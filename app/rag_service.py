"""
Kapselt den kompletten RAG-Stack (Embeddings, Vectorstore, Retriever, LLM)
in einer Klasse, die genau EINMAL beim App-Start instanziiert und "gebaut"
wird (siehe app/main.py -> lifespan).

Warum eine Klasse statt der Modul-Ebene wie im Original main.py?
- Testbarkeit: In Tests kann eine Fake-Instanz per FastAPI
  dependency_overrides eingesetzt werden, ganz ohne echten Ollama-/
  Qdrant-/Gemini-Zugriff.
- Der lokale QdrantClient(path=...) darf pro Prozess nur einmal geöffnet
  werden (File-Lock) - eine Klasse mit explizitem build() macht diesen
  Lebenszyklus (build einmal, ask() beliebig oft) explizit sichtbar.
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from . import config
from .load_python_docs import chunk_splitter, load_python_docs

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self) -> None:
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.client: Optional[QdrantClient] = None
        self.vectorstore: Optional[QdrantVectorStore] = None
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(config.PROMPT_TEMPLATE)

    @property
    def is_ready(self) -> bool:
        return self.retriever is not None and self.llm is not None

    def build(self) -> None:
        """Lädt eine vorhandene Qdrant-Collection oder erstellt sie neu.

        Muss genau einmal aufgerufen werden (idealerweise im FastAPI
        lifespan-Hook), NICHT pro Request.
        """
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
        )

        self.client = QdrantClient(path=config.QDRANT_PATH)
        try:
            self.client.get_collection(collection_name=config.QDRANT_COLLECTION_NAME)
            self.vectorstore = QdrantVectorStore(
                collection_name=config.QDRANT_COLLECTION_NAME,
                embedding=self.embeddings,
                client=self.client,
            )
            logger.info(
                "Bestehende Qdrant-Collection '%s' geladen.",
                config.QDRANT_COLLECTION_NAME,
            )
        except Exception:
            self.client.close()
            self.client = None
            logger.info(
                "Collection '%s' nicht gefunden - scrape & indexiere die "
                "Python-Docs neu (kann beim ersten Start etwas dauern).",
                config.QDRANT_COLLECTION_NAME,
            )
            self.vectorstore = self._index_documents()
            # from_documents() öffnet intern seinen eigenen QdrantClient
            # (wir übergeben nur path=..., keinen client=...). Referenz
            # nachträglich einsammeln, damit close() ihn findet.
            self.client = getattr(self.vectorstore, "client", None)

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVER_K}
        )

    def close(self) -> None:
        """Gibt den Qdrant-Datei-Lock wieder frei. Im lifespan-Hook nach
        yield aufrufen, damit ein Neustart (z.B. bei --reload) nicht mit
        'Storage folder already accessed' fehlschlägt."""
        if self.client is not None:
            self.client.close()
            self.client = None

    def _index_documents(self) -> QdrantVectorStore:
        final_chunks: List[Document] = []
        for link in config.DOC_LINKS:
            markdown_text, page_title = load_python_docs(link)
            chunks = chunk_splitter(markdown_text, page_title)
            final_chunks.extend(chunks)

        logger.info(
            "%d Chunks aus %d Seiten werden indexiert.",
            len(final_chunks),
            len(config.DOC_LINKS),
        )

        return QdrantVectorStore.from_documents(
            final_chunks,
            embedding=self.embeddings,
            path=config.QDRANT_PATH,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )

    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """Beantwortet eine Frage. Ruft den Retriever nur EINMAL auf
        (im Gegensatz zum Original-Terminal-Loop, der ihn zweimal aufrief:
        einmal fürs Debug-Print, einmal versteckt in der RAG-Chain)."""
        if not self.is_ready:
            raise RuntimeError("RagService.build() muss vor ask() aufgerufen werden.")

        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_value = self.prompt.invoke({"context": context, "question": question})
        response = self.llm.invoke(prompt_value)

        # Manche Gemini-Antworten liefern response.content als Liste von
        # Content-Blöcken statt als reinen String (z.B. [{"type": "text",
        # "text": "..."}]). Beide Formen werden hier zu einem String
        # zusammengefasst.
        if isinstance(response.content, list):
            parts = []
            for part in response.content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            answer = "".join(parts)
        else:
            answer = str(response.content)

        return answer, docs