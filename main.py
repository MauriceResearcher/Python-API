from load_python_docs import load_python_docs, chunk_splitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import shutil


from dotenv import load_dotenv

# datanbank löschen
#shutil.rmtree("./qdrant_db", ignore_errors=True)

load_dotenv()

#embeddings = OllamaEmbeddings(model="llama3.2")
embeddings = OllamaEmbeddings(model = "nomic-embed-text")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature = 0)

PERSIST_PATH = "./qdrant_db"
COLLECTION_NAME = "python_docs"


def main():

    client = QdrantClient(path=PERSIST_PATH)

    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        vectorstore = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            client= client
        )
    except Exception:
        client.close()


        links = ["https://docs.python.org/3/tutorial/appetite.html",
                 "https://docs.python.org/3/tutorial/interpreter.html",
                 "https://docs.python.org/3/tutorial/introduction.html",
                 "https://docs.python.org/3/tutorial/controlflow.html",
                 "https://docs.python.org/3/tutorial/datastructures.html",
                 "https://docs.python.org/3/tutorial/modules.html",
                 "https://docs.python.org/3/tutorial/inputoutput.html",
                 "https://docs.python.org/3/tutorial/errors.html",
                 "https://docs.python.org/3/tutorial/classes.html",
                 "https://docs.python.org/3/tutorial/stdlib.html",
                 "https://docs.python.org/3/tutorial/stdlib2.html",
                 "https://docs.python.org/3/tutorial/venv.html",
                 "https://docs.python.org/3/tutorial/whatnow.html",
                 "https://docs.python.org/3/tutorial/interactive.html",
                 "https://docs.python.org/3/tutorial/floatingpoint.html",
                 "https://docs.python.org/3/tutorial/appendix.html"]


        text_list = []
        for link in links:
            markdown_text, page_title = load_python_docs(link)
            text_list.append([markdown_text, page_title])

        final_chunks = []

        for texts in text_list:
            chunks = chunk_splitter(texts[0], texts[1])
            final_chunks.extend(chunks)


        print(f"\nFertig! Insgesamt wurden {len(final_chunks)} Chunks aus {len(links)} Seiten erstellt.")

        vectorstore = QdrantVectorStore.from_documents(
            final_chunks,
            embedding=embeddings,
            path = PERSIST_PATH,
            collection_name = COLLECTION_NAME
        )


    # search engine, saves 15 most similar chunls



    #TODO mmr / similarity_score_threshold probieren
    retriever = vectorstore.as_retriever(search_kwargs={"k":5})

    template = """
    You are a Python expert. EXCLUSIVELY use the following documentation excerpts to answer. 
    If the answer is partly contained, provide the best possible answer based on the context.
    If the answer is not in the context, respond with "This information cannot be found in the Python docs!"
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- The Python Expert is readx to answer your questions ---"
          "\n type exit or quit to exit the chat")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        # 1. DEBUG: Hol dir die Chunks direkt vom Retriever
        docs = retriever.invoke(query)

        print(f"\n🔍 [RETRIEVER DEBUG] {len(docs)} Chunks gefunden:")
        for i, doc in enumerate(docs[:3]):  # Zeigt die TOP 3 Ergebnisse
            source_header = doc.metadata.get("header_2") or doc.metadata.get("header_1") or "Unbekannt"
            print(f"  --- Chunk {i + 1} (Quelle: {source_header}) ---")
            print(f"  {doc.page_content[:200]}...\n")



        response = rag_chain.invoke(query)
        print(f"\n Python Expert: {response}")

if __name__ == "__main__":
    main()
