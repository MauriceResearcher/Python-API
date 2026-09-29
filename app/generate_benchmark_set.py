import sys
from unittest.mock import MagicMock

sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

import os
import random
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

from app import config
from app.rag_service import RagService


def generate_benchmark():
    rag_service = RagService()
    rag_service.build()

    print(f"Lese Chunks aus Qdrant Collection '{config.QDRANT_COLLECTION_NAME}' aus...")

    scroll_result = rag_service.client.scroll(
        collection_name=config.QDRANT_COLLECTION_NAME,
        limit=200,
        with_payload=True
    )[0]

    docs = []
    for point in scroll_result:
        payload = point.payload or {}
        content = payload.get("page_content") or payload.get("text") or payload.get("document")
        if content:
            docs.append(Document(page_content=content, metadata=payload.get("metadata", {})))

    print(f"Erfolgreich {len(docs)} Chunks aus Qdrant geladen.")

    if not docs:
        print("FEHLER: Keine Dokumente in der Qdrant Collection gefunden!")
        rag_service.close()
        return

    if len(docs) > 30:
        docs = random.sample(docs, 20)
    print(f"Verwende {len(docs)} zufällige Chunks für die Testset-Generierung.")

    generator_llm = LangchainLLMWrapper(rag_service.llm)
    generator_embeddings = LangchainEmbeddingsWrapper(rag_service.embeddings)

    run_config = RunConfig(
        max_workers=2,
        max_retries=10,
        max_wait=60
    )

    print("\n--- Initialisiere Ragas TestsetGenerator ---")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )

    print("Generiere Testfälle...")
    test_dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=10,
        run_config=run_config
    )

    # Pfad absolut vom Root aus steuern: <PROJEKT_ROOT>/app/data/generated_benchmark.csv
    data_dir = config.BASE_DIR / "app" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "generated_benchmark.csv"
    df = test_dataset.to_pandas()
    df.to_csv(output_path, index=False)

    print(f"\n================ BENCHMARK GENERATION COMPLETED ================")
    print(f"Testset erfolgreich gespeichert unter: {output_path}")
    print(df[["user_input", "reference"]].head())

    rag_service.close()


if __name__ == "__main__":
    generate_benchmark()