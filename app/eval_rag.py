import sys
from unittest.mock import MagicMock

# --- FIX: Täuscht das gelöschte VertexAI-Modul für Ragas vor ---
# MUSS VOR DEM RAGAS-IMPORT STEHEN!
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

import os
from pathlib import Path
import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall

from app import config
from app.rag_service import RagService


def main():
    # 1. Pfad zur Benchmark CSV ermitteln
    csv_path = config.BASE_DIR / "app" / "data" / "generated_benchmark.csv"
    if not os.path.exists(csv_path):
        print(f"FEHLER: Die Benchmark-Datei '{csv_path}' wurde nicht gefunden!")
        print("Bitte führe zuerst 'generate_benchmark_set.py' aus.")
        return

    print(f"Lade Benchmark-Testfälle aus '{csv_path}'...")
    df_benchmark = pd.read_csv(csv_path)

    # 2. RAG-Service aufbauen
    rag_service = RagService()
    rag_service.build()

    print("\n--- 1. Generiere RAG-Antworten für geladene Testfälle ---")
    samples = []

    for index, row in df_benchmark.iterrows():
        query = row["user_input"]
        reference = row["reference"]

        print(f"[{index + 1}/{len(df_benchmark)}] Verarbeite: '{query}'")

        answer, docs = rag_service.ask(query)
        contexts = [doc.page_content for doc in docs]

        samples.append(
            SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference,
            )
        )

    rag_service.close()

    eval_dataset = EvaluationDataset(samples=samples)

    print("\n--- 2. Initialisiere Ragas Evaluator ---")
    evaluator_llm = LangchainLLMWrapper(rag_service.llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(rag_service.embeddings)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextRecall(llm=evaluator_llm),
    ]

    print("\n--- 3. Starte Evaluierung ---")
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    print("\n================ EVALUATION RESULTS ================")
    df_results = results.to_pandas()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Sicheres Drucken: Gibt alle verfügbaren Metrik-Spalten aus
    metric_cols = [c for c in df_results.columns if
                   c not in ["user_input", "retrieved_contexts", "reference", "response"]]
    print(df_results[["user_input"] + metric_cols])

    # Ergebnisse zusätzlich im data-Ordner abspeichern
    output_eval_path = config.BASE_DIR / "app" / "data" / "evaluation_results.csv"
    df_results.to_csv(output_eval_path, index=False)
    print(f"\nDetaillierte Evaluierungsergebnisse gespeichert unter: {output_eval_path}")


if __name__ == "__main__":
    main()