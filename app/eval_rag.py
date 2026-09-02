import numpy as np
from trulens.apps.app import TruApp, instrument
from trulens.core import Metric, Selector, TruSession
from trulens.dashboard import run_dashboard
from trulens.providers.google import Google

from app import config
from app.rag_service import RagService

# 1. TruLens Session & Provider
session = TruSession()
session.reset_database()

provider = Google(model_engine=config.GEMINI_MODEL)

# 2. Metriken definieren
f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
        "question": Selector.select_record_input(),
    },
)

f_answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)

f_context_relevance = Metric(
    implementation=provider.context_relevance_with_cot_reasons,
    name="Context Relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),
    },
    agg=np.mean,
)


# 3. Custom Class mit @instrument dekorieren
class InstrumentedRagApp:

    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service

    @instrument
    def query(self, question: str) -> str:
        answer, _ = self.rag_service.ask(question)
        return answer


# 4. Instanziieren & mit TruApp wrappen
rag_service = RagService()
rag_service.build()

custom_rag_app = InstrumentedRagApp(rag_service)

tru_rag = TruApp(
    custom_rag_app,
    app_name="Python_Docs_RAG",
    app_version="v1_gemini",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

# 5. Test-Queries ausführen
test_queries = [
    "Wie erstelle ich eine Liste in Python?",
    "Welche Quantencomputer-Algorithmen unterstützt Python native?",
]

with tru_rag as recording:
    for query in test_queries:
        print(f"\n[Frage]: {query}")
        response = custom_rag_app.query(query)
        print(f"[Antwort]: {response[:150]}...\n")

# 6. Leaderboard & Dashboard
print(session.get_leaderboard())
run_dashboard(session)