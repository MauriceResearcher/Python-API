from trulens.apps.app import instrument  # Zukunftssicherer Import
from trulens.apps.custom import TruCustomApp
from trulens.core import Metric, TruSession
from trulens.core.select import Select
from trulens.providers.google import Google

from app import config
from app.rag_service import RagService

# 1. TruLens Session & Provider initialisieren
session = TruSession()
session.reset_database()  # Setzt die lokale SQLite-Eval-DB zurück

provider = Google(model_engine=config.GEMINI_MODEL)

# 2. RAG-Triade mit der OTEL-konformen Syntax definieren

# a) Groundedness: Basiert die Antwort auf dem Kontext?
m_groundedness = (
    Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="Groundedness",
    )
    .on(Select.RecordCalls.query.rets.context)
    .on(Select.RecordCalls.query.rets.answer)
)

# b) Context Relevance: Passt der geholte Kontext zur Frage?
m_context_relevance = (
    Metric(
        implementation=provider.qs_relevance_with_cot_reasons,
        name="Context Relevance",
    )
    .on(Select.RecordCalls.query.args.question)
    .on(Select.RecordCalls.query.rets.context)
)

# c) Answer Relevance: Beantwortet die Antwort die Frage?
m_answer_relevance = (
    Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="Answer Relevance",
    )
    .on(Select.RecordCalls.query.args.question)
    .on(Select.RecordCalls.query.rets.answer)
)

metrics = [m_groundedness, m_context_relevance, m_answer_relevance]


# 3. RAG-Wrapper erstellen
class TruRagWrapper:
    def __init__(self, service: RagService):
        self.service = service

    @instrument
    def query(self, question: str) -> dict:
        answer, docs = self.service.ask(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"answer": answer, "context": context}


# 4. Service instanziieren & bauen
rag_service = RagService()
rag_service.build()

rag_wrapper = TruRagWrapper(rag_service)

# 5. App mit TruCustomApp und den Metrics wrappen
tru_app = TruCustomApp(
    rag_wrapper, app_name="Python_RAG_Evaluator", feedbacks=metrics
)

# 6. Test-Fragen durchführen
test_questions = [
    "Wie erstelle ich eine Liste in Python?",
    "Welche Quantencomputer-Algorithmen unterstützt Python native?",
]

with tru_app as recorder:
    for q in test_questions:
        print(f"\n[Frage]: {q}")
        result = rag_wrapper.query(q)
        print(f"[Antwort]: {result['answer'][:150]}...\n")

# 7. Dashboard starten
session.run_dashboard()