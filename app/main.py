import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from .rag_service import RagService
from .schemas import HealthResponse, QueryRequest, QueryResponse, SourceChunk
from .auth import require_api_key

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ein einziges, prozessweites RagService-Objekt. Wird NICHT pro Request
# gebaut, sondern einmal beim Start (siehe lifespan unten) - u.a. weil
# der lokale Qdrant-Client nur von einem Prozess gleichzeitig geöffnet
# werden darf.
rag_service = RagService()



# Wenn True zeigt die KI die genutzten Quellen in der Antwort an
show_sources = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starte RAG-Service (erster Start kann wegen Scraping+Indexierung dauern)...")
    rag_service.build()
    logger.info("RAG-Service bereit.")
    yield
    logger.info("Fahre herunter.")
    rag_service.close()


def get_rag_service() -> RagService:
    """FastAPI-Dependency. In Tests via app.dependency_overrides ersetzbar,
    damit Tests ohne echten Ollama-/Qdrant-/Gemini-Zugriff laufen.

    Wichtig: das deckt NUR Routen ab, die service per Depends() bekommen
    (health/query unten). Der lifespan-Hook oben ruft rag_service.build()
    direkt auf und ignoriert dependency_overrides komplett - deshalb gibt
    es unten create_app(), mit dem Tests einen eigenen, harmlosen Lifespan
    einsetzen können.
    """
    return rag_service


def create_app(lifespan=lifespan) -> FastAPI:
    """App-Factory statt eines einzigen, fest verdrahteten Modul-Objekts.

    Der Grund dafür ist rein testbezogen: ohne Factory würde JEDE
    TestClient(app)-Instanz denselben, echten lifespan-Hook auslösen und
    damit rag_service.build() erneut aufrufen - also einen echten, neuen
    QdrantClient(path=...) öffnen, ohne den vorherigen zu schliessen.
    Bei mehreren Tests in einem pytest-Lauf führt das zum
    "Storage folder already accessed"-Fehler. Tests übergeben hier
    stattdessen einen No-Op-Lifespan (siehe tests/test_api.py) und
    umgehen so den echten build()-Aufruf komplett.
    """

    limiter = Limiter(key_func=get_remote_address)


    app = FastAPI(
        title="Python Docs RAG API",
        description="Beantwortet Fragen zu den offiziellen Python-Tutorial-Docs per RAG "
        "(Qdrant + Ollama-Embeddings + Gemini).",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/health", response_model=HealthResponse)
    def health(service: RagService = Depends(get_rag_service)) -> HealthResponse:
        return HealthResponse(status="ok", ready=service.is_ready)

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
    @limiter.limit("5/minute")
    def query(
        request: QueryRequest,
        body: QueryRequest,
        service: RagService = Depends(get_rag_service),
    ) -> QueryResponse:
        if not service.is_ready:
            raise HTTPException(status_code=503, detail="RAG-Service ist noch nicht bereit.")

        try:
            answer, docs = service.ask(body.question)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Fehler bei der Beantwortung der Frage")
            raise HTTPException(
                status_code=500,
                detail="Interner Fehler bei der Beantwortung der Frage.",
            ) from exc

        if show_sources:
            sources = [
                SourceChunk(
                    header_1=doc.metadata.get("header_1"),
                    header_2=doc.metadata.get("header_2"),
                    header_3=doc.metadata.get("header_3"),
                    content_preview=doc.page_content[:200],
                )
                for doc in docs
            ]
            return QueryResponse(answer=answer, sources=sources)

        else:
            return QueryResponse(answer=answer)

    return app


# Produktions-App - das ist es, was "uvicorn app.main:app" (lokal & im
# Dockerfile) tatsächlich startet. Nutzt den echten lifespan von oben.
app = create_app()