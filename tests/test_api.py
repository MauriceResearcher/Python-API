"""
Diese Tests ersetzen den echten RagService über FastAPIs
`dependency_overrides` durch ein Fake-Objekt. Damit laufen die Tests
ohne laufenden Ollama-Server, ohne Qdrant und ohne GOOGLE_API_KEY -
wichtig z.B. für CI-Pipelines ohne diese Abhängigkeiten.

Die client/not_ready_client-Fixtures setzen config.API_KEY zusätzlich
explizit auf None, damit die Tests unabhängig davon funktionieren, ob im
lokalen .env zufällig schon ein echter API_KEY eingetragen ist (sonst
würden z.B. test_query_success etc. mit einem lokal gesetzten API_KEY
unerwartet mit 401 fehlschlagen). Die eigentlichen Auth-Tests weiter
unten setzen den Wert gezielt selbst.

Ausführen (nach `pip install -r requirements-dev.txt`):
    pytest -v
"""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from app.main import create_app, get_rag_service
from app import auth


@asynccontextmanager
async def noop_lifespan(app):
    """Ersetzt den echten lifespan-Hook nur für Tests. Der echte Hook
    ruft rag_service.build() auf - das würde bei JEDER TestClient-
    Instanz einen echten QdrantClient(path=...) öffnen (dependency_overrides
    greift dort nicht, da lifespan nicht über Depends() läuft). Bei
    mehreren Tests in einem pytest-Lauf führt das zum
    "Storage folder already accessed"-Fehler. Mit diesem No-Op-Lifespan
    passiert beim Start/Stop der Test-App schlicht nichts."""
    yield


class FakeDocument:
    """Duck-typed Ersatz für langchain_core.documents.Document, damit
    diese Tests keine LangChain-Objekte instanziieren müssen."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FakeRagService:
    def __init__(self, ready: bool = True):
        self._ready = ready
        self.last_question = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    def ask(self, question: str):
        self.last_question = question
        docs = [
            FakeDocument(
                page_content="List comprehensions provide a concise way to create lists.",
                metadata={"header_1": "More Control Flow Tools", "header_2": "List Comprehensions"},
            )
        ]
        return f"Antwort auf: {question}", docs


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth.config, "API_KEY", None)
    test_app = create_app(lifespan=noop_lifespan)
    test_app.dependency_overrides[get_rag_service] = lambda: FakeRagService(ready=True)
    with TestClient(test_app) as c:
        yield c


@pytest.fixture
def not_ready_client(monkeypatch):
    monkeypatch.setattr(auth.config, "API_KEY", None)
    test_app = create_app(lifespan=noop_lifespan)
    test_app.dependency_overrides[get_rag_service] = lambda: FakeRagService(ready=False)
    with TestClient(test_app) as c:
        yield c


def test_health_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ready": True}


def test_health_not_ready(not_ready_client):
    response = not_ready_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ready": False}


def test_query_success(client):
    response = client.post("/query", json={"question": "Was ist eine List Comprehension?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Antwort auf: Was ist eine List Comprehension?"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["header_1"] == "More Control Flow Tools"
    assert data["sources"][0]["header_2"] == "List Comprehensions"
    assert data["sources"][0]["content_preview"].startswith("List comprehensions")


def test_query_empty_question_is_rejected(client):
    # min_length=1 im Schema -> FastAPI/Pydantic muss das mit 422 ablehnen,
    # bevor der RagService überhaupt aufgerufen wird.
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_missing_field_is_rejected(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_returns_503_when_service_not_ready(not_ready_client):
    response = not_ready_client.post("/query", json={"question": "Hallo"})
    assert response.status_code == 503


def test_query_open_when_no_api_key_configured(client):
    # client-Fixture setzt API_KEY bereits auf None -> Endpoint bleibt offen.
    response = client.post("/query", json={"question": "Hallo"})
    assert response.status_code == 200


def test_query_rejected_without_header_when_api_key_configured(client, monkeypatch):
    monkeypatch.setattr(auth.config, "API_KEY", "test-secret")
    response = client.post("/query", json={"question": "Hallo"})
    assert response.status_code == 401


def test_query_rejected_with_wrong_api_key(client, monkeypatch):
    monkeypatch.setattr(auth.config, "API_KEY", "test-secret")
    response = client.post(
        "/query",
        json={"question": "Hallo"},
        headers={"X-API-Key": "falscher-wert"},
    )
    assert response.status_code == 401


def test_query_succeeds_with_correct_api_key(client, monkeypatch):
    monkeypatch.setattr(auth.config, "API_KEY", "test-secret")
    response = client.post(
        "/query",
        json={"question": "Hallo"},
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200


def test_health_is_not_protected_by_api_key(client, monkeypatch):
    # /health bleibt bewusst ungeschützt (Docker HEALTHCHECK schickt keinen
    # Header) - muss auch bei konfiguriertem API_KEY ohne Header erreichbar sein.
    monkeypatch.setattr(auth.config, "API_KEY", "test-secret")
    response = client.get("/health")
    assert response.status_code == 200