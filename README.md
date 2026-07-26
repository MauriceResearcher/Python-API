# Python Docs RAG API

FastAPI-Wrapper um den bestehenden RAG-Stack (Qdrant lokal, Ollama-Embeddings,
Gemini als LLM), containerisiert mit Docker / Docker Compose.

## Endpunkte

- `GET /health` → `{"status": "ok", "ready": true|false}` — **ungeschützt**
  (wird auch vom Docker-`HEALTHCHECK` ohne Header aufgerufen).
- `POST /query` mit Body `{"question": "..."}` →
  `{"answer": "...", "sources": [{"header_1": ..., "header_2": ..., "content_preview": ...}, ...]}`
  — **geschützt**, sobald `API_KEY` in der `.env` gesetzt ist (siehe unten).
  Client schickt dann `X-API-Key: <gleicher Wert>` mit.
- Interaktive Doku (automatisch von FastAPI generiert): `/docs`

## Lokal ohne Docker starten

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Ollama muss lokal laufen und das Embedding-Modell muss vorhanden sein:
ollama pull nomic-embed-text

cp .env.example .env   # GOOGLE_API_KEY eintragen

uvicorn app.main:app --reload
```

Erster Start: die App scraped die 16 Python-Tutorial-Seiten und baut den
Qdrant-Index neu auf (dauert etwas). Danach wird bei jedem weiteren Start
die vorhandene Collection unter `./qdrant_db` wiederverwendet.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

Die Tests in `tests/test_api.py` ersetzen den `RagService` per
`app.dependency_overrides` durch ein Fake-Objekt — sie brauchen also
**keinen** laufenden Ollama-Server, kein Qdrant und keinen `GOOGLE_API_KEY`.

> **Wichtig:** `app/main.py` stellt dafür `create_app(lifespan=...)` als
> Factory bereit. Der Grund: `dependency_overrides` greift nur bei Routen,
> die den Service per `Depends()` bekommen — der `lifespan`-Hook ruft
> `rag_service.build()` aber direkt auf und würde sonst bei **jeder**
> `TestClient`-Instanz einen echten `QdrantClient(path=...)` öffnen, ohne
> den vorherigen zu schliessen. Ergebnis wäre ein
> `RuntimeError: Storage folder ... is already accessed by another
> instance of Qdrant client.` ab dem zweiten Test. Die Tests übergeben
> deshalb einen No-Op-`lifespan`, der `build()` gar nicht erst aufruft.

> Hinweis zur Herkunft dieses Projekts: Ich konnte diese Tests in meiner
> Sandbox nicht selbst ausführen (kein Netzwerkzugriff dort, daher auch kein
> `pip install`). Geprüft habe ich stattdessen mit `python -m py_compile`
> (Syntax aller Dateien ist sauber) und durch sorgfältiges Nachvollziehen der
> FastAPI-/LangChain-APIs. Führe `pytest -v` bei dir lokal aus, um die Logik
> tatsächlich zu verifizieren — falls dabei etwas auffällt, sag mir einfach
> die Fehlermeldung.

## Mit Docker Compose starten (empfohlen)

```bash
cp .env.example .env   # GOOGLE_API_KEY (und optional API_KEY) eintragen
docker compose up --build -d

# Einmalig: Embedding-Modell in den Ollama-Container laden
docker compose exec ollama ollama pull nomic-embed-text

curl http://localhost:8000/health

# Ohne gesetzten API_KEY:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Wie funktioniert eine List Comprehension?"}'

# Mit gesetztem API_KEY (Wert muss zu API_KEY in der .env passen):
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dein-geheimer-key" \
  -d '{"question": "Wie funktioniert eine List Comprehension?"}'
```

## Nur das API-Image bauen (ohne Compose)

```bash
docker build -t python-docs-rag-api .
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --env-file .env \
  python-docs-rag-api
```

(`host.docker.internal` funktioniert unter Docker Desktop/macOS/Windows;
unter Linux ggf. `--add-host=host.docker.internal:host-gateway` ergänzen
oder gleich docker-compose mit dem Ollama-Service nutzen.)

## Sicherheit

Bereits umgesetzt:

- **`POST /query` per API-Key geschützt** (`app/auth.py`). Ohne `API_KEY` in
  der `.env` bleibt der Endpoint bewusst offen (mit Warnung im Log) — für
  jeden Einsatz ausserhalb von `localhost` unbedingt setzen.
- **Keine internen Fehlerdetails mehr im 500er-Response** — nur noch
  serverseitig geloggt (`logger.exception(...)`), der Client bekommt eine
  generische Meldung.
- **Ollama-Port nicht mehr auf den Host exponiert** — Ollama hat keine
  eigene Authentifizierung; der `api`-Container erreicht es intern über
  den Compose-Service-Namen, eine Freigabe nach aussen war unnötig.
- **Timeout beim Scraping** (`requests.get(..., timeout=(5, 30))`) —
  verhindert, dass ein nicht antwortender Server den App-Start für immer
  blockiert.
- **System-/Human-Message-Trennung im Prompt** statt einer einzigen
  Nachricht — erschwert Prompt-Injection über die Nutzerfrage etwas
  (verhindert sie nicht vollständig).

Bewusst (noch) nicht umgesetzt, aber gut zu wissen:

- **Kein Rate Limiting.** Auch mit API-Key kann ein einzelner autorisierter
  Client die API fluten und Gemini-Kosten verursachen. Für echten Einsatz:
  z.B. `slowapi` (FastAPI-Middleware) oder Rate Limiting auf einem
  vorgeschalteten Reverse Proxy.
- **Kein TLS im Container selbst.** Uvicorn spricht nur HTTP — vor einen
  Reverse Proxy (Traefik/Nginx/Caddy) stellen, wenn die API übers Internet
  erreichbar sein soll.
- **`requirements.txt` nicht versionsgepinnt.** Für reproduzierbare,
  auditierbare Builds z.B. mit `pip-compile` (pip-tools) ein Lockfile mit
  Hashes erzeugen.
- **API-Key ist ein einzelnes, geteiltes Geheimnis**, kein Pro-User-Token.
  Reicht für einen Client/eine kleine Gruppe, nicht für ein Produkt mit
  unterschiedlichen Nutzerrechten.

## Bekannte Stolpersteine

- **Nur 1 Uvicorn-Worker / 1 Prozess.** `QdrantClient(path=...)` ist der
  eingebettete Qdrant-Modus (lokale Datei statt Server) und lässt nur einen
  offenen Client gleichzeitig zu. Mehrere Worker (`--workers 2`) oder mehrere
  Container-Replicas gegen denselben `qdrant_db`-Pfad führen zu Lock-Fehlern.
  Für echte Skalierung: auf einen richtigen Qdrant-Server umstellen
  (`QdrantClient(url=...)` statt `path=...`).
- **Erster Start ist langsam.** Scraping + Embedding von 16 Seiten passiert
  synchron beim Startup (`lifespan`). `/health` liefert erst `ready: true`,
  sobald das durch ist.
- **Ollama-Modell muss existieren**, bevor Requests reinkommen — sonst
  schlägt `build()` beim Embedding-Aufruf fehl. Deshalb der explizite
  `ollama pull nomic-embed-text` Schritt oben.
- **GOOGLE_API_KEY** wird von `langchain-google-genai` automatisch aus der
  Umgebung gelesen — es reicht, ihn per `.env` / `env_file` bereitzustellen,
  kein Code-Change nötig.