# Python Docs RAG API

Production-ready **REST API** basierend auf **FastAPI** und **Docker**, die Fragen zu den offiziellen Python-Tutorials beantwortet.

* **Architecture**: Retrieval-Augmented Generation (RAG)
* **Embeddings & Vector Store**: Ollama (`nomic-embed-text`) + Qdrant
* **LLM**: Google Gemini API (`gemini-1.5-flash`)
* **Features**: Rate Limiting (SlowAPI), API-Key Auth, Prompt-Injection Protection

<p align="center">
  <img src="docs/Python_API_1.png" alt="Swagger UI Overview" width="800">
</p>

---

## Quickstart & Testing

1. **Repository klonen:**
   ```bash
   git clone [https://github.com/DeinUser/Python_RAG.git](https://github.com/DeinUser/Python_RAG.git)
   cd Python_RAG
   ```

2. **Umgebungsvariablen konfigurieren:**
   ```bash
   cp .env.example .env
   # GEMINI_API_KEY (und optional API_KEY für Auth) in .env eintragen
   ```

3. **Container starten:**
   ```bash
   docker compose up -d
   ```

4. **Einmalig: Embedding-Modell in Ollama laden:**
   ```bash
   docker compose exec ollama ollama pull nomic-embed-text
   ```

5. **API testen:**
   * **Swagger UI (Browser):** `http://localhost:8000/docs`
   * **cURL Terminal-Test:**
     ```bash
     curl -X 'POST' 'http://localhost:8000/query' \
       -H 'Content-Type: application/json' \
       -d '{"question": "Wie funktioniert eine List Comprehension?"}'
     ```

---

## Setup & Optionen

Die Anwendung lässt sich über die `.env`-Datei flexibel konfigurieren:

1. **Authentication (`API_KEY`)**
   * **Gesetzt**: Zugriffe auf `/query` erfordern den Header `X-API-Key`.
   * **Leer**: Modus "Fail-Open" für einfache lokale Entwicklungszwecke.

2. **Quellenanzeige (`SHOW_SOURCES`)**
   * `SHOW_SOURCES=true`: Liefert die zugrundeliegenden Dokumenten-Header und Textausschnitte im Response-Body mit.

---

## Nutzungsbeispiel

<p align="center">
  <img src="docs/Python_API_2.png" alt="Gültige Anfrage" width="800">
</p>

* **Request**: POST-Anfrage an `/query` mit JSON-Payload `{"question": "..."}`.
* **Response**: Kontextbasierte Antwort auf Basis der geparsten Python-Dokumentation.

---

## Sicherheitsmaßnahmen & Quality Guardrails

<p align="center">
  <img src="docs/Python_API_3.png" alt="Ungültige Anfrage" width="800">
</p>

* **System-Prompt Guardrails**: Verhindert Prompt Injections und Jailbreaks (z. B. *"Ignoriere alle vorherigen Befehle..."*).
* **Strict RAG Scope**: Die API beantwortet ausschließlich Fragen, die durch den hinterlegten Python-Kontext abgedeckt sind.
* **Timing-Attack Protection**: Verwendung von `secrets.compare_digest` bei der API-Key-Validierung.
* **Rate Limiting**: Schutz vor Missbrauch via `slowapi` (Maximal 5 Anfragen pro Minute pro Nutzer).
* **Network Isolation**: Der interne Ollama-Port wird nicht auf den Host-Rechner exponiert.

---

## Roadmap & TODOs

- [ ] Mehr Python-Docs verarbeiten für komplexere Abfragen
- [ ] Prompts für den Chatbot weiter verfeinern
- [ ] Zusätzliche Security-Features (z. B. TLS/HTTPS-Proxy)
- [ ] Eigenes Frontend / UI zur Verbesserung der User Experience