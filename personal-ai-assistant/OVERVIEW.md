# Personal AI Assistant – Project Overview

A simple, detailed guide to the folder structure, how files connect, the request workflow, and how to run the app.

---

## 1. Folder structure (what lives where)

```
personal-ai-assistant/
├── backend/                    # All application code (package via namespace / PYTHONPATH)
│   ├── main.py                 # Entry: FastAPI app + lifespan
│   ├── graph_routing.py        # LangGraph conditional edges (retriever / tool phases)
│   ├── core/
│   │   ├── config.py           # Settings from env (getSettings)
│   │   └── constants.py        # Shared constants (e.g. STM window)
│   ├── api/
│   │   ├── answer_source.py    # answer_source from final graph state
│   │   ├── routes/
│   │   │   ├── __init__.py     # apiRouter → health, tools, documents, query
│   │   │   ├── health.py
│   │   │   ├── tools.py        # GET /api/v1/tools (MCP tool listing)
│   │   │   ├── documents.py    # upload + ingest
│   │   │   └── query.py        # POST /api/v1/query
│   │   └── schemas/            # Pydantic models (+ __init__.py re-exports)
│   ├── database/
│   │   ├── vector_store.py     # pgvector, addDocuments, getRetriever wiring
│   │   ├── checkpoint_store.py # Postgres checkpointer + store (STM/LTM)
│   │   └── ingest_registry.py  # Idempotent ingest metadata table
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── retriever.py
│   │   └── self_rag_pipeline.py
│   ├── tools/
│   │   ├── mcp_client.py
│   │   └── web_search.py
│   └── agents/
│       ├── memory.py
│       ├── nodes.py
│       └── langgraph_flow.py   # buildAgentGraph, AgentState
├── uploads/                    # Uploaded files – gitignored (keep uploads/.gitkeep)
├── tests/
│   └── test_sourced_pipeline.py
├── venv/                       # Optional local venv (gitignored); or use repo-root venv
├── requirements.txt
├── Dockerfile
├── pyproject.toml              # Metadata, pytest/ruff/mypy config
├── OVERVIEW.md                 # This file
└── .gitignore
```

**Important:** We use **pgvector only** for vector storage (PostgreSQL + pgvector). We do **not** use FAISS or Chroma; all document embeddings and retrieval go through the same Postgres instance that also holds the checkpointer (STM) and store (LTM).

---

## 2. How files are connected (who imports whom)

- **`api/`** (routes, schemas) imports from **`core/`**, **`database/`**, **`rag/`**, **`agents/`**, **`tools/`**.
- **`rag/`**, **`agents/`**, **`tools/`**, **`database/`** do **not** import from **`api/`** (clean boundaries, no circular imports).
- **`core/`** is used everywhere (config, constants).

```
                    ┌─────────────┐
                    │  main.py    │
                    └──────┬──────┘
                           │ includes apiRouter, calls getSettings, getCheckpointerAndStore,
                           │ buildAgentGraph, getMcpTools
                           ▼
        ┌──────────────────────────────────────────────────┐
        │  api/routes/__init__.py (apiRouter)                │
        │  → health, tools, documents, query                 │
        └──────────────────────────────────────────────────┘
          │              │              │                    │
          ▼              ▼              ▼                    ▼
     health.py      tools.py    documents.py         query.py
          │              │              │                    │
          │              │              │                    │  graph.ainvoke, QueryRequest/Response
          │              │              │                    └──────────────► core/config, api/schemas,
          │              │              │                                        agents, tools
          │              │              │  getSettings, addDocuments,
          │              │              │  loadAndChunk, IngestRequest/Response
          │              │              └──────────────► core/config, database/vector_store,
          │              │                               rag, api/schemas/documents
          │              │  getMcpTools (MCP listing)
          │              └──────────────► tools/mcp_client, core/config
          │  getSettings, loadExisting, HealthResponse
          └─────────────► core/config, database/vector_store, api/schemas/health

    agents/langgraph_flow.py
          │
          │  buildAgentGraph uses: agent_nodes (nodes.py), getSettings
          │  each node is a thin wrapper that calls nodes.py with store/tools/settings
          ▼
    agents/nodes.py
          │
          │  rememberNode → memory.py (MEMORY_PROMPT, MemoryDecision), store
          │  plannerNode  → LLM
          │  retrieverNode → rag.runSelfRag, rag.getRetriever
          │  reasoningNode → LLM, tools (getMcpTools)
          │  toolNode     → tools.invokeTool
          │  answerNode   → store (LTM), LLM
          ▼
    rag/ (runSelfRag, getRetriever), tools/ (invokeTool), agents/memory.py, database (via rag)
```

---

## 3. What runs first (startup order)

When you start the app with **uvicorn**:

1. **`backend.main`** is loaded → `app = createApp()` runs.
2. **`createApp()`** creates a FastAPI app, sets **`lifespan=lifespan`**, and includes **`apiRouter`**.
3. When the server **starts**, FastAPI runs the **lifespan** function:
   - **`getSettings()`** → load config from env (e.g. `.env`).
   - **`getCheckpointerAndStore(settings)`** → open Postgres checkpointer (STM) and PostgresStore (LTM).
   - **`getMcpTools(settings)`** → discover MCP tools from config.
   - **`buildAgentGraph(checkpointer=..., store=..., tools=..., settings=...)`** → build the LangGraph agent.
   - **`app.state.graph = ...`** → store the compiled graph on the app so the query route can use it.
   - Then **`yield`** → app is “ready”; requests are handled.
4. When the server **stops**, execution resumes after **`yield`** and the checkpointer/store context exits (connections closed).

So the **first** code that “runs” in your project is **`main.py`** (createApp + lifespan). The **first** thing that runs on **each request** is the **route** that matches the URL (health, documents, or query).

---

## 4. Request workflow (what runs when)

### GET `/api/v1/health`

1. **`api/routes/health.py`** → `getHealth(settings=Depends(getSettings))`.
2. **`getSettings()`** provides **Settings** (from env).
3. **`loadExisting(settings)`** (from **database/vector_store.py**) checks that the pgvector store is reachable.
4. Returns **HealthResponse(status, app, vector_store_loaded, postgres_ok)**.

### POST `/api/v1/documents/upload`

1. **`api/routes/documents.py`** → `uploadDocument(file, user_id=Form(...), settings=Depends(getSettings))`.
2. **`user_id`** is required (alphanumeric, `_`, `-`, up to 128 chars); same file bytes for the same user are stored once under **`{uploads_dir}/{user_id}/objects/aa/bb/{sha256}{ext}`** (content-addressed).
3. Returns **UploadResponse**: `path` (relative to `uploads_dir`), `filename`, `content_hash`, **`deduplicated`** (true if the file was already on disk for that user).

### POST `/api/v1/documents/ingest`

1. **`api/routes/documents.py`** → `ingestDocuments(body: IngestRequest, ...)`. Body must include **`user_id`** and either **`paths`** (relative to `uploads_dir`, e.g. `user_id/objects/.../hash.pdf`) or **`ingest_all`: true** (all supported files under `uploads_dir/{user_id}/`).
2. **`database/ingest_registry.py`**: Postgres table **`user_document_ingest`** skips embeddings already done for `(user_id, content_hash)`; skipped paths appear in **`skipped_documents`**.
3. **`loadAndChunk(...)`** (from **rag/ingestion.py**): loads, chunks, attaches **`user_id`** and **`content_hash`** to chunk metadata for retrieval filtering.
4. **`addDocuments(chunks, settings)`** (from **database/vector_store.py**): adds chunks to **pgvector**.
5. Returns **IngestResponse** including **`skipped_documents`** when applicable.

### POST `/api/v1/query` (main flow)

1. **`api/routes/query.py`** → `postQuery(request, body: QueryRequest, settings=Depends(getSettings))`.
2. **Graph:** `graph = request.app.state.graph` (set at startup). If missing, builds a fallback graph with **getMcpTools** and **buildAgentGraph** (no checkpointer/store).
3. **Config:** `thread_id` (from body or new UUID), `user_id` (from body or `""`) → `config = {"configurable": {"thread_id": ..., "user_id": ...}}`.
4. **Initial state:** `query`, `messages=[HumanMessage(content=body.query)]`, and empty lists for `retrieved_docs`, `tool_results`, `tool_calls_used`, `sources`, `final_answer`.
5. **`graph.ainvoke(initialState, config=config)`** runs the agent graph. Internally the graph runs nodes in this order:
   - **remember** → **planner** → **retriever** → **reasoning** → then either **tool** (and back to **reasoning**) or **answer** → END.

**Node chain in short:**

| Node          | File            | What it does                                                                                                                         |
| ------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **remember**  | agents/nodes.py | LTM: extract memories from last user message; write new ones to PostgresStore (namespace `("user", user_id, "details")`).            |
| **planner**   | agents/nodes.py | LLM plans 1–3 steps for the query.                                                                                                   |
| **retriever** | agents/nodes.py | Builds retriever with **metadata filter** (`user_id`) when `configurable.user_id` is set; calls **rag.runSelfRag** → **pgvector** + Self-RAG; puts answer and sources in state. |
| **reasoning** | agents/nodes.py | LLM reasons over query + retrieved docs + tool results; may output **tool_calls** or a direct answer.                                |
| **tool**      | agents/nodes.py | Runs MCP tools via **tools.invokeTool(name, args)**; appends results to state; graph goes back to **reasoning**.                     |
| **answer**    | agents/nodes.py | Builds system prompt with LTM from **store.search(("user", user_id, "details"))**; generates final answer.                           |

6. **Response:** From final state: `final_answer`, `sources`, `tool_calls_used`, `refinement_rounds` → **QueryResponse** is returned.

So the **query** route is the only one that runs the **agent**; the agent then uses **rag** (Self-RAG + pgvector), **tools** (MCP), and **database** (vector_store + store for LTM).

---

## 5. How to run the program

### Prerequisites

- **Python 3.12** (or compatible).
- **PostgreSQL** with **pgvector** extension (same DB for vectors, checkpointer, and store).
- **OpenAI API key** (and optionally LangSmith for tracing).

### Step 1: Go to the project folder

```bash
cd /path/to/AgenticGeneticAI/personal-ai-assistant
```

### Step 2: Create and use the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or:  venv\Scripts\activate   # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure environment

Create **`.env`** in **`personal-ai-assistant/`** (gitignored) with at least:

```bash
# OPENAI_API_KEY=your-key
# POSTGRES_URI=postgresql://user:password@host:port/dbname?sslmode=disable
```

### Step 5: Run the server

From **`personal-ai-assistant/`** (project root), set **PYTHONPATH** so that **`backend`** is importable, then start uvicorn:

```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- **`--host 0.0.0.0`** allows access from other machines (e.g. Docker or LAN).
- For local only: `--host 127.0.0.1` or omit (default is 127.0.0.1).

### Step 6: Try the API

- **Health:**  
  `GET http://localhost:8000/api/v1/health`
- **Upload a file:**  
  `POST http://localhost:8000/api/v1/documents/upload` — multipart: **`file`** (PDF/TXT/MD) and **`user_id`** (form field). Same bytes + same user → `deduplicated: true`.
- **Ingest:**  
  `POST http://localhost:8000/api/v1/documents/ingest` with JSON e.g. `{"user_id": "alice", "ingest_all": true}` or `{"user_id": "alice", "paths": ["alice/objects/ab/cd/<hash>.pdf"]}`. Re-ingesting the same content for the same user yields **`skipped_documents`** with reason `already_ingested`.
- **Query:**  
  `POST http://localhost:8000/api/v1/query` with body e.g.  
  `{"query": "What is in my documents?", "thread_id": "optional-thread-id", "user_id": "optional-user-id"}`.

### Run tests

From **`personal-ai-assistant/`** with venv activated:

```bash
PYTHONPATH=. pytest tests/ -v
```

(Optional) Use a **`.env`** with test-friendly `OPENAI_API_KEY` and `POSTGRES_URI` so health check can connect to Postgres; tests override **getSettings** for the health route so it can run without real keys.

### Run with Docker

Build and run (Postgres must be available at **POSTGRES_URI**):

```bash
docker build -t personal-ai-assistant .
docker run -p 8000:8000 --env-file .env personal-ai-assistant
```

---

## 6. One-page flow summary

- **Entry:** **`backend/main.py`** → creates FastAPI app, registers **apiRouter**, runs **lifespan** (open checkpointer/store, build agent graph, set **app.state.graph**).
- **Routes:** All under **`/api/v1`** (health, tools, documents/upload, documents/ingest, query). Routes live in **backend/api/routes/** and use **backend/api/schemas/** and **backend/core/config** (getSettings).
- **Query flow:** **query.py** gets **app.state.graph** → **ainvoke(state, config)** → graph runs: **remember → planner → retriever → reasoning → (tool)\* → answer** → response built from final state.
- **Storage:** **pgvector** in **database/vector_store.py**; same Postgres has checkpointer (STM), store (LTM) in **database/checkpoint_store.py**, and **ingest_registry** (`user_document_ingest`) for idempotent ingest.
- **RAG:** **rag/ingestion.py** (load + chunk), **rag/retriever.py** (getRetriever, formatDocs), **rag/self_rag_pipeline.py** (runSelfRag); used by the **retriever** node.
- **Tools:** **tools/mcp_client.py** (getMcpTools, invokeTool); used by **reasoning** (tool binding) and **tool** node (invokeTool).
- **Memory:** **agents/memory.py** (LTM models + prompt); **remember** and **answer** nodes use PostgresStore (LTM); **checkpointer** (in **checkpoint_store**) provides STM (conversation history per **thread_id**).

This document is the high-level overview; the plan document remains the single source of truth for detailed behavior and conventions.
