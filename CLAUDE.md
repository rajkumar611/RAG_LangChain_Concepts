# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**AI Learning Hub** — a FastAPI web app demonstrating 5 RAG strategies, 10 LangChain orchestration patterns, and LLM-based RAG evaluation. A single `frontend/index.html` SPA calls backend endpoints. No database, no auth.

## Running the app

```bash
pip install -r requirements.txt   # first time only (downloads ~80 MB embedding model)
python main.py                    # serves on http://localhost:8000
```

`main.py` loads `.env` then delegates to `uvicorn app:app`. The `.env` must contain `ANTHROPIC_API_KEY`.

Optional: add `LANGCHAIN_API_KEY` to `.env` to enable LangSmith tracing automatically.

## Architecture

```
app.py                                  ← FastAPI app; mounts routers; serves frontend/index.html at /
main.py                                 ← uvicorn entrypoint
frontend/
  index.html                            ← entire SPA (single file)
src/
  rag/
    routes.py                           ← 5 RAG strategies + /upload + /rag/evaluate
  langchain_orchestration/
    routes.py                           ← 10 LangChain concept demos
tests/                                  ← test suite (empty, ready to populate)
```

## RAG module (`src/rag/routes.py`)

Uses the **Anthropic SDK directly** (`anthropic.Anthropic()`, model `claude-sonnet-4-6`).

Global in-memory state holds the uploaded document corpus — **reset on every server restart**. Upload a PDF or TXT via `POST /upload` before calling any RAG endpoint.

| Endpoint | Strategy |
|---|---|
| `POST /rag/naive` | Embed → vector search → generate |
| `POST /rag/advanced` | Query rewrite → hybrid search → RRF → LLM re-rank → generate |
| `POST /rag/agentic` | Tool-calling agent that searches iteratively (up to 2 rounds) |
| `POST /rag/hybrid` | Dense (FAISS cosine) + sparse (TF-IDF BM25) fused via RRF |
| `POST /rag/graph` | Seed retrieval + 2-hop BFS on a sequential graph → re-score |
| `POST /rag/evaluate` | LLM-as-Judge: Faithfulness, Answer Relevancy, Context Utilization, optional Correctness |

Shared utilities: `vsearch`, `bsearch`, `rrf`, `chunk_text`, `llm`, `rebuild_indexes`, `no_docs_response`, `ctx_prompt`.

Global state: `DOCS`, `DOC_EMBS`, `TFIDF_MAT`, `G`, `embedder` (`all-MiniLM-L6-v2`, 384-dim), `tfidf`.

Upload limits: max 300 chunks, max chunk size 400 chars.

## LangChain module (`src/langchain_orchestration/routes.py`)

Uses **`langchain-anthropic`** (`ChatAnthropic`, model `claude-haiku-4-5-20251001`).

All imports are deferred (inside each route function) — keeps startup fast.

| Endpoint | Concept |
|---|---|
| `POST /langchain/prompt` | Prompt templates + `StrOutputParser` |
| `POST /langchain/chaining` | Sequential 3-step chain (translate → summarise → JSON) |
| `POST /langchain/rag` | RAG with a fixed 7-doc FAISS vectorstore (lazy-loaded, in-memory) |
| `POST /langchain/memory` | Per-session conversation history via `MessagesPlaceholder` |
| `POST /langchain/tools` | `bind_tools` with calculator / weather / word-count |
| `POST /langchain/documents` | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` |
| `POST /langchain/parsers` | `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser` |
| `POST /langchain/agent` | ReAct agent via `langgraph.prebuilt.create_react_agent` |
| `POST /langchain/multiagent` | Two sequential LLM calls (researcher → blog writer) |
| `POST /langchain/langgraph` | `StateGraph` with manager → research → writer → reviewer + conditional edge (max 2 revisions) |

`LC_SESSIONS` (dict) stores per-session chat history — cleared with `DELETE /langchain/memory/{session_id}`.

## Key constraints

- **All state is in-memory.** Restarting the server clears uploaded docs and chat sessions.
- RAG module uses `claude-sonnet-4-6`; LangChain module uses `claude-haiku-4-5-20251001` — don't conflate.
- The LangChain fixed vectorstore (`_lc_vectorstore`) contains only 7 hardcoded AI/ML facts; it is not the uploaded document store.
- `frontend/index.html` is a single-file SPA — all JS, CSS, and HTML in one file. Edit it directly.
- `app.py` enables CORS with wildcard (`allow_origins=["*"]`) — fine for local dev, not for production.
- LangSmith tracing activates automatically when `LANGCHAIN_API_KEY` is set in `.env`.
