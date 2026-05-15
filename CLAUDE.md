# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**RAG Evaluation Suite** — a FastAPI web app demonstrating 5 RAG strategies, 10 LangChain orchestration patterns, and LLM-based RAG evaluation. A single `frontend/index.html` SPA calls backend endpoints. No database, no auth.

## Running the app

```bash
pip install -r requirements.txt   # first time only (downloads ~80 MB embedding model)
python main.py                    # serves on http://localhost:8080
```

`main.py` loads `.env`, validates `ANTHROPIC_API_KEY` is set (exits with a clear message if not), then starts uvicorn. `reload` is enabled only when `ENV=development` (the default).

Optional: add `LANGCHAIN_API_KEY` to `.env` to enable LangSmith tracing automatically.

### Docker

```bash
docker compose up          # reads .env, sets ENV=production, healthchecks /health
```

### Tests

```bash
python -m pytest tests/ -v     # 91 tests, all LLM/RAGAS calls mocked, no API key needed
```

## Architecture

```
app.py                                  ← FastAPI app; mounts routers; /health; logging setup
main.py                                 ← uvicorn entrypoint; startup validation
src/
  config.py                             ← Pydantic BaseSettings — single source of truth for all env vars
  rag/
    routes.py                           ← 5 RAG strategies + /upload + /rag/evaluate
  langchain_orchestration/
    routes.py                           ← 10 LangChain concept demos
frontend/
  index.html                            ← entire SPA (single file)
tests/
  conftest.py                           ← shared fixtures: client, sample_text, uploaded_client
  test_rag_utils.py                     ← 32 unit tests for chunking, retrieval, and injection detection
  test_rag_endpoints.py                 ← 59 endpoint integration tests (LLM + RAGAS mocked)
Dockerfile                              ← multi-stage build; pre-downloads embedding model
docker-compose.yml                      ← local dev; healthcheck wired to /health
pyproject.toml                          ← pytest, ruff, mypy config
.github/workflows/ci.yml                ← runs tests + lint on every push/PR
```

## Configuration (`src/config.py`)

All settings are defined as a `pydantic_settings.BaseSettings` subclass and imported as `from src.config import settings`. Do **not** use `os.getenv()` directly in route files — add new settings to `config.py` instead.

| Setting | Default | Description |
|---|---|---|
| `anthropic_api_key` | *(required)* | Anthropic API key |
| `sonnet_model` | `claude-sonnet-4-6` | Model used by RAG module |
| `haiku_model` | `claude-haiku-4-5-20251001` | Model used by LangChain module |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `max_chunks` | `300` | Max chunks stored per upload |
| `max_chunk_chars` | `400` | Max chars per chunk |
| `max_search_rounds` | `2` | Max agentic RAG tool-call iterations |
| `port` | `8080` | Server port |
| `env` | `development` | Set to `production` to disable reload |

## RAG module (`src/rag/routes.py`)

Uses the **Anthropic SDK directly** (`anthropic.Anthropic()`, model `claude-sonnet-4-6`).

Global in-memory state holds the uploaded document corpus — **reset on every server restart**. Upload a PDF or TXT via `POST /upload` before calling any RAG endpoint.

| Endpoint | Strategy |
|---|---|
| `POST /rag/naive` | Embed → vector search → generate |
| `POST /rag/advanced` | Query rewrite → hybrid search → RRF → LLM re-rank → generate |
| `POST /rag/agentic` | Tool-calling agent that searches iteratively (up to `max_search_rounds`) |
| `POST /rag/hybrid` | Dense (FAISS cosine) + sparse (TF-IDF BM25) fused via RRF |
| `POST /rag/graph` | Seed retrieval + 2-hop BFS on a sequential graph → re-score |
| `POST /rag/evaluate` | RAGAS evaluation: Faithfulness + Answer Relevancy always; Context Precision + Answer Correctness when `ground_truth` is provided |

**Shared utilities** (all have docstrings):

| Function | Purpose |
|---|---|
| `vector_search(query, k)` | Dense cosine similarity search over `DOC_EMBS` |
| `bm25_search(query, k)` | Sparse TF-IDF keyword search over `TFIDF_MAT` |
| `reciprocal_rank_fusion(lists, k=60)` | Fuses multiple ranked lists; k=60 is the standard RRF constant |
| `chunk_text(text, max_chars)` | Splits text at paragraph then sentence boundaries |
| `llm(prompt, max_tokens)` | Single-turn Claude call; raises `RuntimeError` on failure |
| `rebuild_indexes(docs)` | Rebuilds all globals from a new doc list |
| `no_docs_response()` | Standard response when no document is uploaded |
| `ctx_prompt(docs, question)` | Builds the RAG context+question prompt |
| `RAG_SYSTEM` | System prompt passed to all RAG answer generation calls — instructs the model to stay within retrieved context and acknowledge gaps rather than using training knowledge |
| `_check_prompt_injection(text)` | Regex scan for injection patterns in a query; returns `{"flagged", "reason"}` |
| `_check_indirect_injection(chunks)` | Scans all document chunks for embedded injection before indexing; returns `{"flagged", "reason"}` |
| `_faithfulness_guardrail(answer, docs, threshold)` | Inline faithfulness check after generation; scores grounding via LLM JSON prompt; threshold 0.7; returns `{"passed", "faithfulness_score", "warning"}` |
| `_get_ragas_resources()` | Lazily initialises and caches RAGAS LLM + embeddings wrappers (called on first `/rag/evaluate` request) |

**Request models**: `QueryRequest` (field: `query`), `EvaluationRequest` (fields: `question`, `answer`, `contexts`, `ground_truth`).

Global state: `DOCS`, `DOC_EMBS`, `TFIDF_MAT`, `G`, `embedder` (`all-MiniLM-L6-v2`, 384-dim), `tfidf`, `_ragas_llm`, `_ragas_embeddings`.

**RAGAS evaluation** (`POST /rag/evaluate`) uses the real [ragas](https://github.com/explodinggradients/ragas) library (v0.4.x) with Claude as the judge LLM via `llm_factory(model, provider="anthropic", client=Anthropic(...))`. Resources are lazily initialised on the first call. Response keys map RAGAS metric names: `context_precision` → `context_utilization`, `answer_correctness` → `correctness`.

## LangChain module (`src/langchain_orchestration/routes.py`)

Uses **`langchain-anthropic`** (`ChatAnthropic`, model `claude-haiku-4-5-20251001`).

All LangChain imports are deferred (inside each route function) — keeps startup fast.

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

**Request models**: `PromptRequest`, `TextRequest`, `QuestionRequest`, `TopicRequest`, `MemoryRequest`.

`LC_SESSIONS` (dict) stores per-session chat history — cleared with `DELETE /langchain/memory/{session_id}`.

**Calculator tool**: Uses a whitelist-only AST walker (`_safe_eval_math`) — **no `eval()` anywhere**. Only permits `+`, `-`, `*`, `/`, `**`, and numeric literals. Input is pre-validated by regex before parsing.

## Health endpoint

`GET /health` — returns uptime, `len(DOCS)`, `len(LC_SESSIONS)`, and model names. No external calls. Used as the Docker healthcheck.

## Key constraints

- **All state is in-memory.** Restarting the server clears uploaded docs and chat sessions.
- RAG module uses `claude-sonnet-4-6`; LangChain module uses `claude-haiku-4-5-20251001` — don't conflate.
- The LangChain fixed vectorstore (`_lc_vectorstore`) contains only 7 hardcoded AI/ML facts; it is **not** the uploaded document store.
- `frontend/index.html` is a single-file SPA — all JS, CSS, and HTML in one file. Edit it directly.
- `app.py` enables CORS with wildcard (`allow_origins=["*"]`) — fine for local dev, not for production.
- LangSmith tracing activates automatically when `LANGCHAIN_API_KEY` is set in `.env`.
- All Pydantic request models enforce `min_length` / `max_length` constraints — do not remove these.
- Every route endpoint wraps its body in `try/except` and returns `JSONResponse(status_code=500)` on failure — maintain this pattern when adding new endpoints.

## Adding a new RAG endpoint

1. Add a route function in `src/rag/routes.py` with a docstring explaining the strategy.
2. Use `QueryRequest` as the request model (or extend it if needed).
3. Return `no_docs_response()` if `not DOCS`.
4. Wrap the body in `try/except RuntimeError` → `JSONResponse(500)`.
5. Use `vector_search`, `bm25_search`, `reciprocal_rank_fusion`, `llm`, `ctx_prompt` — don't duplicate logic.
6. Add endpoint tests to `tests/test_rag_endpoints.py` — mock `src.rag.routes.llm` with `unittest.mock.patch`.

## Testing the evaluation endpoint

`/rag/evaluate` uses the real RAGAS library. Tests must mock both the RAGAS resources and the `evaluate()` call. See `_ragas_patches()` helper in `test_rag_endpoints.py` for the required set of patches:
- `src.rag.routes._get_ragas_resources` — returns `(MagicMock(), MagicMock())`
- `ragas.evaluate` — returns a mock result with `.scores = [dict]`
- Metric classes (`ragas.metrics.collections.*`) — replaced with `MagicMock()` to bypass LLM type validation in their constructors
