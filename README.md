# RAG Evaluation Suite

An interactive FastAPI application demonstrating **5 RAG retrieval strategies** and **LLM-based RAG evaluation** — built to production-grade standards with a multi-page guided frontend.

![CI](https://github.com/your-username/RAG_EVALUATION_SUITE/actions/workflows/ci.yml/badge.svg)

---

## Why This Project

Most RAG demos are toy examples. This project shows:
- **Retrieval depth** — five strategies from naive to graph-based, each with visible pipeline steps
- **Pass / Fail sample documents** — purpose-built documents that demonstrate exactly where each strategy succeeds and where it breaks down
- **Evaluation** — LLM-as-Judge scoring across Faithfulness, Answer Relevancy, Context Utilization, and optional Correctness
- **Production rigour** — typed config, structured logging, input validation, 35 tests, CI pipeline, Docker support

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              Browser (SPA)                   │
│   index.html  +  frontend/pages/             │
│                                              │
│  Home │ Naive │ Advanced │ Agentic │ Hybrid  │
│                Graph │ Evaluation            │
└─────────────────────┬────────────────────────┘
                      │ HTTP
            ┌─────────▼─────────┐
            │   rag/routes.py   │
            │                   │
            │  POST /upload     │
            │  POST /rag/naive  │
            │  POST /rag/advanced│
            │  POST /rag/agentic│
            │  POST /rag/hybrid │
            │  POST /rag/graph  │
            │  POST /rag/evaluate│
            └─────────┬─────────┘
                      │
            ┌─────────▼─────────┐
            │   Anthropic SDK   │
            │  claude-sonnet-4-6│
            └───────────────────┘
```

---

## RAG Strategies

| Strategy | When to Use | Key Technique |
|---|---|---|
| **Naive RAG** | Baseline / simple Q&A | Embed → cosine search → generate |
| **Advanced RAG** | Production Q&A | Query rewrite + hybrid search + LLM re-rank |
| **Agentic RAG** | Multi-hop questions | Tool-calling agent with iterative search |
| **Hybrid RAG** | Keyword + semantic needs | Dense (FAISS) + Sparse (BM25) fused via RRF |
| **Graph RAG** | Relational / connected data | Seed retrieval + BFS graph expansion |

Each strategy page explains the pipeline visually, shows step-by-step execution results, and ships with dedicated sample documents to demonstrate pass and fail behaviour.

---

## RAG Evaluation (LLM-as-Judge)

Four RAGAS-aligned metrics evaluated by Claude:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (hallucination detection) |
| **Answer Relevancy** | Does the answer address the question that was asked? |
| **Context Utilization** | Did the retrieved chunks actually contain the right information? |
| **Correctness** *(optional)* | Accuracy against a known ground truth answer |

---

## Sample Documents

Purpose-built pass / fail document pairs for each strategy. Each set includes a `_prompt.txt` explaining exactly what behaviour to expect and why.

| Strategy | Pass Document | Fail Document | Notes |
|---|---|---|---|
| Naive RAG | `naive_rag_pass.txt` | `naive_rag_fail.txt` | Fail: answer split across distant sections |
| Advanced RAG | `advanced_rag_pass.txt` | `advanced_rag_fail.txt` | Fail: causal chain spread across 5+ sections |
| Agentic RAG | `agentic_rag_pass.txt` | `agentic_rag_fail.txt` | Fail: two similar events create conflicting context |
| Hybrid RAG | — | — | Use any of the above |
| Graph RAG | `graph_rag_pass.txt` | `graph_rag_fail.txt` | Also includes 4-document multi-hop scenario |

General purpose documents (`doc1_ai_research_report.txt`, `doc2_product_operations_manual.txt`) work across all strategies.

---

## Quick Start

### Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. Run
python main.py
# → http://localhost:8080
```

### Docker

```bash
# Build and run (reads ANTHROPIC_API_KEY from .env automatically)
docker compose up

# Or without compose
docker build -t rag-eval .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-... rag-eval
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests run fully offline — all LLM calls are mocked. No API key required.

```
tests/test_rag_utils.py      # Unit tests: chunk_text, vector_search, bm25_search, reciprocal_rank_fusion
tests/test_rag_endpoints.py  # Integration tests: /upload, /rag/naive, /rag/hybrid, /rag/evaluate
```

---

## Project Structure

```
.
├── app.py                              # FastAPI app — routers, logging, /health endpoint
├── main.py                             # Uvicorn entrypoint with startup validation
├── requirements.txt
├── pyproject.toml                      # Pytest, ruff (lint + format), mypy config
├── Dockerfile                          # Multi-stage build; pre-downloads embedding model
├── docker-compose.yml
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # Runs tests + lint on every push / PR
│
├── src/
│   ├── config.py                       # Pydantic BaseSettings — single source of truth for all env vars
│   └── rag/
│       └── routes.py                   # 5 RAG strategies + /upload + /rag/evaluate
│
├── frontend/
│   ├── index.html                      # SPA shell — navigation and page routing
│   └── pages/
│       ├── home.html                   # Overview: what RAG is, how strategies differ
│       ├── naive-rag.html
│       ├── advanced-rag.html
│       ├── agentic-rag.html
│       ├── hybrid-rag.html
│       ├── graph-rag.html
│       ├── rag-evaluation.html
│       └── summary.html
│
├── sample_docs/
│   ├── naive_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── advanced_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── agentic_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── graph_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── graph_rag_doc1_teams.txt        # Multi-document Graph RAG scenario
│   ├── graph_rag_doc2_tech_stack.txt
│   ├── graph_rag_doc3_projects.txt
│   ├── graph_rag_doc4_security.txt
│   ├── doc1_ai_research_report.txt     # General purpose documents
│   └── doc2_product_operations_manual.txt
│
└── tests/
    ├── conftest.py                     # Shared fixtures (client, sample_text, uploaded_client)
    ├── test_rag_utils.py               # 16 utility unit tests
    └── test_rag_endpoints.py           # 19 endpoint integration tests
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend SPA |
| `GET` | `/health` | Liveness probe — uptime, doc count, model name |
| `POST` | `/upload` | Upload PDF or TXT; rebuilds all indexes |
| `POST` | `/rag/naive` | Baseline RAG |
| `POST` | `/rag/advanced` | Advanced RAG with query rewrite and re-ranking |
| `POST` | `/rag/agentic` | Agentic tool-calling RAG |
| `POST` | `/rag/hybrid` | Hybrid dense + sparse RAG |
| `POST` | `/rag/graph` | Graph-expanded RAG |
| `POST` | `/rag/evaluate` | LLM-as-Judge evaluation |

Full interactive docs: `http://localhost:8080/docs`

---

## Key Design Decisions

- **LLM-as-Judge over RAGAS library** — avoids heavy dependencies; the same evaluation principle implemented directly with the Anthropic SDK
- **All state in-memory** — intentional for simplicity; upload a document before querying (state resets on server restart)
- **Typed config** — `src/config.py` (pydantic-settings) validates all env vars at startup; the app exits immediately with a clear message if `ANTHROPIC_API_KEY` is missing
- **Pass / Fail sample documents** — each strategy ships with documents designed to expose the strategy's specific failure mode, not just documents that work

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Search | FAISS |
| Sparse Search | TF-IDF / BM25 (scikit-learn) |
| Graph | NetworkX |
| Config | Pydantic Settings |
| Observability | Structured logging |
| Testing | pytest + FastAPI TestClient |
| Containerisation | Docker (multi-stage) + Docker Compose |
| CI | GitHub Actions |

---

## Author

**Raj Kumar** — [rajkumar.novsix@gmail.com](mailto:rajkumar.novsix@gmail.com)
