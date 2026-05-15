"""
RAG Evaluation Suite — unified FastAPI app
Run via: python main.py
"""
import logging
import os
import time
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Enable LangSmith tracing when API key is present
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "rag-evaluation-suite")

from src.config import settings
from src.rag.routes import router as rag_router, DOCS, INJECTION_PATTERN_DESCRIPTIONS
from src.langchain_orchestration.routes import router as lc_router, LC_SESSIONS

_START_TIME = time.time()

app = FastAPI(title="RAG Evaluation Suite")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(rag_router)
app.include_router(lc_router)
app.mount("/pages", StaticFiles(directory="frontend/pages"), name="pages")


@app.get("/")
def root():
    return FileResponse("frontend/index.html")


@app.get("/governance")
def governance():
    """Policy snapshot — every rule this app enforces, live from current config.

    No hardcoded values: thresholds and limits are read directly from settings,
    so the response always reflects what the system is actually doing.
    """
    return {
        "version": "1.0",
        "security_filters": {
            "prompt_injection": {
                "enabled": True,
                "scope": "user query — checked before retrieval",
                "action": "block — HTTP 400, no retrieval or LLM call made",
                "patterns": INJECTION_PATTERN_DESCRIPTIONS,
            },
            "indirect_injection": {
                "enabled": True,
                "scope": "uploaded document — checked per chunk before indexing",
                "action": "block — HTTP 400, document not indexed",
                "patterns": "same pattern set as prompt_injection",
            },
        },
        "guardrails": {
            "faithfulness": {
                "enabled": True,
                "scope": "all RAG answers — runs after every generation",
                "threshold": settings.faithfulness_threshold,
                "model": settings.sonnet_model,
                "action": "warn — amber indicator + HITL suggestions shown in UI",
            },
        },
        "human_in_the_loop": {
            "trigger": f"faithfulness_score < {settings.faithfulness_threshold}",
            "response": "UI surfaces three actionable next steps for the user",
        },
        "models": {
            "rag": settings.sonnet_model,
            "evaluation": settings.haiku_model,
            "embeddings": settings.embedding_model,
        },
        "limits": {
            "max_query_length": 2000,
            "max_chunks_per_upload": settings.max_chunks,
            "max_chunk_chars": settings.max_chunk_chars,
            "max_agentic_search_rounds": settings.max_search_rounds,
        },
    }


@app.get("/health")
def health():
    """Liveness probe — returns runtime state without calling any external service."""
    return {
        "status":          "ok",
        "uptime_seconds":  round(time.time() - _START_TIME, 1),
        "docs_loaded":     len(DOCS),
        "active_sessions": len(LC_SESSIONS),
        "models": {
            "rag":       settings.sonnet_model,
            "langchain": settings.haiku_model,
            "embedding": settings.embedding_model,
        },
    }
