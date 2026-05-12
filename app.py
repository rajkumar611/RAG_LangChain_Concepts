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
from src.rag.routes import router as rag_router, DOCS
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
