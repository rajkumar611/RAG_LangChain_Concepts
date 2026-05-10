"""
AI Learning Hub — unified FastAPI app
Run via: python main.py
"""
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Enable LangSmith tracing when API key is present
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "ai-learning-hub")

from src.rag.routes import router as rag_router
from src.langchain_orchestration.routes import router as lc_router

app = FastAPI(title="AI Learning Hub")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(rag_router)
app.include_router(lc_router)

@app.get("/")
def root():
    return FileResponse("frontend/index.html")
