"""
Centralised application settings loaded from environment / .env file.

Import the singleton: `from src.config import settings`
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore unrecognised env vars
    )

    # ── Required ──────────────────────────────────────────────────────────────
    anthropic_api_key: str

    # ── Models ────────────────────────────────────────────────────────────────
    sonnet_model: str = "claude-sonnet-4-6"
    haiku_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── RAG limits ────────────────────────────────────────────────────────────
    max_chunks: int = 300
    max_chunk_chars: int = 400
    max_search_rounds: int = 2  # agentic RAG: max tool-call iterations

    # ── Server ────────────────────────────────────────────────────────────────
    port: int = 8080
    env: str = "development"  # set to "production" to disable reload

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    langchain_api_key: str = ""
    langchain_project: str = "ai-learning-hub"


settings = Settings()
