"""
Pytest fixtures shared across the test suite.
"""

import os

import pytest
from fastapi.testclient import TestClient

# Provide a dummy key so Settings validation passes without a real .env
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient — imports the app once per test session."""
    from app import app

    return TestClient(app)


@pytest.fixture(scope="session")
def sample_text():
    """A multi-paragraph text used to test chunking and upload."""
    return (
        "Retrieval Augmented Generation (RAG) is a technique that combines "
        "information retrieval with text generation. It retrieves relevant "
        "documents from a knowledge base and uses them as context for the LLM.\n\n"
        "Vector search converts text into dense numerical embeddings and finds "
        "semantically similar documents using cosine similarity. This allows "
        "retrieval based on meaning rather than exact keyword matches.\n\n"
        "BM25 is a sparse retrieval method that scores documents by term "
        "frequency and inverse document frequency. It is fast and effective "
        "for keyword-based queries where exact terms matter.\n\n"
        "Reciprocal Rank Fusion combines multiple ranked lists into a single "
        "list by summing reciprocal ranks. The formula 1/(rank + k) with k=60 "
        "prevents any single list from dominating the final ranking.\n\n"
        "LLM-as-Judge evaluation uses a language model to score generated "
        "answers on dimensions like faithfulness, relevancy, and correctness. "
        "It is more flexible than BLEU or ROUGE for open-ended answers."
    )


@pytest.fixture(scope="session")
def uploaded_client(client, sample_text):
    """TestClient with a document already uploaded — ready for RAG endpoint tests."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", sample_text.encode(), "text/plain")},
    )
    assert response.status_code == 200, f"Upload failed: {response.text}"
    assert response.json()["chunks"] > 0
    return client
