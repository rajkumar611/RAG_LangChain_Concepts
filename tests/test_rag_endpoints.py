"""
Integration tests for RAG API endpoints.
LLM calls are mocked so tests run without an Anthropic API key or network.
"""

import json
import os
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

MOCK_ANSWER = "This is a mocked LLM answer for testing purposes."


def _mock_llm(prompt: str, max_tokens: int = 512) -> str:
    return MOCK_ANSWER


# ── /upload ───────────────────────────────────────────────────────────────────
class TestUpload:
    def test_upload_txt_success(self, client, sample_text):
        resp = client.post(
            "/upload",
            files={"file": ("doc.txt", sample_text.encode(), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "doc.txt"
        assert data["chunks"] > 0

    def test_upload_empty_file_returns_error(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_upload_whitespace_only_returns_error(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("blank.txt", b"   \n\n   ", "text/plain")},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()


# ── /rag/naive ────────────────────────────────────────────────────────────────
class TestNaiveRag:
    def test_no_docs_returns_guidance(self, client):
        # Reset global state to simulate no uploads
        import src.rag.routes as r

        original = r.DOCS[:]
        r.DOCS.clear()
        resp = client.post("/rag/naive", json={"query": "what is RAG?"})
        r.DOCS.extend(original)
        assert resp.status_code == 200
        assert "No document" in resp.json()["answer"]

    def test_returns_expected_keys(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "what is vector search?"})
        assert resp.status_code == 200
        data = resp.json()
        assert {"answer", "docs", "steps"} <= data.keys()

    def test_answer_is_string(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "explain BM25"})
        assert isinstance(resp.json()["answer"], str)

    def test_docs_list_non_empty(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "chunking"})
        assert len(resp.json()["docs"]) > 0

    def test_steps_list_non_empty(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "RAG pipeline"})
        assert len(resp.json()["steps"]) > 0

    def test_query_too_long_rejected(self, client):
        resp = client.post("/rag/naive", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_empty_query_rejected(self, client):
        resp = client.post("/rag/naive", json={"query": ""})
        assert resp.status_code == 422


# ── /rag/hybrid ───────────────────────────────────────────────────────────────
class TestHybridRag:
    def test_returns_both_result_sets(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/hybrid", json={"query": "retrieval"})
        assert resp.status_code == 200
        data = resp.json()
        assert "vector_results" in data
        assert "bm25_results" in data

    def test_docs_have_source_annotation(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/hybrid", json={"query": "embeddings"})
        for doc in resp.json()["docs"]:
            assert "sources" in doc
            assert len(doc["sources"]) > 0


# ── /rag/evaluate ─────────────────────────────────────────────────────────────
class TestRagEvaluate:
    BASE_PAYLOAD = {
        "question": "What is RAG?",
        "answer": "RAG combines retrieval with generation.",
        "contexts": ["RAG stands for Retrieval Augmented Generation."],
    }

    def _mock_llm_score(self, prompt: str, max_tokens: int = 512) -> str:
        return json.dumps({"score": 0.9, "reasoning": "Good answer.", "unsupported_claims": []})

    def test_returns_scores_dict(self, client):
        with patch("src.rag.routes.llm", side_effect=self._mock_llm_score):
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        assert resp.status_code == 200
        assert "scores" in resp.json()

    def test_three_core_metrics_present(self, client):
        with patch("src.rag.routes.llm", side_effect=self._mock_llm_score):
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        scores = resp.json()["scores"]
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_utilization" in scores

    def test_overall_is_average_of_scores(self, client):
        with patch("src.rag.routes.llm", side_effect=self._mock_llm_score):
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        data = resp.json()
        scores = list(data["scores"].values())
        expected_overall = round(sum(scores) / len(scores), 3)
        assert abs(data["overall"] - expected_overall) < 0.001

    def test_correctness_added_when_ground_truth_given(self, client):
        payload = {**self.BASE_PAYLOAD, "ground_truth": "RAG is Retrieval Augmented Generation."}
        with patch("src.rag.routes.llm", side_effect=self._mock_llm_score):
            resp = client.post("/rag/evaluate", json=payload)
        assert "correctness" in resp.json()["scores"]

    def test_scores_bounded_zero_to_one(self, client):
        with patch("src.rag.routes.llm", side_effect=self._mock_llm_score):
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        for val in resp.json()["scores"].values():
            assert 0.0 <= val <= 1.0

    def test_missing_question_rejected(self, client):
        payload = {k: v for k, v in self.BASE_PAYLOAD.items() if k != "question"}
        resp = client.post("/rag/evaluate", json=payload)
        assert resp.status_code == 422

    def test_empty_contexts_rejected(self, client):
        payload = {**self.BASE_PAYLOAD, "contexts": []}
        resp = client.post("/rag/evaluate", json=payload)
        assert resp.status_code == 422
