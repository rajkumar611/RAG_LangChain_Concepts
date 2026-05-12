"""
Unit tests for RAG utility functions.
These tests do NOT call the LLM — they verify retrieval and chunking logic only.
"""
import pytest

# Ensure dummy key is set before importing routes
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

from src.rag.routes import (
    chunk_text, MAX_CHUNKS,
    rebuild_indexes, vector_search, bm25_search, reciprocal_rank_fusion,
)


# ── chunk_text ────────────────────────────────────────────────────────────────
class TestChunkText:

    def test_basic_paragraph_split(self):
        text = "First paragraph with enough content here.\n\nSecond paragraph with enough content here."
        chunks = chunk_text(text)
        assert len(chunks) == 2

    def test_short_paragraphs_skipped(self):
        text = "Hi.\n\nThis paragraph is long enough to be kept as a chunk for sure."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert "long enough" in chunks[0]

    def test_long_paragraph_split_at_sentence(self):
        sentence = "This is a sentence with some words. "
        long_para = sentence * 20          # ~720 chars — exceeds MAX_CHUNK_CHARS=400
        chunks = chunk_text(long_para, max_chars=400)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 400

    def test_empty_string_returns_empty(self):
        assert chunk_text("") == []

    def test_caps_at_max_chunks(self):
        # Generate more paragraphs than MAX_CHUNKS
        text = "\n\n".join(
            f"Paragraph {i}: This paragraph has enough words to pass the length filter."
            for i in range(MAX_CHUNKS + 50)
        )
        chunks = chunk_text(text)
        assert len(chunks) == MAX_CHUNKS

    def test_excessive_blank_lines_collapsed(self):
        # Paragraphs must be >40 chars to pass the length filter
        text = (
            "First paragraph is long enough to pass the forty-char filter.\n\n\n\n\n"
            "Second paragraph is long enough to pass the forty-char filter."
        )
        chunks = chunk_text(text)
        assert len(chunks) == 2


# ── vector_search / bm25_search ───────────────────────────────────────────────
class TestSearchFunctions:

    @pytest.fixture(autouse=True)
    def _seed_indexes(self):
        """Seed the global indexes with known documents before each test."""
        docs = [
            "RAG combines retrieval with language model generation.",
            "Vector search uses cosine similarity on dense embeddings.",
            "BM25 is a sparse keyword-based retrieval algorithm.",
            "FAISS enables fast approximate nearest-neighbour search.",
            "Chunking splits documents into smaller retrieval units.",
        ]
        rebuild_indexes(docs)

    def test_vector_search_returns_k_results(self):
        results = vector_search("retrieval augmented generation", k=3)
        assert len(results) == 3

    def test_vector_search_result_schema(self):
        result = vector_search("embeddings", k=1)[0]
        assert "id" in result
        assert "text" in result
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_vector_search_most_relevant_first(self):
        results = vector_search("cosine similarity embeddings", k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_search_returns_k_results(self):
        results = bm25_search("keyword retrieval BM25", k=3)
        assert len(results) == 3

    def test_bm25_search_result_schema(self):
        result = bm25_search("BM25", k=1)[0]
        assert {"id", "text", "score"} <= result.keys()

    def test_bm25_exact_keyword_scores_highest(self):
        results = bm25_search("BM25 sparse keyword", k=5)
        top_text = results[0]["text"]
        assert "BM25" in top_text


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────
class TestReciprocalRankFusion:

    def _make_list(self, ids: list[int], texts: list[str]) -> list[dict]:
        return [{"id": i, "text": t, "score": 1.0} for i, t in zip(ids, texts)]

    def test_single_list_passthrough(self):
        rebuild_indexes(["doc zero", "doc one", "doc two"])
        ranked = self._make_list([0, 1, 2], ["doc zero", "doc one", "doc two"])
        fused = reciprocal_rank_fusion([ranked])
        assert [r["id"] for r in fused] == [0, 1, 2]

    def test_two_lists_boost_overlap(self):
        # TF-IDF requires real words (not single chars or pure stop-words)
        rebuild_indexes(["alpha document", "beta document", "gamma document", "delta document"])
        list1 = self._make_list([0, 1, 2], ["alpha document", "beta document", "gamma document"])
        list2 = self._make_list([2, 0, 3], ["gamma document", "alpha document", "delta document"])
        fused = reciprocal_rank_fusion([list1, list2])
        # doc 0 and doc 2 appear in both lists — should rank higher than doc 3
        fused_ids = [r["id"] for r in fused]
        assert fused_ids.index(3) > fused_ids.index(0)
        assert fused_ids.index(3) > fused_ids.index(2)

    def test_scores_descending(self):
        rebuild_indexes(["alpha document", "beta document", "gamma document", "delta document"])
        list1 = self._make_list([0, 1, 2, 3], ["alpha document", "beta document", "gamma document", "delta document"])
        list2 = self._make_list([3, 2, 1, 0], ["delta document", "gamma document", "beta document", "alpha document"])
        fused = reciprocal_rank_fusion([list1, list2])
        scores = [r["score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []
