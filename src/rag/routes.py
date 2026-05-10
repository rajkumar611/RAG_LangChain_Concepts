import io
import json
import os
import re

import numpy as np
import networkx as nx

# Suppress HuggingFace token warnings — embedder runs fully offline
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cos

router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
CLAUDE      = os.getenv("SONNET_MODEL", "claude-sonnet-4-6")
EMB_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MAX_CHUNKS      = 300
MAX_CHUNK_CHARS = 400
MAX_SEARCH_ROUNDS = 2   # agentic RAG: max tool-call iterations before forcing a final answer

# ── Clients & models ──────────────────────────────────────────────────────────
print(f"Loading embedding model '{EMB_MODEL}' (first run downloads ~80 MB)…")
embedder = SentenceTransformer(EMB_MODEL)
client   = anthropic.Anthropic()

# ── Global in-memory state (reset on server restart) ─────────────────────────
DOCS: list[str]      = []
DOC_EMBS: np.ndarray = np.array([])
TFIDF_MAT            = None
G                    = nx.Graph()
tfidf                = TfidfVectorizer(stop_words="english")


# ── Request / response models ─────────────────────────────────────────────────
class Q(BaseModel):
    query: str

class EvalReq(BaseModel):
    question:    str
    answer:      str
    contexts:    list[str]
    ground_truth: str = ""


# ── Shared utilities ──────────────────────────────────────────────────────────
def rebuild_indexes(docs: list[str]):
    global DOCS, DOC_EMBS, TFIDF_MAT
    DOCS      = docs
    DOC_EMBS  = embedder.encode(DOCS, normalize_embeddings=True)
    TFIDF_MAT = tfidf.fit_transform(DOCS)
    G.clear()
    for i, text in enumerate(DOCS):
        G.add_node(f"d{i}", text=text)
    for i in range(len(DOCS) - 1):
        G.add_edge(f"d{i}", f"d{i+1}")


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    text  = re.sub(r'\n{3,}', '\n\n', text.strip())
    paras = [p.strip() for p in re.split(r'\n\n+', text) if len(p.strip()) > 40]
    chunks: list[str] = []
    for para in paras:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            cur = ""
            for sent in re.split(r'(?<=[.!?])\s+', para):
                if len(cur) + len(sent) + 1 <= max_chars:
                    cur = (cur + " " + sent).strip() if cur else sent
                else:
                    if cur:
                        chunks.append(cur)
                    cur = sent
            if cur:
                chunks.append(cur)
    return chunks[:MAX_CHUNKS]


def vsearch(query: str, k: int = 5) -> list[dict]:
    """Dense vector search — cosine similarity over document embeddings."""
    qe     = embedder.encode([query], normalize_embeddings=True)
    scores = (qe @ DOC_EMBS.T)[0]
    top    = np.argsort(scores)[::-1][:k]
    return [{"id": int(i), "text": DOCS[i], "score": round(float(scores[i]), 4)} for i in top]


def bsearch(query: str, k: int = 5) -> list[dict]:
    """Sparse keyword search — TF-IDF / BM25-style cosine similarity."""
    qv     = tfidf.transform([query])
    scores = sk_cos(qv, TFIDF_MAT)[0]
    top    = np.argsort(scores)[::-1][:k]
    return [{"id": int(i), "text": DOCS[i], "score": round(float(scores[i]), 4)} for i in top]


def rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion — combines multiple ranked lists into one."""
    scores: dict[int, float] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            scores[item["id"]] = scores.get(item["id"], 0) + 1 / (rank + k)
    order = sorted(scores, key=scores.__getitem__, reverse=True)
    return [{"id": i, "text": DOCS[i], "score": round(scores[i], 6)} for i in order]


def llm(prompt: str, max_tokens: int = 512) -> str:
    """Send a single-turn prompt to Claude and return the text response."""
    try:
        r = client.messages.create(
            model=CLAUDE,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e


def no_docs_response() -> dict:
    return {"answer": "No document uploaded yet. Please upload a PDF or TXT file first.",
            "docs": [], "steps": []}


def ctx_prompt(docs: list[dict], question: str) -> str:
    ctx = "\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer in 2-3 sentences:"


# ── Upload ────────────────────────────────────────────────────────────────────
@router.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()
    fname   = file.filename or ""

    if fname.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text   = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return {"error": f"PDF error: {e}"}
    else:
        text = content.decode("utf-8", errors="ignore")

    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No readable text found in file."}

    rebuild_indexes(chunks)
    return {"filename": fname, "chunks": len(chunks)}


# ════════════════════════════════════════════════════════════════════════════════
# 1 · NAIVE RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/naive")
def naive_rag(q: Q):
    if not DOCS:
        return no_docs_response()

    steps = [{"step": "1. Embed Query",     "detail": f'Encode "{q.query}" → 384-dim vector'}]
    docs  = vsearch(q.query, k=3)
    steps.append({"step": "2. Vector Search",  "detail": f"Cosine similarity over {len(DOCS)} docs → top 3"})
    steps.append({"step": "3. Augment Prompt", "detail": "Prepend top-3 chunks to the user question"})
    ans = llm(ctx_prompt(docs, q.query))
    steps.append({"step": "4. Generate", "detail": "LLM reads augmented prompt → answer"})
    return {"answer": ans, "docs": docs, "steps": steps}


# ════════════════════════════════════════════════════════════════════════════════
# 2 · ADVANCED RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/advanced")
def advanced_rag(q: Q):
    if not DOCS:
        return no_docs_response()
    steps = []
    try:
        rewritten = llm(
            f"Rewrite this query to be specific and retrieval-optimized. "
            f"Return ONLY the rewritten query.\n\nOriginal: {q.query}",
            max_tokens=128,
        )
        steps.append({"step": "1. Query Rewriting", "detail": f'"{q.query}" → "{rewritten}"'})

        vr = vsearch(rewritten, k=5)
        br = bsearch(rewritten, k=5)
        steps.append({"step": "2. Hybrid Search", "detail": "Dense vector + sparse BM25 in parallel"})

        fused = rrf([vr, br])[:5]
        steps.append({"step": "3. RRF Fusion", "detail": "Reciprocal Rank Fusion: score = Σ 1/(rank+60)"})

        passages = "\n".join(f"ID {d['id']}: {d['text']}" for d in fused)
        raw = llm(
            f"Score each passage for relevance to '{q.query}' (0-10). "
            f"Return JSON array [{{\"id\": N, \"score\": X}}]. JSON only.\n\n{passages}",
            max_tokens=256,
        )
        try:
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                id2sc = {r["id"]: r["score"] for r in json.loads(m.group())}
                fused.sort(key=lambda d: id2sc.get(d["id"], 0), reverse=True)
        except Exception:
            pass
        steps.append({"step": "4. LLM Re-ranking", "detail": "LLM scores each candidate — precision over recall"})

        docs = fused[:3]
        ans  = llm(ctx_prompt(docs, q.query))
        steps.append({"step": "5. Generate", "detail": "Answer from rewritten query + re-ranked context"})
        return {"answer": ans, "docs": docs, "steps": steps, "rewritten_query": rewritten}
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ════════════════════════════════════════════════════════════════════════════════
# 3 · AGENTIC RAG
# ════════════════════════════════════════════════════════════════════════════════
TOOLS = [{
    "name": "search_knowledge_base",
    "description": "Search the knowledge base for relevant context. Search once or twice, then synthesize a final answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    },
}]

AGENT_SYSTEM = (
    "You are a helpful assistant with access to a knowledge base search tool. "
    "Search 1-2 times to gather relevant context, then provide a clear, concise answer. "
    "Do not search more than twice. After searching, always end with a final answer."
)

@router.post("/rag/agentic")
def agentic_rag(q: Q):
    if not DOCS:
        return no_docs_response()

    steps        = [{"step": "0. Init Agent", "detail": f'Agent receives task: "{q.query}"'}]
    all_docs:  list[dict] = []
    messages   = [{"role": "user", "content": f"Question: {q.query}"}]
    final_answer = ""

    try:
        for turn in range(MAX_SEARCH_ROUNDS):
            resp      = client.messages.create(
                model=CLAUDE, max_tokens=2048,
                system=AGENT_SYSTEM, tools=TOOLS, messages=messages,
            )
            texts     = [b.text for b in resp.content if hasattr(b, "text") and b.text]
            tool_uses = [b for b in resp.content if b.type == "tool_use"]

            if resp.stop_reason in ("end_turn", "max_tokens") or not tool_uses:
                final_answer = texts[0] if texts else "No answer generated."
                steps.append({"step": f"{turn+1}. Done", "detail": "Agent answered directly"})
                break

            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            for tu in tool_uses:
                sq      = tu.input.get("query", q.query)
                tk      = min(int(tu.input.get("top_k", 3)), 5)
                results = vsearch(sq, k=tk)
                all_docs.extend(results)
                steps.append({
                    "step": f"{turn+1}. Tool Call",
                    "detail": f'search_knowledge_base(query="{sq}", top_k={tk}) → {len(results)} hits',
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps([{"id": r["id"], "text": r["text"]} for r in results]),
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            # Max rounds reached — force a plain-text final answer
            messages.append({"role": "user", "content": "Now provide your final answer based on the search results above."})
            forced       = client.messages.create(model=CLAUDE, max_tokens=2048, system=AGENT_SYSTEM, messages=messages)
            final_answer = forced.content[0].text.strip() if forced.content else "No answer generated."
            steps.append({"step": "Final. Forced Answer", "detail": "Agent prompted to synthesize after max search rounds"})

    except anthropic.APIError as e:
        return JSONResponse(status_code=500, content={"detail": f"Anthropic API error: {e}"})

    # Deduplicate retrieved docs, preserving order
    seen: set[int] = set()
    unique: list[dict] = []
    for d in all_docs:
        if d["id"] not in seen:
            seen.add(d["id"])
            unique.append(d)

    return {"answer": final_answer, "docs": unique[:5], "steps": steps}


# ════════════════════════════════════════════════════════════════════════════════
# 4 · HYBRID RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/hybrid")
def hybrid_rag(q: Q):
    if not DOCS:
        return no_docs_response()
    steps = []

    vr = vsearch(q.query, k=5)
    steps.append({"step": "1. Dense Retrieval",  "detail": "Semantic vector search → top 5 by cosine similarity"})

    br = bsearch(q.query, k=5)
    steps.append({"step": "2. Sparse Retrieval", "detail": "BM25 keyword match → top 5 by TF-IDF weight"})

    fused = rrf([vr, br])[:3]
    steps.append({"step": "3. RRF Fusion", "detail": "1/(rank+60) — prevents any single list from dominating"})

    v_ids = {r["id"] for r in vr}
    b_ids = {r["id"] for r in br}
    for d in fused:
        d["sources"] = (["vector"] if d["id"] in v_ids else []) + (["bm25"] if d["id"] in b_ids else [])

    ans = llm(ctx_prompt(fused, q.query))
    steps.append({"step": "4. Generate", "detail": "LLM answers from hybrid-fused top-3 docs"})
    return {"answer": ans, "docs": fused, "steps": steps,
            "vector_results": vr, "bm25_results": br}


# ════════════════════════════════════════════════════════════════════════════════
# 5 · GRAPH RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/graph")
def graph_rag(q: Q):
    if not DOCS:
        return no_docs_response()
    steps = []

    seeds      = vsearch(q.query, k=2)
    seed_nodes = {f"d{r['id']}" for r in seeds}
    steps.append({"step": "1. Seed Retrieval",
                  "detail": f"Vector search finds seeds: {[f'd{r[\"id\"]}' for r in seeds]}"})

    visited  = set(seed_nodes)
    frontier = set(seed_nodes)
    for hop in range(2):
        new_frontier: set[str] = set()
        for node in frontier:
            new_frontier.update(n for n in G.neighbors(node) if n not in visited)
        visited  |= new_frontier
        frontier  = new_frontier
        steps.append({"step": f"2.{hop+1} Graph Hop {hop+1}",
                      "detail": f"BFS adds {len(new_frontier)} neighbors → {len(visited)} total nodes visited"})

    doc_ids = [int(n[1:]) for n in visited if n.startswith("d")]
    if doc_ids:
        qe   = embedder.encode([q.query], normalize_embeddings=True)
        sc   = (qe @ DOC_EMBS[doc_ids].T)[0]
        best = sorted(zip(doc_ids, sc.tolist()), key=lambda x: x[1], reverse=True)[:3]
        docs = [{"id": i, "text": DOCS[i], "score": round(s, 4)} for i, s in best]
    else:
        docs = seeds[:3]
    steps.append({"step": "3. Re-score",
                  "detail": f"Re-rank {len(doc_ids)} graph-expanded candidates by vector similarity"})

    sub   = {f"d{d['id']}" for d in docs} | seed_nodes
    edges = [[n, nb] for n in sub for nb in G.neighbors(n) if nb in visited]

    ans = llm(ctx_prompt(docs, q.query))
    steps.append({"step": "4. Generate", "detail": "Answer enriched via multi-hop graph traversal"})
    return {"answer": ans, "docs": docs, "steps": steps,
            "graph_edges": edges[:20], "visited_nodes": list(visited)}


# ════════════════════════════════════════════════════════════════════════════════
# 6 · RAG EVALUATION  (LLM-as-Judge)
# ════════════════════════════════════════════════════════════════════════════════
def _parse_score(raw: str) -> dict:
    """Extract a JSON object from an LLM response, falling back gracefully."""
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {"score": 0.0, "reasoning": "Parse error"}
    except Exception:
        return {"score": 0.0, "reasoning": "Parse error"}


@router.post("/rag/evaluate")
def rag_evaluate(req: EvalReq):
    ctx_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(req.contexts))

    faithfulness_prompt = f"""You are an expert RAG evaluator. Score the FAITHFULNESS of the answer.

FAITHFULNESS: Does the answer contain ONLY information that is present in the retrieved context?
Penalise any claim not supported by the context (hallucination).

Retrieved Context:
{ctx_block}

Answer:
{req.answer}

Return a JSON object with exactly these keys:
- score: float between 0.0 and 1.0
- reasoning: one sentence explanation
- unsupported_claims: list of any claims not found in context (empty list if none)

JSON only, no extra text."""

    relevancy_prompt = f"""You are an expert RAG evaluator. Score the ANSWER RELEVANCY.

ANSWER RELEVANCY: Does the answer directly address what the question is asking?
A high score means the answer is on-topic and complete. Penalise off-topic or incomplete answers.

Question: {req.question}
Answer: {req.answer}

Return a JSON object with exactly these keys:
- score: float between 0.0 and 1.0
- reasoning: one sentence explanation

JSON only, no extra text."""

    context_prompt = f"""You are an expert RAG evaluator. Score the CONTEXT UTILIZATION.

CONTEXT UTILIZATION: Did the retrieved context actually contain the information needed to answer the question?
A high score means the context was relevant and sufficient. A low score means the wrong chunks were retrieved.

Question: {req.question}

Retrieved Context:
{ctx_block}

Return a JSON object with exactly these keys:
- score: float between 0.0 and 1.0
- reasoning: one sentence explanation
- missing_information: what key information was absent from the context (empty string if nothing missing)

JSON only, no extra text."""

    faithfulness = _parse_score(llm(faithfulness_prompt, max_tokens=300))
    relevancy    = _parse_score(llm(relevancy_prompt,    max_tokens=200))
    context_util = _parse_score(llm(context_prompt,      max_tokens=200))

    scores = {
        "faithfulness":        round(faithfulness.get("score", 0.0), 3),
        "answer_relevancy":    round(relevancy.get("score",    0.0), 3),
        "context_utilization": round(context_util.get("score", 0.0), 3),
    }

    if req.ground_truth:
        correctness_prompt = f"""You are an expert RAG evaluator. Score the CORRECTNESS of the answer.

CORRECTNESS: Compare the generated answer to the ground truth.
Score 1.0 if fully correct, 0.5 if partially correct, 0.0 if wrong.

Question: {req.question}
Ground Truth: {req.ground_truth}
Generated Answer: {req.answer}

Return a JSON object with exactly these keys:
- score: float between 0.0 and 1.0
- reasoning: one sentence explanation

JSON only, no extra text."""
        scores["correctness"] = round(_parse_score(llm(correctness_prompt, max_tokens=200)).get("score", 0.0), 3)

    overall = round(sum(scores.values()) / len(scores), 3)

    return {
        "scores":  scores,
        "overall": overall,
        "details": {
            "faithfulness":        faithfulness,
            "answer_relevancy":    relevancy,
            "context_utilization": context_util,
        },
        "inputs": {
            "question":         req.question,
            "answer":           req.answer,
            "context_count":    len(req.contexts),
            "has_ground_truth": bool(req.ground_truth),
        },
    }
