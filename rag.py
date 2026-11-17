"""
RAG pipeline — optimised for accuracy and ChatGPT-quality answers.

Improvements over v1:
  • Hybrid retrieval  : FAISS (semantic) + BM25 (keyword) fused with RRF
  • Cross-encoder     : reranks the candidate pool before sending to the LLM
  • Conversation history: follow-up questions maintain full context
  • Structured prompt : produces clear, cited, well-formatted answers
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert document assistant. Your ONLY knowledge source is the document excerpts provided in each message.

## Your rules:
1. **Language**: always reply in the same language as the user's question (French → French, English → English, Arabic → Arabic, etc.).
2. **Grounding**: base every statement strictly on the provided excerpts. Never invent facts or use outside knowledge.
3. **Citations**: cite sources inline as **[filename, p.X]** immediately after each piece of information.
4. **Structure**: use bullet points, numbered lists, or section headers when the answer has multiple parts — this makes answers easier to read.
5. **Synthesis**: if the answer spans several excerpts, combine them into a coherent, flowing response rather than listing raw quotes.
6. **Honesty**: if the information is not in the documents, say clearly: "This is not covered in the provided documents."
7. **Follow-ups**: use the conversation history to understand what "it", "they", "this" etc. refer to — answer follow-up questions naturally.
8. **Conciseness**: be thorough but not verbose. Quality over quantity."""


# ── Tokenizer (shared with ingest) ───────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ── RAG class ─────────────────────────────────────────────────────────────────
class RAG:
    def __init__(self, embeddings=None):
        self._emb = embeddings or config.get_embeddings()
        self.store = FAISS.load_local(
            config.INDEX_DIR,
            self._emb,
            allow_dangerous_deserialization=True,
        )
        self._load_bm25()
        self._reranker = None  # lazy-loaded on first use

    # ── Index loading ─────────────────────────────────────────────────────────
    def _load_bm25(self):
        bm25_path = Path(config.INDEX_DIR) / "bm25.pkl"
        if config.USE_BM25 and bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                self._bm25          = data["bm25"]
                self._bm25_corpus   = data["corpus"]
                self._bm25_meta     = data["metadatas"]
            except Exception:
                self._bm25 = None
        else:
            self._bm25 = None

    def _get_reranker(self):
        if self._reranker is None and config.USE_RERANKER:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(config.RERANKER_MODEL)
            except Exception:
                self._reranker = False   # mark as unavailable
        return self._reranker if self._reranker else None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def _faiss_retrieve(self, question: str, k: int) -> list[tuple[Document, float]]:
        """Semantic similarity search via FAISS. Score = L2 distance (lower = better)."""
        return self.store.similarity_search_with_score(question, k=k)

    def _bm25_retrieve(self, question: str, k: int) -> list[tuple[Document, float]]:
        """Keyword search via BM25. Score = BM25 relevance (higher = better)."""
        if self._bm25 is None:
            return []
        scores  = self._bm25.get_scores(_tokenize(question))
        indices = scores.argsort()[::-1][:k]
        results = []
        for idx in indices:
            if scores[idx] > 0:
                doc = Document(
                    page_content=self._bm25_corpus[idx],
                    metadata=self._bm25_meta[idx],
                )
                results.append((doc, float(scores[idx])))
        return results

    def _rrf_fuse(
        self,
        faiss_results: list,
        bm25_results: list,
        k_rrf: int = 60,
    ) -> list[tuple[Document, float]]:
        """Reciprocal Rank Fusion: combines FAISS and BM25 rankings into one list."""
        rrf_scores: dict[str, float] = {}
        docs:       dict[str, Document] = {}

        for rank, (doc, _) in enumerate(faiss_results):
            key = doc.page_content[:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)
            docs[key] = doc

        for rank, (doc, _) in enumerate(bm25_results):
            key = doc.page_content[:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)
            docs[key] = doc

        ordered = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [(docs[k], rrf_scores[k]) for k in ordered]

    def _rerank(
        self,
        question: str,
        candidates: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Cross-encoder reranking: score each (question, chunk) pair for relevance."""
        reranker = self._get_reranker()
        if reranker is None or not candidates:
            return candidates[:top_k]

        pairs  = [(question, doc.page_content) for doc, _ in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:top_k]]

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[tuple[Document, float]]:
        """
        Full retrieval pipeline:
          1. Fetch candidate pool from FAISS + BM25
          2. Fuse rankings with RRF
          3. Rerank with cross-encoder
          4. Return top-k
        """
        pool = config.TOP_K_RETRIEVE
        faiss_res = self._faiss_retrieve(question, pool)
        bm25_res  = self._bm25_retrieve(question, pool)
        fused     = self._rrf_fuse(faiss_res, bm25_res)
        return self._rerank(question, fused, k)

    # ── Context builder ───────────────────────────────────────────────────────
    @staticmethod
    def build_context(results: list[tuple[Document, float]]) -> str:
        parts = []
        for i, (doc, _) in enumerate(results, 1):
            src  = doc.metadata.get("source", "?")
            page = doc.metadata.get("page",   "?")
            parts.append(f"[Excerpt {i} — {src}, p.{page}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    # ── Answer ────────────────────────────────────────────────────────────────
    def answer(
        self,
        question: str,
        k: int = config.TOP_K,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        results  = self.retrieve(question, k)
        context  = self.build_context(results)
        user_msg = (
            f"### Document excerpts:\n\n{context}\n\n"
            f"### Question:\n{question}"
        )
        text = generate(SYSTEM_PROMPT, user_msg, history=history or [])
        sources = [
            {
                "source":  d.metadata.get("source"),
                "page":    d.metadata.get("page"),
                "score":   round(float(s), 4),
                "snippet": d.page_content[:200].replace("\n", " "),
            }
            for d, s in results
        ]
        return {"answer": text, "sources": sources}


# ── LLM generation ────────────────────────────────────────────────────────────
def generate(
    system: str,
    user: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Call the configured LLM. `history` is a list of {role, content} dicts."""
    history = (history or [])[-config.MAX_HISTORY:]   # cap to avoid context overflow

    if config.LLM_PROVIDER == "anthropic":
        return _generate_anthropic(system, user, history)
    return _generate_ollama(system, user, history)


def _build_messages(user: str, history: list[dict]) -> list[dict]:
    """Build the messages array: history turns + current user message."""
    msgs = []
    for turn in history:
        role = turn.get("role", "user")
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": turn["content"]})
    msgs.append({"role": "user", "content": user})
    return msgs


def _generate_anthropic(system: str, user: str, history: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=system,
        messages=_build_messages(user, history),
    )
    return msg.content[0].text


def _generate_ollama(system: str, user: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": system}] + _build_messages(user, history)
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model":  config.OLLAMA_MODEL,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0.3,   # lower = more factual, less hallucination
                "top_p": 0.9,
                "num_predict": config.LLM_MAX_TOKENS,
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]
