"""Shared settings. Everything can be overridden with environment variables."""
import os

DATA_DIR  = os.getenv("DATA_DIR",  "data")
INDEX_DIR = os.getenv("INDEX_DIR", "index")

# ── Embedding model ───────────────────────────────────────────────────────────
# Multilingual: handles French, English, Arabic, etc.
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ── Chunking ──────────────────────────────────────────────────────────────────
# Smaller chunks → more precise retrieval; overlap preserves sentence boundaries.
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
# TOP_K_RETRIEVE: candidate pool before reranking (larger = better recall)
# TOP_K:          final chunks sent to the LLM after reranking
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "12"))
TOP_K          = int(os.getenv("TOP_K",           "4"))

# ── Reranker (cross-encoder) ──────────────────────────────────────────────────
# Reranks the candidate pool for maximum relevance before sending to the LLM.
USE_RERANKER    = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Hybrid BM25 ───────────────────────────────────────────────────────────────
# Keyword search complements FAISS semantic search via Reciprocal Rank Fusion.
USE_BM25 = os.getenv("USE_BM25", "true").lower() == "true"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "ollama")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen3:4b")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Conversation memory ───────────────────────────────────────────────────────
# How many previous Q&A turns to pass to the LLM for context continuity.
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
