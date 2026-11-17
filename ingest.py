"""Step 1: read PDFs → split into chunks → embed → FAISS + BM25 indices."""
import pickle
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents(data_dir: str):
    pdfs = sorted(Path(data_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in '{data_dir}/'. Add some and re-run.")
    docs = []
    for pdf in pdfs:
        pages = PyPDFLoader(str(pdf)).load()
        for p in pages:
            p.metadata["source"] = pdf.name
            p.metadata["page"]   = p.metadata.get("page", 0) + 1  # 1-indexed
        docs.extend(pages)
        print(f"  {pdf.name}: {len(pages)} pages")
    return docs


def build_index(embeddings=None):
    print(f"Loading PDFs from {config.DATA_DIR}/ ...")
    docs = load_documents(config.DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],  # smart sentence-aware split
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks. Embedding (first run downloads model) ...")

    emb = embeddings or config.get_embeddings()

    # ── FAISS semantic index ──────────────────────────────────────────────────
    store = FAISS.from_documents(chunks, emb)
    store.save_local(config.INDEX_DIR)
    print(f"FAISS index saved to {config.INDEX_DIR}/")

    # ── BM25 keyword index ────────────────────────────────────────────────────
    if config.USE_BM25:
        try:
            from rank_bm25 import BM25Okapi
            corpus    = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            tokenized = [_tokenize(text) for text in corpus]
            bm25      = BM25Okapi(tokenized)
            bm25_path = Path(config.INDEX_DIR) / "bm25.pkl"
            with open(bm25_path, "wb") as f:
                pickle.dump({"bm25": bm25, "corpus": corpus, "metadatas": metadatas}, f)
            print(f"BM25 index saved to {bm25_path}")
        except ImportError:
            print("rank-bm25 not installed — skipping BM25 index (pip install rank-bm25)")

    return store


def _tokenize(text: str) -> list[str]:
    """Simple multilingual tokenizer: lowercase + split on whitespace/punctuation."""
    import re
    return re.findall(r"\w+", text.lower())


if __name__ == "__main__":
    build_index()
