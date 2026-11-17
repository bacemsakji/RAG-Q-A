"""Step 1: read PDFs in data/, split into chunks, embed, save a FAISS index."""
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
        pages = PyPDFLoader(str(pdf)).load()  # one Document per page, with metadata
        for p in pages:
            p.metadata["source"] = pdf.name
            p.metadata["page"] = p.metadata.get("page", 0) + 1  # 1-indexed for humans
        docs.extend(pages)
        print(f"  {pdf.name}: {len(pages)} pages")
    return docs


def build_index(embeddings=None):
    print(f"Loading PDFs from {config.DATA_DIR}/ ...")
    docs = load_documents(config.DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks. Embedding (first run downloads the model) ...")

    store = FAISS.from_documents(chunks, embeddings or config.get_embeddings())
    store.save_local(config.INDEX_DIR)
    print(f"Index saved to {config.INDEX_DIR}/")


if __name__ == "__main__":
    build_index()
