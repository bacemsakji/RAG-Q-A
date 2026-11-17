"""FastAPI backend: RAG Q&A + document management dashboard."""
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from rag import RAG

state: dict = {}


def reload_index():
    try:
        state["rag"] = RAG()
        state.pop("error", None)
    except Exception as e:
        state.pop("rag", None)
        state["error"] = str(e)


def rebuild_index():
    from ingest import build_index
    build_index()
    reload_index()


@asynccontextmanager
async def lifespan(app: FastAPI):
    reload_index()
    yield


app = FastAPI(title="RAG Q&A Dashboard", lifespan=lifespan)

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "Dashboard not found.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "index_loaded": "rag" in state,
        "llm_provider": config.LLM_PROVIDER,
        "reranker":     config.USE_RERANKER,
        "bm25":         config.USE_BM25,
        "error":        state.get("error"),
    }


# ── Document management ───────────────────────────────────────────────────────
DATA_DIR = Path(config.DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)


@app.get("/documents")
def list_documents():
    return [
        {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in sorted(DATA_DIR.glob("*.pdf"))
    ]


@app.post("/documents", status_code=201)
async def upload_document(file: Annotated[UploadFile, File()]):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    dest = DATA_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        rebuild_index()
    except Exception as e:
        raise HTTPException(500, f"File saved but indexing failed: {e}")
    return {"message": f"'{file.filename}' uploaded and index rebuilt.", "name": file.filename}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    target = DATA_DIR / filename
    if not target.exists():
        raise HTTPException(404, f"'{filename}' not found.")
    target.unlink()
    try:
        if any(DATA_DIR.glob("*.pdf")):
            rebuild_index()
        else:
            state.pop("rag", None)
            state["error"] = "No documents. Upload a PDF to start."
    except Exception as e:
        raise HTTPException(500, f"File deleted but re-indexing failed: {e}")
    return {"message": f"'{filename}' deleted."}


@app.put("/documents/{filename}")
async def replace_document(filename: str, file: Annotated[UploadFile, File()]):
    target = DATA_DIR / filename
    if not target.exists():
        raise HTTPException(404, f"'{filename}' not found.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        rebuild_index()
    except Exception as e:
        raise HTTPException(500, f"File replaced but re-indexing failed: {e}")
    return {"message": f"'{filename}' replaced and index rebuilt."}


# ── Chat / Q&A ────────────────────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class Question(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=config.TOP_K, ge=1, le=10)
    history: list[HistoryMessage] = Field(default_factory=list)


@app.post("/ask")
def ask(q: Question):
    if "rag" not in state:
        raise HTTPException(503, state.get("error", "Index not loaded. Upload a PDF first."))
    try:
        history = [{"role": m.role, "content": m.content} for m in q.history]
        return state["rag"].answer(q.question, q.k, history=history)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
