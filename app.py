"""FastAPI backend: RAG Q&A + document management + conversation history."""
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
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
        "llm_model":    config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else config.ANTHROPIC_MODEL,
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


# ── Conversation history ──────────────────────────────────────────────────────
CONV_DIR = Path("conversations")
CONV_DIR.mkdir(exist_ok=True)


class ConvMessage(BaseModel):
    role: str
    content: str
    sources: list = Field(default_factory=list)


class ConversationSave(BaseModel):
    title: str
    messages: list[ConvMessage]


@app.get("/conversations")
def list_conversations():
    convs = []
    for f in sorted(CONV_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            convs.append({
                "id":            f.stem,
                "title":         data.get("title", "Untitled"),
                "created_at":    data.get("created_at", ""),
                "updated_at":    data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            pass
    return convs


@app.post("/conversations", status_code=201)
def create_conversation(conv: ConversationSave):
    cid  = uuid.uuid4().hex[:10]
    now  = datetime.now().isoformat(timespec="seconds")
    data = {
        "id":         cid,
        "title":      conv.title,
        "created_at": now,
        "updated_at": now,
        "messages":   [m.model_dump() for m in conv.messages],
    }
    (CONV_DIR / f"{cid}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"id": cid}


@app.get("/conversations/{cid}")
def get_conversation(cid: str):
    path = CONV_DIR / f"{cid}.json"
    if not path.exists():
        raise HTTPException(404, "Conversation not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.put("/conversations/{cid}")
def update_conversation(cid: str, conv: ConversationSave):
    path = CONV_DIR / f"{cid}.json"
    if not path.exists():
        raise HTTPException(404, "Conversation not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["title"]      = conv.title
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    data["messages"]   = [m.model_dump() for m in conv.messages]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@app.delete("/conversations/{cid}")
def delete_conversation(cid: str):
    path = CONV_DIR / f"{cid}.json"
    if not path.exists():
        raise HTTPException(404, "Conversation not found.")
    path.unlink()
    return {"ok": True}


# ── Chat / Q&A ────────────────────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    role: str
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
