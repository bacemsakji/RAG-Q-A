"""Step 3: expose the RAG pipeline as a FastAPI service.

Run:  uvicorn app:app --reload      then open http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from rag import RAG

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["rag"] = RAG()
    except Exception as e:  # index missing -> API still starts, /ask returns 503
        state["error"] = str(e)
    yield


app = FastAPI(title="RAG Q&A", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=config.TOP_K, ge=1, le=10)


@app.get("/health")
def health():
    return {"index_loaded": "rag" in state, "llm_provider": config.LLM_PROVIDER}


@app.post("/ask")
def ask(q: Question):
    if "rag" not in state:
        raise HTTPException(503, f"Index not loaded. Run `python ingest.py` first. ({state.get('error')})")
    try:
        return state["rag"].answer(q.question, q.k)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
