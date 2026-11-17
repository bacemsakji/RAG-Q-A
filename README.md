# RAG Q&A over PDF documents

Ask questions about your own PDFs. Answers are grounded in the retrieved passages and cite `[file, p.X]`.

**Stack:** Python · LangChain · FAISS · Sentence-Transformers (multilingual) · FastAPI · LLM (Ollama local or Claude API)

```
PDFs → split into chunks → embeddings → FAISS index
question → embed → top-k similar chunks → prompt + LLM → answer + sources
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**LLM option A – free & local (Ollama):** install from ollama.com, then `ollama pull llama3.2:3b`
**LLM option B – Claude API:** `export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...`

## Run
```bash
# 1. put PDFs in data/
python ingest.py                 # builds index/
uvicorn app:app --reload         # API on http://127.0.0.1:8000/docs
```
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Quel est le sujet principal du document ?"}'
```

## Test (no downloads, fake embeddings/LLM)
```bash
pip install reportlab && python tests/smoke_test.py
```

## Configuration
See `config.py`: chunk size/overlap, top-k, embedding model, LLM provider/model (all env vars).
