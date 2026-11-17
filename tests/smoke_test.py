"""Offline pipeline test: fake embeddings + fake LLM, no downloads needed.
Run from the project root:  python tests/smoke_test.py
"""
import os, sys, tempfile
sys.path.insert(0, os.getcwd())

from langchain_core.embeddings import DeterministicFakeEmbedding
from reportlab.pdfgen import canvas

import config, ingest, rag

tmp = tempfile.mkdtemp()
config.DATA_DIR, config.INDEX_DIR = tmp, os.path.join(tmp, "index")
c = canvas.Canvas(os.path.join(tmp, "demo.pdf"))
c.drawString(72, 750, "Terraform provisions AWS infrastructure as code.")
c.showPage(); c.drawString(72, 750, "Trivy scans container images for CVEs."); c.save()

emb = DeterministicFakeEmbedding(size=64)
ingest.build_index(emb)
rag.generate = lambda s, u: "FAKE ANSWER based on: " + u[:60]
out = rag.RAG(emb).answer("What does Trivy do?")
assert out["sources"] and out["sources"][0]["source"] == "demo.pdf", out
print("OK ->", out["answer"], "|", [(s["source"], s["page"]) for s in out["sources"]])
