"""Step 2: retrieve relevant chunks and ask an LLM to answer from them only."""
import os

import requests
from langchain_community.vectorstores import FAISS

import config

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond UNIQUEMENT à partir des extraits de documents fournis. "
    "Si la réponse ne s'y trouve pas, dis clairement que tu ne sais pas d'après les documents. "
    "Réponds dans la langue de la question, de façon concise, et cite les sources "
    "sous la forme [fichier, p.X]."
)


class RAG:
    def __init__(self, embeddings=None):
        self.store = FAISS.load_local(
            config.INDEX_DIR,
            embeddings or config.get_embeddings(),
            allow_dangerous_deserialization=True,  # safe: we created this index ourselves
        )

    def retrieve(self, question: str, k: int = config.TOP_K):
        return self.store.similarity_search_with_score(question, k=k)

    @staticmethod
    def build_context(results) -> str:
        parts = []
        for doc, _score in results:
            src, page = doc.metadata.get("source", "?"), doc.metadata.get("page", "?")
            parts.append(f"[{src}, p.{page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str, k: int = config.TOP_K) -> dict:
        results = self.retrieve(question, k)
        context = self.build_context(results)
        user_msg = f"Extraits :\n{context}\n\nQuestion : {question}"
        text = generate(SYSTEM_PROMPT, user_msg)
        sources = [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "score": float(s),  # L2 distance: lower = closer
                "snippet": d.page_content[:200].replace("\n", " "),
            }
            for d, s in results
        ]
        return {"answer": text, "sources": sources}


def generate(system: str, user: str) -> str:
    if config.LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        msg = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=700,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]
