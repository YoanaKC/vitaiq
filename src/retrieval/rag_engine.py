"""
src/retrieval/rag_engine.py
----------------------------
Core RAG logic:
1. Embeds the user query
2. Retrieves top-k chunks from FAISS
3. Builds an augmented prompt with user profile context
4. Calls Anthropic Claude API for generation
5. Returns answer + citations

Used by the Flask API.
"""

import json
import os
import pickle
import sqlite3
import time
import uuid
import numpy as np
import faiss
import anthropic
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = Path("data/processed/vitaiq.index")
METADATA_PATH = Path("data/processed/chunk_metadata.pkl")
DB_PATH = Path("data/processed/vitaiq.db")

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Module-level singletons (loaded once on first use)
_embedder = None
_index = None
_metadata = None


def _load_artifacts():
    global _embedder, _index, _metadata
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer(MODEL_NAME)
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run src/embeddings/build_index.py first."
            )
        _index = faiss.read_index(str(INDEX_PATH))
    if _metadata is None:
        with open(METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)


def get_user_profile(user_id: str) -> dict | None:
    """Fetch a user profile from SQLite."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if row:
        profile = dict(row)
        profile["health_goals"] = json.loads(profile.get("health_goals") or "[]")
        profile["conditions"] = json.loads(profile.get("conditions") or "[]")
        profile["preferences"] = json.loads(profile.get("preferences") or "{}")
        return profile
    return None


def log_query(user_id: str, query_text: str, doc_ids: list[str],
              response_text: str, response_time_ms: int):
    """Log a query to SQLite for evaluation tracking."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO query_logs
           (query_id, user_id, query_text, retrieved_doc_ids, response_text, response_time_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            user_id,
            query_text,
            json.dumps(doc_ids),
            response_text,
            response_time_ms,
        ),
    )
    conn.commit()
    conn.close()


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and retrieve top-k chunks from FAISS."""
    _load_artifacts()
    query_emb = _embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(_metadata[idx])
        chunk["relevance_score"] = float(score)
        results.append(chunk)

    return results


def build_prompt(query: str, chunks: list[dict], user_profile: dict | None) -> str:
    """Construct the augmented RAG prompt."""
    # User context block
    if user_profile:
        goals = ", ".join(user_profile.get("health_goals", [])) or "general wellness"
        conditions = ", ".join(user_profile.get("conditions", [])) or "none reported"
        age = user_profile.get("age", "unknown")
        sex = user_profile.get("sex", "unknown")
        user_ctx = (
            f"User profile: age {age}, sex {sex}. "
            f"Health goals: {goals}. "
            f"Conditions: {conditions}."
        )
    else:
        user_ctx = "No user profile provided. Provide general evidence-based guidance."

    # Retrieved context block
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        ref = f"[{i}]"
        doi_str = f" DOI: {chunk['doi']}" if chunk.get("doi") else ""
        pub_str = f" ({chunk['pub_date']})" if chunk.get("pub_date") else ""
        context_parts.append(
            f"{ref} Source: PubMed PMID {chunk['pmid']}{pub_str}{doi_str}\n"
            f"Title: {chunk['title']}\n"
            f"Excerpt: {chunk['chunk_text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are VitaIQ, a personal wellness and longevity AI assistant.
Your answers are always grounded in peer-reviewed biomedical research.
You do NOT diagnose medical conditions. You inform and empower.

{user_ctx}

The following research excerpts are retrieved from PubMed to help answer the user's question:

{context}

User question: {query}

Instructions:
- Answer clearly and directly based on the retrieved research.
- Cite sources inline using [1], [2], etc. matching the numbered excerpts above.
- If the research is conflicting or uncertain, say so honestly.
- End with a brief disclaimer that this is not medical advice.
- Keep your answer under 300 words unless the question genuinely requires more.
"""
    return prompt


def query(question: str, user_id: str = "anonymous") -> dict:
    """
    Full RAG pipeline: retrieve -> augment -> generate.
    Returns: { answer, citations, response_time_ms }
    """
    start = time.time()

    chunks = retrieve(question)
    user_profile = get_user_profile(user_id)
    prompt = build_prompt(question, chunks, user_profile)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text
    elapsed_ms = int((time.time() - start) * 1000)

    # Build citation list
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citations.append({
            "ref": f"[{i}]",
            "pmid": chunk["pmid"],
            "title": chunk["title"],
            "doi": chunk.get("doi", ""),
            "pub_date": chunk.get("pub_date", ""),
            "relevance_score": round(chunk["relevance_score"], 4),
        })

    doc_ids = [c["doc_id"] for c in chunks]
    log_query(user_id, question, doc_ids, answer, elapsed_ms)

    return {
        "answer": answer,
        "citations": citations,
        "response_time_ms": elapsed_ms,
    }
