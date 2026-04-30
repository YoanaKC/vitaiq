"""
src/retrieval/rag_engine.py
----------------------------
RAG engine using Ollama (local LLM) instead of Anthropic API.
No API key needed — runs fully locally.
"""

import json
import os
import pickle
import re
import sqlite3
import time
import uuid
import numpy as np
import faiss
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = Path("data/processed/vitaiq.index")
METADATA_PATH = Path("data/processed/chunk_metadata.pkl")
DB_PATH = Path("data/processed/vitaiq.db")

MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 5

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


def get_user_profile(user_id: str):
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


def log_query(user_id, query_text, doc_ids, response_text, response_time_ms):
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO query_logs
           (query_id, user_id, query_text, retrieved_doc_ids, response_text, response_time_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), user_id, query_text,
         json.dumps(doc_ids), response_text, response_time_ms),
    )
    conn.commit()
    conn.close()


def retrieve(query: str, top_k: int = TOP_K):
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


def build_prompt(question: str, chunks: list, user_profile) -> str:
    if user_profile:
        goals = ", ".join(user_profile.get("health_goals", [])) or "general wellness"
        conditions = ", ".join(user_profile.get("conditions", [])) or "none reported"
        age = user_profile.get("age", "unknown")
        sex = user_profile.get("sex", "unknown")
        user_ctx = (f"User profile: age {age}, sex {sex}. "
                    f"Health goals: {goals}. Conditions: {conditions}.")
    else:
        user_ctx = "No user profile provided. Provide general evidence-based guidance."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doi_str = f" DOI: {chunk['doi']}" if chunk.get("doi") else ""
        pub_str = f" ({chunk['pub_date']})" if chunk.get("pub_date") else ""
        context_parts.append(
            f"[{i}] Source: PubMed PMID {chunk['pmid']}{pub_str}{doi_str}\n"
            f"Title: {chunk['title']}\n"
            f"Excerpt: {chunk['chunk_text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""You are VitaIQ, a personal wellness and longevity AI assistant.
Your answers are always grounded in peer-reviewed biomedical research.
You do NOT diagnose medical conditions. You inform and empower.

{user_ctx}

Retrieved research excerpts:

{context}

User question: {question}

Instructions:
- Answer clearly based on the retrieved research, formatted with Markdown (use **bold**, short paragraphs, lists where appropriate).
- Cite sources inline using [1], [2], etc.
- If research is conflicting, say so honestly.
- Keep your answer under 280 words. Do NOT include a disclaimer (the UI adds one).

Answer:"""


def _classify_meta(question: str, chunks: list) -> dict:
    """
    Second LLM call: narrow, JSON-only output. Classifies each chunk's stance,
    overall confidence, and produces 3 follow-up questions.

    Uses Ollama's `format: "json"` mode for reliable structured output.
    """
    default = {
        "consensus": [{"ref": f"[{i}]", "stance": "na"} for i in range(1, len(chunks) + 1)],
        "confidence": "medium",
        "follow_ups": [],
    }
    if not chunks:
        return default

    excerpts = []
    for i, ch in enumerate(chunks, 1):
        snippet = (ch.get("chunk_text") or "")[:600]
        excerpts.append(f'[{i}] "{ch.get("title", "")}"\n{snippet}')
    excerpts_str = "\n\n".join(excerpts)

    prompt = f"""You are a research-evidence classifier. Read the user's question and each numbered source excerpt.

Question: {question}

Sources:
{excerpts_str}

Output ONLY a single JSON object with these exact keys:
  "consensus": array of {{"ref": "[N]", "stance": one of "yes" | "possibly" | "mixed" | "no" | "na"}}
                — one entry for EACH source [1]..[{len(chunks)}].
                — "yes" = the source supports answering the question affirmatively
                — "no"  = the source supports answering the question negatively
                — "possibly" = source weakly suggests yes
                — "mixed" = source presents both sides
                — "na"  = source does not address the question
  "confidence": "low" | "medium" | "high"  (your overall confidence in the answer based on these sources)
  "follow_ups": array of exactly 3 short follow-up questions (each under 90 characters)

No prose. JSON only."""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }, timeout=60)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    except Exception:
        return default

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return default
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return default

    # Normalize
    valid_stances = {"yes", "possibly", "mixed", "no", "na"}
    consensus = []
    raw_consensus = data.get("consensus") if isinstance(data.get("consensus"), list) else []
    by_ref = {}
    for item in raw_consensus:
        if isinstance(item, dict):
            ref = str(item.get("ref", "")).strip()
            stance = str(item.get("stance", "na")).lower().strip()
            if stance not in valid_stances:
                stance = "na"
            by_ref[ref] = stance
    for i in range(1, len(chunks) + 1):
        ref = f"[{i}]"
        consensus.append({"ref": ref, "stance": by_ref.get(ref, "na")})

    confidence = str(data.get("confidence", "medium")).lower().strip()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    follow_ups_raw = data.get("follow_ups") if isinstance(data.get("follow_ups"), list) else []
    follow_ups = [str(q).strip() for q in follow_ups_raw if str(q).strip()][:3]

    return {"consensus": consensus, "confidence": confidence, "follow_ups": follow_ups}


def _confidence_score(chunks: list) -> float:
    """Mean similarity of top-3 chunks, in [0,1]."""
    if not chunks:
        return 0.0
    top = sorted((c["relevance_score"] for c in chunks), reverse=True)[:3]
    return round(float(sum(top) / len(top)), 4)


def query(question: str, user_id: str = "anonymous") -> dict:
    start = time.time()

    chunks = retrieve(question)
    user_profile = get_user_profile(user_id)
    prompt = build_prompt(question, chunks, user_profile)

    # Call A: generate the Markdown answer
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=180)
    response.raise_for_status()
    answer = response.json().get("response", "").strip()

    # Call B: narrow JSON-only classification (stances + confidence + follow-ups)
    meta = _classify_meta(question, chunks)

    elapsed_ms = int((time.time() - start) * 1000)

    stance_by_ref = {item.get("ref"): item.get("stance", "na") for item in meta["consensus"]}
    citations = []
    for i, chunk in enumerate(chunks, 1):
        ref = f"[{i}]"
        citations.append({
            "ref": ref,
            "pmid": chunk["pmid"],
            "title": chunk["title"],
            "doi": chunk.get("doi", ""),
            "pub_date": chunk.get("pub_date", ""),
            "relevance_score": round(chunk["relevance_score"], 4),
            "stance": stance_by_ref.get(ref, "na"),
        })

    tallies = {"yes": 0, "possibly": 0, "mixed": 0, "no": 0, "na": 0}
    for c in citations:
        tallies[c["stance"]] = tallies.get(c["stance"], 0) + 1

    doc_ids = [c["doc_id"] for c in chunks]
    query_id = log_query_with_id(user_id, question, doc_ids, answer, elapsed_ms)

    return {
        "query_id": query_id,
        "answer": answer,
        "citations": citations,
        "response_time_ms": elapsed_ms,
        "consensus": tallies,
        "confidence": meta["confidence"],
        "confidence_score": _confidence_score(chunks),
        "follow_ups": meta["follow_ups"],
    }


def log_query_with_id(user_id, query_text, doc_ids, response_text, response_time_ms):
    """Same as log_query but returns the query_id for later feedback linkage."""
    qid = str(uuid.uuid4())
    if not DB_PATH.exists():
        return qid
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO query_logs
           (query_id, user_id, query_text, retrieved_doc_ids, response_text, response_time_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (qid, user_id, query_text,
         json.dumps(doc_ids), response_text, response_time_ms),
    )
    conn.commit()
    conn.close()
    return qid