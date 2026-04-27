"""
src/embeddings/build_index.py
------------------------------
Chunks cleaned PubMed abstracts, generates sentence-transformer embeddings,
and builds a FAISS vector index saved to data/processed/.

Uses: sentence-transformers/all-MiniLM-L6-v2
Chunk size: 512 tokens (~400 words), 64-token overlap

Run: python src/embeddings/build_index.py
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_PATH = Path("data/processed/pubmed_clean.json")
INDEX_PATH = Path("data/processed/vitaiq.index")
METADATA_PATH = Path("data/processed/chunk_metadata.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_WORDS = 150      # approximate words per chunk (abstracts are short)
OVERLAP_WORDS = 20


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap
    return chunks


def run():
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {INPUT_PATH}")
        print("Run src/preprocessing/text_cleaner.py first.")
        return

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Chunking {len(records)} documents...")
    chunks = []       # list of text strings
    metadata = []     # parallel list of dicts with source info

    for rec in records:
        text = f"{rec['title']}. {rec['abstract']}"
        doc_chunks = chunk_text(text)
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadata.append({
                "doc_id": rec["doc_id"],
                "pmid": rec["pmid"],
                "title": rec["title"],
                "doi": rec.get("doi", ""),
                "pub_date": rec.get("pub_date", ""),
                "chunk_index": i,
                "chunk_text": chunk,
            })

    print(f"Total chunks: {len(chunks)}")
    print("Generating embeddings (this may take a few minutes)...")

    batch_size = 64
    all_embeddings = []

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Build FAISS index (Inner Product = cosine similarity when normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save artifacts
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index saved:    {INDEX_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print("Done.")


if __name__ == "__main__":
    run()
