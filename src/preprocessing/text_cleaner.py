"""
src/preprocessing/text_cleaner.py
-----------------------------------
Normalizes PubMed abstract text:
- Lowercases
- Strips HTML artifacts
- Expands common medical abbreviations
- Sentence tokenizes with spaCy
- Outputs clean records to data/processed/pubmed_clean.json

Run: python src/preprocessing/text_cleaner.py
"""

import json
import re
import spacy
from pathlib import Path

INPUT_PATH = Path("data/raw/pubmed_abstracts.json")
OUTPUT_PATH = Path("data/processed/pubmed_clean.json")

# Common medical abbreviations to expand for better retrieval
ABBREV_MAP = {
    r"\bBMI\b": "body mass index",
    r"\bHbA1c\b": "glycated hemoglobin",
    r"\bNMN\b": "nicotinamide mononucleotide",
    r"\bNAD\+?\b": "nicotinamide adenine dinucleotide",
    r"\bRCT\b": "randomized controlled trial",
    r"\bCV\b": "cardiovascular",
    r"\bT2D\b": "type 2 diabetes",
    r"\bBP\b": "blood pressure",
    r"\bHDL\b": "high-density lipoprotein",
    r"\bLDL\b": "low-density lipoprotein",
    r"\bVO2\b": "maximal oxygen uptake",
    r"\bCRP\b": "C-reactive protein",
    r"\bIL-6\b": "interleukin 6",
    r"\bTNF\b": "tumor necrosis factor",
    r"\bREM\b": "rapid eye movement sleep",
    r"\bPUFA\b": "polyunsaturated fatty acids",
    r"\bDHA\b": "docosahexaenoic acid",
    r"\bEPA\b": "eicosapentaenoic acid",
    r"\bVD\b": "vitamin D",
}


def expand_abbreviations(text: str) -> str:
    for pattern, expansion in ABBREV_MAP.items():
        text = re.sub(pattern, expansion, text)
    return text


def clean_text(text: str) -> str:
    """Apply all text cleaning steps."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove special XML/HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Expand abbreviations
    text = expand_abbreviations(text)
    return text


def run():
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {INPUT_PATH}")
        print("Run src/ingestion/pubmed_fetcher.py first.")
        return

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Cleaning {len(records)} abstracts...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned = []

    for rec in records:
        abstract = clean_text(rec.get("abstract", ""))
        title = clean_text(rec.get("title", ""))

        if not abstract:
            continue

        # Sentence tokenize for chunk-level operations later
        doc = nlp(abstract)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        cleaned.append({
            "doc_id": f"pubmed_{rec['pmid']}",
            "pmid": rec["pmid"],
            "title": title,
            "abstract": abstract,
            "sentences": sentences,
            "doi": rec.get("doi", ""),
            "pub_date": rec.get("pub_date", ""),
            "source": "PubMed",
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(cleaned)} cleaned abstracts -> {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
