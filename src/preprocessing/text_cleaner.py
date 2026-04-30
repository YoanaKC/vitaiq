"""
src/preprocessing/text_cleaner.py
"""

import json
import re
from pathlib import Path

INPUT_PATH = Path("data/raw/pubmed_abstracts.json")
OUTPUT_PATH = Path("data/processed/pubmed_clean.json")

ABBREV_MAP = {
    r"\bBMI\b": "body mass index",
    r"\bHbA1c\b": "glycated hemoglobin",
    r"\bNMN\b": "nicotinamide mononucleotide",
    r"\bNAD\+?\b": "nicotinamide adenine dinucleotide",
    r"\bRCT\b": "randomized controlled trial",
    r"\bCRP\b": "C-reactive protein",
    r"\bLDL\b": "low-density lipoprotein",
    r"\bHDL\b": "high-density lipoprotein",
    r"\bT2D\b": "type 2 diabetes",
    r"\bDHA\b": "docosahexaenoic acid",
    r"\bEPA\b": "eicosapentaenoic acid",
}

def expand_abbreviations(text):
    for pattern, expansion in ABBREV_MAP.items():
        text = re.sub(pattern, expansion, text)
    return text

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = expand_abbreviations(text)
    return text

def sentence_split(text):
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip()]

def run():
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {INPUT_PATH}")
        print("Run src/ingestion/pubmed_fetcher.py first.")
        return
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
        cleaned.append({
            "doc_id": f"pubmed_{rec['pmid']}",
            "pmid": rec["pmid"],
            "title": title,
            "abstract": abstract,
            "sentences": sentence_split(abstract),
            "doi": rec.get("doi", ""),
            "pub_date": rec.get("pub_date", ""),
            "source": "PubMed",
        })
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(cleaned)} cleaned abstracts -> {OUTPUT_PATH}")

if __name__ == "__main__":
    run()