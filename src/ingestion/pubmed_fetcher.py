"""
src/ingestion/pubmed_fetcher.py
--------------------------------
Fetches biomedical abstracts from PubMed via the NCBI Entrez API
and saves them to data/raw/pubmed_abstracts.json.

Covers longevity topics: NMN, NAD+, sleep, nutrition, biomarkers,
metabolic health, micronutrients.

Run: python src/ingestion/pubmed_fetcher.py
"""

import json
import time
import os
from pathlib import Path
from Bio import Entrez

# ── Config ──────────────────────────────────────────────────────────────────
Entrez.email = "vitaiq@hcc.edu"  # NCBI requires an email for API access

SEARCH_TERMS = [
    "NMN supplementation longevity",
    "NAD+ aging metabolism",
    "sleep quality health outcomes",
    "micronutrient deficiency prevention",
    "intermittent fasting metabolic health",
    "omega-3 fatty acids cardiovascular",
    "vitamin D supplementation immune function",
    "gut microbiome nutrition health",
    "caloric restriction longevity",
    "exercise biomarkers aging",
    "magnesium supplementation sleep",
    "astaxanthin antioxidant longevity",
]

MAX_PER_TERM = 50          # abstracts per search term
OUTPUT_PATH = Path("data/raw/pubmed_abstracts.json")


def fetch_pmids(term: str, max_results: int) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    handle = Entrez.esearch(
        db="pubmed",
        term=term,
        retmax=max_results,
        sort="relevance",
        datetype="pdat",
        mindate="2019",   # last 5 years preferred per data plan
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch full abstract records for a list of PMIDs."""
    if not pmids:
        return []

    ids = ",".join(pmids)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    results = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]

            # Extract fields
            pmid = str(medline["PMID"])
            title = str(art.get("ArticleTitle", ""))
            abstract_texts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(t) for t in abstract_texts)

            # DOI
            doi = ""
            for loc in art.get("ELocationID", []):
                if loc.attributes.get("EIdType") == "doi":
                    doi = str(loc)
                    break

            # Publication date
            pub_date = ""
            date_info = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            pub_date = f"{date_info.get('Year', '')} {date_info.get('Month', '')}".strip()

            if abstract:  # only keep records that have an abstract
                results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "pub_date": pub_date,
                    "source": "PubMed",
                })
        except Exception as e:
            print(f"  [WARN] Skipping record: {e}")
            continue

    return results


def run():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_abstracts = []
    seen_pmids = set()

    for term in SEARCH_TERMS:
        print(f"Searching: '{term}' ...")
        try:
            pmids = fetch_pmids(term, MAX_PER_TERM)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            abstracts = fetch_abstracts(new_pmids)
            all_abstracts.extend(abstracts)
            print(f"  Got {len(abstracts)} abstracts (total so far: {len(all_abstracts)})")

            time.sleep(1)  # be polite to NCBI — max 3 req/sec without API key
        except Exception as e:
            print(f"  [ERROR] Failed for term '{term}': {e}")
            time.sleep(5)  # backoff on error

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_abstracts, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_abstracts)} abstracts saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
