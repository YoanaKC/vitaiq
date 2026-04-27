# VitaIQ — Personal Wellness & Longevity Coach

**ITAI 2377 · Group 1 · Domain-Specific AI Assistant — Final Implementation**

A retrieval-augmented generation (RAG) AI assistant that synthesizes biomedical research and nutrition science to deliver personalized, evidence-based wellness guidance.

---

## Team

| Member | Role |
|---|---|
| Yoana Cook | Data architecture, pipeline lead, Flask API, embeddings |
| Juphens Cherfils | Executive summary, presentation, submission coordination |
| Kaden Glover | Evaluation framework, testing strategy |
| Richard Rodriguez | Domain research, project definition |

**Instructor:** Prof. Sitaram Ayyagari — Houston Community College · ITAI 2377

---

## Project Overview

VitaIQ answers natural language health questions by:
1. Retrieving relevant biomedical abstracts from a local FAISS vector index (built from PubMed data)
2. Augmenting the query with user profile context and retrieved chunks
3. Generating a cited, evidence-grounded response via Claude API

It does **not** diagnose. It informs and empowers.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | PubMed Entrez API, USDA FoodData Central REST API |
| Preprocessing | Python, Pandas, spaCy |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (CPU) |
| Structured data | SQLite |
| Generation | Anthropic Claude API |
| API layer | Flask |
| Environment | Python 3.10, VS Code / Google Colab |

---

## Project Structure

```
vitaiq/
├── data/
│   ├── raw/               # Raw PubMed XML, USDA JSON
│   └── processed/         # Cleaned CSVs, SQLite DB
├── src/
│   ├── ingestion/         # PubMed + USDA data fetchers
│   ├── preprocessing/     # Text normalization, structured cleaning
│   ├── embeddings/        # Chunk + embed documents, build FAISS index
│   ├── retrieval/         # RAG query logic
│   └── api/               # Flask REST API + web UI
├── notebooks/             # Colab-compatible exploration notebooks
├── tests/                 # Unit + integration tests
├── docs/                  # Project plan PDF, evaluation report
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YoanaKC/vitaiq.git
cd vitaiq
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 5. Ingest data (run in order)
```bash
python src/ingestion/pubmed_fetcher.py       # Fetch PubMed abstracts
python src/ingestion/usda_fetcher.py         # Fetch USDA food data
python src/preprocessing/text_cleaner.py     # Normalize text
python src/preprocessing/structured_cleaner.py  # Clean USDA data
python src/embeddings/build_index.py         # Chunk, embed, build FAISS index
```

### 6. Run the Flask API
```bash
python src/api/app.py
# Visit http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Submit a health question, get cited answer |
| POST | `/profile` | Create or update a user profile |
| GET | `/health` | Health check |

### Example Query
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does research say about NMN supplementation for longevity?", "user_id": "user_001"}'
```

---

## Evaluation Metrics

| Metric | Target |
|---|---|
| Retrieval precision@5 | > 75% |
| Query response time | < 2 seconds |
| Citation rate | 100% of responses |
| User helpfulness rating | > 4/5 |

---

## Disclaimer

VitaIQ is an academic project. It does not provide medical diagnoses or replace professional clinical care. Always consult a licensed healthcare provider for medical decisions.
