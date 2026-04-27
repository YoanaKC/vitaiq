"""
tests/test_pipeline.py
-----------------------
Unit tests for preprocessing and retrieval components.
Run: python -m pytest tests/ -v
"""

import json
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Text Cleaner Tests ───────────────────────────────────────────────────────

class TestTextCleaner:
    def setup_method(self):
        from src.preprocessing.text_cleaner import clean_text, expand_abbreviations, chunk_text as ct
        self.clean_text = clean_text
        self.expand_abbrev = expand_abbreviations

    def test_removes_html_tags(self):
        result = self.clean_text("<b>Hello</b> world")
        assert "<b>" not in result
        assert "Hello" in result

    def test_expands_bmi(self):
        result = self.expand_abbrev("The patient's BMI was elevated.")
        assert "body mass index" in result

    def test_expands_nmn(self):
        result = self.expand_abbrev("NMN supplementation improved outcomes.")
        assert "nicotinamide mononucleotide" in result

    def test_normalizes_whitespace(self):
        result = self.clean_text("Hello    world\n\nfoo")
        assert "  " not in result


# ── Chunking Tests ───────────────────────────────────────────────────────────

class TestChunking:
    def setup_method(self):
        from src.embeddings.build_index import chunk_text
        self.chunk_text = chunk_text

    def test_short_text_single_chunk(self):
        text = "This is a short abstract."
        chunks = self.chunk_text(text, chunk_words=50)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        text = " ".join(["word"] * 400)
        chunks = self.chunk_text(text, chunk_words=150, overlap=20)
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        text = " ".join(["word"] * 200)
        chunks = self.chunk_text(text)
        assert all(c.strip() for c in chunks)


# ── RAG Engine Tests (mock) ──────────────────────────────────────────────────

class TestPromptBuilder:
    def setup_method(self):
        from src.retrieval.rag_engine import build_prompt
        self.build_prompt = build_prompt

    def make_chunk(self, i=1):
        return {
            "pmid": f"1234{i}",
            "title": f"Test Study {i}",
            "doi": f"10.1000/test{i}",
            "pub_date": "2023",
            "chunk_text": f"This study examined longevity intervention {i}.",
            "doc_id": f"pubmed_1234{i}",
            "relevance_score": 0.9,
        }

    def test_prompt_contains_question(self):
        prompt = self.build_prompt("How does NMN work?", [self.make_chunk()], None)
        assert "How does NMN work?" in prompt

    def test_prompt_has_citations(self):
        prompt = self.build_prompt("test", [self.make_chunk(1), self.make_chunk(2)], None)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_prompt_includes_user_profile(self):
        profile = {
            "age": 35,
            "sex": "female",
            "health_goals": ["longevity", "sleep"],
            "conditions": [],
            "preferences": {},
        }
        prompt = self.build_prompt("test question", [self.make_chunk()], profile)
        assert "35" in prompt
        assert "longevity" in prompt

    def test_no_profile_uses_defaults(self):
        prompt = self.build_prompt("test", [self.make_chunk()], None)
        assert "No user profile" in prompt


# ── Flask API Tests ──────────────────────────────────────────────────────────

class TestFlaskAPI:
    def setup_method(self):
        from src.api.app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"

    def test_query_requires_question(self):
        resp = self.client.post(
            "/query",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_index_returns_html(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"VitaIQ" in resp.data
