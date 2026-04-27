"""
src/api/app.py
---------------
Flask REST API for VitaIQ.
Endpoints:
  GET  /           → web UI (HTML)
  GET  /health     → health check
  POST /query      → RAG query
  POST /profile    → create/update user profile
  GET  /biomarkers → list reference ranges

Run: python src/api/app.py
Visit: http://localhost:5000
"""

import json
import os
import sqlite3
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Import RAG engine lazily to avoid loading models at import time
from src.retrieval.rag_engine import query as rag_query, DB_PATH

app = Flask(__name__)
CORS(app)

# ── Web UI HTML ──────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VitaIQ — Wellness & Longevity Coach</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
  }
  header {
    text-align: center;
    margin-bottom: 2rem;
  }
  header h1 { font-size: 2rem; color: #38bdf8; font-weight: 700; }
  header p { color: #94a3b8; margin-top: 0.4rem; font-size: 0.95rem; }
  .container { width: 100%; max-width: 780px; }
  .card {
    background: #1e2330;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }
  textarea {
    width: 100%;
    background: #0f1117;
    border: 1px solid #2d3748;
    border-radius: 8px;
    color: #e2e8f0;
    font-size: 1rem;
    padding: 0.75rem 1rem;
    resize: vertical;
    min-height: 90px;
    font-family: inherit;
  }
  textarea:focus { outline: none; border-color: #38bdf8; }
  .btn {
    background: #0ea5e9;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    margin-top: 0.75rem;
    transition: background 0.2s;
  }
  .btn:hover { background: #38bdf8; }
  .btn:disabled { background: #334155; cursor: not-allowed; }
  #answer-section { display: none; }
  .answer-text {
    line-height: 1.7;
    white-space: pre-wrap;
    font-size: 0.97rem;
  }
  .citations { margin-top: 1rem; }
  .citations h3 { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
  .citation-item {
    font-size: 0.82rem;
    color: #64748b;
    padding: 0.4rem 0;
    border-top: 1px solid #2d3748;
  }
  .citation-item a { color: #38bdf8; text-decoration: none; }
  .citation-item a:hover { text-decoration: underline; }
  .meta { display: flex; gap: 1rem; margin-top: 0.75rem; font-size: 0.8rem; color: #64748b; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 3px solid #334155; border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { color: #f87171; }
  .disclaimer { background: #1a2535; border-left: 3px solid #0ea5e9; padding: 0.6rem 1rem; border-radius: 4px; font-size: 0.82rem; color: #94a3b8; margin-top: 1rem; }
  label { font-size: 0.85rem; color: #94a3b8; display: block; margin-bottom: 0.3rem; }
  input[type=text] {
    background: #0f1117;
    border: 1px solid #2d3748;
    border-radius: 6px;
    color: #e2e8f0;
    padding: 0.45rem 0.75rem;
    font-size: 0.9rem;
    width: 200px;
  }
</style>
</head>
<body>
<header>
  <h1>🧬 VitaIQ</h1>
  <p>Personal Wellness &amp; Longevity Coach · Evidence-Based · Always Cited</p>
  <p style="font-size:0.75rem;color:#475569;margin-top:0.3rem">ITAI 2377 · Group 1 · Houston Community College</p>
</header>

<div class="container">
  <div class="card">
    <div style="margin-bottom:1rem">
      <label for="user-id">User ID (optional — for personalized context)</label>
      <input type="text" id="user-id" placeholder="e.g. user_001">
    </div>
    <label for="question">Ask a health or longevity question</label>
    <textarea id="question" placeholder="e.g. What does research say about NMN supplementation for longevity? How can I improve my sleep quality? What foods are highest in magnesium?"></textarea>
    <button class="btn" id="ask-btn" onclick="askQuestion()">Ask VitaIQ</button>
  </div>

  <div class="card" id="answer-section">
    <div id="answer-content"></div>
  </div>
</div>

<script>
async function askQuestion() {
  const question = document.getElementById('question').value.trim();
  const userId = document.getElementById('user-id').value.trim() || 'anonymous';
  if (!question) return;

  const btn = document.getElementById('ask-btn');
  const section = document.getElementById('answer-section');
  const content = document.getElementById('answer-content');

  btn.disabled = true;
  section.style.display = 'block';
  content.innerHTML = '<span class="spinner"></span> Searching research...';

  try {
    const resp = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, user_id: userId }),
    });
    const data = await resp.json();

    if (!resp.ok) {
      content.innerHTML = `<span class="error">Error: ${data.error || 'Unknown error'}</span>`;
      return;
    }

    let html = `<div class="answer-text">${escapeHtml(data.answer)}</div>`;
    html += `<div class="meta"><span>⏱ ${data.response_time_ms}ms</span><span>📄 ${data.citations.length} sources retrieved</span></div>`;

    if (data.citations.length) {
      html += '<div class="citations"><h3>Sources</h3>';
      for (const c of data.citations) {
        const doiLink = c.doi
          ? `<a href="https://doi.org/${c.doi}" target="_blank">DOI ↗</a>`
          : `<a href="https://pubmed.ncbi.nlm.nih.gov/${c.pmid}/" target="_blank">PubMed ↗</a>`;
        html += `<div class="citation-item">${c.ref} ${escapeHtml(c.title)} ${c.pub_date ? '(' + c.pub_date + ')' : ''} — ${doiLink}</div>`;
      }
      html += '</div>';
    }

    html += `<div class="disclaimer">⚠️ VitaIQ provides research-grounded information only. This is not medical advice. Always consult a licensed healthcare professional for clinical decisions.</div>`;

    content.innerHTML = html;
  } catch (err) {
    content.innerHTML = `<span class="error">Request failed: ${err.message}</span>`;
  } finally {
    btn.disabled = false;
  }
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

document.getElementById('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) askQuestion();
});
</script>
</body>
</html>"""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "VitaIQ"})


@app.route("/query", methods=["POST"])
def query_endpoint():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    user_id = data.get("user_id", "anonymous")

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        result = rag_query(question, user_id)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route("/profile", methods=["POST"])
def upsert_profile():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    if not DB_PATH.exists():
        return jsonify({"error": "Database not initialized. Run structured_cleaner.py first."}), 503

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO user_profiles (user_id, age, sex, weight_kg, height_cm, health_goals, conditions, preferences)
        VALUES (:user_id, :age, :sex, :weight_kg, :height_cm, :health_goals, :conditions, :preferences)
        ON CONFLICT(user_id) DO UPDATE SET
          age=excluded.age, sex=excluded.sex, weight_kg=excluded.weight_kg,
          height_cm=excluded.height_cm, health_goals=excluded.health_goals,
          conditions=excluded.conditions, preferences=excluded.preferences,
          updated_at=CURRENT_TIMESTAMP
    """, {
        "user_id": user_id,
        "age": data.get("age"),
        "sex": data.get("sex"),
        "weight_kg": data.get("weight_kg"),
        "height_cm": data.get("height_cm"),
        "health_goals": json.dumps(data.get("health_goals", [])),
        "conditions": json.dumps(data.get("conditions", [])),
        "preferences": json.dumps(data.get("preferences", {})),
    })
    conn.commit()
    conn.close()
    return jsonify({"status": "saved", "user_id": user_id})


@app.route("/biomarkers")
def list_biomarkers():
    if not DB_PATH.exists():
        return jsonify({"error": "Database not initialized."}), 503
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM biomarkers ORDER BY name").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\nVitaIQ running at http://localhost:{port}\n")
    app.run(debug=os.getenv("FLASK_DEBUG", "1") == "1", host="0.0.0.0", port=port)
