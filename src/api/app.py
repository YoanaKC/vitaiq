"""
src/api/app.py
---------------
Flask REST API for VitaIQ.
Endpoints:
  GET  /            → web UI (HTML)
  GET  /health      → health check (DB + FAISS + Ollama)
  GET  /methodology → describes the RAG pipeline
  POST /query       → RAG query
  GET  /profile     → fetch user profile
  POST /profile     → create/update user profile
  POST /feedback    → thumbs up/down on a query
  GET  /biomarkers  → list reference ranges

Run: PYTHONPATH=. python src/api/app.py  (or just: python src/api/app.py)
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Lazy import — model loads on first query, not at startup
DB_PATH = Path("data/processed/vitaiq.db")
INDEX_PATH = Path("data/processed/vitaiq.index")

def rag_query(question, user_id="anonymous"):
    from src.retrieval.rag_engine import query
    return query(question, user_id)


def ensure_feedback_table():
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT,
            user_id TEXT,
            rating TEXT,           -- 'up' | 'down'
            comment TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


app = Flask(__name__)
CORS(app)
ensure_feedback_table()

HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VitaIQ — Personal Wellness & Longevity</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --cream: #fbf8f3; --white: #ffffff; --ink: #1a1a2e;
    --ink-light: #4a4a6a; --ink-muted: #8b8ba8;
    --teal: #2d8f7a; --teal-deep: #1f6b5a; --teal-light: #d8efe8; --teal-mid: #a8d9cd;
    --plum: #6b4d8a; --plum-light: #ebe2f4;
    --coral: #ed7e63; --coral-light: #fde6dd;
    --gold: #e8b04a; --gold-light: #fbeed1;
    --sky: #5b8dbf; --sky-light: #e0ecf6;
    --berry: #c14a6f; --berry-light: #f9e2ea;
    --border: #ece8e0; --surface: #ffffff; --surface-2: #fbf8f3;
    --shadow-sm: 0 1px 3px rgba(26,26,46,0.06);
    --shadow-md: 0 6px 24px rgba(26,26,46,0.08);
    --shadow-lg: 0 12px 40px rgba(26,26,46,0.10);
    --shadow-glow: 0 8px 32px rgba(45,143,122,0.18);
    --radius: 20px; --radius-sm: 10px;
    --grad-hero: linear-gradient(135deg, #d8efe8 0%, #ebe2f4 50%, #fde6dd 100%);
    --grad-brand: linear-gradient(135deg, #2d8f7a 0%, #6b4d8a 100%);
    --grad-warm: linear-gradient(135deg, #ed7e63 0%, #e8b04a 100%);
    --grad-text: linear-gradient(135deg, #1f6b5a 0%, #6b4d8a 60%, #c14a6f 100%);
  }
  html[data-theme="dark"] {
    --cream: #14141f; --white: #1c1c2c; --ink: #f0f0f5;
    --ink-light: #c4c4d8; --ink-muted: #7c7c98;
    --teal-light: #1d3a35; --teal-mid: #2d5d52;
    --plum-light: #2a2138; --coral-light: #3a2820;
    --gold-light: #3a2f18; --sky-light: #1a2a3a; --berry-light: #38222b;
    --border: #2c2c40; --surface: #1c1c2c; --surface-2: #14141f;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.4);
    --shadow-md: 0 6px 24px rgba(0,0,0,0.4);
    --shadow-lg: 0 12px 40px rgba(0,0,0,0.5);
    --shadow-glow: 0 8px 32px rgba(45,143,122,0.30);
  }
  html { scroll-behavior: smooth; }
  body { font-family: 'DM Sans', sans-serif; background: var(--cream); color: var(--ink); min-height: 100vh; font-size: 15px; line-height: 1.6; overflow-x: hidden; transition: background 0.3s, color 0.3s; }
  .bg-decor { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
  .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.45; animation: float 18s ease-in-out infinite; }
  html[data-theme="dark"] .orb { opacity: 0.20; }
  .orb-1 { width: 480px; height: 480px; background: var(--teal-light); top: -120px; left: -100px; }
  .orb-2 { width: 420px; height: 420px; background: var(--plum-light); top: 200px; right: -120px; animation-delay: -6s; }
  .orb-3 { width: 380px; height: 380px; background: var(--coral-light); top: 600px; left: 30%; animation-delay: -12s; }
  @keyframes float { 0%, 100% { transform: translate(0, 0) scale(1); } 33% { transform: translate(30px, -30px) scale(1.05); } 66% { transform: translate(-20px, 20px) scale(0.95); } }
  nav, .hero, .main-container, footer { position: relative; z-index: 1; }
  nav { position: sticky; top: 0; z-index: 100; background: color-mix(in srgb, var(--cream) 85%, transparent); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border-bottom: 1px solid var(--border); padding: 0 1.5rem; display: flex; align-items: center; justify-content: space-between; height: 68px; gap: 1rem; }
  .nav-logo { display: flex; align-items: center; gap: 12px; font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 600; color: var(--ink); text-decoration: none; }
  .nav-logo-mark { width: 28px; height: 28px; border-radius: 8px; background: var(--grad-brand); display: flex; align-items: center; justify-content: center; color: white; font-size: 0.85rem; font-weight: 700; box-shadow: var(--shadow-glow); }
  .nav-right { display: flex; align-items: center; gap: 0.4rem; }
  .nav-btn { background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 7px 12px; font-size: 0.82rem; font-weight: 500; color: var(--ink-light); cursor: pointer; font-family: inherit; transition: all 0.18s; display: inline-flex; align-items: center; gap: 6px; }
  .nav-btn:hover { background: var(--surface); border-color: var(--border); color: var(--ink); }
  .nav-btn-primary { background: var(--teal-light); color: var(--teal-deep); border-color: var(--teal-mid); }
  .nav-badge { font-size: 0.7rem; font-weight: 500; color: var(--teal-deep); background: var(--teal-light); padding: 5px 12px; border-radius: 100px; letter-spacing: 0.04em; }
  @media (max-width: 720px) { .nav-btn span.label { display: none; } .nav-badge { display: none; } }
  .hero { max-width: 820px; margin: 0 auto; padding: 4.5rem 2rem 2rem; text-align: center; }
  .hero-eyebrow { display: inline-flex; align-items: center; gap: 8px; background: var(--surface); color: var(--teal-deep); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 8px 18px; border-radius: 100px; margin-bottom: 1.75rem; box-shadow: var(--shadow-sm); border: 1px solid var(--teal-light); }
  .hero-eyebrow-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 0 4px rgba(45,143,122,0.18); animation: pulse 2.4s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 4px rgba(45,143,122,0.18); } 50% { box-shadow: 0 0 0 8px rgba(45,143,122,0.06); } }
  h1 { font-family: 'Playfair Display', serif; font-size: clamp(2.2rem, 5.5vw, 3.6rem); font-weight: 500; line-height: 1.15; color: var(--ink); margin-bottom: 1.25rem; letter-spacing: -0.01em; }
  h1 em { font-style: italic; font-weight: 600; background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
  .hero-sub { color: var(--ink-light); font-size: 1.05rem; max-width: 520px; margin: 0 auto 2rem; font-weight: 400; }
  .hero-meta { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 2.5rem; font-size: 0.78rem; color: var(--ink-muted); }
  .hero-meta-item { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 100px; background: var(--surface); border: 1px solid var(--border); }
  .hero-meta-item.live { color: var(--teal-deep); border-color: var(--teal-mid); background: var(--teal-light); }
  .hero-meta-item.warn { color: #a87c1f; border-color: var(--gold); background: var(--gold-light); }
  .stats { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 3rem; max-width: 760px; margin-left: auto; margin-right: auto; }
  .stat { flex: 1; min-width: 140px; background: var(--surface); padding: 1.1rem 0.75rem; border-radius: var(--radius-sm); border: 1px solid var(--border); box-shadow: var(--shadow-sm); transition: transform 0.2s, box-shadow 0.2s; }
  .stat:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
  .stat-num { font-family: 'Playfair Display', serif; font-size: 1.85rem; font-weight: 600; line-height: 1; }
  .stat:nth-child(1) .stat-num { color: var(--teal); }
  .stat:nth-child(2) .stat-num { color: var(--plum); }
  .stat:nth-child(3) .stat-num { color: var(--coral); }
  .stat:nth-child(4) .stat-num { color: var(--gold); }
  .stat-label { font-size: 0.7rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; font-weight: 500; }
  .main-container { max-width: 820px; margin: 0 auto; padding: 0 2rem 4rem; }
  .query-card { position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 2rem; box-shadow: var(--shadow-md); margin-bottom: 1.5rem; overflow: hidden; }
  .query-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: var(--grad-brand); }
  .suggestions-label { font-size: 0.7rem; font-weight: 600; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.85rem; display: flex; align-items: center; gap: 8px; }
  .suggestions-label::before { content: ''; width: 16px; height: 1px; background: var(--ink-muted); }
  .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem; }
  .suggestion-chip { background: var(--surface-2); border: 1px solid var(--border); border-radius: 100px; padding: 7px 14px; font-size: 0.8rem; color: var(--ink-light); cursor: pointer; transition: all 0.2s ease; font-family: 'DM Sans', sans-serif; font-weight: 500; }
  .suggestion-chip:nth-child(5n+1):hover { background: var(--teal-light); border-color: var(--teal-mid); color: var(--teal-deep); transform: translateY(-1px); }
  .suggestion-chip:nth-child(5n+2):hover { background: var(--plum-light); border-color: var(--plum); color: var(--plum); transform: translateY(-1px); }
  .suggestion-chip:nth-child(5n+3):hover { background: var(--coral-light); border-color: var(--coral); color: var(--coral); transform: translateY(-1px); }
  .suggestion-chip:nth-child(5n+4):hover { background: var(--gold-light); border-color: var(--gold); color: #a87c1f; transform: translateY(-1px); }
  .suggestion-chip:nth-child(5n+5):hover { background: var(--sky-light); border-color: var(--sky); color: var(--sky); transform: translateY(-1px); }
  textarea { width: 100%; min-height: 110px; background: var(--surface-2); border: 1.5px solid var(--border); border-radius: var(--radius-sm); padding: 1rem 1.1rem; font-family: 'DM Sans', sans-serif; font-size: 0.95rem; color: var(--ink); resize: vertical; outline: none; transition: all 0.2s; line-height: 1.6; }
  textarea:focus { border-color: var(--teal); background: var(--surface); box-shadow: 0 0 0 4px rgba(45,143,122,0.10); }
  textarea::placeholder { color: var(--ink-muted); }
  .input-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 1rem; gap: 1rem; flex-wrap: wrap; }
  .input-left { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
  .user-input-wrap { display: flex; align-items: center; gap: 8px; }
  .user-input-wrap label { font-size: 0.78rem; color: var(--ink-muted); font-weight: 500; }
  input[type=text], input[type=number], select { background: var(--surface-2); border: 1.5px solid var(--border); border-radius: var(--radius-sm); padding: 7px 12px; font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: var(--ink); width: 140px; outline: none; transition: all 0.2s; }
  input[type=text]:focus, input[type=number]:focus, select:focus { border-color: var(--teal); background: var(--surface); }
  .kbd-hint { font-size: 0.72rem; color: var(--ink-muted); display: inline-flex; align-items: center; gap: 4px; }
  .kbd { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px; padding: 1px 6px; color: var(--ink-light); }
  .btn-ask { background: var(--grad-brand); color: white; border: none; border-radius: var(--radius-sm); padding: 11px 28px; font-family: 'DM Sans', sans-serif; font-size: 0.92rem; font-weight: 600; cursor: pointer; transition: all 0.25s; display: flex; align-items: center; gap: 8px; box-shadow: var(--shadow-glow); letter-spacing: 0.01em; }
  .btn-ask:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(45,143,122,0.30); }
  .btn-ask:active { transform: translateY(0); }
  .btn-ask:disabled { background: var(--ink-muted); transform: none; cursor: not-allowed; box-shadow: none; }
  .btn-ask::after { content: '→'; font-size: 1.05rem; transition: transform 0.2s; }
  .btn-ask:hover::after { transform: translateX(3px); }
  .answer-card { position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow-lg); display: none; animation: fadeUp 0.4s cubic-bezier(0.2, 0.8, 0.2, 1); }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
  .answer-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 5px; background: var(--grad-warm); }
  .answer-header { background: linear-gradient(135deg, var(--teal-light) 0%, var(--plum-light) 100%); border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent); padding: 1.1rem 1.5rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
  .answer-header-left { display: flex; align-items: center; gap: 12px; }
  .answer-icon { width: 38px; height: 38px; border-radius: 10px; background: var(--grad-brand); color: white; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; box-shadow: var(--shadow-glow); }
  .answer-title { font-family: 'Playfair Display', serif; font-size: 1.05rem; font-weight: 600; color: var(--ink); }
  .answer-meta { font-size: 0.72rem; color: var(--ink-light); margin-top: 2px; font-weight: 500; }
  .header-pills { display: flex; gap: 6px; flex-wrap: wrap; }
  .answer-meta-pill { display: inline-flex; align-items: center; gap: 5px; background: color-mix(in srgb, var(--surface) 70%, transparent); padding: 4px 10px; border-radius: 100px; font-size: 0.72rem; color: var(--teal-deep); font-weight: 600; }
  .conf-pill { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 100px; font-size: 0.72rem; font-weight: 600; }
  .conf-pill.high { background: var(--teal-light); color: var(--teal-deep); }
  .conf-pill.medium { background: var(--gold-light); color: #8a6a2a; }
  .conf-pill.low { background: var(--berry-light); color: var(--berry); }
  .conf-pill::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  .answer-body { padding: 1.75rem 1.75rem 1.25rem; }
  .answer-text { font-size: 0.97rem; line-height: 1.78; color: var(--ink); }
  .answer-text p { margin-bottom: 0.85rem; }
  .answer-text p:last-child { margin-bottom: 0; }
  .answer-text strong { color: var(--ink); font-weight: 600; }
  .answer-text ul, .answer-text ol { margin: 0.75rem 0 0.85rem 1.25rem; }
  .answer-text li { margin-bottom: 0.35rem; }
  .answer-text h2, .answer-text h3 { font-family: 'Playfair Display', serif; font-weight: 600; margin: 1rem 0 0.5rem; color: var(--ink); }
  .answer-text h2 { font-size: 1.1rem; }
  .answer-text h3 { font-size: 1rem; }
  .answer-text code { font-family: 'JetBrains Mono', monospace; font-size: 0.85em; background: var(--surface-2); padding: 1px 6px; border-radius: 4px; }
  .answer-text a { color: var(--teal); }
  /* Consensus meter */
  .consensus { margin: 0 1.75rem 0.5rem; padding: 1.25rem; background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .consensus-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.85rem; flex-wrap: wrap; gap: 0.5rem; }
  .consensus-title { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; color: var(--ink); display: flex; align-items: center; gap: 8px; }
  .consensus-n { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--ink-muted); background: var(--surface); padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border); }
  .consensus-bar { display: flex; height: 28px; border-radius: 8px; overflow: hidden; gap: 2px; background: var(--border); }
  .consensus-seg { display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.78rem; transition: filter 0.2s; min-width: 0; }
  .consensus-seg:hover { filter: brightness(1.1); }
  .consensus-seg.yes { background: var(--teal); }
  .consensus-seg.possibly { background: var(--gold); }
  .consensus-seg.mixed { background: var(--coral); }
  .consensus-seg.no { background: var(--berry); }
  .consensus-seg.na { background: var(--ink-muted); }
  .consensus-legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 0.5rem; margin-top: 0.85rem; }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.78rem; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .legend-dot.yes { background: var(--teal); }
  .legend-dot.possibly { background: var(--gold); }
  .legend-dot.mixed { background: var(--coral); }
  .legend-dot.no { background: var(--berry); }
  .legend-dot.na { background: var(--ink-muted); }
  .legend-label { color: var(--ink-light); font-weight: 500; }
  .legend-pct { color: var(--ink-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; margin-left: auto; }
  /* Action bar */
  .action-bar { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; padding: 0.75rem 1.75rem 0.5rem; flex-wrap: wrap; }
  .action-group { display: flex; gap: 0.4rem; flex-wrap: wrap; }
  .action-btn { background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; font-size: 0.8rem; color: var(--ink-light); cursor: pointer; font-family: inherit; transition: all 0.18s; display: inline-flex; align-items: center; gap: 6px; }
  .action-btn:hover { background: var(--teal-light); border-color: var(--teal-mid); color: var(--teal-deep); }
  .action-btn.active { background: var(--teal); border-color: var(--teal); color: white; }
  .action-btn.active.down { background: var(--berry); border-color: var(--berry); }
  .action-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  /* Follow-ups */
  .followups { padding: 0 1.75rem 1.25rem; }
  .followups-title { font-size: 0.7rem; font-weight: 600; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.7rem; display: flex; align-items: center; gap: 8px; }
  .followups-title::before { content: '↳'; font-size: 1rem; color: var(--plum); }
  .followups-list { display: flex; flex-direction: column; gap: 6px; }
  .followup-chip { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 9px 14px; text-align: left; font-size: 0.85rem; color: var(--ink-light); cursor: pointer; font-family: inherit; transition: all 0.18s; display: flex; align-items: center; gap: 10px; }
  .followup-chip:hover { background: var(--plum-light); border-color: var(--plum); color: var(--plum); transform: translateX(2px); }
  .followup-chip::before { content: '?'; width: 22px; height: 22px; border-radius: 50%; background: var(--plum); color: white; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700; flex-shrink: 0; }
  /* Citations */
  .citations-section { padding: 0 1.75rem 1.5rem; }
  .citations-title { font-size: 0.7rem; font-weight: 600; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.85rem; padding-top: 1.25rem; border-top: 1px dashed var(--border); display: flex; align-items: center; gap: 8px; }
  .citations-title::before { content: '📚'; font-size: 0.95rem; }
  .citation-item { display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
  .citation-item:last-child { border-bottom: none; }
  .citation-ref { background: var(--grad-brand); color: white; font-weight: 600; padding: 3px 9px; border-radius: 6px; font-size: 0.7rem; white-space: nowrap; margin-top: 1px; box-shadow: var(--shadow-sm); }
  .citation-body { flex: 1; }
  .citation-title { color: var(--ink-light); line-height: 1.5; }
  .citation-meta { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; margin-top: 4px; font-size: 0.72rem; color: var(--ink-muted); }
  .stance-tag { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 7px; border-radius: 4px; }
  .stance-tag.yes { background: var(--teal-light); color: var(--teal-deep); }
  .stance-tag.possibly { background: var(--gold-light); color: #8a6a2a; }
  .stance-tag.mixed { background: var(--coral-light); color: var(--coral); }
  .stance-tag.no { background: var(--berry-light); color: var(--berry); }
  .stance-tag.na { background: var(--surface-2); color: var(--ink-muted); }
  .sim-bar { display: inline-flex; align-items: center; gap: 4px; }
  .sim-track { width: 50px; height: 5px; background: var(--border); border-radius: 100px; overflow: hidden; }
  .sim-fill { height: 100%; background: var(--grad-brand); }
  .citation-link { color: var(--teal); text-decoration: none; font-size: 0.75rem; white-space: nowrap; font-weight: 600; padding: 3px 8px; border-radius: 6px; background: var(--teal-light); transition: all 0.2s; }
  .citation-link:hover { background: var(--teal); color: white; }
  .disclaimer { margin: 1rem 1.75rem 1.5rem; background: linear-gradient(135deg, var(--gold-light) 0%, var(--coral-light) 100%); border: 1px solid color-mix(in srgb, var(--gold) 30%, transparent); border-radius: var(--radius-sm); padding: 12px 16px; font-size: 0.78rem; color: #8a6a2a; display: flex; align-items: flex-start; gap: 10px; }
  html[data-theme="dark"] .disclaimer { color: #d6b577; }
  .disclaimer-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
  .loading-state { padding: 2.5rem; text-align: center; }
  .loading-dots { display: inline-flex; gap: 7px; align-items: center; margin-bottom: 0.75rem; }
  .loading-dots span { width: 9px; height: 9px; border-radius: 50%; animation: bounce 1.2s infinite; }
  .loading-dots span:nth-child(1) { background: var(--teal); }
  .loading-dots span:nth-child(2) { background: var(--plum); animation-delay: 0.2s; }
  .loading-dots span:nth-child(3) { background: var(--coral); animation-delay: 0.4s; }
  @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); opacity: 0.4; } 30% { transform: translateY(-8px); opacity: 1; } }
  .loading-text { font-size: 0.88rem; color: var(--ink-light); font-weight: 500; }
  .error-text { color: var(--berry); font-size: 0.9rem; padding: 1.5rem; background: var(--berry-light); border-radius: var(--radius-sm); margin: 1rem; }
  .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2.5rem; }
  @media (max-width: 600px) { .features { grid-template-columns: 1fr; } .stats { gap: 0.5rem; } .stat { min-width: 130px; } }
  .feature-card { position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem 1.25rem; transition: all 0.25s; overflow: hidden; }
  .feature-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
  .feature-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
  .feature-card:nth-child(1)::before { background: var(--teal); }
  .feature-card:nth-child(2)::before { background: var(--plum); }
  .feature-card:nth-child(3)::before { background: var(--coral); }
  .feature-icon-wrap { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 0.75rem; font-size: 1.2rem; }
  .feature-card:nth-child(1) .feature-icon-wrap { background: var(--teal-light); }
  .feature-card:nth-child(2) .feature-icon-wrap { background: var(--plum-light); }
  .feature-card:nth-child(3) .feature-icon-wrap { background: var(--coral-light); }
  .feature-title { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; color: var(--ink); margin-bottom: 0.35rem; }
  .feature-desc { font-size: 0.8rem; color: var(--ink-light); line-height: 1.55; }
  footer { text-align: center; padding: 2.5rem 2rem; font-size: 0.78rem; color: var(--ink-muted); border-top: 1px solid var(--border); margin-top: 2rem; }
  footer strong { color: var(--ink-light); font-weight: 600; }
  /* Modals */
  .modal-overlay { position: fixed; inset: 0; background: rgba(20, 20, 35, 0.55); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); z-index: 200; display: none; align-items: flex-start; justify-content: center; padding: 4rem 1rem 2rem; overflow-y: auto; animation: fadeIn 0.2s ease; }
  .modal-overlay.open { display: flex; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  .modal { background: var(--surface); border-radius: var(--radius); width: 100%; max-width: 560px; box-shadow: var(--shadow-lg); border: 1px solid var(--border); overflow: hidden; animation: fadeUp 0.3s ease; }
  .modal-head { padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
  .modal-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 600; }
  .modal-close { background: transparent; border: none; font-size: 1.4rem; line-height: 1; cursor: pointer; color: var(--ink-muted); padding: 4px 8px; border-radius: 6px; }
  .modal-close:hover { background: var(--surface-2); color: var(--ink); }
  .modal-body { padding: 1.5rem; max-height: 70vh; overflow-y: auto; }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; }
  .form-row { display: flex; flex-direction: column; gap: 4px; }
  .form-row.full { grid-column: 1 / -1; }
  .form-row label { font-size: 0.75rem; font-weight: 600; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .form-row input, .form-row select { width: 100%; }
  .form-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 1.25rem; }
  .btn-secondary { background: var(--surface-2); border: 1px solid var(--border); color: var(--ink-light); border-radius: var(--radius-sm); padding: 9px 18px; font-size: 0.85rem; font-weight: 500; cursor: pointer; font-family: inherit; }
  .btn-secondary:hover { background: var(--surface); color: var(--ink); }
  .biomarker-row { display: grid; grid-template-columns: 1.5fr 1fr 1.5fr; gap: 0.75rem; padding: 0.75rem 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; align-items: center; }
  .biomarker-row:last-child { border-bottom: none; }
  .biomarker-name { font-weight: 600; color: var(--ink); }
  .biomarker-range { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--teal-deep); background: var(--teal-light); padding: 3px 8px; border-radius: 6px; display: inline-block; }
  .biomarker-interp { font-size: 0.78rem; color: var(--ink-light); }
  .history-item { display: flex; gap: 10px; padding: 0.75rem 0; border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.15s; padding-left: 0.5rem; padding-right: 0.5rem; border-radius: 8px; }
  .history-item:hover { background: var(--surface-2); }
  .history-item:last-child { border-bottom: none; }
  .history-q { flex: 1; font-size: 0.88rem; color: var(--ink); }
  .history-time { font-size: 0.72rem; color: var(--ink-muted); white-space: nowrap; }
  .history-empty { text-align: center; padding: 2rem; color: var(--ink-muted); font-size: 0.85rem; }
  .method-block { margin-bottom: 1.25rem; }
  .method-h { font-size: 0.72rem; font-weight: 700; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
  .method-v { font-size: 0.9rem; color: var(--ink); line-height: 1.6; }
  .method-v code { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; background: var(--surface-2); padding: 1px 6px; border-radius: 4px; }
  .method-list { padding-left: 1.25rem; }
  .method-list li { font-size: 0.88rem; margin-bottom: 0.35rem; color: var(--ink-light); }
  .toast { position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%); background: var(--ink); color: var(--cream); padding: 10px 18px; border-radius: 100px; font-size: 0.85rem; box-shadow: var(--shadow-lg); z-index: 300; opacity: 0; transition: opacity 0.2s, transform 0.2s; pointer-events: none; }
  .toast.show { opacity: 1; transform: translateX(-50%) translateY(-6px); }
</style>
</head>
<body>
<div class="bg-decor">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
</div>
<nav>
  <a class="nav-logo" href="#"><span class="nav-logo-mark">V</span>VitaIQ</a>
  <div class="nav-right">
    <button class="nav-btn" onclick="openModal('history')" title="History">🕘<span class="label">History</span></button>
    <button class="nav-btn" onclick="openModal('profile')" title="Profile">👤<span class="label">Profile</span></button>
    <button class="nav-btn" onclick="openModal('biomarkers')" title="Biomarkers">🧪<span class="label">Biomarkers</span></button>
    <button class="nav-btn" onclick="openModal('about')" title="Methodology">ℹ️<span class="label">About</span></button>
    <a class="nav-btn" href="/admin" title="Admin / eval dashboard">📊<span class="label">Admin</span></a>
    <button class="nav-btn" id="theme-btn" onclick="toggleTheme()" title="Toggle theme">🌙</button>
    <span class="nav-badge">ITAI 2377 · Group 1</span>
  </div>
</nav>
<div class="hero">
  <div class="hero-eyebrow"><span class="hero-eyebrow-dot"></span>Evidence-Based · Always Cited</div>
  <h1>Your personal<br><em>longevity coach</em></h1>
  <p class="hero-sub">Ask any health or wellness question. Get answers grounded in peer-reviewed research — with citations you can verify.</p>
  <div class="hero-meta" id="hero-meta"></div>
  <div class="stats">
    <div class="stat"><div class="stat-num">520</div><div class="stat-label">PubMed Abstracts</div></div>
    <div class="stat"><div class="stat-num">1,082</div><div class="stat-label">Research Chunks</div></div>
    <div class="stat"><div class="stat-num">53</div><div class="stat-label">USDA Foods</div></div>
    <div class="stat"><div class="stat-num">100%</div><div class="stat-label">Cited Answers</div></div>
  </div>
</div>
<div class="main-container">
  <div class="query-card">
    <div class="suggestions-label">Suggested questions</div>
    <div class="suggestions">
      <span class="suggestion-chip" onclick="setQ(this)">What does research say about NMN for longevity?</span>
      <span class="suggestion-chip" onclick="setQ(this)">How does omega-3 affect cardiovascular health?</span>
      <span class="suggestion-chip" onclick="setQ(this)">What supplements improve sleep quality?</span>
      <span class="suggestion-chip" onclick="setQ(this)">How does intermittent fasting affect metabolism?</span>
      <span class="suggestion-chip" onclick="setQ(this)">What does elevated CRP indicate?</span>
    </div>
    <textarea id="question" placeholder="Ask a health or longevity question..."></textarea>
    <div class="input-footer">
      <div class="input-left">
        <div class="user-input-wrap">
          <label for="user-id">User ID</label>
          <input type="text" id="user-id" placeholder="optional">
        </div>
        <span class="kbd-hint"><span class="kbd">⌘</span><span class="kbd">↵</span> to send</span>
      </div>
      <button class="btn-ask" id="ask-btn" onclick="askQuestion()">
        <span id="btn-text">Ask VitaIQ</span>
      </button>
    </div>
  </div>
  <div class="answer-card" id="answer-section">
    <div id="answer-content"></div>
  </div>
  <div class="features">
    <div class="feature-card"><div class="feature-icon-wrap">🔬</div><div class="feature-title">PubMed-Grounded</div><div class="feature-desc">Every answer retrieves from 520 peer-reviewed biomedical abstracts via FAISS semantic search.</div></div>
    <div class="feature-card"><div class="feature-icon-wrap">📎</div><div class="feature-title">Always Cited</div><div class="feature-desc">Inline citations link directly to PubMed and DOI — every claim is traceable and verifiable.</div></div>
    <div class="feature-card"><div class="feature-icon-wrap">🧬</div><div class="feature-title">RAG Architecture</div><div class="feature-desc">Retrieval-augmented generation means answers are grounded in retrieved documents, not hallucinated.</div></div>
  </div>
</div>
<footer><strong>VitaIQ</strong> · ITAI 2377 Data Science · Group 1 · Houston Community College · Not medical advice</footer>

<!-- Modals -->
<div class="modal-overlay" id="modal-history" onclick="if(event.target===this)closeModal('history')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title">🕘 Your question history</div><button class="modal-close" onclick="closeModal('history')">×</button></div>
    <div class="modal-body" id="history-body"></div>
  </div>
</div>
<div class="modal-overlay" id="modal-profile" onclick="if(event.target===this)closeModal('profile')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title">👤 Personalize answers</div><button class="modal-close" onclick="closeModal('profile')">×</button></div>
    <div class="modal-body">
      <div class="form-grid">
        <div class="form-row full"><label>User ID</label><input type="text" id="pf-user-id" placeholder="e.g. yoanna"></div>
        <div class="form-row"><label>Age</label><input type="number" id="pf-age" placeholder="34" min="0" max="120"></div>
        <div class="form-row"><label>Sex</label><select id="pf-sex"><option value="">—</option><option>female</option><option>male</option><option>other</option></select></div>
        <div class="form-row"><label>Weight (kg)</label><input type="number" id="pf-weight" step="0.1"></div>
        <div class="form-row"><label>Height (cm)</label><input type="number" id="pf-height" step="0.1"></div>
        <div class="form-row full"><label>Health goals (comma-separated)</label><input type="text" id="pf-goals" placeholder="longevity, energy, sleep" style="width:100%"></div>
        <div class="form-row full"><label>Conditions (comma-separated)</label><input type="text" id="pf-cond" placeholder="hypertension, prediabetes" style="width:100%"></div>
      </div>
      <div class="form-actions">
        <button class="btn-secondary" onclick="loadProfile()">Load existing</button>
        <button class="btn-ask" onclick="saveProfile()"><span>Save</span></button>
      </div>
    </div>
  </div>
</div>
<div class="modal-overlay" id="modal-biomarkers" onclick="if(event.target===this)closeModal('biomarkers')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title">🧪 Biomarker reference ranges</div><button class="modal-close" onclick="closeModal('biomarkers')">×</button></div>
    <div class="modal-body" id="biomarkers-body"></div>
  </div>
</div>
<div class="modal-overlay" id="modal-about" onclick="if(event.target===this)closeModal('about')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title">ℹ️ Methodology</div><button class="modal-close" onclick="closeModal('about')">×</button></div>
    <div class="modal-body" id="about-body"></div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const STORAGE_HISTORY = 'vitaiq.history';
const STORAGE_THEME = 'vitaiq.theme';
const STORAGE_USER = 'vitaiq.user_id';
let lastQuery = null;

// ── Theme ───────────────────────────────────────────────────────────────────
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('theme-btn').textContent = t === 'dark' ? '☀️' : '🌙';
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  localStorage.setItem(STORAGE_THEME, next);
  applyTheme(next);
}
applyTheme(localStorage.getItem(STORAGE_THEME) || 'light');

// ── Toast ───────────────────────────────────────────────────────────────────
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 2200);
}

// ── Modals ──────────────────────────────────────────────────────────────────
function openModal(name) {
  document.getElementById('modal-' + name).classList.add('open');
  if (name === 'history') renderHistory();
  if (name === 'biomarkers') loadBiomarkers();
  if (name === 'about') loadMethodology();
  if (name === 'profile') {
    const uid = document.getElementById('user-id').value.trim() || localStorage.getItem(STORAGE_USER) || '';
    document.getElementById('pf-user-id').value = uid;
  }
}
function closeModal(name) { document.getElementById('modal-' + name).classList.remove('open'); }
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.querySelectorAll('.modal-overlay.open').forEach(m => m.classList.remove('open'));
});

// ── Suggestions / question helpers ──────────────────────────────────────────
function setQ(el) {
  document.getElementById('question').value = el.textContent;
  document.getElementById('question').focus();
}

// ── Markdown renderer (safe) ────────────────────────────────────────────────
function renderMarkdown(md) {
  if (!window.marked || !window.DOMPurify) return escapeHtml(md);
  marked.setOptions({ breaks: true, gfm: true });
  return DOMPurify.sanitize(marked.parse(md));
}
function escapeHtml(str) { return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Consensus meter ─────────────────────────────────────────────────────────
function renderConsensus(consensus) {
  const order = ['yes', 'possibly', 'mixed', 'no', 'na'];
  const labels = { yes: 'Yes', possibly: 'Possibly', mixed: 'Mixed', no: 'No', na: 'Not addressed' };
  const total = order.reduce((s, k) => s + (consensus[k] || 0), 0);
  if (total === 0) return '';
  const segs = order.map(k => {
    const n = consensus[k] || 0;
    if (n === 0) return '';
    const pct = (n / total) * 100;
    return `<div class="consensus-seg ${k}" style="flex:${pct}" title="${labels[k]}: ${n} of ${total}">${n}</div>`;
  }).join('');
  const legend = order.map(k => {
    const n = consensus[k] || 0;
    if (n === 0 && k === 'na') return '';
    const pct = total ? Math.round((n / total) * 100) : 0;
    return `<div class="legend-item"><span class="legend-dot ${k}"></span><span class="legend-label">${labels[k]}</span><span class="legend-pct">${pct}%</span></div>`;
  }).join('');
  return `
    <div class="consensus">
      <div class="consensus-head">
        <div class="consensus-title">📊 Consensus Meter</div>
        <span class="consensus-n">N = ${total}</span>
      </div>
      <div class="consensus-bar">${segs}</div>
      <div class="consensus-legend">${legend}</div>
    </div>`;
}

// ── Render answer ───────────────────────────────────────────────────────────
function renderAnswer(data) {
  lastQuery = data._question;
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const conf = data.confidence || 'medium';
  const confScore = data.confidence_score != null ? ' · ' + (data.confidence_score * 100).toFixed(0) + '%' : '';
  let html = `
    <div class="answer-header">
      <div class="answer-header-left">
        <div class="answer-icon">✶</div>
        <div>
          <div class="answer-title">VitaIQ Response</div>
          <div class="answer-meta">${data.citations.length} sources · ${data.response_time_ms}ms · ${now}</div>
        </div>
      </div>
      <div class="header-pills">
        <span class="conf-pill ${conf}">${conf} confidence${confScore}</span>
        <span class="answer-meta-pill">✨ Research-grounded</span>
      </div>
    </div>
    <div class="answer-body"><div class="answer-text">${renderMarkdown(data.answer)}</div></div>`;
  html += renderConsensus(data.consensus || {});
  // Action bar
  html += `
    <div class="action-bar">
      <div class="action-group">
        <button class="action-btn" onclick="copyAnswer()" title="Copy answer">📋 Copy</button>
        <button class="action-btn" onclick="shareLink()" title="Copy shareable link">🔗 Share</button>
        <button class="action-btn" onclick="regenerate()" title="Re-run query">🔄 Regenerate</button>
      </div>
      <div class="action-group">
        <button class="action-btn" id="thumb-up" onclick="rate('up')" title="Helpful">👍</button>
        <button class="action-btn" id="thumb-down" onclick="rate('down')" title="Not helpful">👎</button>
      </div>
    </div>`;
  // Follow-ups
  if (data.follow_ups && data.follow_ups.length) {
    html += '<div class="followups"><div class="followups-title">Try next</div><div class="followups-list">';
    for (const q of data.follow_ups) {
      html += `<button class="followup-chip" onclick="askFollowup(this)">${escapeHtml(q)}</button>`;
    }
    html += '</div></div>';
  }
  // Citations
  if (data.citations.length) {
    html += '<div class="citations-section"><div class="citations-title">Sources</div>';
    for (const c of data.citations) {
      const link = c.doi
        ? `<a class="citation-link" href="https://doi.org/${c.doi}" target="_blank">View ↗</a>`
        : `<a class="citation-link" href="https://pubmed.ncbi.nlm.nih.gov/${c.pmid}/" target="_blank">PubMed ↗</a>`;
      const stance = c.stance || 'na';
      const stanceLabel = { yes: 'supports', possibly: 'possibly', mixed: 'mixed', no: 'opposes', na: 'context' }[stance];
      const sim = Math.round((c.relevance_score || 0) * 100);
      html += `
        <div class="citation-item">
          <span class="citation-ref">${c.ref}</span>
          <div class="citation-body">
            <div class="citation-title">${escapeHtml(c.title)}${c.pub_date ? ` <span style="color:var(--ink-muted)">(${c.pub_date})</span>` : ''}</div>
            <div class="citation-meta">
              <span class="stance-tag ${stance}">${stanceLabel}</span>
              <span class="sim-bar"><span class="sim-track"><span class="sim-fill" style="width:${sim}%"></span></span>${sim}% match</span>
              ${c.pmid ? `<span>PMID ${c.pmid}</span>` : ''}
            </div>
          </div>
          ${link}
        </div>`;
    }
    html += '</div>';
  }
  html += `<div class="disclaimer"><span class="disclaimer-icon">⚠️</span><span>VitaIQ provides research-grounded information only. This is not medical advice. Always consult a licensed healthcare professional for clinical decisions.</span></div>`;
  document.getElementById('answer-content').innerHTML = html;
  document.getElementById('answer-content')._data = data;
}

// ── Ask question ────────────────────────────────────────────────────────────
async function askQuestion(textOverride) {
  const question = (textOverride || document.getElementById('question').value).trim();
  const userId = document.getElementById('user-id').value.trim() || localStorage.getItem(STORAGE_USER) || 'anonymous';
  if (!question) return;
  if (textOverride) document.getElementById('question').value = textOverride;
  if (userId !== 'anonymous') localStorage.setItem(STORAGE_USER, userId);
  const btn = document.getElementById('ask-btn');
  const btnText = document.getElementById('btn-text');
  const section = document.getElementById('answer-section');
  const content = document.getElementById('answer-content');
  btn.disabled = true; btnText.textContent = 'Searching...';
  section.style.display = 'block';
  content.innerHTML = '<div class="loading-state"><div class="loading-dots"><span></span><span></span><span></span></div><div class="loading-text">Searching research corpus…</div></div>';
  try {
    const resp = await fetch('/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, user_id: userId }) });
    const data = await resp.json();
    if (!resp.ok) {
      content.innerHTML = '<div class="error-text">Error: ' + (data.error || 'Unknown error') + '</div>';
      return;
    }
    data._question = question;
    saveHistory(question);
    renderAnswer(data);
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    content.innerHTML = '<div class="error-text">Request failed: ' + err.message + '</div>';
  } finally {
    btn.disabled = false; btnText.textContent = 'Ask VitaIQ';
  }
}
function askFollowup(el) { askQuestion(el.textContent); }

// ── Action handlers ─────────────────────────────────────────────────────────
function copyAnswer() {
  const data = document.getElementById('answer-content')._data;
  if (!data) return;
  navigator.clipboard.writeText(data.answer).then(() => toast('Answer copied'));
}
function shareLink() {
  const url = new URL(window.location);
  url.searchParams.set('q', lastQuery || '');
  navigator.clipboard.writeText(url.toString()).then(() => toast('Link copied'));
}
function regenerate() { if (lastQuery) askQuestion(lastQuery); }
async function rate(rating) {
  const data = document.getElementById('answer-content')._data;
  if (!data || !data.query_id) { toast('Nothing to rate yet'); return; }
  const userId = document.getElementById('user-id').value.trim() || 'anonymous';
  try {
    const resp = await fetch('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query_id: data.query_id, user_id: userId, rating }) });
    if (!resp.ok) { toast('Could not save feedback'); return; }
    document.getElementById('thumb-up').classList.toggle('active', rating === 'up');
    document.getElementById('thumb-down').classList.toggle('active', rating === 'down');
    document.getElementById('thumb-down').classList.toggle('down', rating === 'down');
    toast(rating === 'up' ? 'Thanks for the feedback!' : 'Noted — thanks');
  } catch (err) { toast('Network error'); }
}

// ── History ─────────────────────────────────────────────────────────────────
function saveHistory(q) {
  const list = JSON.parse(localStorage.getItem(STORAGE_HISTORY) || '[]');
  list.unshift({ q, t: Date.now() });
  localStorage.setItem(STORAGE_HISTORY, JSON.stringify(list.slice(0, 30)));
}
function renderHistory() {
  const body = document.getElementById('history-body');
  const list = JSON.parse(localStorage.getItem(STORAGE_HISTORY) || '[]');
  if (!list.length) { body.innerHTML = '<div class="history-empty">No questions yet — ask something to get started.</div>'; return; }
  let h = '';
  for (const item of list) {
    const ago = timeAgo(item.t);
    h += `<div class="history-item" onclick="closeModal('history');askQuestion(${JSON.stringify(item.q).replace(/"/g, '&quot;')})"><div class="history-q">${escapeHtml(item.q)}</div><div class="history-time">${ago}</div></div>`;
  }
  h += '<div style="text-align:center;margin-top:1rem;"><button class="btn-secondary" onclick="clearHistory()">Clear history</button></div>';
  body.innerHTML = h;
}
function clearHistory() { localStorage.removeItem(STORAGE_HISTORY); renderHistory(); }
function timeAgo(t) {
  const s = Math.floor((Date.now() - t) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
}

// ── Profile ─────────────────────────────────────────────────────────────────
async function loadProfile() {
  const uid = document.getElementById('pf-user-id').value.trim();
  if (!uid) { toast('Enter a user ID first'); return; }
  try {
    const resp = await fetch('/profile?user_id=' + encodeURIComponent(uid));
    const data = await resp.json();
    if (!data.profile) { toast('No profile found — create one'); return; }
    const p = data.profile;
    document.getElementById('pf-age').value = p.age || '';
    document.getElementById('pf-sex').value = p.sex || '';
    document.getElementById('pf-weight').value = p.weight_kg || '';
    document.getElementById('pf-height').value = p.height_cm || '';
    document.getElementById('pf-goals').value = (p.health_goals || []).join(', ');
    document.getElementById('pf-cond').value = (p.conditions || []).join(', ');
    toast('Profile loaded');
  } catch (e) { toast('Could not load'); }
}
async function saveProfile() {
  const uid = document.getElementById('pf-user-id').value.trim();
  if (!uid) { toast('User ID required'); return; }
  const payload = {
    user_id: uid,
    age: parseInt(document.getElementById('pf-age').value) || null,
    sex: document.getElementById('pf-sex').value || null,
    weight_kg: parseFloat(document.getElementById('pf-weight').value) || null,
    height_cm: parseFloat(document.getElementById('pf-height').value) || null,
    health_goals: document.getElementById('pf-goals').value.split(',').map(s => s.trim()).filter(Boolean),
    conditions: document.getElementById('pf-cond').value.split(',').map(s => s.trim()).filter(Boolean),
  };
  try {
    const resp = await fetch('/profile', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (resp.ok) {
      document.getElementById('user-id').value = uid;
      localStorage.setItem(STORAGE_USER, uid);
      toast('Profile saved');
      closeModal('profile');
    } else { toast('Save failed'); }
  } catch (e) { toast('Network error'); }
}

// ── Biomarkers ──────────────────────────────────────────────────────────────
async function loadBiomarkers() {
  const body = document.getElementById('biomarkers-body');
  body.innerHTML = '<div class="history-empty">Loading…</div>';
  try {
    const resp = await fetch('/biomarkers');
    const rows = await resp.json();
    if (!Array.isArray(rows) || !rows.length) { body.innerHTML = '<div class="history-empty">No biomarkers loaded.</div>'; return; }
    let h = '';
    for (const r of rows) {
      const range = (r.ref_low != null && r.ref_high != null) ? `${r.ref_low}–${r.ref_high} ${r.unit || ''}` : (r.unit || '—');
      h += `<div class="biomarker-row"><div class="biomarker-name">${escapeHtml(r.name || '')}</div><div><span class="biomarker-range">${escapeHtml(range)}</span></div><div class="biomarker-interp">${escapeHtml(r.interpretation || '')}</div></div>`;
    }
    body.innerHTML = h;
  } catch (e) { body.innerHTML = '<div class="error-text">Could not load biomarkers.</div>'; }
}

// ── Methodology ─────────────────────────────────────────────────────────────
async function loadMethodology() {
  const body = document.getElementById('about-body');
  body.innerHTML = '<div class="history-empty">Loading…</div>';
  try {
    const resp = await fetch('/methodology');
    const m = await resp.json();
    let h = '';
    h += `<div class="method-block"><div class="method-h">Embedding model</div><div class="method-v"><code>${escapeHtml(m.embedding_model)}</code></div></div>`;
    h += `<div class="method-block"><div class="method-h">Vector store</div><div class="method-v">${escapeHtml(m.vector_store)}</div></div>`;
    h += `<div class="method-block"><div class="method-h">LLM</div><div class="method-v"><code>${escapeHtml(m.llm)}</code> · top-k = ${m.top_k}</div></div>`;
    h += `<div class="method-block"><div class="method-h">Corpus</div><div class="method-v">${m.corpus.pubmed_abstracts} PubMed abstracts · ${m.corpus.research_chunks} chunks · ${m.corpus.usda_foods} foods</div></div>`;
    h += `<div class="method-block"><div class="method-h">Pipeline</div><ol class="method-list">${(m.pipeline || []).map(s => `<li>${escapeHtml(s.replace(/^\d+\.\s*/, ''))}</li>`).join('')}</ol></div>`;
    h += `<div class="method-block"><div class="method-h">Guardrails</div><ul class="method-list">${(m.guardrails || []).map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul></div>`;
    body.innerHTML = h;
  } catch (e) { body.innerHTML = '<div class="error-text">Could not load methodology.</div>'; }
}

// ── Health badge ────────────────────────────────────────────────────────────
async function loadHealth() {
  try {
    const resp = await fetch('/health');
    const h = await resp.json();
    const meta = document.getElementById('hero-meta');
    let items = [];
    if (h.status === 'ok') items.push('<span class="hero-meta-item live">● All systems operational</span>');
    else items.push('<span class="hero-meta-item warn">● ' + h.status + '</span>');
    if (h.index_built_at) {
      const d = new Date(h.index_built_at);
      items.push('<span class="hero-meta-item">📦 Indexed ' + d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) + '</span>');
    }
    items.push('<span class="hero-meta-item">' + (h.ollama ? '🟢 Ollama' : '🔴 Ollama offline') + '</span>');
    meta.innerHTML = items.join('');
  } catch (e) { /* silent */ }
}
loadHealth();

// ── Deep link ───────────────────────────────────────────────────────────────
const params = new URLSearchParams(window.location.search);
if (params.get('q')) {
  const q = params.get('q');
  document.getElementById('question').value = q;
  setTimeout(() => askQuestion(q), 300);
}
const savedUser = localStorage.getItem(STORAGE_USER);
if (savedUser) document.getElementById('user-id').value = savedUser;

// ── Keyboard ────────────────────────────────────────────────────────────────
document.getElementById('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) askQuestion();
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
    checks = {
        "service": "VitaIQ",
        "db": DB_PATH.exists(),
        "faiss_index": INDEX_PATH.exists(),
    }
    if INDEX_PATH.exists():
        from datetime import datetime, timezone
        mtime = datetime.fromtimestamp(INDEX_PATH.stat().st_mtime, tz=timezone.utc)
        checks["index_built_at"] = mtime.isoformat()
    try:
        import requests as _r
        _r.get("http://localhost:11434/api/tags", timeout=1.5)
        checks["ollama"] = True
    except Exception:
        checks["ollama"] = False
    checks["status"] = "ok" if all([checks["db"], checks["faiss_index"]]) else "degraded"
    return jsonify(checks)


@app.route("/methodology")
def methodology():
    return jsonify({
        "embedding_model": "all-MiniLM-L6-v2 (384-dim, sentence-transformers)",
        "vector_store": "FAISS (IndexFlatIP, cosine similarity via normalized embeddings)",
        "llm": "llama3.2 via Ollama (local, no external API)",
        "top_k": 5,
        "corpus": {
            "pubmed_abstracts": 520,
            "research_chunks": 1082,
            "usda_foods": 53,
        },
        "pipeline": [
            "1. Encode query with MiniLM",
            "2. Top-5 cosine search in FAISS",
            "3. Build prompt with retrieved excerpts + user profile",
            "4. Local LLM generates Markdown answer + structured meta",
            "5. Per-source stance, confidence, follow-ups parsed from meta",
        ],
        "guardrails": [
            "No diagnosis — informational only",
            "Inline citations [N] required",
            "UI disclaimer appended to every answer",
        ],
    })


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


@app.route("/profile", methods=["GET"])
def get_profile():
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not DB_PATH.exists():
        return jsonify({"error": "Database not initialized."}), 503
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"profile": None})
    profile = dict(row)
    profile["health_goals"] = json.loads(profile.get("health_goals") or "[]")
    profile["conditions"] = json.loads(profile.get("conditions") or "[]")
    profile["preferences"] = json.loads(profile.get("preferences") or "{}")
    return jsonify({"profile": profile})


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    if not DB_PATH.exists():
        return jsonify({"error": "Database not initialized."}), 503
    data = request.get_json(force=True)
    query_id = data.get("query_id")
    rating = data.get("rating")
    if rating not in {"up", "down"}:
        return jsonify({"error": "rating must be 'up' or 'down'"}), 400
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO query_feedback (query_id, user_id, rating, comment) VALUES (?, ?, ?, ?)",
        (query_id, data.get("user_id", "anonymous"), rating, data.get("comment", "")),
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "saved"})


@app.route("/profile", methods=["POST"])
def upsert_profile():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not DB_PATH.exists():
        return jsonify({"error": "Database not initialized."}), 503
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
        "user_id": user_id, "age": data.get("age"), "sex": data.get("sex"),
        "weight_kg": data.get("weight_kg"), "height_cm": data.get("height_cm"),
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


@app.route("/admin/stats")
def admin_stats():
    if not DB_PATH.exists():
        return jsonify({"error": "Database not initialized."}), 503
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) AS n FROM query_logs").fetchone()["n"]
    avg_ms = conn.execute("SELECT AVG(response_time_ms) AS m FROM query_logs").fetchone()["m"]
    p95_ms = conn.execute("""
        SELECT response_time_ms FROM query_logs
        ORDER BY response_time_ms LIMIT 1
        OFFSET (SELECT CAST(COUNT(*) * 0.95 AS INT) FROM query_logs)
    """).fetchone()
    p95_ms = p95_ms["response_time_ms"] if p95_ms else None
    distinct_users = conn.execute("SELECT COUNT(DISTINCT user_id) AS n FROM query_logs").fetchone()["n"]

    # Citations per query (parse JSON arrays)
    citation_counts = []
    for row in conn.execute("SELECT retrieved_doc_ids FROM query_logs").fetchall():
        try:
            citation_counts.append(len(json.loads(row["retrieved_doc_ids"] or "[]")))
        except Exception:
            pass
    avg_citations = round(sum(citation_counts) / len(citation_counts), 2) if citation_counts else 0

    # Daily volume (last 30 days)
    daily = [dict(r) for r in conn.execute("""
        SELECT DATE(timestamp) AS day, COUNT(*) AS n
        FROM query_logs
        WHERE timestamp >= DATE('now', '-30 days')
        GROUP BY DATE(timestamp)
        ORDER BY day
    """).fetchall()]

    # Feedback breakdown
    fb_total = conn.execute("SELECT COUNT(*) AS n FROM query_feedback").fetchone()["n"]
    fb_up = conn.execute("SELECT COUNT(*) AS n FROM query_feedback WHERE rating='up'").fetchone()["n"]
    fb_down = conn.execute("SELECT COUNT(*) AS n FROM query_feedback WHERE rating='down'").fetchone()["n"]
    thumbs_rate = round(fb_up / fb_total, 3) if fb_total else None

    # Top questions
    top_questions = [dict(r) for r in conn.execute("""
        SELECT query_text, COUNT(*) AS n, AVG(response_time_ms) AS avg_ms
        FROM query_logs
        GROUP BY LOWER(TRIM(query_text))
        ORDER BY n DESC, query_text
        LIMIT 10
    """).fetchall()]

    # Recent queries with feedback joined
    recent = [dict(r) for r in conn.execute("""
        SELECT q.query_id, q.user_id, q.query_text, q.response_time_ms, q.timestamp,
               (SELECT rating FROM query_feedback f WHERE f.query_id = q.query_id ORDER BY feedback_id DESC LIMIT 1) AS rating
        FROM query_logs q
        ORDER BY q.timestamp DESC
        LIMIT 25
    """).fetchall()]

    # Active users
    active_users = [dict(r) for r in conn.execute("""
        SELECT user_id, COUNT(*) AS n
        FROM query_logs
        GROUP BY user_id
        ORDER BY n DESC
        LIMIT 10
    """).fetchall()]

    conn.close()
    return jsonify({
        "totals": {
            "queries": total,
            "distinct_users": distinct_users,
            "avg_response_ms": int(avg_ms) if avg_ms is not None else None,
            "p95_response_ms": p95_ms,
            "avg_citations_per_query": avg_citations,
        },
        "feedback": {
            "total": fb_total,
            "up": fb_up,
            "down": fb_down,
            "thumbs_up_rate": thumbs_rate,
        },
        "daily_volume": daily,
        "top_questions": top_questions,
        "recent": recent,
        "active_users": active_users,
    })


ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VitaIQ — Admin / Eval</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --cream: #fbf8f3; --surface: #ffffff; --ink: #1a1a2e; --ink-light: #4a4a6a; --ink-muted: #8b8ba8;
    --teal: #2d8f7a; --teal-deep: #1f6b5a; --teal-light: #d8efe8;
    --plum: #6b4d8a; --plum-light: #ebe2f4;
    --coral: #ed7e63; --coral-light: #fde6dd;
    --gold: #e8b04a; --gold-light: #fbeed1;
    --berry: #c14a6f; --berry-light: #f9e2ea;
    --border: #ece8e0;
    --shadow-sm: 0 1px 3px rgba(26,26,46,0.06);
    --shadow-md: 0 6px 24px rgba(26,26,46,0.08);
    --grad-brand: linear-gradient(135deg, #2d8f7a 0%, #6b4d8a 100%);
    --grad-text: linear-gradient(135deg, #1f6b5a 0%, #6b4d8a 60%, #c14a6f 100%);
  }
  body { font-family: 'DM Sans', sans-serif; background: var(--cream); color: var(--ink); padding: 0 0 4rem; min-height: 100vh; }
  .topbar { display: flex; align-items: center; justify-content: space-between; padding: 1rem 2rem; border-bottom: 1px solid var(--border); background: var(--surface); }
  .topbar a { color: var(--ink-light); text-decoration: none; font-size: 0.85rem; }
  .topbar a:hover { color: var(--teal); }
  .brand { font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }
  .brand-mark { width: 26px; height: 26px; border-radius: 7px; background: var(--grad-brand); color: white; display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.78rem; }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
  h1 { font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.5rem; }
  h1 em { background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; font-style: italic; }
  .subtitle { color: var(--ink-light); margin-bottom: 2rem; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem; box-shadow: var(--shadow-sm); position: relative; overflow: hidden; }
  .kpi::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
  .kpi:nth-child(5n+1)::before { background: var(--teal); }
  .kpi:nth-child(5n+2)::before { background: var(--plum); }
  .kpi:nth-child(5n+3)::before { background: var(--coral); }
  .kpi:nth-child(5n+4)::before { background: var(--gold); }
  .kpi:nth-child(5n+5)::before { background: var(--berry); }
  .kpi-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-muted); font-weight: 600; }
  .kpi-value { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 600; color: var(--ink); margin-top: 4px; line-height: 1.1; }
  .kpi-sub { font-size: 0.75rem; color: var(--ink-muted); margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
  .panels { display: grid; grid-template-columns: 1.4fr 1fr; gap: 1rem; margin-bottom: 1rem; }
  @media (max-width: 800px) { .panels { grid-template-columns: 1fr; } }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem; box-shadow: var(--shadow-sm); }
  .panel h2 { font-family: 'Playfair Display', serif; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.85rem; display: flex; align-items: center; justify-content: space-between; }
  .panel h2 small { font-size: 0.7rem; color: var(--ink-muted); font-weight: 500; font-family: 'DM Sans', sans-serif; text-transform: uppercase; letter-spacing: 0.08em; }
  .row { display: flex; align-items: center; gap: 0.75rem; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
  .row:last-child { border-bottom: none; }
  .row .q { flex: 1; color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .row .n { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--ink-muted); }
  .row .ms { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--teal-deep); background: var(--teal-light); padding: 2px 7px; border-radius: 6px; }
  .row .rating { font-size: 0.85rem; }
  .row .ts { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--ink-muted); }
  .empty { padding: 1rem; text-align: center; color: var(--ink-muted); font-size: 0.85rem; }
  /* Sparkline / bar chart */
  .spark { display: flex; gap: 3px; align-items: flex-end; height: 80px; padding: 0.5rem 0; }
  .spark-bar { flex: 1; background: var(--grad-brand); border-radius: 3px 3px 0 0; min-height: 2px; position: relative; transition: opacity 0.15s; }
  .spark-bar:hover { opacity: 0.75; }
  .spark-bar:hover::after { content: attr(data-tip); position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: var(--ink); color: var(--cream); padding: 4px 8px; border-radius: 6px; font-size: 0.72rem; white-space: nowrap; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
  .thumbs-bar { display: flex; height: 24px; border-radius: 6px; overflow: hidden; gap: 2px; background: var(--border); margin-top: 0.5rem; }
  .thumbs-up { background: var(--teal); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.78rem; min-width: 0; }
  .thumbs-down { background: var(--berry); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.78rem; min-width: 0; }
  .refresh-btn { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; font-size: 0.82rem; color: var(--ink-light); cursor: pointer; font-family: inherit; transition: all 0.15s; }
  .refresh-btn:hover { background: var(--teal-light); color: var(--teal-deep); border-color: var(--teal); }
</style>
</head>
<body>
<div class="topbar">
  <div class="brand"><span class="brand-mark">V</span>VitaIQ <span style="color:var(--ink-muted);font-size:0.85rem;font-family:'DM Sans';font-weight:500;">/ Admin</span></div>
  <div><a href="/">← Back to app</a></div>
</div>
<div class="container">
  <h1>System <em>evaluation</em></h1>
  <p class="subtitle">Live metrics from <code style="font-family:'JetBrains Mono';">query_logs</code> and <code style="font-family:'JetBrains Mono';">query_feedback</code>. <button class="refresh-btn" onclick="load()">↻ Refresh</button> <span id="refreshed" style="font-size:0.75rem;color:var(--ink-muted);margin-left:8px;"></span></p>

  <div class="kpis" id="kpis"></div>

  <div class="panel" style="margin-bottom:1rem;">
    <h2>Daily volume <small>last 30 days</small></h2>
    <div class="spark" id="spark"></div>
  </div>

  <div class="panels">
    <div class="panel">
      <h2>Recent queries <small>last 25</small></h2>
      <div id="recent"></div>
    </div>
    <div class="panel">
      <h2>Top questions <small>by frequency</small></h2>
      <div id="top"></div>
    </div>
  </div>

  <div class="panels">
    <div class="panel">
      <h2>User feedback <small>thumbs ratio</small></h2>
      <div id="feedback"></div>
    </div>
    <div class="panel">
      <h2>Active users <small>by query count</small></h2>
      <div id="users"></div>
    </div>
  </div>
</div>

<script>
function escapeHtml(s) { return String(s == null ? '' : s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fmtNum(n) { return n == null ? '—' : Number(n).toLocaleString(); }
function fmtMs(n) { return n == null ? '—' : n + ' ms'; }

async function load() {
  const resp = await fetch('/admin/stats');
  const d = await resp.json();
  if (d.error) { document.getElementById('kpis').innerHTML = '<div class="empty">' + escapeHtml(d.error) + '</div>'; return; }

  // KPIs
  const t = d.totals, fb = d.feedback;
  document.getElementById('kpis').innerHTML = `
    <div class="kpi"><div class="kpi-label">Total queries</div><div class="kpi-value">${fmtNum(t.queries)}</div><div class="kpi-sub">${fmtNum(t.distinct_users)} distinct users</div></div>
    <div class="kpi"><div class="kpi-label">Avg response</div><div class="kpi-value">${fmtMs(t.avg_response_ms)}</div><div class="kpi-sub">p95 ${fmtMs(t.p95_response_ms)}</div></div>
    <div class="kpi"><div class="kpi-label">Avg citations</div><div class="kpi-value">${t.avg_citations_per_query ?? '—'}</div><div class="kpi-sub">per query</div></div>
    <div class="kpi"><div class="kpi-label">Thumbs up rate</div><div class="kpi-value">${fb.thumbs_up_rate != null ? Math.round(fb.thumbs_up_rate * 100) + '%' : '—'}</div><div class="kpi-sub">${fmtNum(fb.total)} ratings</div></div>
    <div class="kpi"><div class="kpi-label">Total feedback</div><div class="kpi-value">${fmtNum(fb.total)}</div><div class="kpi-sub">${fb.up} up · ${fb.down} down</div></div>
  `;

  // Daily sparkline
  const spark = document.getElementById('spark');
  if (!d.daily_volume.length) { spark.innerHTML = '<div class="empty">No queries yet.</div>'; }
  else {
    const max = Math.max(...d.daily_volume.map(r => r.n));
    spark.innerHTML = d.daily_volume.map(r => {
      const h = max ? (r.n / max) * 100 : 0;
      return `<div class="spark-bar" style="height:${h}%" data-tip="${r.day}: ${r.n}"></div>`;
    }).join('');
  }

  // Recent
  const recent = document.getElementById('recent');
  if (!d.recent.length) { recent.innerHTML = '<div class="empty">No queries yet.</div>'; }
  else {
    recent.innerHTML = d.recent.map(r => {
      const rating = r.rating === 'up' ? '👍' : r.rating === 'down' ? '👎' : '';
      return `<div class="row"><div class="q" title="${escapeHtml(r.query_text)}">${escapeHtml(r.query_text)}</div><span class="rating">${rating}</span><span class="ms">${fmtMs(r.response_time_ms)}</span><span class="ts">${escapeHtml((r.timestamp || '').slice(0, 16).replace('T', ' '))}</span></div>`;
    }).join('');
  }

  // Top questions
  const top = document.getElementById('top');
  if (!d.top_questions.length) { top.innerHTML = '<div class="empty">No data.</div>'; }
  else {
    top.innerHTML = d.top_questions.map(r => `<div class="row"><div class="q">${escapeHtml(r.query_text)}</div><span class="n">×${r.n}</span><span class="ms">${r.avg_ms ? Math.round(r.avg_ms) + ' ms' : '—'}</span></div>`).join('');
  }

  // Feedback bar
  const feedback = document.getElementById('feedback');
  if (!fb.total) { feedback.innerHTML = '<div class="empty">No feedback yet — try thumbs up/down on an answer.</div>'; }
  else {
    const upPct = (fb.up / fb.total) * 100;
    const downPct = (fb.down / fb.total) * 100;
    feedback.innerHTML = `
      <div style="font-size:0.85rem;color:var(--ink-light);">${fb.up} up · ${fb.down} down · ${fmtNum(fb.total)} total</div>
      <div class="thumbs-bar">${fb.up ? `<div class="thumbs-up" style="flex:${upPct}">👍 ${fb.up}</div>` : ''}${fb.down ? `<div class="thumbs-down" style="flex:${downPct}">👎 ${fb.down}</div>` : ''}</div>
      <div style="font-size:0.78rem;color:var(--ink-muted);margin-top:0.5rem;">Thumbs-up rate: <strong>${Math.round((fb.thumbs_up_rate || 0) * 100)}%</strong></div>`;
  }

  // Active users
  const users = document.getElementById('users');
  if (!d.active_users.length) { users.innerHTML = '<div class="empty">No users yet.</div>'; }
  else {
    users.innerHTML = d.active_users.map(r => `<div class="row"><div class="q">${escapeHtml(r.user_id || '—')}</div><span class="n">×${r.n}</span></div>`).join('');
  }

  document.getElementById('refreshed').textContent = 'Updated ' + new Date().toLocaleTimeString();
}
load();
setInterval(load, 30000);
</script>
</body>
</html>"""


@app.route("/admin")
def admin():
    return render_template_string(ADMIN_HTML)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\nVitaIQ running at http://localhost:{port}\n")
    app.run(debug=os.getenv("FLASK_DEBUG", "1") == "1", host="0.0.0.0", port=port)
