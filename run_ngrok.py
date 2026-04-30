"""
run_ngrok.py
-------------
Starts the VitaIQ Flask app AND opens an ngrok tunnel simultaneously.
Prints the public URL to the terminal — share it for live demos.

Requirements:
  pip install pyngrok

One-time setup:
  1. Sign up free at https://ngrok.com
  2. Copy your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
  3. Run once: ngrok config add-authtoken YOUR_TOKEN
     (or set env var NGROK_AUTHTOKEN in your .env)

Then just run:
  python run_ngrok.py
"""

import os
import threading
import time
from dotenv import load_dotenv

load_dotenv()


PORT = int(os.getenv("PORT", 5000))


def start_flask():
    """Start Flask in a background thread."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.api.app import app
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


def check_ollama():
    """Quick check that Ollama is reachable — warn but don't block."""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


def main():
    # ── 1. Start Flask in background ────────────────────────────────────────
    print("Starting VitaIQ Flask server...")
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    time.sleep(2)  # give Flask a moment to bind

    if not check_ollama():
        print("[WARN] Ollama is not reachable at http://localhost:11434")
        print("       Start it with: ollama serve   (and pull the model: ollama pull llama3.2)")

    # ── 2. Open ngrok tunnel ─────────────────────────────────────────────────
    try:
        from pyngrok import ngrok, conf

        auth_token = os.getenv("NGROK_AUTHTOKEN")
        if auth_token:
            conf.get_default().auth_token = auth_token

        # Modern pyngrok API: addr first, then proto. HTTPS is the default.
        try:
            tunnel = ngrok.connect(addr=PORT, proto="http", bind_tls=True)
        except TypeError:
            # Newer pyngrok dropped bind_tls
            tunnel = ngrok.connect(addr=PORT, proto="http", schemes=["https"])
        public_url = tunnel.public_url

        print("\n" + "=" * 55)
        print("  ✅  VitaIQ is LIVE")
        print(f"  🌐  Public URL:  {public_url}")
        print(f"  🏠  Local URL:   http://localhost:{PORT}")
        print(f"  🔗  Share link:  {public_url}/?q=Your+question+here")
        print("=" * 55)
        print("\n  Share the public URL for your demo.")
        print("  Press Ctrl+C to stop.\n")

        # ── 3. Keep alive ────────────────────────────────────────────────────
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            ngrok.kill()

    except ImportError:
        print("\n[ERROR] pyngrok is not installed.")
        print("Run: pip install pyngrok")
        print("Then try again.\n")
    except Exception as e:
        print(f"\n[ERROR] ngrok failed to start: {e}")
        print("Make sure your ngrok auth token is set:")
        print("  ngrok config add-authtoken YOUR_TOKEN")
        print("or add NGROK_AUTHTOKEN=your_token to your .env file\n")


if __name__ == "__main__":
    main()
