#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> Starting AI Image Tracker"
echo ""

# ── Backend ──────────────────────────────────────────────────
cd "$ROOT/backend"

if [ ! -f .env ]; then
  echo "[!] No .env found. Copying .env.example → .env"
  cp .env.example .env
  echo "    → Set ANTHROPIC_API_KEY in backend/.env before running"
fi

if [ ! -d .venv ]; then
  echo "[backend] Creating virtual environment..."
  python3 -m venv .venv
fi

echo "[backend] Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

echo "[backend] Starting FastAPI on http://localhost:8000"
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# ── Frontend ─────────────────────────────────────────────────
cd "$ROOT/frontend"

if ! command -v node &>/dev/null; then
  echo "[!] Node.js not found. Please install Node.js 18+ from https://nodejs.org"
  kill $BACKEND_PID
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "[frontend] Installing npm packages..."
  npm install
fi

echo "[frontend] Starting Vite dev server on http://localhost:5173"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop both servers"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
