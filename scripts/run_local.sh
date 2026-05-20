#!/usr/bin/env bash
# Starts FastAPI and Streamlit in parallel.
#
# Debug each service individually:
#   API only:  uv run uvicorn api.main:app --reload --port 8000
#   UI only:   uv run streamlit run ui/app.py --server.port 8501

set -euo pipefail

cleanup() {
    echo "Shutting down..."
    kill "$API_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "Starting API on http://localhost:8000 ..."
uv run uvicorn api.main:app --reload --port 8000 &
API_PID=$!

echo "Starting Streamlit on http://localhost:8501 ..."
uv run streamlit run ui/app.py --server.port 8501
