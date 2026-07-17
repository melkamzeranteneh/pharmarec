#!/usr/bin/env bash
set -e

echo "[RENDER] Running preprocessing (downloads dataset + builds embeddings)..."
python -m app.preprocessing || echo "[RENDER] Preprocessing skipped/failed, continuing..."

echo "[RENDER] Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-10000}"
