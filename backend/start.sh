#!/usr/bin/env bash
set -e

echo "[RENDER] PharmaRec Backend starting..."

# Make sure the pre-computed artifacts exist (fast no-op when already present).
python -m app.bootstrap || echo "[BOOTSTRAP] skipped/failed, continuing with available artifacts"

# Single worker: free-tier RAM budget is 512 MB (one process keeps us well inside it)
echo "[RENDER] Starting API server on port ${PORT:-10000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-10000}" --workers 1