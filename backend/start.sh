#!/usr/bin/env bash
set -e

echo "[RENDER] PharmaRec Backend Starting..."

# Step 1: Try to download pre-computed weights from Hugging Face Hub
echo "[RENDER] Checking for pre-computed weights on HF Hub..."
python -c "from app.utils import ensure_weights; ensure_weights()" || echo "[RENDER] Hub download skipped/failed"

# Step 2: Run preprocessing if cleaned dataset is missing (fallback)
if [ ! -f "data/cleaned_dataset.csv" ]; then
  echo "[RENDER] Cleaned dataset not found. Running preprocessing..."
  python -m app.preprocessing || echo "[RENDER] Preprocessing skipped/failed, continuing..."
else
  echo "[RENDER] Cleaned dataset found, skipping preprocessing"
fi

# Step 3: Start the API server
echo "[RENDER] Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-10000}"
