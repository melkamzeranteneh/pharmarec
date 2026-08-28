"""
PharmaRec - Hugging Face Hub Download Utility

Downloads pre-computed weights from Hugging Face Hub on first startup.
Configured via the HF_REPO environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Files expected on the Hub
REQUIRED_FILES = [
    "cleaned_dataset.csv",
    "embeddings_cache.pkl",
    "collab_stats.pkl",
]

DEFAULT_REPO = "melkamzer/pharmarec-weights"


def get_repo_id() -> str:
    """Return the HF Hub repo ID from env or default."""
    return os.environ.get("HF_REPO", DEFAULT_REPO)


def all_weights_present(data_dir: Path | None = None) -> bool:
    """Check if all pre-computed weight files exist locally."""
    d = data_dir or _DATA_DIR
    return all((d / f).exists() for f in REQUIRED_FILES)


def download_weights(data_dir: Path | None = None) -> bool:
    """Download pre-computed weights from Hugging Face Hub.

    Args:
        data_dir: Directory to save files. Defaults to backend/data/.

    Returns:
        True if all files downloaded successfully, False otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[HUB] huggingface_hub not installed. Skipping weight download.")
        return False

    repo_id = get_repo_id()
    d = data_dir or _DATA_DIR
    d.mkdir(parents=True, exist_ok=True)

    print(f"[HUB] Downloading weights from {repo_id}...")

    all_ok = True
    for filename in REQUIRED_FILES:
        local_path = d / filename
        if local_path.exists():
            print(f"[HUB] {filename} already exists locally, skipping")
            continue

        try:
            print(f"[HUB] Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(d),
                repo_type="model",
            )
            print(f"[HUB] Downloaded {filename}")
        except Exception as e:
            print(f"[HUB] Failed to download {filename}: {e}")
            all_ok = False

    if all_ok:
        print("[HUB] All weights downloaded successfully")
    else:
        print("[HUB] Some weights failed to download. Will fall back to full training.")

    return all_ok


def ensure_weights(data_dir: Path | None = None) -> bool:
    """Ensure weights are available locally. Download from Hub if missing.

    Args:
        data_dir: Directory to check/save files. Defaults to backend/data/.

    Returns:
        True if weights are available (cached or downloaded).
    """
    d = data_dir or _DATA_DIR

    if all_weights_present(d):
        print("[HUB] Pre-computed weights found locally")
        return True

    return download_weights(d)
