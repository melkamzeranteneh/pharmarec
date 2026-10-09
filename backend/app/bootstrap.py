"""
PharmaRec - Runtime Bootstrap

Ensures the pre-computed artifacts in ``backend/data/`` are ready before the API
starts. Running this keeps the app self-contained and boots in seconds instead
of training models at startup.

Pip install order in deployments is tiny (pandas + numpy + sklearn only) because
none of the heavy ML frameworks are needed to *read* the artifacts: embeddings
are loaded with pickle + numpy, collaborative stats with plain pandas.

Generated artifacts:
    - cleaned_dataset.csv   from the raw Kaggle dump (already tracked in git)
    - colab_stats.pkl       collaborative per-drug stats (+ SVD RMSE/MAE if the
                            ``surprise`` package happens to be installed)
    - embeddings_cache.pkl  semantic embeddings (encoded once; drawn from a
                            Hugging Face Hub model repo if provided via HF_REPO)

The embeddings artifact is *optional*: without it, the content recommender
ranks with its tuned TF-IDF model (still good quality, zero extra dependencies).
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_RAW_DATASET = "drug_review.csv"
_CLEANED_DATASET = "cleaned_dataset.csv"
COLLAB_STATS_CACHE = _DATA_DIR / "collab_stats.pkl"
EMBEDDING_CACHE = _DATA_DIR / "embeddings_cache.pkl"

DEFAULT_HF_REPO = os.environ.get("HF_REPO", "")


# ---------------------------------------------------------------------------
# Collaborative stats
# ---------------------------------------------------------------------------

def _build_collab_stats() -> Path:
    """Rebuild the collaborative per-drug stats from the cleaned dataset."""
    import pandas as pd

    print("[BOOTSTRAP] Building collaborative stats...")
    cleaned_path = _DATA_DIR / _CLEANED_DATASET
    df = pd.read_csv(cleaned_path)

    required = ["drugName", "rating", "condition"]
    if "usefulCount" not in df.columns:
        df["usefulCount"] = 0
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    global_mean = float(df["rating"].mean())

    stats = (
        df.groupby("drugName")
        .agg(
            count=("rating", "size"),
            mean_rating=("rating", "mean"),
            useful=("usefulCount", "sum"),
            condition=("condition", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
        )
        .reset_index()
    )

    m = 10.0  # prior strength (matches CollaborativeRecommender default)
    c = global_mean
    stats["weighted_score"] = (
        (stats["count"] / (stats["count"] + m)) * stats["mean_rating"]
        + (m / (stats["count"] + m)) * c
    )

    drug_condition = dict(zip(stats["drugName"], stats["condition"]))
    drug_mean_ratings = dict(zip(stats["drugName"], stats["mean_rating"]))
    drug_weighted_scores = dict(zip(stats["drugName"], stats["weighted_score"]))
    drug_map = {drug: idx for idx, drug in enumerate(stats["drugName"])}
    drug_lookup = {drug.lower(): drug for drug in stats["drugName"]}

    condition_drugs: dict[str, list[str]] = {}
    for cond_name, grp in stats.sort_values("weighted_score", ascending=False).groupby("condition"):
        condition_drugs[cond_name] = grp["drugName"].tolist()

    high_rated = set(stats.loc[stats["mean_rating"] >= 7.0, "drugName"])

    rmse, mae = _svd_accuracy(df)

    cache = {
        "drug_condition": drug_condition,
        "drug_mean_ratings": drug_mean_ratings,
        "drug_weighted_scores": drug_weighted_scores,
        "drug_map": drug_map,
        "drug_lookup": drug_lookup,
        "condition_drugs": condition_drugs,
        "global_mean": global_mean,
        "prior_strength": m,
        "rmse": rmse,
        "mae": mae,
        "n_drugs": len(stats),
        "n_conditions": int(stats["condition"].nunique()),
        "drug_stats_df": stats.to_dict(orient="records"),
        "high_rated_drugs": list(high_rated),
    }

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(COLLAB_STATS_CACHE, "wb") as f:
        pickle.dump(cache, f)

    size_mb = COLLAB_STATS_CACHE.stat().st_size / (1024 * 1024)
    print(f"[BOOTSTRAP] Saved collab stats: {COLLAB_STATS_CACHE} ({size_mb:.1f} MB)")
    return COLLAB_STATS_CACHE


def _svd_accuracy(df):
    """Compute SVD RMSE/MAE when ``surprise`` is installed; else a free baseline.

    The baseline predicts each rating as the drug's mean rating - a simple,
    honest lower-bound that keeps the reported metrics meaningful on installs
    without the optional ``surprise`` dependency.
    """
    import pandas as pd

    if "userId" not in df.columns:
        df = df.copy()
        df["userId"] = df.groupby(["drugName", "condition", "review"]).ngroup().astype(str)

    try:
        from surprise import Dataset, Reader, SVD, accuracy
        from surprise.model_selection import train_test_split

        ratings_df = pd.DataFrame(
            {
                "user_id": df["userId"].astype(str),
                "item_id": df["drugName"].astype(str),
                "rating": df["rating"].values,
            }
        )
        reader = Reader(rating_scale=(float(df["rating"].min()), float(df["rating"].max())))
        data = Dataset.load_from_df(ratings_df, reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        model = SVD()
        model.fit(trainset)
        preds = model.test(testset)
        rmse = float(accuracy.rmse(preds, verbose=False))
        mae = float(accuracy.mae(preds, verbose=False))
        print(f"[BOOTSTRAP] SVD accuracy - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae
    except Exception as e:
        print(f"[BOOTSTRAP] surprise unavailable ({e}); using mean-rating baseline")
        actual = df["rating"].astype(float)
        predicted = df["drugName"].map(df.groupby("drugName")["rating"].mean()).astype(float)
        se = ((actual - predicted) ** 2).mean()
        ae = (actual - predicted).abs().mean()
        return float(se ** 0.5), float(ae)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _ensure_embeddings() -> None:
    """Try to obtain the embeddings cache.

    Priority: local file -> Hugging Face Hub (HF_REPO env) -> silent skip.
    Skipping is fine: the content recommender uses TF-IDF in that case.
    """
    if EMBEDDING_CACHE.exists():
        print("[BOOTSTRAP] Embeddings cache found locally")
        return

    if not DEFAULT_HF_REPO:
        print("[BOOTSTRAP] No HF_REPO set; embeddings left to local build or TF-IDF fallback")
        return

    try:
        from huggingface_hub import hf_hub_download

        print(f"[BOOTSTRAP] Downloading embeddings from HF Hub ({DEFAULT_HF_REPO})...")
        hf_hub_download(
            repo_id=DEFAULT_HF_REPO,
            filename="embeddings_cache.pkl",
            local_dir=str(_DATA_DIR),
            repo_type="model",
        )
        print("[BOOTSTRAP] Embeddings downloaded")
    except Exception as e:
        print(f"[BOOTSTRAP] Could not download embeddings ({e}); using TF-IDF fallback")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def ensure_data() -> None:
    """Ensure all runtime artifacts exist (create them if needed)."""
    print("=" * 60)
    print("  PharmaRec - Runtime Bootstrap")
    print("=" * 60)

    cleaned_path = _DATA_DIR / _CLEANED_DATASET
    if not cleaned_path.exists():
        print(f"[BOOTSTRAP] Cleaned dataset missing at {cleaned_path}")
        if (_DATA_DIR / _RAW_DATASET).exists():
            print("[BOOTSTRAP] Rebuilding cleaned dataset from raw dump...")
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from app.preprocessing import run_preprocessing

            run_preprocessing()
        else:
            print("[BOOTSTRAP] No raw dataset either; recommenders may not start.")

    if COLLAB_STATS_CACHE.exists():
        print("[BOOTSTRAP] Collab stats found locally")
    else:
        if cleaned_path.exists():
            _build_collab_stats()
        else:
            print("[BOOTSTRAP] Skipping collab stats (no cleaned dataset)")

    _ensure_embeddings()

    print("[BOOTSTRAP] Done.")


if __name__ == "__main__":
    ensure_data()