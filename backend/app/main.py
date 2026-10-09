"""
PharmaRec - FastAPI Application Entry Point

This module defines the FastAPI application and exposes endpoints for
drug recommendation, evaluation, and preprocessing.

Deployment notes (free tier):
    - Heavy libraries (sentence-transformers, surprise, kagglehub, nltk) are
      imported lazily so a fast ``uvicorn`` boot only needs FastAPI + pandas.
    - Pre-computed artifacts (embeddings cache, collaborative stats) are loaded
      from ``backend/data/``. If they are present, no model training, no HF
      downloads and no network access happen at startup.
    - Evaluation is computed once and memoized: ``/analysis``, ``/metrics`` and
      ``/metrics/comparison`` share a single on-demand run instead of re-running
      a 50-query benchmark on every request.

Endpoints:
    GET  /                  - Service info
    GET  /health            - Detailed health check
    GET  /drugs             - List drugs from cleaned dataset
    GET  /drugs/names       - List unique drug names
    POST /recommend         - Get recommendations (content, collaborative, or hybrid)
    GET  /metrics           - Get evaluation metrics comparison
    GET  /metrics/comparison- Get comparison table of all algorithms
    GET  /analysis          - Winner + ranking + explanation
    POST /preprocess        - Re-run cleaning pipeline and reload all recommenders
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Data models for request/response
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Request model for recommendation endpoint."""
    method: str  # "content", "collaborative", or "hybrid"
    query: str  # Drug name for content, user_id for collaborative, or either for hybrid
    previous_searches: list[str] = []  # Session history of previously searched drugs


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

# Global state for recommenders (initialized on startup)
_content_recommender = None
_collaborative_recommender = None
_hybrid_recommender = None
_cleaned_df: pd.DataFrame | None = None

# Memoized evaluation results (computed once on first request, see _evaluate_once)
_comparison_cache: dict | None = None
_eval_lock = threading.Lock()

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CLEANED_DATASET = "cleaned_dataset.csv"
_EVALUATION_ARTIFACT = _DATA_DIR / "evaluation_cache.json"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _cleaned_path() -> Path:
    return _DATA_DIR / _CLEANED_DATASET


def _load_cleaned_df() -> pd.DataFrame:
    """Load the cleaned dataset (cached in module state)."""
    global _cleaned_df
    if _cleaned_df is None:
        path = _cleaned_path()
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="Cleaned dataset not found. Run /preprocess first.",
            )
        _cleaned_df = pd.read_csv(path)
    return _cleaned_df


def load_recommenders() -> None:
    """Load and initialize all recommender systems.

    Relies on pre-computed artifacts in ``backend/data/``. If the cleaned
    dataset is missing it is rebuilt (lazy ``nltk``/``kagglehub`` imports).
    Content embeddings are read from cache when present; otherwise the
    content recommender falls back to its tuned TF-IDF model.
    """
    global _content_recommender, _collaborative_recommender, _hybrid_recommender

    from app.recommenders.collaborative import CollaborativeRecommender
    from app.recommenders.content import ContentRecommender
    from app.recommenders.hybrid import HybridRecommender

    cleaned_path = _cleaned_path()

    if not cleaned_path.exists():
        print("[STARTUP] Cleaned dataset not found. Running preprocessing...")
        try:
            from app.preprocessing import run_preprocessing

            run_preprocessing()
        except Exception as e:
            print(f"[STARTUP] Preprocessing failed ({e}); continuing with what is available")

    print("[STARTUP] Initializing ContentRecommender...")
    _content_recommender = ContentRecommender()
    _content_recommender.fit()

    print("[STARTUP] Initializing CollaborativeRecommender...")
    _collaborative_recommender = CollaborativeRecommender()
    _collaborative_recommender.train()

    print("[STARTUP] Initializing HybridRecommender...")
    _hybrid_recommender = HybridRecommender(_content_recommender, _collaborative_recommender)

    print("[STARTUP] All recommenders initialized successfully")


# ---------------------------------------------------------------------------
# Evaluation (computed once, then memoized)
# ---------------------------------------------------------------------------

def _load_evaluation_artifact() -> dict | None:
    """Load a pre-computed evaluation cache (generated at build/bootstrap time).

    The benchmark over ~50 query drugs is the most CPU-heavy task the API has,
    so a committed artifact lets every metrics/analysis request be instant and
    keeps the free-tier CPU budget nearly idle.
    """
    if not _EVALUATION_ARTIFACT.exists():
        return None
    try:
        import json

        with open(_EVALUATION_ARTIFACT, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("comparison") and data.get("analysis"):
            print(f"[INFO] Loaded pre-computed evaluation from {_EVALUATION_ARTIFACT.name}")
            return data
    except Exception as e:
        print(f"[WARNING] Could not load evaluation artifact: {e}")
    return None


def _save_evaluation_artifact(data: dict) -> None:
    """Persist a computed evaluation so future boots can skip the benchmark."""
    try:
        import json

        _EVALUATION_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        with open(_EVALUATION_ARTIFACT, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved evaluation artifact to {_EVALUATION_ARTIFACT.name}")
    except Exception as e:
        print(f"[WARNING] Could not save evaluation artifact: {e}")


def _evaluate_once() -> dict:
    """Return comparison + collab metrics + analysis, cached in memory.

    Priority: in-memory memo -> pre-computed artifact on disk -> one live
    benchmark run (which is then persisted for future boots).
    """
    global _comparison_cache
    if _comparison_cache is not None:
        return _comparison_cache

    with _eval_lock:
        if _comparison_cache is not None:
            return _comparison_cache

        if _hybrid_recommender is None:
            raise HTTPException(status_code=500, detail="Recommenders not initialized")

        artifact = _load_evaluation_artifact()
        if artifact is not None:
            _comparison_cache = artifact
            return artifact

        table = _hybrid_recommender.get_comparison_table(test_users=None, top_n=10)

        comparison = {
            "hybrid": table.loc["Hybrid"].to_dict(),
            "content_based": table.loc["Content-Based"].to_dict(),
            "collaborative": table.loc["Collaborative"].to_dict(),
        }

        collab_metrics: dict[str, float] = {"rmse": 0.0, "mae": 0.0}
        if _collaborative_recommender is not None:
            try:
                rmse_val, mae_val = _collaborative_recommender.evaluate()
                collab_metrics = {"rmse": float(rmse_val), "mae": float(mae_val)}
            except Exception:
                pass

        result = {
            "comparison": comparison,
            "collab_metrics": collab_metrics,
            "analysis": _build_analysis(comparison, collab_metrics),
        }
        _comparison_cache = result
        _save_evaluation_artifact(result)
        return result


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager."""
    if _content_recommender is None:
        load_recommenders()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="PharmaRec API",
    description="Comparative Drug Recommendation System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "PharmaRec API", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Detailed health check."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "content_recommender": "loaded" if _content_recommender is not None else "not loaded",
        "collaborative_recommender": "loaded" if _collaborative_recommender is not None else "not loaded",
        "hybrid_recommender": "loaded" if _hybrid_recommender is not None else "not loaded",
    }


@app.get("/drugs", tags=["Data"])
async def get_drugs(limit: int = 100, offset: int = 0) -> list[dict]:
    """Retrieve drugs from the cleaned dataset.

    Args:
        limit: Maximum number of records to return (default: 100).
        offset: Number of records to skip (default: 0).

    Returns:
        List of drug records.
    """
    df = _load_cleaned_df()
    result_df = df[["drugName", "condition", "review", "rating", "usefulCount"]].iloc[offset:offset + limit]
    return result_df.to_dict(orient="records")


@app.get("/drugs/names", tags=["Data"])
async def get_drug_names(limit: int = 100) -> list[str]:
    """Retrieve unique drug names.

    Args:
        limit: Maximum number of unique drug names to return (default: 100).

    Returns:
        List of unique drug names.
    """
    df = _load_cleaned_df()
    unique_drugs = df["drugName"].unique().tolist()
    return unique_drugs[:limit]


@app.post("/recommend", tags=["Recommendations"])
async def recommend(request: RecommendRequest) -> dict:
    """Get drug recommendations using the specified method.

    Request body:
    {
        "method": "content" | "collaborative" | "hybrid",
        "query": "Paracetamol",
        "previous_searches": ["DrugA", "DrugB"]
    }

    Returns:
        {
            "recommendations": [...],
            "method": "content" | "collaborative" | "hybrid",
            "query": "Paracetamol",
            "status": "success"
        }
    """
    method = request.method.lower()
    query = request.query
    previous_searches = request.previous_searches

    try:
        if method == "content":
            if _content_recommender is None:
                raise HTTPException(status_code=500, detail="Content recommender not initialized")

            # If we have previous searches, aggregate content scores across history
            if previous_searches:
                all_scores: dict[str, float] = {}
                all_conditions: dict[str, str] = {}
                for q in [query] + previous_searches:
                    try:
                        for drug, condition, score in _content_recommender.recommend(q, top_n=20):
                            if drug not in all_scores:
                                all_scores[drug] = 0.0
                                all_conditions[drug] = condition
                            all_scores[drug] += score
                    except ValueError:
                        continue
                sorted_drugs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                formatted = [
                    {"drugName": drug, "condition": all_conditions.get(drug, ""), "similarityScore": float(score)}
                    for drug, score in sorted_drugs
                ]
            else:
                formatted = [
                    {
                        "drugName": drug,
                        "condition": condition,
                        "similarityScore": float(score),
                    }
                    for drug, condition, score in _content_recommender.recommend(query, top_n=10)
                ]

        elif method == "collaborative":
            if _collaborative_recommender is None:
                raise HTTPException(status_code=500, detail="Collaborative recommender not initialized")
            formatted = [
                {"drugName": drug, "predictedRating": float(rating)}
                for drug, rating in _collaborative_recommender.recommend(query, top_n=10)
            ]

        elif method == "hybrid":
            if _hybrid_recommender is None:
                raise HTTPException(status_code=500, detail="Hybrid recommender not initialized")

            if previous_searches:
                all_hybrid_scores: dict[str, tuple[float, float, float]] = {}
                for q in [query] + previous_searches:
                    try:
                        recs = _hybrid_recommender.recommend(drug_name=q, top_n=20)
                        for drug, content, collab, hybrid in recs:
                            if drug not in all_hybrid_scores:
                                all_hybrid_scores[drug] = (0.0, 0.0, 0.0)
                            old_c, old_cl, old_h = all_hybrid_scores[drug]
                            all_hybrid_scores[drug] = (
                                old_c + content,
                                max(old_cl, collab),
                                old_h + hybrid,
                            )
                    except (ValueError, Exception):
                        continue
                sorted_drugs = sorted(all_hybrid_scores.items(), key=lambda x: x[1][2], reverse=True)[:10]
                formatted = [
                    {
                        "drugName": drug,
                        "contentScore": float(scores[0]),
                        "collabScore": float(scores[1]),
                        "hybridScore": float(scores[2]),
                    }
                    for drug, scores in sorted_drugs
                ]
            else:
                formatted = [
                    {
                        "drugName": drug,
                        "contentScore": float(content),
                        "collabScore": float(collab),
                        "hybridScore": float(hybrid),
                    }
                    for drug, content, collab, hybrid in _hybrid_recommender.recommend(drug_name=query, top_n=10)
                ]

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method: {method}. Must be 'content', 'collaborative', or 'hybrid'",
            )

        return {
            "recommendations": formatted,
            "method": method,
            "query": query,
            "status": "success",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/metrics", tags=["Evaluation"])
async def get_metrics() -> dict:
    """Get evaluation metrics for all recommendation methods.

    Returns:
        {
            "metrics": {
                "hybrid": {"precision_at_10": ..., "recall_at_10": ..., ...},
                "collaborative": {"rmse": ..., "mae": ..., ...},
                "content": {...}
            },
            "status": "ok"
        }
    """
    try:
        result = _evaluate_once()
    except HTTPException:
        raise
    except Exception as e:
        return {"metrics": {}, "status": "error", "error": str(e)}

    comparison = result["comparison"]
    collab_metrics = result["collab_metrics"]

    metrics = {
        "hybrid": {
            "precision_at_10": float(comparison["hybrid"].get("precision_at_10", 0.0)),
            "recall_at_10": float(comparison["hybrid"].get("recall_at_10", 0.0)),
            "coverage": float(comparison["hybrid"].get("coverage", 0.0)),
            "execution_time": float(comparison["hybrid"].get("execution_time", 0.0)),
        },
        "collaborative": {
            "rmse": float(collab_metrics["rmse"]),
            "mae": float(collab_metrics["mae"]),
            "precision_at_10": float(comparison["collaborative"].get("precision_at_10", 0.0)),
            "recall_at_10": float(comparison["collaborative"].get("recall_at_10", 0.0)),
        },
        "content": {
            "precision_at_10": float(comparison["content_based"].get("precision_at_10", 0.0)),
            "recall_at_10": float(comparison["content_based"].get("recall_at_10", 0.0)),
            "coverage": float(comparison["content_based"].get("coverage", 0.0)),
        },
    }
    return {"metrics": metrics, "status": "ok"}


@app.get("/metrics/comparison", tags=["Evaluation"])
async def get_comparison_table() -> dict:
    """Get comparison table of all recommendation methods.

    Returns:
        {
            "comparison_table": [
                {"metric": "precision_at_10", "hybrid": 0.XX, "content_based": 0.XX, "collaborative": 0.XX},
                ...
            ],
            "status": "ok"
        }
    """
    try:
        result = _evaluate_once()
        comparison = result["comparison"]
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e), "status": "error"}

    label_map = {
        "hybrid": "Hybrid",
        "content_based": "Content-Based",
        "collaborative": "Collaborative",
    }

    comparison_data = []
    for metric_name in ["precision_at_10", "recall_at_10", "coverage", "execution_time"]:
        row = {"metric": metric_name}
        for key, label in label_map.items():
            row[label.lower().replace("-", "_")] = float(comparison.get(key, {}).get(metric_name, 0.0))
        comparison_data.append(row)

    return {"comparison_table": comparison_data, "status": "ok"}


def _build_analysis(comparison: dict, collab_metrics: dict) -> dict:
    """Build a data-driven explanation of which method performs best.

    Args:
        comparison: Mapping of method name -> {precision_at_10, recall_at_10,
            coverage, execution_time}.
        collab_metrics: {"rmse": ..., "mae": ...} for collaborative filtering.

    Returns:
        Dict with the winning method, a ranked summary and human-readable
        explanation paragraphs.
    """
    label_map = {
        "hybrid": "Hybrid",
        "content_based": "Content-Based",
        "collaborative": "Collaborative",
    }

    # Rank primarily by Precision@10, tie-break by Recall@10.
    ranked = sorted(
        comparison.items(),
        key=lambda kv: (kv[1].get("precision_at_10", 0.0), kv[1].get("recall_at_10", 0.0)),
        reverse=True,
    )

    winner_key, winner_metrics = ranked[0]
    winner_label = label_map.get(winner_key, winner_key)

    explanations: list[str] = []
    explanations.append(
        f"Based on the current dataset, the {winner_label} approach performs best, "
        f"achieving a Precision@10 of {winner_metrics.get('precision_at_10', 0.0):.4f} "
        f"and Recall@10 of {winner_metrics.get('recall_at_10', 0.0):.4f}."
    )

    if winner_key == "hybrid":
        explanations.append(
            "The Hybrid method wins because it combines the textual similarity signal "
            "of content-based filtering with the user-behaviour signal of collaborative "
            "filtering. This lets it recommend drugs that are both semantically related "
            "to the query and historically well-rated, mitigating the weaknesses of each "
            "individual approach (cold-start for collaborative, popularity-blindness for content)."
        )
    elif winner_key == "content_based":
        explanations.append(
            "Content-Based filtering wins here because the drug reviews contain rich "
            "condition/symptom text, so TF-IDF + cosine similarity captures strong "
            "semantic relationships even when user-overlap in ratings is sparse."
        )
    else:
        explanations.append(
            "Collaborative filtering wins here because there is enough overlapping "
            "user-rating signal for SVD matrix factorisation to learn latent "
            "preferences that generalise across users."
        )

    explanations.append(
        f"Collaborative filtering rating-accuracy: RMSE = {collab_metrics.get('rmse', 0.0):.4f}, "
        f"MAE = {collab_metrics.get('mae', 0.0):.4f} (lower is better)."
    )

    ranking = [
        {
            "method": label_map.get(k, k),
            "precision_at_10": v.get("precision_at_10", 0.0),
            "recall_at_10": v.get("recall_at_10", 0.0),
            "coverage": v.get("coverage", 0.0),
            "execution_time": v.get("execution_time", 0.0),
        }
        for k, v in ranked
    ]

    return {
        "winner": winner_label,
        "winner_key": winner_key,
        "ranking": ranking,
        "explanation": explanations,
    }


@app.get("/analysis", tags=["Evaluation"])
async def get_analysis() -> dict:
    """Compare all three methods and explain why one performs better.

    Returns:
        {
            "winner": "Hybrid",
            "ranking": [...],
            "explanation": ["...", "..."],
            "status": "ok"
        }
    """
    try:
        result = _evaluate_once()
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e), "status": "error"}

    analysis = dict(result["analysis"])
    analysis["status"] = "ok"
    return analysis


@app.post("/preprocess", tags=["Data"])
async def preprocess_data() -> dict:
    """Trigger preprocessing pipeline manually.

    Returns:
        {"status": "success", "message": "..."}
    """
    global _content_recommender, _collaborative_recommender, _hybrid_recommender, _cleaned_df, _comparison_cache

    try:
        from app.preprocessing import run_preprocessing

        run_preprocessing()
        _cleaned_df = None
        _content_recommender = None
        _collaborative_recommender = None
        _hybrid_recommender = None
        _comparison_cache = None
        # Stale benchmark results no longer match the (rebuilt) data.
        if _EVALUATION_ARTIFACT.exists():
            _EVALUATION_ARTIFACT.unlink()
        load_recommenders()
        return {"status": "success", "message": "Preprocessing completed and recommenders reloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))