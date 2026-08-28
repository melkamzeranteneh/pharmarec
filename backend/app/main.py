"""
PharmaRec - FastAPI Application Entry Point

This module defines the FastAPI application and exposes endpoints for
drug recommendation, preprocessing, and evaluation.

Endpoints:
    GET  /              - Health check
    GET  /health        - Detailed health check
    GET  /drugs        - List drugs from cleaned dataset
    GET  /drugs/names  - List unique drug names
    POST /recommend     - Get recommendations (content, collaborative, or hybrid)
    GET  /metrics       - Get evaluation metrics comparison
    GET  /metrics/comparison - Get comparison table of all algorithms
    POST /preprocess    - Trigger preprocessing pipeline
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.preprocessing import run_preprocessing

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
_cleaned_df = None


def load_recommenders():
    """Load and initialize all recommender systems.

    First tries to download pre-computed weights from Hugging Face Hub.
    Falls back to full training if Hub download fails or weights are unavailable.
    """
    global _content_recommender, _collaborative_recommender, _hybrid_recommender, _cleaned_df
    
    from app.recommenders.content import ContentRecommender
    from app.recommenders.collaborative import CollaborativeRecommender
    from app.recommenders.hybrid import HybridRecommender
    from app.utils import ensure_weights
    
    _DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    _CLEANED_DATASET = "cleaned_dataset.csv"
    cleaned_path = _DATA_DIR / _CLEANED_DATASET
    
    # Step 1: Try downloading pre-computed weights from HF Hub
    print("[STARTUP] Checking for pre-computed weights...")
    hub_ok = ensure_weights(_DATA_DIR)
    
    # Step 2: If Hub download failed, try preprocessing locally
    if not hub_ok or not cleaned_path.exists():
        if not cleaned_path.exists():
            print("[STARTUP] Cleaned dataset not found. Running preprocessing...")
            try:
                run_preprocessing()
            except Exception as e:
                print(f"[STARTUP] Preprocessing failed: {e}")
                print("[STARTUP] Will attempt to continue with whatever is available")
    
    # Step 3: Load cleaned dataset
    if cleaned_path.exists():
        _cleaned_df = pd.read_csv(cleaned_path)
    else:
        print("[STARTUP] WARNING: No cleaned dataset available")
        _cleaned_df = pd.DataFrame()
    
    # Step 4: Initialize recommenders (they will use cached weights if available)
    print("[STARTUP] Initializing ContentRecommender...")
    _content_recommender = ContentRecommender()
    _content_recommender.fit()
    
    print("[STARTUP] Initializing CollaborativeRecommender...")
    _collaborative_recommender = CollaborativeRecommender()
    _collaborative_recommender.train()
    
    print("[STARTUP] Initializing HybridRecommender...")
    _hybrid_recommender = HybridRecommender(
        _content_recommender, 
        _collaborative_recommender
    )
    
    print("[STARTUP] All recommenders initialized successfully")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager."""
    global _content_recommender, _collaborative_recommender, _hybrid_recommender
    
    # Startup
    if _content_recommender is None:
        load_recommenders()
    
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="PharmaRec API",
    description="Comparative Drug Recommendation System",
    version="1.0.0",
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
    return {"status": "ok", "service": "PharmaRec API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Detailed health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
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
    global _cleaned_df
    
    if _cleaned_df is None:
        _DATA_DIR = Path(__file__).resolve().parent.parent / "data"
        cleaned_path = _DATA_DIR / "cleaned_dataset.csv"
        if not cleaned_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Cleaned dataset not found. Run /preprocess first."
            )
        _cleaned_df = pd.read_csv(cleaned_path)
    
    result_df = _cleaned_df[[
        "drugName", "condition", "review", "rating", "usefulCount"
    ]].iloc[offset:offset+limit]
    
    return result_df.to_dict(orient="records")


@app.get("/drugs/names", tags=["Data"])
async def get_drug_names(limit: int = 100) -> list[str]:
    """Retrieve unique drug names.
    
    Args:
        limit: Maximum number of unique drug names to return (default: 100).
    
    Returns:
        List of unique drug names.
    """
    global _cleaned_df
    
    if _cleaned_df is None:
        _DATA_DIR = Path(__file__).resolve().parent.parent / "data"
        cleaned_path = _DATA_DIR / "cleaned_dataset.csv"
        if not cleaned_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Cleaned dataset not found. Run /preprocess first."
            )
        _cleaned_df = pd.read_csv(cleaned_path)
    
    unique_drugs = _cleaned_df["drugName"].unique().tolist()
    return unique_drugs[:limit]


@app.post("/recommend", tags=["Recommendations"])
async def recommend(request: RecommendRequest) -> dict:
    """Get drug recommendations using the specified method.
    
    Request body:
    {
        "method": "content" | "collaborative" | "hybrid",
        "query": "Paracetamol" | "user123"
    }
    
    Returns:
        {
            "recommendations": [...],
            "method": "content" | "collaborative" | "hybrid",
            "query": "Paracetamol",
            "status": "success"
        }
    """
    global _content_recommender, _collaborative_recommender, _hybrid_recommender
    
    method = request.method.lower()
    query = request.query
    previous_searches = request.previous_searches
    
    try:
        if method == "content":
            if _content_recommender is None:
                raise HTTPException(status_code=500, detail="Content recommender not initialized")
            
            # If we have previous searches, find drugs similar to the combined history
            if previous_searches:
                # Get recommendations for each search in history, combine scores
                all_scores: dict[str, float] = {}
                all_conditions: dict[str, str] = {}
                
                # Include current query
                all_queries = [query] + previous_searches
                
                for q in all_queries:
                    try:
                        recs = _content_recommender.recommend(q, top_n=20)
                        for drug, condition, score in recs:
                            if drug not in all_scores:
                                all_scores[drug] = 0.0
                                all_conditions[drug] = condition
                            all_scores[drug] += score
                    except ValueError:
                        continue
                
                # Sort by combined score and take top 10
                sorted_drugs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                formatted = [
                    {
                        "drugName": drug,
                        "condition": all_conditions.get(drug, ""),
                        "similarityScore": float(score),
                    }
                    for drug, score in sorted_drugs
                ]
            else:
                recommendations = _content_recommender.recommend(query, top_n=10)
                formatted = [
                    {
                        "drugName": drug,
                        "condition": condition,
                        "similarityScore": float(score),
                    }
                    for drug, condition, score in recommendations
                ]
            
        elif method == "collaborative":
            if _collaborative_recommender is None:
                raise HTTPException(status_code=500, detail="Collaborative recommender not initialized")
            recommendations = _collaborative_recommender.recommend(query, top_n=10)
            formatted = [
                {
                    "drugName": drug,
                    "predictedRating": float(rating),
                }
                for drug, rating in recommendations
            ]
            
        elif method == "hybrid":
            if _hybrid_recommender is None:
                raise HTTPException(status_code=500, detail="Hybrid recommender not initialized")
            
            # For hybrid with history, aggregate content scores from all searches
            if previous_searches:
                all_hybrid_scores: dict[str, tuple[float, float, float]] = {}
                all_queries = [query] + previous_searches
                
                for q in all_queries:
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
                recommendations = _hybrid_recommender.recommend(
                    drug_name=query,
                    user_id=query,
                    top_n=10
                )
                formatted = [
                    {
                        "drugName": drug,
                        "contentScore": float(content),
                        "collabScore": float(collab),
                        "hybridScore": float(hybrid),
                    }
                    for drug, content, collab, hybrid in recommendations
                ]
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method: {method}. Must be 'content', 'collaborative', or 'hybrid'"
            )
        
        return {
            "recommendations": formatted,
            "method": method,
            "query": query,
            "status": "success"
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
                "hybrid": {"precision_at_10": ..., "recall_at_10": ..., "coverage": ..., "execution_time": ...},
                "collaborative": {"rmse": ..., "mae": ..., ...},
                "content": {"coverage": ..., ...}
            },
            "status": "ok"
        }
    """
    global _hybrid_recommender, _collaborative_recommender, _content_recommender
    
    metrics = {}
    
    # Query drugs auto-selected by the recommender (dataset has no repeat users)
    test_users = None
    
    # Hybrid metrics
    try:
        if _hybrid_recommender is not None:
            hybrid_metrics = _hybrid_recommender.evaluate(test_users=test_users, top_n=10)
            metrics["hybrid"] = {
                "precision_at_10": hybrid_metrics.get("precision_at_10", 0.0),
                "recall_at_10": hybrid_metrics.get("recall_at_10", 0.0),
                "coverage": hybrid_metrics.get("coverage", 0.0),
                "execution_time": hybrid_metrics.get("execution_time", 0.0),
            }
        else:
            metrics["hybrid"] = {"error": "not initialized"}
    except Exception as e:
        metrics["hybrid"] = {"error": str(e)}
    
    # Collaborative metrics
    try:
        if _collaborative_recommender is not None:
            rmse, mae = _collaborative_recommender.evaluate()
            metrics["collaborative"] = {
                "rmse": float(rmse),
                "mae": float(mae),
            }
        else:
            metrics["collaborative"] = {"error": "not initialized"}
    except Exception as e:
        metrics["collaborative"] = {"error": str(e)}
    
    # Content metrics
    try:
        if _content_recommender is not None:
            metrics["content"] = {"status": "loaded"}
        else:
            metrics["content"] = {"error": "not initialized"}
    except Exception as e:
        metrics["content"] = {"error": str(e)}
    
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
    global _hybrid_recommender, _collaborative_recommender
    
    try:
        if _hybrid_recommender is not None:
            table = _hybrid_recommender.get_comparison_table(
                test_users=None, top_n=10
            )
            
            # Convert to list of dicts.
            # table has methods as rows (index) and metrics as columns.
            comparison_data = []
            for metric_name in ["precision_at_10", "recall_at_10", "coverage", "execution_time"]:
                row = {"metric": metric_name}
                if metric_name in table.columns:
                    row["hybrid"] = float(table.loc["Hybrid", metric_name])
                    row["content_based"] = float(table.loc["Content-Based", metric_name])
                    row["collaborative"] = float(table.loc["Collaborative", metric_name])
                else:
                    row["hybrid"] = 0.0
                    row["content_based"] = 0.0
                    row["collaborative"] = 0.0
                comparison_data.append(row)
            
            return {"comparison_table": comparison_data, "status": "ok"}
        else:
            return {"error": "Hybrid recommender not initialized", "status": "error"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


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
    global _hybrid_recommender, _collaborative_recommender

    if _hybrid_recommender is None:
        return {"error": "Recommenders not initialized", "status": "error"}

    try:
        table = _hybrid_recommender.get_comparison_table(test_users=None, top_n=10)

        comparison = {
            "hybrid": table.loc["Hybrid"].to_dict(),
            "content_based": table.loc["Content-Based"].to_dict(),
            "collaborative": table.loc["Collaborative"].to_dict(),
        }

        collab_metrics = {"rmse": 0.0, "mae": 0.0}
        if _collaborative_recommender is not None:
            try:
                rmse_val, mae_val = _collaborative_recommender.evaluate()
                collab_metrics = {"rmse": float(rmse_val), "mae": float(mae_val)}
            except Exception:
                pass

        analysis = _build_analysis(comparison, collab_metrics)
        analysis["status"] = "ok"
        return analysis
    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.post("/preprocess", tags=["Data"])
async def preprocess_data() -> dict:
    """Trigger preprocessing pipeline manually.
    
    Returns:
        {"status": "success", "message": "..."}
    """
    global _content_recommender, _collaborative_recommender, _hybrid_recommender
    
    try:
        run_preprocessing()
        # Reload recommenders
        _content_recommender = None
        _collaborative_recommender = None
        _hybrid_recommender = None
        load_recommenders()
        return {"status": "success", "message": "Preprocessing completed and recommenders reloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
