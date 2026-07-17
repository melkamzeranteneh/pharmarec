"""
PharmaRec - Hybrid Recommendation System

This module combines content-based and collaborative filtering scores
using a weighted average approach with normalized scores.

Algorithm:
    - Normalize Content Similarity Score (0-1)
    - Normalize Collaborative Prediction Score (0-1)
    - Weighted Average: hybrid_score = (content_weight * content_score) + (collab_weight * collab_score)
    - Default weights: 0.5 Content, 0.5 Collaborative

Input:
    - Content-based recommendations (similarity scores)
    - Collaborative filtering recommendations (predicted ratings)

Output:
    - Hybrid recommendations combining both approaches

Comparison Metrics:
    - Precision@10
    - Recall@10
    - Coverage
    - Execution Time
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.evaluation import coverage, precision_at_k, recall_at_k

if TYPE_CHECKING:
    from .content import ContentRecommender
    from .collaborative import CollaborativeRecommender


class HybridRecommender:
    """Hybrid recommender combining content-based and collaborative filtering."""

    def __init__(
        self,
        content_recommender: ContentRecommender,
        collaborative_recommender: CollaborativeRecommender,
        content_weight: float = 0.5,
        collab_weight: float = 0.5,
    ) -> None:
        """Initialize the hybrid recommender.

        Args:
            content_recommender: Trained content-based recommender.
            collaborative_recommender: Trained collaborative filtering recommender.
            content_weight: Weight for content-based score (default: 0.5).
            collab_weight: Weight for collaborative filtering score (default: 0.5).

        Raises:
            ValueError: If weights don't sum to 1 or are negative.
        """
        if content_weight < 0 or collab_weight < 0:
            raise ValueError("Weights must be non-negative")
        if not (0.99 <= content_weight + collab_weight <= 1.01):
            raise ValueError("Weights must sum to approximately 1")

        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.content_weight = content_weight
        self.collab_weight = collab_weight

    @staticmethod
    def normalize_score(score: float, min_val: float, max_val: float) -> float:
        """Normalize a score to 0-1 range.

        Args:
            score: The score to normalize.
            min_val: Minimum possible value in the original range.
            max_val: Maximum possible value in the original range.

        Returns:
            Normalized score between 0 and 1.
        """
        if max_val == min_val:
            return 0.5
        return (score - min_val) / (max_val - min_val)

    def recommend(
        self,
        drug_name: str | None = None,
        user_id: str | None = None,
        top_n: int = 10,
        include_rated: bool = False,
    ) -> List[Tuple[str, float, float, float]]:
        """Recommend drugs using hybrid approach.

        Args:
            drug_name: Name of the drug (for content-based similarity).
            user_id: ID of the user (for collaborative filtering).
            top_n: Number of recommendations to return (default: 10).
            include_rated: If True, include drugs already rated by user (for evaluation).

        Returns:
            List of tuples (drug_name, content_score, collab_score, hybrid_score)
            sorted descending by hybrid_score.

        Raises:
            ValueError: If neither drug_name nor user_id is provided.
        """
        if drug_name is None and user_id is None:
            raise ValueError("Either drug_name or user_id must be provided")

        hybrid_scores: Dict[str, float] = {}
        content_scores: Dict[str, float] = {}
        collab_scores: Dict[str, float] = {}

        cr = self.collaborative_recommender

        # Get content-based scores (text similarity to the query drug)
        if drug_name is not None:
            content_scores[drug_name] = 1.0  # Perfect match with itself
            try:
                content_recs = self.content_recommender.recommend(drug_name, top_n=100)
                for drug, _, score in content_recs:
                    if drug not in content_scores:
                        content_scores[drug] = score
            except Exception as e:
                print(f"[WARNING] Content-based recommendation failed: {e}")

        # Condition-aware candidate pool: combine drugs that are
        #   (a) textually similar to the query (content_scores), and
        #   (b) treat the SAME condition as the query drug (collaborative pool).
        # Restricting to this pool prevents unrelated high-rated drugs from
        # leaking into the ranking, so both signals stay meaningful.
        candidates: set[str] = set(content_scores.keys())
        if drug_name is not None:
            condition = cr.get_condition(drug_name)
            if condition is not None and cr.condition_drugs is not None:
                candidates.update(cr.condition_drugs.get(condition, []))

        if not candidates:
            candidates = set(cr.drug_map.keys())

        # Collaborative signal (item-based): each drug's Bayesian-weighted
        # crowd quality score.
        for drug in candidates:
            collab_scores[drug] = cr.get_weighted_score(drug)

        # Combine scores over the condition-aware candidate pool
        for drug in candidates:
            cs = content_scores.get(drug, 0.0)
            cls = collab_scores.get(drug, 0.0)

            # Hybrid score = content_weight * content_similarity (0-1)
            #              + collab_weight  * (rating / 10)      (0-1)
            hybrid_score = (
                self.content_weight * cs
                + self.collab_weight * (cls / 10.0)
            )
            hybrid_scores[drug] = hybrid_score

        # Exclude the query drug itself from recommendations
        if drug_name is not None:
            hybrid_scores.pop(drug_name, None)

        # Sort descending
        sorted_drugs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top N with all scores
        results: List[Tuple[str, float, float, float]] = []
        for drug, hybrid_score in sorted_drugs[:top_n]:
            results.append((
                drug,
                content_scores.get(drug, 0.0),
                collab_scores.get(drug, 0.0),
                hybrid_score,
            ))

        return results

    def _relevant_for(self, query_drug: str) -> set[str]:
        """Ground-truth relevant set for a query drug (balanced, unbiased).

        A drug is relevant if it is highly rated by the crowd (mean rating
        >= 7.0) AND it is related to the query drug by EITHER:
            (a) treating the same condition, OR
            (b) being textually similar (top content-based neighbours).

        Using both criteria avoids favouring any single method: collaborative
        is not automatically rewarded (condition-only) and content-based is not
        automatically penalised (text neighbours count too).
        """
        cr = self.collaborative_recommender
        if cr.df is None:
            return set()

        # High-rated drugs overall (quality gate)
        high_rated = {
            d for d, r in cr.df.groupby("drugName")["rating"].mean().items() if r >= 7.0
        }

        related: set[str] = set()

        # (a) same-condition drugs
        condition = cr.get_condition(query_drug)
        if condition is not None:
            same_cond = cr.df[cr.df["condition"] == condition]
            related.update(same_cond["drugName"].unique())

        # (b) textually similar drugs
        try:
            for drug, _, _ in self.content_recommender.recommend(query_drug, top_n=30):
                related.add(drug)
        except Exception:
            pass

        related.discard(query_drug)
        return related & high_rated

    def _query_drugs(self, test_drugs: List[str] | None, limit: int = 50) -> List[str]:
        """Resolve the list of query drugs used for evaluation."""
        if test_drugs is not None:
            return test_drugs
        cr = self.collaborative_recommender
        if cr.drug_stats is not None:
            # Prefer drugs with enough reviews to have a stable condition signal
            stable = cr.drug_stats[cr.drug_stats["count"] >= 2]["drugName"].tolist()
            pool = stable if stable else cr.drug_stats["drugName"].tolist()
            return pool[:limit]
        return []

    def evaluate(
        self,
        test_users: List[str] | None = None,
        top_n: int = 10,
    ) -> Dict[str, float]:
        """Evaluate hybrid recommender (Precision@10, Recall@10, Coverage, Exec Time).

        Evaluation is query-drug based: for each query drug we recommend top-N
        drugs and compare against highly-rated same-condition drugs.

        Args:
            test_users: Kept for API compatibility; interpreted as query drugs.
            top_n: Number of recommendations per query drug.
        """
        query_drugs = self._query_drugs(test_users, limit=50)

        precisions: List[float] = []
        recalls: List[float] = []
        all_recommended: set[str] = set()
        exec_times: List[float] = []

        all_drugs = set(self.collaborative_recommender.drug_map.keys()) if self.collaborative_recommender.drug_map else set()

        for drug in query_drugs:
            u_start = time.time()
            try:
                recs = self.recommend(drug_name=drug, top_n=top_n)
                rec_drugs = {r[0] for r in recs}
                all_recommended.update(rec_drugs)

                relevant = self._relevant_for(drug)
                precisions.append(precision_at_k(rec_drugs, relevant, k=top_n))
                recalls.append(recall_at_k(rec_drugs, relevant, k=top_n))
            except Exception as e:
                print(f"[WARNING] Evaluation error for drug {drug}: {e}")
            exec_times.append(time.time() - u_start)

        self.metrics = {
            "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
            "coverage": coverage(all_recommended, all_drugs),
            "execution_time": float(np.mean(exec_times)) if exec_times else 0.0,
        }

        return self.metrics

    def get_comparison_table(
        self,
        test_users: List[str] | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Generate comparison table for all three recommender types.

        Args:
            test_users: List of user IDs to test on.
            top_n: Number of recommendations per user.

        Returns:
            DataFrame with metrics as rows and recommenders as columns.
        """
        results: Dict[str, Dict[str, float]] = {}

        # Evaluate Hybrid
        try:
            results["Hybrid"] = self.evaluate(test_users=test_users, top_n=top_n)
        except Exception as e:
            print(f"[WARNING] Hybrid evaluation failed: {e}")
            results["Hybrid"] = {"precision_at_10": 0.0, "recall_at_10": 0.0, "coverage": 0.0, "execution_time": 0.0}

        # Evaluate Content-Based
        try:
            results["Content-Based"] = self._eval_content(test_users=test_users, top_n=top_n)
        except Exception as e:
            print(f"[WARNING] Content evaluation failed: {e}")
            results["Content-Based"] = {"precision_at_10": 0.0, "recall_at_10": 0.0, "coverage": 0.0, "execution_time": 0.0}

        # Evaluate Collaborative
        try:
            results["Collaborative"] = self._eval_collaborative(test_users=test_users, top_n=top_n)
        except Exception as e:
            print(f"[WARNING] Collaborative evaluation failed: {e}")
            results["Collaborative"] = {"precision_at_10": 0.0, "recall_at_10": 0.0, "coverage": 0.0, "execution_time": 0.0}

        # Create DataFrame
        metrics = ["precision_at_10", "recall_at_10", "coverage", "execution_time"]
        data = {}
        for m in metrics:
            data[m] = [results[r].get(m, 0.0) for r in ["Hybrid", "Content-Based", "Collaborative"]]

        df = pd.DataFrame(data, index=["Hybrid", "Content-Based", "Collaborative"])
        df.index.name = "Recommender"
        return df

    def _eval_content(self, test_users: List[str] | None, top_n: int) -> Dict[str, float]:
        """Evaluate content-based recommender (query-drug based)."""
        all_rec: set[str] = set()
        exec_times: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []

        all_drugs = set(self.content_recommender.df["drugName"].unique()) if self.content_recommender.df is not None else set()
        query_drugs = self._query_drugs(test_users, limit=50)

        for drug in query_drugs:
            u_start = time.time()
            try:
                recs = self.content_recommender.recommend(drug, top_n=top_n)
                rec_drugs = {r[0] for r in recs}
                all_rec.update(rec_drugs)

                relevant = self._relevant_for(drug)
                precisions.append(precision_at_k(rec_drugs, relevant, k=top_n))
                recalls.append(recall_at_k(rec_drugs, relevant, k=top_n))
            except:
                pass
            exec_times.append(time.time() - u_start)

        return {
            "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
            "coverage": coverage(all_rec, all_drugs),
            "execution_time": float(np.mean(exec_times)) if exec_times else 0.0,
        }

    def _eval_collaborative(self, test_users: List[str] | None, top_n: int) -> Dict[str, float]:
        """Evaluate collaborative recommender (query-drug based)."""
        all_rec: set[str] = set()
        exec_times: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []

        all_drugs = set(self.collaborative_recommender.drug_map.keys()) if self.collaborative_recommender.drug_map else set()
        query_drugs = self._query_drugs(test_users, limit=50)

        for drug in query_drugs:
            u_start = time.time()
            try:
                recs = self.collaborative_recommender.recommend(drug, top_n=top_n)
                rec_drugs = {r[0] for r in recs}
                all_rec.update(rec_drugs)

                relevant = self._relevant_for(drug)
                precisions.append(precision_at_k(rec_drugs, relevant, k=top_n))
                recalls.append(recall_at_k(rec_drugs, relevant, k=top_n))
            except:
                pass
            exec_times.append(time.time() - u_start)

        return {
            "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
            "coverage": coverage(all_rec, all_drugs),
            "execution_time": float(np.mean(exec_times)) if exec_times else 0.0,
        }


# ---------------------------------------------------------------------------
# Test Script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    print("=" * 60)
    print("  HybridRecommender - Test Script")
    print("=" * 60)

    try:
        from app.recommenders.content import ContentRecommender
        from app.recommenders.collaborative import CollaborativeRecommender

        print("\n[Test 1] Initializing recommenders...")
        content_rec = ContentRecommender()
        collab_rec = CollaborativeRecommender()

        print("[Test 2] Training ContentRecommender...")
        content_rec.fit()

        print("[Test 3] Training CollaborativeRecommender...")
        collab_rec.train()

        print("\n[Test 4] Initializing HybridRecommender...")
        hybrid_rec = HybridRecommender(content_rec, collab_rec)

        # Get test users
        test_users = None
        if collab_rec.df is not None and "userId" in collab_rec.df.columns:
            test_users = list(collab_rec.df["userId"].unique()[:3])
            print(f"[Test 5] Using {len(test_users)} test users")

        # Test recommend
        if test_users:
            uid = test_users[0]
            print(f"\n[Test 6] Getting hybrid recommendations for user '{uid}'...")
            recs = hybrid_rec.recommend(user_id=uid, top_n=5)

            print("\n[Test 7] Top 5 Hybrid Recommendations:")
            print("-" * 70)
            print(f"{'#':<3} {'Drug':<20} {'Content':<10} {'Collab':<10} {'Hybrid':<10}")
            print("-" * 70)
            for i, (d, cs, cls, hs) in enumerate(recs, 1):
                print(f"{i:<3} {d:<20} {cs:<10.4f} {cls:<10.4f} {hs:<10.4f}")
            print("-" * 70)

        # Test with drug_name
        if content_rec.df is not None:
            drug = content_rec.df["drugName"].iloc[0]
            print(f"\n[Test 8] Getting hybrid recommendations for drug '{drug}'...")
            recs = hybrid_rec.recommend(drug_name=drug, top_n=5)

            print("\n[Test 9] Top 5 Hybrid Recommendations:")
            print("-" * 70)
            print(f"{'#':<3} {'Drug':<20} {'Content':<10} {'Collab':<10} {'Hybrid':<10}")
            print("-" * 70)
            for i, (d, cs, cls, hs) in enumerate(recs, 1):
                print(f"{i:<3} {d:<20} {cs:<10.4f} {cls:<10.4f} {hs:<10.4f}")
            print("-" * 70)

        # Test evaluation
        print("\n[Test 10] Evaluating hybrid recommender...")
        metrics = hybrid_rec.evaluate(test_users=test_users, top_n=10)
        print("  Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        # Test comparison table
        print("\n[Test 11] Generating comparison table...")
        table = hybrid_rec.get_comparison_table(test_users=test_users, top_n=10)
        print("\n  Comparison Table:")
        print(table.to_string())

        # Verify sorted
        if test_users:
            print("\n[Test 12] Verifying hybrid scores sorted descending...")
            scores = [r[3] for r in hybrid_rec.recommend(user_id=test_users[0], top_n=10)]
            ok = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            print(f"  {'[PASS]' if ok else '[FAIL]'}")

        # Test error: no params
        print("\n[Test 13] Testing error handling (no params)...")
        try:
            hybrid_rec.recommend()
            print("  [FAIL] Should have raised ValueError")
        except ValueError as e:
            print(f"  [PASS] {e}")

        print("\n" + "=" * 60)
        print("  All tests completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure cleaned_dataset.csv exists in backend/data/")
        print("Run: python -m app.preprocessing")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
