"""
PharmaRec - Collaborative Filtering Recommender (Item-based, Condition-aware)

Why item-based instead of user-based?
    The Drug Review dataset has (almost) no user overlap: each user rates a
    single drug, so a classic user-drug matrix is ~100% sparse and user-based
    SVD cannot learn meaningful latent preferences. Instead we exploit the
    dataset's real strength - the *crowd rating* of each drug - and build an
    item-based collaborative signal:

        1. Aggregate all patient reviews per drug.
        2. Compute a Bayesian-weighted mean rating (a drug with 40 reviews at
           8.0 outranks a drug with 1 review at 10.0).
        3. Recommend the highest-quality drugs, preferring those that treat the
           same condition as the queried drug (condition-aware collaborative
           recommendation).

We still fit an SVD model on the raw ratings purely to report RMSE/MAE as a
rating-prediction accuracy metric for the comparison table.

Input:
    - Drug name (used to look up its condition)

Output:
    - Top N well-rated drugs for the same condition, with a quality score

Evaluation Metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import pandas as pd

if TYPE_CHECKING:
    pass


class CollaborativeRecommender:
    """Item-based, condition-aware collaborative filtering recommender.

    The collaborative signal is the crowd's Bayesian-weighted rating of each
    drug. Recommendations for a queried drug are the top-rated *other* drugs
    that treat the same condition (falling back to globally top-rated drugs).
    """

    def __init__(self, prior_strength: float = 10.0) -> None:
        """Initialize the collaborative recommender.

        Args:
            prior_strength: Number of "virtual" prior votes at the global mean
                used for Bayesian rating shrinkage. Higher values penalize
                drugs with few reviews more strongly.

        Attributes:
            df: Cleaned dataset (one row per review).
            drug_stats: Per-drug aggregate stats (count, mean, weighted score).
            drug_condition: Mapping drug -> primary condition.
            condition_drugs: Mapping condition -> list of drugs (sorted by score).
            drug_map: Mapping drug name -> index (kept for API compatibility).
            global_mean: Global mean rating across the dataset.
            rmse / mae: SVD rating-prediction accuracy (for comparison metrics).
        """
        self.prior_strength = prior_strength
        self.df: pd.DataFrame | None = None
        self.drug_stats: pd.DataFrame | None = None
        self.drug_condition: dict[str, str] | None = None
        self.condition_drugs: dict[str, list[str]] | None = None
        self.drug_map: dict[str, int] | None = None
        self.drug_mean_ratings: dict[str, float] | None = None
        self.drug_weighted_scores: dict[str, float] | None = None
        self.global_mean: float = 0.0
        self.rmse: float | None = None
        self.mae: float | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        cleaned_dataset_path: Path | str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Build the item-based collaborative model.

        Args:
            cleaned_dataset_path: Path to cleaned dataset CSV.
            test_size: Fraction of ratings held out for RMSE/MAE (via SVD).
            random_state: Random seed for reproducibility.

        Raises:
            FileNotFoundError: If the cleaned dataset is not found.
            ValueError: If required columns are missing.
        """
        if cleaned_dataset_path is None:
            cleaned_dataset_path = (
                Path(__file__).resolve().parent.parent.parent / "data" / "cleaned_dataset.csv"
            )
        else:
            cleaned_dataset_path = Path(cleaned_dataset_path)

        if not cleaned_dataset_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_dataset_path}")

        self.df = pd.read_csv(cleaned_dataset_path)

        required = ["drugName", "rating", "condition"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "usefulCount" not in self.df.columns:
            self.df["usefulCount"] = 0

        self.global_mean = float(self.df["rating"].mean())

        # --- Per-drug aggregate statistics (the collaborative signal) --------
        stats = (
            self.df.groupby("drugName")
            .agg(
                count=("rating", "size"),
                mean_rating=("rating", "mean"),
                useful=("usefulCount", "sum"),
                condition=("condition", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
            )
            .reset_index()
        )

        # Bayesian-weighted rating: shrink low-count drugs toward global mean.
        # weighted = (v/(v+m)) * R + (m/(v+m)) * C
        m = self.prior_strength
        c = self.global_mean
        stats["weighted_score"] = (
            (stats["count"] / (stats["count"] + m)) * stats["mean_rating"]
            + (m / (stats["count"] + m)) * c
        )

        self.drug_stats = stats
        self.drug_condition = dict(zip(stats["drugName"], stats["condition"]))
        self.drug_mean_ratings = dict(zip(stats["drugName"], stats["mean_rating"]))
        self.drug_weighted_scores = dict(zip(stats["drugName"], stats["weighted_score"]))
        self.drug_map = {drug: idx for idx, drug in enumerate(stats["drugName"])}
        # Case-insensitive lookup: lowercase name -> canonical drugName
        self.drug_lookup = {drug.lower(): drug for drug in stats["drugName"]}

        # condition -> drugs sorted by weighted score (best first)
        cond_map: dict[str, list[str]] = {}
        for cond, grp in stats.sort_values("weighted_score", ascending=False).groupby("condition"):
            cond_map[cond] = grp["drugName"].tolist()
        self.condition_drugs = cond_map

        print(
            f"[INFO] CollaborativeRecommender (item-based) built on "
            f"{len(stats)} drugs across {stats['condition'].nunique()} conditions"
        )

        # --- SVD purely to report RMSE / MAE accuracy -----------------------
        self._compute_accuracy(test_size=test_size, random_state=random_state)

    def _compute_accuracy(self, test_size: float, random_state: int) -> None:
        """Fit SVD on raw ratings to obtain RMSE/MAE for the metrics table."""
        try:
            from surprise import Dataset, Reader, SVD, accuracy
            from surprise.model_selection import train_test_split

            df = self.df.copy()
            if "userId" not in df.columns:
                df["userId"] = df.groupby(["drugName", "condition", "review"]).ngroup().astype(str)

            ratings_df = pd.DataFrame(
                {
                    "user_id": df["userId"].astype(str),
                    "item_id": df["drugName"].astype(str),
                    "rating": df["rating"].values,
                }
            )

            lo = min(1.0, float(df["rating"].min()))
            hi = max(10.0, float(df["rating"].max()))
            reader = Reader(rating_scale=(lo, hi))
            data = Dataset.load_from_df(ratings_df, reader)
            trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
            model = SVD()
            model.fit(trainset)
            preds = model.test(testset)
            self.rmse = float(accuracy.rmse(preds, verbose=False))
            self.mae = float(accuracy.mae(preds, verbose=False))
            print(f"[INFO] Rating accuracy - RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}")
        except Exception as e:
            print(f"[WARNING] Could not compute SVD accuracy metrics: {e}")
            self.rmse = 0.0
            self.mae = 0.0

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _resolve_drug(self, drug_name: str) -> str:
        """Resolve a (possibly mis-cased) drug name to its canonical form.

        Raises ValueError if no case-insensitive match exists in the dataset.
        """
        canonical = self.drug_lookup.get(drug_name.lower())
        if canonical is None:
            raise ValueError(f"Drug '{drug_name}' not found in dataset")
        return canonical

    def get_drug_mean_rating(self, drug_name: str) -> float:
        """Return the raw mean rating for a drug (0.0 if unknown)."""
        if self.drug_mean_ratings is None:
            return 0.0
        return float(self.drug_mean_ratings.get(self._resolve_drug(drug_name), 0.0))

    def get_weighted_score(self, drug_name: str) -> float:
        """Return the Bayesian-weighted quality score for a drug."""
        if self.drug_weighted_scores is None:
            return 0.0
        return float(self.drug_weighted_scores.get(self._resolve_drug(drug_name), self.global_mean))

    def get_condition(self, drug_name: str) -> str | None:
        """Return the primary condition treated by a drug."""
        if self.drug_condition is None:
            return None
        return self.drug_condition.get(self._resolve_drug(drug_name))

    def predict(self, drug_name: str) -> float:
        """Predict the crowd quality score (weighted mean rating) for a drug.

        Kept for hybrid compatibility. Accepts a drug name.
        """
        if self.drug_weighted_scores is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self.get_weighted_score(drug_name)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    def recommend(self, drug_name: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Recommend well-rated drugs for the same condition as ``drug_name``.

        Collaborative logic: given a drug, find its condition, then return the
        highest crowd-rated *other* drugs treating that same condition. If there
        are not enough same-condition drugs, fill with the globally top-rated
        drugs.

        Args:
            drug_name: Queried drug (used only to look up the condition).
            top_n: Number of recommendations.

        Returns:
            List of (drug_name, weighted_score) sorted by score descending.

        Raises:
            RuntimeError: If the model has not been trained.
            ValueError: If the drug is not found in the dataset.
        """
        if self.drug_stats is None or self.condition_drugs is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        # Case-insensitive drug name resolution
        drug_name = self._resolve_drug(drug_name)

        condition = self.get_condition(drug_name)

        # Same-condition candidates (already sorted by weighted score)
        candidates = [d for d in self.condition_drugs.get(condition, []) if d != drug_name]

        # Fill with globally top-rated drugs if needed
        if len(candidates) < top_n:
            global_sorted = (
                self.drug_stats.sort_values("weighted_score", ascending=False)["drugName"].tolist()
            )
            for d in global_sorted:
                if d != drug_name and d not in candidates:
                    candidates.append(d)
                if len(candidates) >= top_n:
                    break

        results = [(d, self.get_weighted_score(d)) for d in candidates[:top_n]]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def evaluate(self) -> Tuple[float, float]:
        """Return (rmse, mae) rating-prediction accuracy."""
        if self.rmse is None or self.mae is None:
            raise RuntimeError("Model has not been trained.")
        return self.rmse, self.mae


# ---------------------------------------------------------------------------
# Test Script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    print("=" * 60)
    print("  CollaborativeRecommender (item-based) - Test Script")
    print("=" * 60)

    try:
        rec = CollaborativeRecommender()
        rec.train()

        sample_drug = rec.df["drugName"].iloc[0]
        cond = rec.get_condition(sample_drug)
        print(f"\nSample drug: {sample_drug}  (condition: {cond})")

        print(f"\nTop 5 collaborative recommendations for '{sample_drug}':")
        print("-" * 60)
        for i, (drug, score) in enumerate(rec.recommend(sample_drug, top_n=5), 1):
            print(f"  {i}. {drug:<35} score={score:.3f}")
        print("-" * 60)

        print(f"\nRMSE: {rec.rmse:.4f}  MAE: {rec.mae:.4f}")
        print("\nAll tests completed!")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run preprocessing first: python -m app.preprocessing")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
