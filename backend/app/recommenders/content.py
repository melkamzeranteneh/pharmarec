"""
PharmaRec - Content-Based Filtering Recommender

This module implements content-based filtering using TF-IDF and Cosine Similarity.

Algorithm:
    - TF-IDF vectorization of drug features (combined_text)
    - Cosine Similarity for finding similar drugs

Input:
    - Cleaned dataset with combined_text column

Output:
    - Top N similar drugs based on content
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from typing import Tuple


class ContentRecommender:
    """Content-based recommender using TF-IDF and Cosine Similarity."""

    def __init__(self) -> None:
        """Initialize the content-based recommender.

        Attributes:
            df: DataFrame containing the cleaned dataset.
            vectorizer: TF-IDF vectorizer fitted on combined_text.
            similarity_matrix: Cosine similarity matrix between all drugs.
            drug_indices: Mapping from drug names to DataFrame indices.
        """
        self.df: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.similarity_matrix: pd.DataFrame | None = None
        self.drug_indices: dict[str, int] | None = None

    def fit(self, cleaned_dataset_path: Path | str | None = None) -> None:
        """Fit the TF-IDF vectorizer and compute similarity matrix.

        Args:
            cleaned_dataset_path: Path to the cleaned dataset CSV file.
                                 Defaults to backend/data/cleaned_dataset.csv.

        Raises:
            FileNotFoundError: If the cleaned dataset is not found.
            ValueError: If required columns are missing.
        """
        # Load cleaned dataset
        if cleaned_dataset_path is None:
            cleaned_dataset_path = (
                Path(__file__).resolve().parent.parent.parent / "data" / "cleaned_dataset.csv"
            )
        else:
            cleaned_dataset_path = Path(cleaned_dataset_path)

        if not cleaned_dataset_path.exists():
            raise FileNotFoundError(
                f"Cleaned dataset not found at: {cleaned_dataset_path}"
            )

        self.df = pd.read_csv(cleaned_dataset_path)

        # Verify required columns exist
        required_columns = ["drugName", "condition", "combined_text"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

        # Compute cosine similarity matrix
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(tfidf_matrix),
            index=self.df.index,
            columns=self.df.index,
        )

        print(f"[INFO] ContentRecommender fitted on {len(self.df)} drugs")

    def recommend(self, drug_name: str, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Recommend top N similar drugs based on content.

        Args:
            drug_name: Name of the drug to find similar drugs for.
            top_n: Number of recommendations to return (default: 10).

        Returns:
            List of tuples containing (drug_name, condition, similarity_score) for
            the top N most similar drugs, sorted in descending order of similarity.
            The queried drug is excluded from the results.

        Raises:
            ValueError: If the drug is not found in the dataset.
            RuntimeError: If the recommender has not been fitted yet.
        """
        if self.df is None or self.similarity_matrix is None:
            raise RuntimeError("Recommender has not been fitted. Call fit() first.")

        if drug_name not in self.df["drugName"].values:
            raise ValueError(f"Drug '{drug_name}' not found in dataset")

        # Get all indices where drugName matches
        drug_indices = self.df[self.df["drugName"] == drug_name].index.tolist()

        # Get similarity scores for the queried drug
        # Average similarity if drug appears multiple times
        sim_scores: pd.Series = self.similarity_matrix.loc[drug_indices].mean(axis=0)

        # Get top N+1 most similar drugs (including the queried drug itself)
        top_n_plus_1 = sim_scores.nlargest(top_n + len(drug_indices))

        # Filter out the queried drug
        recommendations: List[Tuple[str, str, float]] = []
        for idx, score in top_n_plus_1.items():
            current_drug = self.df.loc[idx, "drugName"]
            if current_drug != drug_name:
                condition = self.df.loc[idx, "condition"]
                recommendations.append((current_drug, condition, float(score)))

            # Stop when we have enough recommendations
            if len(recommendations) >= top_n:
                break

        # Sort by similarity score descending (already sorted from nlargest)
        # Take only top_n
        recommendations = recommendations[:top_n]

        return recommendations


# ---------------------------------------------------------------------------
# Test Script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    print("=" * 60)
    print("  ContentRecommender - Test Script")
    print("=" * 60)

    try:
        # Initialize and fit the recommender
        print("\n[Test 1] Initializing ContentRecommender...")
        recommender = ContentRecommender()

        print("[Test 2] Fitting the recommender...")
        recommender.fit()

        # Get a sample drug from the dataset
        sample_drug = recommender.df["drugName"].iloc[0]
        print(f"\n[Test 3] Sample drug: {sample_drug}")

        # Get recommendations
        print(f"[Test 4] Getting recommendations for '{sample_drug}'...")
        recommendations = recommender.recommend(sample_drug, top_n=5)

        print(f"\n[Test 5] Top 5 Recommendations for '{sample_drug}':")
        print("-" * 60)
        for i, (drug, condition, score) in enumerate(recommendations, 1):
            print(f"  {i}. {drug} | Condition: {condition} | Score: {score:.4f}")
        print("-" * 60)

        # Test with non-existent drug
        print("\n[Test 6] Testing with non-existent drug...")
        try:
            recommender.recommend("NonExistentDrug123")
            print("  [FAIL] Should have raised ValueError")
        except ValueError as e:
            print(f"  [PASS] Correctly raised ValueError: {e}")

        # Test recommend before fit
        print("\n[Test 7] Testing recommend before fit...")
        unfitted_recommender = ContentRecommender()
        try:
            unfitted_recommender.recommend(sample_drug)
            print("  [FAIL] Should have raised RuntimeError")
        except RuntimeError as e:
            print(f"  [PASS] Correctly raised RuntimeError: {e}")

        # Verify queried drug is excluded
        print(f"\n[Test 8] Verifying queried drug '{sample_drug}' is excluded...")
        recommendations = recommender.recommend(sample_drug, top_n=10)
        drug_names = [r[0] for r in recommendations]
        if sample_drug not in drug_names:
            print(f"  [PASS] Queried drug not in recommendations")
        else:
            print(f"  [FAIL] Queried drug found in recommendations")

        # Verify scores are sorted descending
        print("\n[Test 9] Verifying scores are sorted descending...")
        scores = [r[2] for r in recommendations]
        is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        if is_sorted:
            print("  [PASS] Scores are sorted in descending order")
        else:
            print("  [FAIL] Scores are not sorted correctly")

        print("\n" + "=" * 60)
        print("  All tests completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure cleaned_dataset.csv exists in backend/data/")
        print("Run preprocessing first: python -m app.preprocessing")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
