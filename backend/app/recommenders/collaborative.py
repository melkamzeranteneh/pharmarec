"""
PharmaRec - Collaborative Filtering Recommender

This module implements collaborative filtering using SVD from the Surprise library.

Algorithm:
    - Matrix Factorization with SVD (Singular Value Decomposition)
    - User-Drug-Rating matrix

Input:
    - User IDs, Drug IDs, Ratings from cleaned dataset

Output:
    - Top N recommended drugs for a given user with predicted ratings

Evaluation Metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

if TYPE_CHECKING:
    from surprise import Trainset


class CollaborativeRecommender:
    """Collaborative filtering recommender using SVD from Surprise library."""

    def __init__(self) -> None:
        """Initialize the collaborative filtering recommender.

        Attributes:
            df: DataFrame containing the cleaned dataset with user ratings.
            model: SVD model from Surprise library.
            trainset: Training dataset used for fitting the model.
            testset: Test dataset for evaluation.
            user_map: Mapping from user IDs to internal indices.
            drug_map: Mapping from drug names to internal indices.
            rmse: RMSE score after training (if evaluated).
            mae: MAE score after training (if evaluated).
        """
        self.df: pd.DataFrame | None = None
        self.model: SVD | None = None
        self.trainset: Trainset | None = None
        self.testset: list | None = None
        self.user_map: dict[str, int] | None = None
        self.drug_map: dict[str, int] | None = None
        self.drug_mean_ratings: dict[str, float] | None = None
        self.rmse: float | None = None
        self.mae: float | None = None

    def train(
        self,
        cleaned_dataset_path: Path | str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Train the SVD model on user-drug-rating data.

        Args:
            cleaned_dataset_path: Path to the cleaned dataset CSV file.
                                 Defaults to backend/data/cleaned_dataset.csv.
            test_size: Proportion of data to use for testing (default: 0.2).
            random_state: Random seed for reproducibility (default: 42).

        Raises:
            FileNotFoundError: If the cleaned dataset is not found.
            ValueError: If required columns are missing or no ratings exist.
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
        required_columns = ["drugName", "rating"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Create user IDs (assuming each unique user has a userId column or we generate one)
        # If userId doesn't exist, we'll create a synthetic one based on index
        if "userId" not in self.df.columns:
            # Create synthetic user IDs based on unique review patterns
            # This is a placeholder - real dataset should have user IDs
            self.df = self.df.copy()
            self.df["userId"] = self.df.groupby(
                ["drugName", "condition", "review"]
            ).ngroup().astype(str)
            print(f"[WARNING] No userId column found. Created {len(self.df['userId'].unique())} synthetic user IDs")

        # Create mappings for user and drug IDs
        self.user_map = {user: idx for idx, user in enumerate(self.df["userId"].unique())}
        self.drug_map = {drug: idx for idx, drug in enumerate(self.df["drugName"].unique())}

        # Precompute mean rating per drug (used as fallback collaborative signal)
        self.drug_mean_ratings = (
            self.df.groupby("drugName")["rating"].mean().to_dict()
        )

        # Prepare data for Surprise
        # Surprise expects: user_id, item_id, rating
        ratings_data = {
            "user_id": self.df["userId"].map(self.user_map).astype(str),
            "item_id": self.df["drugName"].map(self.drug_map).astype(str),
            "rating": self.df["rating"].values,
        }
        ratings_df = pd.DataFrame(ratings_data)

        # Define rating scale
        rating_scale = (1, 10)  # Assuming ratings are on a scale of 1-10
        if self.df["rating"].min() < 1 or self.df["rating"].max() > 10:
            rating_scale = (float(self.df["rating"].min()), float(self.df["rating"].max()))

        # Create Surprise dataset
        reader = Reader(rating_scale=rating_scale)
        data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)

        # Split into train and test sets
        self.trainset, self.testset = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        # Train SVD model
        self.model = SVD()
        self.model.fit(self.trainset)

        # Evaluate on test set
        if self.testset:
            predictions = self.model.test(self.testset)
            self.rmse = accuracy.rmse(predictions)
            self.mae = accuracy.mae(predictions)
            print(f"[INFO] CollaborativeRecommender trained on {len(self.trainset.all_users())} users and {len(self.trainset.all_items())} drugs")
            print(f"[INFO] Evaluation - RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}")
        else:
            print(f"[INFO] CollaborativeRecommender trained on {len(self.trainset.all_users())} users and {len(self.trainset.all_items())} drugs")

    def _resolve_user_id(self, user_id: str) -> str | int:
        """Resolve user_id to the key stored in user_map (may be str or int)."""
        if user_id in self.user_map:
            return user_id
        try:
            as_int = int(user_id)
            if as_int in self.user_map:
                return as_int
        except (ValueError, TypeError):
            pass
        raise ValueError(f"User '{user_id}' not found in training data")

    def get_drug_mean_rating(self, drug_name: str) -> float:
        """Return the mean rating for a drug across all users.

        Used as a collaborative-style signal when no specific user is provided
        (e.g. drug-name hybrid searches).
        """
        if self.drug_mean_ratings is None:
            return 0.0
        return float(self.drug_mean_ratings.get(drug_name, 0.0))

    def predict(self, user_id: str, drug_name: str) -> float:
        """Predict the rating a user would give to a drug.

        Args:
            user_id: ID of the user.
            drug_name: Name of the drug.

        Returns:
            Predicted rating (float).

        Raises:
            ValueError: If user or drug is not found in the training data.
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None or self.trainset is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        resolved_uid = self._resolve_user_id(user_id)

        if drug_name not in self.drug_map:
            raise ValueError(f"Drug '{drug_name}' not found in training data")

        # Get internal IDs
        uid = self.user_map[resolved_uid]
        iid = self.drug_map[drug_name]

        # Predict using the trained model
        prediction = self.model.predict(uid=str(uid), iid=str(iid))
        return prediction.est

    def recommend(self, user_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Recommend top N drugs for a given user based on predicted ratings.

        Args:
            user_id: ID of the user to recommend drugs for.
            top_n: Number of recommendations to return (default: 10).

        Returns:
            List of tuples containing (drug_name, predicted_rating) for the top N
            recommended drugs, sorted in descending order of predicted rating.
            Only includes drugs that the user has not yet rated.

        Raises:
            ValueError: If user is not found in the training data.
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None or self.trainset is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        resolved_uid = self._resolve_user_id(user_id)

        # Get all drugs the user has already rated
        rated_drugs = set()
        if self.df is not None:
            user_ratings = self.df[self.df["userId"] == resolved_uid]
            rated_drugs = set(user_ratings["drugName"].values)

        # Get all drugs in the dataset
        all_drugs = list(self.drug_map.keys())

        # Predict ratings for all unrated drugs
        recommendations: List[Tuple[str, float]] = []
        for drug in all_drugs:
            if drug not in rated_drugs:
                try:
                    predicted_rating = self.predict(user_id, drug)
                    recommendations.append((drug, predicted_rating))
                except:
                    # Skip if prediction fails
                    continue

        # Sort by predicted rating descending
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return recommendations[:top_n]

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate the model on the test set.

        Returns:
            Tuple of (rmse, mae) scores.

        Raises:
            RuntimeError: If the model has not been trained yet or test set is empty.
        """
        if self.rmse is None or self.mae is None:
            if self.model is None or self.testset is None:
                raise RuntimeError("Model has not been trained or no test set available")

            # Re-compute if not already computed
            predictions = self.model.test(self.testset)
            self.rmse = accuracy.rmse(predictions)
            self.mae = accuracy.mae(predictions)

        return self.rmse, self.mae


# ---------------------------------------------------------------------------
# Test Script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    print("=" * 60)
    print("  CollaborativeRecommender - Test Script")
    print("=" * 60)

    try:
        # Initialize and train the recommender
        print("\n[Test 1] Initializing CollaborativeRecommender...")
        recommender = CollaborativeRecommender()

        print("[Test 2] Training the recommender...")
        recommender.train()

        # Test evaluation metrics
        print("\n[Test 2b] Getting evaluation metrics...")
        if recommender.rmse is not None and recommender.mae is not None:
            print(f"  RMSE: {recommender.rmse:.4f}")
            print(f"  MAE: {recommender.mae:.4f}")
            print("  [PASS] Evaluation metrics computed")
        else:
            print("  [INFO] No test set available for evaluation")

        # Get a sample user from the dataset
        if recommender.df is not None and "userId" in recommender.df.columns:
            sample_user = recommender.df["userId"].iloc[0]
            print(f"\n[Test 3] Sample user: {sample_user}")

            # Get recommendations
            print(f"[Test 4] Getting recommendations for user '{sample_user}'...")
            recommendations = recommender.recommend(sample_user, top_n=5)

            print(f"\n[Test 5] Top 5 Recommendations for user '{sample_user}':")
            print("-" * 60)
            for i, (drug, predicted_rating) in enumerate(recommendations, 1):
                print(f"  {i}. {drug} | Predicted Rating: {predicted_rating:.4f}")
            print("-" * 60)

            # Test prediction for a specific drug
            print(f"\n[Test 6] Predicting rating for specific drug...")
            sample_drug = recommender.df["drugName"].iloc[0]
            predicted = recommender.predict(sample_user, sample_drug)
            print(f"  User '{sample_user}' predicted rating for '{sample_drug}': {predicted:.4f}")

            # Verify recommendations are sorted descending
            print("\n[Test 7] Verifying recommendations are sorted descending...")
            ratings = [r[1] for r in recommendations]
            is_sorted = all(ratings[i] >= ratings[i + 1] for i in range(len(ratings) - 1))
            if is_sorted:
                print("  [PASS] Recommendations are sorted in descending order")
            else:
                print("  [FAIL] Recommendations are not sorted correctly")

            # Test with non-existent user
            print("\n[Test 8] Testing with non-existent user...")
            try:
                recommender.recommend("NonExistentUser123")
                print("  [FAIL] Should have raised ValueError")
            except ValueError as e:
                print(f"  [PASS] Correctly raised ValueError: {e}")

            # Test recommend before train
            print("\n[Test 9] Testing recommend before train...")
            untrained_recommender = CollaborativeRecommender()
            try:
                untrained_recommender.recommend(sample_user)
                print("  [FAIL] Should have raised RuntimeError")
            except RuntimeError as e:
                print(f"  [PASS] Correctly raised RuntimeError: {e}")

        else:
            print("\n[WARNING] No userId column in dataset. Cannot run tests.")

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
