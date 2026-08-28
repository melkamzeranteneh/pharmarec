"""
PharmaRec - Content-Based Filtering Recommender

This module implements content-based filtering using semantic text embeddings
(sentence-transformers) with a TF-IDF fallback.

Algorithm:
    - Encode each drug's features (condition + review text) into dense
      semantic vectors via a sentence-transformer model.
    - Cosine Similarity between embeddings for finding similar drugs.

Using semantic embeddings (instead of bag-of-words TF-IDF) captures meaning:
e.g. "depression" is close to "anxiety", and "birth control" is close to
"contraception", which the original TF-IDF pipeline could not express. The
clean ``condition`` field is prepended to the text so the therapeutic area
becomes a strong, shared signal.

Input:
    - Cleaned dataset with combined_text and condition columns

Output:
    - Top N similar drugs based on content
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from typing import Tuple

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "embeddings_cache.pkl"


class ContentRecommender:
    """Content-based recommender using semantic embeddings (TF-IDF fallback)."""

    def __init__(self, use_embeddings: bool = True) -> None:
        """Initialize the content-based recommender.

        Args:
            use_embeddings: If True (default), use a sentence-transformer model.
                Falls back to tuned TF-IDF if the model cannot be loaded.

        Attributes:
            df: DataFrame containing the cleaned dataset.
            embeddings: Dense semantic vectors (numpy array) per drug row.
            vectorizer: TF-IDF vectorizer (only used in fallback mode).
            similarity_matrix: Cosine similarity matrix between all drugs.
            mode: "embeddings" or "tfidf" depending on what was fitted.
        """
        self.df: pd.DataFrame | None = None
        self.embeddings: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.similarity_matrix: pd.DataFrame | None = None
        self.mode: str = "embeddings"
        self._use_embeddings = use_embeddings
        # Case-insensitive lookup: lowercase name -> canonical drugName
        self._drug_lookup: dict[str, str] = {}

    def _feature_text(self) -> List[str]:
        """Build the per-row feature text: condition + review text.

        Prepending the condition makes the therapeutic area a dominant,
        repeated signal so semantically/conditionally related drugs cluster.
        """
        assert self.df is not None
        cond = self.df["condition"].fillna("").astype(str)
        text = self.df["combined_text"].fillna("").astype(str)
        return (cond + ". " + cond + ". " + text).tolist()

    def _load_or_build_embeddings(
        self, texts: List[str], source_path: Optional[Path] = None
    ) -> Optional[np.ndarray]:
        """Load cached embeddings or compute (and cache) them.

        The cache is keyed by a hash of the source dataset (path + mtime + row
        count) so that ANY incoming/new data automatically triggers a rebuild,
        guaranteeing the content model is always trained on the latest data.
        """
        if source_path is not None and Path(source_path).exists():
            sp = Path(source_path)
            identity = f"{sp.resolve()}:{sp.stat().st_mtime}:{len(texts)}"
        else:
            identity = f"rows:{len(texts)}"
        key = hashlib.md5(identity.encode("utf-8")).hexdigest()
        if EMBEDDING_CACHE.exists():
            try:
                with open(EMBEDDING_CACHE, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("key") == key:
                    return cache["embeddings"]
            except Exception:
                pass
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME)
            emb = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            with open(EMBEDDING_CACHE, "wb") as f:
                pickle.dump({"key": key, "embeddings": emb}, f)
            return emb
        except Exception as e:
            print(f"[WARNING] Embedding model failed ({e}); falling back to TF-IDF")
            return None

    def fit(self, cleaned_dataset_path: Path | str | None = None) -> None:
        """Fit the content recommender and compute the similarity matrix.

        Tries to load pre-computed embeddings from cache first. If unavailable,
        falls back to computing embeddings from scratch.

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

        # Try loading pre-computed embeddings from cache first
        embeddings = None
        if self._use_embeddings:
            embeddings = self._load_cached_embeddings(cleaned_dataset_path)
            if embeddings is not None:
                print("[INFO] Loaded pre-computed embeddings from cache")

        # If no cache, compute from scratch
        if embeddings is None and self._use_embeddings:
            texts = self._feature_text()
            embeddings = self._load_or_build_embeddings(texts, cleaned_dataset_path)

        if embeddings is not None:
            self.mode = "embeddings"
            self.embeddings = embeddings
            sim = cosine_similarity(embeddings)
        else:
            # Fallback: tuned TF-IDF on the same feature text
            self.mode = "tfidf"
            texts = self._feature_text()
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.6,
            )
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            sim = cosine_similarity(tfidf_matrix)

        self.similarity_matrix = pd.DataFrame(
            sim,
            index=self.df.index,
            columns=self.df.index,
        )

        # Build case-insensitive lookup (lowercase -> canonical drug name)
        self._drug_lookup = {name.lower(): name for name in self.df["drugName"].unique()}

        print(
            f"[INFO] ContentRecommender fitted on {len(self.df)} drugs "
            f"(mode={self.mode})"
        )

    def _load_cached_embeddings(self, cleaned_dataset_path: Path) -> Optional[np.ndarray]:
        """Try to load pre-computed embeddings from the local cache file.

        Returns the embeddings array if the cache is valid, None otherwise.
        """
        if not EMBEDDING_CACHE.exists():
            return None
        try:
            with open(EMBEDDING_CACHE, "rb") as f:
                cache = pickle.load(f)
            # Validate cache key matches current dataset
            sp = Path(cleaned_dataset_path)
            identity = f"{sp.resolve()}:{sp.stat().st_mtime}:{len(self.df)}"
            expected_key = hashlib.md5(identity.encode("utf-8")).hexdigest()
            # Also accept the Colab-generated key (row-count based)
            colab_identity = f"rows:{len(self.df)}"
            colab_key = hashlib.md5(colab_identity.encode("utf-8")).hexdigest()
            if cache.get("key") in (expected_key, colab_key):
                emb = cache["embeddings"]
                if hasattr(emb, "shape") and emb.shape[0] == len(self.df):
                    return emb
        except Exception as e:
            print(f"[WARNING] Could not load cached embeddings: {e}")
        return None

    def _resolve_drug(self, drug_name: str) -> str:
        """Resolve a (possibly mis-cased) drug name to its canonical form.

        Raises ValueError if no case-insensitive match exists in the dataset.
        """
        canonical = self._drug_lookup.get(drug_name.lower())
        if canonical is None:
            raise ValueError(f"Drug '{drug_name}' not found in dataset")
        return canonical

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

        # Case-insensitive drug name resolution
        drug_name = self._resolve_drug(drug_name)

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
