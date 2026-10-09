"""
PharmaRec - Content-Based Filtering Recommender

Implements content-based filtering using dense semantic text embeddings
(sentence-transformers) with a tuned TF-IDF fallback.

Algorithm:
    - Each drug's feature text is built as ``condition + condition + review``
      (the therapeutic area is repeated so it dominates the signal).
    - Feature text is encoded into dense semantic vectors.
    - Recommendations are the nearest neighbours by cosine similarity.

Memory note (free tiers):
    The classic approach stores a full N x N cosine-similarity matrix, which
    would consume ~800 MB for 10,000 rows. Instead we keep one aggregated,
    L2-normalised vector per *drug* (a 10k x 384 float32 array ≈ 15 MB) and
    compute cosine similarity to that drug vector on the fly per request
    (a single cheap matmul, a few milliseconds). This preserves identical
    ranking quality while staying comfortably inside a 512 MB budget.

    The sentence-transformers model is only imported *lazily*, when embeddings
    actually need to be *encoded*. Loading a pre-computed embeddings cache only
    requires numpy + pickle, so runtime never touches torch.

Input:
    - Cleaned dataset with combined_text and condition columns

Output:
    - Top N similar drugs based on content
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from typing import List, Optional, Tuple

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "embeddings_cache.pkl"

# Hard cap so the on-the-fly query stays trivially fast even on the smallest CPU.
_QUERY_TOP_N = 100


class ContentRecommender:
    """Content-based recommender using semantic embeddings (TF-IDF fallback).

    Attributes:
        df: Cleaned dataset (one row per review).
        mode: ``"embeddings"`` or ``"tfidf"`` depending on what was fitted.
        drug_names: Canonical drug names, aligned with ``drug_vectors``.
        drug_vectors: L2-normalised float32 matrix (n_drugs x dim).
        vectorizer / tfidf_matrix: Used only in the TF-IDF fallback mode.
    """

    def __init__(self, use_embeddings: bool = True) -> None:
        """Initialize the content-based recommender.

        Args:
            use_embeddings: Prefer semantic embeddings over TF-IDF. ``True`` by
                default. If embeddings cannot be loaded/encoded, the recommender
                automatically falls back to tuned TF-IDF.
        """
        self.df: pd.DataFrame | None = None
        self.embeddings: Optional[np.ndarray] = None
        self.drug_names: list[str] = []
        self.drug_vectors: Optional[np.ndarray] = None
        self._drug_row_indices: dict[str, list[int]] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: object = None
        self.mode: str = "embeddings"
        self._use_embeddings = use_embeddings
        # Case-insensitive lookup: lowercase name -> canonical drugName
        self._drug_lookup: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Feature text
    # ------------------------------------------------------------------
    def _feature_text(self) -> List[str]:
        """Build the per-row feature text: ``condition + condition + review``.

        Prepending the condition twice makes the therapeutic area a dominant,
        repeated signal so semantically/conditionally related drugs cluster.
        """
        assert self.df is not None
        cond = self.df["condition"].fillna("").astype(str)
        text = self.df["combined_text"].fillna("").astype(str)
        return (cond + ". " + cond + ". " + text).tolist()

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(rows: int, source_path: Optional[Path]) -> str:
        """Stable cache key: fits both colab-generated (row-count only) caches."""
        if source_path is not None and Path(source_path).exists():
            sp = Path(source_path)
            identity = f"{sp.resolve()}:{sp.stat().st_mtime}:{rows}"
        else:
            identity = f"rows:{rows}"
        return hashlib.md5(identity.encode("utf-8")).hexdigest()

    def _load_cached_embeddings(self, cleaned_dataset_path: Path) -> Optional[np.ndarray]:
        """Load pre-computed per-row embeddings from the local cache file."""
        if not EMBEDDING_CACHE.exists():
            return None
        try:
            with open(EMBEDDING_CACHE, "rb") as f:
                cache = pickle.load(f)
            expected_key = self._cache_key(len(self.df or []), cleaned_dataset_path)
            colab_key = self._cache_key(len(self.df or []), None)
            if cache.get("key") in (expected_key, colab_key):
                emb = cache["embeddings"]
                if hasattr(emb, "shape") and emb.shape[0] == len(self.df):
                    return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            print(f"[WARNING] Could not load cached embeddings: {e}")
        return None

    def _build_embeddings(self, texts: List[str], source_path: Path) -> Optional[np.ndarray]:
        """Encode texts with the sentence-transformer (lazy import) and cache them."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[WARNING] sentence-transformers not installed; using TF-IDF fallback")
            return None

        print(f"[INFO] Encoding {len(texts)} rows with '{MODEL_NAME}' (first run only)...")
        model = SentenceTransformer(MODEL_NAME)
        emb = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emb = np.asarray(emb, dtype=np.float32)
        key = self._cache_key(len(texts), source_path)
        try:
            EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(EMBEDDING_CACHE, "wb") as f:
                pickle.dump({"key": key, "embeddings": emb}, f)
        except Exception as e:
            print(f"[WARNING] Could not cache embeddings: {e}")
        return emb

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, cleaned_dataset_path: Path | str | None = None) -> None:
        """Fit the content recommender and prepare per-drug vectors.

        Args:
            cleaned_dataset_path: Path to the cleaned dataset CSV.
                Defaults to ``backend/data/cleaned_dataset.csv``.

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

        self.df = pd.read_csv(cleaned_dataset_path, usecols=["drugName", "condition", "combined_text"])

        required = ["drugName", "condition", "combined_text"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        embeddings: Optional[np.ndarray] = None
        if self._use_embeddings:
            embeddings = self._load_cached_embeddings(cleaned_dataset_path)
            if embeddings is not None:
                print("[INFO] Loaded pre-computed embeddings from cache")
            else:
                embeddings = self._build_embeddings(self._feature_text(), cleaned_dataset_path)

        if embeddings is not None:
            self.mode = "embeddings"
            self.embeddings = embeddings
            self._build_drug_vectors()
        else:
            # Fallback: tuned TF-IDF on the same feature text.
            self.mode = "tfidf"
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.6,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self._feature_text())

        self._drug_lookup = {name.lower(): name for name in self.df["drugName"].unique()}
        print(f"[INFO] ContentRecommender fitted on {len(self.df)} rows (mode={self.mode})")

    def _build_drug_vectors(self) -> None:
        """Aggregate the per-row embeddings into one vector per drug.

        Row indices are grouped per drug and averaged, then L2-normalised so the
        dot product between two vectors equals their cosine similarity.
        """
        assert self.df is not None and self.embeddings is not None

        unique_names = self.df["drugName"].unique().tolist()
        self.drug_names = unique_names
        self._drug_row_indices = {name: [] for name in unique_names}
        for row_idx, name in enumerate(self.df["drugName"]):
            self._drug_row_indices[name].append(row_idx)

        vectors = np.empty((len(unique_names), self.embeddings.shape[1]), dtype=np.float32)
        for i, name in enumerate(unique_names):
            row_idx = self._drug_row_indices[name]
            vectors[i] = self.embeddings[row_idx].mean(axis=0)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.drug_vectors = (vectors / norms).astype(np.float32)

        # Free the per-row array after aggregation to keep memory minimal.
        self.embeddings = None
        self._drug_row_indices = {}

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _resolve_drug(self, drug_name: str) -> str:
        """Resolve a (possibly mis-cased) drug name to its canonical form.

        Raises ValueError if no case-insensitive match exists in the dataset.
        """
        canonical = self._drug_lookup.get(drug_name.lower())
        if canonical is None:
            raise ValueError(f"Drug '{drug_name}' not found in dataset")
        return canonical

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    def _recommend_embeddings(self, drug_name: str, top_n: int) -> List[Tuple[str, str, float]]:
        """Nearest neighbours by cosine similarity against drug vectors."""
        assert self.drug_vectors is not None

        drug_index = self.drug_names.index(drug_name)
        query = self.drug_vectors[drug_index]
        sims = self.drug_vectors @ query  # both L2-normalised -> cosine similarity

        order = np.argsort(-sims)  # descending
        results: List[Tuple[str, str, float]] = []
        for idx in order:
            if idx == drug_index:
                continue
            name = self.drug_names[int(idx)]
            condition = self.df.loc[self.df["drugName"] == name, "condition"].iat[0]
            results.append((name, condition, float(sims[idx])))
            if len(results) >= top_n:
                break
        return results

    def _recommend_tfidf(self, drug_name: str, top_n: int) -> List[Tuple[str, str, float]]:
        """Nearest neighbours over the sparse TF-IDF matrix.

        Mimics the classic behaviour: average the per-review similarity scores
        of the query drug's rows, then rank neighbour drugs by that average.
        """
        assert self.df is not None and self.vectorizer is not None and self.tfidf_matrix is not None

        query_rows = self.df.index[self.df["drugName"] == drug_name].tolist()
        # average of the cosine similarities of the query drug's rows vs all rows
        row_sims = cosine_similarity(self.tfidf_matrix[query_rows], self.tfidf_matrix).mean(axis=0)
        sims = pd.Series(row_sims, index=self.df.index)
        per_drug = sims.groupby(self.df["drugName"]).mean()

        results: List[Tuple[str, str, float]] = []
        for candidate, score in per_drug.sort_values(ascending=False).items():
            if candidate == drug_name:
                continue
            condition = self.df.loc[self.df["drugName"] == candidate, "condition"].iat[0]
            results.append((candidate, condition, float(score)))
            if len(results) >= top_n:
                break
        return results

    def recommend(self, drug_name: str, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Recommend top N similar drugs based on content.

        Args:
            drug_name: Name of the drug to find similar drugs for.
            top_n: Number of recommendations to return (default: 10).

        Returns:
            List of tuples ``(drug_name, condition, similarity_score)`` sorted
            descending by similarity. The queried drug is excluded.

        Raises:
            ValueError: If the drug is not found in the dataset.
            RuntimeError: If the recommender has not been fitted yet.
        """
        if self.df is None:
            raise RuntimeError("Recommender has not been fitted. Call fit() first.")

        drug_name = self._resolve_drug(drug_name)

        if self.mode == "embeddings":
            if self.drug_vectors is None:
                raise RuntimeError("Recommender has not been fitted. Call fit() first.")
            return self._recommend_embeddings(drug_name, top_n)
        return self._recommend_tfidf(drug_name, top_n)


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
        recommender = ContentRecommender()
        recommender.fit()

        sample_drug = recommender.df["drugName"].iloc[0]
        print(f"\nSample drug: {sample_drug}  (mode={recommender.mode})")

        recommendations = recommender.recommend(sample_drug, top_n=5)
        print(f"\nTop 5 Recommendations for '{sample_drug}':")
        print("-" * 60)
        for i, (drug, condition, score) in enumerate(recommendations, 1):
            print(f"  {i}. {drug} | Condition: {condition} | Score: {score:.4f}")
        print("-" * 60)

        try:
            recommender.recommend("NonExistentDrug123")
            print("  [FAIL] Should have raised ValueError")
        except ValueError:
            print("  [PASS] Correctly raised ValueError")

        unfitted = ContentRecommender()
        try:
            unfitted.recommend(sample_drug)
            print("  [FAIL] Should have raised RuntimeError")
        except RuntimeError:
            print("  [PASS] Correctly raised RuntimeError")

        recs = recommender.recommend(sample_drug, top_n=10)
        if sample_drug not in [r[0] for r in recs]:
            print("  [PASS] Queried drug not in recommendations")
        else:
            print("  [FAIL] Queried drug found in recommendations")

        scores = [r[2] for r in recs]
        if all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
            print("  [PASS] Scores sorted descending")
        else:
            print("  [FAIL] Scores not sorted")

        print("\nAll tests completed!")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run preprocessing first: python -m app.preprocessing")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)