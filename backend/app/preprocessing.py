"""
PharmaRec - Data Preprocessing Pipeline

This module handles loading, cleaning, and preprocessing the Drug Review dataset
for downstream recommendation algorithms.

Pipeline Steps:
    1. Auto-download dataset from Kaggle (if missing or synthetic)
    2. Load raw dataset
    3. Remove duplicate records
    4. Handle missing values
    5. Lowercase text fields
    6. Remove punctuation from review text
    7. Remove stop words
    8. Lemmatize review text
    9. Create combined_text = condition + review
    10. Save cleaned_dataset.csv
"""

from __future__ import annotations

import string
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# NLTK data (loaded lazily so importing this module never blocks on downloads)
# ---------------------------------------------------------------------------
_NLTK_RESOURCES: list[str] = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


@lru_cache(maxsize=1)
def _ensure_nltk():
    """Download the NLTK corpora/tokenizers needed by the cleaning pipeline."""
    import nltk

    for _resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(
                f"tokenizers/{_resource}" if _resource.startswith("punkt") else f"corpora/{_resource}"
            )
        except LookupError:
            nltk.download(_resource, quiet=True)
    return True

# ---------------------------------------------------------------------------
# Kaggle dataset config
# ---------------------------------------------------------------------------
_KAGGLE_DATASET: str = "mohamedabdelwahabali/drugreview"
_MIN_REAL_ROWS: int = 10000  # require at least this many rows in the raw dataset
_TRAIN_SIZE: int = 9000
_TEST_SIZE: int = 1000
_TARGET_ROWS: int = 10000  # total rows to keep in the raw dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
_RAW_DATASET: str = "drug_review.csv"
_CLEANED_DATASET: str = "cleaned_dataset.csv"

_PUNCTUATION: str = string.punctuation


def _stop_words() -> set[str]:
    """English stop-words (ensures NLTK data is available first)."""
    _ensure_nltk()
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


def _lemmatizer():
    """WordNet lemmatizer (ensures NLTK data is available first)."""
    _ensure_nltk()
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    """Download the real Drug Review dataset from Kaggle using kagglehub.

    Returns:
        Path to the directory containing the downloaded dataset files.
    """
    try:
        import kagglehub
    except ImportError:
        print("[WARNING] kagglehub not installed. Run: pip install kagglehub")
        print("[INFO] Falling back to local drug_review.csv")
        return _DATA_DIR

    print(f"[INFO] Downloading dataset '{_KAGGLE_DATASET}' from Kaggle...")
    try:
        path: Path = Path(kagglehub.dataset_download(_KAGGLE_DATASET))
        print(f"[INFO] Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[WARNING] Kaggle download failed: {e}")
        print("[INFO] Falling back to local drug_review.csv")
        return _DATA_DIR


def _combine_kaggle_csvs(download_dir: Path) -> Path:
    """Combine train/test/validation CSVs from Kaggle download into a single CSV.

    The Kaggle dataset provides three cleaned splits (train, test, validation).
    We combine them into a single raw dataset for preprocessing.

    Args:
        download_dir: Directory containing the downloaded dataset files.

    Returns:
        Path to the combined CSV file.
    """
    import glob

    csv_patterns = ["drug_review_train.csv", "drug_review_test.csv", "drug_review_validation.csv"]
    parts: list[pd.DataFrame] = []

    for pattern in csv_patterns:
        files = list(download_dir.glob(pattern))
        if files:
            df = pd.read_csv(files[0])
            parts.append(df)
            print(f"[INFO] Loaded {files[0].name}: {len(df):,} rows")

    if not parts:
        # Fall back to raw TSV files
        tsv_patterns = ["drugsComTrain_raw.tsv", "drugsComTest_raw.tsv"]
        for pattern in tsv_patterns:
            files = list(download_dir.glob(pattern))
            if files:
                df = pd.read_csv(files[0], sep="\t")
                parts.append(df)
                print(f"[INFO] Loaded {files[0].name}: {len(df):,} rows")

    if not parts:
        raise FileNotFoundError(f"No CSV or TSV files found in {download_dir}")

    combined = pd.concat(parts, ignore_index=True)

    # Take the first N rows (no random sampling, to keep the pipeline fast and
    # deterministic). This keeps _TARGET_ROWS total rows for training/inference.
    if len(combined) > _TARGET_ROWS:
        combined = combined.head(_TARGET_ROWS)
        print(f"[INFO] Kept first {_TARGET_ROWS:,} rows (of {len(combined):,} available)")
    else:
        print(f"[INFO] Using all {len(combined):,} rows (less than requested {_TARGET_ROWS:,})")

    # Drop unnecessary columns
    drop_cols = [c for c in ["Unnamed: 0", "review_length"] if c in combined.columns]
    if drop_cols:
        combined = combined.drop(columns=drop_cols)

    # Rename 'patient_id' to 'userId' if present (for collaborative filtering)
    if "patient_id" in combined.columns:
        combined = combined.rename(columns={"patient_id": "userId"})

    # Ensure core columns exist
    core_cols = ["drugName", "condition", "review", "rating", "usefulCount"]
    missing = [c for c in core_cols if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Save combined raw dataset
    raw_path = _DATA_DIR / _RAW_DATASET
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(raw_path, index=False)
    print(f"[INFO] Combined dataset saved to {raw_path} ({len(combined):,} rows, {len(combined.columns)} cols)")
    return raw_path


def ensure_real_dataset() -> Path:
    """Ensure the real Kaggle dataset is available in the data directory.

    If the existing drug_review.csv is synthetic (too few rows), it will be
    replaced with the real dataset from Kaggle.

    Returns:
        Path to the raw dataset CSV file.
    """
    raw_path = _DATA_DIR / _RAW_DATASET

    # Check if we already have the real dataset
    if raw_path.exists():
        try:
            row_count = sum(1 for _ in open(raw_path)) - 1  # skip header
            if row_count >= _MIN_REAL_ROWS:
                print(f"[INFO] Using existing dataset: {raw_path} ({row_count:,} rows)")
                return raw_path
            print(f"[INFO] Existing dataset is synthetic ({row_count} rows). Downloading real dataset...")
        except Exception:
            pass

    # Download from Kaggle
    download_dir = download_dataset()

    if download_dir != _DATA_DIR:
        return _combine_kaggle_csvs(download_dir)

    raise FileNotFoundError(
        f"Dataset not found at {raw_path}. "
        f"Place the Kaggle CSV file at this path or ensure kagglehub is configured."
    )


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters from text."""
    return text.translate(str.maketrans("", "", _PUNCTUATION))


def remove_stop_words(text: str) -> str:
    """Remove English stop words from tokenized text."""
    from nltk.tokenize import word_tokenize

    tokens: list[str] = word_tokenize(text)
    filtered: list[str] = [token for token in tokens if token not in _stop_words()]
    return " ".join(filtered)


def lemmatize(text: str) -> str:
    """Lemmatize each token in the text."""
    from nltk.tokenize import word_tokenize

    tokens: list[str] = word_tokenize(text)
    lemmatized: list[str] = [_lemmatizer().lemmatize(token) for token in tokens]
    return " ".join(lemmatized)


def clean_review(text: str) -> str:
    """Apply the full text-cleaning pipeline to a single review string.

    Steps:
        1. Lowercase
        2. Remove punctuation
        3. Remove stop words
        4. Lemmatize
    """
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_stop_words(text)
    text = lemmatize(text)
    return text


# ---------------------------------------------------------------------------
# Dataset operations
# ---------------------------------------------------------------------------

def load_dataset(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw Drug Review dataset from CSV.

    If no filepath is given, the real Kaggle dataset is automatically
    downloaded and used.

    Args:
        filepath: Optional explicit path to the CSV file.
                  Defaults to ``backend/data/drug_review.csv`` (auto-downloaded).

    Returns:
        A pandas DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if filepath is None:
        filepath = ensure_real_dataset()
    else:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df: pd.DataFrame = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset with {len(df):,} records from {filepath.name}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    before: int = len(df)
    df = df.drop_duplicates()
    removed: int = before - len(df)
    print(f"[INFO] Removed {removed:,} duplicate records ({before:,} -> {len(df):,})")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset.

    - Drops rows where ``drugName`` or ``review`` is missing (critical fields).
    - Fills missing ``condition`` with 'Unknown'.
    - Fills missing ``rating`` with the column median.
    - Fills missing ``usefulCount`` with 0.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    before: int = len(df)

    # Drop rows missing critical fields
    df = df.dropna(subset=["drugName", "review"]).copy()

    # Fill non-critical missing values
    df["condition"] = df["condition"].fillna("Unknown")
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["usefulCount"] = df["usefulCount"].fillna(0).astype(int)

    dropped: int = before - len(df)
    print(f"[INFO] Dropped {dropped:,} rows with missing critical fields")
    print(f"[INFO] Remaining records: {len(df):,}")
    return df


def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create a ``combined_text`` column by concatenating condition and review.

    The combined text is used as the primary input for content-based filtering.

    Args:
        df: Input DataFrame with ``condition`` and ``review`` columns.

    Returns:
        DataFrame with an added ``combined_text`` column.
    """
    df["combined_text"] = df["condition"].astype(str) + " " + df["review"].astype(str)
    print("[INFO] Created 'combined_text' column (condition + review)")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    input_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Execute the full preprocessing pipeline.

    Args:
        input_path:  Optional path to the raw dataset CSV.
        output_path: Optional path for the cleaned output CSV.
                     Defaults to ``backend/data/cleaned_dataset.csv``.

    Returns:
        The cleaned pandas DataFrame.
    """
    print("=" * 60)
    print("  PharmaRec - Data Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load dataset
    print("\n[Step 1/8] Loading dataset...")
    df = load_dataset(input_path)

    # 2. Remove duplicates
    print("\n[Step 2/8] Removing duplicates...")
    df = remove_duplicates(df)

    # 3. Handle missing values
    print("\n[Step 3/8] Handling missing values...")
    df = handle_missing_values(df)

    # 4. Lowercase text
    print("\n[Step 4/8] Lowercasing review text...")
    df["review"] = df["review"].apply(lowercase)
    print("[INFO] Lowercased 'review' column")

    # 5. Remove punctuation
    print("\n[Step 5/8] Removing punctuation...")
    df["review"] = df["review"].apply(remove_punctuation)
    print("[INFO] Removed punctuation from 'review' column")

    # 6. Remove stop words
    print("\n[Step 6/8] Removing stop words...")
    df["review"] = df["review"].apply(remove_stop_words)
    print("[INFO] Removed stop words from 'review' column")

    # 7. Lemmatize
    print("\n[Step 7/8] Lemmatizing review text...")
    df["review"] = df["review"].apply(lemmatize)
    print("[INFO] Lemmatized 'review' column")

    # 8. Create combined_text (after cleaning so it uses cleaned review)
    print("\n[Step 8/8] Creating combined_text column...")
    df = create_combined_text(df)

    # 9. Save cleaned dataset
    if output_path is None:
        output_path = _DATA_DIR / _CLEANED_DATASET
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Cleaned dataset saved to: {output_path}")
    print(f"[DONE] Final dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
