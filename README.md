# PharmaRec

**PharmaRec** is a comparative drug-recommendation system built for academic evaluation.
It implements and benchmarks **three recommendation strategies** side-by-side on a real
drug-review dataset, then explains — with measured metrics — why one approach wins.

> ⚠️ **Scope:** PharmaRec recommends *similar / related drugs* based on crowd reviews and
> therapeutic area. It is **not** a diagnostic or prescription tool, and it does not give
> medical advice.

---

## Table of Contents

- [Overview](#overview)
- [The Three Approaches](#the-three-approaches)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Evaluation Methodology](#evaluation-methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Frontend (React + TypeScript)](#frontend-react--typescript)
- [API Reference](#api-reference)
- [How to Reproduce the Metrics](#how-to-reproduce-the-metrics)
- [Key Design Decisions](#key-design-decisions)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

The repository compares three classical recommender paradigms on the same data:

| Approach | What it uses | What it returns |
|---|---|---|
| **Content-Based** | Drug text + condition, encoded as semantic embeddings | Drugs with similar reviews / therapeutic area |
| **Collaborative** | Crowd ratings, grouped by condition (item-based, quality-weighted) | Top-rated drugs for the same condition |
| **Hybrid** | Both of the above, combined | Best of both worlds, ranked by a weighted score |

A FastAPI backend serves recommendations and live evaluation metrics; a React frontend
lets a user search a drug, pick a method, and read an auto-generated **analysis & conclusion**
panel explaining which method performs best and why.

---

## The Three Approaches

### 1. Content-Based Filtering (`recommenders/content.py`)
- Each drug's feature text is built as `condition + condition + review` (condition is
  doubled so the therapeutic area dominates the signal).
- Text is encoded into dense **semantic vectors** with a sentence-transformer
  (`all-MiniLM-L6-v2`).
- Recommendations are the nearest neighbours by cosine similarity.
- A tuned **TF-IDF fallback** (`stop_words`, `ngram_range=(1,2)`, `min_df=3`, `max_df=0.6`)
  is used automatically if the embedding model cannot be loaded.
- Embeddings are cached to `data/embeddings_cache.pkl`, keyed by dataset identity
  (path + mtime + row count), so they rebuild automatically when the data changes.

### 2. Collaborative Filtering (`recommenders/collaborative.py`)
- **Item-based, condition-aware.** For a given drug it finds the condition, then returns
  the highest crowd-rated *other* drugs treating that same condition.
- Each drug's quality is a **Bayesian-weighted mean rating**
  `weighted = (v/(v+m))·R + (m/(v+m))·C`, which shrinks low-review-count drugs toward the
  global mean and prevents obscure drugs from dominating.
- An SVD model (surprise) is trained purely to report **RMSE / MAE** rating accuracy.
- *Why item-based and not user-based?* The dataset has no repeat users (each user rates
  exactly one drug), so a user–user matrix would be degenerate. The item/condition signal
  is the meaningful collaborative axis.

### 3. Hybrid (`recommenders/hybrid.py`)
- Candidate pool = text-similar drugs **∪** same-condition drugs (condition-aware, so
  unrelated high-rated drugs cannot leak in).
- `hybrid_score = content_weight · similarity + collab_weight · (rating / 10)`,
  with `content_weight = collab_weight = 0.5` by default (weights sum to 1).
- This keeps the semantic gain of content-based **and** the quality/condition signal of
  collaborative, while suppressing each method's individual weakness.

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────────────────┐
│   React + TypeScript    │  HTTP   │            FastAPI (backend)          │
│   Vite frontend         │ ──────▶ │                                       │
│  - DrugSearch           │         │  /recommend   /metrics /analysis ...  │
│  - MethodSelector       │ ◀────── │         │                             │
│  - RecommendationTable  │  JSON   │         ▼                             │
│  - AnalysisPanel        │         │  HybridRecommender                    │
└─────────────────────────┘         │     ├── ContentRecommender            │
                                     │     └── CollaborativeRecommender      │
                                     │              │                        │
                                     │              ▼                        │
                                     │   cleaned_dataset.csv (10,000 rows)   │
                                     └──────────────────────────────────────┘
```

The backend loads all three recommenders once at startup (and again after `/preprocess`),
so every method is always trained on the latest data.

---

## Dataset

- Source: Kaggle *Drug Review* dataset (`mohamedabdelwahabali/drugreview`), auto-downloaded
  via `kagglehub` on first run.
- After cleaning, **10,000 rows** are kept (first 10,000 — no random sampling, for speed and
  determinism).
- Pipeline (`app/preprocessing.py`): drop duplicates → handle missing values →
  lowercase → remove punctuation → remove stop-words → lemmatize → build `combined_text`.
- Final schema: `userId, drugName, condition, review, rating, date, usefulCount, combined_text`.

> Note: every `userId` in this dataset rates exactly one drug, so there is **no user overlap**.
> This shaped the evaluation design (see below).

---

## Evaluation Methodology

Because there are no repeat users, classic user-based evaluation is impossible. Instead,
evaluation is **query-drug based**:

- For a query drug, the **relevant set** = drugs that are highly rated by the crowd
  (mean rating ≥ 7.0) **and** related to the query by *either*:
  - treating the **same condition**, *or*
  - being **textually similar** (top content-based neighbours).
- This balanced definition avoids favouring any single method (the earlier version defined
  relevance as same-condition only, which made collaborative win by construction).

**Metrics** (centralized in `app/evaluation.py`):
- `precision_at_k` — fraction of top-K recs that are relevant.
- `recall_at_k` — fraction of relevant items captured in top-K.
- `coverage` — proportion of the catalog ever recommended.
- `rmse` / `mae` — rating prediction error (collaborative SVD).
- `execution_time` — latency per query.

---

## Results

Evaluated on the 10,000-row dataset (Precision@10 / Recall@10, k=10):

| Recommender   | Precision@10 | Recall@10 | Coverage | Exec time (s) |
|---------------|-------------:|----------:|---------:|--------------:|
| **Hybrid**        | **0.766** | **0.501** | 0.175 | 0.010 |
| Collaborative | 0.642 | 0.350 | 0.169 | 0.016 |
| Content-Based | 0.464 | 0.278 | 0.177 | 0.011 |

**Why the Hybrid wins:**
1. Content-based alone retrieves *semantically* similar drugs, but those often treat a
   *related yet different* condition (e.g. anxiety vs depression), which fails the relevance
   gate — hence its lower precision.
2. Collaborative alone is strong on same-condition quality but ignores textual nuance and
   has weaker recall.
3. The Hybrid combines both signals **within a condition-aware candidate pool**, so it keeps
   the semantic gain *and* the crowd-quality/condition signal, while filtering out the
   cross-condition noise that hurts content-based. The result is the highest precision and
   recall of the three.

---

## Project Structure

```
pharmarec/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, lifespan loading, endpoints
│   │   ├── preprocessing.py        # dataset download + cleaning pipeline
│   │   ├── evaluation.py           # metric implementations (single source of truth)
│   │   └── recommenders/
│   │       ├── content.py          # semantic-embedding content-based
│   │       ├── collaborative.py    # item-based, condition-aware CF
│   │       └── hybrid.py           # weighted combination + comparison table
│   └── data/
│       ├── drug_review.csv         # raw dataset
│       ├── cleaned_dataset.csv     # 10,000 cleaned rows
│       └── embeddings_cache.pkl    # cached content embeddings
└── frontend/
    └── src/
        ├── pages/Home.tsx
        ├── components/
        │   ├── DrugSearch.tsx
        │   ├── MethodSelector.tsx
        │   ├── RecommendationTable.tsx
        │   └── AnalysisPanel.tsx
        └── App.tsx
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) `kagglehub` credentials for first-time dataset download; otherwise place a
  `drug_review.csv` in `backend/data/`.

### Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # or: pip install fastapi uvicorn pandas scikit-learn sentence-transformers surprise kagglehub nltk

# First run downloads + cleans the dataset and builds embeddings (can take a few minutes)
python -m app.preprocessing

# Start the API
uvicorn app.main:app --reload --port 8000
```

The API docs are available at `http://localhost:8000/docs`.

### Frontend (React + TypeScript)

```bash
cd frontend
npm install
npm run dev          # Vite dev server (defaults to http://localhost:5173)
```

The frontend expects the backend at `http://localhost:8000` (configure the base URL in the
API client if different).

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health/root info |
| GET | `/health` | Detailed health check (which recommenders are loaded) |
| GET | `/drugs?limit=&offset=` | Paginated drug records |
| GET | `/drugs/names?limit=` | Unique drug names (for the search box) |
| POST | `/recommend` | Body: `{"method": "content|collaborative|hybrid", "query": "<drug>"}` |
| GET | `/metrics` | Per-method evaluation metrics |
| GET | `/metrics/comparison` | Side-by-side comparison table |
| GET | `/analysis` | Winner + ranking + textual explanation |
| POST | `/preprocess` | Re-run cleaning pipeline and reload all recommenders |

**Drug name search is case-insensitive** — `VENLAFAXINE`, `venlafaxine`, and `Venlafaxine`
all resolve to the same drug.

---

## How to Reproduce the Metrics

```bash
cd backend
source venv/bin/activate
python -c "
from app.recommenders.content import ContentRecommender
from app.recommenders.collaborative import CollaborativeRecommender
from app.recommenders.hybrid import HybridRecommender

c = ContentRecommender(); c.fit()
cf = CollaborativeRecommender(); cf.train()
h = HybridRecommender(c, cf)
print(h.get_comparison_table(test_users=None, top_n=10))
"
```

Or via the live API: `GET /analysis` and `GET /metrics/comparison`.

---

## Key Design Decisions

- **Item-based / condition-aware collaborative** instead of user-based, because the dataset
  has no repeat users.
- **Semantic embeddings** (not raw TF-IDF) for content-based, with a TF-IDF fallback.
- **Condition-aware hybrid candidate pool** so unrelated high-rated drugs cannot dilute results.
- **Balanced, query-drug-based relevance** so the comparison is fair to all three methods.
- **First-10,000 rows** (no random sampling) for fast, deterministic preprocessing.
- **Case-insensitive search** across all three recommenders.

---

## Limitations & Future Work

- The dataset lacks repeat users, limiting classic collaborative-filtering evaluation.
- Content-based precision is capped by cross-condition semantic neighbours; conditioning
  embeddings on diagnosis codes could improve it.
- Coverage is modest because the candidate pool is intentionally condition-restricted.
- Future: add a user-history-aware mode, expose tunable hybrid weights via the UI, and
  benchmark a neural collaborative model.

---

## License

For academic / educational use.
