# PharmaRec Backend

FastAPI-based backend for the PharmaRec comparative drug recommendation system.

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── recommend.py    # Recommendation API endpoints
│   │   └── metrics.py      # Evaluation metrics API endpoints
│   ├── recommenders/
│   │   ├── __init__.py
│   │   ├── content.py      # Content-Based Filtering (Phase 2)
│   │   ├── collaborative.py # Collaborative Filtering (Phase 3)
│   │   └── hybrid.py       # Hybrid Recommendation (Phase 4)
│   ├── __init__.py
│   ├── main.py            # FastAPI application entry point
│   ├── preprocessing.py   # Data preprocessing pipeline (Phase 1)
│   └── evaluation.py      # Evaluation metrics (Phase 5)
├── data/
│   └── drug_review.csv    # Raw dataset (to be downloaded from Kaggle)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate   # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the dataset from Kaggle:

```bash
# Ensure you have kaggle CLI installed and configured
pip install kaggle

# Set up your Kaggle API token
# Create ~/.kaggle/kaggle.json with:
# {"username": "your_username", "key": "your_api_key"}
# chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download -d jessicali9530/kuc-hackathon-winter-2018

# Unzip and move to data directory
unzip kuc-hackathon-winter-2018.zip -d data/
mv data/drugsCom_train.csv data/drug_review.csv  # Rename if needed
```

**Note:** The dataset file should be named `drug_review.csv` and placed in the `backend/data/` directory.

Expected columns in the dataset:
- drugName
- condition
- review
- rating
- usefulCount

### 4. Run Preprocessing

```bash
# Method 1: Run preprocessing directly
python -m app.preprocessing

# Method 2: Run via FastAPI (auto-runs on startup if cleaned_dataset.csv doesn't exist)
python -m uvicorn app.main:app --reload
```

The preprocessing pipeline will:
1. Load the raw dataset
2. Remove duplicate records
3. Handle missing values (drop critical, fill non-critical)
4. Lowercase text
5. Remove punctuation
6. Remove stop words
7. Lemmatize text
8. Create combined_text column (condition + review)
9. Save cleaned_dataset.csv

Output: `backend/data/cleaned_dataset.csv`

## Running the API

```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload
```

The API will be available at: http://localhost:8000

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health check |
| GET | `/drugs?limit=100` | Get drugs from cleaned dataset |
| POST | `/preprocess` | Manually trigger preprocessing |
| POST | `/recommend/content` | Content-based recommendations (Phase 2) |
| POST | `/recommend/collaborative` | Collaborative filtering recommendations (Phase 3) |
| POST | `/recommend/hybrid` | Hybrid recommendations (Phase 4) |
| GET | `/metrics` | Get evaluation metrics (Phase 5) |
| GET | `/metrics/comparison` | Get comparison table (Phase 5) |

## Implementation Phases

### Phase 1: Data Preprocessing ✅ COMPLETE

- [x] Backend folder structure
- [x] requirements.txt
- [x] preprocessing.py with full pipeline
- [x] Dataset loading and cleaning
- [x] cleaned_dataset.csv generation

### Phase 2: Content-Based Filtering (To Do)

- [ ] TF-IDF vectorization
- [ ] Cosine similarity
- [ ] ContentRecommender class
- [ ] API endpoint /recommend/content

### Phase 3: Collaborative Filtering (To Do)

- [ ] User-Drug-Rating matrix
- [ ] SVD implementation (Surprise)
- [ ] CollaborativeRecommender class
- [ ] API endpoint /recommend/collaborative

### Phase 4: Hybrid Recommendation (To Do)

- [ ] Weighted average of scores
- [ ] HybridRecommender class
- [ ] API endpoint /recommend/hybrid

### Phase 5: Evaluation (To Do)

- [ ] Precision@K
- [ ] Recall@K
- [ ] RMSE
- [ ] MAE
- [ ] Coverage
- [ ] Execution Time
- [ ] Comparison table

### Phase 6: Backend API (Partially Complete)

- [x] Health endpoints
- [x] Data endpoints
- [x] Preprocessing endpoint
- [ ] Recommendation endpoints (Phases 2-4)
- [ ] Metrics endpoints (Phase 5)

### Phase 7: Frontend (To Do)

React frontend with Vite and TypeScript.

## Testing

After downloading the dataset and running preprocessing:

```bash
# Check if cleaned_dataset.csv was created
ls -lh backend/data/cleaned_dataset.csv

# Run the API and test endpoints
python -m uvicorn app.main:app --reload

# Test with curl
curl http://localhost:8000/
curl http://localhost:8000/drugs?limit=5
```

## Notes

- The preprocessing pipeline automatically runs on application startup if `cleaned_dataset.csv` doesn't exist.
- NLTK data (punkt, stopwords, wordnet, omw-1.4) is automatically downloaded on first run.
- The dataset from Kaggle is: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
