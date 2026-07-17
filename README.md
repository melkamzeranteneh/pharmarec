# PharmaRec - Comparative Healthcare Drug Recommendation System

PharmaRec is a full-stack web application that compares three recommendation algorithms (Content-Based Filtering, Collaborative Filtering, and Hybrid) on healthcare drug review data.

## Tech Stack

### Backend
- **Python** 3.12
- **FastAPI** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (TF-IDF, Cosine Similarity)
- **Surprise** - Collaborative filtering (SVD)
- **NLTK** - Natural language processing

### Frontend
- **React** 18 - UI library
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool
- **Tailwind CSS** - Utility-first CSS (via CDN)
- **Axios** - HTTP client

## Project Structure

```
pharmarec/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application with all endpoints
│   │   ├── preprocessing.py     # Data preprocessing pipeline
│   │   ├── evaluation.py       # Evaluation metrics (stub)
│   │   └── recommenders/
│   │       ├── __init__.py
│   │       ├── content.py      # Content-Based Filtering (TF-IDF + Cosine)
│   │       ├── collaborative.py # Collaborative Filtering (SVD)
│   │       └── hybrid.py       # Hybrid Recommendation (Weighted Average)
│   ├── data/
│   │   └── drug_review.csv     # Raw dataset (from Kaggle)
│   │   └── cleaned_dataset.csv # Preprocessed dataset (generated)
│   ├── requirements.txt         # Python dependencies
│   └── README.md              # Backend documentation
│
└── frontend/
    ├── public/
    │   └── index.html          # HTML entry point
    ├── src/
    │   ├── components/
    │   │   ├── DrugSearch.tsx          # Search input with autocomplete
    │   │   ├── MethodSelector.tsx      # Dropdown for recommendation method
    │   │   ├── RecommendationTable.tsx # Results table
    │   │   └── MetricsComparisonTable.tsx # Metrics comparison
    │   ├── pages/
    │   │   └── Home.tsx                # Main page
    │   ├── App.tsx                     # Root component
    │   ├── main.tsx                    # Application entry
    │   └── index.css                   # Global styles
    ├── package.json
    ├── tsconfig.json
    ├── tsconfig.node.json
    ├── vite.config.ts
    └── README.md                      # Frontend documentation

└── README.md                          # This file
```

## Setup Instructions

### Prerequisites

- Python 3.12
- Node.js 18+ (for frontend)
- pip (Python package manager)
- npm (Node package manager)

### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Download and Prepare Dataset

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

```bash
# Configure Kaggle API (one-time setup)
mkdir -p ~/.kaggle
echo '{"username":"your_kaggle_username","key":"your_kaggle_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
cd backend
kaggle datasets download -d jessicali9530/kuc-hackathon-winter-2018

# Extract and rename the dataset file
# The downloaded file might be named differently
# Extract it and rename/move to backend/data/drug_review.csv
unzip kuc-hackathon-winter-2018.zip -d data/
# Check the extracted file name and rename if needed
# The CSV should have columns: drugName, condition, review, rating, usefulCount
mv data/drugsCom_train.csv data/drug_review.csv  # Example - adjust as needed

# Run preprocessing to create cleaned_dataset.csv
python -m app.preprocessing
```

**Note:** If the dataset file has a different name, adjust accordingly. The preprocessing expects a CSV with columns: drugName, condition, review, rating, usefulCount.

### 3. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install Node dependencies
npm install
```

## Running the Application

### Development Mode (Recommended)

**Terminal 1: Backend**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```
- Backend will be available at: http://localhost:8000
- Auto-reloads on code changes

**Terminal 2: Frontend**
```bash
cd frontend
npm run dev
```
- Frontend will be available at: http://localhost:3000
- Auto-reloads on code changes
- API requests are proxied to http://localhost:8000

Open http://localhost:3000 in your browser.

### Production Mode

```bash
# Backend
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (in separate terminal)
cd frontend
npm run build
npm run preview
```

## API Endpoints

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| GET | `/` | Health check | - |
| GET | `/health` | Detailed health check | - |
| GET | `/drugs` | Get drugs list (`?limit=100`, `?offset=0`) | - |
| GET | `/drugs/names` | Get unique drug names (`?limit=100`) | - |
| POST | `/recommend` | Get recommendations | `{"method": "content\|collaborative\|hybrid", "query": "..."}` |
| GET | `/metrics` | Get evaluation metrics | - |
| GET | `/metrics/comparison` | Get comparison table | - |
| POST | `/preprocess` | Run preprocessing pipeline | - |

### Recommend Request Example

```json
{
  "method": "content",
  "query": "Paracetamol"
}
```

### Recommend Response Example (Content-Based)

```json
{
  "recommendations": [
    {
      "drugName": "Aspirin",
      "condition": "Headache",
      "similarityScore": 0.8523
    },
    {
      "drugName": "Ibuprofen",
      "condition": "Back Pain",
      "similarityScore": 0.7845
    }
  ],
  "method": "content",
  "query": "Paracetamol",
  "status": "success"
}
```

### Collaborative Response Example

```json
{
  "recommendations": [
    {
      "drugName": "Aspirin",
      "predictedRating": 4.5
    },
    {
      "drugName": "Ibuprofen",
      "predictedRating": 4.2
    }
  ],
  "method": "collaborative",
  "query": "user123",
  "status": "success"
}
```

### Hybrid Response Example

```json
{
  "recommendations": [
    {
      "drugName": "Aspirin",
      "contentScore": 0.85,
      "collabScore": 4.5,
      "hybridScore": 0.78
    },
    {
      "drugName": "Ibuprofen",
      "contentScore": 0.78,
      "collabScore": 4.2,
      "hybridScore": 0.72
    }
  ],
  "method": "hybrid",
  "query": "Paracetamol",
  "status": "success"
}
```

## Features

### Recommendation Methods

1. **Content-Based Filtering**
   - **Algorithm**: TF-IDF + Cosine Similarity
   - **Input**: Drug name
   - **Output**: Similar drugs with similarity scores (0-1)
   - **Use Case**: Find drugs similar to a known drug

2. **Collaborative Filtering**
   - **Algorithm**: SVD (Singular Value Decomposition) from Surprise library
   - **Input**: User ID
   - **Output**: Predicted ratings for drugs (1-10 scale)
   - **Use Case**: Predict what rating a user would give to drugs they haven't rated

3. **Hybrid Recommendation**
   - **Algorithm**: Weighted average of normalized Content and Collaborative scores
   - **Default Weights**: 0.5 (Content) / 0.5 (Collaborative)
   - **Input**: Drug name or User ID
   - **Output**: Combined recommendations with Content, Collaborative, and Hybrid scores
   - **Normalization**: Content scores (0-1) and Collaborative ratings (normalized to 0-1)
   - **Use Case**: Best of both worlds - text similarity + user behavior

### Evaluation Metrics

| Metric | Description | Applies To |
|--------|-------------|------------|
| **Precision@10** | Fraction of recommended drugs that are positively rated (≥7.0) | Hybrid, Content-Based, Collaborative |
| **Recall@10** | Fraction of positively rated drugs that are recommended | Hybrid, Content-Based, Collaborative |
| **Coverage** | Proportion of unique drugs recommended from total catalog | All |
| **Execution Time** | Average time per recommendation in seconds | All |
| **RMSE** | Root Mean Squared Error | Collaborative |
| **MAE** | Mean Absolute Error | Collaborative |

### UI Features

- **Responsive Design**: Works on mobile, tablet, and desktop
- **Clean Card-based Layout**: White cards with subtle shadows
- **Drug Search**: Input with autocomplete for drug names
- **Method Selector**: Dropdown to choose recommendation algorithm
- **Recommendation Table**: Results displayed in clean, sortable table
- **Metrics Comparison Table**: Compare all three methods side-by-side
- **Sample Drugs**: Quick access buttons for common drugs
- **Loading States**: Animated spinners during async operations
- **Error Handling**: User-friendly error messages with guidance
- **Tailwind CSS**: Utility-first CSS framework via CDN

## Using the Application

### Step 1: Get Recommendations

1. Select a **Method** from the dropdown:
   - **Content-Based**: Find similar drugs (enter drug name)
   - **Collaborative**: Predict ratings for a user (enter user ID)
   - **Hybrid**: Combine both approaches (enter drug name or user ID)

2. Enter your **Search Query**:
   - For Content-Based/Hybrid: Enter a drug name (e.g., "Paracetamol", "Aspirin")
   - For Collaborative/Hybrid: Enter a user ID (e.g., "user123")

3. Click **Get Recommendations**

### Step 2: View Results

- Recommendations appear in a table below the search form
- Each recommendation shows:
  - **Content-Based**: Drug name, condition, similarity score
  - **Collaborative**: Drug name, predicted rating
  - **Hybrid**: Drug name, content score, collaborative score, hybrid score

### Step 3: Compare Metrics

- The **Model Comparison Metrics** section shows performance metrics
- Compare Precision@10, Recall@10, Coverage, and Execution Time
- Click **Refresh Metrics** to update the comparison

### Step 4: Explore Sample Drugs

- Click on any **Sample Drug** button at the bottom to quickly get recommendations
- This is useful for testing without typing

## Data Preprocessing

The preprocessing pipeline automatically runs on backend startup if `cleaned_dataset.csv` doesn't exist.

**Pipeline Steps:**
1. Load raw dataset from `backend/data/drug_review.csv`
2. Remove duplicate records
3. Handle missing values:
   - Drop rows with missing `drugName` or `review` (critical fields)
   - Fill missing `condition` with "Unknown"
   - Fill missing `rating` with column median
   - Fill missing `usefulCount` with 0
4. Lowercase all text
5. Remove punctuation from review text
6. Remove English stop words
7. Lemmatize review text (reduce words to base form)
8. Create `combined_text` column (condition + " " + review)
9. Save cleaned dataset to `backend/data/cleaned_dataset.csv`

**Manual Preprocessing:**
```bash
# Via API
curl -X POST http://localhost:8000/preprocess

# Via Python
cd backend
python -m app.preprocessing
```

## Project Implementation Details

### Backend Implementation

**main.py** - FastAPI Application:
- Global state for recommender instances (loaded on startup)
- Lifespan management for initialization
- CORS enabled for frontend integration
- All API endpoints implemented

**preprocessing.py** - Data Pipeline:
- Automatic NLTK data download (punkt, stopwords, wordnet, omw-1.4)
- Modular functions for each preprocessing step
- Type hints throughout
- Error handling for missing data

**recommenders/content.py** - Content-Based:
- `ContentRecommender` class with `fit()` and `recommend()`
- TF-IDF vectorization of `combined_text`
- Cosine similarity matrix computation
- Returns (drug_name, condition, similarity_score) tuples

**recommenders/collaborative.py** - Collaborative:
- `CollaborativeRecommender` class with `train()`, `predict()`, `recommend()`, `evaluate()`
- SVD algorithm from Surprise library
- Automatic user ID generation if not in dataset
- RMSE and MAE evaluation on test set
- Returns (drug_name, predicted_rating) tuples

**recommenders/hybrid.py** - Hybrid:
- `HybridRecommender` class with `recommend()`, `evaluate()`, `get_comparison_table()`
- Normalizes both content (0-1) and collaborative scores (to 0-1)
- Weighted average with configurable weights
- Precision@10, Recall@10, Coverage, Execution Time metrics
- Comparison table generation

### Frontend Implementation

**App.tsx** - Root Component:
- Health check on startup
- Error boundary for connection issues
- Helpful setup instructions if backend not running

**Home.tsx** - Main Page:
- State management for drugs, recommendations, metrics
- Data fetching with useEffect and useCallback
- Form handling for search
- Responsive grid layout

**Components:**
- **DrugSearch**: Controlled input with autocomplete suggestions
- **MethodSelector**: Dropdown with descriptions
- **RecommendationTable**: Dynamic columns based on method
- **MetricsComparisonTable**: Formatted comparison display

## Troubleshooting

### Backend not starting

**Symptom:** `ModuleNotFoundError` when running `uvicorn`

**Solution:**
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

**Symptom:** Cleaned dataset not found

**Solution:**
```bash
# Run preprocessing first
python -m app.preprocessing
# Or via API
curl -X POST http://localhost:8000/preprocess
```

**Symptom:** Kaggle download fails

**Solution:**
- Ensure `~/.kaggle/kaggle.json` exists with valid credentials
- Check Kaggle API token is still valid
- Manually download dataset from Kaggle and place in `backend/data/drug_review.csv`

### Frontend not starting

**Symptom:** `npm ERR!` during install

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**Symptom:** Blank page or errors in browser console

**Solution:**
- Check browser console (F12) for error messages
- Ensure backend is running: `curl http://localhost:8000/health`
- Verify frontend can reach backend: `curl http://localhost:8000/drugs?limit=5`

### API connection errors

**Symptom:** CORS errors or connection refused

**Solution:**
- Backend must be running: `uvicorn app.main:app --reload`
- Frontend dev server automatically proxies `/api` to `http://localhost:8000`
- If running frontend separately, update `API_BASE` in `src/App.tsx` and `src/pages/Home.tsx`

### Dataset column mismatch

**Symptom:** `ValueError: Missing required columns`

**Solution:**
- Ensure your dataset CSV has columns: drugName, condition, review, rating, usefulCount
- If column names differ, rename them or modify the preprocessing.py to match

## Educational Disclaimer

**This project is intended for educational and research purposes only.**

The recommendations generated by PharmaRec are based on historical user reviews and ratings from a public dataset. They should **NOT** be considered medical advice.

Always consult with a qualified healthcare professional before making any decisions about medication or treatment.

The dataset used contains anonymous user reviews and does not represent real patient data for medical decision-making.

## License

MIT License - Feel free to use this project for educational purposes.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Open a Pull Request

## Acknowledgments

- **Dataset**: [KUC Hackathon Winter 2018](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) on Kaggle
- **Backend Libraries**: FastAPI, Scikit-learn, Surprise, Pandas, NumPy, NLTK
- **Frontend Libraries**: React, TypeScript, Vite, Axios, Tailwind CSS

## Contact

For questions or issues, please refer to the documentation or open an issue in the repository.
