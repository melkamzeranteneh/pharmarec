# PharmaRec - Implementation Plan

## Overview

PharmaRec is a healthcare recommendation system that compares three recommendation algorithms:

1. Content-Based Filtering
2. Collaborative Filtering
3. Hybrid Recommendation

The objective is to compare their performance on the same healthcare dataset and evaluate them using standard recommender system metrics.

---

# Tech Stack

## Backend

- Python 3.12
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- Surprise
- NLTK

## Frontend

- React
- TypeScript
- Vite
- Axios

---

# Dataset

Drug Review Dataset

Columns

- drugName
- condition
- review
- rating
- usefulCount

---

# Project Structure

pharmarec/

    backend/
        app/
            api/
            recommenders/
                content.py
                collaborative.py
                hybrid.py
            preprocessing.py
            evaluation.py
            main.py

        data/

    frontend/

    README.md

---

# Phase 1

## Data Processing

Tasks

- Load dataset
- Remove duplicates
- Handle missing values
- Clean review text
- Remove stop words
- Lemmatize
- Create combined_text

Deliverable

cleaned_dataset.csv

---

# Phase 2

## Content-Based Filtering

Algorithm

TF-IDF

Similarity

Cosine Similarity

Output

Top N similar drugs

Deliverable

ContentRecommender

---

# Phase 3

## Collaborative Filtering

Algorithm

SVD (Surprise)

Input

User-Drug-Rating matrix

Output

Top N recommended drugs

Deliverable

CollaborativeRecommender

---

# Phase 4

## Hybrid Recommendation

Combine

Content Score

Collaborative Score

Weighted Average

Default

50%

50%

Deliverable

HybridRecommender

---

# Phase 5

## Evaluation

Metrics

- Precision@K
- Recall@K
- RMSE
- MAE
- Coverage
- Execution Time

Deliverable

Comparison Table

---

# Phase 6

## Backend API

Endpoints

GET /drugs

POST /recommend

GET /metrics

Deliverable

Working REST API

---

# Phase 7

## React Frontend

Pages

Home

Components

- Drug Search
- Method Selector
- Recommendation Table
- Metrics Table

Deliverable

Fully working UI

---

# Final Deliverables

✓ Content-Based Filtering

✓ Collaborative Filtering

✓ Hybrid Filtering

✓ Evaluation

✓ React Frontend

✓ FastAPI Backend

✓ Documentation
