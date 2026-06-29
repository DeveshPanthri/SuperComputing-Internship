# 🎓 AI Personalized Learning Agent

An AI agent that designs a personalized learning journey for every individual student.

## Problem Statement

In colleges and universities, every student learns at a different pace and has different weak areas. Identifying gaps manually is time-consuming and impossible at scale. Students end up studying the same material regardless of what they already know — wasting time on strong topics while weak areas remain unaddressed.

## Solution

This AI agent:
- **Analyzes** quiz results, attendance patterns, assignment scores, and performance logs
- **Identifies** weak knowledge concepts using a Deep Knowledge Tracing (DKT) model
- **Generates** a personalized 7-day study plan
- **Recommends** specific YouTube videos, PDFs, and practice problems
- **Reports** weekly progress — all automatically without teacher involvement

## Architecture

```
Student Data → EDA Pipeline → Feature Engineering → DKT (LSTM) → Gap Detector (XGBoost)
                                                                        ↓
                              Streamlit UI ← LangChain Agent ← Resource Recommender (SVD)
                                                    ↓
                                          Study Planner + Progress Reporter
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch (DKT/LSTM) |
| ML | Scikit-learn, XGBoost |
| NLP | HuggingFace (TF-IDF / BERT) |
| Recommender | Surprise (Collaborative Filtering / SVD) |
| Agent | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| Forecasting | Prophet |

## Datasets

| Dataset | Size | Type |
|---------|------|------|
| Open University Learning Analytics (OULA) | ~32,000 students | Tabular + Time-series |
| UCI Student Performance | 649 students, 33 features | Mixed |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run EDA notebooks
jupyter notebook notebooks/

# Run Streamlit dashboard
streamlit run app/streamlit_app.py

# Run FastAPI backend
uvicorn api.main:app --reload
```

## Project Structure

```
project/
├── data/raw/              # Raw downloaded datasets
├── data/processed/        # Cleaned, feature-engineered data
├── notebooks/             # EDA & experiment notebooks (Steps 1-8)
├── src/
│   ├── knowledge_tracer/  # DKT LSTM model
│   ├── gap_detector/      # XGBoost classifier
│   ├── recommender/       # Collaborative filtering (SVD)
│   ├── study_planner/     # Rule-based + LLM planner
│   ├── progress_reporter/ # Weekly report generator
│   └── agent/             # LangChain agent integration
├── app/                   # Streamlit UI
├── api/                   # FastAPI endpoints
├── models/                # Saved model weights
├── reports/               # Generated student reports
├── requirements.txt
└── README.md
```

## Evaluation Metrics

| Component | Metric | Benchmark |
|-----------|--------|-----------|
| DKT Model | AUC-ROC | > 0.75 |
| Gap Detector | F1-score | High |
| Recommender | NDCG@5, Precision@3 | Competitive |
