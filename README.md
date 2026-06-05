# Heart Disease Prediction System

A machine learning web app that predicts cardiovascular disease risk from 13 clinical features using a 4-model pipeline with SHAP explainability — deployed on Streamlit Cloud.

**Live Demo:** [codealphaheartdiseaseprediction-ypddp926uffnkst6sagsnr.streamlit.app](https://codealphaheartdiseaseprediction-ypddp926uffnkst6sagsnr.streamlit.app/)

---

## Features

- Four classifiers trained and compared: SVM, Logistic Regression, Random Forest, XGBoost
- Best model auto-selected by ROC-AUC on a held-out 20% test set
- Class imbalance handled via `class_weight="balanced"` (no external dependencies)
- SHAP KernelExplainer for per-prediction feature attribution
- Clinical minimalist Streamlit UI — sidebar input form, result card, tabbed analysis panel
- Confidence score, SHAP bar chart, confusion matrix, and ROC curves for all models
- Cold-start pipeline: models built automatically on first run, no manual setup needed

## Tech Stack

Python · Streamlit · scikit-learn · XGBoost · SHAP · pandas · matplotlib · seaborn · joblib

## Dataset

UCI Cleveland Heart Disease Dataset — 303 patients, 13 clinical features, binary classification (disease / no disease)

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Random Forest** ✓ | 86.67% | 0.8400 | **0.9637** |
| Logistic Regression | 91.67% | 0.9020 | 0.9542 |
| XGBoost | 85.00% | 0.8302 | 0.9442 |
| SVM | 90.00% | 0.8800 | 0.9431 |

Best model selected by ROC-AUC. Random Forest wins on AUC despite lower raw accuracy — better at ranking risk, which matters more in clinical screening.

## Project Structure

```
├── app/
│   └── app.py              # Streamlit app (cold-start + UI + SHAP)
├── src/
│   ├── preprocess.py       # Load, split, scale, save artifacts
│   ├── train.py            # Train 4 models, export best_metrics.json
│   └── evaluate.py         # Confusion matrix, ROC curves, SHAP summary
├── data/
│   └── heart.csv           # UCI Cleveland dataset
├── models/                 # Generated at runtime (gitignored)
├── outputs/                # PNG charts generated at runtime (gitignored)
└── requirements.txt
```

## Local Setup

```bash
git clone https://github.com/vishvrajsolanki-dev/CodeAlpha_HeartDiseasePrediction.git
cd CodeAlpha_HeartDiseasePrediction
pip install -r requirements.txt
streamlit run app/app.py
```

The app auto-runs the full pipeline (preprocess → train → evaluate) on first launch. No manual script execution needed.

## Internship

CodeAlpha Machine Learning Internship · Task 4 · June 2026

## Author

Vishvrajsinh Solanki · B.Tech AI & Data Science, ADIT (2025–2029)