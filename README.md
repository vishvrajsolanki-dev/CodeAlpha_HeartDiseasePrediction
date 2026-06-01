# Heart Disease Prediction System

A machine learning web app that predicts heart disease risk from clinical patient data using four ML models with SHAP explainability.

## Features

- Multi-model pipeline: SVM, Logistic Regression, Random Forest, XGBoost
- Best model auto-selected by ROC-AUC score
- SMOTE balancing to handle class imbalance
- SHAP values for per-prediction explainability
- Interactive Streamlit web app with patient input form
- Confidence score, SHAP bar chart, confusion matrix, and ROC curve

## Tech Stack

Python · Streamlit · scikit-learn · XGBoost · imbalanced-learn · SHAP · pandas · matplotlib · seaborn

## Dataset

UCI Cleveland Heart Disease Dataset — 303 patients, 13 features, binary classification (disease / no disease)

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 91.67% | 0.9020 | 0.9542 |
| SVM | 90.00% | 0.8800 | 0.9408 |
| XGBoost | 86.67% | 0.8462 | 0.9542 |
| Random Forest | 85.00% | 0.8235 | 0.9408 |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

## Usage

1. Run preprocessing:
```bash
python src/preprocess.py
```

2. Train models:
```bash
python src/train.py
```

3. Generate evaluation charts:
```bash
python src/evaluate.py
```

4. Launch the app:
```bash
streamlit run app/app.py
```

## Live Demo

[Streamlit Cloud link — add after deployment]

## Author

Vishvrajsinh Solanki · B.Tech AI & Data Science, ADIT (2025–2029)