import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import os

os.makedirs("outputs", exist_ok=True)

X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")
feature_names = joblib.load("models/feature_names.pkl")
best_model = joblib.load("models/best_model.pkl")
best_name = joblib.load("models/best_model_name.pkl")

models = {
    "svm": joblib.load("models/svm.pkl"),
    "logistic_regression": joblib.load("models/logistic_regression.pkl"),
    "random_forest": joblib.load("models/random_forest.pkl"),
    "xgboost": joblib.load("models/xgboost.pkl")
}

print(f"Best model: {best_name}\n")
print(f"{'Model':25s} | Accuracy | F1     | ROC-AUC")
print("-" * 55)

for name, model in models.items():
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"{name:25s} | {acc:.4f}   | {f1:.4f} | {auc:.4f}")

# --- confusion matrix ---
preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved: outputs/confusion_matrix.png")

# --- ROC curve for all models ---
plt.figure(figsize=(8, 6))
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150)
plt.close()
print("Saved: outputs/roc_curves.png")

# --- SHAP summary plot ---
# KernelExplainer works with any model — it treats the model as a black box
# and estimates SHAP values by sampling. masker sets the background baseline.
masker = shap.maskers.Independent(X_test)
explainer = shap.Explainer(best_model.predict_proba, masker)
shap_values = explainer(X_test)

# shap_values[..., 1] selects SHAP values for the "Disease" class (class 1)
shap.summary_plot(
    shap_values[..., 1],
    X_test,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/shap_summary.png")
print("\nEvaluation complete.")