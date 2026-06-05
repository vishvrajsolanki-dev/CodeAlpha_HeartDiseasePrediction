import os
import json
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

X_train = joblib.load(os.path.join(MODELS_DIR, "X_train.pkl"))
y_train = joblib.load(os.path.join(MODELS_DIR, "y_train.pkl"))
X_test  = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
y_test  = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))

models = {
    "svm":                SVC(probability=True, class_weight="balanced", random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest":      RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "xgboost":            XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    results[name] = {"accuracy": acc, "f1": f1, "roc_auc": auc, "model": model}
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
    print(f"{name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

best_name  = max(results, key=lambda x: results[x]["roc_auc"])
best_model = results[best_name]["model"]

joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
joblib.dump(best_name,  os.path.join(MODELS_DIR, "best_model_name.pkl"))

with open(os.path.join(MODELS_DIR, "best_metrics.json"), "w") as f:
    json.dump({
        "accuracy": results[best_name]["accuracy"],
        "f1":       results[best_name]["f1"],
        "roc_auc":  results[best_name]["roc_auc"],
    }, f)

print(f"\nBest: {best_name} (AUC: {results[best_name]['roc_auc']:.4f})")