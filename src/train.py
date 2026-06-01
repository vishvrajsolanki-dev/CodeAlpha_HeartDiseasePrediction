import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

X_train = joblib.load("models/X_train.pkl")
y_train = joblib.load("models/y_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

models = {
    "svm": SVC(probability=True, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    results[name] = {"accuracy": acc, "f1": f1, "roc_auc": auc, "model": model}
    joblib.dump(model, f"models/{name}.pkl")
    print(f"{name:25s} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

best_name = max(results, key=lambda x: results[x]["roc_auc"])
best_model = results[best_name]["model"]

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(best_name, "models/best_model_name.pkl")

print(f"\nBest model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.4f})")
print("Saved to models/best_model.pkl")