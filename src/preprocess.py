import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def preprocess():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "heart.csv"))

    if "condition" in df.columns:
        df.rename(columns={"condition": "target"}, inplace=True)

    df["target"] = (df["target"] > 0).astype(int)

    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Class distribution:\n{df['target'].value_counts()}")

    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"\nTrain size after SMOTE: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train)}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(X_train, os.path.join(MODELS_DIR, "X_train.pkl"))
    joblib.dump(y_train, os.path.join(MODELS_DIR, "y_train.pkl"))
    joblib.dump(X_test, os.path.join(MODELS_DIR, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(MODELS_DIR, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    print("\nAll files saved to models/")

if __name__ == "__main__":
    preprocess()