import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess():
    df = pd.read_csv("data/heart.csv")

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

    os.makedirs("models", exist_ok=True)

    joblib.dump(X_train, "models/X_train.pkl")
    joblib.dump(y_train, "models/y_train.pkl")
    joblib.dump(X_test, "models/X_test.pkl")
    joblib.dump(y_test, "models/y_test.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    print("\nAll files saved to models/")

if __name__ == "__main__":
    preprocess()