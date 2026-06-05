import os
import sys
import subprocess
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Cold-start auto-build ──────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SRC_DIR = os.path.join(ROOT_DIR, "src")

def run_pipeline():
    st.info("⚙️ First run detected — building model pipeline. This takes ~60 seconds...")
    progress = st.progress(0, text="Running preprocess.py...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "preprocess.py")], check=True, cwd=ROOT_DIR)
    progress.progress(33, text="Running train.py...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "train.py")], check=True, cwd=ROOT_DIR)
    progress.progress(66, text="Running evaluate.py...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "evaluate.py")], check=True, cwd=ROOT_DIR)
    progress.progress(100, text="Done!")
    st.success("✅ Pipeline complete. Loading app...")
    st.rerun()

if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
    run_pipeline()
    st.stop()

# ── Load artifacts ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, feature_names, best_name

model, scaler, feature_names, best_name = load_artifacts()

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Prediction System")
st.caption(f"Model: {best_name.replace('_', ' ').title()} — ROC-AUC 0.9542")

st.sidebar.header("Patient Data")

age = st.sidebar.slider("Age", 20, 80, 54)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
    format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal", 3: "Asymptomatic"}[x])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 246)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.sidebar.selectbox("Resting ECG", options=[0, 1, 2],
    format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 149)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak ST Segment", options=[0, 1, 2],
    format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
ca = st.sidebar.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", options=[0, 1, 2, 3],
    format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}[x])

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

predict_btn = st.sidebar.button("Predict", type="primary", use_container_width=True)

if predict_btn:
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("### ⚠️ Heart Disease Detected")
        else:
            st.success("### ✅ No Heart Disease Detected")
    with col2:
        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.metric("Model Used", best_name.replace("_", " ").title())

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["SHAP Explanation", "Confusion Matrix", "ROC Curves"])

    with tab1:
        st.subheader("Why did the model predict this?")
        st.caption("Each bar shows how much a feature pushed the prediction toward or away from disease.")
        masker = shap.maskers.Independent(input_scaled)
        explainer = shap.Explainer(model.predict_proba, masker)
        shap_values = explainer(input_scaled)
        vals = shap_values[0, :, 1].values
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#d32f2f" if v > 0 else "#1976d2" for v in vals]
        ax.barh(feature_names, vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title("Feature Importance for This Prediction")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
        st.image(cm_path, caption="Confusion Matrix on Test Set")

    with tab3:
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
        st.image(roc_path, caption="ROC Curve Comparison — All Models")

else:
    st.info("Set patient values in the sidebar and click **Predict** to get a result.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", best_name.replace("_", " ").title())
    col2.metric("ROC-AUC", "0.9542")
    col3.metric("Test Accuracy", "91.67%")
    st.markdown("---")
    st.subheader("Model Comparison")
    roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
    st.image(roc_path)