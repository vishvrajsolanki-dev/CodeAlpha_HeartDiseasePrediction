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

# ── Paths ──────────────────────────────────────────────────────────────────
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(APP_DIR)
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SRC_DIR     = os.path.join(ROOT_DIR, "src")

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark medical theme */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2a45;
}
[data-testid="stSidebar"] * {
    color: #c8cfe0 !important;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f1629 0%, #1a1f3a 50%, #0f1629 100%);
    border: 1px solid #1e2a45;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(220,50,50,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.9rem;
    color: #6b7a99;
    margin: 0;
    font-family: 'DM Mono', monospace;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: #0f1629;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e05c5c;
    font-family: 'DM Mono', monospace;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Result cards */
.result-positive {
    background: linear-gradient(135deg, #2d0a0a, #1a0f0f);
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-negative {
    background: linear-gradient(135deg, #0a2d15, #0f1a12);
    border: 1px solid #14532d;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
}
.result-conf {
    font-size: 0.85rem;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
}

/* Section headers */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b7a99;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2a45;
}

/* Sidebar section label */
.sidebar-label {
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a5568;
    margin: 1.2rem 0 0.5rem 0;
}

/* Tab styling */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #6b7a99 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e05c5c !important;
    border-bottom-color: #e05c5c !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(220,38,38,0.3) !important;
}

/* Sliders accent */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #6b7a99 !important;
}

/* Info box */
.stAlert {
    background: #0f1629 !important;
    border: 1px solid #1e2a45 !important;
    border-radius: 10px !important;
    color: #9ca3af !important;
}

/* Plot background */
.stPlotlyChart, [data-testid="stImage"] {
    background: transparent !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Cold-start auto-build ──────────────────────────────────────────────────
def run_pipeline():
    st.markdown("""
    <div style="background:#0f1629;border:1px solid #1e2a45;border-radius:12px;
    padding:2rem;text-align:center;margin:2rem 0;">
        <div style="font-size:2rem;margin-bottom:0.5rem;">⚙️</div>
        <div style="font-size:1.1rem;font-weight:600;color:#e8eaf0;margin-bottom:0.3rem;">
            First Launch — Building Pipeline
        </div>
        <div style="font-size:0.85rem;color:#6b7a99;">
            Training 4 models on the UCI Cleveland dataset. Takes ~60 seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)
    progress = st.progress(0, text="Preprocessing data...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "preprocess.py")], check=True, cwd=ROOT_DIR)
    progress.progress(33, text="Training SVM, Logistic Regression, Random Forest, XGBoost...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "train.py")], check=True, cwd=ROOT_DIR)
    progress.progress(66, text="Generating evaluation charts...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "evaluate.py")], check=True, cwd=ROOT_DIR)
    progress.progress(100, text="Done!")
    st.success("Pipeline complete. Loading app...")
    st.rerun()

if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
    run_pipeline()
    st.stop()

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_name     = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, feature_names, best_name

model, scaler, feature_names, best_name = load_artifacts()
model_label = best_name.replace("_", " ").title()

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-title">🫀 Heart Disease Risk Predictor</div>
    <div class="hero-sub">CodeAlpha ML Internship · Task 4 · Model: {model_label} · UCI Cleveland Dataset</div>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card"><div class="val">91.67%</div><div class="lbl">Test Accuracy</div></div>
    <div class="metric-card"><div class="val">0.9542</div><div class="lbl">ROC-AUC Score</div></div>
    <div class="metric-card"><div class="val">0.9020</div><div class="lbl">F1 Score</div></div>
    <div class="metric-card"><div class="val">4</div><div class="lbl">Models Compared</div></div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0;">
        <div style="font-size:1rem;font-weight:600;color:#e8eaf0;">Patient Input</div>
        <div style="font-size:0.75rem;color:#6b7a99;margin-top:0.2rem;">
            Adjust values and click Predict
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Demographics</div>', unsafe_allow_html=True)
    age = st.slider("Age", 20, 80, 54)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    st.markdown('<div class="sidebar-label">Cardiac Symptoms</div>', unsafe_allow_html=True)
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
        format_func=lambda x: {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-Anginal", 3:"Asymptomatic"}[x])
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 149)
    exang   = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope   = st.selectbox("ST Slope", [0, 1, 2],
        format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])

    st.markdown('<div class="sidebar-label">Lab Results</div>', unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 246)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    restecg  = st.selectbox("Resting ECG", [0, 1, 2],
        format_func=lambda x: {0:"Normal", 1:"ST-T Abnormality", 2:"LV Hypertrophy"}[x])

    st.markdown('<div class="sidebar-label">Advanced</div>', unsafe_allow_html=True)
    ca   = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
        format_func=lambda x: {0:"Normal", 1:"Fixed Defect", 2:"Reversible Defect", 3:"Unknown"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

# ── Main content ───────────────────────────────────────────────────────────
input_data   = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

if predict_btn:
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-title" style="color:#f87171;">⚠️ Elevated Heart Disease Risk Detected</div>
            <div class="result-conf">Confidence: {confidence:.1f}% · Model: {model_label}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-title" style="color:#4ade80;">✅ Low Heart Disease Risk</div>
            <div class="result-conf">Confidence: {confidence:.1f}% · Model: {model_label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔍 SHAP Feature Importance", "📊 Confusion Matrix", "📈 ROC Curves"])

    with tab1:
        st.caption("Which features drove this prediction — red pushes toward disease, blue away from it.")
        with st.spinner("Computing SHAP values..."):
            masker      = shap.maskers.Independent(input_scaled)
            explainer   = shap.Explainer(model.predict_proba, masker)
            shap_values = explainer(input_scaled)
            vals        = shap_values[0, :, 1].values

        # Sort by absolute impact
        sorted_idx = np.argsort(np.abs(vals))
        sorted_vals   = vals[sorted_idx]
        sorted_feats  = [feature_names[i] for i in sorted_idx]
        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#0f1629")
        bars = ax.barh(sorted_feats, sorted_vals, color=colors, height=0.6, edgecolor="none")
        ax.axvline(0, color="#374151", linewidth=1)
        ax.set_xlabel("SHAP Value", color="#6b7a99", fontsize=9)
        ax.set_title("Feature Impact on This Prediction", color="#e8eaf0", fontsize=11, pad=12)
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.spines[:].set_color("#1e2a45")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(cm_path, caption=f"Confusion Matrix — {model_label}", use_container_width=True)
        else:
            st.warning("Confusion matrix not found.")

    with tab3:
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            col1, col2, col3 = st.columns([0.5, 3, 0.5])
            with col2:
                st.image(roc_path, caption="ROC Curve — All 4 Models", use_container_width=True)
        else:
            st.warning("ROC curves not found.")

else:
    # Landing state
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve Comparison — All 4 Models", use_container_width=True)
    with col2:
        st.markdown("""
        <div style="padding:1rem;">
        <div style="font-size:0.8rem;color:#6b7a99;margin-bottom:1rem;line-height:1.6;">
        This app uses a <strong style="color:#e8eaf0;">4-model ML pipeline</strong> trained on the 
        UCI Cleveland Heart Disease dataset to predict cardiovascular risk from 13 clinical features.
        </div>
        <div style="margin-bottom:0.8rem;">
            <div style="font-size:0.7rem;color:#6b7a99;text-transform:uppercase;letter-spacing:0.08em;">Models Trained</div>
            <div style="font-size:0.85rem;color:#e8eaf0;margin-top:0.3rem;">
                SVM · Logistic Regression<br>Random Forest · XGBoost
            </div>
        </div>
        <div style="margin-bottom:0.8rem;">
            <div style="font-size:0.7rem;color:#6b7a99;text-transform:uppercase;letter-spacing:0.08em;">Selection Criteria</div>
            <div style="font-size:0.85rem;color:#e8eaf0;margin-top:0.3rem;">Best ROC-AUC on held-out test set</div>
        </div>
        <div style="margin-bottom:0.8rem;">
            <div style="font-size:0.7rem;color:#6b7a99;text-transform:uppercase;letter-spacing:0.08em;">Explainability</div>
            <div style="font-size:0.85rem;color:#e8eaf0;margin-top:0.3rem;">Per-prediction SHAP feature importance</div>
        </div>
        <div style="margin-bottom:0.8rem;">
            <div style="font-size:0.7rem;color:#6b7a99;text-transform:uppercase;letter-spacing:0.08em;">Data Integrity</div>
            <div style="font-size:0.85rem;color:#e8eaf0;margin-top:0.3rem;">No leakage — scaler fit on train only</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0f1629;border:1px solid #1e2a45;border-radius:10px;
    padding:1rem 1.5rem;margin-top:1rem;font-size:0.85rem;color:#6b7a99;text-align:center;">
        👈 Set patient values in the sidebar and click <strong style="color:#e8eaf0;">Run Prediction</strong>
    </div>
    """, unsafe_allow_html=True)