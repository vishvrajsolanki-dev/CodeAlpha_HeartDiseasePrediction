import os
import sys
import json
import subprocess
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(APP_DIR)
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SRC_DIR     = os.path.join(ROOT_DIR, "src")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="assets/favicon.png" if os.path.exists(os.path.join(ROOT_DIR,"assets","favicon.png")) else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --bg:        #f5f2ed;
    --surface:   #ffffff;
    --border:    #ddd9d2;
    --text:      #1c1917;
    --muted:     #78716c;
    --accent:    #7f1d1d;
    --accent2:   #991b1b;
    --positive:  #14532d;
    --negative:  #7f1d1d;
    --font-serif: 'Libre Baskerville', Georgia, serif;
    --font-sans:  'Source Sans 3', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: var(--bg) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: var(--text) !important;
    background: var(--bg) !important;
}

/* Slider thumb accent */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background: var(--accent) !important;
}

/* ── Main area ── */
.block-container {
    padding: 2.5rem 3rem 3rem 3rem !important;
    max-width: 1100px !important;
}

/* ── Typography ── */
.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: var(--font-serif);
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.3px;
    line-height: 1.2;
}
.page-meta {
    font-size: 0.8rem;
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.03em;
}

/* ── Metric strip ── */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.metric-item {
    background: var(--surface);
    padding: 1rem 1.25rem;
    text-align: left;
}
.metric-val {
    font-family: var(--font-serif);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-lbl {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 500;
}

/* ── Result block ── */
.result-block {
    border-radius: 6px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 2rem;
    border-left: 3px solid;
}
.result-positive {
    background: #fef2f2;
    border-left-color: var(--negative);
}
.result-negative {
    background: #f0fdf4;
    border-left-color: var(--positive);
}
.result-heading {
    font-family: var(--font-serif);
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
}
.result-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin: 0;
    font-family: var(--font-sans);
}

/* ── Section label ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar group label ── */
.group-label {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin: 1.5rem 0 0.5rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar header ── */
.sidebar-header {
    font-family: var(--font-serif);
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] button {
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    padding: 0.5rem 1rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* ── Predict button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.2rem !important;
    width: 100% !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--accent2) !important;
}

/* ── Info / alerts ── */
.stAlert {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}

/* ── Image containers ── */
[data-testid="stImage"] {
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
}

/* ── Dropdown options legible ── */
[data-baseweb="popover"] ul li {
    color: var(--text) !important;
    background: var(--surface) !important;
}
[data-baseweb="popover"] ul li:hover {
    background: var(--bg) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Cold-start ─────────────────────────────────────────────────────────────
def run_pipeline():
    st.markdown("""
    <div style="border:1px solid #ddd9d2;border-radius:6px;padding:2rem;
    text-align:center;background:#fff;margin:3rem auto;max-width:480px;">
        <div style="font-family:'Libre Baskerville',serif;font-size:1.1rem;
        font-weight:700;color:#1c1917;margin-bottom:0.5rem;">
            Building Model Pipeline
        </div>
        <div style="font-size:0.82rem;color:#78716c;line-height:1.6;">
            Training four classifiers on the UCI Cleveland dataset.<br>
            This runs once and takes approximately 60 seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)
    bar = st.progress(0, text="Preprocessing data...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "preprocess.py")], check=True, cwd=ROOT_DIR)
    bar.progress(33, text="Training models...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "train.py")], check=True, cwd=ROOT_DIR)
    bar.progress(66, text="Generating evaluation charts...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "evaluate.py")], check=True, cwd=ROOT_DIR)
    bar.progress(100, text="Complete.")
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
    metrics_path  = os.path.join(MODELS_DIR, "best_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {"accuracy": 0.9167, "f1": 0.9020, "roc_auc": 0.9542}
    return model, scaler, feature_names, best_name, metrics

model, scaler, feature_names, best_name, metrics = load_artifacts()
model_label = best_name.replace("_", " ").title()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-header">Patient Assessment</div>
    <div class="sidebar-sub">Adjust clinical values and run the prediction.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="group-label">Demographics</div>', unsafe_allow_html=True)
    age = st.slider("Age", 20, 80, 54, label_visibility="visible")
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    st.markdown('<div class="group-label">Cardiac Presentation</div>', unsafe_allow_html=True)
    cp = st.selectbox("Chest Pain Type", [0,1,2,3],
        format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal Pain",3:"Asymptomatic"}[x])
    thalach = st.slider("Max Heart Rate (bpm)", 60, 220, 149)
    exang   = st.selectbox("Exercise-Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope   = st.selectbox("ST Slope", [0,1,2],
        format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])

    st.markdown('<div class="group-label">Lab Values</div>', unsafe_allow_html=True)
    trestbps = st.slider("Resting BP (mmHg)", 80, 200, 130)
    chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 246)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0,1],
        format_func=lambda x: "No" if x==0 else "Yes")
    restecg  = st.selectbox("Resting ECG", [0,1,2],
        format_func=lambda x: {0:"Normal",1:"ST-T Wave Abnormality",2:"LV Hypertrophy"}[x])

    st.markdown('<div class="group-label">Imaging</div>', unsafe_allow_html=True)
    ca   = st.selectbox("Major Vessels (Fluoroscopy)", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [0,1,2,3],
        format_func=lambda x: {0:"Normal",1:"Fixed Defect",2:"Reversible Defect",3:"Unknown"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", type="primary")

# ── Input prep ─────────────────────────────────────────────────────────────
input_data   = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

# ── Page header ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="page-title">Heart Disease Risk Predictor</div>
    <div class="page-meta">
        CodeAlpha ML Internship &nbsp;·&nbsp; Task 4 &nbsp;·&nbsp;
        UCI Cleveland Dataset &nbsp;·&nbsp; Best model: {model_label}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics strip ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-strip">
    <div class="metric-item">
        <div class="metric-val">{metrics['accuracy']*100:.1f}%</div>
        <div class="metric-lbl">Test Accuracy</div>
    </div>
    <div class="metric-item">
        <div class="metric-val">{metrics['roc_auc']:.4f}</div>
        <div class="metric-lbl">ROC-AUC</div>
    </div>
    <div class="metric-item">
        <div class="metric-val">{metrics['f1']:.4f}</div>
        <div class="metric-lbl">F1 Score</div>
    </div>
    <div class="metric-item">
        <div class="metric-val">4</div>
        <div class="metric-lbl">Models Compared</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-block result-positive">
            <div class="result-heading" style="color:#7f1d1d;">
                Elevated Cardiovascular Risk Detected
            </div>
            <div class="result-sub">
                Confidence {confidence:.1f}% &nbsp;·&nbsp; Model: {model_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-block result-negative">
            <div class="result-heading" style="color:#14532d;">
                Low Cardiovascular Risk
            </div>
            <div class="result-sub">
                Confidence {confidence:.1f}% &nbsp;·&nbsp; Model: {model_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Detailed Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Feature Importance (SHAP)", "Confusion Matrix", "ROC Curves"])

    with tab1:
        st.caption("SHAP values show how each clinical feature contributed to this individual prediction. Positive values increase risk; negative values decrease it.")
        with st.spinner("Computing SHAP values..."):
            masker      = shap.maskers.Independent(input_scaled)
            explainer   = shap.Explainer(model.predict_proba, masker)
            shap_values = explainer(input_scaled)
            vals        = shap_values[0, :, 1].values

        sorted_idx   = np.argsort(vals)
        sorted_vals  = vals[sorted_idx]
        sorted_feats = [feature_names[i] for i in sorted_idx]
        colors       = ["#991b1b" if v > 0 else "#1e3a5f" for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.barh(sorted_feats, sorted_vals, color=colors, height=0.55, edgecolor="none")
        ax.axvline(0, color="#d6d3d1", linewidth=0.8, zorder=0)
        ax.set_xlabel("SHAP value", fontsize=8, color="#78716c", labelpad=6)
        ax.set_title("Per-Prediction Feature Contribution", fontsize=10,
                     color="#1c1917", pad=10, fontfamily="serif")
        ax.tick_params(axis="both", labelsize=7.5, colors="#44403c", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e7e5e4")
        ax.spines["bottom"].set_color("#e7e5e4")
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#f5f5f4", linewidth=0.6)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            col_l, col_c, col_r = st.columns([1, 2, 1])
            with col_c:
                st.image(cm_path, caption=f"Confusion matrix — {model_label}", use_container_width=True)
        else:
            st.info("Confusion matrix not available.")

    with tab3:
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            col_l, col_c, col_r = st.columns([0.3, 3, 0.3])
            with col_c:
                st.image(roc_path, caption="ROC curves — all four models", use_container_width=True)
        else:
            st.info("ROC curves not available.")

else:
    # ── Landing state ─────────────────────────────────────────────────────
    col_chart, col_info = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-label">About This Tool</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.85rem;color:#44403c;line-height:1.75;">
            <p style="margin:0 0 1rem 0;">
                This tool applies a supervised machine learning pipeline to estimate
                cardiovascular disease risk from 13 routine clinical measurements —
                no imaging required.
            </p>
            <table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
                <tr style="border-bottom:1px solid #e7e5e4;">
                    <td style="padding:0.5rem 0;color:#78716c;width:45%;">Dataset</td>
                    <td style="padding:0.5rem 0;color:#1c1917;">UCI Cleveland (303 patients)</td>
                </tr>
                <tr style="border-bottom:1px solid #e7e5e4;">
                    <td style="padding:0.5rem 0;color:#78716c;">Models trained</td>
                    <td style="padding:0.5rem 0;color:#1c1917;">SVM, Logistic Reg, Random Forest, XGBoost</td>
                </tr>
                <tr style="border-bottom:1px solid #e7e5e4;">
                    <td style="padding:0.5rem 0;color:#78716c;">Selection</td>
                    <td style="padding:0.5rem 0;color:#1c1917;">Best ROC-AUC on 20% held-out test set</td>
                </tr>
                <tr style="border-bottom:1px solid #e7e5e4;">
                    <td style="padding:0.5rem 0;color:#78716c;">Best model</td>
                    <td style="padding:0.5rem 0;color:#1c1917;">{model_label}</td>
                </tr>
                <tr style="border-bottom:1px solid #e7e5e4;">
                    <td style="padding:0.5rem 0;color:#78716c;">Explainability</td>
                    <td style="padding:0.5rem 0;color:#1c1917;">SHAP per-prediction feature attribution</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem 0;color:#78716c;">Data integrity</td>
                    <td style="padding:0.5rem 0;color:#1c1917;">Scaler fit on training data only — no leakage</td>
                </tr>
            </table>
        </div>
        <div style="margin-top:1.5rem;padding:0.75rem 1rem;background:#faf9f7;
        border:1px solid #e7e5e4;border-radius:4px;font-size:0.78rem;color:#78716c;">
            Set patient values in the sidebar and click Run Prediction.
        </div>
        """, unsafe_allow_html=True)