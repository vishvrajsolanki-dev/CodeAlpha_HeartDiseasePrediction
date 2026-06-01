import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    best_name = joblib.load("models/best_model_name.pkl")
    return model, scaler, feature_names, best_name

model, scaler, feature_names, best_name = load_artifacts()

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
            st.error(f"### ⚠️ Heart Disease Detected")
        else:
            st.success(f"### ✅ No Heart Disease Detected")

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

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(shap_values[0, :, 1], feature_names=feature_names, show=False, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.image("outputs/confusion_matrix.png", caption="Confusion Matrix on Test Set")

    with tab3:
        st.image("outputs/roc_curves.png", caption="ROC Curve Comparison — All Models")

else:
    st.info("Set patient values in the sidebar and click **Predict** to get a result.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", best_name.replace("_", " ").title())
    col2.metric("ROC-AUC", "0.9542")
    col3.metric("Test Accuracy", "91.67%")

    st.markdown("---")
    st.subheader("Model Comparison")
    st.image("outputs/roc_curves.png")