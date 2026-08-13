"""
app.py — Cancer Stage Predictor
Streamlit web app for the Global Cancer Patients ML Project.

Run locally : streamlit run app.py
Deploy free : https://share.streamlit.io  (connect your GitHub repo)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Stage Predictor",
    page_icon="🎗️",
    layout="centered",
)

# ── Load saved model & transformers ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model          = joblib.load("model.pkl")
    scaler         = joblib.load("scaler.pkl")
    label_encoder  = joblib.load("label_encoder.pkl")
    feature_names  = joblib.load("feature_names.pkl")
    return model, scaler, label_encoder, feature_names

try:
    model, scaler, le, feature_names = load_artifacts()
    artifacts_ok = True
except FileNotFoundError:
    artifacts_ok = False

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🎗️ Cancer Stage Predictor")
st.markdown(
    "Enter patient information below to predict the **cancer stage** "
    "(I, II, III, or IV) using our trained Random Forest model."
)
st.divider()

# ── Sidebar — Project info ────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.markdown("""
    **Dataset:** Global Cancer Patients 2015–2024  
    **Model:** Tuned Random Forest Classifier  
    **Target:** Cancer Stage (I / II / III / IV)  
    **Features:** 10 selected clinical + demographic variables
    """)
    st.divider()
    st.subheader("📊 Model Performance")
    st.metric("Accuracy",  "≥ 0.75")
    st.metric("Precision", "≥ 0.75")
    st.metric("Recall",    "≥ 0.75")
    st.caption("Exact values depend on your final trained model.")

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=10, max_value=90, value=50, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cancer_type = st.selectbox(
        "Cancer Type",
        ["Lung", "Breast", "Colon", "Prostate", "Liver",
         "Stomach", "Cervical", "Leukemia", "Lymphoma", "Skin"]
    )
    country = st.selectbox(
        "Country / Region",
        ["USA", "India", "China", "Brazil", "UK",
         "Germany", "France", "Australia", "Japan", "Other"]
    )
    treatment_type = st.selectbox(
        "Treatment Type",
        ["Surgery", "Chemotherapy", "Radiation", "Immunotherapy",
         "Targeted Therapy", "Combination"]
    )

with col2:
    survival_years  = st.slider("Survival Years (so far)", 0.0, 20.0, 3.0, step=0.5)
    treatment_cost  = st.number_input(
        "Treatment Cost (USD)", min_value=0, max_value=500_000,
        value=25_000, step=1000
    )
    year            = st.selectbox("Diagnosis Year", list(range(2015, 2025)), index=5)

st.divider()

# ── Engineer the same features as the notebook ────────────────────────────────
def build_input_row(age, gender, cancer_type, country,
                    treatment_type, survival_years, treatment_cost, year):
    """Build a feature-engineered row matching the training pipeline."""

    gender_enc        = 1 if gender == "Male" else 0
    is_senior         = 1 if age >= 60 else 0
    treatment_eff     = survival_years / (treatment_cost + 1)

    age_group_map     = {"Young": 0, "Middle": 1, "Senior": 2, "Elderly": 3}
    if age < 30:    age_group = 0
    elif age < 45:  age_group = 1
    elif age < 60:  age_group = 2
    else:           age_group = 3

    # Simple ordinal encoding for categoricals (matches LabelEncoder order)
    cancer_types = ["Breast", "Cervical", "Colon", "Leukemia", "Liver",
                    "Lung", "Lymphoma", "Prostate", "Skin", "Stomach"]
    cancer_enc = cancer_types.index(cancer_type) if cancer_type in cancer_types else 0

    treatment_types = ["Chemotherapy", "Combination", "Immunotherapy",
                       "Radiation", "Surgery", "Targeted Therapy"]
    treatment_enc = treatment_types.index(treatment_type) if treatment_type in treatment_types else 0

    countries = ["Australia", "Brazil", "China", "France", "Germany",
                 "India", "Japan", "Other", "UK", "USA"]
    country_enc = countries.index(country) if country in countries else 0

    row = {
        "Age"                  : age,
        "Gender"               : gender_enc,
        "Cancer_Type"          : cancer_enc,
        "Country"              : country_enc,
        "Year"                 : year,
        "Treatment_Type"       : treatment_enc,
        "Survival_Years"       : survival_years,
        "Treatment_Cost_USD"   : treatment_cost,
        "Treatment_Efficiency" : treatment_eff,
        "Is_Senior"            : is_senior,
        "Age_Group"            : age_group,
    }
    return row


# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Cancer Stage", use_container_width=True, type="primary"):
    if not artifacts_ok:
        st.error(
            "⚠️ Model files not found. Make sure `model.pkl`, `scaler.pkl`, "
            "`label_encoder.pkl`, and `feature_names.pkl` are in the same folder as `app.py`."
        )
    else:
        row = build_input_row(
            age, gender, cancer_type, country,
            treatment_type, survival_years, treatment_cost, year
        )

        # Align columns to training feature set
        input_df = pd.DataFrame([row])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Predict
        pred_encoded = model.predict(input_df)[0]
        pred_proba   = model.predict_proba(input_df)[0]
        pred_label   = le.inverse_transform([pred_encoded])[0]

        # Display result
        stage_colors = {"Stage I": "🟢", "Stage II": "🟡",
                        "Stage III": "🟠", "Stage IV": "🔴"}
        icon = stage_colors.get(pred_label, "⚪")

        st.success(f"### {icon} Predicted Cancer Stage: **{pred_label}**")

        # Confidence bar
        st.subheader("Prediction Confidence")
        proba_df = pd.DataFrame({
            "Stage"       : le.classes_,
            "Probability" : pred_proba
        }).sort_values("Stage")

        for _, r in proba_df.iterrows():
            st.progress(float(r["Probability"]),
                        text=f"{r['Stage']} — {r['Probability']*100:.1f}%")

        # Feature summary
        st.divider()
        with st.expander("📋 Input Summary"):
            st.json(row)

        st.caption(
            "⚠️ This tool is for **educational purposes only**. "
            "It is not a substitute for professional medical diagnosis."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · Model: Random Forest (GridSearchCV tuned) · "
    "Dataset: Global Cancer Patients 2015–2024 (Kaggle)"
)
