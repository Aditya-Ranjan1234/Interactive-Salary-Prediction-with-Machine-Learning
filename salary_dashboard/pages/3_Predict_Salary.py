"""Streamlit page: Input form and salary prediction using trained models."""
from __future__ import annotations

# Ensure project root in path
import sys, pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from salary_dashboard.data_loader import load_data
from salary_dashboard.models import load_models

st.header("🔮 Predict Employee Salary")

models = load_models()
if not models:
    st.info("No trained models found. Please go to **Model Training** page and train models first.")
    st.stop()

# Load dataset once to infer columns and categories
base_df = load_data()

# Build input form dynamically
with st.form("prediction_form"):
    st.subheader("Enter Employee Details")

    # Collect numerical features
    num_cols = [
        "age",
        "fnlwgt",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    num_inputs = {
        col: st.number_input(col.replace("-", " ").title(), value=float(base_df[col].median())) for col in num_cols
    }

    # Collect categorical features with select boxes (use unique values from dataset)
    cat_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ]

    cat_inputs = {}
    for col in cat_cols:
        options = sorted(base_df[col].unique())
        default = options[0]
        cat_inputs[col] = st.selectbox(col.replace("-", " ").title(), options, index=options.index(default))

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    input_data = {
        **num_inputs,
        **cat_inputs,
    }
    input_df = pd.DataFrame([input_data])

    st.subheader("Model Predictions")
    result_rows = []
    for name, model in models.items():
        pred = model.predict(input_df)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        label = ">50K" if pred == 1 else "<=50K"
        result_rows.append((name, label, prob))

    res_df = pd.DataFrame(result_rows, columns=["Model", "Prediction", "Prob >50K"],)
    st.dataframe(res_df.style.background_gradient(cmap="Blues"), use_container_width=True)
