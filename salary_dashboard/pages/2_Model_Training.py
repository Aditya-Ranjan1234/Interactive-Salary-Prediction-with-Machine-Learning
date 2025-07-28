"""Streamlit page: Train multiple ML algorithms."""
from __future__ import annotations

# Ensure project root in path
import sys, pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from salary_dashboard.data_loader import load_data
from salary_dashboard.models import train_models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

st.header("🤖 Model Training & Evaluation")

if "training_results" not in st.session_state:
    st.session_state["training_results"] = None

cols = st.columns(2)
with cols[0]:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
with cols[1]:
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

if st.button("Train / Retrain Models"):
    with st.spinner("Training models, please wait…"):
        df = load_data()
        results = train_models(df, test_size=test_size, random_state=random_state)
        st.session_state["training_results"] = results
    st.success("Training complete!")

results = st.session_state["training_results"]
if results:
    st.subheader("Performance Metrics (Test Set)")
    metrics_table = {
        name: {
            "Accuracy": info["metrics"]["accuracy"],
            "Precision": info["metrics"]["precision"],
            "Recall": info["metrics"]["recall"],
            "F1": info["metrics"]["f1"],
        }
        for name, info in results.items()
    }
    st.dataframe(metrics_table, use_container_width=True)

    # Metric bar chart
    import pandas as pd
    mdf = (
        pd.DataFrame(metrics_table).T.reset_index().rename(columns={"index": "Model"})
    )
    st.bar_chart(mdf.set_index("Model"))

    # Confusion matrices
    st.subheader("Confusion Matrices")
    for name, info in results.items():
        y_test = info["y_test"]
        X_test = info["X_test"]
        y_pred = info["pipeline"].predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        st.pyplot(fig)
else:
    st.info("Train the models to view performance.")
