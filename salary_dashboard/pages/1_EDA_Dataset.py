"""Streamlit page: Dataset preview and EDA."""
from __future__ import annotations

# Ensure project root in path
import sys, pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from salary_dashboard.data_loader import load_data
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



st.header("🗂️ Dataset Preview & EDA")

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# --- Basic Preview ---
st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

# Dataset dimensions
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# --- Visual Explorations ---

st.subheader("Feature Distributions")
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

with st.expander("Numeric Feature Histogram"):
    num_feat = st.selectbox("Select numeric column", num_cols)
    chart = alt.Chart(df).mark_bar(opacity=0.7, color="#3182bd").encode(
        alt.X(f"{num_feat}:Q", bin=alt.Bin(maxbins=30), title=num_feat),
        y="count()",
    )
    st.altair_chart(chart, use_container_width=True)

with st.expander("Categorical Feature Counts"):
    cat_feat = st.selectbox("Select categorical column", cat_cols)
    bar_ct = (
        df[cat_feat]
        .value_counts()
        .rename_axis(cat_feat)
        .reset_index(name="count")
    )
    chart_c = (
        alt.Chart(bar_ct)
        .mark_bar(color="#6baed6")
        .encode(x=alt.X(f"{cat_feat}:N", sort="-y"), y="count")
    )
    st.altair_chart(chart_c, use_container_width=True)

# --- Summary statistics ---
with st.expander("Summary Statistics"):
    st.dataframe(df.describe(include="all").T)

# Column info
with st.expander("Column Information"):
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ["column", "dtype"]
    st.dataframe(dtypes)
