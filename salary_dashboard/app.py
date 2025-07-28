"""Streamlit entrypoint for Employee Salary Prediction dashboard."""
from __future__ import annotations

# Ensure parent directory is in Python path so `import salary_dashboard` works
import sys
import pathlib

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💰 Employee Salary Prediction Dashboard")

st.write("""Use the sidebar to navigate between pages:
1. **Dataset Preview & EDA** – inspect the raw census data.
2. **Model Training** – train several algorithms and compare metrics.
3. **Predict Salary** – input employee details and see predictions from all models.
""")

st.write("Built with ❤️ using Streamlit.")
