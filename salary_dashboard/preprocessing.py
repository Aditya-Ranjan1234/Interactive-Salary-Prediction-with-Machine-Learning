"""Data cleaning and preprocessing utilities."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_PLACEHOLDER = "?"
CATEGORICAL_REPLACEMENT = "Others"


def _replace_placeholders(df: pd.DataFrame, cols: List[str]) -> None:
    """Replace '?' placeholders in *cols* with CATEGORICAL_REPLACEMENT (in-place)."""
    for col in cols:
        df[col] = df[col].replace({CATEGORICAL_PLACEHOLDER: CATEGORICAL_REPLACEMENT})


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataset following the notebook logic."""
    df = df.copy()

    cat_cols_with_missing = [
        "workclass",
        "occupation",
    ]
    _replace_placeholders(df, cat_cols_with_missing)

    # Remove rare workclass categories that add noise
    df = df[~df["workclass"].isin(["Without-pay", "Never-worked"])]

    # Age bounds (17-75) as used in the notebook
    df = df[(df["age"] >= 17) & (df["age"] <= 75)]

    # Reset index to keep things tidy
    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    """Return (numeric_cols, categorical_cols, target_col)."""
    target = "income"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target) if target in numeric_cols else None
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]
    return numeric_cols, categorical_cols, target


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Construct a sklearn ColumnTransformer for mixed-type preprocessing."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor
