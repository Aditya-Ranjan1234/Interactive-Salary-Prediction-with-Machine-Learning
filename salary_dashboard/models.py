"""Model training utilities."""
from __future__ import annotations

import pathlib
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Optional xgboost
try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover
    XGBClassifier = None
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, clean_data, get_feature_lists

MODEL_DIR = pathlib.Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)


def _build_pipeline(model, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """Return a complete preprocessing + estimator Pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", model),
        ]
    )


def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict]:
    """Train several algorithms and return performance + models.

    Returns a mapping of model name to a dict with keys: pipeline, metrics, model_path.
    """
    df = clean_data(df)
    numeric_cols, categorical_cols, target_col = get_feature_lists(df)

    X = df.drop(columns=[target_col])
    y = (df[target_col] == ">50K").astype(int)  # binary 0/1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10),
    }
    if XGBClassifier is not None:
        algorithms["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    results: Dict[str, Dict] = {}
    for name, estimator in algorithms.items():
        pipe = _build_pipeline(estimator, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Persist model for reuse
        model_path = MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(pipe, model_path)

        results[name] = {
            "pipeline": pipe,
            "metrics": metrics,
            "model_path": str(model_path),
            "y_test": y_test,  # actual labels for inspection
            "X_test": X_test,  # features of test set
        }

    return results

def load_models() -> Dict[str, Pipeline]:
    """Load all trained models from artifacts directory.

    Returns a mapping from model name (derived from filename) to Pipeline.
    """
    models: Dict[str, Pipeline] = {}
    for path in MODEL_DIR.glob("*.joblib"):
        name = path.stem.replace("_", " ").title()
        models[name] = joblib.load(path)
    return models
