"""Data loading utilities for Employee Salary Prediction dashboard."""

from __future__ import annotations

import pathlib
from typing import Union

import pandas as pd


DEFAULT_CSV_NAME = "adult 3.csv"


def load_data(csv_path: Union[str, pathlib.Path, None] = None) -> pd.DataFrame:
    """Load census-salary dataset.

    If *csv_path* is None, it looks for `adult 3.csv` one level above the
    project folder.
    """
    if csv_path is None:
        base_dir = pathlib.Path(__file__).resolve().parent.parent
        csv_path = base_dir / DEFAULT_CSV_NAME
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df
