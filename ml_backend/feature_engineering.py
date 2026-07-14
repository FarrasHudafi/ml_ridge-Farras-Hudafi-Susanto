"""
Feature engineering untuk model Ridge Regression.

Mengubah DataFrame mentah hasil cleaning menjadi matriks fitur X dan target y
yang sudah siap di-train. One-hot encoding diterapkan pada variabel
kategorikal dan StandardScaler diterapkan pada variabel numerik.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = StandardScaler()
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

def build_feature_matrix(df: pd.DataFrame) -> FeatureMatrix:
    required = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang dari dataset: {missing}")

    X_df = df[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(float).to_numpy()

    preprocessor = _build_preprocessor()
    X = preprocessor.fit_transform(X_df)
    feature_names = _expand_feature_names(preprocessor)

    return FeatureMatrix(X=X, y=y, feature_names=feature_names, preprocessor=preprocessor)

def _expand_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = list(NUMERIC_FEATURES) + list(BINARY_FEATURES)
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    names.extend(cat_names.tolist())
    return names




def transform_for_inference(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    return preprocessor.transform(df[cols])


if __name__ == "__main__":
    from data_loader import load_clean_dataset

    df = load_clean_dataset()
    fm = build_feature_matrix(df)
    print(f"X shape: {fm.X.shape}, y shape: {fm.y.shape}")
    print(f"#features: {len(fm.feature_names)}")
    print("Contoh fitur:", fm.feature_names[:8], "...")
