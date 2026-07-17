"""
Pelatihan model Ridge Regression untuk dynamic pricing.

Ridge Regression meminimalkan fungsi objektif:

    L(w) = sum_i (y_i - x_i^T w)^2 + alpha * ||w||_2^2

Parameter alpha (lambda regularisasi) dipilih menggunakan k-fold
cross-validation pada training set sehingga model tidak overfit terhadap
pola spesifik dari sampel pelatihan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

from config import ALPHA_GRID, CV_FOLDS, RANDOM_STATE, TEST_SIZE


@dataclass
class TrainedRidge:
    model: Ridge
    best_alpha: float
    cv_alphas: list[float]
    cv_mse: list[float]
    feature_names: list[str]


def split_dataset(X: np.ndarray, y: np.ndarray):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )


def train_ridge_with_cv(X_train, y_train, feature_names) -> TrainedRidge:
    ridge_cv = RidgeCV(
        alphas=ALPHA_GRID,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
    )
    ridge_cv.fit(X_train, y_train)
    best_alpha = float(ridge_cv.alpha_)

    cv_alphas, cv_mse = _evaluate_alpha_grid(X_train, y_train)

    final_model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)

    return TrainedRidge(
        model=final_model,
        best_alpha=best_alpha,
        cv_alphas=cv_alphas,
        cv_mse=cv_mse,
        feature_names=feature_names,
    )


def _evaluate_alpha_grid(X_train, y_train):
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    alphas, mse_values = [], []
    for alpha in ALPHA_GRID:
        fold_mse = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            m = Ridge(alpha=alpha, random_state=RANDOM_STATE).fit(X_tr, y_tr)
            fold_mse.append(mean_squared_error(y_val, m.predict(X_val)))
        alphas.append(alpha)
        mse_values.append(float(np.mean(fold_mse)))
    return alphas, mse_values


if __name__ == "__main__":
    from data_loader import load_clean_dataset
    from feature_engineering import build_feature_matrix

    df = load_clean_dataset()
    fm = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_dataset(fm.X, fm.y)
    
    # Info pembagian dataset (untuk sub-bab 3.4.1)
    print(f"\n[SPLIT] Total record       : {fm.X.shape[0]}")
    print(f"[SPLIT] Training set        : {X_train.shape[0]} record")
    print(f"[SPLIT] Testing set         : {X_test.shape[0]} record")
    
    trained = train_ridge_with_cv(X_train, y_train, fm.feature_names)
    
    # Info hasil tuning (untuk sub-bab 3.4.3)
    print(f"\n[TUNING] Grid CV-MSE per alpha:")
    for a, m in zip(trained.cv_alphas, trained.cv_mse):
        marker = "  <-- optimal" if a == trained.best_alpha else ""
        print(f"         alpha={a:>8.3f}  cv_mse={m:.6f}{marker}")
    
    print(f"\n[RESULT] Best alpha = {trained.best_alpha}")