"""
Evaluasi model dan pembuatan visualisasi untuk laporan ilmiah.

Modul ini menghitung MAE, MSE, RMSE, MAPE, R^2 baik pada skala rasio harga
(target prediksi langsung) maupun pada skala rupiah (rasio dikalikan base
price). Selain itu modul ini juga menghasilkan figur-figur yang siap
disertakan di paper:
  * Distribusi target
  * Tren rata-rata rasio harga per hari
  * Korelasi fitur numerik
  * Kurva alpha vs CV-MSE
  * Scatter aktual vs prediksi
  * Plot residual
  * Top koefisien Ridge
  * Ringkasan metrik Ridge (test set)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from config import (
    BASE_PRICE_COLUMN,
    FIGURE_DIR,
    NUMERIC_FEATURES,
    REPORT_DIR,
    TARGET_COLUMN,
)

sns.set_theme(style="whitegrid", context="paper")


@dataclass
class EvaluationResult:
    metrics_ratio: dict
    metrics_rupiah: dict


def _ensure_dirs():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_predictions(
    y_true_ratio: np.ndarray,
    y_pred_ratio: np.ndarray,
    base_price: np.ndarray,
) -> EvaluationResult:
    metrics_ratio = _metrics(y_true_ratio, y_pred_ratio)
    actual_rupiah = y_true_ratio * base_price
    pred_rupiah = y_pred_ratio * base_price
    metrics_rupiah = _metrics(actual_rupiah, pred_rupiah)
    return EvaluationResult(metrics_ratio=metrics_ratio, metrics_rupiah=metrics_rupiah)


def write_metrics_report(
    ridge_eval: EvaluationResult,
    best_alpha: float,
    n_train: int,
    n_test: int,
    n_features: int,
):
    _ensure_dirs()
    report = {
        "summary": {
            "n_train": n_train,
            "n_test": n_test,
            "n_features": n_features,
            "best_alpha": best_alpha,
        },
        "ridge_regression": {
            "ratio_scale": ridge_eval.metrics_ratio,
            "rupiah_scale": ridge_eval.metrics_rupiah,
        },
    }
    json_path = REPORT_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = REPORT_DIR / "metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Hasil Evaluasi Ridge Regression\n\n")
        f.write(f"- Jumlah data train: **{n_train}**\n")
        f.write(f"- Jumlah data test:  **{n_test}**\n")
        f.write(f"- Jumlah fitur (setelah encoding): **{n_features}**\n")
        f.write(f"- Alpha optimal (RidgeCV): **{best_alpha}**\n\n")
        f.write("## Skala Price Ratio\n\n")
        f.write("| Metrik | Nilai |\n|---|---|\n")
        for k in ["MAE", "MSE", "RMSE", "MAPE", "R2"]:
            f.write(f"| {k} | {ridge_eval.metrics_ratio[k]:.6f} |\n")
        f.write("\n## Skala Rupiah (rasio x base price)\n\n")
        f.write("| Metrik | Nilai |\n|---|---|\n")
        for k in ["MAE", "MSE", "RMSE", "MAPE", "R2"]:
            f.write(f"| {k} | {ridge_eval.metrics_rupiah[k]:.2f} |\n")

    return json_path, md_path


def plot_target_distribution(df: pd.DataFrame):
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[TARGET_COLUMN], bins=40, kde=True, ax=ax, color="#3b82f6")
    ax.set_title("Distribusi Price Ratio (Actual / Base Price)")
    ax.set_xlabel("Price Ratio")
    ax.set_ylabel("Frekuensi")
    fig.tight_layout()
    path = FIGURE_DIR / "01_distribusi_price_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_avg_ratio_by_day(df: pd.DataFrame):
    _ensure_dirs()
    order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    avg = df.groupby("Day")[TARGET_COLUMN].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=avg.index, y=avg.values, ax=ax, color="#10b981")
    ax.set_title("Rata-rata Price Ratio per Hari")
    ax.set_ylabel("Price Ratio")
    ax.set_xlabel("Hari")
    fig.tight_layout()
    path = FIGURE_DIR / "02_rata_rata_per_hari.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame):
    _ensure_dirs()
    cols = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Korelasi Fitur Numerik vs Price Ratio")
    fig.tight_layout()
    path = FIGURE_DIR / "03_korelasi_fitur.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_alpha_curve(alphas, cv_mse, best_alpha):
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, cv_mse, marker="o", color="#ef4444")
    ax.axvline(best_alpha, color="black", linestyle="--",
               label=f"alpha optimal = {best_alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (skala log)")
    ax.set_ylabel("Cross-Validation MSE")
    ax.set_title("Kurva Tuning Hyperparameter Ridge")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "04_kurva_alpha.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18, color="#6366f1")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residuals(y_true, y_pred, filename):
    _ensure_dirs()
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=18, color="#0ea5e9")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Prediksi")
    axes[0].set_ylabel("Residual (aktual - prediksi)")
    axes[0].set_title("Plot Residual")

    sns.histplot(residuals, bins=40, kde=True, ax=axes[1], color="#f59e0b")
    axes[1].set_title("Distribusi Residual")
    axes[1].set_xlabel("Residual")
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_top_coefficients(model, feature_names, top_k=15):
    _ensure_dirs()
    coefs = pd.Series(model.coef_, index=feature_names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:top_k]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#dc2626" if v < 0 else "#16a34a" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_title(f"Top {top_k} Koefisien Ridge")
    ax.set_xlabel("Bobot")
    fig.tight_layout()
    path = FIGURE_DIR / "07_koefisien_ridge.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ridge_metrics_summary(ridge_eval: EvaluationResult):
    """Bar chart ringkasan MAE / RMSE / MAPE pada skala price ratio (test set)."""
    _ensure_dirs()
    metrics = ["MAE", "RMSE", "MAPE"]
    vals = [ridge_eval.metrics_ratio[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics, vals, color="#2563eb")
    ax.set_title("Ringkasan metrik Ridge Regression (skala price ratio, test set)")
    ax.set_ylabel("Nilai metrik")
    fig.tight_layout()
    path = FIGURE_DIR / "08_ringkasan_metrik_ridge.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "EvaluationResult",
    "evaluate_predictions",
    "write_metrics_report",
    "plot_target_distribution",
    "plot_avg_ratio_by_day",
    "plot_correlation_heatmap",
    "plot_alpha_curve",
    "plot_actual_vs_predicted",
    "plot_residuals",
    "plot_top_coefficients",
    "plot_ridge_metrics_summary",
]
