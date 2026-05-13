"""
Pipeline utama: penerapan dynamic pricing menggunakan algoritma Ridge
Regression untuk penetapan harga booking hotel.

Alur pipeline:
  1. Memuat dataset       (data_loader.py)
  2. Membersihkan data    (data_loader.py)
  3. Feature engineering  (feature_engineering.py)
  4. Train/test split     (model.py)
  5. Tuning alpha + train Ridge (model.py)
  6. Evaluasi & figur     (evaluation.py)
  7. Persist model        (joblib)
"""
from __future__ import annotations

from datetime import datetime

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    BASE_PRICE_COLUMN,
    FIGURE_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from data_loader import load_clean_dataset
from evaluation import (
    evaluate_predictions,
    plot_actual_vs_predicted,
    plot_alpha_curve,
    plot_avg_ratio_by_day,
    plot_correlation_heatmap,
    plot_residuals,
    plot_target_distribution,
    plot_ridge_metrics_summary,
    plot_top_coefficients,
    write_metrics_report,
)
from feature_engineering import build_feature_matrix
from model import train_ridge_with_cv


def _print_header(title: str):
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


def run_pipeline():
    started = datetime.now()
    _print_header("1. LOAD & CLEAN DATASET")
    df = load_clean_dataset()
    print(f"Jumlah data setelah cleaning : {len(df)}")

    _print_header("2. EXPLORATORY VISUALIZATION")
    p1 = plot_target_distribution(df)
    p2 = plot_avg_ratio_by_day(df)
    p3 = plot_correlation_heatmap(df)
    print(f"Saved figure : {p1.name}")
    print(f"Saved figure : {p2.name}")
    print(f"Saved figure : {p3.name}")

    _print_header("3. FEATURE ENGINEERING")
    fm = build_feature_matrix(df)
    print(f"Bentuk matriks fitur X : {fm.X.shape}")
    print(f"Jumlah fitur akhir     : {len(fm.feature_names)}")

    _print_header("4. TRAIN / TEST SPLIT")
    base_prices = df[BASE_PRICE_COLUMN].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test, base_train, base_test = train_test_split(
        fm.X, fm.y, base_prices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(f"n_train={len(y_train)}  n_test={len(y_test)}")

    _print_header("5. TRAIN RIDGE + CROSS-VALIDATION")
    trained = train_ridge_with_cv(X_train, y_train, fm.feature_names)
    print(f"Alpha optimal terpilih : {trained.best_alpha}")
    p4 = plot_alpha_curve(trained.cv_alphas, trained.cv_mse, trained.best_alpha)
    print(f"Saved figure : {p4.name}")

    _print_header("6. EVALUASI MODEL (Ridge)")
    y_pred_ridge = trained.model.predict(X_test)

    ridge_eval = evaluate_predictions(y_test, y_pred_ridge, base_test)

    _print_metrics_table(ridge_eval)

    p5 = plot_actual_vs_predicted(
        y_test, y_pred_ridge,
        title="Aktual vs Prediksi Price Ratio (Ridge)",
        filename="05_aktual_vs_prediksi.png",
    )
    p6 = plot_residuals(
        y_test, y_pred_ridge, filename="06_residual.png",
    )
    p7 = plot_top_coefficients(trained.model, trained.feature_names)
    p8 = plot_ridge_metrics_summary(ridge_eval)
    print("Saved figure :", p5.name, p6.name, p7.name, p8.name)

    _print_header("7. SIMPAN MODEL & LAPORAN")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = MODEL_DIR / "ridge_pipeline.joblib"
    joblib.dump(
        {
            "preprocessor": fm.preprocessor,
            "model": trained.model,
            "feature_names": trained.feature_names,
            "best_alpha": trained.best_alpha,
        },
        bundle_path,
    )
    json_path, md_path = write_metrics_report(
        ridge_eval=ridge_eval,
        best_alpha=trained.best_alpha,
        n_train=len(y_train),
        n_test=len(y_test),
        n_features=len(fm.feature_names),
    )
    print(f"Model disimpan ke : {bundle_path.relative_to(MODEL_DIR.parent.parent)}")
    print(f"Laporan JSON      : {json_path.relative_to(REPORT_DIR.parent.parent)}")
    print(f"Laporan Markdown  : {md_path.relative_to(REPORT_DIR.parent.parent)}")

    duration = (datetime.now() - started).total_seconds()
    _print_header(f"PIPELINE SELESAI ({duration:.1f} detik)")


def _print_metrics_table(ridge_eval):
    rows = []
    for k in ["MAE", "MSE", "RMSE", "MAPE", "R2"]:
        rows.append({
            "Metrik": k,
            "Rasio": f"{ridge_eval.metrics_ratio[k]:.6f}",
            "Rupiah": f"{ridge_eval.metrics_rupiah[k]:,.2f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
