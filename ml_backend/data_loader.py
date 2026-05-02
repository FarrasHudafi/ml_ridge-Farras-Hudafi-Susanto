"""
Pemuatan dan pembersihan dataset booking hotel.

Tahapan ini bertanggung jawab atas:
  1. Membaca file Excel dataset_hotel_dynamic_pricing.xlsx
  2. Memfilter reservasi yang valid (status Confirmed)
  3. Membuang outlier rasio harga yang tidak realistis
  4. Mengembalikan DataFrame yang sudah siap untuk feature engineering
"""
from __future__ import annotations

import pandas as pd

from config import (
    ACTUAL_PRICE_COLUMN,
    BASE_PRICE_COLUMN,
    DATASET_PATH,
    PRICE_RATIO_LOWER,
    PRICE_RATIO_UPPER,
    TARGET_COLUMN,
)


def load_raw_dataset(path=DATASET_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)

    df = df[df["Status"].astype(str).str.lower() == "confirmed"].copy()
    after_status = len(df)

    df = df[df[BASE_PRICE_COLUMN] > 0].copy()
    df = df[df[ACTUAL_PRICE_COLUMN] > 0].copy()
    after_price = len(df)

    mask_ratio = (df[TARGET_COLUMN] >= PRICE_RATIO_LOWER) & (
        df[TARGET_COLUMN] <= PRICE_RATIO_UPPER
    )
    df = df[mask_ratio].copy()
    after_ratio = len(df)

    df = df.dropna(
        subset=[
            TARGET_COLUMN,
            "Occupancy Rate",
            "Lead Time (days)",
            "Length of Stay",
            "Room Type",
        ]
    ).reset_index(drop=True)
    after_dropna = len(df)

    summary = {
        "initial_rows": initial,
        "after_status_filter": after_status,
        "after_price_filter": after_price,
        "after_ratio_filter": after_ratio,
        "after_dropna": after_dropna,
    }
    print("[CLEAN] " + " -> ".join(f"{k}={v}" for k, v in summary.items()))

    return df


def load_clean_dataset(path=DATASET_PATH) -> pd.DataFrame:
    return clean_dataset(load_raw_dataset(path))


if __name__ == "__main__":
    df = load_clean_dataset()
    print(df.head())
    print(f"Final shape: {df.shape}")
