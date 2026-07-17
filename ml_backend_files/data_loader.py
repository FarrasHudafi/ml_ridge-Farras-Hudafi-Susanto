"""
Pemuatan dan pembersihan dataset booking hotel (Hotel Booking Demand,
Antonio, Almeida & Nunes, 2019).

Tahapan:
  1. Membaca file hotel_bookings.csv
  2. Memfilter reservasi valid (is_canceled == 0 / tidak dibatalkan)
  3. Membuang harga tidak valid (adr <= 0)
  4. Membentuk target Price Ratio = adr / BASE_REFERENCE
  5. Membuang outlier rasio harga di luar rentang wajar
  6. Menurunkan fitur (Length of Stay, Total Guests, Month Sin/Cos,
     Is Weekend/Sunday/Midweek, Season) dari kolom mentah
  7. Mengembalikan DataFrame siap feature engineering
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ADR_COLUMN,
    BASE_REFERENCE,
    DATASET_PATH,
    PRICE_RATIO_LOWER,
    PRICE_RATIO_UPPER,
    TARGET_COLUMN,
)

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
_DAYS_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def _season_from_month(month: int) -> str:
    if month in (6, 7, 8, 12):
        return "High"
    if month in (1, 2, 11):
        return "Low"
    return "Mid"


def load_raw_dataset(path=DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _engineer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turunkan kolom fitur dari atribut mentah dataset."""
    df = df.copy()
    df["children"] = df["children"].fillna(0)

    month_num = df["arrival_date_month"].map(_MONTHS).astype(int)
    arrival = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=month_num,
        day=df["arrival_date_day_of_month"],
    ), errors="coerce")

    dow = arrival.dt.weekday          # Senin=0 ... Minggu=6
    df["Lead Time (days)"] = df["lead_time"].astype(float)
    df["Lead Time Norm"] = (df["lead_time"].clip(upper=90) / 90).astype(float)
    df["Length of Stay"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    ).astype(float)
    df["Total Guests"] = (
        df["adults"] + df["children"] + df["babies"]
    ).astype(float)
    df["Month Sin"] = np.sin(2 * np.pi * (month_num - 1) / 12)
    df["Month Cos"] = np.cos(2 * np.pi * (month_num - 1) / 12)
    df["Is Weekend"] = dow.isin([4, 5]).astype(int)   # Jumat, Sabtu
    df["Is Sunday"] = (dow == 6).astype(int)
    df["Is Midweek"] = dow.isin([1, 2, 3]).astype(int)
    df["Segment"] = df["market_segment"].astype(str)
    df["Channel"] = df["distribution_channel"].astype(str)
    df["Season"] = month_num.map(_season_from_month).astype(str)
    df["Day"] = dow.map(dict(enumerate(_DAYS_ID)))
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)

    df = df[df["is_canceled"] == 0].copy()
    after_status = len(df)

    df = df[df[ADR_COLUMN] > 0].copy()
    after_price = len(df)

    df = _engineer_columns(df)
    df[TARGET_COLUMN] = df[ADR_COLUMN] / BASE_REFERENCE

    mask_ratio = (df[TARGET_COLUMN] >= PRICE_RATIO_LOWER) & (
        df[TARGET_COLUMN] <= PRICE_RATIO_UPPER
    )
    df = df[mask_ratio].copy()
    after_ratio = len(df)

    df = df.dropna(
        subset=[
            TARGET_COLUMN,
            "Lead Time (days)",
            "Length of Stay",
            "Total Guests",
            "Month Sin",
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
    print(f"Final shape: {df.shape}")
    print(df.head())
