"""
Skrip inferensi: penerapan dynamic pricing pada kasus baru.

Fitur model diturunkan HANYA dari tanggal menginap dan konteks booking
(lead time, lama menginap, jumlah tamu, bulan, akhir pekan, segment,
channel, musim). Room Type & Occupancy TIDAK dipakai model — kelas kamar
sudah diwakili base price kamar (dari basis data HotelKu).

harga_rekomendasi = base_price x clamp(Price Ratio prediksi, 0.7, 1.5)

Contoh:
  python predict.py --night-date 2026-12-25 --booking-date 2026-05-01 \
      --base-price 850000 --segment Direct --channel Direct
"""
from __future__ import annotations

import argparse
import math
from datetime import date, datetime

import joblib
import pandas as pd

from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    NUMERIC_FEATURES,
    PRICE_RATIO_CLAMP,
)


def _season_from_month(month: int) -> str:
    if month in (6, 7, 8, 12):
        return "High"
    if month in (1, 2, 11):
        return "Low"
    return "Mid"


def build_input_row(
    night_date: date,
    booking_date: date,
    base_price: float,
    room_type: str | None = None,   # diterima untuk kompatibilitas; tidak dipakai model
    segment: str = "Direct",
    channel: str = "Direct",
    season: str | None = None,
    length_of_stay: int = 1,
    total_guests: int = 2,
    **_ignored,
) -> pd.DataFrame:
    dow = night_date.weekday()                 # Senin=0 ... Minggu=6
    is_weekend = int(dow in (4, 5))            # Jumat, Sabtu
    is_sunday = int(dow == 6)
    is_midweek = int(dow in (1, 2, 3))
    month = night_date.month
    month_sin = math.sin(2 * math.pi * (month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12)
    lead_time = max(0, (night_date - booking_date).days)
    lead_time_norm = min(lead_time, 90) / 90

    if season is None:
        season = _season_from_month(month)

    row = {
        "Lead Time (days)": lead_time,
        "Lead Time Norm": lead_time_norm,
        "Length of Stay": length_of_stay,
        "Total Guests": total_guests,
        "Month Sin": month_sin,
        "Month Cos": month_cos,
        "Is Weekend": is_weekend,
        "Is Sunday": is_sunday,
        "Is Midweek": is_midweek,
        "Segment": segment,
        "Channel": channel,
        "Season": season,
    }
    cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    return pd.DataFrame([row])[cols]


def predict_price(
    night_date: date,
    booking_date: date,
    base_price: float,
    room_type: str | None = None,
    **kwargs,
) -> dict:
    bundle = joblib.load(MODEL_DIR / "ridge_pipeline.joblib")
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    row = build_input_row(
        night_date=night_date,
        booking_date=booking_date,
        base_price=base_price,
        room_type=room_type,
        **kwargs,
    )
    X = preprocessor.transform(row)
    raw_ratio = float(model.predict(X)[0])
    lo, hi = PRICE_RATIO_CLAMP
    clamped_ratio = max(lo, min(hi, raw_ratio))
    recommended_price = round(base_price * clamped_ratio, -3)

    return {
        "input": row.iloc[0].to_dict(),
        "raw_price_ratio": raw_ratio,
        "clamped_price_ratio": clamped_ratio,
        "base_price": base_price,
        "recommended_price": recommended_price,
        "delta_rupiah": recommended_price - base_price,
        "delta_pct": (recommended_price - base_price) / base_price * 100,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Dynamic pricing inference")
    p.add_argument("--night-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--booking-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--base-price", type=float, required=True)
    p.add_argument("--room-type", default=None,
                   choices=["Standard", "Superior", "Deluxe", "Suite"],
                   help="Opsional; hanya untuk konteks, tidak dipakai model")
    p.add_argument("--segment", default="Direct")
    p.add_argument("--channel", default="Direct")
    p.add_argument("--length-of-stay", type=int, default=1)
    p.add_argument("--total-guests", type=int, default=2)
    return p.parse_args()


def main():
    args = _parse_args()
    result = predict_price(
        night_date=date.fromisoformat(args.night_date),
        booking_date=date.fromisoformat(args.booking_date),
        base_price=args.base_price,
        room_type=args.room_type,
        segment=args.segment,
        channel=args.channel,
        length_of_stay=args.length_of_stay,
        total_guests=args.total_guests,
    )
    print("=== Hasil Dynamic Pricing ===")
    print(f"Base Price          : Rp {args.base_price:,.0f}")
    print(f"Raw Price Ratio     : {result['raw_price_ratio']:.4f}")
    print(f"Clamped Price Ratio : {result['clamped_price_ratio']:.4f}")
    print(f"Recommended Price   : Rp {result['recommended_price']:,.0f}")
    print(f"Delta vs Base       : Rp {result['delta_rupiah']:+,.0f} "
          f"({result['delta_pct']:+.2f}%)")


if __name__ == "__main__":
    main()
