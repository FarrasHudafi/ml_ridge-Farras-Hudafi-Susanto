"""
Skrip inferensi: penerapan dynamic pricing pada kasus baru.

Setelah model dilatih oleh main.py dan disimpan di outputs/models/,
skrip ini dapat dipakai untuk:
  * Menghitung Price Ratio prediksi untuk satu kombinasi input
  * Menghitung harga rekomendasi (Rp) = Base Price x Price Ratio prediksi
  * Memberikan harga clamp pada batas operasional (default 0.7..1.5)

Contoh pemanggilan dari command line:
  python predict.py \
      --night-date 2026-12-25 --booking-date 2026-05-01 \
      --room-type Deluxe --base-price 950000 \
      --occupancy 0.85 --segment Leisure --channel Website
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


HOLIDAYS_2024_2026 = {
    "2024-01-01","2024-02-08","2024-02-10","2024-03-11","2024-04-10","2024-04-11",
    "2024-05-01","2024-05-23","2024-06-01","2024-06-17","2024-07-07","2024-08-17",
    "2024-09-16","2024-12-24","2024-12-25",
    "2025-01-01","2025-01-27","2025-01-29","2025-03-29","2025-03-31","2025-04-01",
    "2025-05-01","2025-05-12","2025-05-29","2025-06-01","2025-06-06","2025-06-27",
    "2025-08-17","2025-09-05","2025-12-24","2025-12-25",
    "2026-01-01","2026-01-16","2026-02-17","2026-03-19","2026-03-20","2026-03-21",
    "2026-05-01","2026-05-14","2026-05-27","2026-06-01","2026-06-16","2026-08-17",
    "2026-08-26","2026-12-24","2026-12-25",
}

SCHOOL_HOLIDAYS = [
    (date(2024,3,22),  date(2024,3,31)),
    (date(2024,6,14),  date(2024,6,30)),
    (date(2024,12,22), date(2025,1,5)),
    (date(2025,3,21),  date(2025,3,30)),
    (date(2025,6,13),  date(2025,6,29)),
    (date(2025,12,20), date(2026,1,4)),
    (date(2026,3,20),  date(2026,3,29)),
    (date(2026,6,12),  date(2026,6,28)),
    (date(2026,12,21), date(2027,1,3)),
]


def _is_school_holiday(d: date) -> int:
    return int(any(s <= d <= e for s, e in SCHOOL_HOLIDAYS))


def _day_category(night: date, is_holiday: int, is_school: int) -> str:
    dow = night.weekday()
    if is_holiday:
        return "Holiday"
    if is_school:
        return "School_Holiday"
    if dow == 6:
        return "Sunday"
    if dow in (4, 5):
        return "Weekend"
    return "Weekday"


def build_input_row(
    night_date: date,
    booking_date: date,
    room_type: str,
    base_price: float,
    occupancy: float,
    segment: str = "Leisure",
    channel: str = "Website",
    season: str | None = None,
    length_of_stay: int = 1,
    total_guests: int = 2,
) -> pd.DataFrame:
    dow = night_date.weekday()
    js_dow = (dow + 1) % 7
    is_weekend = int(js_dow in (5, 6))
    is_sunday = int(js_dow == 0)
    is_midweek = int(js_dow in (2, 3))
    is_holiday = int(night_date.isoformat() in HOLIDAYS_2024_2026)
    is_near_holiday = int(any(
        abs((night_date - datetime.fromisoformat(h).date()).days) <= 1
        for h in HOLIDAYS_2024_2026
    ))
    is_school = _is_school_holiday(night_date)
    month = night_date.month
    month_sin = math.sin(2 * math.pi * (month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12)
    lead_time = max(0, (night_date - booking_date).days)
    lead_time_norm = min(lead_time, 90) / 90

    if season is None:
        season = _infer_season(month, is_holiday, is_school)

    row = {
        "Lead Time (days)": lead_time,
        "Length of Stay": length_of_stay,
        "Lead Time Norm": lead_time_norm,
        "Occupancy Rate": max(0.0, min(1.0, occupancy)),
        "Month Sin": month_sin,
        "Month Cos": month_cos,
        "Total Guests": total_guests,
        "Is Weekend": is_weekend,
        "Is Sunday": is_sunday,
        "Is Midweek": is_midweek,
        "Is Holiday": is_holiday,
        "Is Near Holiday": is_near_holiday,
        "Is School Holiday": is_school,
        "Room Type": room_type,
        "Segment": segment,
        "Channel": channel,
        "Season": season,
        "Day Category": _day_category(night_date, is_holiday, is_school),
    }
    cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    return pd.DataFrame([row])[cols]


def _infer_season(month: int, is_holiday: int, is_school: int) -> str:
    if is_holiday or is_school or month in (6, 7, 12):
        return "High"
    if month in (1, 2, 11):
        return "Low"
    return "Mid"


def predict_price(
    night_date: date,
    booking_date: date,
    room_type: str,
    base_price: float,
    occupancy: float,
    **kwargs,
) -> dict:
    bundle = joblib.load(MODEL_DIR / "ridge_pipeline.joblib")
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    row = build_input_row(
        night_date=night_date,
        booking_date=booking_date,
        room_type=room_type,
        base_price=base_price,
        occupancy=occupancy,
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
    p.add_argument("--room-type", required=True,
                   choices=["Standard", "Superior", "Deluxe", "Suite"])
    p.add_argument("--base-price", type=float, required=True)
    p.add_argument("--occupancy", type=float, required=True,
                   help="Occupancy rate 0..1")
    p.add_argument("--segment", default="Leisure",
                   choices=["Leisure", "Business", "Family", "Couple", "Group"])
    p.add_argument("--channel", default="Website",
                   choices=["OTA_Traveloka", "Website", "OTA_Booking",
                            "Corporate", "Phone", "Walk-in"])
    p.add_argument("--length-of-stay", type=int, default=1)
    p.add_argument("--total-guests", type=int, default=2)
    return p.parse_args()


def main():
    args = _parse_args()
    result = predict_price(
        night_date=date.fromisoformat(args.night_date),
        booking_date=date.fromisoformat(args.booking_date),
        room_type=args.room_type,
        base_price=args.base_price,
        occupancy=args.occupancy,
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
