"""
HTTP server FastAPI untuk dynamic pricing.

Endpoint:
  GET  /health     -> status server + apakah model sudah dimuat.
  POST /predict    -> menerima BookingRequest (skema lokal Indonesia)
                      dan mengembalikan harga rekomendasi dalam Rupiah.

Schema input/output sepenuhnya berbasis dataset lokal
``dataset_hotel_dynamic_pricing.xlsx``. Server memanggil utility
``predict_price`` di ``predict.py`` agar logika inferensi (feature
engineering, clamp rasio, pembulatan harga) konsisten antara CLI dan
HTTP API.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.schemas import (  # noqa: E402
    BookingRequest,
    HealthResponse,
    PredictionResponse,
)
from config import MODEL_DIR, PRICE_RATIO_CLAMP  # noqa: E402
from predict import build_input_row  # noqa: E402

MODEL_VERSION = "ridge-id-v1"

app = FastAPI(
    title="Hotel Dynamic Pricing — Ridge Regression",
    version=MODEL_VERSION,
    description=(
        "API prediksi harga booking hotel menggunakan algoritma Ridge "
        "Regression. Skema input langsung berbasis dataset lokal "
        "(dataset_hotel_dynamic_pricing.xlsx)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_state: dict = {"preprocessor": None, "model": None, "best_alpha": None}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _occupancy_proxy(night_date, is_weekend: int, is_holiday: int,
                     is_school: int, lead_time_norm: float) -> float:
    """Estimasi occupancy_rate ketika klien tidak mengirim datanya.

    Bobot per faktor sengaja kecil karena model sudah memiliki koefisien
    sendiri untuk Is Holiday / Is Weekend / Is School Holiday. Memompa
    occupancy lewat sinyal kalender yang sama menyebabkan double-counting
    dan extrapolasi ke ekor distribusi training (>0.9). Median occupancy
    pada training = 0.537, 75th percentile = 0.713 — proxy ini mencoba
    tetap berada di rentang itu.
    """
    occ = 0.50
    if is_holiday:
        occ += 0.10
    if is_school:
        occ += 0.05
    if is_weekend:
        occ += 0.08
    occ += min(0.05, lead_time_norm * 0.05)
    return float(min(0.80, max(0.30, occ)))


@app.on_event("startup")
def load_model() -> None:
    bundle_path = MODEL_DIR / "ridge_pipeline.joblib"
    if not bundle_path.exists():
        print(f"[WARN] Model bundle belum ada di {bundle_path}. "
              f"Jalankan `python main.py` lebih dahulu.")
        return
    bundle = joblib.load(bundle_path)
    _state["preprocessor"] = bundle["preprocessor"]
    _state["model"] = bundle["model"]
    _state["best_alpha"] = bundle.get("best_alpha")
    print(f"[OK] Model dimuat dari {bundle_path}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _state["preprocessor"] is not None and _state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        timestamp=_now_iso(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: BookingRequest) -> PredictionResponse:
    if _state["model"] is None or _state["preprocessor"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum dimuat. Jalankan training (python main.py) dahulu.",
        )

    night_date = payload.check_in
    booking_date = datetime.now(timezone.utc).date()
    if booking_date > night_date:
        booking_date = night_date
    length_of_stay = max(1, (payload.check_out - payload.check_in).days)
    lead_time_days = max(0, (night_date - booking_date).days)

    occupancy = payload.occupancy_rate
    if occupancy is None:
        row_for_calendar = build_input_row(
            night_date=night_date,
            booking_date=booking_date,
            room_type=payload.room_type,
            base_price=payload.base_price,
            occupancy=0.5,
            segment=payload.segment,
            channel=payload.channel,
            length_of_stay=length_of_stay,
            total_guests=payload.total_guests,
        )
        occupancy = _occupancy_proxy(
            night_date=night_date,
            is_weekend=int(row_for_calendar["Is Weekend"].iloc[0]),
            is_holiday=int(row_for_calendar["Is Holiday"].iloc[0]),
            is_school=int(row_for_calendar["Is School Holiday"].iloc[0]),
            lead_time_norm=float(row_for_calendar["Lead Time Norm"].iloc[0]),
        )

    row = build_input_row(
        night_date=night_date,
        booking_date=booking_date,
        room_type=payload.room_type,
        base_price=payload.base_price,
        occupancy=occupancy,
        segment=payload.segment,
        channel=payload.channel,
        length_of_stay=length_of_stay,
        total_guests=payload.total_guests,
    )

    X = _state["preprocessor"].transform(row)
    raw_ratio = float(_state["model"].predict(X)[0])

    lo, hi = PRICE_RATIO_CLAMP
    clamped_ratio = max(lo, min(hi, raw_ratio))
    predicted_price = round(payload.base_price * clamped_ratio, -3)
    delta_rupiah = predicted_price - payload.base_price
    delta_pct = (delta_rupiah / payload.base_price) * 100

    return PredictionResponse(
        predicted_price=predicted_price,
        base_price=payload.base_price,
        raw_price_ratio=round(raw_ratio, 6),
        clamped_price_ratio=round(clamped_ratio, 6),
        delta_rupiah=delta_rupiah,
        delta_pct=round(delta_pct, 2),
        night_date=night_date,
        lead_time_days=lead_time_days,
        length_of_stay=length_of_stay,
        occupancy_rate_used=round(float(occupancy), 4),
        currency="IDR",
        model_version=MODEL_VERSION,
        status="ok",
        timestamp=_now_iso(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
