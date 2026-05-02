"""
Pydantic schemas untuk endpoint dynamic pricing.

Schema ini selaras dengan dataset lokal
``dataset_hotel_dynamic_pricing.xlsx``. Field input merupakan informasi
minimum yang dibutuhkan untuk menentukan rasio harga: tanggal check-in,
check-out, tipe kamar, dan harga base. Field opsional (occupancy_rate,
segment, channel, total_guests) diisi default representatif jika tidak
disediakan oleh klien.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


RoomType = Literal["Standard", "Superior", "Deluxe", "Suite"]
Segment = Literal["Leisure", "Business", "Family", "Couple", "Group"]
Channel = Literal[
    "OTA_Traveloka", "Website", "OTA_Booking",
    "Corporate", "Phone", "Walk-in",
]


class BookingRequest(BaseModel):
    """Permintaan prediksi harga.

    - ``check_in`` & ``check_out`` dalam format ISO ``YYYY-MM-DD``.
    - ``base_price`` dalam Rupiah, harga normal kamar (sebelum dynamic pricing).
    - ``occupancy_rate`` 0..1; jika ``None`` server akan memakai proxy
      berbasis kalender (libur/weekend → tinggi, midweek non-libur → rendah).
    """

    check_in: date
    check_out: date
    room_type: RoomType
    base_price: float = Field(gt=0)
    occupancy_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    total_guests: int = Field(default=2, ge=1, le=20)
    segment: Segment = "Leisure"
    channel: Channel = "Website"

    @model_validator(mode="after")
    def _validate_dates(self) -> "BookingRequest":
        if self.check_out <= self.check_in:
            raise ValueError("check_out harus setelah check_in")
        return self


class PredictionResponse(BaseModel):
    predicted_price: float
    base_price: float
    raw_price_ratio: float
    clamped_price_ratio: float
    delta_rupiah: float
    delta_pct: float
    night_date: date
    lead_time_days: int
    length_of_stay: int
    occupancy_rate_used: float
    currency: str
    model_version: str
    status: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
