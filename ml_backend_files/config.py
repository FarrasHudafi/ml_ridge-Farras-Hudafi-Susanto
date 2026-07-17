"""
Konfigurasi global pipeline Ridge Regression untuk dynamic pricing hotel.

File ini memusatkan seluruh parameter eksperimen agar mudah dirujuk dalam
penulisan ilmiah dan direplikasi oleh peneliti lain.

Dataset: Hotel Booking Demand (Antonio, Almeida & Nunes, 2019),
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "hotel_bookings.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Konstruksi target (Price Ratio) ---------------------------------------
# Dataset hanya memiliki satu kolom harga: adr (average daily rate, EUR).
# Price Ratio dibentuk sebagai adr / BASE_REFERENCE, di mana BASE_REFERENCE
# adalah referensi tetap = median adr seluruh reservasi terkonfirmasi (harga
# "normal" pasar). Karena BASE_REFERENCE konstan (bukan turunan per-baris dari
# adr yang sama), pembentukan target ini BEBAS dari kebocoran data.
ADR_COLUMN = "adr"
BASE_REFERENCE = 94.5          # median adr (EUR) pada data terkonfirmasi
PRICE_RATIO_LOWER = 0.3
PRICE_RATIO_UPPER = 3.0

# Kurs untuk penyajian harga pada skala Rupiah. TIDAK memengaruhi model
# (Price Ratio tidak berdimensi); hanya dipakai untuk metrik/tampilan Rupiah.
# GANTI dengan kurs tengah Bank Indonesia pada tanggal tertentu + sitasi.
EXCHANGE_RATE_EUR_IDR = 17000.0

ALPHA_GRID = [
    0.001, 0.01, 0.05, 0.1, 0.3, 0.5,
    1.0, 3.0, 5.0, 10.0, 30.0, 100.0,
]
CV_FOLDS = 5

# --- Fitur model ------------------------------------------------------------
# Room Type & Occupancy Rate DIHAPUS: room type pada dataset ini anonim
# (A-H) dan tidak selaras dengan kelas kamar HotelKu; kelas kamar diwakili
# base price saat inference. Occupancy Rate tidak tersedia pada dataset.
# Fitur libur nasional Indonesia dilepas (data berasal dari hotel di Portugal).
NUMERIC_FEATURES = [
    "Lead Time (days)",
    "Lead Time Norm",
    "Length of Stay",
    "Total Guests",
    "Month Sin",
    "Month Cos",
]

BINARY_FEATURES = [
    "Is Weekend",
    "Is Sunday",
    "Is Midweek",
]

CATEGORICAL_FEATURES = [
    "Segment",
    "Channel",
    "Season",
]

TARGET_COLUMN = "Price Ratio"

# Batas operasional harga rekomendasi (dipakai saat inference di HotelKu):
# harga_rekomendasi = base_price_kamar x clamp(Price Ratio, 0.7, 1.5)
PRICE_RATIO_CLAMP = (0.7, 1.5)
