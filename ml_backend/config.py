"""
Konfigurasi global pipeline Ridge Regression untuk dynamic pricing hotel.

File ini memusatkan seluruh parameter eksperimen agar mudah dirujuk dalam
penulisan ilmiah dan direplikasi oleh peneliti lain.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "dataset_hotel_dynamic_pricing.xlsx"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2

PRICE_RATIO_LOWER = 0.3
PRICE_RATIO_UPPER = 3.0

ALPHA_GRID = [
    0.001, 0.01, 0.05, 0.1, 0.3, 0.5,
    1.0, 3.0, 5.0, 10.0, 30.0, 100.0,
]
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "Lead Time (days)",
    "Length of Stay",
    "Lead Time Norm",
    "Occupancy Rate",
    "Month Sin",
    "Month Cos",
    "Total Guests",
]

BINARY_FEATURES = [
    "Is Weekend",
    "Is Sunday",
    "Is Midweek",
    "Is Holiday",
    "Is Near Holiday",
    "Is School Holiday",
]

CATEGORICAL_FEATURES = [
    "Room Type",
    "Segment",
    "Channel",
    "Season",
    "Day Category",
]

TARGET_COLUMN = "Price Ratio"
BASE_PRICE_COLUMN = "Base Price (Rp)"
ACTUAL_PRICE_COLUMN = "Actual Price (Rp)"

PRICE_RATIO_CLAMP = (0.7, 1.5)
