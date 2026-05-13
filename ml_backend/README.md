# Penerapan Dynamic Pricing menggunakan Algoritma Ridge Regression untuk Penetapan Harga Booking Hotel

Proyek ini merupakan implementasi *machine learning backend* untuk penulisan
ilmiah berjudul **"Penerapan Dynamic Pricing menggunakan Algoritma Ridge
Regression untuk Penetapan Harga Booking Hotel"**. Backend ini melatih model
Ridge Regression pada dataset historis reservasi hotel dan menghasilkan
rekomendasi harga (Rp) untuk setiap kombinasi tipe kamar, tanggal menginap,
lead time, segmen tamu, dan tingkat hunian (occupancy).

## 1. Latar Belakang Singkat

*Dynamic pricing* adalah strategi penetapan harga yang dapat berubah-ubah
sesuai kondisi permintaan dan supply. Pada industri perhotelan, harga kamar
ideal merupakan fungsi kompleks dari hari dalam minggu, hari libur, lead
time pemesanan, musim, tipe kamar, dan tingkat hunian saat ini. Regresi linear
biasa rentan terhadap *multicollinearity* karena banyak fitur yang saling
berkorelasi (misal hari libur sekolah & hari libur nasional). **Ridge
Regression** menambahkan penalti L2 pada bobot:

```
L(w) = Σ (yᵢ − xᵢᵀ w)² + α · ‖w‖₂²
```

Penalti ini menstabilkan koefisien dan menurunkan varian prediksi, yang
membuatnya cocok untuk skenario fitur saling berkorelasi.

## 2. Struktur Direktori

```
ml_backend/
├── dataset_hotel_dynamic_pricing.xlsx   # data primer
├── requirements.txt
├── config.py                            # konstanta global
├── data_loader.py                       # load + cleaning
├── feature_engineering.py               # one-hot + scaling
├── model.py                             # Ridge + RidgeCV
├── evaluation.py                        # metrik & visualisasi
├── predict.py                           # CLI inferensi harga
├── main.py                              # pipeline end-to-end
└── outputs/
    ├── figures/                         # grafik untuk paper
    ├── models/ridge_pipeline.joblib     # model siap pakai
    └── reports/                         # metrics.json + metrics.md
```

## 3. Instalasi

```powershell
cd ml_backend
python -m pip install -r requirements.txt
```

## 4. Menjalankan Pipeline Pelatihan

```powershell
python main.py
```

Output yang dihasilkan:

| Lokasi | Deskripsi |
|---|---|
| `outputs/figures/01_distribusi_price_ratio.png` | Distribusi target Price Ratio |
| `outputs/figures/02_rata_rata_per_hari.png` | Rata-rata Price Ratio per hari |
| `outputs/figures/03_korelasi_fitur.png` | Korelasi fitur numerik |
| `outputs/figures/04_kurva_alpha.png` | Kurva tuning hyperparameter |
| `outputs/figures/05_aktual_vs_prediksi.png` | Scatter aktual vs prediksi |
| `outputs/figures/06_residual.png` | Plot residual |
| `outputs/figures/07_koefisien_ridge.png` | Top koefisien Ridge |
| `outputs/figures/08_ringkasan_metrik_ridge.png` | Ringkasan MAE / RMSE / MAPE (Ridge) |
| `outputs/reports/metrics.json` | Ringkasan numerik metrik |
| `outputs/reports/metrics.md` | Ringkasan markdown untuk paper |
| `outputs/models/ridge_pipeline.joblib` | Pipeline preprocessor + model |

## 5. Inferensi (Penerapan Dynamic Pricing)

### 5.1 CLI

```powershell
python predict.py `
  --night-date 2026-12-25 --booking-date 2026-05-01 `
  --room-type Deluxe --base-price 950000 `
  --occupancy 0.85 --segment Leisure --channel Website
```

Contoh keluaran:

```
=== Hasil Dynamic Pricing ===
Base Price          : Rp 950,000
Raw Price Ratio     : 1.2143
Clamped Price Ratio : 1.2143
Recommended Price   : Rp 1,154,000
Delta vs Base       : Rp +204,000 (+21.47%)
```

### 5.2 HTTP API (untuk frontend Next.js bookhotel/)

Server FastAPI di [api/server.py](api/server.py) mengekspos endpoint
`/health` dan `/predict` di port 8000 — sesuai dengan kontrak
[bookhotel/lib/ml-client.ts](../bookhotel/lib/ml-client.ts).

Cara menjalankan:

```powershell
# pastikan main.py sudah dijalankan dulu agar model tersimpan di outputs/
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

atau langsung:

```powershell
python api/server.py
```

Verifikasi:

```powershell
# health check
curl http://localhost:8000/health

# contoh prediksi (skema lokal)
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "@samples/predict_request_christmas_deluxe.json"
```

Setelah server berjalan, buka frontend di `http://localhost:3000/predict`.
Banner merah "Tidak dapat terhubung ke server prediksi" akan hilang dan
estimasi harga muncul setelah form di-submit.

**Skema request `POST /predict`:**

| Field | Tipe | Keterangan |
|---|---|---|
| `check_in` | `string` (yyyy-mm-dd) | Tanggal mulai menginap |
| `check_out` | `string` (yyyy-mm-dd) | Tanggal selesai menginap |
| `room_type` | `Standard\|Superior\|Deluxe\|Suite` | Tipe kamar |
| `base_price` | `number` (IDR) | Harga normal kamar |
| `occupancy_rate` | `number` 0..1 (opsional) | Jika kosong, server pakai proxy |
| `total_guests` | `int` (opsional, default 2) | Total tamu |
| `segment` | `Segment` (opsional) | Default `Leisure` |
| `channel` | `Channel` (opsional) | Default `Website` |

**Alur end-to-end:**

```
[bookhotel /predict form]
        |  BookingRequest (skema lokal Indonesia)
        v
[FastAPI /predict] -> build_input_row -> Ridge.predict -> clamp 0.7..1.5
        |  PredictionResponse {predicted_price (IDR), price_ratio, dst.}
        v
[hasil estimasi rupiah ditampilkan langsung di UI]
```

## 6. Variabel pada Model

**Target**: `Price Ratio = Actual Price / Base Price`

**Fitur numerik** (distandarisasi dengan StandardScaler):
- Lead Time (days)
- Length of Stay
- Lead Time Norm
- Occupancy Rate
- Month Sin / Month Cos (encoding siklik bulan)
- Total Guests

**Fitur biner** (passthrough):
- Is Weekend, Is Sunday, Is Midweek
- Is Holiday, Is Near Holiday, Is School Holiday

**Fitur kategorikal** (one-hot encoding):
- Room Type, Segment, Channel, Season, Day Category

## 7. Metodologi Pelatihan

1. **Cleaning**: hanya status `Confirmed`, base price > 0, dan
   `Price Ratio ∈ [0.3, 3.0]` untuk membuang outlier.
2. **Split**: 80% train / 20% test, `random_state = 42`.
3. **Standarisasi**: fitur numerik → mean 0, std 1.
4. **Tuning α**: grid `[0.001 … 100]` dengan 5-fold cross-validation
   (`RidgeCV`, scoring = neg-MSE).
5. **Evaluasi**: MAE, MSE, RMSE, MAPE, R² baik pada skala rasio maupun skala
   rupiah (rasio × base price) pada test set.

## 8. Replikasi

Pipeline bersifat deterministik (`RANDOM_STATE = 42`), sehingga setiap
eksekusi menghasilkan metrik dan figur yang persis sama. Anda dapat
mengubah parameter di `config.py` untuk eksperimen tambahan
(misalnya menambah `ALPHA_GRID`, mengganti fitur, atau menyesuaikan
batas clamp Price Ratio).
