# Hasil Evaluasi Ridge Regression

- Jumlah data train: **1324**
- Jumlah data test:  **332**
- Jumlah fitur (setelah encoding): **36**
- Alpha optimal (RidgeCV): **1.0**

## Skala Price Ratio

| Metrik | Ridge | OLS |
|---|---|---|
| MAE | 0.062648 | 0.062686 |
| MSE | 0.007363 | 0.007385 |
| RMSE | 0.085807 | 0.085937 |
| MAPE | 0.055941 | 0.055972 |
| R2 | 0.877033 | 0.876660 |

## Skala Rupiah (rasio x base price)

| Metrik | Ridge | OLS |
|---|---|---|
| MAE | 41749.70 | 41796.62 |
| MSE | 3340463281.63 | 3350551087.48 |
| RMSE | 57796.74 | 57883.94 |
| MAPE | 0.06 | 0.06 |
| R2 | 0.96 | 0.96 |
