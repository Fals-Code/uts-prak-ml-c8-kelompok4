# Catatan Serah Terima (Handoff) Tahap 2: LightGBM — untuk Azzam

**PIC Tahap 2:** Ah. Dliya'ul Adlha Jamalul Lail (434241126)  
**Penerima (Tahap 3):** Azzam (Perbandingan Model, UI Streamlit, Laporan Final)  
**Branch:** `adlha/week-11-lightgbm`  
**Dataset:** Gold GLD ETF (`data/raw/FINAL_USO.csv`)  
**Target Prediksi:** GLD ETF Adjusted Closing Price pada observasi tercatat berikutnya ($t+1$) dalam *dataset price units*.

---

## 1. Daftar Artefak LightGBM & Provenance

Seluruh artefak hasil eksperimen LightGBM disimpan pada namespace terisolasi dan telah diverifikasi melalui unit test read-only audit:

| Tipe Artefak | Path File | Keterangan / Hash |
|---|---|---|
| **Fitted Model** | `models/gold/lightgbm_pipeline.joblib` | SHA-256: `cbf223e3d5733817f7dd849a52a95ca7a999a4c2d32bd1114d02a124de8cf43b` |
| **Metrik JSON** | `results/gold/lightgbm_metrics.json` | Metrik train, 5-fold CV, test, dan persistence |
| **Metrik CSV** | `results/gold/lightgbm_metrics.csv` | Format tabel 5 metrik standar |
| **Hasil CV** | `results/gold/lightgbm_cv_results.csv` | 12 kandidat hyperparameter dan skor fold |
| **Metadata** | `results/gold/lightgbm_metadata.json` | Konfigurasi lengkap, timing, parameter terbaik, provenance |
| **Prediksi Test** | `results/gold/lightgbm_test_predictions.csv` | 342 baris test: source/target dates/rows, actual, pred, residual, baseline |
| **Execution Log** | `results/gold/lightgbm_execution_log.json` | Log eksekusi eksperimen dan audit snapshot |
| **Visualisasi Actual vs Pred** | `figures/gold/lightgbm_actual_vs_predicted.png` | Scatter plot actual vs predicted pada test set |
| **Visualisasi Residuals** | `figures/gold/lightgbm_residuals.png` | Scatter plot residual terhadap prediksi |
| **Visualisasi Importance** | `figures/gold/lightgbm_feature_importance.png` | Bar chart normalized gain feature importance |
| **Arsip Source Trainer** | `results/gold/lightgbm_training_source.py.txt` | SHA-256: `7cb28832522ee9bbc804225e562db1a8a756a2ddd0e2838647feaa7ef3829aca` |

---

## 2. Angka Kunci Perbandingan (XGBoost vs LightGBM vs Persistence)

### A. Metrik Seleksi Utama (Training-only TimeSeriesSplit 5 Folds)
Sesuai kebijakan metodologi, pemilihan model terbaik didasarkan murni pada **mean CV RMSE**:
- **XGBoost mean CV RMSE:** `5.406863028` (std: `6.534881232`)
- **LightGBM mean CV RMSE:** `5.509848611` (std: `6.744125471`)
- **Selisih CV RMSE:** XGBoost memiliki mean CV RMSE lebih rendah sebesar `0.102986` (~`1.87%` lebih baik dari LightGBM pada tahap validasi temporal training).

### B. Hasil Evaluasi Akhir (Holdout Test Set — 342 Observasi)
Holdout test set tidak pernah digunakan dalam proses pemilihan hyperparameter:

| Model / Baseline | MAE | MAPE (%) | MSE | RMSE | $R^2$ | Keterangan |
|---|---:|---:|---:|---:|---:|---|
| **Persistence Baseline** | **0.567222** | **0.468659** | **0.550178** | **0.741740** | **0.974475** | Prediksi $t+1$ dengan $\text{Close}(t)$ |
| **XGBoost** | 0.636765 | 0.526921 | 0.646805 | 0.804242 | 0.969992 | Test RMSE +8.43% vs persistence |
| **LightGBM** | 0.647531 | 0.536290 | 0.671981 | 0.819745 | 0.968824 | Test RMSE +10.52% vs persistence |

**Temuan Kunci untuk Bagian Perbandingan & Laporan:**
1. Baik XGBoost maupun LightGBM **tidak mengungguli** persistence baseline pada test set holdout.
2. Selisih RMSE LightGBM terhadap XGBoost pada test set sangat tipis: `0.819745` vs `0.804242` (selisih `0.015502` atau `1.93%`).
3. Keduanya menunjukkan fenomena di mana fitur harga observasi terakhir ($\text{Close}$, $\text{High}$, $\text{Low}$) mendominasi kontribusi gain (>99%), namun lag/rolling historis tidak mampu memberikan alpha prediktif melampaui naive persistence.

---

## 3. Cara Memuat Model untuk Aplikasi Streamlit

Model LightGBM disimpan dalam format `Pipeline` scikit-learn lengkap:

```python
import joblib
import pandas as pd
from src.gold_data_pipeline import FEATURES

# 1. Muat pipeline yang sudah difit
model = joblib.load("models/gold/lightgbm_pipeline.joblib")

# 2. Pastikan input DataFrame memiliki 11 kolom persis sesuai skema FEATURES (urutan sama)
# FEATURES = ('Open', 'High', 'Low', 'Close', 'Volume', 'close_lag_1',
#             'close_lag_5', 'close_mean_prev_5', 'close_std_prev_5',
#             'close_return_1', 'close_return_5')
X_input = df[list(FEATURES)]

# 3. Lakukan inferensi
y_pred = model.predict(X_input)
```

Untuk membandingkan kedua model di Streamlit secara berdampingan:
- XGBoost: `joblib.load("models/gold/xgboost_pipeline.joblib")`
- LightGBM: `joblib.load("models/gold/lightgbm_pipeline.joblib")`
- Persistence baseline: `df["Close"].to_numpy()`

---

## 4. Kebijakan Seleksi Model untuk Tahap 3 (Azzam)

1. **Kriteria Objektif:**
   - Evaluasi model terbaik harus mengacu pada **Mean CV RMSE** pada data training (`XGBoost 5.406863` vs `LightGBM 5.509849`).
   - Tie-break: Mean CV $R^2$ lalu Mean CV MAE.
   - Final test set hanya berfungsi sebagai evaluasi generalisasi out-of-sample di masa depan, bukan feedback loop untuk memilih model.
2. **Kejujuran Ilmiah:**
   - Wajib mencantumkan bahwa model ML tidak mengalahkan naive persistence ($y_{t+1} \approx \text{Close}_t$). Tingginya nilai $R^2$ (>0.96) disebabkan oleh sifat non-stasioner data harga deret waktu finansial (random walk-like), bukan bukti keunggulan prediksi tren.

---

## 5. Konfirmasi Integritas Workspace

Telah diverifikasi melalui `git diff main --stat`:
- Dataset mentah `data/raw/FINAL_USO.csv` **tidak disentuh**.
- Manifest split `results/gold/split_manifest.json` dan skema 11 fitur **identik 100%**.
- Kode dan seluruh artefak XGBoost milik Falah **tidak berubah sedikit pun**.
- Suite unit test: **60 passed (100% PASS)** tanpa warning/error.
- Branch `adlha/week-11-lightgbm` siap untuk review dan integrasi UI Streamlit oleh Azzam.
