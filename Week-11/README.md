# Week 11 — Gold XGBoost Regression

Praktikum Machine Learning C8, Universitas Airlangga — Kelompok 1 (ganjil).
Owner tahap Falah: Ahmad Mathlaul Falah (434241123).
Urutan serah terima: Falah (data/XGBoost), Jamal (LightGBM), Azzam (perbandingan/UI/laporan).

Solusi aktif Week-11 menggunakan Gold GLD ETF. Prototipe dataset sebelumnya
tersimpan dalam Git history; implementasi dan artefaknya telah dipensiunkan
dari working tree melalui commit migrasi biasa.

**XGBoost tidak mengungguli persistence pada final test set.**
LightGBM diimplementasikan (Tahap 2); UI belum. Kesepakatan seluruh kelas atas
dataset Gold belum diverifikasi; pemilihan ini adalah keputusan pengguna.

## Dataset dan target

Sumber resmi:
https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset

File lokal: data/raw/FINAL_USO.csv; 1718 baris, 81 kolom.
Rentang tanggal asli: 2011-12-15 sampai 2018-12-31.

SHA-256:
f1a366b4edfbdf074136727f3653058be6d91456e0128a462220797400ae6087

Ini harga **GLD ETF adjusted closing price dalam dataset price units**,
bukan harga emas fisik per gram dan tidak diberi label rupiah.
Detail currency, lisensi, dan point-in-time adjustment masih perlu konfirmasi.

Pada akhir observasi t, prediksi Adj Close pada **next recorded observation t+1**.
Cutoff fitur adalah akhir t. Close dan Adj Close identik dalam CSV, sehingga
Close(t+1) membocorkan target; Close(t) sah digunakan untuk target berikutnya.
Tanggal/indeks disimpan sebagai metadata, bukan fitur ordinal.

## Sebelas fitur historis

| Fitur | Definisi |
|---|---|
| Open, High, Low, Close, Volume | Nilai saat t, tersedia pada cutoff akhir observasi |
| close_lag_1, close_lag_5 | Close(t-1), Close(t-5) |
| close_mean_prev_5 | Mean Close(t-5) sampai Close(t-1), min_periods=5 |
| close_std_prev_5 | Sample std periode yang sama, ddof=1 |
| close_return_1, close_return_5 | Close(t)/Close(t-k)-1, k=1 atau 5 |

Tidak ada centered rolling, future backfill, scaling, atau learned preprocessing.
Data numerik nonfinite ditolak. Fitur eksternal tidak digunakan.

## Gap policy, split, dan CV

Pasangan source-target dengan gap lebih dari 6 hari kalender dikeluarkan:
2013-04-15 ke 2013-05-01 dan 2013-12-13 ke 2014-01-02.
Tidak ada observasi yang direkayasa. Lima baris awal belum memiliki histori
cukup dan baris terakhir tidak memiliki target berikutnya.

1710 pasangan eligible dibagi tanpa shuffle:
- Training: 1367; source 2011-12-22 sampai 2017-08-10.
- Purge: satu source row 1374 (2017-08-11).
- Testing: 342; source 2017-08-14 sampai 2018-12-28.
- Training target terakhir 2017-08-11 mendahului first test cutoff 2017-08-14.

Manifest: results/gold/split_manifest.json.
Fingerprint:
07224b5958086dc8c76fca249c26e960e7748cb21e70c94650ce8d9459f259f2

TimeSeriesSplit(n_splits=5, gap=1) hanya diterapkan pada training.
Setiap fold memvalidasi urutan source/target dan batas informasi.
Test set tidak masuk fitting, tuning, atau pemilihan fitur.
Jumlah observasi per fold sama tidak berarti durasi kalender identik.

## Workflow XGBoost

Pipeline berisi XGBRegressor; tidak ada transformasi fitted tambahan.
Parameter dasar: objective=reg:squarederror, random_state=42,
tree_method=hist, device=cpu, n_jobs=1, importance_type=gain.

RandomizedSearchCV: 12 kandidat, scoring=neg_root_mean_squared_error,
random_state=42, n_jobs=1, refit=True, error_score=raise.
Ruang pencarian:
- n_estimators: [100, 200, 300]
- max_depth: [2, 3, 4]
- learning_rate: [0.03, 0.05, 0.1]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]
- reg_lambda: [1, 5]

60 search CV fits, 5 fit CV deskriptif pada parameter terpilih, dan 1 refit final.
Seluruh CV selesai sebelum final test. Tidak ada tuning ulang berdasarkan test.

Best parameters aktual:
```json
{
  "model__subsample": 1.0,
  "model__reg_lambda": 1,
  "model__n_estimators": 100,
  "model__max_depth": 2,
  "model__learning_rate": 0.1,
  "model__colsample_bytree": 0.8
}
```

## Workflow LightGBM

Pipeline berisi LGBMRegressor; tidak ada transformasi fitted tambahan.
Parameter dasar: objective=regression, random_state=42, n_jobs=1,
deterministic=True, force_row_wise=True, subsample_freq=1,
importance_type=gain, verbose=-1.

RandomizedSearchCV: 12 kandidat, scoring=neg_root_mean_squared_error,
random_state=42, n_jobs=1, refit=True, error_score=raise.
Ruang pencarian:
- n_estimators: [100, 200, 300]
- learning_rate: [0.03, 0.05, 0.1]
- num_leaves: [4, 8, 15]
- max_depth: [-1, 3, 5]
- min_child_samples: [10, 20, 40]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]
- reg_lambda: [0.0, 1.0, 5.0]

60 search CV fits, 5 fit CV deskriptif pada parameter terpilih, dan 1 refit final.
Perintah eksperimen:
```powershell
& './.venv/Scripts/python.exe' -B -m src.train_lightgbm_gold
```
Artefak disimpan di `models/gold/lightgbm_pipeline.joblib`, `results/gold/lightgbm_*`, dan `figures/gold/lightgbm_*`.

Best parameters aktual:
```json
{
  "model__subsample": 1.0,
  "model__reg_lambda": 5.0,
  "model__num_leaves": 8,
  "model__n_estimators": 200,
  "model__min_child_samples": 10,
  "model__max_depth": 3,
  "model__learning_rate": 0.05,
  "model__colsample_bytree": 1.0
}
```

## Hasil aktual

| Model / subset | MAE | MAPE (%) | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| XGBoost training | 0.847445769 | 0.658597558 | 1.395211869 | 1.181190869 | 0.996168582 |
| XGBoost CV mean | 4.099111051 | 3.313673037 | 71.938840521 | 5.406863028 | 0.502982133 |
| XGBoost testing | 0.636764737 | 0.526921073 | 0.646805375 | 0.804242112 | 0.969991780 |
| LightGBM training | 0.779815076 | 0.607158678 | 1.159619746 | 1.076856418 | 0.996815546 |
| LightGBM CV mean | 4.181892029 | 3.377142843 | 75.841660087 | 5.509848611 | 0.483193606 |
| LightGBM testing | 0.647531209 | 0.536290075 | 0.671981204 | 0.819744597 | 0.968823759 |
| Persistence testing | 0.567221865 | 0.468659243 | 0.550177837 | 0.741739736 | 0.974474767 |

MAE/RMSE memakai dataset price units; MSE squared units; R2 unitless.
MAPE = sklearn MAPE × 100. Input nol untuk target MAPE ditolak.
CV adalah mean metrik per fold, bukan pooled out-of-fold.

Persistence memprediksi target berikutnya dengan Close(t), pada test targets identik.
Test RMSE XGBoost 8.426456% lebih tinggi daripada persistence; Test RMSE LightGBM 10.516473% lebih tinggi daripada persistence.
R2 tinggi saja tidak membuktikan nilai tambah forecasting.

Best mean CV RMSE XGBoost 5.406863028 (std 6.534881232); Best mean CV RMSE LightGBM 5.509848611 (std 6.744125471).
RMSE lima fold XGBoost: 18.384214403, 2.869771104, 3.099724435, 1.713239257, 0.967365941.
RMSE lima fold LightGBM: 18.902082288, 2.851969883, 3.171376075, 1.690077427, 0.933737382.
CV sensitif terhadap periode: 140/227 target validation fold pertama berada di
luar rentang target training-nya. Ini petunjuk perubahan distribusi, bukan
bukti penyebab tunggal. Perbandingan formal komparatif antara XGBoost dan LightGBM dilakukan pada Tahap 3 oleh Azzam.

## Environment, verifikasi, dan reproduksi

Gunakan Python 3.14.7 dan requirements.txt yang mem-pin dependency langsung.
Dependency transitif tidak di-lock sepenuhnya; lintas platform tidak dijamin bitwise-identik.

Dari folder Week-11, pada tahap yang mengizinkan setup:
```powershell
& 'C:/Users/falah/AppData/Local/Programs/Python/Python314/python.exe' -m venv .venv
& './.venv/Scripts/python.exe' -m pip install --only-binary=:all: -r requirements.txt
& './.venv/Scripts/python.exe' -B -m pytest tests -v -p no:cacheprovider
& './.venv/Scripts/python.exe' -B -m src.gold_data_pipeline
```

Venv existing jangan ditimpa. Perintah data pipeline hanya menerima manifest
existing jika isi rekonstruksinya identik; manifest berbeda ditolak.

Training default menolak overwrite artefak yang sudah tersedia.
Untuk reproduksi terpisah pada tahap yang mengizinkan training, gunakan
namespace baru di dalam Week-11:
```powershell
& './.venv/Scripts/python.exe' -B -m src.train_xgboost_gold --help
& './.venv/Scripts/python.exe' -B -m src.train_xgboost_gold --output-root results/reproduction
```

Perintah reproduksi training tidak dijalankan pada integrasi ini. Jangan
memakai hasil test untuk mengganti parameter atau mengulang pemilihan model.
Output baru berada di results/reproduction/{models,results,figures}/gold/.
Namespace existing tidak ditimpa. Verifikasi parameter, split, dan metrik;
perbedaan runtime/platform dapat memengaruhi byte model/timing.

Atribut Git Week-11 mempertahankan byte raw dataset dan artefak hasil supaya
hash tidak berubah hanya karena normalisasi newline saat checkout.

## Artefak dan provenance

- results/gold/split_manifest.json: split, exclusions, purge, dan fold boundaries.
- results/gold/xgboost_metrics.json dan .csv: train/CV/test/persistence.
- results/gold/xgboost_cv_results.csv: 12 kandidat dan skor fold.
- results/gold/xgboost_metadata.json: konfigurasi, versi, hash, hasil, keterbatasan.
- results/gold/xgboost_test_predictions.csv: source/target row dan tanggal, prediksi/residual.
- results/gold/xgboost_execution_log.json: bukti eksekusi asli; dipertahankan.
- results/gold/xgboost_training_source.py.txt: source trainer persis saat eksperimen.
- models/gold/xgboost_pipeline.joblib: fitted pipeline Gold.
- figures/gold/: actual vs predicted, residuals, dan normalized-gain importance.

Model SHA-256:
fe5f7a2cc001bc32ae336311c85730e404154ce13ecd1af1a66f3f3f0f3d8425

Integrasi menambah routing output/reproduksi dan menyesuaikan guard preservasi
ke proyek Gold-only; konfigurasi model dan hasil tidak berubah.
trainer_source_sha256 dalam metadata merujuk arsip source eksperimen asli
(.py.txt), bukan wrapper integrasi terbaru. Metadata/log tidak diregenerasi.
Referensi file prototipe lama dalam log adalah bukti historis, bukan dependency aktif.

## Handoff LightGBM

```python
import joblib
from src.gold_data_pipeline import replay_gold_manifest, make_time_series_cv, validate_cv_folds
from src.evaluation import evaluate_regression
split = replay_gold_manifest()
cv = make_time_series_cv()
fold_audit = validate_cv_folds(split)
model = joblib.load("models/gold/xgboost_pipeline.joblib")
predictions = model.predict(split.X_test)
```

Jamal telah menyelesaikan implementasi dan training LightGBM menggunakan manifest, 11 fitur, metrik, persistence baseline, dan
training-only CV yang sama.

Dokumen detail: docs/GOLD_MIGRATION_DESIGN.md, docs/GOLD_XGBOOST_RESULTS.md, dan docs/GOLD_LIGHTGBM_RESULTS.md.

## Keterbatasan

- Next recorded observation bukan jaminan sesi bursa berikutnya.
- Lag/rolling berbasis observasi dapat melintasi gap kalender.
- Data berakhir pada 2018; tidak ada klaim performa pasar terkini.
- Detail currency, lisensi, dan point-in-time adjusted prices belum terverifikasi.
- Kesepakatan kelas atas Gold belum diverifikasi.
- Importance adalah kontribusi prediktif, bukan bukti kausalitas.
