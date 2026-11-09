# Week 11 — TM11 Regresi

Kelompok 1 (ganjil), Kelas C8 — Praktikum Machine Learning, Universitas Airlangga.
Algoritma tugas: XGBoost dan LightGBM. Target: memprediksi **Body Mass (g)**.
Tahap 3A menyediakan data, preprocessing, dan split. Tahap 3B menambahkan evaluasi bersama dan eksperimen XGBoost; LightGBM belum diimplementasikan.

## Dataset dan integritas

Sumber lokal read-only: `../Week-10/data/raw/Penguins.csv` (Palmer Penguins).
Dataset sekunder online yang disepakati kelas; bibliografi dan tautan sumber
spesifik belum diverifikasi. File tidak disalin, dipindahkan, atau diubah.

SHA-256: `5ea908ae6fd5363b6d649f9eecdd63c010caaa389ee9498483e9ec30647dee2d`.

Pembaca memverifikasi hash sebelum parsing, memakai `encoding="utf-8-sig"`,
menormalisasi header dengan `strip()`, menolak duplikasi header setelah
normalisasi, dan memvalidasi semua kolom wajib. Nilai numerik nonmissing yang
tidak dapat dikonversi atau tidak finite menyebabkan error.

## Target dan fitur

Target: `Body Mass (g)`; target tidak pernah menjadi fitur.
Urutan fitur selalu:

1. Culmen Length (mm)
2. Culmen Depth (mm)
3. Flipper Length (mm)
4. Species
5. Island
6. Sex

Hanya enam fitur tersebut yang dipilih. Identifier, komentar, tanggal,
koordinat, isotop, dan metadata lainnya tidak digunakan. Indeks baris sumber
dipertahankan sebagai index DataFrame, bukan sebagai kolom fitur.

## Kebijakan missing value dan preprocessing

- Target kosong: keluarkan baris dari supervised dataset, tanpa imputasi target.
- Numerik: `SimpleImputer(strategy="median", keep_empty_features=True)`.
  Median dipelajari hanya ketika pipeline di-fit pada fold training.
- Kategorikal: strip string; string kosong menjadi missing. Sex dinormalisasi
  uppercase; token titik juga menjadi missing. Missing diwakili `numpy.nan`
  agar konsisten dengan scikit-learn.
- Pipeline kategorikal: `SimpleImputer(strategy="constant",
  fill_value="Unknown", keep_empty_features=True)`, lalu
  `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Unknown bukan MALE/FEMALE. Kategori baru saat transform diabaikan encoder;
  lebar fitur hasil transform tetap mengikuti vocabulary train.
- Tidak memakai StandardScaler. `remainder="drop"` mempertahankan allowlist fitur.
- Jika suatu fitur numerik seluruhnya kosong dalam fold training,
  `keep_empty_features=True` mempertahankan kolom dengan fallback nol dari
  SimpleImputer. Kondisi ini tidak ditemukan pada fitur numerik data valid aktual.

`make_preprocessor()` selalu mengembalikan transformer **belum fitted**.
Pasang transformer itu pada Pipeline estimator nanti, sehingga setiap fold
cross-validation melakukan fitting preprocessing sendiri. Jangan fit
preprocessor pada seluruh dataset atau seluruh training sebelum tuning CV.
Pada tahap ini, fitting transformer hanya dijalankan pada fixture unit test.

## Split dan hasil eksekusi aktual

`train_test_split(test_size=0.20, random_state=42, shuffle=True)`.

- Data asli: 344 baris × 19 kolom; BOM UTF-8.
- Target valid: 342 baris; 2 baris target kosong dikeluarkan.
- Training: 273 baris; testing: 69 baris (pembulatan ukuran test oleh scikit-learn).
- Missing numerik awal: masing-masing 2 untuk tiga fitur numerik.
  Setelah filtering target: masing-masing 0.
- Sex awal: 10 kosong dan 1 titik. Setelah filtering target dan normalisasi:
  9 Sex missing dipertahankan; akan diimputasi Unknown saat fitting pipeline.
- Split disjoint, mencakup setiap baris eligible tepat sekali, dan urutan
  indeks dapat direproduksi serta diputar ulang dari manifest.
- SHA-256 split terurut:
  `fd37f6636ccce38aa9f7f6b6ab9b3ccd53f269a0e83ad60e032efc471ee0b319`.

Manifest: `results/split_manifest.json`. Berisi hash dataset, fitur/target,
kebijakan missing, jumlah aktual, serta indeks train/test terurut.
Indeks adalah posisi baris data CSV **zero-based, tidak menghitung header**.
Tidak ada data mentah lengkap atau hasil model disimpan di manifest.

Test set tidak digunakan untuk fitting preprocessing, tuning, pemilihan
fitur, atau penyesuaian model. Hasil XGBoost tahap 3B tersedia pada bagian berikut.

## Cara menjalankan

Jalankan dari folder Week-11 menggunakan environment khusus:

```powershell
& './.venv/Scripts/python.exe' -B -m src.data_pipeline
& './.venv/Scripts/python.exe' -B -m pytest tests -v -p no:cacheprovider
```

Perintah pertama memverifikasi dataset, membuat split dan manifest, serta
memeriksa reproducibility dan replay. Perintah itu tidak fit preprocessor
atau model. Manifest dihasilkan deterministik tanpa timestamp.

## Antarmuka serah terima

Dari working directory Week-11:

```python
from src.data_pipeline import load_dataset, split_from_manifest, make_preprocessor

dataset = load_dataset()
split = split_from_manifest(dataset)
preprocessor = make_preprocessor()  # belum fitted
# split.X_train/X_test dan y_train/y_test masih dalam satuan asli.
# Pasang preprocessor di dalam Pipeline model pada tahap berikutnya.
```

`load_dataset()` menghasilkan fitur terpilih dan target valid.
`split_dataset()` membuat split baru dengan default yang disepakati.
`split_from_manifest()` memakai indeks tersimpan tanpa resplit dan memvalidasi
hash dataset, skema, cakupan/overlap indeks, jumlah, serta hash split.

Unit test menggunakan fixture sintetis kecil untuk memeriksa perilaku
fungsi, bukan sebagai hasil eksperimen penelitian. Eksekusi tahap 3A:
**15 test lulus** pada tahap 3A. Hasil metrik, tuning, dan model XGBoost tahap 3B dijelaskan di bawah.

## Keterbatasan

Random row split berpotensi memasukkan observasi individu yang berkorelasi
ke train dan test jika identifier penguin ternyata merepresentasikan individu
berulang. Risiko tersebut belum dibuktikan; perlu analisis identitas tambahan.
Pemeriksaan tahap ini tidak mengklaim seluruh bentuk leakage sudah dieliminasi.

[Kembali ke halaman utama](../README.md)

## Eksperimen XGBoost — Tahap 3B

Pipeline bersama: preprocessing dari `make_preprocessor()`, kemudian
`XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1,
tree_method="hist", importance_type="gain")`. Tidak ada perubahan fitur,
dataset, preprocessor bersama, atau manifest tahap 3A.

### Prosedur training dan seleksi

- Replay split dari manifest: 273 training dan 69 testing, tanpa resplit.
- RandomizedSearchCV: 12 kandidat, random_state=42, n_jobs=1, refit=True,
  error_score="raise", scoring="neg_root_mean_squared_error".
- KFold: 5 fold, shuffle=True, random_state=42.
- Ruang pencarian: n_estimators=[100,200], max_depth=[2,3,4],
  learning_rate=[0.05,0.10], subsample=[0.8,1.0],
  colsample_bytree=[0.8,1.0], reg_lambda=[1.0,5.0].
  Seluruh parameter search memakai prefix `model__`.
- Preprocessing berada di dalam Pipeline CV dan di-fit ulang pada setiap
  fold training. Test set tidak masuk fitting, tuning, atau early stopping.
- Semua 60 fit CV berhasil, diikuti satu refit kandidat terbaik pada training.
- Test set dievaluasi final setelah pemilihan parameter. Tidak ada tuning
  ulang berdasarkan hasil test.

Best parameters aktual:

```json
{
  "model__subsample": 0.8,
  "model__reg_lambda": 5.0,
  "model__n_estimators": 100,
  "model__max_depth": 3,
  "model__learning_rate": 0.05,
  "model__colsample_bytree": 0.8
}
```

Best mean CV RMSE: **307.90695043 g**; standar deviasi antar-fold: **24.93766525 g**.

### Evaluasi bersama dan hasil aktual

Fungsi `src.evaluation.evaluate_regression(y_true, y_pred)` digunakan bersama
untuk XGBoost dan LightGBM nantinya. Keys: MAE, MAPE, MSE, RMSE, R2.

- MAE: mean_absolute_error, satuan g.
- MAPE: mean_absolute_percentage_error × 100, satuan persen (%).
- MSE: mean_squared_error, satuan g².
- RMSE: sqrt(MSE), satuan g.
- R2: r2_score, tanpa satuan.
- Input harus 1D numerik finite, panjang sama, minimal dua observasi.
  NaN/inf, target nol (MAPE), dan target konstan (R2 tidak terdefinisi)
  ditolak secara eksplisit.

Angka berikut dibaca dari file eksperimen, ditampilkan dengan pembulatan:

| Subset | MAE (g) | MAPE (%) | MSE (g²) | RMSE (g) | R² |
|---|---:|---:|---:|---:|---:|
| training | 180.73212855 | 4.42468981 | 53305.70636660 | 230.88028579 | 0.91590757 |
| testing | 259.32660708 | 6.77916850 | 103505.60557528 | 321.72287077 | 0.84433330 |

Nilai penuh tersedia dalam JSON/CSV. Ini hasil XGBoost saja, bukan keputusan
model terbaik antaralgoritma. LightGBM merupakan pekerjaan Jamal selanjutnya.
Estimasi test set kecil memiliki ketidakpastian; R² training lebih tinggi
dari testing tidak menjadi alasan melakukan tuning ulang pada test.

### Artefak dan grafik

- `results/xgboost_metrics.json` dan `xgboost_metrics.csv`: metrik train/test.
- `results/xgboost_cv_results.csv`: 12 kandidat beserta skor setiap fold.
- `results/xgboost_metadata.json`: parameter, konfigurasi, versi, hash,
  reproducibility, hasil, dan catatan koreksi serialisasi.
- `results/xgboost_test_predictions.csv`: source_row, actual_g, predicted_g,
  residual_g; urutan 69 indeks sama dengan manifest, tanpa identitas penguin.
- `models/xgboost_pipeline.joblib`: seluruh fitted preprocessing dan estimator.
- `figures/xgboost_actual_vs_predicted.png`: scatter test dan garis ideal y=x.
- `figures/xgboost_residuals.png`: residual aktual − prediksi terhadap prediksi,
  dengan garis nol.
- `figures/xgboost_feature_importance.png`: importance gain untuk 12 fitur
  hasil encoding, memakai nama fitur dari fitted preprocessor.
  Importance adalah kontribusi prediktif, bukan bukti hubungan kausal.

Ketiga PNG adalah grafik hasil program dari eksperimen nyata, bukan screenshot
training/UI. Prediksi reload joblib sama persis dengan sebelum penyimpanan.

Pemeriksaan awal roundtrip CSV menemukan pembulatan float32 pada kolom
prediksi. Serialisasi diperbaiki dengan kolom float64 dan test regresi khusus.
CSV diperbaiki dari model tersimpan tanpa retraining, retuning, atau
penghitungan ulang metrik test. Model dan metrik tidak berubah.

SHA-256 model tersimpan:

`fe82491c0e79b9015ed0c4ebc4792732d36bac77a1102fd3c1a7db5ff6cdb980`

### Menjalankan, memuat model, dan serah terima

Dari folder Week-11:

```powershell
& './.venv/Scripts/python.exe' -B -m src.train_xgboost
& './.venv/Scripts/python.exe' -B -m pytest tests -v -p no:cacheprovider
```

Training menolak menimpa artefak yang sudah ada. Untuk reproduksi terpisah
pada tahap yang mengizinkan eksperimen ulang, gunakan
`--output-root results/reproduction`; seluruh output tetap di bawah Week-11.
Reproduksi memakai manifest yang sama. Jangan menjalankan pencarian baru
untuk memilih parameter berdasarkan hasil testing.

Memuat model tidak memerlukan training ulang:

```python
import joblib
from src.data_pipeline import load_dataset, split_from_manifest

split = split_from_manifest(load_dataset())
pipeline = joblib.load("models/xgboost_pipeline.joblib")
predictions = pipeline.predict(split.X_test)
```

Jamal menggunakan `src/data_pipeline.py`, manifest yang sama, dan
`src/evaluation.py`; preprocessing harus tetap di dalam Pipeline LightGBM CV.
Model joblib dan hasil masih lokal/untracked; belum ada commit atau push.
Seluruh **36 unit test** Week-11 lulus, termasuk 15 test preprocessing.
Seed, single-thread CPU, dan versi library disimpan untuk reproduksi;
identitas bitwise lintas platform tidak dijamin.
