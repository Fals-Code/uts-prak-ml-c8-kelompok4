# Week 10 — Clustering

**Kelompok C8 — Week 10**

## Soal praktikum

> Buatlah program clustering pada dataset Mall Customer minggu kemarin dengan menggunakan metode clustering yang lain: Fuzzy C-Means dan DBSCAN!
>
> 1. Lakukan analisa apakah dengan metode lain tersebut nilai akurasi menjadi lebih baik dibandingkan dengan metode K-Means!
> 2. Tugas dikumpulkan pada Rabu, 4 November 2026 pukul 23.59 WIB.

Clustering adalah unsupervised learning. Accuracy klasifikasi tidak langsung berlaku tanpa label dan definisi pencocokan kelas. Karena itu evaluasi utama memakai metrik internal; untuk Penguins tersedia evaluasi eksternal menggunakan Species.

## Tujuan

Membandingkan metode clustering (**K-Means**, **Fuzzy C-Means**, dan **DBSCAN**) pada Mall Customer dan Palmer Penguins dengan preprocessing terstandardisasi dan evaluasi yang konsisten. Pipeline integratif, modul evaluasi, unit test, dan artefak visualisasi perbandingan telah lengkap tersedia.

## Dataset

| Dataset | Path lokal | Dimensi mentah | Fitur clustering | Catatan | SHA-256 |
|---|---|---:|---|---|---|
| Mall Customer | `data/raw/Mall_Customers.csv` | 200 × 5 | Age, Annual Income (k$), Spending Score (1-100) | CustomerID tidak digunakan; tidak ada missing | `416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77` |
| Palmer Penguins | `data/raw/Penguins.csv` | 344 × 19 | Culmen Length, Culmen Depth, Flipper Length, Body Mass | Species hanya evaluasi eksternal; URL dan lisensi Kaggle spesifik perlu konfirmasi | `5ea908ae6fd5363b6d649f9eecdd63c010caaa389ee9498483e9ec30647dee2d` |

Kedua CSV mentah bersifat lokal dan tidak tercatat di Git. Mall berasal dari file Week 9. Penguins mempertahankan header asli Kaggle.

## Struktur Direktori

- `src/preprocessing.py` — pemilihan fitur, validasi, dan StandardScaler.
- `src/pipeline.py` — runner pipeline terpadu (`kmeans`, `fcm`, `dbscan`).
- `src/fcm.py` — implementasi Fuzzy C-Means mandiri berbasis NumPy/SciPy dengan metrik PC dan PE.
- `src/dbscan_runner.py` — runner DBSCAN, analisis noise, dan perhitungan k-distance.
- `src/tune_fcm.py` — modul grid search hyperparameter FCM ($c \in [2, 10], m \in [1.25, 2.5]$).
- `src/tune_dbscan.py` — modul grid search hyperparameter DBSCAN ($\text{eps}, \text{min\_samples}$).
- `src/generate_figures.py` — pembuat gambar visualisasi perbandingan 2D, k-distance plot, dan kurva metrik.
- `src/evaluation.py` — metrik internal (Silhouette, DB, CH), eksternal (ARI, NMI, Purity), fuzzy (PC, PE), dan noise handling.
- `tests/test_pipeline.py` — unittest integritas baseline pipeline.
- `tests/test_fcm.py` — unittest sifat keanggotaan, PC/PE bounds, reproduksibilitas FCM.
- `tests/test_edge_cases.py` — unittest skenario 100% noise, klaster tunggal, dan penanganan N/A.
- `docs/` — dokumentasi audit pipeline dan catatan parameter (`AUDIT_PIPELINE.md`, `PARAMETER_NOTES.md`).
- `figures/` — visualisasi komparatif 2D resolusi tinggi ($\ge 150$ DPI).
- `results/` — tabel ringkasan metrik komparasi, membership matrix FCM, pencarian parameter, dan metadata eksekusi.

## Preprocessing

Mall memakai tiga fitur numerik; CustomerID dan Gender tidak masuk model. Semua 200 sampel digunakan (reject missing). Penguins mengonversi empat ukuran tubuh Kaggle menjadi numerik dan memakai complete-case: 2 dari 344 baris dikeluarkan dari data kerja karena setidaknya satu fitur ukuran tubuh hilang (342 baris valid). Kedua dataset memakai `StandardScaler`. CSV mentah tidak pernah diubah.

Satu fungsi `prepare_dataset(dataset_name)` menyiapkan matriks fitur yang sama untuk seluruh metode clustering tanpa data leakage.

## Konfigurasi Parameter Terpilih

Seluruh parameter hyperparameter dipilih secara murni unsupervised menggunakan metrik internal tanpa menyentuh ground truth:
- **Mall Customer ($N=200$):**
  - K-Means: $k=6$, `random_state=42`
  - FCM: $c=6, m=1.25$, `random_state=42`
  - DBSCAN: $\text{eps}=0.45, \text{min\_samples}=4$, metric Euclidean
- **Palmer Penguins ($N=342$):**
  - K-Means: $k=2$, `random_state=42`
  - FCM: $c=2, m=1.25$, `random_state=42`
  - DBSCAN: $\text{eps}=0.60, \text{min\_samples}=6$, metric Euclidean

## Hasil Komparasi Lengkap (3 Algoritma × 2 Dataset)

### 1. Mall Customer
| Metode | Konfigurasi | Klaster | Cakupan | Silhouette | Davies-Bouldin | Calinski-Harabasz | Partition Coeff (PC) | Partition Entropy (PE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **K-Means** | $k=6$ | 6 | 100.0% | 0.428417 | 0.825354 | 135.102104 | N/A | N/A |
| **FCM** | $c=6, m=1.25$ | 6 | 100.0% | 0.427428 | 0.827743 | 135.102217 | 0.946564 | 0.103846 |
| **DBSCAN** | $\text{eps}=0.45, \text{min}=4$ | 7 | 71.5% | **0.493716** | **0.607155** | 112.798824 | N/A | N/A |

### 2. Palmer Penguins
| Metode | Konfigurasi | Klaster | Cakupan | Silhouette | DB | CH | ARI | NMI | Purity | PC | PE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **K-Means** | $k=2$ | 2 | 100.0% | 0.531540 | 0.714064 | 482.191469 | 0.655066 | 0.767077 | 0.801170 | N/A | N/A |
| **FCM** | $c=2, m=1.25$ | 2 | 100.0% | 0.531540 | 0.714064 | 482.191469 | 0.655066 | 0.767077 | 0.801170 | 0.981232 | 0.034162 |
| **DBSCAN** | $\text{eps}=0.60, \text{min}=6$ | 2 | 89.18% | **0.561290** | **0.664558** | **519.570108** | **0.681146** | **0.778193** | **0.816393** | N/A | N/A |

## Menjalankan Program dan Reproduksi

Dari PowerShell di direktori `Week-10/`:

```powershell
# 1. Menjalankan seluruh test suite (baseline, FCM, dan edge cases)
& "..\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test*.py" -v

# 2. Menjalankan pipeline clustering utama
# Menjalankan seluruh kombinasi dataset dan algoritma:
& "..\.venv\Scripts\python.exe" -m src.pipeline --dataset all --method all

# Atau secara spesifik per dataset / metode:
& "..\.venv\Scripts\python.exe" -m src.pipeline --dataset mall --method fcm
& "..\.venv\Scripts\python.exe" -m src.pipeline --dataset penguins --method dbscan

# 3. Menjalankan tuning parameter dan generate figur
& "..\.venv\Scripts\python.exe" -m src.tune_fcm
& "..\.venv\Scripts\python.exe" -m src.tune_dbscan
& "..\.venv\Scripts\python.exe" -m src.generate_figures
```

## Pembagian Tugas dan Status

| Anggota | Pekerjaan | Status |
|---|---|---|
| Ahmad Mathlaul Falah — 434241123 | Dataset, preprocessing, pipeline reusable, baseline K-Means, fondasi Git | Selesai |
| Ah. Dliya'ul Adlha Jamalul Lail — 434241126 | Fuzzy C-Means, DBSCAN, evaluasi lanjutan (PC/PE, noise rules), tuning hyperparameter, visualisasi perbandingan | Selesai |
| Abdullah Azzam — 434241118 | Analisis, laporan sintesis kelompok, kesimpulan | Sedang berjalan |
