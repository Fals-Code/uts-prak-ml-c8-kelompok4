# Week 10 — Clustering

**Kelompok C8 — Week 10**

## Soal praktikum

> Buatlah program clustering pada dataset Mall Customer minggu kemarin dengan menggunakan metode clustering yang lain: Fuzzy C-Means dan DBSCAN!
>
> 1. Lakukan analisa apakah dengan metode lain tersebut nilai akurasi menjadi lebih baik dibandingkan dengan metode K-Means!
> 2. Tugas dikumpulkan pada Rabu, 4 November 2026 pukul 23.59 WIB.

Clustering adalah unsupervised learning. Accuracy klasifikasi tidak langsung berlaku tanpa label dan definisi pencocokan kelas. Karena itu evaluasi utama memakai metrik internal; untuk Penguins tersedia evaluasi eksternal menggunakan Species.

## Tujuan

Membandingkan metode clustering pada Mall Customer dan Palmer Penguins dengan preprocessing konsisten dan evaluasi yang dapat digunakan bersama. Baseline K-Means tersedia. FCM, DBSCAN, dan analisis perbandingan akhir belum selesai.

## Dataset

| Dataset | Path lokal | Dimensi mentah | Fitur clustering | Catatan | SHA-256 |
|---|---|---:|---|---|---|
| Mall Customer | `data/raw/Mall_Customers.csv` | 200 × 5 | Age, Annual Income (k$), Spending Score (1-100) | CustomerID tidak digunakan; tidak ada missing | `416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77` |
| Palmer Penguins | `data/raw/Penguins.csv` | 344 × 19 | Culmen Length, Culmen Depth, Flipper Length, Body Mass | Species hanya evaluasi eksternal; URL dan lisensi Kaggle spesifik perlu konfirmasi | `5ea908ae6fd5363b6d649f9eecdd63c010caaa389ee9498483e9ec30647dee2d` |

Kedua CSV mentah bersifat lokal dan tidak tercatat di Git. Mall berasal dari file Week 9. Penguins mempertahankan header asli Kaggle; URL sumber dan lisensinya belum terverifikasi.

## Struktur direktori

- `src/preprocessing.py` — pemilihan fitur, validasi, scaling.
- `src/pipeline.py` — runner K-Means reusable.
- `src/evaluation.py` — metrik internal/eksternal dan noise handling.
- `tests/test_pipeline.py` — lima unittest.
- `data/raw/` — dataset lokal (tidak dilacak Git).
- `results/` — tabel metrik, pencarian k, metadata, dan plot; sebagian output berbasis baris tetap lokal.

## Preprocessing

Mall memakai tiga fitur numerik; CustomerID dan Gender tidak masuk model. Semua 200 sampel digunakan. Jika Mall memiliki missing value, pipeline berhenti tanpa imputasi diam-diam. Penguins mengonversi empat ukuran tubuh Kaggle menjadi numerik dan memakai complete-case: 2 dari 344 baris dikeluarkan dari data kerja karena setidaknya satu fitur ukuran tubuh hilang. Nilai nonmissing yang gagal dikonversi menyebabkan error. Kedua dataset memakai StandardScaler. CSV mentah tidak diubah.

Satu fungsi `prepare_dataset(dataset_name)` menyiapkan matriks fitur yang sama untuk metode clustering berikutnya. Label Species dan source row disimpan selaras, tetapi Species tidak pernah menjadi fitur.

## Metode clustering dan konfigurasi

K-Means adalah baseline baru Week 10 karena source K-Means Week 9 tidak ditemukan. Konfigurasi: `random_state=42`, `n_init=10`, kandidat k=2 sampai 10. k dipilih per dataset berdasarkan Silhouette Score tertinggi yang valid; jika seri, dipilih k terkecil. FCM dan DBSCAN akan ditambahkan kemudian menggunakan data preprocessing yang sama. FCM nantinya melaporkan Partition Coefficient dan Partition Entropy.

## Evaluasi

Metrik internal: Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index. Silhouette bukan accuracy. Penguins juga memakai Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), dan Purity terhadap Species sebagai evaluasi eksternal.

Untuk metode yang menghasilkan label noise `-1`, evaluator mencatat jumlah noise dan coverage; metrik internal dihitung hanya pada titik non-noise. Metrik yang tidak terdefinisi ditandai N/A. Perbandingan internal pada metode dengan noise memakai cakupan sampel yang berbeda.

## Hasil baseline K-Means

| Dataset | n | k | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | Purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mall Customer | 200 | 6 | 0.428417 | 0.825354 | 135.102104 | N/A | N/A | N/A |
| Palmer Penguins | 342 | 2 | 0.531540 | 0.714064 | 482.191469 | 0.655066 | 0.767077 | 0.801170 |

## Visualisasi

Untuk kedua dataset tersedia elbow plot dan Silhouette Score untuk k=2–10. Scatter cluster memakai Annual Income dan Spending Score untuk Mall serta Flipper Length dan Body Mass untuk Penguins. Scatter adalah proyeksi dua dimensi; model dilatih menggunakan semua fitur yang ditentukan. File hasil yang dilacak Git berada di `results/`; scatter berbasis baris berada lokal.

## Menjalankan program dan unittest

Dari PowerShell di root repository, working directory pengujian dan modul harus `Week-10/`:

```powershell
Push-Location Week-10
try {
    & "..\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test*.py" -v
    & "..\.venv\Scripts\python.exe" -m src.pipeline --dataset all
}
finally {
    Pop-Location
}
```

Untuk satu dataset, ganti `all` dengan `mall` atau `penguins`. Runner mendukung `--dataset` dan `--output-dir`; tidak ada argumen CLI lain. Hasil baseline dan metadata disimpan di `results/`.

## Pembagian tugas dan status

| Anggota | Pekerjaan | Status |
|---|---|---|
| Ahmad Mathlaul Falah — 434241123 | Dataset, preprocessing, pipeline reusable, baseline K-Means, fondasi Git | Selesai |
| Ah. Dliya'ul Adlha Jamalul Lail — 434241126 | Fuzzy C-Means, DBSCAN, evaluasi lanjutan, visualisasi perbandingan | Belum selesai |
| Abdullah Azzam — 434241118 | Analisis, laporan, kesimpulan, finalisasi dokumen | Belum selesai |

Status Week 10: dataset dan preprocessing selesai; baseline K-Means selesai; FCM belum selesai; DBSCAN belum selesai; analisis perbandingan akhir belum selesai.
