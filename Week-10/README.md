# Week 10 — Clustering

## Tujuan

Week 10 menyiapkan analisis clustering pada dataset Mall Customer dan Palmer Penguins. Pipeline preprocessing dan baseline K-Means kini tersedia untuk kedua dataset.

## Dataset dan skema

- Mall Customer: `data/raw/Mall_Customers.csv`, 200 baris × 5 kolom. Header asli: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`. Tidak ada missing value atau duplikasi baris. Tiga kolom numerik (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`) terbaca numerik. SHA-256: `416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77`.
- Palmer Penguins Kaggle: `data/raw/Penguins.csv`, 344 baris × 19 kolom. Skema dan hash lengkap ada di [`data/README.md`](data/README.md). Nama header Kaggle dipertahankan; URL Kaggle dan lisensi spesifik belum terverifikasi.

Mall menggunakan dataset asli Week 9 yang disalin byte-identik. File `mall_kmodes.csv` tidak menjadi input K-Means. Penguins menggunakan file Kaggle yang disediakan pengguna.

## Rancangan analisis

K-Means menjadi baseline baru Week 10 karena source implementasi K-Means Week 9 tidak ditemukan. Satu pipeline reusable direncanakan untuk kedua dataset dengan tahap pemilihan fitur, penanganan nilai hilang, transformasi/scaling yang sesuai, fitting model, dan evaluasi. Rincian preprocessing perlu disesuaikan per dataset: Mall memiliki atribut kategorikal `Gender` dan fitur numerik; Penguins memiliki label `Species`, empat ukuran tubuh numerik, dan missing value pada ukuran tubuh. `Species` hanya untuk evaluasi eksternal dan tidak menjadi fitur clustering. Implementasi baseline tersedia di `src/`.


## Menjalankan dan menguji

Dari root repository, gunakan interpreter environment yang sudah tersedia:

```powershell
Set-Location Week-10
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe -m src.pipeline --dataset all
# Atau --dataset mall / --dataset penguins
```

`requirements.txt` mencatat versi library dari environment eksekusi. Python dan versi library juga disimpan dalam metadata hasil. Tidak ada library yang dipasang dalam pengerjaan ini.

## Preprocessing final

Mall memakai hanya `Age`, `Annual Income (k$)`, dan `Spending Score (1-100)`; CustomerID dan Gender dikeluarkan. Semua 200 baris digunakan. Missing value Mall menyebabkan error agar tidak ada imputasi diam-diam. Penguins memakai empat ukuran tubuh dalam header Kaggle dan complete-case: 344 menjadi 342 baris, 2 baris dikeluarkan karena setidaknya satu ukuran tubuh hilang. Nilai nonmissing yang gagal dikonversi numerik menyebabkan error. Species disimpan terpisah dengan source_row yang sama. Semua fitur model ditransformasi StandardScaler (fit pada batch clustering). CSV asli tidak ditulis ulang.

## Antarmuka untuk integrasi Jamal

- `src.preprocessing.prepare_dataset(dataset_name)` mengembalikan PreparedDataset: X terstandardisasi, features dalam satuan asli, source_rows, external_labels, scaler, metadata. Nama dataset: mall atau penguins.
- `src.pipeline.run_clustering(dataset_name, method="kmeans", k_values=range(2, 11), random_state=42, n_init=10, output_dir=None)` mengembalikan model, labels, metrics, candidates, dan prepared. output_dir=None tidak menulis hasil. Metode lain belum didukung.
- `src.evaluation.evaluate_clustering(X, labels, external_labels=None, noise_label=-1)` mengevaluasi matriks preprocessing yang sama dan label yang selaras.

FCM dan DBSCAN dapat memanggil prepare_dataset dan evaluate_clustering tanpa menduplikasi preprocessing. FCM menggunakan k terpilih per dataset sebagai baseline perbandingan; jangan memaksa kedua dataset memakai k sama.

## Evaluasi dan hasil

K-Means memakai random_state=42, n_init=10 dan k=2–10. Pemilihan memakai Silhouette Score valid tertinggi, dengan k lebih kecil untuk nilai yang persis sama. Silhouette bukan accuracy. Elbow dan scatter membantu interpretasi; scatter 2D hanya proyeksi dari model pada seluruh fitur.

Purity = jumlah anggota kelas eksternal terbanyak pada setiap cluster, dibagi jumlah sampel evaluasi. Species tidak digunakan untuk memilih k. Purity dapat meningkat ketika jumlah cluster meningkat, sehingga perlu dibaca bersama ARI/NMI dan metrik internal.

Untuk label noise -1, evaluasi mencatat n_noise, n_evaluated, coverage. Semua metrik dievaluasi pada titik non-noise; label eksternal ikut disaring dengan mask yang sama. Metrik internal hanya tersedia ketika 2 <= jumlah cluster < sampel non-noise; selain itu None (JSON null/CSV kosong, berarti N/A), beserta alasannya. Perbandingan internal dengan metode yang menghasilkan noise menggunakan cakupan sampel berbeda. DBSCAN belum diimplementasikan.

Hasil tersedia di results/<dataset>_kmeans_*: k_search.csv untuk semua kandidat, metrics.csv untuk k terpilih, metadata.json untuk parameter/versi/preprocessing/hash, labels.csv untuk label dan source_row, serta elbow.png, silhouette.png, clusters.png. Labels dan scatter berasal dari data pengguna dan tetap lokal sampai izin redistribusi jelas.

Hasil aktual mall: k=6, dipilih karena Silhouette tertinggi dari kandidat valid k=2–10.

Hasil aktual penguins: k=2, dipilih karena Silhouette tertinggi dari kandidat valid k=2–10.
