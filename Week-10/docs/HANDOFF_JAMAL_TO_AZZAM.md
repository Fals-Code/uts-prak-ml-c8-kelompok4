# Catatan Serah Terima (Handoff Jamal ke Azzam) — Week 10

**Kepada:** Abdullah Azzam (434241118) — PIC Analisis, Laporan Akhir, dan Kesimpulan  
**Dari:** Ah. Dliya'ul Adlha Jamalul Lail (434241126) — PIC FCM, DBSCAN, Metrik, & Visualisasi  
**Tanggal:** Selasa, 3 November 2026  
**Status Tahap 2:** Selesai (100% kode, tuning, hasil komparasi, dan visualisasi telah siap)

---

## 1. Ringkasan Aset yang Diserahkan

Seluruh modul dan artefak telah diintegrasikan langsung di branch `adlha/week10-fcm-dbscan` tanpa merusak pipeline baseline Falah:

| Kategori | Lokasi File | Deskripsi Isi |
|---|---|---|
| **Pipeline & Runner** | `Week-10/src/pipeline.py` | Runner integratif mendukung `--method kmeans`, `fcm`, `dbscan`, `all`. |
| **Implementasi FCM** | `Week-10/src/fcm.py` | Modul FCM mandiri (NumPy/SciPy) dengan perhitungan Partition Coefficient (PC) dan Partition Entropy (PE). |
| **Implementasi DBSCAN** | `Week-10/src/dbscan_runner.py` | Wrapper DBSCAN dengan analisis noise/coverage dan fungsi diagnosa `compute_k_distance`. |
| **Modul Tuning** | `Week-10/src/tune_fcm.py`<br>`Week-10/src/tune_dbscan.py` | Script pencarian grid search hyperparameter murni unsupervised. |
| **Generator Visualisasi**| `Week-10/src/generate_figures.py` | Generator scatter 2D komparatif, k-distance plot, dan kurva evaluasi ($\ge 150$ DPI). |
| **Modul Evaluasi** | `Week-10/src/evaluation.py` | Evaluator metrik internal (Silhouette, DB, CH), eksternal (ARI, NMI, Purity), dan fuzzy (PC, PE). |
| **Unit Test Suite** | `Week-10/tests/test_fcm.py`<br>`Week-10/tests/test_edge_cases.py` | Pengujian properti keanggotaan, PC/PE bounds, reproduksibilitas, skenario 100% noise, dan klaster tunggal. |
| **Tabel Hasil Metrik** | `Week-10/results/metrics_summary_mall.csv`<br>`Week-10/results/metrics_summary_penguins.csv` | Tabel komparasi resmi seluruh metrik aktual untuk disalin ke laporan. |
| **Membership Matrix** | `Week-10/results/mall_fcm_membership.csv`<br>`Week-10/results/penguins_fcm_membership.csv` | Matriks derajat keanggotaan fuzzy $U$ ($N \times c$) per baris data. |
| **Parameter Final** | `Week-10/results/params_final.json` | Konfigurasi parameter model terpilih. |
| **Artefak Gambar** | `Week-10/figures/*.png` | Gambar plot visualisasi siap pakai untuk bab pembahasan laporan. |
| **Dokumentasi Teknis**| `Week-10/docs/AUDIT_PIPELINE.md`<br>`Week-10/docs/PARAMETER_NOTES.md` | Catatan audit arsitektur dan justifikasi matematis pemilihan parameter. |

---

## 2. Tabel Ringkasan Metrik Komparasi untuk Laporan

Berikut data resmi hasil eksekusi aktual untuk dicantumkan pada Bab Hasil dan Pembahasan:

### A. Mall Customer ($N = 200$, 3 Fitur Terstandardisasi)
| Algoritma | Parameter Terpilih | Klaster ($k/c$) | Cakupan (Coverage) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Partition Coeff (PC) | Partition Entropy (PE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **K-Means** | $k=6$ | 6 | 100.0% | 0.428417 | 0.825354 | 135.102104 | N/A | N/A |
| **FCM** | $c=6, m=1.25$ | 6 | 100.0% | 0.427428 | 0.827743 | 135.102217 | 0.946564 | 0.103846 |
| **DBSCAN** | $\text{eps}=0.45, \text{min}=4$ | 7 | 71.5% | **0.493716** | **0.607155** | 112.798824 | N/A | N/A |

### B. Palmer Penguins ($N = 342$, 4 Fitur Terstandardisasi)
| Algoritma | Parameter Terpilih | Klaster ($k/c$) | Cakupan (Coverage) | Silhouette Score | DB Index | CH Index | ARI | NMI | Purity | PC | PE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **K-Means** | $k=2$ | 2 | 100.0% | 0.531540 | 0.714064 | 482.191469 | 0.655066 | 0.767077 | 0.801170 | N/A | N/A |
| **FCM** | $c=2, m=1.25$ | 2 | 100.0% | 0.531540 | 0.714064 | 482.191469 | 0.655066 | 0.767077 | 0.801170 | 0.981232 | 0.034162 |
| **DBSCAN** | $\text{eps}=0.60, \text{min}=6$ | 2 | 89.18% | **0.561290** | **0.664558** | **519.570108** | **0.681146** | **0.778193** | **0.816393** | N/A | N/A |

---

## 3. Poin-Poin Kritis untuk Analisis dan Sintesis Azzam

Saat menyusun laporan akhir kelompok untuk menjawab pertanyaan dosen:
> *"Lakukan analisa apakah dengan metode lain tersebut nilai akurasi menjadi lebih baik dibandingkan dengan metode K-Means!"*

Harap perhatikan argumentasi metodologis berikut:

1. **Koreksi Konsep "Akurasi" pada Unsupervised Learning:**
   - Dalam clustering murni tanpa supervisi (khususnya Mall Customer), istilah "akurasi" tidak relevan secara teknis karena tidak ada label kelas acuan (*ground truth*).
   - Penilaian kualitas pengelompokan harus didasarkan pada **validasi internal**: kepadatan klaster (*compactness*) dan pemisahan antar klaster (*separation*), yang diukur melalui Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index.
   - Pada Palmer Penguins, kita memiliki label `Species`, sehingga evaluasi kecocokan biologis dapat dinilai melalui metrik eksternal (ARI, NMI, dan Purity).

2. **Performa Fuzzy C-Means (FCM) vs K-Means:**
   - **Karakteristik Geometris yang Serupa:** Baik K-Means maupun FCM mengoptimalkan fungsi objektif berbasis jarak Euclidean terhadap pusat titik prototipe (*centroid*).
   - Pada Mall ($c=6$) dan Penguins ($c=2$), FCM dengan $m=1.25$ menghasilkan partisi yang sangat mendekati K-Means (Silhouette Mall: `0.427428` vs `0.428417`; Penguins: identik `0.531540`).
   - **Kelebihan FCM:** FCM memberikan nilai tambah berupa derajat keanggotaan kontinu ($u_{ij} \in [0, 1]$). Titik-titik yang berada di perbatasan antar kelompok tidak dipaksa ke dalam satu klaster secara biner (*hard assignment*), melainkan memiliki probabilitas keanggotaan berbobot (terdokumentasi di `results/*_fcm_membership.csv`).
   - Nilai $PC$ tinggi ($0.946$ di Mall, $0.981$ di Penguins) dan $PE$ rendah ($0.103$ di Mall, $0.034$ di Penguins) membuktikan partisi klaster yang terbentuk stabil dan tidak ambigu.

3. **Performa DBSCAN vs K-Means & FCM (Peringatan Kritis Noise!):**
   - **Tampilan Metrik:** Nilai Silhouette Score DBSCAN terlihat lebih tinggi (`0.493716` pada Mall dan `0.561290` pada Penguins) serta Davies-Bouldin lebih rendah (`0.607` dan `0.664`).
   - **Konteks Pembanding yang Wajib Ditegaskan:** Metrik internal DBSCAN hanya dihitung pada **titik non-noise (core & border points)**:
     - Mall Customer mengecualikan **28.5% data sebagai noise** (cakupan 71.5%).
     - Palmer Penguins mengecualikan **10.8% data sebagai noise** (cakupan 89.18%).
   - K-Means dan FCM mengevaluasi **100% sampel**. Peningkatan nilai Silhouette pada DBSCAN terjadi karena DBSCAN membuang titik-titik anomali dan daerah berdensitas rendah yang berpotensi menurunkan kerapatan klaster.
   - **Evaluasi Eksternal Penguins:** DBSCAN mencatatkan ARI (`0.681`), NMI (`0.778`), dan Purity (`0.816`) yang sedikit lebih tinggi dari K-Means/FCM (`0.655`, `0.767`, `0.801`), menunjukkan bahwa penyaringan individu penguin dengan ukuran morfometrik outlier mampu mempertajam batas pemisahan spesies.

---

## 4. Panduan Verifikasi Cepat untuk Azzam

Azzam dapat memverifikasi seluruh hasil kapan saja menggunakan perintah berikut di terminal:
```powershell
Set-Location Week-10

# 1. Jalankan semua unit test
& "..\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test*.py" -v

# 2. Cek visualisasi figur komparatif yang sudah siap
Get-ChildItem figures\
```

Semua pekerjaan Tahap 2 telah diserahterimakan dengan tuntas. Selamat menyusun draf laporan akhir kelompok!
