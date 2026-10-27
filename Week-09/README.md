# Week 09 — Dataset, EDA, dan preprocessing K-Modes

Penanggung jawab: **Ahmad Mathlaul Falah — 434241123**, kelas C8.
Penugasan dataset: Kelompok 1 menurut informasi pengguna. Nama repository tetap
memuat Kelompok 4; identitas laporan mengikuti sampul dokumen kelompok.
Deadline: 28 Oktober 2026 pukul 23.59 WIB.

Bagian ini menyiapkan data untuk Jamal. Model, jumlah cluster, cost, evaluasi,
dan interpretasi cluster belum dikerjakan di sini. Azzam menangani integrasi
laporan, analisis akhir, dan ZIP. Lihat [audit dan catatan finalisasi](AUDIT.md).

## Distribusi publik dan input lokal

Versi publik ini memuat kode, dokumentasi, analisis agregat, serta CSV Telco
dengan lisensi sumbernya. **Mall Customers tidak disertakan**, termasuk CSV
hasil preprocessing, reference, grafik, dan bukti yang berisi baris pelanggan.
Seluruh folder data/processed dan outputs dihasilkan serta disimpan lokal.
Lihat [kebijakan publikasi](PUBLICATION.md) dan [hasil validasi](VALIDATION.md).

Sebelum menjalankan program lengkap, dapatkan Mall_Customers.csv dari materi
praktikum yang berhak Anda akses atau
[halaman sumber Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
Ikuti ketentuan sumber; akses unduhan tidak dianggap izin mengunggah ulang.
Letakkan berkas tersebut tepat di **Week-09/data/raw/Mall_Customers.csv**.
Sumber, SHA-256 acuan, dan status lisensi dijelaskan di data/raw/SOURCES.md.
Jangan menambahkan berkas atau output lokal dengan git add -f.

Tanpa Mall lokal, dua tes mandiri tetap berjalan dan dua tes yang membutuhkan
Mall dilaporkan SKIP dengan alasan yang jelas. Setelah Mall tersedia, keempat
tes dapat dijalankan. Pipeline penuh dan make_evidence.py memerlukan kedua
dataset; tidak ada dataset pengganti otomatis dan tidak ada perubahan metode.

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath Week-09/data/raw/Mall_Customers.csv
git check-ignore Week-09/data/raw/Mall_Customers.csv
```

## Menjalankan

Gunakan Python 3.12+ karena NumPy 2.5.3 mensyaratkan versi tersebut;
lingkungan yang benar-benar diuji: Python 3.14.6,
Pandas 3.0.6, NumPy 2.5.3, Matplotlib 3.11.2. Python lain belum diuji.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Week-09/requirements.txt
.\.venv\Scripts\python.exe Week-09/preprocessing.py
.\.venv\Scripts\python.exe -m unittest discover -s Week-09 -p test_preprocessing.py -v
.\.venv\Scripts\python.exe Week-09/make_evidence.py
```

Contoh di atas untuk PowerShell Windows tanpa perlu aktivasi environment.
Pada Linux/macOS gunakan `.venv/bin/python` untuk perintah setelah pembuatan venv.
Pastikan `python --version` memenuhi versi minimum sebelum membuat environment.

Jalankan perintah di atas dari root repository. Di workspace Falah interpreter
yang digunakan ialah `.venv/Scripts/python.exe`. Path data dihitung dari lokasi
script sehingga entry point juga bekerja dari direktori lain. `--output-dir folder`
menyimpan seluruh hasil pada folder alternatif, termasuk fitur di `folder/processed`.
Eksekusi setelah instalasi dependency dan penyediaan input tidak memerlukan
internet, scikit-learn, kmodes, atau aplikasi web. Jika memakai `--output-dir`,
pilih lokasi di luar repository: aturan ignore hanya mencakup folder standar,
bukan sembarang folder alternatif.

## File publik dan hasil lokal

CSV Mall, seluruh data/processed, dan seluruh outputs pada tabel berikut merupakan
berkas **lokal hasil eksekusi**, bukan berkas yang disertakan di Git.

| Lokasi | Fungsi |
| --- | --- |
| data/raw/Mall_Customers.csv | Input lokal pengguna, tidak disertakan; acuan 200 × 5 |
| data/raw/Telco-Customer-Churn.csv | Sumber asli Telco, 7.043 × 21 |
| data/raw/SOURCES.md | Asal, hash asli, dan verifikasi versi IBM |
| preprocessing.py | Pemeriksaan, EDA, transformasi, ekspor, validasi |
| data/processed/mall_kmodes.csv | **Input Jamal, 200 × 4** |
| data/processed/telco_kmodes.csv | **Input Jamal, 7.043 × 12** |
| data/processed/mall_reference.csv | source_row dan CustomerID, 200 × 2 |
| data/processed/telco_reference.csv | source_row, customerID, Churn, 7.043 × 3 |
| outputs/mall_inspection.txt, telco_inspection.txt | head, shape, info, describe, kualitas, distribusi |
| outputs/preprocessing_metadata.json | Batas kuantil presisi penuh, kategori, jumlah, hash, versi, validasi |
| outputs/mall_category_counts.csv, telco_category_counts.csv | Frekuensi/persentase semua kategori final |
| outputs/figures/*.png | Empat gambar EDA asli |
| test_preprocessing.py | Uji integrasi, reproducibility, kasus tepi, dan input invalid |
| make_evidence.py | Eksekusi ulang program/tes dan gambar cuplikan kode/output aktual |
| outputs/evidence/ | Log aktual dan PNG bukti; provenance dijelaskan di bawah |
| LAPORAN_FALAH.md | Salinan delapan subbagian laporan dan petunjuk gambar |

Tidak ada file mentah yang ditimpa. Tidak ada baris dihapus, disortir, diacak,
dibagi train-test, atau ditambahkan. Pengulangan profil tetap dipertahankan.

## Pemeriksaan dan interpretasi EDA

Mall tidak memiliki missing value, string kosong, duplikasi penuh, maupun
duplikasi setelah ID dikeluarkan. Seluruh 200 ID unik. Usia 18–70 tahun
(rata-rata 38,85; median 36), pendapatan 15–137 ribu dolar per tahun
(rata-rata 60,56; median 61,5), spending score 1–99 (rata-rata 50,2; median 50).
Gender terdiri dari 112 Female (56%) dan 88 Male (44%).

- `mall_distributions.png`: usia dan pendapatan tersebar cukup lebar; pendapatan
  tinggi lebih jarang. IQR menandai dua pendapatan 137 sebagai nilai ekstrem,
  tetapi nilainya tidak otomatis salah sehingga tetap digunakan.
- `mall_income_spending.png`: pada pendapatan tinggi terdapat spending score
  rendah maupun tinggi. Pendapatan saja tidak cukup menggambarkan perilaku
  belanja; income dan spending dipertahankan sebagai dimensi berbeda.
  Pola ini deskriptif, bukan bukti sebab-akibat atau hasil clustering.

Telco memiliki 7.043 ID unik dan nol duplikasi penuh. Ada 22 pengulangan setelah
customerID dikeluarkan (Churn masih termasuk dalam pembandingan); ini bukan
bukti pelanggan yang sama. `isna()` mentah melaporkan nol, tetapi 11 TotalCharges
berisi spasi. Konversi numerik untuk EDA mengubahnya menjadi missing, seluruhnya
pada tenure 0. Tidak ada teks nonkosong lain yang gagal dikonversi. Tidak ada
konflik PhoneService–MultipleLines maupun InternetService–layanan tambahan.
Kategori tanpa layanan valid, bukan missing value.

- `telco_characteristics.png`: SeniorCitizen sebanyak 1.142 (16,21%), Partner
  3.402, dan Dependents 2.110. Fiber optic 3.096, DSL 2.421, tanpa internet 1.526.
  Kontrak bulanan paling umum: 3.875 (55,02%), satu tahun 1.473, dua tahun 1.695.
  Electronic check paling umum di antara pembayaran (2.365; 33,58%). Variasi
  ini mendasari dipertahankannya fitur layanan dan kontrak.
- `telco_numeric.png`: tenure 0–72 bulan (median 29; rata-rata 32,37), biaya
  bulanan 18,25–118,75 dolar (median 70,35; rata-rata 64,76). TotalCharges valid
  18,80–8.684,80 dolar dengan median 1.397,475; ekor kanan lebar mencerminkan
  akumulasi biaya. Grafik TotalCharges memakai 7.032 nilai valid; ini hanya
  pengecualian saat menggambar, bukan penghapusan 11 pelanggan. IQR pada tenure
  dan kedua biaya tidak menandai nilai ekstrem. SeniorCitizen merupakan kode
  kategori biner, sehingga tidak diperiksa sebagai outlier kontinu.

## Pemilihan fitur

Mall mempertahankan empat dimensi profil dasar. Gender memberi konteks
demografis tetapi tidak diasumsikan menentukan perilaku belanja.

| Atribut Mall | Keputusan dan alasan |
| --- | --- |
| CustomerID | Dikeluarkan; identitas disimpan terpisah |
| Gender | Dipertahankan; demografi dasar |
| Age | Menjadi AgeCategory; usia relatif dalam sampel |
| Annual Income (k$) | Menjadi IncomeCategory; karakteristik ekonomi |
| Spending Score (1-100) | Menjadi SpendingCategory; perilaku belanja |

Telco dipersempit menjadi 12 fitur untuk segmentasi rumah tangga, layanan inti,
dan pola berlangganan. Churn tidak digunakan untuk memilih fitur/batas kuantil.

| Atribut Telco | Keputusan dan alasan |
| --- | --- |
| customerID | Dikeluarkan; identitas disimpan terpisah |
| Churn | Dikeluarkan; hanya label pembanding eksternal dalam reference |
| gender | Dikeluarkan; fokus demografi pada status senior/rumah tangga, bukan jenis kelamin; bukan klaim fitur tidak berguna |
| SeniorCitizen | Dipertahankan; kode 0/1 menjadi No/Yes, bukan skala numerik |
| Partner, Dependents | Dipertahankan; karakteristik rumah tangga berbeda |
| PhoneService | Dikeluarkan; tanpa telepon sudah tercakup MultipleLines |
| MultipleLines | Dipertahankan; tanpa telepon, layanan tunggal, atau beberapa jalur |
| InternetService | Dipertahankan; DSL, Fiber optic, atau tanpa internet |
| OnlineSecurity, TechSupport | Dipertahankan; kebutuhan keamanan dan dukungan |
| OnlineBackup, DeviceProtection, StreamingTV, StreamingMovies | Dikeluarkan untuk membatasi dominasi banyak atribut layanan tambahan yang saling terkait; bukan klaim tidak bernilai |
| Contract | Dipertahankan; komitmen kontrak |
| PaperlessBilling, PaymentMethod | Dipertahankan; pilihan tagihan dan cara pembayaran |
| tenure | Menjadi TenureCategory; pengalaman berlangganan |
| MonthlyCharges | Menjadi MonthlyChargesCategory; tingkat biaya |
| TotalCharges | Dikeluarkan; akumulasi berpotensi menambah bobot tenure/biaya, serta memiliki 11 nilai kosong |

String di-strip pada salinan kerja. Kategori asing atau missing pada fitur
terpilih menyebabkan error yang jelas, bukan imputasi diam-diam. Pada input
aktual tidak diperlukan imputasi. TotalCharges hanya dikonversi untuk EDA;
tidak diisi nol/median karena dikeluarkan dari fitur. `No`, `No internet service`,
dan `No phone service` tetap berbeda: tanpa layanan bukan penolakan fitur tambahan.

## Diskretisasi

Tertil dihitung terpisah per dataset dengan kuantil linear 0, 1/3, 2/3, 1,
lalu `pd.cut(..., right=True, include_lowest=True)`. Tiga kategori adalah
ringkasan eksploratif sederhana, bukan berdasarkan Churn atau hasil model.
Jumlah kategori per fitur **bukan** penentuan jumlah cluster.

| Fitur | Rendah | Sedang | Tinggi | Jumlah rendah/sedang/tinggi |
| --- | --- | --- | --- | --- |
| AgeCategory | [18; 31] | (31; 45] | (45; 70] | 70 / 64 / 66 |
| IncomeCategory | [15; 48] | (48; 72] | (72; 137] | 70 / 64 / 66 |
| SpendingCategory | [1; 41,333333…] | (41,333333…; 59] | (59; 99] | 67 / 68 / 65 |
| TenureCategory | [0; 14] | (14; 47] | (47; 72] | 2.371 / 2.369 / 2.303 |
| MonthlyChargesCategory | [18,25; 50,40] | (50,40; 83,90] | (83,90; 118,75] | 2.351 / 2.345 / 2.347 |

Tabel memakai koma desimal; JSON menyimpan angka presisi penuh dengan titik.
Batas bawah pertama dan semua batas kanan termasuk interval. Spending score
bulat ekuivalen 1–41, 42–59, dan 60–99. Nilai sama tidak dipecah, sehingga
jumlah anggota tidak harus persis sama. Batas berulang dihapus, interval kosong
digabung, fitur konstan diberi kategori tunggal. Tidak ada kasus tersebut pada
lima fitur numerik aktual; perilakunya diuji pada sampel kecil.

Batas relatif terhadap sampel, bukan standar usia/penghasilan universal.
Diskretisasi menghilangkan informasi numerik. K-Modes menganggap rendah,
sedang, tinggi sebagai kategori nominal: ketidakcocokan rendah–tinggi tidak
otomatis lebih besar daripada rendah–sedang. Beberapa kategori tanpa internet
masih dapat memberi bobot berulang pada ketiadaan layanan. Keterbatasan ini
perlu dibahas Jamal tanpa mengubah input diam-diam. Tidak ada scaling, one-hot
encoding, train-test split, penghapusan outlier, atau model dalam tahap ini.

## Handoff untuk Jamal

Gunakan **hanya** dua CSV `*_kmodes.csv` sebagai fitur. Semua nilai adalah string.

- Mall: `Gender, AgeCategory, IncomeCategory, SpendingCategory`.
- Telco: `SeniorCitizen, Partner, Dependents, MultipleLines, InternetService,
  OnlineSecurity, TechSupport, Contract, PaperlessBilling, PaymentMethod,
  TenureCategory, MonthlyChargesCategory`.

```python
from pathlib import Path
import pandas as pd

# Notebook dijalankan dari root repository.
folder = Path("Week-09/data/processed")
mall = pd.read_csv(folder / "mall_kmodes.csv", dtype=str)
telco = pd.read_csv(folder / "telco_kmodes.csv", dtype=str)
mall_reference = pd.read_csv(folder / "mall_reference.csv")
telco_reference = pd.read_csv(folder / "telco_reference.csv")

# Pemeriksaan ringkas setelah preprocessing; empat tes menyediakan pemeriksaan lengkap.
assert mall.shape == (200, 4) and telco.shape == (7043, 12)
for features in [mall, telco]:
    assert not features.isna().any().any()
    assert not features.apply(lambda column: column.str.strip().eq("")).any().any()
    assert set(features).isdisjoint({"CustomerID", "customerID", "Churn", "source_row"})
assert mall_reference.source_row.tolist() == list(range(len(mall)))
assert telco_reference.source_row.tolist() == list(range(len(telco)))

# Setelah Jamal menghasilkan label, cocokkan secara posisi TANPA mengurutkan.
# telco_reference["cluster"] = labels_telco
# Churn hanya untuk analisis eksternal sesudah clustering, bukan input model.
```

`source_row` 0 sampai n−1 menunjuk posisi baris sumber tanpa header; baris
CSV fisiknya `source_row + 2`. Jangan masukkan reference ke matriks fitur.
Bila perlu array, gunakan `mall.to_numpy(dtype=str)` dan `telco.to_numpy(dtype=str)`.
Kategori/jumlah lengkap tersedia di `*_category_counts.csv`; pakai metadata
`binning` untuk merekonstruksi batas melalui `pd.cut`. Data baru di luar rentang
sumber memerlukan kebijakan tersendiri; pipeline ini untuk batch praktikum.

Ada 160 pengulangan profil kategori Mall dan 4.107 Telco setelah transformasi
(40 dan 2.936 profil unik). Baris tersebut mewakili pelanggan berbeda atau
informasi yang menyatu akibat diskretisasi; seluruhnya dipertahankan.

## Validasi yang dijalankan

Empat tes unittest lulus. Uji integrasi menjalankan pipeline dua kali dan
membandingkan seluruh CSV, JSON, teks, dan empat PNG byte demi byte; juga
memeriksa shape, kekosongan, larangan ID/Churn/source_row, urutan ID/Churn,
hash data asli, dan grafik. Kasus tepi mencakup batas sama, interval kosong,
fitur konstan, missing, usia negatif, kategori asing, ID kosong, konflik
layanan, dan strip label. Tidak ada pengujian model clustering.

`make_evidence.py` menyimpan log aktual pada `outputs/evidence/validation.txt`.
PNG kode/output adalah render literal dari berkas/log aktual, bukan tangkapan
layar IDE/terminal. Skrip memisahkan bukti ini dari grafik EDA asli.

LAPORAN_FALAH.md memuat teks metodologi dan hasil agregat. Gambar dan bukti
berisi baris data hanya dibuat lokal setelah input tersedia. Google Docs yang
sudah disetujui pengguna tidak diubah oleh penyiapan versi publik ini.

## Rujukan metodologi

- [Pandas qcut](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html): kuantil dan batas berulang.
- [Pustaka kmodes](https://github.com/nicodv/kmodes): kategori, ketidakcocokan, dan input tanpa NaN.
- [IBM Telco](https://github.com/IBM/telco-customer-churn-on-icp4d): verifikasi versi dataset, bukan acuan pipeline klasifikasi.

[Kembali ke halaman utama](../README.md)
