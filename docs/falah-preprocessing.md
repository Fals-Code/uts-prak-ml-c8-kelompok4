# Catatan Bagian Falah — Dataset & Preprocessing

## Dataset

Dataset yang digunakan adalah **IBM HR Analytics Employee Attrition & Performance** dari Kaggle.

- File: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Jumlah data sumber: 1.470 baris
- Jumlah kolom sumber: 35 kolom
- Target: `Attrition` (`Yes` / `No`)
- Distribusi target yang dikenal pada dataset sumber: 1.233 `No` dan 237 `Yes`

Dataset harus diunduh dari Kaggle lalu ditempatkan pada folder `data/`.

## Langkah Preprocessing

### 1. Pemeriksaan missing value

Program mengecek jumlah nilai kosong pada seluruh kolom menggunakan `isna().sum()`.

Jika ditemukan missing value:

- fitur numerik diisi menggunakan median;
- fitur kategorikal diisi menggunakan modus.

### 2. Pemeriksaan data duplikat

Program menghitung baris duplikat menggunakan `duplicated().sum()` dan menghapus duplikasi penuh menggunakan `drop_duplicates()`.

### 3. Penghapusan kolom noninformatif

Kolom yang hanya berfungsi sebagai identifier atau memiliki nilai konstan dihapus sebelum pemodelan.

Pada dataset ini, program secara eksplisit memperlakukan `EmployeeNumber` sebagai identifier. Kolom konstan dideteksi otomatis.

### 4. Deteksi dan handling outlier dengan IQR

Outlier diperiksa pada fitur numerik kontinu menggunakan metode **Interquartile Range (IQR)**.

Batas bawah dan atas:

- Lower Bound = Q1 - 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

Fitur numerik dengan kardinalitas rendah tidak dikenai IQR karena lebih merepresentasikan kode ordinal/kategorikal daripada nilai kontinu.

Outlier yang ditemukan tidak langsung menghapus baris. Nilainya di-*cap* ke batas bawah/atas IQR agar jumlah data tetap terjaga.

### 5. Encoding data kategorikal

Fitur kategorikal diubah menjadi numerik menggunakan **One-Hot Encoding** melalui `pandas.get_dummies()`.

Target `Attrition` dikonversi menjadi:

- `No` → `0`
- `Yes` → `1`

Output tahap ini adalah:

- `X`: seluruh fitur dalam format numerik;
- `y`: target Attrition dalam format 0/1.

Output ini menjadi input untuk tahap milik anggota berikutnya: split data, Min-Max scaling, dan seleksi fitur.

## Cara Menjalankan

Pastikan dataset sudah ada pada:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Lalu dari root project jalankan:

```bash
python -m src.preprocessing
```

Program akan menampilkan ringkasan preprocessing yang dapat digunakan sebagai bukti output pada laporan.

## Screenshot yang Perlu Diambil untuk Laporan

1. Dataset berhasil dibaca / informasi jumlah data dan kolom.
2. Hasil pengecekan missing value.
3. Hasil pengecekan data duplikat.
4. Ringkasan deteksi outlier IQR.
5. Kolom ID/konstan yang dihapus.
6. Daftar fitur kategorikal yang di-encoding.
7. Jumlah fitur setelah encoding.
8. Distribusi target setelah encoding (`0=No`, `1=Yes`).

## Batas Pengerjaan Falah

Bagian ini berhenti setelah menghasilkan `X` dan `y` hasil preprocessing. Tahapan berikut **tidak** dikerjakan di modul ini agar pembagian kerja tetap jelas:

- split 80% training dan 20% testing;
- Min-Max scaling;
- Mutual Information;
- L1-based Feature Selection;
- Decision Tree;
- evaluasi model.
