# Preprocessing - Falah

Dataset yang dipakai adalah **IBM HR Analytics Employee Attrition & Performance**.

File dataset:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Target yang digunakan adalah `Attrition` dengan nilai `Yes` dan `No`.

## Alur preprocessing

1. Baca dataset.
2. Cek missing value.
   - numerik diisi median jika ada data kosong;
   - kategorikal diisi modus.
3. Cek dan hapus data duplikat.
4. Hapus kolom yang tidak digunakan seperti `EmployeeNumber` dan kolom konstan.
5. Cek outlier pada data numerik menggunakan IQR.
6. Outlier ditangani dengan capping pada batas bawah dan batas atas IQR.
7. Ubah target `Attrition` menjadi `No = 0` dan `Yes = 1`.
8. Ubah fitur kategorikal menggunakan One-Hot Encoding.

Hasil preprocessing berupa `X` sebagai fitur dan `y` sebagai target. Data ini nantinya dipakai untuk tahap split data, Min-Max, dan feature selection.

## Menjalankan program

```bash
python -m src.preprocessing
```

Hasil pengujian pada dataset:

- jumlah data: 1470;
- missing value: 0;
- data duplikat: 0;
- fitur sebelum encoding: 30;
- fitur setelah encoding: 51;
- target: 1233 `No` dan 237 `Yes`.

Kolom yang dihapus:

```text
EmployeeNumber, EmployeeCount, Over18, StandardHours
```

Bagian Falah selesai sampai preprocessing dan encoding. Split data, Min-Max, feature selection, Decision Tree, dan evaluasi dilanjutkan oleh anggota lain.
