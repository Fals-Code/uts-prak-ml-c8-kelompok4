# Dataset UTS

Dataset: **IBM HR Analytics Employee Attrition & Performance**.
Target: `Attrition` (`Yes` / `No`).

Sumber: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Unduh dan ekstrak dataset, lalu simpan file mentah dengan lokasi persis berikut dari root repository:

```text
UTS/data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Dataset mentah tidak tersedia dalam repository sejak sebelum restrukturisasi. File ini tetap diabaikan oleh Git karena disediakan melalui unduhan terpisah dari sumber aslinya.

`employee_attrition_preprocessed.csv` adalah hasil preprocessing UTS yang sudah tersimpan (1.470 baris, 51 fitur dan 1 target). File ini dipertahankan sebagai hasil historis, bukan pengganti input mentah untuk `main.py`. Menjalankan pipeline akan memperbaruinya; periksa diff sebelum commit.

Folder `split/`, `scaled/`, `mi_selected/`, dan `l1_selected/` dibuat ulang saat pipeline berjalan dan tidak disimpan di Git.

Jalankan program dari folder `UTS/` sesuai [README UTS](../README.md).
