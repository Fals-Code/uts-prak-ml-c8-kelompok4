# UTS Praktikum Machine Learning (Kelompok 4 C8)

Project ini dibuat untuk UTS Praktikum Machine Learning 2026 pada **Soal D**. Dataset yang digunakan adalah **IBM HR Analytics Employee Attrition & Performance**.

Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  
Target klasifikasi: `Attrition` (`Yes` / `No`).

## Alur Pengerjaan

1. Preprocessing data
2. Split data 80% training dan 20% testing
3. Min-Max scaling
4. Feature selection menggunakan Mutual Information dan L1-based Feature Selection
5. Klasifikasi menggunakan Decision Tree
6. Evaluasi menggunakan Confusion Matrix, Accuracy, Precision, Recall, dan F1-Score

## Progress Pengerjaan

### Dataset & Preprocessing (Ahmad Mathlaul Falah) ✅

Bagian preprocessing sudah selesai dan sudah masuk ke `main`.

Tahap yang dikerjakan:
- membaca dataset dan menentukan `Attrition` sebagai target;
- mengecek missing value dan menyiapkan penanganan dengan median atau modus;
- mengecek dan menghapus data duplikat;
- menghapus `EmployeeNumber` serta kolom konstan `EmployeeCount`, `Over18`, dan `StandardHours`;
- mendeteksi dan menangani outlier numerik menggunakan metode IQR dengan capping;
- mengubah target `Attrition` menjadi 0 dan 1;
- melakukan One-Hot Encoding pada fitur kategorikal;
- menyimpan hasil preprocessing ke file CSV agar bisa langsung dipakai pada tahap berikutnya.

Hasil preprocessing:
- jumlah data: **1.470 baris**;
- missing value: **0**;
- data duplikat: **0**;
- fitur sebelum encoding: **30**;
- fitur setelah encoding: **51**;
- total kolom pada file hasil: **52 kolom** (51 fitur + 1 target);
- seluruh kolom pada file hasil sudah berbentuk numerik;
- distribusi target: **1.233 No** dan **237 Yes**.

Hasil preprocessing tersimpan di:

```text
data/employee_attrition_preprocessed.csv
```

### Transformasi & Seleksi Fitur (Ah. Dliya'ul Adlha Jamalul Lail) ✅

Bagian transformasi dan feature selection juga sudah selesai dan sudah masuk ke `main`.

Tahap yang dikerjakan:
- membagi data menjadi **80% training** dan **20% testing** menggunakan `random_state=42` dan stratifikasi target;
- menerapkan **MinMaxScaler** dengan `fit` hanya pada data training agar tidak terjadi data leakage;
- menerapkan **Mutual Information** pada data training dan memilih **20 fitur terbaik**;
- mengenali fitur biner hasil One-Hot Encoding sebagai fitur diskrit pada perhitungan Mutual Information;
- menerapkan **L1-based Feature Selection** menggunakan Logistic Regression dengan `penalty="l1"`, `solver="saga"`, dan `C=0.1`;
- memakai fitur terpilih dari data training untuk data testing.

Saat program dijalankan, hasil tahap ini dibuat pada folder:

```text
data/split/
data/scaled/
data/mi_selected/
data/l1_selected/
```

File-file tersebut bisa dibuat ulang dari pipeline, jadi tidak perlu disimpan sebagai artefak utama repository.

### Modeling & Evaluasi (Abdullah Azzam) ⏳

Bagian ini masih dikerjakan pada branch `azzam/model-evaluation`.

Yang dikerjakan:
- training dan testing Decision Tree;
- Confusion Matrix;
- Accuracy, Precision, Recall, dan F1-Score;
- perbandingan hasil kedua metode seleksi fitur.

## Struktur Project

```text
.
├── data/
│   ├── README.md
│   └── employee_attrition_preprocessed.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── modeling.py
│   └── evaluation.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Menjalankan Project

Install dependency terlebih dahulu:

```bash
pip install -r requirements.txt
```

Download dataset asli dari Kaggle, lalu simpan dengan nama:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Setelah itu jalankan:

```bash
python main.py
```

Untuk saat ini, `main.py` sudah menjalankan pipeline sampai tahap feature selection:

```text
Preprocessing
→ Split 80/20
→ Min-Max Scaling
→ Mutual Information
→ L1-based Feature Selection
```

Tahap berikutnya adalah integrasi Decision Tree dan evaluasi model.

## Branch

- `main` (kode dan hasil yang sudah terintegrasi)
- `falah/preprocessing` (selesai dan sudah masuk `main`)
- `adlha/feature-selection` (selesai dan sudah masuk `main`)
- `azzam/model-evaluation` (modeling dan evaluasi)
