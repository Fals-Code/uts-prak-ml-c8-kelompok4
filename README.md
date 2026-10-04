# UTS Praktikum Machine Learning — Kelompok 4 C8

Project UTS Praktikum Machine Learning 2026 untuk **Soal D** menggunakan dataset **IBM HR Analytics Employee Attrition & Performance**.

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

### Dataset & Preprocessing — Ahmad Mathlaul Falah ✅

Bagian preprocessing sudah selesai dan sudah terintegrasi ke `main`.

Proses yang dikerjakan:
- membaca dataset dan menentukan `Attrition` sebagai target;
- memeriksa missing value dan menyiapkan penanganan median/modus;
- memeriksa dan menghapus data duplikat;
- menghapus `EmployeeNumber` serta kolom konstan `EmployeeCount`, `Over18`, dan `StandardHours`;
- mendeteksi dan menangani outlier numerik menggunakan metode IQR dengan capping;
- mengubah target `Attrition` menjadi 0 dan 1;
- melakukan One-Hot Encoding pada fitur kategorikal;
- menyimpan hasil preprocessing sebagai file CSV untuk tahap berikutnya.

Hasil preprocessing:
- jumlah data: **1.470 baris**;
- missing value: **0**;
- data duplikat: **0**;
- fitur sebelum encoding: **30**;
- fitur setelah encoding: **51**;
- total kolom pada file hasil: **52 kolom** (51 fitur + 1 target);
- seluruh kolom pada file hasil sudah berbentuk numerik;
- distribusi target: **1.233 No** dan **237 Yes**.

Output preprocessing tersedia pada:

```text
data/employee_attrition_preprocessed.csv
```

### Transformasi & Seleksi Fitur — Ah. Dliya'ul Adlha Jamalul Lail ✅

Bagian transformasi dan feature selection sudah selesai dan sudah terintegrasi ke `main`.

Proses yang dikerjakan:
- membagi data menjadi **80% training** dan **20% testing** menggunakan `random_state=42` dan stratifikasi target;
- menerapkan **MinMaxScaler** dengan proses `fit` hanya pada data training untuk menghindari data leakage;
- menerapkan **Mutual Information** pada data training dan memilih **20 fitur terbaik**;
- mendeteksi fitur biner/One-Hot Encoding sebagai fitur diskrit pada perhitungan Mutual Information;
- menerapkan **L1-based Feature Selection** menggunakan Logistic Regression dengan `penalty="l1"`, `solver="saga"`, dan `C=0.1`;
- menerapkan fitur yang terpilih dari data training ke data testing.

Output tahap ini dibuat saat program dijalankan pada folder:

```text
data/split/
data/scaled/
data/mi_selected/
data/l1_selected/
```

File output tersebut dapat diregenerate dari pipeline dan tidak disimpan sebagai artefak utama repository.

### Modeling & Evaluasi — Abdullah Azzam ⏳

Dikerjakan pada branch `azzam/model-evaluation`:
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

Install dependency:

```bash
pip install -r requirements.txt
```

Download dataset asli dari Kaggle dan simpan sebagai:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Jalankan program:

```bash
python main.py
```

Saat ini `main.py` menjalankan pipeline sampai tahap feature selection:

```text
Preprocessing
→ Split 80/20
→ Min-Max Scaling
→ Mutual Information
→ L1-based Feature Selection
```

Tahap berikutnya adalah integrasi Decision Tree dan evaluasi model.

## Branch

- `main` — kode dan hasil yang sudah terintegrasi
- `falah/preprocessing` — dataset dan preprocessing, **selesai dan sudah masuk `main`**
- `adlha/feature-selection` — transformasi dan seleksi fitur, **selesai dan sudah masuk `main`**
- `azzam/model-evaluation` — modeling dan evaluasi
