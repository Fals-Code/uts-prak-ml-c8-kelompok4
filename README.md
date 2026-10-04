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
- melakukan One-Hot Encoding pada fitur kategorikal.

Hasil preprocessing saat ini:
- jumlah data: **1.470 baris**;
- missing value: **0**;
- data duplikat: **0**;
- fitur sebelum encoding: **30**;
- fitur setelah encoding: **51**;
- distribusi target: **1.233 No** dan **237 Yes**.

### Transformasi & Seleksi Fitur — Ah. Dliya'ul Adlha Jamalul Lail ⏳

Dikerjakan pada branch `adlha/feature-selection`:
- split data 80% training dan 20% testing;
- Min-Max scaling;
- Mutual Information;
- L1-based Feature Selection.

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
│   └── README.md
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

Download dataset dari Kaggle dan simpan sebagai:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Jalankan program:

```bash
python main.py
```

Saat ini `main.py` menjalankan tahap preprocessing. Pipeline akan dilanjutkan setelah bagian transformasi, seleksi fitur, modeling, dan evaluasi diintegrasikan.

## Branch

- `main` — kode yang sudah terintegrasi
- `falah/preprocessing` — dataset dan preprocessing, **selesai dan sudah masuk `main`**
- `adlha/feature-selection` — transformasi dan seleksi fitur
- `azzam/model-evaluation` — modeling dan evaluasi
