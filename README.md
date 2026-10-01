# UTS Praktikum Machine Learning — Kelompok 4 C8

Project kelompok untuk **UTS Praktikum Machine Learning 2026**, Kelas **C8**, Kelompok **4**, dengan **Soal D**.

## Deskripsi

Project ini menggunakan **IBM HR Analytics Employee Attrition & Performance Dataset** untuk membangun model klasifikasi **Decision Tree** dengan dua metode seleksi fitur.

Dataset Kaggle:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Target klasifikasi: `Attrition` (`Yes` / `No`).

## Ketentuan Soal D

Alur utama yang dikerjakan:

1. Dataset memiliki atribut kategorikal dan numerik.
2. Preprocessing: deteksi missing value, duplikasi data, dan outlier.
3. Transformasi: Min-Max.
4. Split data: 80% training dan 20% testing.
5. Seleksi fitur menggunakan 2 metode:
   - Mutual Information
   - L1-based Feature Selection
6. Klasifikasi: Decision Tree.
7. Evaluasi:
   - Confusion Matrix
   - Accuracy
   - Precision
   - Recall
   - F1-Score

## Anggota dan Pembagian Task

### Ahmad Mathlaul Falah — Dataset & Preprocessing

- Menyiapkan dataset IBM HR Analytics Employee Attrition.
- Memahami atribut dan target `Attrition`.
- Mendeteksi dan menangani missing value.
- Mendeteksi dan menangani data duplikat.
- Mendeteksi dan menangani outlier menggunakan IQR.
- Melakukan encoding fitur kategorikal.
- Menyiapkan data hasil preprocessing untuk tahap berikutnya.

Branch kerja: `falah/preprocessing`

### Ah. Dliya'ul Adlha Jamalul Lail — Transformasi & Seleksi Fitur

- Melakukan split data 80% training dan 20% testing.
- Menerapkan Min-Max scaling.
- Mengimplementasikan Mutual Information.
- Mengimplementasikan L1-based Feature Selection.
- Mencatat fitur yang terpilih dari masing-masing metode.

Branch kerja: `adlha/feature-selection`

### Abdullah Azzam — Modeling & Evaluasi

- Melatih Decision Tree pada hasil kedua metode seleksi fitur.
- Melakukan testing model.
- Membuat Confusion Matrix.
- Menghitung Accuracy, Precision, Recall, dan F1-Score.
- Membandingkan hasil kedua metode seleksi fitur.

Branch kerja: `azzam/model-evaluation`

## Struktur Project

```text
uts-prak-ml-c8-kelompok4/
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

## Setup

Clone repository:

```bash
git clone https://github.com/Fals-Code/uts-prak-ml-c8-kelompok4.git
cd uts-prak-ml-c8-kelompok4
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependency:

```bash
pip install -r requirements.txt
```

Download dataset dari Kaggle lalu simpan file CSV di folder `data/`. File dataset tidak disimpan ke Git karena diabaikan oleh `.gitignore`.

## Workflow Git

Jangan mengerjakan task langsung di `main`.

Sebelum mulai:

```bash
git checkout main
git pull origin main
```

Pindah ke branch masing-masing:

```bash
git checkout falah/preprocessing
# atau
git checkout adlha/feature-selection
# atau
git checkout azzam/model-evaluation
```

Setelah mengerjakan perubahan:

```bash
git add .
git commit -m "feat: deskripsi perubahan"
git push origin nama-branch
```

Setelah itu buat Pull Request ke `main`.

## Aturan Kolaborasi

- `main` adalah sumber kode yang sudah terintegrasi.
- Masing-masing anggota bekerja di branch sendiri.
- Hindari mengedit modul milik anggota lain tanpa koordinasi.
- Pull perubahan terbaru dari `main` sebelum mulai sesi kerja baru.
- Merge dilakukan setelah kode dapat dijalankan dan tidak merusak bagian anggota lain.
- Laporan dikerjakan bersama melalui Google Docs, sedangkan source code dikelola melalui GitHub.

## Status

Baseline repository sudah disiapkan. Implementasi setiap modul dikerjakan sesuai pembagian task kelompok.
