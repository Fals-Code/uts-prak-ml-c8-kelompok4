# Week 08 — Klasifikasi Employee Attrition

## Tujuan dan Dataset

Tugas ini mengklasifikasikan `Attrition` pada dataset IBM HR Analytics Employee Attrition & Performance. Week 08 menggunakan kembali `UTS/data/employee_attrition_preprocessed.csv`, berisi 1.470 baris, 51 fitur, dan target `Attrition` (0 = No, 1 = Yes). File ini sudah dipreproses pada UTS.

## Pembagian Tugas

- **Falah:** pipeline, preprocessing, split 80:20, Naive Bayes, evaluasi, dan laporan bagian Falah.
- **Jamal:** SVM, Decision Tree pembanding, evaluasi kedua model, dan laporan bagian Jamal.
- **Azzam:** perbandingan model, UI Streamlit, penggabungan laporan, dan ZIP pengumpulan.

## Program

```text
Week-08/
├── README.md
├── requirements.txt
├── main.py
├── app.py
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── naive_bayes.py
│   ├── svm_model.py
│   ├── evaluation.py
│   └── comparison.py
├── tests/
└── outputs/
    ├── metrics.json
    ├── split/
    │   ├── X_train.csv
    │   ├── X_test.csv
    │   ├── y_train.csv
    │   └── y_test.csv
    └── models/
        └── naive_bayes.joblib
```

`prepare_pipeline()` di `src/data_pipeline.py` menghasilkan split yang dipakai semua model: 80% training dan 20% testing, `random_state=42`, stratifikasi target, MinMaxScaler, dan 20 fitur hasil Mutual Information. Scaler dan seleksi fitur hanya di-fit pada training.

## Instalasi dan Menjalankan

Dari root repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Week-08\requirements.txt
.\.venv\Scripts\python.exe Week-08\main.py
```

Program menjalankan eksperimen pada pipeline bersama dan membuat output di `Week-08/outputs/`. Empat CSV split tersimpan di `outputs/split/`; indeks asli disimpan pada kolom `index`, sedangkan urutan 20 fitur dipertahankan pada header `X_train.csv` dan `X_test.csv`. `y_train.csv` dan `y_test.csv` berisi kolom `Attrition`.

Jamal dapat membaca CSV tanpa preprocessing atau split ulang:

```python
import pandas as pd

X_train = pd.read_csv("Week-08/outputs/split/X_train.csv", index_col="index")
X_test = pd.read_csv("Week-08/outputs/split/X_test.csv", index_col="index")
y_train = pd.read_csv("Week-08/outputs/split/y_train.csv", index_col="index")["Attrition"]
y_test = pd.read_csv("Week-08/outputs/split/y_test.csv", index_col="index")["Attrition"]
```

Model hasil training Gaussian Naive Bayes tersimpan di `outputs/models/naive_bayes.joblib`. Azzam dapat memuatnya dengan `joblib.load("Week-08/outputs/models/naive_bayes.joblib")` dan memberikan `X_test` yang kolom serta urutannya tetap dipertahankan.

Pengujian:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Week-08\tests -v
```

## Metode dan Hasil Naive Bayes

Model Falah menggunakan `GaussianNB()`. Kelas positif pada Precision, Recall, dan F1-Score adalah `Attrition = Yes`.

| Hasil | Nilai |
| --- | ---: |
| Confusion Matrix (No, Yes) | `[[160, 87], [14, 33]]` |
| Accuracy | 0.6565 |
| Precision | 0.2750 |
| Recall | 0.7021 |
| F1-Score | 0.3952 |

Pada 47 pegawai dengan attrition di data testing, model mengenali 33 dan melewatkan 14. Dari 120 prediksi `Yes`, 87 merupakan false positive.

## Batasan Metodologis dan Status

Imputasi, capping outlier, dan One-Hot Encoding pada file hasil UTS dilakukan sebelum split Week 08. Tahapan tersebut berpotensi memakai informasi dari seluruh data. Arsip UTS tidak diubah; pada pipeline Week 08, scaler dan Mutual Information tetap hanya di-fit pada data training.

Pipeline dan evaluasi Naive Bayes Falah sudah divalidasi. Hasil SVM dan Decision Tree perlu ditinjau Jamal; perbandingan, UI Streamlit, penggabungan laporan, dan ZIP menjadi pekerjaan Azzam. UI dan laporan gabungan belum dinyatakan final.
