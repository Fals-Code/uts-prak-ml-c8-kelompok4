# Week 08 - Klasifikasi

## Status

?? Implementasi dan evaluasi model selesai. Aplikasi Streamlit sudah tersedia, tetapi startup server belum dapat divalidasi pada environment kerja karena paket Streamlit tidak tersedia dan akses instalasi eksternal dibatasi.

## Tujuan

Week 08 membandingkan Naive Bayes, SVM, dan Decision Tree dari UTS pada dataset serta pipeline yang sama. Model dengan accuracy tertinggi digunakan oleh UI prediksi.

## Dataset dan Pipeline

Dataset yang digunakan adalah **IBM HR Analytics Employee Attrition & Performance** dengan target `Attrition`. Implementasi memakai `UTS/data/employee_attrition_preprocessed.csv`, yaitu output preprocessing UTS yang sudah tersimpan. Dataset mentah tidak digandakan ke folder Week 08.

Tahapan pipeline:

1. membaca 1.470 baris dan 51 fitur hasil preprocessing UTS;
2. membagi data secara terstratifikasi menjadi 80% training dan 20% testing dengan `random_state=42`;
3. melakukan Min-Max scaling, dengan scaler hanya di-fit pada data training;
4. memilih 20 fitur menggunakan Mutual Information yang dihitung hanya dari data training;
5. memberikan hasil split dan transformasi yang sama kepada seluruh model.

Hasil split terdiri atas 1.176 baris training dan 294 baris testing. Jalur Mutual Information dipilih agar perbandingan konsisten dengan salah satu konfigurasi utama pada UTS. Decision Tree dijalankan kembali dengan parameter UTS, yaitu `DecisionTreeClassifier(random_state=42)`, pada split yang sama.

## Model dan Parameter

- Naive Bayes: `GaussianNB()`.
- SVM: `SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)`.
- Decision Tree: konfigurasi UTS tanpa tuning tambahan.

Tidak dilakukan hyperparameter tuning besar karena tidak diminta pada soal.

## Hasil Evaluasi

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.6565 | 0.2750 | 0.7021 | 0.3952 |
| SVM | **0.8469** | **0.5833** | 0.1489 | 0.2373 |
| Decision Tree | 0.7823 | 0.3607 | 0.4681 | **0.4074** |

Seluruh angka di atas dihasilkan oleh `python main.py` dan tersimpan pada `outputs/metrics.json` serta `outputs/model_comparison.csv`.

### Confusion Matrix

Urutan matriks adalah `[[TN, FP], [FN, TP]]`.

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 160 | 87 | 14 | 33 |
| SVM | 242 | 5 | 40 | 7 |
| Decision Tree | 208 | 39 | 25 | 22 |

Naive Bayes mengenali 33 dari 47 pegawai yang benar-benar mengalami attrition. Recall-nya paling tinggi, tetapi menghasilkan 87 false positive sehingga precision dan accuracy lebih rendah.

SVM menghasilkan accuracy tertinggi dan hanya lima false positive. Namun, model ini melewatkan 40 kasus attrition sehingga recall kelas `Yes` rendah. Tingginya accuracy perlu dibaca bersama distribusi kelas yang tidak seimbang.

Decision Tree mengenali 22 kasus attrition dengan 39 false positive. Accuracy-nya berada di bawah SVM, tetapi F1-Score-nya paling tinggi karena keseimbangan precision dan recall lebih baik daripada dua model lain.

Sesuai aturan soal, **SVM dipilih untuk UI karena memiliki accuracy tertinggi, yaitu 0.8469**.

## Struktur Project

```text
Week-08/
|-- README.md
|-- requirements.txt
|-- main.py
|-- app.py
|-- src/
|   |-- data_pipeline.py
|   |-- naive_bayes.py
|   |-- svm_model.py
|   |-- evaluation.py
|   `-- comparison.py
|-- tests/
|   |-- test_week08.py
|   `-- test_falah_naive_bayes.py
|-- outputs/
`-- docs/
    |-- Laporan_Falah_Week08.md
    |-- generate_assets.py
    `-- assets/
```

## Instalasi

Dari root repository:

```bash
python -m venv .venv
```

Aktifkan virtual environment, lalu:

```bash
pip install -r Week-08/requirements.txt
```

## Menjalankan Eksperimen

```bash
python Week-08/main.py
```

Perintah tersebut menjalankan pipeline, ketiga model, evaluasi, dan pembuatan output. Untuk menjalankan tes integrasi:

```bash
python -m unittest discover -s Week-08/tests -v
```

## Menjalankan UI

```bash
streamlit run Week-08/app.py
```

UI memilih SVM dari hasil accuracy aktual. Pengguna dapat memilih satu sampel testing atau mengunggah CSV yang memiliki 51 kolom hasil preprocessing UTS. Input melewati scaler dan daftar fitur yang sama dengan proses training sebelum prediksi dijalankan.

## Pembagian Tugas

### Ahmad Mathlaul Falah — Data Pipeline, Naive Bayes, dan Evaluasi

Branch: `falah/week-08-pipeline-naive-bayes`

- Menyiapkan dan memvalidasi penggunaan pipeline bersama, split 80:20, Naive Bayes, serta evaluasinya.
- Menyerahkan hasil melalui fungsi `prepare_pipeline()` agar semua model memakai split dan transformasi yang sama.
- Menyiapkan screenshot, output, dan laporan bagian Falah.

### Ah. Dliya'ul Adlha Jamalul Lail (Jamal) — SVM, Decision Tree, dan Evaluasi

Branch: `adlha/week-08-svm`

- Melatih dan mengevaluasi SVM serta Decision Tree pembanding pada split pipeline yang sama.
- Menyiapkan screenshot, output, dan laporan bagian Jamal.

### Abdullah Azzam — Perbandingan Model, UI, dan Finalisasi

- Membandingkan ketiga model dan menganalisis hasil keseluruhan.
- Membuat UI Streamlit, menggabungkan laporan, dan menyiapkan ZIP pengumpulan.

## Serah Terima Pipeline

Jamal dapat menggunakan split dan transformasi yang sama dengan memanggil `prepare_pipeline()` dari `src/data_pipeline.py`. Fungsi tersebut menghasilkan `X_train`, `X_test`, `y_train`, dan `y_test` dalam objek `PipelineData`, menggunakan `test_size=0.2`, `random_state=42`, stratifikasi target, MinMaxScaler, dan 20 fitur Mutual Information. Scaler serta seleksi fitur di-fit hanya pada data training. Tidak perlu membuat split baru atau menyimpan empat CSV salinan.

Hasil evaluasi program tersimpan pada `outputs/metrics.json`, tabel metrik pada `outputs/model_comparison.csv`, dan Confusion Matrix tiap model dalam PNG pada `outputs/`. `outputs/console_output.txt` menyimpan ringkasan terminal. Hasil Naive Bayes berada di kunci `Naive Bayes` dalam JSON tersebut.

Jalankan dari root repository:

```bash
python Week-08/main.py
python -m unittest discover -s Week-08/tests -v
```

Laporan bagian Falah ada pada [docs/Laporan_Falah_Week08.md](docs/Laporan_Falah_Week08.md). Screenshot kode dan output masih perlu diambil manual melalui VS Code.

## Urutan Kerja

1. Falah menyiapkan pipeline bersama dan hasil Naive Bayes.
2. Jamal menggunakan pipeline yang sama untuk SVM dan Decision Tree.
3. Azzam membandingkan hasil ketiga model, menganalisisnya, dan menyiapkan UI Streamlit.
4. Setiap anggota menyiapkan screenshot, output, dan laporan bagiannya.
5. Azzam menggabungkan laporan dan menyiapkan ZIP pengumpulan.

Desain UI dan kerangka laporan boleh mulai dicicil lebih awal jika tidak bergantung pada hasil model.

## Progress

| Bagian | PIC | Status |
| --- | --- | --- |
| Data pipeline & split 80:20 | Ahmad Mathlaul Falah | ✅ Selesai |
| Naive Bayes | Ahmad Mathlaul Falah | ✅ Selesai |
| Evaluasi Naive Bayes | Ahmad Mathlaul Falah | ✅ Selesai |
| SVM | Ah. Dliya'ul Adlha Jamalul Lail | ✅ Selesai |
| Evaluasi SVM | Ah. Dliya'ul Adlha Jamalul Lail | ✅ Selesai |
| Perbandingan NB vs SVM vs Decision Tree | Abdullah Azzam | ✅ Selesai |
| UI Streamlit | Abdullah Azzam | 🚧 Perlu validasi |
| Pull Request dan integrasi | Falah, Jamal, dan Azzam (masing-masing untuk bagiannya) | 🚧 Berjalan |
| Penggabungan laporan dan ZIP | Abdullah Azzam | ⏳ Belum |
| Review akhir dan tindak lanjut temuan | Falah, Jamal, dan Azzam | ⏳ Belum |

## Aset Laporan

Hasil evaluasi yang dapat diperiksa ulang berada di `outputs/`. Confusion Matrix PNG dibuat langsung dari hasil prediksi Python; JPG pasangannya di `docs/assets/` merupakan salinan untuk keperluan visual laporan. `ui_prediction_preview.jpg` adalah preview statis dari fungsi prediksi, bukan screenshot aplikasi atau bukti server Streamlit berjalan. Screenshot kode dan terminal akan diambil manual melalui VS Code.

UI Streamlit dan penggabungan laporan/ZIP menjadi tanggung jawab Azzam dan belum dinyatakan final di repository ini.

[Kembali ke halaman utama](../README.md)

