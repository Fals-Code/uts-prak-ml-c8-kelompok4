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
��� README.md
��� requirements.txt
��� main.py
��� app.py
��� src/
�   ��� data_pipeline.py
�   ��� naive_bayes.py
�   ��� svm_model.py
�   ��� evaluation.py
�   ��� comparison.py
��� tests/
�   ��� test_week08.py
��� outputs/
��� docs/
    ��� assets/
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

### Ahmad Mathlaul Falah — Naive Bayes, Evaluasi, dan Streamlit

Branch: `falah/week-08-naive-bayes` dan `falah/week-08-ui`

- Mengimplementasikan Naive Bayes, training, testing, dan prediksi.
- Membuat Confusion Matrix serta menghitung Accuracy, Precision, Recall, dan F1-Score.
- Setelah perbandingan selesai, membuat UI Streamlit untuk model dengan accuracy tertinggi.
- Menyiapkan screenshot, output, dan penjelasan bagian Naive Bayes serta UI.
- Membuat Pull Request untuk bagiannya dan menindaklanjuti konflik atau error pada modul Naive Bayes dan UI.

### Ah. Dliya'ul Adlha Jamalul Lail (Jamal) — SVM, Evaluasi, dan Finalisasi Laporan

Branch: `adlha/week-08-svm`

- Mengimplementasikan SVM, training, testing, dan prediksi.
- Membuat Confusion Matrix serta menghitung Accuracy, Precision, Recall, dan F1-Score.
- Menyiapkan screenshot, output, dan penjelasan bagian SVM.
- Menggabungkan materi laporan yang disiapkan anggota, merapikan format, dan menyiapkan ZIP sesuai ketentuan.
- Membuat Pull Request untuk bagiannya dan menindaklanjuti konflik atau error pada modul SVM.

### Abdullah Azzam — Data Pipeline dan Perbandingan Model

Branch: `azzam/week-08-pipeline` dan `azzam/week-08-comparison`

- Menyiapkan pipeline memakai dataset dan tahapan UTS serta split 80% training dan 20% testing.
- Memastikan seluruh model memakai data dan skenario pengujian yang konsisten.
- Menjalankan Decision Tree UTS sebagai pembanding apabila diperlukan.
- Membandingkan Naive Bayes, SVM, dan Decision Tree berdasarkan hasil eksperimen aktual.
- Menganalisis Confusion Matrix dan metrik ketiga model, termasuk kemampuan mengenali Attrition = Yes.
- Menyiapkan screenshot, output, dan penjelasan bagian pipeline serta perbandingan model.
- Membuat Pull Request untuk bagiannya dan menindaklanjuti konflik atau error pada modul pipeline/perbandingan.
- Pekerjaan tambahan di luar pembagian ini dibahas dan disepakati terlebih dahulu, tidak otomatis dialihkan kepada Azzam.

## Urutan Kerja

1. Azzam menyiapkan pipeline dan format data yang digunakan bersama.
2. Falah dan Jamal mengerjakan Naive Bayes dan SVM secara paralel.
3. Setelah kedua model selesai, Azzam membandingkan hasilnya dengan Decision Tree UTS.
4. Setelah model dengan accuracy tertinggi diketahui, Falah menyelesaikan UI Streamlit.
5. Setiap anggota menyerahkan screenshot, output, dan penjelasan untuk bagiannya.
6. Jamal menggabungkan materi laporan, merapikan format, dan menyiapkan ZIP pengumpulan.
7. Ketiga anggota memeriksa bagian masing-masing dan melakukan review akhir. Pemilik bagian menindaklanjuti temuannya.

Desain UI dan kerangka laporan boleh mulai dicicil lebih awal jika tidak bergantung pada hasil model.

## Progress

| Bagian | PIC | Status |
| --- | --- | --- |
| Data pipeline & split 80:20 | Abdullah Azzam | ✅ Selesai |
| Naive Bayes | Ahmad Mathlaul Falah | ✅ Selesai |
| Evaluasi Naive Bayes | Ahmad Mathlaul Falah | ✅ Selesai |
| SVM | Ah. Dliya'ul Adlha Jamalul Lail | ✅ Selesai |
| Evaluasi SVM | Ah. Dliya'ul Adlha Jamalul Lail | ✅ Selesai |
| Perbandingan NB vs SVM vs Decision Tree | Abdullah Azzam | ✅ Selesai |
| UI model terbaik | Ahmad Mathlaul Falah | 🚧 Proses |
| Pull Request dan integrasi | Falah, Jamal, dan Azzam (masing-masing untuk bagiannya) | ✅ Selesai |
| Penggabungan laporan dan ZIP | Ah. Dliya'ul Adlha Jamalul Lail | ⏳ Belum |
| Review akhir dan tindak lanjut temuan | Falah, Jamal, dan Azzam | ⏳ Belum |

## Aset Laporan

Kode penting, output eksperimen, Confusion Matrix, perbandingan model, dan preview prediksi tersimpan pada `docs/assets/`. Preview UI dibuat dari fungsi prediksi nyata yang juga digunakan oleh `app.py`; preview tersebut bukan bukti bahwa server Streamlit telah berhasil dijalankan pada environment ini.

[Kembali ke halaman utama](../README.md)

