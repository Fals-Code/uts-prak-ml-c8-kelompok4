# Laporan Bagian Falah — Praktikum Machine Learning Week 08

## A. Persiapan Dataset

Dataset yang digunakan adalah IBM HR Analytics Employee Attrition & Performance. Target klasifikasinya adalah `Attrition`: nilai 0 menyatakan pegawai tidak mengalami attrition (`No`), sedangkan nilai 1 menyatakan pegawai mengalami attrition (`Yes`). Week 08 menggunakan kembali `UTS/data/employee_attrition_preprocessed.csv`, yaitu keluaran preprocessing yang sudah tersedia dari UTS.

File tersebut berisi 1.470 baris, 51 fitur, dan satu kolom target. Pemeriksaan pada file yang dipakai untuk eksperimen menunjukkan tidak ada missing value maupun baris duplikat. Distribusi target terdiri atas 1.233 pegawai `No` dan 237 pegawai `Yes`, sehingga kelas target tidak seimbang.

**[TEMPAT SCREENSHOT F1: Pembacaan dataset dan preprocessing]**

*Keterangan gambar: Pembacaan data hasil preprocessing UTS dan pemeriksaan target serta kondisi data yang digunakan pada Week 08.*



## B. Preprocessing Data

Tahap UTS membaca dataset IBM HR, menangani missing value numerik dengan median dan kategorikal dengan modus, menghapus baris duplikat, membuang `EmployeeNumber` serta kolom konstan, dan melakukan capping outlier numerik menggunakan batas IQR. Target `Attrition` dipetakan menjadi 0 dan 1, sedangkan fitur kategorikal diubah menjadi kolom numerik melalui One-Hot Encoding. Hasilnya disimpan sebagai CSV dan dipakai kembali pada Week 08 tanpa preprocessing ulang.

Ada batasan metodologis yang perlu dicatat. Preprocessing UTS dilakukan sebelum pembagian data Week 08, sehingga statistik imputasi, batas capping IQR, dan kategori One-Hot Encoding terbentuk sebelum split. Tahapan upstream tersebut berpotensi menggunakan informasi dari keseluruhan dataset. Arsip UTS tidak diubah; pada Week 08, MinMaxScaler dan perhitungan Mutual Information di-fit hanya pada data training.

## C. Pembagian Data

Fungsi `prepare_pipeline()` memisahkan fitur dari target dan membagi data dengan `test_size=0.2`, `random_state=42`, serta `stratify=y`. Pembagian ini menghasilkan 1.176 data training dan 294 data testing. Karena menggunakan stratifikasi, distribusi kelas pada training adalah 986 `No` dan 190 `Yes`, sedangkan pada testing terdapat 247 `No` dan 47 `Yes`.

Fungsi tersebut mengembalikan objek `PipelineData` yang memuat `X_train`, `X_test`, `y_train`, dan `y_test`, termasuk indeks dan nama fitur yang tetap selaras. Model berikutnya sebaiknya menggunakan keluaran ini agar evaluasi berjalan pada split yang sama.

**[TEMPAT SCREENSHOT F2: Pembagian data training dan testing]**

*Keterangan gambar: Parameter split 80:20, random state 42, stratifikasi target, dan jumlah baris training/testing.*



## D. Transformasi dan Seleksi Fitur

Pipeline menerapkan `MinMaxScaler` pada fitur. Scaler di-fit menggunakan `X_train` saja, kemudian dipakai untuk mentransformasi training dan testing. Setelah itu, Mutual Information menghitung skor dari `X_train` dan `y_train`; 20 fitur dengan skor tertinggi dipilih. Penanda fitur biner One-Hot Encoding juga ditentukan dari data training, dan daftar fitur terpilih yang sama digunakan pada data testing.

**[TEMPAT SCREENSHOT F3: MinMaxScaler dan feature selection]**

*Keterangan gambar: Bagian pipeline yang melakukan fit scaler pada training dan seleksi 20 fitur Mutual Information dari data training.*



## E. Implementasi Naive Bayes

Klasifikasi menggunakan Gaussian Naive Bayes melalui `GaussianNB()`. Model ini menerima fitur numerik, termasuk fitur numerik asli dan hasil encoding kategori. Gaussian Naive Bayes memperkirakan distribusi setiap fitur pada masing-masing kelas dengan asumsi Gaussian; untuk fitur hasil One-Hot Encoding, asumsi tersebut menjadi pendekatan terhadap nilai biner.

Fungsi `train_naive_bayes()` melatih model menggunakan `X_train` dan `y_train`. Selanjutnya, `predict_naive_bayes()` menghasilkan prediksi dari `X_test`. Data testing tidak digunakan untuk melatih model.

**[TEMPAT SCREENSHOT F4: Kode training Naive Bayes]**

*Keterangan gambar: Fungsi `train_naive_bayes()` yang membuat GaussianNB dan memanggil `fit()` pada data training.*



**[TEMPAT SCREENSHOT F5: Kode prediction dan evaluasi]**

*Keterangan gambar: Pemanggilan prediksi Naive Bayes untuk data testing dan fungsi evaluasi metrik.*



## F. Evaluasi Naive Bayes

Evaluasi dihitung pada 294 baris data testing. Confusion Matrix menggunakan urutan kelas 0 lalu 1, yaitu `No` dan `Yes`, untuk baris aktual maupun kolom prediksi.

| Aktual \ Prediksi | No | Yes |
| --- | ---: | ---: |
| No | 160 | 87 |
| Yes | 14 | 33 |

| Metrik | Hasil |
| --- | ---: |
| Accuracy | 0.6565 |
| Precision | 0.2750 |
| Recall | 0.7021 |
| F1-Score | 0.3952 |

Untuk Precision, Recall, dan F1-Score, kelas positif adalah `Attrition = Yes` (label 1). Angka presisi lengkap serta confusion matrix juga tercatat pada `Week-08/outputs/metrics.json` di bawah kunci `Naive Bayes`. Program menghasilkan gambar confusion matrix pada `Week-08/outputs/confusion_matrix_naive_bayes.png`.

**[TEMPAT SCREENSHOT F6: Output terminal hasil Naive Bayes]**

*Keterangan gambar: Blok output Gaussian Naive Bayes dari eksekusi program, mencakup confusion matrix dan empat metrik evaluasi.*



**[TEMPAT SCREENSHOT F7: Confusion Matrix Naive Bayes]**

*Keterangan gambar: Confusion matrix Naive Bayes dengan label aktual dan prediksi No/Yes dari data testing.*



## G. Analisis Hasil

Model memprediksi 33 dari 47 pegawai yang benar-benar mengalami attrition sebagai `Yes`, sementara 14 pegawai `Yes` masih salah diprediksi sebagai `No`. Hasil ini menghasilkan Recall 0.7021. Model juga memberikan 87 prediksi `Yes` yang keliru; dari 120 prediksi `Yes`, hanya 33 yang benar sehingga Precision sebesar 0.2750. Dengan demikian, pada data uji ini model cukup banyak menangkap kasus attrition, tetapi prediksi positifnya masih sering keliru. F1-Score 0.3952 merangkum keseimbangan Precision dan Recall, sedangkan Accuracy mencapai 0.6565.

## H. Ringkasan Pekerjaan dan Serah Terima

Pipeline bersama tersedia di `Week-08/src/data_pipeline.py`, dengan fungsi `prepare_pipeline()` sebagai titik masuk untuk memperoleh satu split dan transformasi yang sama. Implementasi Gaussian Naive Bayes, prediksi, serta evaluasi berada di `Week-08/src/naive_bayes.py` dan `Week-08/src/evaluation.py`. Hasil seluruh model yang dibuat oleh program tersedia di `Week-08/outputs/metrics.json`; hasil Naive Bayes dapat dibaca dari entri `Naive Bayes`. Pengujian integrasi tersedia di `Week-08/tests/`, dan pengujian tambahan khusus Naive Bayes mencakup split, fit transformasi pada training, metrik, dan reproducibility.

## Panduan Screenshot melalui VS Code

Buka root repository di VS Code. Jalankan eksperimen dari terminal root dengan perintah `\.venv\Scripts\python.exe Week-08\main.py`; program memperbarui hasil pada `Week-08/outputs/`. Untuk pengujian, jalankan `\.venv\Scripts\python.exe -m unittest discover -s Week-08\tests -v`.

| Gambar | File/bagian yang dibuka | Perintah | Informasi yang perlu terlihat |
| --- | --- | --- | --- |
| F1 | `UTS/src/preprocessing.py` pada `preprocess_data()` dan `Week-08/src/data_pipeline.py` pada `load_preprocessed()` | Jalankan `\.venv\Scripts\python.exe Week-08\main.py` dari root | Pembacaan dataset mentah di UTS, penyimpanan CSV hasil preprocessing, dan pembacaan kembali CSV tersebut pada Week 08. Ambil dua tampilan bila kode tidak muat di satu layar. |
| F2 | `Week-08/src/data_pipeline.py`, fungsi `prepare_pipeline()` | Perintah eksperimen di atas | `test_size=0.2`, `random_state=42`, `stratify=y`; terminal menunjukkan 1.176 training dan 294 testing. |
| F3 | `Week-08/src/data_pipeline.py`, bagian scaler dan Mutual Information | Perintah eksperimen di atas | `fit_transform()` hanya untuk training, `transform()` untuk testing, serta seleksi 20 fitur. |
| F4 | `Week-08/src/naive_bayes.py`, fungsi `train_naive_bayes()` | Perintah eksperimen di atas | Pembuatan `GaussianNB()` dan `fit(X_train, y_train)`. |
| F5 | `Week-08/src/naive_bayes.py` pada `predict_naive_bayes()` dan `Week-08/src/evaluation.py` pada `evaluate_predictions()` | Perintah eksperimen di atas | Prediksi untuk X testing dan metrik evaluasi dengan label positif 1 (`Yes`). |
| F6 | Terminal VS Code | `\.venv\Scripts\python.exe Week-08\main.py` | Bagian output Naive Bayes: confusion matrix, Accuracy, Precision, Recall, dan F1-Score. |
| F7 | `Week-08/outputs/confusion_matrix_naive_bayes.png` | Perintah eksperimen di atas untuk memperbarui gambar | Plot confusion matrix yang dibuat dari prediksi program, label No/Yes, dan jumlah pada setiap sel. |

Screenshot akan ditambahkan sendiri melalui VS Code setelah file atau output terkait dibuka. Nilai hasil pada laporan mengikuti keluaran program yang terakhir dijalankan.
