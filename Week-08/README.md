# Week 08 — Klasifikasi

## Status

🚧 Sedang dikerjakan

## Tugas

1. Tugas dikerjakan secara kelompok sesuai kelompok UTS.
2. Membuat program klasifikasi menggunakan Naive Bayes.
3. Membuat program klasifikasi menggunakan SVM.
4. Menggunakan dataset yang sama dengan UTS.
5. Mengikuti tahapan pengerjaan UTS.
6. Membagi data menjadi 80% training dan 20% testing.
7. Menganalisis hasil Confusion Matrix.
8. Membandingkan performa Naive Bayes, SVM, dan Decision Tree hasil UTS.
9. Membuat UI/UX menggunakan Streamlit atau Flask untuk metode dengan accuracy tertinggi.
10. Membuat laporan berisi screenshot coding, hasil output program, dan penjelasan.
11. Mengumpulkan file Python dan laporan.
12. Mengikuti penamaan file pada soal: `Tugas klasifikasi_NIM.Zip`.

Model terbaik belum ditentukan. Pemilihannya harus berdasarkan hasil pengujian aktual seluruh model.

## Dataset

Menggunakan **IBM HR Analytics Employee Attrition & Performance** yang sama dengan UTS, dengan target `Attrition`.

Sumber dan petunjuk dataset tersedia di [dokumentasi dataset UTS](../UTS/data/README.md). Jangan commit dataset mentah duplikat tanpa kebutuhan.

## Metode

- Naive Bayes
- SVM
- Decision Tree hasil UTS sebagai pembanding

## Split Data

- 80% training
- 20% testing

## Evaluasi

- Membuat dan menganalisis Confusion Matrix.
- Menggunakan metrik evaluasi untuk membandingkan model.
- Menentukan model dengan accuracy tertinggi berdasarkan hasil aktual.

Nilai evaluasi belum tersedia dan tidak dicantumkan sebelum implementasi serta pengujian selesai.

## UI

Model dengan accuracy tertinggi berdasarkan pengujian aktual akan digunakan pada UI. Soal mengizinkan **Streamlit atau Flask**; pilihan framework belum ditentukan.

## Pembagian Tugas

### Ahmad Mathlaul Falah — Naive Bayes & Evaluasi

Branch: `falah/week-08-naive-bayes`

Tanggung jawab:

- mengimplementasikan Naive Bayes;
- melakukan training dan testing menggunakan data pipeline Week 08;
- menghasilkan prediction;
- membuat Confusion Matrix;
- menghitung hasil evaluasi yang diperlukan;
- menyiapkan output dan screenshot bagian Naive Bayes;
- menyiapkan penjelasan bagian Naive Bayes untuk laporan.

### Ah. Dliya'ul Adlha Jamalul Lail — SVM & Evaluasi

Branch: `adlha/week-08-svm`

Tanggung jawab:

- mengimplementasikan SVM;
- melakukan training dan testing menggunakan data pipeline Week 08;
- menghasilkan prediction;
- membuat Confusion Matrix;
- menghitung hasil evaluasi yang diperlukan;
- menyiapkan output dan screenshot bagian SVM;
- menyiapkan penjelasan bagian SVM untuk laporan.

### Abdullah Azzam — Data Pipeline, Integrasi & UI

Branch: `azzam/week-08-pipeline-ui`

Tanggung jawab:

- menyiapkan dataset dan pipeline berdasarkan tahapan UTS;
- memastikan split 80% training dan 20% testing;
- memastikan Naive Bayes dan SVM menggunakan pipeline serta data yang konsisten;
- mengintegrasikan hasil Naive Bayes dan SVM dengan hasil Decision Tree UTS;
- setelah hasil perbandingan tersedia, membuat UI menggunakan Streamlit atau Flask untuk model dengan accuracy tertinggi;
- menyiapkan output dan screenshot pipeline, integrasi, dan UI;
- menyiapkan penjelasan bagian tersebut untuk laporan.

## Dikerjakan Bersama

- memastikan dataset sama dengan UTS;
- menjalankan integrasi seluruh pipeline;
- memastikan split 80:20;
- membandingkan Naive Bayes, SVM, dan Decision Tree UTS;
- menganalisis Confusion Matrix;
- menentukan model dengan accuracy tertinggi berdasarkan hasil aktual;
- memastikan UI menggunakan model terbaik tersebut;
- melakukan review Pull Request;
- menyusun laporan akhir;
- melakukan final review sebelum pengumpulan.

## Workflow Git

```text
falah/week-08-naive-bayes
adlha/week-08-svm
azzam/week-08-pipeline-ui
        ↓
Pull Request
        ↓
review
        ↓
main
```

## Progress

| Bagian | PIC | Status |
| --- | --- | --- |
| Data pipeline & split 80:20 | Abdullah Azzam | ⏳ Belum |
| Naive Bayes | Ahmad Mathlaul Falah | ⏳ Belum |
| Evaluasi Naive Bayes | Ahmad Mathlaul Falah | ⏳ Belum |
| SVM | Ah. Dliya'ul Adlha Jamalul Lail | ⏳ Belum |
| Evaluasi SVM | Ah. Dliya'ul Adlha Jamalul Lail | ⏳ Belum |
| Perbandingan NB vs SVM vs Decision Tree | Bersama | ⏳ Belum |
| UI model terbaik | Abdullah Azzam | ⏳ Belum |
| Integrasi GitHub | Bersama | ⏳ Belum |
| Laporan | Bersama | ⏳ Belum |
| Final Review | Bersama | ⏳ Belum |

Status yang digunakan: **⏳ Belum**, **🚧 Proses**, dan **✅ Selesai**.

## Hasil

Belum tersedia. Bagian ini diisi setelah implementasi dan pengujian selesai.

## Cara Menjalankan

Diisi setelah implementasi dan dependency Week 08 tersedia.

[Kembali ke halaman utama](../README.md)
