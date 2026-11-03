# Praktikum Machine Learning

## Informasi Repository

Repository ini memuat praktikum Machine Learning kelompok C8 untuk UTS, Week 08–13, dan UAS. Setiap tugas memiliki folder dan dokumentasi masing-masing. Identitas atau pembagian tugas historis dapat berbeda antarpraktikum.

## Struktur Repository

- `Week-08/` — pipeline klasifikasi dan evaluasi.
- `Week-09/` — EDA dan preprocessing dataset untuk K-Modes.
- `Week-10/` — clustering; K-Means baseline tersedia.
- `Week-11/`, `Week-12/`, `Week-13/` — materi praktikum masing-masing; status Week 13 belum diverifikasi.
- `UTS/` — pekerjaan UTS.
- `UAS/` — dokumentasi UAS.

## Daftar Praktikum

| Pertemuan | Materi atau fokus | Lokasi | Status |
|---|---|---|---|
| UTS | Praktikum UTS | [`UTS/`](UTS/) | Tersedia |
| Week 08 | Klasifikasi: Naive Bayes, SVM, Decision Tree | [`Week-08/`](Week-08/) | Tersedia |
| Week 09 | EDA dan preprocessing K-Modes | [`Week-09/`](Week-09/) | Tersedia |
| Week 10 | Clustering: baseline K-Means; FCM dan DBSCAN direncanakan | [`Week-10/`](Week-10/) | Berjalan |
| Week 11 | Praktikum Week 11 | [`Week-11/`](Week-11/) | Belum diverifikasi |
| Week 12 | Praktikum Week 12 | [`Week-12/`](Week-12/) | Belum diverifikasi |
| Week 13 | Praktikum Week 13 | [`Week-13/`](Week-13/) | Belum diverifikasi |
| UAS | Praktikum UAS | [`UAS/`](UAS/) | Tersedia |

## Week 10 — Clustering

Week 10 menggunakan dataset Mall Customer dan Palmer Penguins. Baseline K-Means sudah tersedia dengan preprocessing dan evaluasi reusable. Fuzzy C-Means (FCM) dan DBSCAN masih dalam tahap pengerjaan; analisis perbandingan akhir juga belum selesai. Silhouette, Davies-Bouldin, dan Calinski-Harabasz digunakan untuk evaluasi internal; label Species pada Penguins hanya untuk evaluasi eksternal.

Lihat [dokumentasi Week 10](Week-10/README.md) untuk metodologi, hasil baseline, cara menjalankan, dan pembagian tugas.

## Cara Menjalankan

Petunjuk dan perintah teknis pipeline serta unittest tersedia di [Week-10/README.md](Week-10/README.md). Perintah Week 08 dan Week 09 tersedia di README masing-masing.

## Dokumentasi

- [Week 08](Week-08/README.md)
- [Week 09](Week-09/README.md)
- [Week 10](Week-10/README.md)
- [Week 11](Week-11/README.md)
- [Week 12](Week-12/README.md)
- [Week 13](Week-13/README.md)
- [UTS](UTS/README.md)
- [UAS](UAS/README.md)

## Anggota Kelompok Week 10

1. Ahmad Mathlaul Falah — 434241123
2. Ah. Dliya'ul Adlha Jamalul Lail — 434241126
3. Abdullah Azzam — 434241118

## Alur Git

Kerjakan tugas di branch masing-masing, lalu ajukan Pull Request ke `main` untuk review. Jangan commit virtual environment, cache, dataset lokal yang tidak diperlukan, atau file sementara.
