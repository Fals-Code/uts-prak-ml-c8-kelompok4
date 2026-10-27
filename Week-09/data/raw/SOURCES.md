# Sumber dan penempatan input

## Mall Customers — hanya input lokal

Materi praktikum pengguna menggunakan Mall_Customers.csv (200 baris, lima kolom).
Audit membandingkan lampiran dengan
[Kaggle versi 1](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python):
berkasnya identik byte demi byte. SHA-256 acuan:
`416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77`.

[Metadata Kaggle](https://www.kaggle.com/api/v1/datasets/view/vjchoudhary7/customer-segmentation-tutorial-in-python)
yang diperiksa pada 27 Oktober 2026 mencantumkan lisensi
**Other (specified in description)**. Deskripsi menyebut materi kursus, tetapi
izin redistribusi publik belum dapat dikonfirmasi. Tidak dinyatakan CC0.

Peroleh berkas melalui materi yang berhak Anda akses atau halaman sumber tersebut,
ikuti ketentuannya, kemudian tempatkan di Week-09/data/raw/Mall_Customers.csv.
Berkas mentah dan semua turunannya tetap lokal dan diabaikan Git. Tidak ada
unduhan otomatis atau versi pengganti. Skema: CustomerID, Gender, Age,
Annual Income (k$), Spending Score (1-100). Nama/skema dan checksum bukan isi
record pelanggan.

## Telco Customer Churn — disertakan dengan lisensi sumber

CSV Telco-Customer-Churn.csv berasal dari lampiran pengguna dan telah diverifikasi
terhadap [sumber IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).
Jumlah acuan 7.043 baris dan 21 kolom; ukuran 977.501 byte. SHA-256:
`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
Git blob setelah normalisasi CRLF ke LF dalam memori identik dengan blob IBM
`3de7a612d1609f25f21a455bda77948729369002`. Berkas lokal tidak dinormalisasi.

Repository IBM menyertakan [Apache License 2.0](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE).
Salinan lengkap dipertahankan dalam TELCO_SOURCE_LICENSE.txt. CSV mentah tidak
diubah. Transformasi menghasilkan berkas terpisah di data/processed secara lokal
dan dijelaskan pada README. Atribusi ini tidak memberikan lisensi bagi dataset Mall.
