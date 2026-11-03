# Data Week 10

## Mall Customer

Lokasi: `raw/Mall_Customers.csv`. Sumber salinan: objek Git `codex/falah-week09-preprocessing:Week-09/data/raw/Mall_Customers.csv` (commit `28e916f`), disalin byte-identik tanpa mengubah Week 9. Dimensi 200 × 5. Header asli: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`. Missing value: 0 di semua kolom; duplikasi baris penuh: 0. `Age`, `Annual Income (k$)`, dan `Spending Score (1-100)` terbaca numerik. SHA-256: `416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77`.

## Palmer Penguins

Lokasi: `raw/Penguins.csv`. Dataset diberikan pengguna dan dinyatakan berasal dari Kaggle; URL dan lisensi spesifik belum terverifikasi. Dimensi 344 × 19. Header asli dipertahankan: `studyName`, `Sample Number`, `Species`, `Region`, `Latitude `, `Longitude`, `Island`, `Stage`, `Individual ID`, `Clutch Completion`, `Date Egg`, `Culmen Length (mm)`, `Culmen Depth (mm)`, `Flipper Length (mm)`, `Body Mass (g)`, `Sex`, `Delta 15 N (o/oo)`, `Delta 13 C (o/oo)`, `Comments`. Missing: masing-masing ukuran tubuh 2; `Sex` 10; `Delta 15 N (o/oo)` 14; `Delta 13 C (o/oo)` 13; `Comments` 290; kolom lain 0. Duplikasi baris penuh: 0. Empat ukuran tubuh berhasil dikonversi numerik pada 342 baris masing-masing; nilai nonmissing yang gagal dikonversi: 0. SHA-256: `5ea908ae6fd5363b6d649f9eecdd63c010caaa389ee9498483e9ec30647dee2d`.

Nilai `Species`: Adelie Penguin (Pygoscelis adeliae), Chinstrap penguin (Pygoscelis antarctica), Gentoo penguin (Pygoscelis papua). Species hanya digunakan untuk evaluasi eksternal, bukan input clustering. Pemetaan nama kolom ke nama kerja dapat dilakukan dalam data kerja saat preprocessing; CSV sumber tidak diubah.

## Penggunaan dan batasan

K-Means adalah baseline baru Week 10; implementasi K-Means Week 9 tidak ditemukan. Rencananya satu pipeline reusable dipakai untuk kedua dataset dengan preprocessing khusus sesuai struktur masing-masing (fitur kategorikal Mall dan missing value/label Penguins). Baseline kini memproses salinan data kerja: Mall 200 baris dan Penguins 342 complete-case, keduanya StandardScaler. Hasil aktual ada di ../results; data mentah tetap utuh.
