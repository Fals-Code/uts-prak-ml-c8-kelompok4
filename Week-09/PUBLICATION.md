# Batas distribusi publik

Versi ini menyiapkan kode dan analisis agregat untuk ditinjau. Ini bukan izin
redistribusi dataset Mall. CSV Mall dari materi/Kaggle harus disediakan pengguna
secara lokal sesuai ketentuan sumber.

## Allowlist berkas Week-09 dalam commit publik

- .gitattributes
- .gitignore
- AUDIT.md
- LAPORAN_FALAH.md
- PUBLICATION.md
- README.md
- VALIDATION.md
- preprocessing.py
- make_evidence.py
- test_preprocessing.py
- requirements.txt
- data/raw/SOURCES.md
- data/raw/Telco-Customer-Churn.csv
- data/raw/TELCO_SOURCE_LICENSE.txt

## Tetap lokal, tidak dipublikasikan

- data/raw/Mall_Customers.csv dan alias/arsip salinannya.
- Seluruh data/processed: mall_kmodes.csv, mall_reference.csv,
  telco_kmodes.csv, telco_reference.csv.
- Seluruh outputs: inspection.txt berisi head(), metadata, distribusi, seluruh
  gambar EDA, serta evidence/log/screenshot. Ini termasuk scatter yang menampilkan
  titik pelanggan dan output gambar yang memuat contoh baris Mall.
- Bundle backup, stash, ZIP, dan riwayat asli yang memuat data tersebut.

Pengecualian seluruh output dipilih agar data tingkat baris dan turunan Mall tidak
terbawa tanpa sengaja. Teks laporan/README hanya memuat hasil agregat dan metode.
Telco dan lisensi sumbernya tetap disertakan; hasil Telco dapat dibuat ulang lokal.

Jangan gunakan git add -f pada berkas lokal. Jangan push --all atau --mirror:
repository lokal juga menyimpan branch asli berisi Mall. Kelak push hanya ref
branch publik yang telah diverifikasi. Menghapus data pada commit paling akhir
saja tidak cukup bila commit ancestor masih memuatnya.

## Menjalankan ulang

Letakkan Mall pada path di SOURCES.md, instal requirements.txt, lalu jalankan
preprocessing.py. Hasil lokal Jamal tetap identik dengan metode semula dan dapat
dibaca dengan pd.read_csv(..., dtype=str). File reference tetap terpisah.
make_evidence.py boleh dijalankan lokal; bukti yang dibuat tidak otomatis aman
untuk dipublikasikan. Tanpa Mall, dua tes independen tetap berjalan, sedangkan
dua tes yang membutuhkan input itu menyatakan SKIP.

## Catatan peninjauan

Audit ini memisahkan kode, hasil agregat, dan distribusi record pelanggan; bukan
penetapan hak cipta menyeluruh. Perubahan berikutnya memerlukan pemeriksaan isi
dan riwayat ulang, termasuk nama alternatif, notebook, lampiran, dan gambar.
