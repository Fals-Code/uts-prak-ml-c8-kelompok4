# Audit penyiapan publik Week 09

Versi publik dibuat pada branch lokal codex/falah-week09-public dari main
47d49aa38c4f5d4b03456fc3135eadc2aacb7026, dalam worktree terpisah.
File dipindahkan melalui allowlist, bukan cherry-pick atau merge dua commit asli.
Commit 28e916f0365a9348785a072eaa5da9ff46a93710 dan
a279a62980c5320d54f3407eb1c855ab81664d13 tetap ada pada branch asli dan backup
privat; keduanya tidak menjadi ancestor versi publik.

preprocessing.py, make_evidence.py, requirements.txt, CSV Telco, dan teks lisensi
Telco disalin tanpa perubahan dari blob commit asli. Metode tertil, fitur,
penanganan nilai kosong, dan keterlacakan baris tidak diubah. Test hanya ditambah
skip bersyarat untuk dua metode yang memerlukan Mall lokal; isi pengujiannya tetap.

Tidak disertakan: seluruh CSV Mall, semua data/processed, semua outputs termasuk
head()/info() mentah, PNG scatter, screenshot output, metadata dan arsip lama.
Dokumentasi mempertahankan metodologi, batas kategori, jumlah anggota, dan hasil
statistik agregat tanpa mereproduksi record pelanggan. Semua output baru tetap
lokal dan diabaikan Git. Lihat PUBLICATION.md dan VALIDATION.md.

Tidak ada push, penghapusan remote, force push, merge ke main, perubahan stash,
atau perubahan Google Docs pada tahap ini. Daftar 12 branch lama tetap kandidat
pembersihan; eksekusinya membutuhkan persetujuan daftar terbaru.

## Audit akhir tanggung jawab Falah setelah migrasi

Branch aktif untuk perbaikan akhir adalah falah. Empat remote branch aktual
telah diperiksa; main, jamal, dan azzam tetap pada commit hasil integrasi Week 08.
Kode preprocessing, pemilihan fitur, tertil, dan tes tidak diimplementasikan ulang.
Satu koreksi kompatibilitas diperlukan: NumPy 2.5.3 memerlukan Python >=3.12,
sehingga petunjuk Python 3.11+ diperbaiki. Identitas repository dipisahkan dari
identitas sampul laporan yang telah diedit kelompok; sampul tidak diubah.

Dengan kedua input lokal pipeline berjalan dan empat tes lulus. Clone tanpa
Mall menjalankan dua tes mandiri dan melewati dua tes bergantung Mall dengan
alasan eksplisit. Pemanggilan pipeline tanpa Mall berhenti dengan exit 1 yang
menyebut file hilang, tanpa traceback atau klaim sukses. Hasil tetap lokal.

Laporan bagian Falah diperiksa terhadap hasil aktual, termasuk keterbatasan
tertil dan kategori nominal. Penyebutan Zoo/class.csv pada pembagian lama dan
daftar isi Week 08 tetap dicatat untuk finalisasi Azzam, sesuai instruksi terbaru
yang memakai Mall dan Telco. Bagian anggota lain tidak dikerjakan.
