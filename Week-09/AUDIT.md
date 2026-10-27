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
