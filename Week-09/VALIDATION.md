# Validasi versi persiapan publik

Pengujian pada 27 Oktober 2026 menggunakan Python 3.14.6, Pandas 3.0.6,
NumPy 2.5.3, dan Matplotlib 3.11.2. Ini ringkasan agregat, bukan salinan head()
atau screenshot baris pelanggan. Tidak ada pengujian clustering pada tahap ini.

## Dengan input Mall lokal

Perintah yang dijalankan dari root worktree:

```powershell
python Week-09/preprocessing.py
python -m unittest discover -s Week-09 -p test_preprocessing.py -v
```

Pipeline selesai tanpa error. Mall menghasilkan 200 × 4 fitur kategorikal dan
Telco 7.043 × 12. Tidak ada missing pada fitur final, identitas/Churn tidak masuk
matriks fitur, serta jumlah dan urutan baris dipertahankan. Hash data asli tidak
berubah. Empat gambar EDA dibuat hanya sebagai hasil lokal yang diabaikan Git.

Keempat metode tes lulus (11,458 detik pada eksekusi verifikasi):

- test_binary_status_is_not_a_continuous_outlier
- test_binning_ties_constant_and_missing
- test_pipeline_preserves_rows_and_repeats_exactly
- test_rejects_invalid_input_without_silent_imputation

Tes integrasi menjalankan pipeline dua kali dan membandingkan CSV, JSON,
teks pemeriksaan, dan PNG byte demi byte. Berkas keluaran dibandingkan juga
dengan implementasi asli, tanpa mengubah metode preprocessing.

## Tanpa input Mall lokal

Pengujian sebelum input Mall diletakkan menghasilkan dua tes lulus dan dua
SKIP beralasan. Tidak ada kegagalan. Dua SKIP ini tidak dilaporkan sebagai PASS.
Tes mandiri yang tetap berjalan ialah pemeriksaan kategori biner Telco dan kasus
tepi tertil. Pengujian yang memerlukan Mall dijalankan penuh setelah input ada.

## Distribusi dan riwayat

Allowlist publik tercantum pada PUBLICATION.md. Dataset Mall, seluruh
data/processed, dan seluruh outputs tidak masuk commit. Kode preprocessing.py
identik dengan blob pada commit implementasi asli. Penyesuaian tes hanya
menambahkan skip bersyarat pada dua metode yang membutuhkan input lokal.

Audit commit dan clone publik dilakukan terpisah dari berkas ini; hasil rinci
serta SHA persiapan dicatat pada artefak migrasi privat. Jangan mengartikan
berbagi kode/hasil agregat sebagai izin mendistribusikan kembali dataset Mall.
