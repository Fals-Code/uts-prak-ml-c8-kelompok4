# Praktikum Machine Learning — C8 Kelompok 4

Repository kerja kelompok Praktikum Machine Learning 2026 dari **UTS → Week 08 → Week 09 → ... → UAS**.

## Navigasi

| Periode | Folder | Status |
| --- | --- | --- |
| UTS | [UTS](UTS/) | ✅ Implementasi selesai; hasil pengujian sebelumnya terdokumentasi |
| Week 08 | [Week-08](Week-08/) | ⏳ Menunggu soal |
| Week 09 | [Week-09](Week-09/) | ⏳ Menunggu soal |
| Week 10 | [Week-10](Week-10/) | ⏳ Menunggu soal |
| Week 11 | [Week-11](Week-11/) | ⏳ Menunggu soal |
| Week 12 | [Week-12](Week-12/) | ⏳ Menunggu soal |
| Week 13 | [Week-13](Week-13/) | ⏳ Menunggu soal |
| Week 14 | [Week-14](Week-14/) | ⏳ Menunggu soal |
| UAS | [UAS](UAS/) | ⏳ Menunggu soal |

Setiap periode menyimpan kode, dataset yang diperlukan, dependency, dan petunjuk menjalankannya di folder masing-masing. Folder minggu dan UAS saat ini hanya placeholder.

## Anggota Kelompok

| Nama | Kontribusi UTS |
| --- | --- |
| Ahmad Mathlaul Falah | Dataset dan preprocessing |
| Ah. Dliya'ul Adlha Jamalul Lail | Transformasi dan seleksi fitur |
| Abdullah Azzam | Modeling dan evaluasi |

Nama dan kontribusi merujuk dokumentasi UTS, source code, serta riwayat kontribusi repository. Pembagian tugas mingguan ditentukan saat soal tersedia.

## Menjalankan UTS

Download dataset sesuai [petunjuk dataset](UTS/data/README.md), lalu dari root repository:

```bash
cd UTS
pip install -r requirements.txt
python main.py
```

Dataset mentah harus tersedia di `UTS/data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.
CSV hasil preprocessing yang sudah tersimpan tetap dipertahankan, tetapi tidak menggantikan dataset mentah untuk menjalankan `main.py`.
Lihat [README UTS](UTS/README.md) untuk pipeline dan hasil terdahulu.

## Cara Kerja Kelompok

Alur: **branch anggota → Pull Request → main**.

Sebelum mulai, pastikan perubahan lokal sebelumnya sudah disimpan:

```bash
git checkout main
git pull origin main
git checkout -b falah/week-08-nama-task
```

Kerjakan tugas di folder periodenya. Setelah selesai, periksa perubahan dan pastikan tidak ada kredensial atau file sementara yang ikut:

```bash
git status
git diff
git add .
git commit -m "week 08: deskripsi pekerjaan"
git push -u origin falah/week-08-nama-task
```

Buka Pull Request menuju `main`. Tulis ringkasan perubahan dan cara pengecekannya; minta anggota lain memeriksa sebelum merge. Setelah merge, ambil perubahan terbaru dari `main` sebelum membuat branch berikutnya.

Format branch: `nama/week-08-task`, `nama/week-09-task`, dan seterusnya; untuk UAS: `nama/uas-task`.

Contoh:
- `falah/week-08-preprocessing`
- `adlha/week-08-feature-selection`
- `azzam/week-08-modeling`
- `falah/uas-task`, `adlha/uas-task`, `azzam/uas-task`

Contoh commit:
- `week 08: add preprocessing`
- `week 08: add model training`
- `week 09: update evaluation`
- `uas: finalize model`

## Aturan File

- Lengkapi README periode dengan materi, dataset, pembagian tugas, hasil, dan cara menjalankan setelah informasinya tersedia.
- Simpan dependency yang diperlukan di folder periode terkait.
- Jangan commit virtual environment, cache Python/notebook, `.env`, atau kredensial.
- Output UTS yang dapat dibuat ulang diabaikan oleh Git. CSV preprocessing historis tetap disimpan untuk menjaga hasil UTS.
- Dataset mentah UTS diunduh terpisah sesuai petunjuknya; dataset minggu lain tidak otomatis diabaikan.
- Jangan mengubah hasil UTS untuk mengerjakan tugas minggu berikutnya.

## Branch UTS Lama

Audit sebelum restrukturisasi pada 17 Oktober 2026 terhadap commit `3661384`:
`falah/preprocessing`, `adlha/feature-selection`, dan `azzam/model-evaluation` semuanya sudah masuk `main` (0 commit di depan main).
PR [#1](https://github.com/Fals-Code/uts-prak-ml-c8-kelompok4/pull/1) dan [#2](https://github.com/Fals-Code/uts-prak-ml-c8-kelompok4/pull/2) sudah merged.
Branch tersebut tidak diperlukan untuk pekerjaan baru dan tetap dipertahankan sebagai referensi.
