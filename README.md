# Praktikum Machine Learning - C8 Kelompok 4

Repository kerja kelompok Praktikum Machine Learning 2026 dari **UTS  Week 08  Week 09  ...  UAS**.

## Navigasi

| Periode | Folder | Status |
| --- | --- | --- |
| UTS | [UTS](UTS/) | ? Selesai |
| Week 08 | [Week-08](Week-08/) | ?? Berjalan |
| Week 09 | [Week-09](Week-09/) | ? Menunggu soal |
| Week 10 | [Week-10](Week-10/) | ? Menunggu soal |
| Week 11 | [Week-11](Week-11/) | ? Menunggu soal |
| Week 12 | [Week-12](Week-12/) | ? Menunggu soal |
| Week 13 | [Week-13](Week-13/) | ? Menunggu soal |
| UAS | [UAS](UAS/) | ? Menunggu soal |

Implementasi dan evaluasi model Week 08 sudah tersedia. Validasi runtime Streamlit masih dilakukan. Week 09-13 dan UAS tetap menunggu soal; requirement dan pembagian tugasnya belum ditentukan.

## Anggota Kelompok

| Nama | Kontribusi UTS |
| --- | --- |
| Ahmad Mathlaul Falah | Dataset dan preprocessing |
| Ah. Dliya'ul Adlha Jamalul Lail | Transformasi dan seleksi fitur |
| Abdullah Azzam | Modeling dan evaluasi |

Nama dan kontribusi merujuk dokumentasi UTS, source code, serta riwayat kontribusi repository. Pembagian Week 08 tersedia pada [README Week 08](Week-08/README.md).

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

Alur: **Branch anggota → Pull Request → Review → main**.

Sebelum mulai, pastikan perubahan lokal sebelumnya sudah disimpan:

```bash
git checkout main
git pull origin main
git checkout -b falah/week-08-naive-bayes
```

Kerjakan tugas di folder periodenya. Setelah selesai, periksa perubahan dan pastikan tidak ada kredensial atau file sementara yang ikut:

```bash
git status
git diff
git add .
git commit -m "week 08: deskripsi pekerjaan"
git push -u origin falah/week-08-naive-bayes
```

Buka Pull Request menuju `main`. Tulis ringkasan perubahan dan cara pengecekannya; minta anggota lain melakukan review sebelum merge. Setelah merge, ambil perubahan terbaru dari `main` sebelum membuat branch berikutnya.

Branch Week 08 sesuai penanggung jawab:

- Falah: `falah/week-08-naive-bayes`
- Falah: `falah/week-08-ui`
- Jamal: `adlha/week-08-svm`
- Azzam: `azzam/week-08-pipeline`
- Azzam: `azzam/week-08-comparison`

Nama branch untuk periode berikutnya baru ditentukan setelah soalnya tersedia.

Contoh commit:

- `week 08: add naive bayes classifier`
- `week 08: add svm classifier`
- `week 08: add data pipeline`
- `week 08: add model comparison and UI`

## Aturan File

- Lengkapi README periode dengan requirement, pembagian tugas, hasil, dan cara menjalankan setelah informasinya tersedia.
- Simpan dependency yang diperlukan di folder periode terkait.
- Jangan commit virtual environment, cache Python/notebook, `.env`, atau kredensial.
- Output UTS yang dapat dibuat ulang diabaikan oleh Git. CSV preprocessing historis tetap disimpan untuk menjaga hasil UTS.
- Dataset mentah UTS diunduh terpisah sesuai petunjuknya; jangan menggandakan dataset mentah untuk Week 08 tanpa kebutuhan.
- Jangan mengubah hasil UTS untuk mengerjakan tugas minggu berikutnya.

## Branch UTS Lama

Audit sebelum restrukturisasi pada 17 Oktober 2026 terhadap commit `3661384`:
`falah/preprocessing`, `adlha/feature-selection`, dan `azzam/model-evaluation` semuanya sudah masuk `main` (0 commit di depan main).
PR [#1](https://github.com/Fals-Code/uts-prak-ml-c8-kelompok4/pull/1) dan [#2](https://github.com/Fals-Code/uts-prak-ml-c8-kelompok4/pull/2) sudah merged.
Branch tersebut tidak diperlukan untuk pekerjaan baru dan tetap dipertahankan sebagai referensi.


