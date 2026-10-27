# Bagian laporan Ahmad Mathlaul Falah — Week 09

> Salinan publik berupa teks metodologi dan hasil agregat. Tidak ada tabel baris
> pelanggan atau gambar data yang disematkan. Rujukan gambar di bawah menunjuk
> hasil lokal yang dapat dibuat ulang dengan dataset yang diperoleh pengguna.
> Dokumen Google Docs asli tidak diubah.

Salinan isi untuk delapan subbagian pada tab Week 09 → Laporan, Kelompok5.
Angka berasal dari preprocessing.py yang dijalankan pada kedua CSV lampiran.
Daftar isi lama Week 08 dan bagian anggota lain menjadi catatan integrasi Azzam.

## 1.1 Latar Belakang

Pelanggan memiliki karakteristik yang beragam, baik dari sisi demografi,
kemampuan ekonomi, perilaku belanja, maupun pilihan layanan. Pengelompokan
pelanggan dapat membantu merangkum keragaman tersebut menjadi pola yang
lebih mudah dipelajari. Pada pembelajaran tanpa pengawasan, pengelompokan
dilakukan berdasarkan kemiripan fitur tanpa menggunakan label sebagai target.

Praktikum Week 09 menggunakan Mall Customers dan Telco Customer Churn untuk
menyiapkan analisis pengelompokan dengan K-Modes. Mall menggambarkan profil
pelanggan pusat perbelanjaan, sedangkan Telco memuat karakteristik pelanggan
telekomunikasi dan langganannya. Kedua dataset dianalisis secara terpisah
karena memiliki atribut dan konteks yang berbeda.

K-Modes bekerja pada atribut kategorikal melalui ketidakcocokan kategori
antarobservasi. Karena kedua dataset juga memiliki atribut numerik, pemilihan
dan pengubahan representasi fitur harus dijelaskan sebelum pemodelan.
Pemeriksaan kualitas data dan EDA diperlukan agar transformasi tidak sekadar
mengikuti tipe penyimpanan kolom. Dasar penggunaan data kategorikal merujuk
pada dokumentasi pustaka kmodes (https://github.com/nicodv/kmodes).

## 1.2 Tujuan

Tahap ini bertujuan memeriksa struktur, kelengkapan, distribusi, serta
konsistensi kedua dataset; memahami karakteristik pelanggan melalui EDA;
dan memilih fitur yang relevan bagi pengelompokan. Tujuan berikutnya adalah
menyiapkan dua matriks fitur kategorikal yang bebas dari identitas dan label
Churn, tanpa kehilangan keterlacakan pelanggan. Hasil disimpan bersama batas
diskretisasi, dokumentasi keputusan, serta bukti validasi agar dapat digunakan
ulang pada tahap K-Modes oleh anggota berikutnya.

## 2.1 Dataset

Mall_Customers.csv diperoleh dari berkas materi praktikum yang disertakan
pengguna. Dataset berisi 200 pelanggan dan lima atribut. Telco-Customer-Churn.csv
juga diperoleh dari lampiran pengguna, berisi 7.043 pelanggan dan 21 atribut.
Kesesuaian versi Telco diperiksa terhadap berkas pada repository resmi IBM,
https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv.
Hash Git blob identik setelah perbedaan CRLF dan LF dinormalisasi hanya di
memori; berkas asli lokal tidak diubah. Asal langsung dan SHA-256 kedua berkas
dicatat pada data/raw/SOURCES.md.

Mall digunakan untuk menyiapkan profil pelanggan berdasarkan demografi,
pendapatan, dan spending score. Telco digunakan untuk menyiapkan profil
berdasarkan rumah tangga, layanan, kontrak, pembayaran, masa berlangganan,
dan biaya bulanan. Churn disimpan sebagai informasi pembanding eksternal,
tanpa menjadi masukan pengelompokan. Pemakaian Telco mengikuti instruksi
terbaru tugas; penyebutan Zoo pada dokumen pembagian lama perlu diselaraskan
pada finalisasi. Nomor penugasan dataset adalah Kelompok 1 sesuai informasi
pengguna, sedangkan identitas repository dan laporan tetap Kelompok 4 kelas C8.

## 2.2 Karakteristik Dataset

Mall terdiri atas CustomerID sebagai identitas, Gender sebagai atribut
kategorikal, serta Age, Annual Income (k$), dan Spending Score (1-100) sebagai
atribut numerik. Usia dinyatakan dalam tahun dan pendapatan tahunan dalam
ribuan dolar sesuai nama kolom. Spending score menunjukkan skor yang
disediakan dataset; skor ini tidak diperlakukan sebagai pengeluaran nominal.

Telco memuat customerID; gender, SeniorCitizen, Partner, dan Dependents sebagai
karakteristik pelanggan; tenure sebagai masa berlangganan; serta atribut
telepon, internet, dan layanan tambahan. Contract, PaperlessBilling, dan
PaymentMethod menjelaskan pola berlangganan dan pembayaran. MonthlyCharges
dan TotalCharges menunjukkan biaya bulanan dan kumulatif. Churn menunjukkan
status keluar pelanggan. SeniorCitizen tersimpan sebagai integer 0/1 tetapi
bermakna kategori biner. TotalCharges justru terbaca sebagai teks karena
memuat spasi pada sejumlah baris; tipe penyimpanan ini memerlukan pemeriksaan
sebelum statistik numerik dihitung.

## 2.3 Alur Pengerjaan

Pengerjaan dimulai dengan menyimpan salinan data asli dan mencatat hash
integritas. Setiap dataset kemudian dibaca menggunakan Pandas untuk
pemeriksaan struktur, nilai kosong, duplikasi, distribusi, dan konsistensi.
EDA dilakukan pada atribut asli agar gambaran data dapat dipahami sebelum
informasi numerik diringkas menjadi kategori.

Setelah EDA, fitur dipilih berdasarkan tujuan segmentasi. Identitas serta
label pembanding dipisahkan, label string dirapikan pada salinan kerja,
dan fitur numerik terpilih didiskretisasi. Hasil akhir diperiksa dari sisi
kelengkapan, tipe kategori, jumlah dan urutan baris, serta kesesuaian setelah
CSV dibaca ulang. Batas kategori dan distribusinya disimpan sebagai metadata.
Pengujian dua eksekusi dilakukan untuk memastikan hasil dapat direproduksi.
Tahap ini berhenti pada penyerahan data; penentuan jumlah cluster dan
pemodelan dilakukan oleh Jamal.

## 2.4 Pemeriksaan Data

Pemeriksaan memakai head(), shape, info(), dan describe(include='all') untuk
meninjau lima observasi pertama, ukuran, tipe data, dan statistik atribut.
Pemeriksaan missing value dilengkapi pencarian string kosong setelah strip,
karena isna() tidak mendeteksi spasi sebagai nilai hilang. Distribusi kategori
dihitung dengan value_counts(), sedangkan nilai numerik diperiksa melalui
rentang, kuartil, histogram, dan boxplot yang relevan.

Duplikasi diperiksa pada seluruh kolom dan setelah identitas dikeluarkan.
Hasil kedua pemeriksaan dibedakan karena pelanggan berbeda dapat memiliki
karakteristik yang sama. Identitas harus unik dan urutannya harus sesuai
dengan sumber. Konsistensi layanan Telco diperiksa antara PhoneService dan
MultipleLines serta antara InternetService dan layanan tambahannya.
Pemeriksaan IQR digunakan sebagai penanda nilai numerik ekstrem, bukan
aturan penghapusan. SeniorCitizen tidak diperiksa dengan IQR karena
merupakan kategori biner, bukan besaran kontinu.

## 2.5 Preprocessing Data

Mall mempertahankan Gender, Age, Annual Income, dan Spending Score untuk
mewakili demografi, karakteristik ekonomi, dan perilaku belanja. CustomerID
dikeluarkan karena hanya mengidentifikasi pelanggan. Gender tetap berupa
label, sedangkan tiga fitur numerik diubah menjadi AgeCategory,
IncomeCategory, dan SpendingCategory. Gender memberi konteks demografis;
keputusan ini tidak mengasumsikan jenis kelamin menentukan perilaku belanja.

Telco mempertahankan SeniorCitizen, Partner, Dependents, MultipleLines,
InternetService, OnlineSecurity, TechSupport, Contract, PaperlessBilling,
PaymentMethod, serta kategori tenure dan MonthlyCharges. Status SeniorCitizen
diubah dari 0/1 menjadi No/Yes. PhoneService dikeluarkan karena informasi
tanpa telepon sudah tercakup dalam MultipleLines. OnlineBackup,
DeviceProtection, StreamingTV, dan StreamingMovies dikeluarkan untuk membatasi
dominasi kelompok fitur tambahan yang saling terkait. gender tidak digunakan
karena fokus demografi dibatasi pada status senior dan rumah tangga.
Pemilihan tersebut bersifat konseptual, bukan klaim bahwa fitur yang
dikeluarkan tidak bermanfaat pada tujuan analisis lain.

TotalCharges dikeluarkan karena mengandung informasi kumulatif yang
berpotensi memberi bobot berulang terhadap lama dan biaya berlangganan.
Sebelas nilai kosong pada atribut ini tetap dicatat dan ditangani pada
pemeriksaan numerik, tetapi tidak diimputasi karena atributnya tidak masuk
matriks fitur. customerID dan Churn disimpan terpisah. Label tanpa layanan
dipertahankan berbeda dari No agar maknanya tidak berubah. Tidak ditemukan
missing value pada fitur akhir sehingga imputasi tidak diperlukan.

Diskretisasi menggunakan tiga kelompok berdasarkan kuantil linear 0, 1/3,
2/3, dan 1 dari masing-masing fitur. Tertil dipilih sebagai ringkasan
deskriptif sederhana yang mempertahankan nilai sama dalam kategori yang
sama. Batas bawah pertama dan seluruh batas kanan termasuk interval.
Jumlah tiga kategori per fitur tidak menentukan jumlah cluster. Batas
berulang dibuang, interval kosong digabung, dan fitur konstan ditangani
dengan kategori tunggal. Implementasi memanfaatkan kuantil Pandas dan cut;
rujukan kuantil dan batas berulang tersedia pada
https://pandas.pydata.org/docs/reference/api/pandas.qcut.html.

Tertil merangkum posisi relatif dalam sampel, bukan ambang usia atau
pendapatan universal. Diskretisasi menghilangkan sebagian informasi
numerik, sedangkan K-Modes memperlakukan kategori rendah, sedang, dan tinggi
secara nominal. Beberapa indikator tanpa internet juga masih berkaitan,
sehingga kontribusi layanan terhadap ketidakcocokan perlu dipertimbangkan
dalam pembahasan model berikutnya. Scaling, one-hot encoding, train-test
split, dan penghapusan outlier tidak digunakan pada tahap ini.

## 3.1 Hasil Pemeriksaan dan Preprocessing Data

Mall memiliki 200 baris dengan 200 CustomerID unik. Tidak ditemukan missing
value, string kosong, duplikasi penuh, maupun duplikasi setelah ID dikeluarkan.
Terdapat 112 Female (56%) dan 88 Male (44%). Usia berkisar 18–70 tahun dengan
rata-rata 38,85 dan median 36. Pendapatan tahunan berkisar 15–137 ribu dolar
dengan rata-rata 60,56 dan median 61,5. Spending score berkisar 1–99 dengan
rata-rata 50,2 dan median 50. Dua observasi pendapatan 137 ditandai IQR,
tetapi dipertahankan karena penanda statistik tidak membuktikan kesalahan.

Gambar 3.1, mall_distributions.png, memperlihatkan variasi profil Mall dan
pendapatan tinggi yang lebih jarang. Gambar 3.2, mall_income_spending.png,
menunjukkan spending score rendah maupun tinggi pada tingkat pendapatan
tinggi. Karena pendapatan tidak sepenuhnya mewakili spending score, keduanya
dipertahankan sebagai dimensi berbeda. Grafik ini belum menunjukkan hasil
cluster dan tidak digunakan untuk menyimpulkan sebab-akibat.

Telco memiliki 7.043 ID unik dan tidak memiliki duplikasi penuh. Sebanyak
22 baris memiliki karakteristik sama setelah customerID dikeluarkan,
dengan Churn masih diperhitungkan dalam pemeriksaan tersebut. Seluruhnya
dipertahankan sebagai pelanggan berbeda. isna() pada data mentah melaporkan
nol, tetapi pemeriksaan spasi menemukan 11 TotalCharges kosong, seluruhnya
pada tenure 0. Tidak ada teks nonkosong lain yang gagal dikonversi ke angka
dan tidak ditemukan konflik kategori layanan yang diperiksa.

Gambar 3.3, telco_characteristics.png, menunjukkan kontrak bulanan sebagai
kategori terbanyak, yaitu 3.875 pelanggan (55,02%), dibandingkan satu tahun
1.473 dan dua tahun 1.695. Fiber optic dipakai 3.096 pelanggan, DSL 2.421,
dan 1.526 tanpa internet. Sebanyak 1.142 pelanggan berstatus senior, 3.402
memiliki partner, dan 2.110 memiliki dependents. Electronic check merupakan
metode pembayaran terbanyak, yaitu 2.365 pelanggan. Keragaman ini menjadi
dasar mempertahankan karakteristik rumah tangga, layanan, dan kontrak.

Gambar 3.4, telco_numeric.png, menunjukkan tenure 0–72 bulan (rata-rata 32,37;
median 29) dan biaya bulanan 18,25–118,75 dolar (rata-rata 64,76; median 70,35).
TotalCharges valid berkisar 18,80–8.684,80 dolar dengan median 1.397,475.
Distribusi biaya kumulatif memiliki ekor kanan yang panjang; atribut ini
tetap dikeluarkan untuk membatasi pengulangan informasi masa dan biaya
langganan. Grafik TotalCharges hanya memakai 7.032 nilai valid, tanpa
menghapus 11 pelanggan dari dataset akhir.

Hasil diskretisasi Mall adalah AgeCategory dengan interval [18; 31],
(31; 45], (45; 70] berjumlah 70, 64, 66; IncomeCategory [15; 48],
(48; 72], (72; 137] berjumlah 70, 64, 66; serta SpendingCategory
[1; 41,333333…], (41,333333…; 59], (59; 99] berjumlah 67, 68, 65.
Untuk spending score bulat, rentangnya ekuivalen 1–41, 42–59, dan 60–99.
TenureCategory Telco memakai [0; 14], (14; 47], (47; 72] bulan dengan
2.371, 2.369, 2.303 pelanggan. MonthlyChargesCategory memakai [18,25; 50,40],
(50,40; 83,90], (83,90; 118,75] dolar dengan 2.351, 2.345, 2.347 pelanggan.
Urutan jumlah pada seluruh interval adalah rendah, sedang, tinggi.
Semua kategori terisi; jumlah tidak harus sama persis karena nilai yang
sama tidak dipecah. Batas presisi penuh disimpan pada metadata JSON.

Sebelum preprocessing, Mall berukuran 200 × 5 dan Telco 7.043 × 21.
Sesudah preprocessing, mall_kmodes.csv berukuran 200 × 4 dan telco_kmodes.csv
7.043 × 12, seluruhnya berisi label kategori tanpa missing value.
Urutan fitur Mall ialah Gender, AgeCategory, IncomeCategory, SpendingCategory.
Urutan fitur Telco ialah SeniorCitizen, Partner, Dependents, MultipleLines,
InternetService, OnlineSecurity, TechSupport, Contract, PaperlessBilling,
PaymentMethod, TenureCategory, MonthlyChargesCategory. Identitas dan Churn
tidak ada pada kedua matriks. File reference menyimpan source_row mulai dari
0 dan ID asli; reference Telco juga menyimpan Churn. Posisi baris final sama
dengan posisi sumber, sehingga hasil model berikutnya dapat ditelusuri.

Transformasi menghasilkan 40 profil kategori unik pada Mall dan 2.936 pada
Telco. Sebanyak 160 dan 4.107 baris berikutnya mengulang profil kategori
yang sudah ada; ini merupakan konsekuensi representasi dan tidak dihapus.
Hasil tersebut menegaskan adanya kehilangan rincian setelah diskretisasi,
bukan penemuan pelanggan duplikat atau cluster tertentu.

Gambar 3.5, code_preprocessing.png, menampilkan cuplikan literal fungsi yang
memisahkan fitur Mall, mengategorikan numerik, dan menyimpan referensi baris.
Gambar 3.6, output_preprocessing.png, menampilkan output eksekusi aktual dan
lima baris CSV yang dibaca ulang. Kedua gambar adalah render berkas/log,
bukan tangkapan antarmuka IDE atau terminal. Bukti lengkap tersedia pada
outputs/evidence/validation.txt dan dua berkas inspection.txt.

Empat tes otomatis lulus, mencakup kontrak output dan reproduksibilitas,
kasus tepi tertil, penolakan input tidak valid, serta pembedaan status biner
dari outlier kontinu. Dua eksekusi menghasilkan CSV, JSON, teks pemeriksaan,
dan empat PNG EDA identik byte demi byte. Pembacaan ulang CSV sama dengan
fitur di memori, jumlah dan urutan pelanggan tetap, serta SHA-256 data asli
tidak berubah. Lingkungan pengujian memakai Python 3.14.6, Pandas 3.0.6,
NumPy 2.5.3, dan Matplotlib 3.11.2. Hasil ini memvalidasi tahap persiapan data;
belum menyatakan kualitas model clustering.

## Berkas gambar untuk integrasi

1. Gambar 3.1: outputs/figures/mall_distributions.png.
2. Gambar 3.2: outputs/figures/mall_income_spending.png.
3. Gambar 3.3: outputs/figures/telco_characteristics.png.
4. Gambar 3.4: outputs/figures/telco_numeric.png.
5. Gambar 3.5: outputs/evidence/code_preprocessing.png.
6. Gambar 3.6: outputs/evidence/output_preprocessing.png.

Penomoran gambar akhir mengikuti integrasi Azzam. Daftar pustaka gabungan
belum diubah; rujukan yang dipakai sudah disertakan langsung pada subbagian
terkait. Tidak ada bagian Jamal/Azzam yang disalin atau ditulis di sini.
