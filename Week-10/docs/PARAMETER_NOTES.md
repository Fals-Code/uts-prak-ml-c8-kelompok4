# Parameter Notes — Hyperparameter Tuning FCM & DBSCAN

Dokumen ini mencatat rasional empiris dan justifikasi pemilihan parameter clustering untuk Fuzzy C-Means (FCM) dan DBSCAN pada dataset **Mall Customer** dan **Palmer Penguins**.

---

## 1. Prinsip Evaluasi & Pencegahan Data Leakage
1. **Unsupervised Selection Only:**  
   Seluruh seleksi parameter terbaik ($c$, $m$, $\text{eps}$, $\text{min\_samples}$) dilakukan **secara ketat hanya menggunakan metrik internal validasi cluster**:
   - Silhouette Score (semakin tinggi semakin baik)
   - Davies-Bouldin Index (semakin rendah semakin baik)
   - Calinski-Harabasz Index (semakin tinggi semakin baik)
   - Partition Coefficient (PC) dan Partition Entropy (PE) khusus FCM
   - Non-noise coverage dan cluster count $\ge 2$ khusus DBSCAN
2. **Tidak Ada Kebocoran Label (Zero Leakage):**  
   Kolom ground truth `Species` pada Palmer Penguins **tidak pernah disentuh atau dijadikan acuan seleksi model**, melainkan hanya dihitung pada tahap evaluasi post-hoc.

---

## 2. Tuning Hyperparameter Fuzzy C-Means (FCM)

### 2.1 Ruang Pencarian (Grid Search)
- Jumlah cluster $c \in \{2, 3, 4, 5, 6, 7, 8, 9, 10\}$
- Fuzzifier $m \in \{1.25, 1.5, 1.75, 2.0, 2.5\}$
- `random_state = 42`, `tol = 1e-5`, `max_iter = 300`
- Total kombinasi per dataset: 45 kombinasi.

### 2.2 Hasil dan Justifikasi Parameter Terpilih

#### A. Mall Customer
- **Parameter Terbaik:** $c = 6, m = 1.25$
- **Rasional Pemilihan:**
  - Menghasilkan **Silhouette Score tertinggi** sebesar `0.427428` di antara seluruh 45 kombinasi.
  - Davies-Bouldin Index paling optimal (`0.827743`).
  - Calinski-Harabasz Index tertinggi (`135.102217`).
  - Nilai $m=1.25$ memberikan partisi fuzzy yang cukup terdefinisikan secara kompak dengan Partition Coefficient ($PC$) sebesar `0.946564` dan Partition Entropy ($PE$) sebesar `0.103846`.
  - Secara konsisten mengonfirmasi struktur klaster alami data Mall Customer di mana K-Means baseline juga memilih $k=6$.

#### B. Palmer Penguins
- **Parameter Terbaik:** $c = 2, m = 1.25$
- **Rasional Pemilihan:**
  - Menghasilkan **Silhouette Score tertinggi** sebesar `0.531540` dari seluruh kombinasi.
  - Davies-Bouldin Index terendah (`0.714064`).
  - Calinski-Harabasz Index tertinggi (`482.191469`).
  - Partition Coefficient sangat kuat ($PC = 0.981232$) dan Partition Entropy sangat rendah ($PE = 0.034162$).
  - Hasil ini selaras dengan baseline K-Means yang juga memilih $k=2$ berdasarkan geometri densitas fitur numerik terstandardisasi.

---

## 3. Catatan Relevan Terhadap $m$ (Fuzzifier)
- Ketika nilai $m$ dinaikkan ($m \ge 2.0$), tingkat kekaburan (*fuzziness*) antar batas klaster meningkat pesat, menyebabkan Partition Coefficient menurun (misal $PC < 0.5$ pada Mall) dan nilai Silhouette score menurun secara gradual.
- Fuzzifier $m=1.25$ hingga $m=1.50$ memberikan kestabilan separasi geometri yang sangat kokoh untuk kedua dataset tanpa kehilangan sifat representasi keanggotaan derajat probabilitas fuzzy.

---

## 4. Tuning Hyperparameter DBSCAN (`eps` dan `min_samples`)

### 4.1 Metodologi & K-Distance Diagnostics
- Sesuai kaidah pustaka clustering berbasis densitas:
  - Nilai $\text{min\_samples}$ diuji di sekitar $D+1$ dan $2D$, di mana $D=3$ untuk Mall Customer ($\text{min\_samples} \in \{4, 5, 6\}$) dan $D=4$ untuk Palmer Penguins ($\text{min\_samples} \in \{5, 6, 8\}$).
  - Grafik k-nearest neighbor distance ($k=\text{min\_samples}$) menunjukkan titik tekukan (*elbow*) tajam pada rentang jarak $[0.40, 0.65]$ untuk Mall Customer dan $[0.55, 0.75]$ untuk Palmer Penguins.
- **Kriteria Kelayakan:**
  - Jumlah klaster non-noise $\ge 2$.
  - Cakupan observasi valid non-noise ($\text{coverage}$) $\ge 70\%$ (noise tidak melebihi 30%).
  - Silhouette Score non-noise tertinggi sebagai fungsi objektif utama.

### 4.2 Hasil dan Justifikasi Parameter DBSCAN Terpilih

#### A. Mall Customer
- **Parameter Terpilih:** $\text{eps} = 0.45, \text{min\_samples} = 4$
- **Rasional Pemilihan:**
  - Menghasilkan Silhouette Score tertinggi: `0.493716` pada non-noise subset.
  - Davies-Bouldin Index: `0.607155`.
  - Membentuk 7 klaster non-noise dengan coverage $71.5\%$ (57 titik noise terdeteksi sebagai outlier belanja yang tidak lazim).
  - Sebagai pembanding pada klaster yang lebih sedikit ($\ge 5$ klaster), kombinasi $\text{eps} = 0.55, \text{min\_samples} = 6$ juga memberikan Silhouette tinggi `0.483962` dengan coverage $72.5\%$.

#### B. Palmer Penguins
- **Parameter Terpilih:** $\text{eps} = 0.60, \text{min\_samples} = 6$
- **Rasional Pemilihan:**
  - Menghasilkan Silhouette Score tertinggi: `0.561290`.
  - Davies-Bouldin Index: `0.664558` dan Calinski-Harabasz: `519.570108`.
  - Coverage sangat tinggi: $89.18\%$ (hanya 37 titik noise dari 342 observasi).
  - Membentuk 2 klaster densitas utama yang sangat tegas, konsisten dengan pemisahan morfometrik dua kelompok ukuran tubuh penguin di ruang Euclidean.


