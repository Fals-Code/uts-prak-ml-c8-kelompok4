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
