# Audit Kontrak Pipeline Falah (Week-10 Baseline)

Dokumen ini mencatat audit arsitektur dan kontrak antarmuka modul pipeline baseline Falah (`Week-10/src/`) untuk memastikan integrasi Fuzzy C-Means (FCM) dan DBSCAN berjalan konsisten tanpa merusak kontrak atau membuat pipeline paralel.

---

## 1. Modul Preprocessing (`Week-10/src/preprocessing.py`)

### 1.1 Struktur Data & Tipe Kembalian
Fungsi utama `prepare_dataset(dataset_name: str) -> PreparedDataset` mengembalikan dataclass `PreparedDataset`:
- `name: str` — nama dataset (`"mall"` atau `"penguins"`).
- `X: np.ndarray` — matriks fitur numerik hasil standardisasi (`StandardScaler`), bentuk $(N, D)$, bertipe floating-point, bebas NaN dan Inf.
- `features: pd.DataFrame` — dataframe fitur numerik sebelum scaling (berisi baris-baris valid).
- `source_rows: np.ndarray` — indeks baris asli dari CSV mentah (0-indexed). Digunakan untuk perataan (*alignment*) dengan metadata atau ground truth.
- `external_labels: np.ndarray | None` — array label eksternal ground truth (kolom `Species` untuk penguins dengan panjang $N$, atau `None` untuk mall).
- `scaler: StandardScaler` — instance scaler yang sudah di-`fit`.
- `metadata: dict` — kamus metadata pembersihan:
  - `dataset`: `"mall"` / `"penguins"`
  - `rows_before`, `rows_after`, `rows_removed`, `removed_source_rows`, `removal_reason`
  - `features`: daftar kolom terpilih
  - `scaling`: `"StandardScaler"`
  - `missing_before`, `missing_policy`: `"reject missing"` (mall) / `"complete-case"` (penguins)
  - `sha256`: hash integritas file CSV mentah
  - `scaler_mean`, `scaler_scale`: parameter hasil fit scaler

### 1.2 Kebijakan Fitur & Data Mentah
- **Mall Customer (`data/raw/Mall_Customers.csv`):**
  - Fitur: `["Age", "Annual Income (k$)", "Spending Score (1-100)"]` ($D=3$, $N=200$).
  - Integritas: Tidak ada missing value yang ditoleransi (`reject missing`). Kolom `CustomerID` dan `Genre`/`Gender` diabaikan secara eksplisit.
- **Palmer Penguins (`data/raw/Penguins.csv`):**
  - Fitur: `["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]` ($D=4$, $N=342$ valid dari 344 baris).
  - Integritas: Menghapus baris yang memiliki missing value pada fitur numerik (`complete-case`), yaitu 2 baris terhapus (indeks 3 dan 339).
  - Ground truth: Kolom `Species` diekstrak terpisah ke `external_labels` dan **tidak pernah disertakan ke dalam fitur matriks `X`**.
- File CSV mentah bersifat **read-only** dan diverifikasi menggunakan hash SHA-256.

---

## 2. Modul Evaluasi (`Week-10/src/evaluation.py`)

### 2.1 Fungsi `purity_score(y_true, labels)`
- Menghitung nilai purity evaluasi eksternal via contingency matrix: $\frac{1}{N} \sum_k \max_j |c_k \cap t_j|$.
- Memvalidasi keselarasan panjang label dan memastikan input tidak kosong.

### 2.2 Fungsi `evaluate_clustering(X, labels, external_labels=None, noise_label=-1)`
- **Penanganan Noise:**
  - Mask non-noise dihitung dengan `keep = labels != noise_label`.
  - Sampel yang dievaluasi untuk metrik internal hanya subset non-noise: `X[keep]` dan `labels[keep]`.
  - Dicatat atribut: `n_samples`, `n_evaluated`, `n_noise`, `coverage = n_evaluated / n_samples`, dan `n_clusters`.
- **Metrik Internal:**
  - Hanya dihitung jika $2 \le \text{clusters} < \text{non-noise samples}$. Jika kondisi tidak terpenuhi, bernilai `None` dengan `internal_na_reason = "requires 2 <= clusters < non-noise samples"`.
  - Metrik: `silhouette` (`silhouette_score`), `davies_bouldin` (`davies_bouldin_score`), `calinski_harabasz` (`calinski_harabasz_score`).
- **Metrik Eksternal (bila `external_labels` disediakan):**
  - Dihitung terhadap subset `keep` jika $n > 0$: `ari` (`adjusted_rand_score`), `nmi` (`normalized_mutual_info_score`), dan `purity` (`purity_score`).
- **Kebutuhan Ekstensi untuk FCM:**
  - FCM memerlukan metrik validasi fuzzy spesifik: **Partition Coefficient (PC)** dan **Partition Entropy (PE)** yang dihitung dari matriks keanggotaan $U$. Modul evaluasi atau fungsi helper FCM perlu menyediakan kalkulasi metrik ini secara presisi.

---

## 3. Modul Runner & Pipeline (`Week-10/src/pipeline.py`)

### 3.1 Fungsi `run_clustering`
- Parameter saat ini: `(dataset_name, method="kmeans", k_values=range(2, 11), random_state=42, n_init=10, output_dir=None)`
- Melakukan iterasi kandidat parameter, menghitung metrik untuk tiap kandidat, memilih kandidat terbaik berdasarkan kriteria:
  - Valid silhouette tertinggi.
  - Jika ada seri (exact tie), memilih $k$ yang lebih kecil.
- Menghasilkan return dictionary:
  - `"prepared"`: Objek `PreparedDataset`.
  - `"model"`: Model terlatih / hasil clustering.
  - `"labels"`: Array label final cluster (atau hard labels).
  - `"metrics"`: Record metrik konfigurasi terbaik.
  - `"candidates"`: List semua record hasil kandidat parameter.

### 3.2 Kontrak File Artefak Output (`Week-10/results/`)
Jika `output_dir` diberikan, pipeline menyimpan file dengan prefix `{dataset_name}_{method}`:
1. `{prefix}_k_search.csv` (atau `{prefix}_search.csv`): Tabel ringkasan semua kandidat parameter beserta metrik evaluasinya.
2. `{prefix}_metrics.csv`: Baris metrik tunggal untuk konfigurasi terbaik terpilih.
3. `{prefix}_labels.csv`: Pemetaan `source_row`, `cluster` (label prediksi), dan `species` (ground truth jika ada).
4. `{prefix}_metadata.json`: Informasi lengkap preprocessing, model parameters, seleksi parameter terbaik, dan versi library (`python`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`).
5. Plot visualisasi:
   - Evaluasi metrik parameter (misal elbow/inertia dan silhouette curve).
   - Proyeksi 2D cluster scatter plot (`{prefix}_clusters.png`) menggunakan 2 kolom numerik terakhir (misal Annual Income vs Spending Score untuk Mall, Flipper Length vs Body Mass untuk Penguins).

---

## 4. Rencana Integrasi FCM dan DBSCAN

1. **Fuzzy C-Means (FCM):**
   - Menambahkan opsi `method="fcm"` ke pipeline.
   - Menggunakan parameter $c$ (jumlah cluster), fuzzifier $m$ (default 2.0), `max_iter`, `tol`, `random_state`.
   - Menghasilkan hard labels `np.argmax(U, axis=1)` dan matriks fuzzy membership $U$.
   - Menyertakan PC dan PE ke dalam pencatatan metrik di `*_search.csv` dan `*_metrics.csv`.
2. **DBSCAN:**
   - Menambahkan opsi `method="dbscan"` ke pipeline.
   - Hyperparameter search pada kombinasi grid `eps` dan `min_samples`.
   - Memanfaatkan k-distance graph (k-NN distance plot) untuk membantu penentuan rentang `eps`.
   - Mengintegrasikan penanganan noise (`label == -1`) secara transparan dengan `evaluate_clustering`.
3. **Kompatibilitas Backward:**
   - Pemanggilan K-Means baseline tetap berfungsi persis seperti semula tanpa mengubah output baseline yang sudah ada.
