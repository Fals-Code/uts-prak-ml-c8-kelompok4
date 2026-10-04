"""Transformasi dan seleksi fitur.

PIC: Ah. Dliya'ul Adlha Jamalul Lail

Ruang lingkup:
- split data 80% training / 20% testing
- Min-Max scaling
- Mutual Information
- L1-based Feature Selection
- daftar fitur terpilih
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PREPROCESSED_PATH = "data/employee_attrition_preprocessed.csv"
TARGET = "Attrition"

SPLIT_OUTPUT_DIR  = Path("data/split")
SCALED_OUTPUT_DIR = Path("data/scaled")
MI_OUTPUT_DIR     = Path("data/mi_selected")

MI_TOP_K = 20  # jumlah fitur terbaik yang dipilih dari Mutual Information

RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ---------------------------------------------------------------------------
# Split data
# ---------------------------------------------------------------------------

def load_preprocessed(path=PREPROCESSED_PATH):
    """Membaca hasil preprocessing menjadi X (fitur) dan y (target)."""
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di {path}")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Membagi dataset menjadi 80% training dan 20% testing.

    Stratifikasi dilakukan agar proporsi kelas target tetap terjaga
    pada kedua subset (distribusi asli: ~84% No dan ~16% Yes).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def save_split(X_train, X_test, y_train, y_test, output_dir=SPLIT_OUTPUT_DIR):
    """Menyimpan hasil split ke file CSV di folder data/split/."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv",  index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(output_dir / "y_test.csv",  index=False, header=True)

    return {
        "X_train": output_dir / "X_train.csv",
        "X_test":  output_dir / "X_test.csv",
        "y_train": output_dir / "y_train.csv",
        "y_test":  output_dir / "y_test.csv",
    }


def print_split_result(X_train, X_test, y_train, y_test):
    """Menampilkan ringkasan hasil split data."""
    total = len(X_train) + len(X_test)
    print("=" * 70)
    print("HASIL SPLIT DATA - Ah. Dliya'ul Adlha Jamalul Lail")
    print("=" * 70)
    print(f"Total data          : {total}")
    print(f"Data training (80%) : {len(X_train)} baris")
    print(f"Data testing  (20%) : {len(X_test)} baris")
    print(f"Jumlah fitur        : {X_train.shape[1]}")
    print(f"Random state        : {RANDOM_STATE}")
    print(f"Stratifikasi        : Ya (berdasarkan kolom '{TARGET}')")

    print("\nDistribusi target — Training (0=No, 1=Yes):")
    print(y_train.value_counts().sort_index().to_string())
    print(f"  Proporsi: {y_train.value_counts(normalize=True).sort_index().mul(100).round(2).to_string()}")

    print("\nDistribusi target — Testing (0=No, 1=Yes):")
    print(y_test.value_counts().sort_index().to_string())
    print(f"  Proporsi: {y_test.value_counts(normalize=True).sort_index().mul(100).round(2).to_string()}")


# ---------------------------------------------------------------------------
# Min-Max scaling
# ---------------------------------------------------------------------------

def scale_data(X_train, X_test):
    """Menerapkan Min-Max scaling pada X_train dan X_test.

    Scaler di-fit hanya pada X_train untuk menghindari data leakage;
    hasilnya kemudian di-transform ke X_test.
    Semua nilai akan berada dalam rentang [0, 1].
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def save_scaled(X_train_scaled, X_test_scaled, output_dir=SCALED_OUTPUT_DIR):
    """Menyimpan hasil scaling ke file CSV di folder data/scaled/."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(output_dir  / "X_test_scaled.csv",  index=False)

    return {
        "X_train_scaled": output_dir / "X_train_scaled.csv",
        "X_test_scaled":  output_dir / "X_test_scaled.csv",
    }


def print_scaling_result(X_train_scaled, X_test_scaled, scaler):
    """Menampilkan ringkasan hasil Min-Max scaling."""
    print("=" * 70)
    print("HASIL MIN-MAX SCALING - Ah. Dliya'ul Adlha Jamalul Lail")
    print("=" * 70)
    print(f"Metode              : MinMaxScaler (sklearn)")
    print(f"Rentang nilai       : [0, 1]")
    print(f"Fit pada            : X_train ({X_train_scaled.shape[0]} baris)")
    print(f"Transform pada      : X_train & X_test ({X_test_scaled.shape[0]} baris)")
    print(f"Jumlah fitur        : {X_train_scaled.shape[1]}")

    summary = pd.DataFrame({
        "min_train":  X_train_scaled.min(),
        "max_train":  X_train_scaled.max(),
        "min_test":   X_test_scaled.min(),
        "max_test":   X_test_scaled.max(),
    })
    print("\nSampel statistik fitur setelah scaling (5 fitur pertama):")
    print(summary.head().to_string())


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def select_features_mi(X_train_scaled, X_test_scaled, y_train,
                       k=MI_TOP_K, random_state=RANDOM_STATE):
    """Memilih k fitur terbaik menggunakan Mutual Information.

    MI dihitung pada X_train_scaled terhadap y_train saja
    (tidak menyentuh X_test) agar tidak terjadi data leakage.
    random_state dipakai agar hasil MI reproducible.
    """
    mi_scores = mutual_info_classif(
        X_train_scaled, y_train,
        discrete_features=False,
        random_state=random_state,
    )

    mi_series = pd.Series(mi_scores, index=X_train_scaled.columns)
    mi_series = mi_series.sort_values(ascending=False)

    top_features = mi_series.head(k).index.tolist()

    X_train_mi = X_train_scaled[top_features]
    X_test_mi  = X_test_scaled[top_features]

    return X_train_mi, X_test_mi, mi_series


def save_mi_selected(X_train_mi, X_test_mi, output_dir=MI_OUTPUT_DIR):
    """Menyimpan dataset hasil seleksi MI ke folder data/mi_selected/."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_mi.to_csv(output_dir / "X_train_mi.csv", index=False)
    X_test_mi.to_csv(output_dir  / "X_test_mi.csv",  index=False)

    return {
        "X_train_mi": output_dir / "X_train_mi.csv",
        "X_test_mi":  output_dir / "X_test_mi.csv",
    }


def print_mi_result(X_train_mi, X_test_mi, mi_series, k=MI_TOP_K):
    """Menampilkan ringkasan hasil seleksi Mutual Information."""
    print("=" * 70)
    print("HASIL MUTUAL INFORMATION - Ah. Dliya'ul Adlha Jamalul Lail")
    print("=" * 70)
    print(f"Total fitur awal    : {len(mi_series)}")
    print(f"Fitur terpilih (k)  : {k}")
    print(f"Ukuran X_train_mi   : {X_train_mi.shape}")
    print(f"Ukuran X_test_mi    : {X_test_mi.shape}")

    print(f"\nRanking skor MI (top {k}):")
    top_df = pd.DataFrame({
        "fitur": mi_series.head(k).index,
        "mi_score": mi_series.head(k).values,
    }).reset_index(drop=True)
    top_df.index += 1
    print(top_df.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = load_preprocessed()
    X_train, X_test, y_train, y_test = split_data(X, y)
    split_paths = save_split(X_train, X_test, y_train, y_test)
    print_split_result(X_train, X_test, y_train, y_test)
    print("\nFile split disimpan di:")
    for k, v in split_paths.items():
        print(f"  {k}: {v}")

    print()
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    scaled_paths = save_scaled(X_train_scaled, X_test_scaled)
    print_scaling_result(X_train_scaled, X_test_scaled, scaler)
    print("\nFile scaled disimpan di:")
    for k, v in scaled_paths.items():
        print(f"  {k}: {v}")

    print()
    X_train_mi, X_test_mi, mi_series = select_features_mi(
        X_train_scaled, X_test_scaled, y_train
    )
    mi_paths = save_mi_selected(X_train_mi, X_test_mi)
    print_mi_result(X_train_mi, X_test_mi, mi_series)
    print("\nFile MI selected disimpan di:")
    for k, v in mi_paths.items():
        print(f"  {k}: {v}")
