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
from sklearn.model_selection import train_test_split

PREPROCESSED_PATH = "data/employee_attrition_preprocessed.csv"
TARGET = "Attrition"

SPLIT_OUTPUT_DIR = Path("data/split")
TRAIN_X_PATH = SPLIT_OUTPUT_DIR / "X_train.csv"
TRAIN_Y_PATH = SPLIT_OUTPUT_DIR / "y_train.csv"
TEST_X_PATH  = SPLIT_OUTPUT_DIR / "X_test.csv"
TEST_Y_PATH  = SPLIT_OUTPUT_DIR / "y_test.csv"

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = load_preprocessed()
    X_train, X_test, y_train, y_test = split_data(X, y)
    paths = save_split(X_train, X_test, y_train, y_test)
    print_split_result(X_train, X_test, y_train, y_test)
    print("\nFile split disimpan di:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
