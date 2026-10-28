"""
Week 09 - Clustering K-Modes
Tahap 2: Ah. Dliya'ul Adlha Jamalul Lail

Input  : mall_kmodes.csv dan telco_kmodes.csv (hasil Falah, tidak diubah)
Metode : K-Modes (kmodes.kmodes.KModes), jarak matching, pusat cluster = mode
Output : outputs/tables, outputs/figures, outputs/evidence
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # simpan gambar tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kmodes.kmodes import KModes
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = Path(os.environ.get("W9_OUT_DIR", BASE_DIR / "outputs"))
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
EVID_DIR = OUT_DIR / "evidence"

SEED = 42
K_RANGE = list(range(1, 11))
INIT = "Huang"            # "Huang" atau "Cao" (Cao deterministik, n_init dipaksa 1)
N_INIT_ELBOW = 5
N_INIT_FINAL = 10
N_INIT_STABILITY = 3
STABILITY_SEEDS = [1, 7, 123, 2024]
SIL_SAMPLE_SIZE = 3000    # sampel acak untuk silhouette bila baris lebih banyak

# Diisi pada Step 4 setelah melihat hasil elbow + silhouette (🛑 keputusan Jamal)
K_FINAL = {
    "mall":  {"k": None, "reason": ""},
    "telco": {"k": None, "reason": ""},
}

# Urutan kategori ordinal untuk grafik (bahasa Indonesia lowercase sesuai preprocessing Falah)
ORDINAL_ORDERS = [
    ["rendah", "sedang", "tinggi"],
    ["Rendah", "Sedang", "Tinggi"],
    ["Low", "Medium", "High"],
    ["low", "medium", "high"],
]

DATASETS = {
    "mall": {
        "title": "Mall Customers",
        "features_csv": DATA_DIR / "mall_kmodes.csv",
        "reference_csv": DATA_DIR / "mall_reference.csv",
        "id_col": "CustomerID",
        "expected_shape": (200, 4),
        "expected_features": ["Gender", "AgeCategory", "IncomeCategory", "SpendingCategory"],
        "key_features": ["Gender", "AgeCategory", "IncomeCategory", "SpendingCategory"],
    },
    "telco": {
        "title": "Telco Customer Churn",
        "features_csv": DATA_DIR / "telco_kmodes.csv",
        "reference_csv": DATA_DIR / "telco_reference.csv",
        "id_col": "customerID",
        "expected_shape": (7043, 12),
        "expected_features": [
            "SeniorCitizen", "Partner", "Dependents", "MultipleLines",
            "InternetService", "OnlineSecurity", "TechSupport", "Contract",
            "PaperlessBilling", "PaymentMethod", "TenureCategory",
            "MonthlyChargesCategory",
        ],
        "key_features": [
            "Contract", "InternetService", "TenureCategory",
            "MonthlyChargesCategory", "PaymentMethod", "SeniorCitizen",
        ],
    },
}


def ensure_dirs() -> None:
    for d in (TABLE_DIR, FIG_DIR, EVID_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: LOADER + VALIDASI
# ---------------------------------------------------------------------------
def check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_dataset(name: str):
    """Baca fitur (kategorikal) dan reference; validasi sebelum dipakai K-Modes."""
    cfg = DATASETS[name]
    if not cfg["features_csv"].exists():
        raise FileNotFoundError(
            f"[{name}] Berkas {cfg['features_csv']} tidak ditemukan. "
            "Pastikan pipeline preprocessing Falah sudah dijalankan: "
            "python Week-09/preprocessing.py"
        )
    if not cfg["reference_csv"].exists():
        raise FileNotFoundError(
            f"[{name}] Berkas {cfg['reference_csv']} tidak ditemukan."
        )

    # keep_default_na=False agar label seperti "No"/"None" tidak dibaca sebagai NaN
    X = pd.read_csv(cfg["features_csv"], dtype=str, keep_default_na=False)
    ref = pd.read_csv(cfg["reference_csv"], dtype=str, keep_default_na=False)

    check(X.shape == cfg["expected_shape"],
          f"[{name}] ukuran {X.shape}, diharapkan {cfg['expected_shape']}")
    check(list(X.columns) == cfg["expected_features"],
          f"[{name}] urutan/nama fitur tidak sesuai: {list(X.columns)}")
    check(not (X.apply(lambda s: s.str.strip() == "")).any().any(),
          f"[{name}] ada string kosong pada fitur")
    check(not X.isna().any().any(), f"[{name}] ada missing value pada fitur")
    check(cfg["id_col"] not in X.columns, f"[{name}] ID ikut menjadi fitur")
    check("Churn" not in X.columns, f"[{name}] Churn ikut menjadi fitur")

    check(len(ref) == len(X), f"[{name}] jumlah baris reference != fitur")
    check(cfg["id_col"] in ref.columns, f"[{name}] kolom {cfg['id_col']} tidak ada di reference")
    if "source_row" in ref.columns:
        check((ref["source_row"].astype(int).to_numpy() == np.arange(len(ref))).all(),
              f"[{name}] source_row tidak berurutan 0..n-1")
    if name == "telco":
        check("Churn" in ref.columns, "[telco] kolom Churn tidak ada di reference")

    print(f"[{name}] OK: {X.shape[0]} baris x {X.shape[1]} fitur, "
          f"profil unik = {X.drop_duplicates().shape[0]}")
    for c in X.columns:
        print(f"    {c}: {sorted(X[c].unique())}")
    return X.reset_index(drop=True), ref.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 2: ELBOW (COST K-MODES)
# ---------------------------------------------------------------------------
def fit_kmodes(X: pd.DataFrame, k: int, n_init: int, init: str = INIT, seed: int = SEED):
    """Fit K-Modes. Cao deterministik, jadi n_init dipaksa 1."""
    if init == "Cao":
        n_init = 1
    km = KModes(n_clusters=k, init=init, n_init=n_init, max_iter=100,
                random_state=seed, verbose=0)
    labels = km.fit_predict(X.values)
    return km, labels


def run_elbow(name: str, X: pd.DataFrame):
    """Hitung cost untuk setiap k. Kembalikan tabel cost dan label tiap k."""
    rows, labels_by_k = [], {}
    for k in K_RANGE:
        km, labels = fit_kmodes(X, k, N_INIT_ELBOW)
        rows.append({"k": k, "cost": float(km.cost_), "n_iter": int(km.n_iter_)})
        labels_by_k[k] = labels
        print(f"[{name}] k={k:2d}  cost={km.cost_:.1f}  n_iter={km.n_iter_}")
    df = pd.DataFrame(rows)
    df["cost_drop"] = -df["cost"].diff()
    df["drop_pct"] = df["cost_drop"] / df["cost"].shift(1) * 100
    df.to_csv(TABLE_DIR / f"{name}_cost_per_k.csv", index=False)
    return df, labels_by_k


def plot_elbow(name: str, cost_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cost_df["k"], cost_df["cost"], "o-", alpha=0.8)
    ax.set_xticks(cost_df["k"])
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Cost K-Modes")
    ax.set_title(f"Metode Elbow K-Modes - {DATASETS[name]['title']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_elbow.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN (diperluas pada step berikutnya)
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["elbow", "final", "all"], default="all")
    parser.add_argument("--dataset", choices=["mall", "telco", "all"], default="all")
    args = parser.parse_args()
    ensure_dirs()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    if args.stage in ("elbow", "all"):
        for name in names:
            X, ref = load_dataset(name)
            cost_df, labels_by_k = run_elbow(name, X)
            plot_elbow(name, cost_df)
            print(cost_df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
