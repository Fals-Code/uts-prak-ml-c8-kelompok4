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
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score, silhouette_score

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
    "mall": {
        "k": 3,
        "reason": "Titik tekuk elbow otomatis dengan penurunan cost signifikan 24.31% dan silhouette Hamming 0.3029; menghasilkan 3 segmen yang seimbang dan mudah diinterpretasikan pada 4 fitur kategorikal.",
    },
    "telco": {
        "k": 4,
        "reason": "Titik elbow optimal dengan penurunan cost 10.87% dan silhouette 0.2429; penurunan cost setelah k=4 melandai drastis (<5%) sementara kualitas dan pemisahan cluster tetap stabil.",
    },
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
# STEP 3: SILHOUETTE DENGAN JARAK HAMMING (kategorikal)
# ---------------------------------------------------------------------------
def encode_codes(X: pd.DataFrame) -> np.ndarray:
    """Ubah label kategori menjadi kode integer per kolom (hanya untuk menghitung jarak)."""
    return np.column_stack([pd.factorize(X[c])[0] for c in X.columns])


def sample_indices(n: int, size: int, seed: int = SEED) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def prepare_distance(X: pd.DataFrame):
    """Matriks jarak Hamming (proporsi fitur yang berbeda). Pakai sampel bila data besar."""
    idx = sample_indices(len(X), SIL_SAMPLE_SIZE)
    D = squareform(pdist(encode_codes(X.iloc[idx]), metric="hamming"))
    return idx, D


def suggest_k_elbow(cost_df: pd.DataFrame) -> int:
    """Saran k = titik terjauh dari garis lurus antara titik pertama dan terakhir (kneedle sederhana).
    HANYA saran; keputusan akhir tetap manual."""
    k = cost_df["k"].to_numpy(float)
    c = cost_df["cost"].to_numpy(float)
    kn = (k - k.min()) / (k.max() - k.min())
    cn = (c - c.min()) / (c.max() - c.min())
    x1, y1, x2, y2 = kn[0], cn[0], kn[-1], cn[-1]
    dist = np.abs((x2 - x1) * (cn - y1) - (y2 - y1) * (kn - x1)) / math.hypot(x2 - x1, y2 - y1)
    return int(k[int(np.argmax(dist))])


def add_silhouette(name: str, X: pd.DataFrame, cost_df: pd.DataFrame, labels_by_k: dict):
    idx, D = prepare_distance(X)
    sil = []
    for k in cost_df["k"]:
        s = np.nan
        if k >= 2:
            lab = labels_by_k[int(k)][idx]
            if len(np.unique(lab)) >= 2:
                s = float(silhouette_score(D, lab, metric="precomputed"))
        sil.append(s)
    sel = cost_df.copy()
    sel["silhouette_hamming"] = sil
    sel["silhouette_sample_size"] = len(idx)
    sel["elbow_suggestion"] = sel["k"] == suggest_k_elbow(cost_df)
    sel.to_csv(TABLE_DIR / f"{name}_k_selection.csv", index=False)
    return sel


def plot_k_selection(name: str, sel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(sel["k"], sel["cost"], "o-")
    axes[0].set_title("Cost K-Modes (elbow)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Cost")
    axes[1].plot(sel["k"], sel["silhouette_hamming"], "s-", color="tab:orange")
    axes[1].set_title("Silhouette (jarak Hamming)")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    for ax in axes:
        ax.set_xticks(sel["k"]); ax.grid(alpha=0.3)
    fig.suptitle(f"Pemilihan jumlah cluster - {DATASETS[name]['title']}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_k_selection.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# STEP 4: KEPUTUSAN K FINAL
# ---------------------------------------------------------------------------
def require_k(name: str) -> int:
    k = K_FINAL[name]["k"]
    if k is None:
        raise SystemExit(
            f"K_FINAL['{name}'] belum diisi. Jalankan --stage elbow, pilih k, lalu isi K_FINAL."
        )
    return int(k)


def save_k_decision() -> None:
    out = {n: {"k": K_FINAL[n]["k"], "reason": K_FINAL[n]["reason"]} for n in DATASETS}
    with open(TABLE_DIR / "k_final.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# STEP 5: K-MODES FINAL
# ---------------------------------------------------------------------------
def fit_final(name: str, X: pd.DataFrame, ref: pd.DataFrame, k: int):
    """Fit final. Label diurutkan ulang: cluster 0 = terbesar, dst (agar stabil dibaca)."""
    km, labels = fit_kmodes(X, k, N_INIT_FINAL)
    counts = pd.Series(labels).value_counts()
    order = [c for c, _ in sorted(counts.items(), key=lambda t: (-t[1], t[0]))]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])

    centroids = pd.DataFrame(km.cluster_centroids_[order], columns=X.columns)
    centroids.insert(0, "cluster", range(k))
    centroids.to_csv(TABLE_DIR / f"{name}_centroids_modes.csv", index=False)

    clustered = pd.concat([ref.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    clustered["cluster"] = labels
    clustered.to_csv(TABLE_DIR / f"{name}_clustered.csv", index=False)

    print(f"[{name}] K-Modes final: k={k}, init={INIT}, n_init={N_INIT_FINAL}, "
          f"seed={SEED}, cost={km.cost_:.1f}, n_iter={km.n_iter_}")
    print(clustered["cluster"].value_counts().sort_index().to_string())
    return km, labels, clustered


# ---------------------------------------------------------------------------
# STEP 6: PROFIL CLUSTER
# ---------------------------------------------------------------------------
def order_categories(cats):
    cats = list(cats)
    for order in ORDINAL_ORDERS:
        if set(cats) <= set(order):
            return [c for c in order if c in cats]
    return sorted(cats)


def build_proportions(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Proporsi tiap kategori di tiap cluster, proporsi keseluruhan, dan lift = share / overall."""
    rows = []
    for f in X.columns:
        ct = pd.crosstab(labels, X[f], normalize="index")
        ct = ct[order_categories(ct.columns)]
        overall = X[f].value_counts(normalize=True)
        for c in ct.index:
            for cat in ct.columns:
                share = float(ct.loc[c, cat])
                ov = float(overall[cat])
                rows.append({"cluster": int(c), "feature": f, "category": cat,
                             "share": share, "overall_share": ov, "lift": share / ov})
    return pd.DataFrame(rows)


def cluster_profile(name: str, X: pd.DataFrame, clustered: pd.DataFrame):
    labels = clustered["cluster"]
    rows = []
    for c in sorted(labels.unique()):
        sub = X[labels == c]
        row = {"cluster": int(c), "size": len(sub), "pct": round(len(sub) / len(X) * 100, 2)}
        for f in X.columns:
            vc = sub[f].value_counts(normalize=True)
            row[f] = f"{vc.index[0]} ({vc.iloc[0] * 100:.1f}%)"
        rows.append(row)
    profile = pd.DataFrame(rows)
    profile.to_csv(TABLE_DIR / f"{name}_cluster_profile.csv", index=False)

    prop = build_proportions(X, labels)
    prop.to_csv(TABLE_DIR / f"{name}_cluster_category_proportions.csv", index=False)

    distinctive = (prop[prop["share"] >= 0.20]
                   .sort_values(["cluster", "lift"], ascending=[True, False])
                   .groupby("cluster").head(5))
    distinctive.to_csv(TABLE_DIR / f"{name}_cluster_distinctive.csv", index=False)

    print(f"\n[{name}] Profil cluster (mode per fitur dan persentasenya di dalam cluster):")
    print(profile.to_string(index=False))
    print(f"\n[{name}] Kategori khas (share >= 20%, urut lift):")
    print(distinctive.round(3).to_string(index=False))
    return profile, prop


# ---------------------------------------------------------------------------
# STEP 7: EVALUASI + STABILITAS
# ---------------------------------------------------------------------------
def evaluate_final(name: str, X: pd.DataFrame, km, labels: np.ndarray, k: int) -> dict:
    idx, D = prepare_distance(X)
    lab = labels[idx]
    sil = float(silhouette_score(D, lab, metric="precomputed")) if len(np.unique(lab)) > 1 else float("nan")
    n, m = X.shape
    result = {
        "dataset": name, "k": int(k), "n_rows": int(n), "n_features": int(m),
        "init": INIT, "n_init": N_INIT_FINAL, "seed": SEED,
        "cost": float(km.cost_), "n_iter": int(km.n_iter_),
        "mean_mismatch_per_feature": float(km.cost_) / (n * m),
        "silhouette_hamming": sil,
        "silhouette_sample_size": int(len(idx)),
        "silhouette_on_sample": bool(len(idx) < n),
    }
    with open(TABLE_DIR / f"{name}_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{name}] Evaluasi final:")
    for key, val in result.items():
        print(f"    {key}: {val}")
    return result


def stability_check(name: str, X: pd.DataFrame, k: int, final_labels: np.ndarray) -> pd.DataFrame:
    """Fit ulang dengan seed lain; ARI terhadap hasil final (1.0 = partisi identik)."""
    rows = []
    for s in STABILITY_SEEDS:
        km, lab = fit_kmodes(X, k, N_INIT_STABILITY, seed=s)
        rows.append({"seed": s, "cost": float(km.cost_),
                     "ari_vs_final": float(adjusted_rand_score(final_labels, lab))})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / f"{name}_stability.csv", index=False)
    print(f"\n[{name}] Uji stabilitas antar seed:")
    print(df.round(4).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# STEP 8: VISUALISASI
# ---------------------------------------------------------------------------
def plot_cluster_sizes(name: str, clustered: pd.DataFrame) -> None:
    counts = clustered["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=sns.color_palette("Set2", len(counts)))
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / len(clustered) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Jumlah anggota")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(f"Ukuran cluster - {DATASETS[name]['title']}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_cluster_sizes.png", dpi=150)
    plt.close(fig)


def plot_heatmap(name: str, prop: pd.DataFrame) -> None:
    p = prop.copy()
    p["feature_category"] = p["feature"] + " = " + p["category"]
    pivot = p.pivot_table(index="feature_category", columns="cluster",
                          values="share", sort=False)
    fig, ax = plt.subplots(figsize=(1.6 * pivot.shape[1] + 5, max(4, 0.34 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1,
                cbar_kws={"label": "Proporsi dalam cluster"}, ax=ax)
    ax.set_xlabel("Cluster"); ax.set_ylabel("")
    ax.set_title(f"Proporsi kategori per cluster - {DATASETS[name]['title']}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_heatmap_proportions.png", dpi=150)
    plt.close(fig)


def plot_stacked(name: str, X: pd.DataFrame, clustered: pd.DataFrame) -> None:
    feats = DATASETS[name]["key_features"]
    n = len(feats)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        ct = pd.crosstab(clustered["cluster"], X[f], normalize="index") * 100
        ct = ct[order_categories(ct.columns)]
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", width=0.8)
        ax.set_title(f); ax.set_xlabel("Cluster"); ax.set_ylabel("Persentase (%)")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Komposisi fitur utama per cluster - {DATASETS[name]['title']}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_stacked_features.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# STEP 9: CHURN SEBAGAI PEMBANDING (SETELAH CLUSTERING)
# ---------------------------------------------------------------------------
def churn_crosstab(name: str, clustered: pd.DataFrame) -> None:
    check("Churn" in clustered.columns, "kolom Churn tidak ada")
    counts = pd.crosstab(clustered["cluster"], clustered["Churn"])
    pct = pd.crosstab(clustered["cluster"], clustered["Churn"], normalize="index") * 100
    counts.to_csv(TABLE_DIR / f"{name}_churn_crosstab_counts.csv")
    pct.round(2).to_csv(TABLE_DIR / f"{name}_churn_crosstab_pct.csv")

    chi2, p, dof, _ = chi2_contingency(counts)
    n = counts.to_numpy().sum()
    cramers_v = math.sqrt(chi2 / (n * (min(counts.shape) - 1)))
    overall = clustered["Churn"].eq("Yes").mean() * 100

    print(f"\n[{name}] Crosstab cluster x Churn (jumlah):\n{counts.to_string()}")
    print(f"\n[{name}] Persentase per cluster (%):\n{pct.round(2).to_string()}")
    print(f"[{name}] Churn keseluruhan = {overall:.2f}% | chi2={chi2:.2f}, dof={dof}, "
          f"p={p:.3g}, Cramer's V={cramers_v:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rate = pct["Yes"] if "Yes" in pct.columns else pct.iloc[:, -1]
    bars = ax.bar(rate.index.astype(str), rate.values, color=sns.color_palette("Set2", len(rate)))
    ax.axhline(overall, color="red", linestyle="--", label=f"Rata-rata keseluruhan {overall:.1f}%")
    for b, v in zip(bars, rate.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Churn = Yes (%)")
    ax.set_title("Persentase Churn per cluster (pembanding, bukan fitur)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_churn_by_cluster.png", dpi=150)
    plt.close(fig)
    pd.DataFrame([{"chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v,
                   "overall_churn_pct": overall}]).to_csv(
        TABLE_DIR / f"{name}_churn_association.csv", index=False)


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
            sel = add_silhouette(name, X, cost_df, labels_by_k)
            plot_k_selection(name, sel)
            print(sel.round(4).to_string(index=False))
            print(f"[{name}] saran elbow (otomatis, bukan keputusan): k={suggest_k_elbow(cost_df)}")
            if len(X) > SIL_SAMPLE_SIZE:
                print(f"[{name}] silhouette dihitung pada sampel acak {SIL_SAMPLE_SIZE} baris (seed={SEED})")

    if args.stage in ("final", "all"):
        save_k_decision()
        for name in names:
            X, ref = load_dataset(name)
            k = require_k(name)
            km, labels, clustered = fit_final(name, X, ref, k)
            profile, prop = cluster_profile(name, X, clustered)
            evaluate_final(name, X, km, labels, k)
            stability_check(name, X, k, labels)
            plot_cluster_sizes(name, clustered)
            plot_heatmap(name, prop)
            plot_stacked(name, X, clustered)
            if name == "telco":
                churn_crosstab(name, clustered)


if __name__ == "__main__":
    main()
