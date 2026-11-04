"""Evaluasi eksternal Penguins pada SEMUA titik (noise DBSCAN = kelompok tersendiri).

Dijalankan SETELAH: python -m src.pipeline --dataset all --method all
Species hanya dipakai di sini (pasca-clustering), tidak pernah untuk tuning.
Keluaran:
  results/penguins_external_all_points.csv   (ARI, NMI, Purity: semua titik vs non-noise saja)
  results/penguins_contingency_<method>.csv  (tabel silang species x cluster)
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .evaluation import purity_score
from .preprocessing import ROOT

METHODS = ["kmeans", "fcm", "dbscan"]


def main(results_dir: Path = ROOT / "results"):
    rows = []
    for method in METHODS:
        df = pd.read_csv(results_dir / f"penguins_{method}_labels.csv")
        y, c = df["species"].to_numpy(), df["cluster"].to_numpy()
        keep = c != -1
        # cluster -1 dibiarkan sebagai satu kelompok (label -1) pada evaluasi semua titik
        rows.append({
            "method": method, "n_samples": len(df), "n_noise": int((~keep).sum()),
            "ari_all": adjusted_rand_score(y, c),
            "nmi_all": normalized_mutual_info_score(y, c),
            "purity_all": purity_score(y, c),
            "n_nonnoise": int(keep.sum()),
            "ari_nonnoise": adjusted_rand_score(y[keep], c[keep]),
            "nmi_nonnoise": normalized_mutual_info_score(y[keep], c[keep]),
            "purity_nonnoise": purity_score(y[keep], c[keep]),
        })
        pd.crosstab(df["species"], df["cluster"]).to_csv(
            results_dir / f"penguins_contingency_{method}.csv")
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "penguins_external_all_points.csv", index=False)
    print(out.round(6).to_string(index=False))


if __name__ == "__main__":
    main()