"""Membuat tabel Markdown untuk laporan langsung dari file hasil (tanpa mengetik angka manual).

Dijalankan SETELAH pipeline dan src.external_all_points.
Keluaran: results/report_tables.md
"""
import json
from pathlib import Path
import pandas as pd
from .preprocessing import ROOT, prepare_dataset

R = ROOT / "results"
NAMES = {"kmeans": "K-Means", "fcm": "Fuzzy C-Means", "dbscan": "DBSCAN"}


def f(x, nd=4):
    return "N/A" if pd.isna(x) else f"{x:.{nd}f}"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def param_text(row):
    if row.method == "kmeans":
        return f"k={int(row.k)}, n_init=10, seed=42"
    if row.method == "fcm":
        return f"c={int(row.c)}, m={row.m}, iter={int(row.n_iter)}, seed=42"
    return f"eps={row.eps}, min_samples={int(row.min_samples)}"


def audit_table(ds):
    meta = json.loads((R / f"{ds}_kmeans_metadata.json").read_text())["preprocessing"]
    return pd.DataFrame([{
        "Baris awal": meta["rows_before"], "Baris akhir": meta["rows_after"],
        "Dibuang": meta["rows_removed"], "Indeks dibuang": meta["removed_source_rows"],
        "Fitur": ", ".join(meta["features"]), "Scaler": meta["scaling"],
        "Kebijakan missing": meta["missing_policy"], "SHA-256": meta["sha256"][:12] + "...",
    }])


def main():
    out = ["# Tabel laporan Week 10 (otomatis dari results/)\n"]
    for ds, title in [("mall", "Mall Customer"), ("penguins", "Palmer Penguins")]:
        s = pd.read_csv(R / f"metrics_summary_{ds}.csv")
        out.append(f"## {title}\n\n### Data\n\n" + md(audit_table(ds)) + "\n")
        out.append("### Parameter\n\n" + md(pd.DataFrame({
            "Metode": [NAMES[m] for m in s.method], "Parameter": [param_text(r) for r in s.itertuples()]})) + "\n")
        dist = []
        for m in ["kmeans", "fcm", "dbscan"]:
            lab = pd.read_csv(R / f"{ds}_{m}_labels.csv")["cluster"].value_counts().sort_index()
            dist.append({"Metode": NAMES[m], "Anggota per cluster": "; ".join(
                f"{'noise' if k == -1 else 'C' + str(k)}={v}" for k, v in lab.items())})
        out.append("### Distribusi cluster dan noise\n\n" + md(pd.DataFrame(dist)) + "\n")
        t = pd.DataFrame({
            "Metode": [NAMES[m] for m in s.method], "Cluster": s.n_clusters,
            "Sampel dievaluasi": [f"{int(a)}/{int(b)}" for a, b in zip(s.n_evaluated, s.n_samples)],
            "Noise (%)": [f"{int(n)} ({100 * n / t:.2f}%)" for n, t in zip(s.n_noise, s.n_samples)],
            "Silhouette (naik)": [f(v, 6) for v in s.silhouette],
            "Davies-Bouldin (turun)": [f(v, 6) for v in s.davies_bouldin],
            "Calinski-Harabasz (naik)": [f(v, 4) for v in s.calinski_harabasz]})
        out.append("### Metrik internal\n\n" + md(t) + "\n")
        feats = prepare_dataset(ds).features
        for m in ["kmeans", "fcm", "dbscan"]:
            lab = pd.read_csv(R / f"{ds}_{m}_labels.csv")["cluster"].to_numpy()
            prof = feats.assign(cluster=lab).groupby("cluster").agg(["mean"]).round(2)
            prof.columns = [c[0] for c in prof.columns]
            prof.insert(0, "n", pd.Series(lab).value_counts().sort_index().to_numpy())
            prof = prof.reset_index().rename(columns={"cluster": "cluster (-1 = noise)"})
            out.append(f"### Profil rata-rata fitur asli per cluster: {NAMES[m]}\n\n" + md(prof) + "\n")
        fz = s[s.method == "fcm"]
        out.append("### PC dan PE (FCM)\n\n" + md(pd.DataFrame({
            "PC (naik)": [f(v, 6) for v in fz.partition_coefficient],
            "PE (turun)": [f(v, 6) for v in fz.partition_entropy]})) + "\n")
    e = pd.read_csv(R / "penguins_external_all_points.csv")
    out.append("## Penguins: metrik eksternal terhadap species\n\n" + md(pd.DataFrame({
        "Metode": [NAMES[m] for m in e.method],
        "ARI semua titik": [f(v, 6) for v in e.ari_all], "NMI semua titik": [f(v, 6) for v in e.nmi_all],
        "Purity semua titik": [f(v, 6) for v in e.purity_all],
        "ARI non-noise": [f(v, 6) for v in e.ari_nonnoise], "NMI non-noise": [f(v, 6) for v in e.nmi_nonnoise],
        "Purity non-noise": [f(v, 6) for v in e.purity_nonnoise]})) + "\n")
    (R / "report_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("Tersimpan: results/report_tables.md")


if __name__ == "__main__":
    main()