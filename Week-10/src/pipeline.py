"""Run with python -m src.pipeline --dataset all from Week-10."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .preprocessing import prepare_dataset, ROOT
from .evaluation import evaluate_clustering

def run_clustering(dataset_name, method="kmeans", k_values=range(2, 11), random_state=42, n_init=10, output_dir=None):
    if method != "kmeans":
        raise ValueError("Only kmeans is implemented")
    data = prepare_dataset(dataset_name)
    candidates = sorted(set(k_values))
    if not candidates or any(not isinstance(k, int) or k < 2 or k >= len(data.X) for k in candidates):
        raise ValueError("k must be integer and 2 <= k < sample count")
    rows, models = [], {}
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(data.X)
        metrics = evaluate_clustering(data.X, labels, data.external_labels)
        rows.append({"dataset": dataset_name, "method": method, "k": k,
                     "inertia": float(model.inertia_), **metrics})
        models[k] = model
    valid = [r for r in rows if r["silhouette"] is not None and np.isfinite(r["silhouette"])]
    if not valid:
        raise ValueError("No candidate has a valid silhouette score")
    best = max(valid, key=lambda r: (r["silhouette"], -r["k"]))
    model = models[best["k"]]
    result = {"prepared": data, "model": model, "labels": model.labels_, "metrics": best, "candidates": rows}
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset_name}_{method}"
        pd.DataFrame(rows).to_csv(out / f"{prefix}_k_search.csv", index=False)
        pd.DataFrame([best]).to_csv(out / f"{prefix}_metrics.csv", index=False)
        pd.DataFrame({"source_row": data.source_rows, "cluster": model.labels_,
                      **({"species": data.external_labels} if data.external_labels is not None else {})}).to_csv(out / f"{prefix}_labels.csv", index=False)
        metadata = {"preprocessing": data.metadata, "model_parameters": model.get_params(),
                    "selected_k": best["k"], "selection": "highest valid silhouette; exact tie chooses smaller k",
                    "versions": {"python": platform.python_version(), **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]}}}
        (out / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        for metric, suffix, ylabel in [("inertia", "elbow", "Inertia"), ("silhouette", "silhouette", "Silhouette Score")]:
            fig, ax = plt.subplots()
            ax.plot(candidates, [r[metric] for r in rows], marker="o", label=metric)
            ax.axvline(best["k"], linestyle="--", color="gray", label=f"Selected k={best['k']}")
            ax.set(title=f"{dataset_name}: K-Means {suffix}", xlabel="k", ylabel=ylabel, xticks=candidates)
            ax.legend(); fig.tight_layout(); fig.savefig(out / f"{prefix}_{suffix}.png", dpi=150); plt.close(fig)
        x, y = data.features.columns[-2:]
        fig, ax = plt.subplots()
        for cluster in np.unique(model.labels_):
            keep = model.labels_ == cluster
            ax.scatter(data.features.loc[keep, x], data.features.loc[keep, y], label=f"Cluster {cluster}")
        ax.set(title=f"{dataset_name}: K-Means k={best['k']} (2D projection)", xlabel=x, ylabel=y)
        ax.legend(); fig.tight_layout(); fig.savefig(out / f"{prefix}_clusters.png", dpi=150); plt.close(fig)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mall", "penguins", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    for name in (["mall", "penguins"] if args.dataset == "all" else [args.dataset]):
        result = run_clustering(name, output_dir=args.output_dir)
        print(json.dumps(result["metrics"], indent=2))

if __name__ == "__main__":
    main()
