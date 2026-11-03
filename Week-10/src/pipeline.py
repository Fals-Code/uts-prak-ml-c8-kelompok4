"""Run clustering pipeline from Week-10.

Supports:
- kmeans: baseline K-Means
- fcm: Fuzzy C-Means (standalone NumPy/SciPy implementation with PC and PE)
- dbscan: Density-Based Spatial Clustering of Applications with Noise
"""
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
from .fcm import run_fcm
from .dbscan_runner import run_dbscan


def run_clustering(
    dataset_name,
    method="kmeans",
    k_values=range(2, 11),
    m=2.0,
    eps=0.5,
    min_samples=5,
    random_state=42,
    n_init=10,
    output_dir=None,
):
    """Run clustering on prepared dataset with specified method.

    Parameters
    ----------
    dataset_name : str
        'mall' or 'penguins'.
    method : str
        'kmeans', 'fcm', or 'dbscan'.
    k_values : iterable of int, default=range(2, 11)
        Cluster candidate counts for KMeans and FCM.
    m : float, default=2.0
        Fuzzifier exponent for FCM.
    eps : float, default=0.5
        Epsilon neighborhood distance for DBSCAN.
    min_samples : int, default=5
        Core point minimum sample count for DBSCAN.
    random_state : int, default=42
        Seed for reproducibility.
    n_init : int, default=10
        Initialization iterations for KMeans.
    output_dir : Path or str, optional
        Directory where outputs and artifacts are saved.

    Returns
    -------
    dict
        Execution summary with prepared data, model/result, labels, metrics, and candidates.
    """
    if method not in {"kmeans", "fcm", "dbscan"}:
        raise ValueError(f"Unknown method '{method}'; expected 'kmeans', 'fcm', or 'dbscan'")

    data = prepare_dataset(dataset_name)

    if method == "kmeans":
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
        labels = model.labels_
        model_params = model.get_params()
        result = {"prepared": data, "model": model, "labels": labels, "metrics": best, "candidates": rows}

    elif method == "fcm":
        candidates = sorted(set(k_values))
        if not candidates or any(not isinstance(k, int) or k < 2 or k >= len(data.X) for k in candidates):
            raise ValueError("c must be integer and 2 <= c < sample count")
        rows, models = [], {}
        for c in candidates:
            fcm_res = run_fcm(data.X, c=c, m=m, random_state=random_state)
            metrics = evaluate_clustering(
                data.X,
                fcm_res.hard_labels,
                external_labels=data.external_labels,
                membership_matrix=fcm_res.u,
                partition_coefficient=fcm_res.fpc,
                partition_entropy=fcm_res.pe,
            )
            rows.append({
                "dataset": dataset_name,
                "method": method,
                "c": c,
                "m": float(m),
                "n_iter": fcm_res.n_iter,
                **metrics,
            })
            models[c] = fcm_res

        valid = [r for r in rows if r["silhouette"] is not None and np.isfinite(r["silhouette"])]
        if not valid:
            raise ValueError("No candidate has a valid silhouette score")
        best = max(valid, key=lambda r: (r["silhouette"], -r["c"]))
        model = models[best["c"]]
        labels = model.hard_labels
        model_params = {"c": best["c"], "m": float(m), "random_state": random_state, "n_iter": model.n_iter}
        result = {"prepared": data, "model": model, "labels": labels, "metrics": best, "candidates": rows}

    elif method == "dbscan":
        db_res = run_dbscan(data.X, eps=eps, min_samples=min_samples)
        metrics = evaluate_clustering(
            data.X,
            db_res.labels,
            external_labels=data.external_labels,
            noise_label=-1,
        )
        row = {
            "dataset": dataset_name,
            "method": method,
            "eps": float(eps),
            "min_samples": int(min_samples),
            "noise_ratio": db_res.noise_ratio,
            **metrics,
        }
        rows = [row]
        best = row
        model = db_res
        labels = db_res.labels
        model_params = {"eps": float(eps), "min_samples": int(min_samples), "metric": db_res.metric}
        result = {"prepared": data, "model": model, "labels": labels, "metrics": best, "candidates": rows}

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset_name}_{method}"

        # 1. Search candidates CSV
        search_file_name = f"{prefix}_k_search.csv" if method in {"kmeans", "fcm"} else f"{prefix}_search.csv"
        pd.DataFrame(rows).to_csv(out / search_file_name, index=False)

        # 2. Metrics CSV
        pd.DataFrame([best]).to_csv(out / f"{prefix}_metrics.csv", index=False)

        # 3. Labels CSV
        label_df = {
            "source_row": data.source_rows,
            "cluster": labels,
            **({"species": data.external_labels} if data.external_labels is not None else {}),
        }
        pd.DataFrame(label_df).to_csv(out / f"{prefix}_labels.csv", index=False)

        # 4. FCM Membership Matrix CSV
        if method == "fcm":
            u_cols = [f"cluster_{j}" for j in range(best["c"])]
            u_df = pd.DataFrame(model.u, columns=u_cols)
            u_df.insert(0, "source_row", data.source_rows)
            u_df.to_csv(out / f"{prefix}_membership.csv", index=False)

        # 5. Metadata JSON
        selected_param = (
            {"selected_k": best["k"]} if method == "kmeans" else
            {"selected_c": best["c"], "m": float(m)} if method == "fcm" else
            {"selected_eps": float(eps), "selected_min_samples": int(min_samples)}
        )
        selection_rule = (
            "highest valid silhouette; exact tie chooses smaller k/c" if method in {"kmeans", "fcm"} else
            "single specified eps and min_samples"
        )
        metadata = {
            "preprocessing": data.metadata,
            "model_parameters": model_params,
            **selected_param,
            "selection": selection_rule,
            "versions": {
                "python": platform.python_version(),
                **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]},
            },
        }
        (out / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

        # 6. Curves (for KMeans and FCM)
        if method == "kmeans":
            for metric, suffix, ylabel in [("inertia", "elbow", "Inertia"), ("silhouette", "silhouette", "Silhouette Score")]:
                fig, ax = plt.subplots()
                ax.plot(candidates, [r[metric] for r in rows], marker="o", label=metric)
                ax.axvline(best["k"], linestyle="--", color="gray", label=f"Selected k={best['k']}")
                ax.set(title=f"{dataset_name}: K-Means {suffix}", xlabel="k", ylabel=ylabel, xticks=candidates)
                ax.legend(); fig.tight_layout(); fig.savefig(out / f"{prefix}_{suffix}.png", dpi=150); plt.close(fig)
        elif method == "fcm":
            for metric, suffix, ylabel in [("silhouette", "silhouette", "Silhouette Score"), ("partition_coefficient", "pc", "Partition Coefficient")]:
                fig, ax = plt.subplots()
                ax.plot(candidates, [r[metric] for r in rows], marker="o", label=metric)
                ax.axvline(best["c"], linestyle="--", color="gray", label=f"Selected c={best['c']}")
                ax.set(title=f"{dataset_name}: FCM {suffix}", xlabel="c", ylabel=ylabel, xticks=candidates)
                ax.legend(); fig.tight_layout(); fig.savefig(out / f"{prefix}_{suffix}.png", dpi=150); plt.close(fig)

        # 7. 2D projection scatter plot
        x_col, y_col = data.features.columns[-2:]
        fig, ax = plt.subplots()
        unique_labels = sorted(set(labels))
        for cluster_id in unique_labels:
            keep = labels == cluster_id
            cluster_name = "Noise (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
            ax.scatter(data.features.loc[keep, x_col], data.features.loc[keep, y_col], label=cluster_name)
        param_label = f"k={best['k']}" if method == "kmeans" else f"c={best['c']}" if method == "fcm" else f"eps={eps}, min_pts={min_samples}"
        ax.set(
            title=f"{dataset_name}: {method.upper()} {param_label} (2D projection)",
            xlabel=x_col,
            ylabel=y_col,
        )
        ax.legend(); fig.tight_layout(); fig.savefig(out / f"{prefix}_clusters.png", dpi=150); plt.close(fig)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mall", "penguins", "all"], default="all")
    parser.add_argument("--method", choices=["kmeans", "fcm", "dbscan", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    datasets = ["mall", "penguins"] if args.dataset == "all" else [args.dataset]
    methods = ["kmeans", "fcm", "dbscan"] if args.method == "all" else [args.method]

    for d_name in datasets:
        for m_name in methods:
            result = run_clustering(d_name, method=m_name, output_dir=args.output_dir)
            print(f"[{d_name.upper()} - {m_name.upper()}] Metrics:")
            print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
