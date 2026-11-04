"""Tuning DBSCAN parameters (eps and min_samples) using k-distance diagnostics and grid search.

Selection is strictly based on internal validation metrics, ensuring coverage >= 0.70 and clusters >= 2.
No ground truth labels are accessed during search.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessing import prepare_dataset, ROOT
from src.dbscan_runner import run_dbscan
from src.evaluation import evaluate_clustering


def tune_dbscan_grid(dataset_name: str, eps_values, min_samples_values):
    data = prepare_dataset(dataset_name)
    rows = []

    for min_samples in min_samples_values:
        for eps in eps_values:
            res = run_dbscan(data.X, eps=eps, min_samples=min_samples)
            metrics = evaluate_clustering(
                data.X,
                res.labels,
                external_labels=None,
                noise_label=-1,
            )
            rows.append({
                "dataset": dataset_name,
                "eps": round(float(eps), 4),
                "min_samples": int(min_samples),
                "noise_ratio": round(res.noise_ratio, 4),
                **metrics,
            })

    df = pd.DataFrame(rows)
    return df


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Search grids guided by k-distance curves:
    # Mall: dim=3 -> min_samples in [4, 5, 6], eps in [0.35, 0.75]
    mall_eps = np.arange(0.35, 0.80, 0.05)
    mall_min_samples = [4, 5, 6]

    # Penguins: dim=4 -> min_samples in [5, 6, 8], eps in [0.50, 1.10]
    penguins_eps = np.arange(0.50, 1.15, 0.05)
    penguins_min_samples = [5, 6, 8]

    # 1. Mall DBSCAN Grid
    print("--- Running DBSCAN Grid Search for Mall ---")
    df_mall = tune_dbscan_grid("mall", mall_eps, mall_min_samples)
    df_mall.to_csv(out_dir / "mall_dbscan_search.csv", index=False)
    
    # Filter candidates with >= 2 clusters and coverage >= 0.70
    valid_mall = df_mall[(df_mall["n_clusters"] >= 2) & (df_mall["coverage"] >= 0.70) & (df_mall["silhouette"].notna())]
    if not valid_mall.empty:
        best_mall = valid_mall.loc[valid_mall["silhouette"].idxmax()]
        print(f"Best Mall DBSCAN: eps={best_mall['eps']}, min_samples={best_mall['min_samples']}, clusters={best_mall['n_clusters']}, coverage={best_mall['coverage']:.2f}, silhouette={best_mall['silhouette']:.4f}")
    else:
        print("No valid Mall DBSCAN configuration met criteria")

    # 2. Penguins DBSCAN Grid
    print("\n--- Running DBSCAN Grid Search for Penguins ---")
    df_penguins = tune_dbscan_grid("penguins", penguins_eps, penguins_min_samples)
    df_penguins.to_csv(out_dir / "penguins_dbscan_search.csv", index=False)

    valid_penguins = df_penguins[(df_penguins["n_clusters"] >= 2) & (df_penguins["coverage"] >= 0.70) & (df_penguins["silhouette"].notna())]
    if not valid_penguins.empty:
        best_penguins = valid_penguins.loc[valid_penguins["silhouette"].idxmax()]
        print(f"Best Penguins DBSCAN: eps={best_penguins['eps']}, min_samples={best_penguins['min_samples']}, clusters={best_penguins['n_clusters']}, coverage={best_penguins['coverage']:.2f}, silhouette={best_penguins['silhouette']:.4f}")
    else:
        print("No valid Penguins DBSCAN configuration met criteria")


if __name__ == "__main__":
    main()
