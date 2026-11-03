"""Hyperparameter tuning for Fuzzy C-Means (FCM).

Sweeps cluster counts c in [2, 10] and fuzzifiers m in [1.25, 1.5, 1.75, 2.0, 2.5]
using internal validation metrics exclusively (Silhouette, Davies-Bouldin, Calinski-Harabasz, PC, PE).
Ground truth labels (Species) are NOT accessed for model selection to avoid data leakage.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src.preprocessing import prepare_dataset, ROOT
from src.fcm import run_fcm
from src.evaluation import evaluate_clustering


def tune_fcm_grid(dataset_name: str, c_values=range(2, 11), m_values=(1.25, 1.5, 1.75, 2.0, 2.5), random_state=42):
    data = prepare_dataset(dataset_name)
    rows = []

    for m in m_values:
        for c in c_values:
            fcm_res = run_fcm(data.X, c=c, m=m, random_state=random_state)
            # Evaluate strictly without external labels during search to prevent leakage
            metrics = evaluate_clustering(
                data.X,
                fcm_res.hard_labels,
                external_labels=None,
                membership_matrix=fcm_res.u,
                partition_coefficient=fcm_res.fpc,
                partition_entropy=fcm_res.pe,
            )
            rows.append({
                "dataset": dataset_name,
                "c": c,
                "m": float(m),
                "n_iter": fcm_res.n_iter,
                **metrics,
            })

    df = pd.DataFrame(rows)
    return df


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    c_candidates = list(range(2, 11))
    m_candidates = [1.25, 1.5, 1.75, 2.0, 2.5]

    for dataset in ["mall", "penguins"]:
        print(f"--- Running FCM Grid Search for: {dataset} ---")
        df_search = tune_fcm_grid(dataset, c_values=c_candidates, m_values=m_candidates, random_state=42)
        out_csv = out_dir / f"{dataset}_fcm_c_search.csv"
        df_search.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        # Find best according to internal silhouette score
        valid = df_search[df_search["silhouette"].notna()]
        best_row = valid.loc[valid["silhouette"].idxmax()]
        print(f"Best configuration for {dataset}:")
        print(f"  c = {int(best_row['c'])}, m = {best_row['m']}")
        print(f"  Silhouette = {best_row['silhouette']:.6f}")
        print(f"  Davies-Bouldin = {best_row['davies_bouldin']:.6f}")
        print(f"  Calinski-Harabasz = {best_row['calinski_harabasz']:.6f}")
        print(f"  Partition Coefficient (PC) = {best_row['partition_coefficient']:.6f}")
        print(f"  Partition Entropy (PE) = {best_row['partition_entropy']:.6f}")
        print()


if __name__ == "__main__":
    main()
