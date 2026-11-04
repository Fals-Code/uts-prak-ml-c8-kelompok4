"""Generate high-resolution (>=150 DPI) visualization plots for Week 10.

Plots generated:
1. 2D cluster comparison scatter plots (K-Means vs FCM vs DBSCAN) for Mall Customer.
2. 2D cluster comparison scatter plots (K-Means vs FCM vs DBSCAN) for Palmer Penguins.
3. K-distance diagnostic plots for Mall Customer and Palmer Penguins with chosen eps marked.
4. FCM parameter evaluation curves (Silhouette, PC, PE across c and m).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.preprocessing import prepare_dataset, ROOT
from src.dbscan_runner import compute_k_distance


def plot_2d_comparisons(dataset_name: str, x_col: str, y_col: str, results_dir: Path, figures_dir: Path):
    data = prepare_dataset(dataset_name)
    df_features = data.features

    # Load labels
    km_labels = pd.read_csv(results_dir / f"{dataset_name}_kmeans_labels.csv")["cluster"].to_numpy()
    fcm_labels = pd.read_csv(results_dir / f"{dataset_name}_fcm_labels.csv")["cluster"].to_numpy()
    db_labels = pd.read_csv(results_dir / f"{dataset_name}_dbscan_labels.csv")["cluster"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    configs = [
        ("K-Means (k=6)" if dataset_name == "mall" else "K-Means (k=2)", km_labels, axes[0]),
        ("FCM (c=6, m=1.25)" if dataset_name == "mall" else "FCM (c=2, m=1.25)", fcm_labels, axes[1]),
        ("DBSCAN (eps=0.45, min=4)" if dataset_name == "mall" else "DBSCAN (eps=0.60, min=6)", db_labels, axes[2]),
    ]

    for title, labels, ax in configs:
        unique_labels = sorted(set(labels))
        for cid in unique_labels:
            mask = labels == cid
            if cid == -1:
                ax.scatter(df_features.loc[mask, x_col], df_features.loc[mask, y_col],
                           c="black", marker="x", s=35, label="Noise (-1)", alpha=0.6)
            else:
                ax.scatter(df_features.loc[mask, x_col], df_features.loc[mask, y_col],
                           s=35, label=f"Cluster {cid}", alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")

    dataset_title = "Mall Customer" if dataset_name == "mall" else "Palmer Penguins"
    fig.suptitle(f"{dataset_title} — 2D Projection Comparison (K-Means vs FCM vs DBSCAN)\n(Note: clustering fitted on all scaled dimensions)", fontsize=13, y=1.03)
    fig.tight_layout()

    out_file = figures_dir / f"{dataset_name}_comparison_2d.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_k_distance_diagnostics(figures_dir: Path):
    mall_data = prepare_dataset("mall")
    pen_data = prepare_dataset("penguins")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Mall: k=4 (min_samples=4)
    kd_mall_4 = compute_k_distance(mall_data.X, k=4)
    kd_mall_6 = compute_k_distance(mall_data.X, k=6)
    ax1.plot(kd_mall_4, label="k=4 (min_samples=4)", color="#1f77b4", linewidth=2)
    ax1.plot(kd_mall_6, label="k=6 (2*D)", color="#aec7e8", linestyle="--")
    ax1.axhline(0.45, color="red", linestyle=":", linewidth=2, label="Selected eps = 0.45")
    ax1.set_title("Mall Customer — K-Distance Plot (D=3)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sorted Points (ascending)", fontsize=10)
    ax1.set_ylabel("4th Nearest Neighbor Distance", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Penguins: k=6 (min_samples=6)
    kd_pen_6 = compute_k_distance(pen_data.X, k=6)
    kd_pen_8 = compute_k_distance(pen_data.X, k=8)
    ax2.plot(kd_pen_6, label="k=6 (min_samples=6)", color="#2ca02c", linewidth=2)
    ax2.plot(kd_pen_8, label="k=8 (2*D)", color="#98df8a", linestyle="--")
    ax2.axhline(0.60, color="red", linestyle=":", linewidth=2, label="Selected eps = 0.60")
    ax2.set_title("Palmer Penguins — K-Distance Plot (D=4)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sorted Points (ascending)", fontsize=10)
    ax2.set_ylabel("6th Nearest Neighbor Distance", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle("DBSCAN Diagnostic: K-Nearest Neighbor Distance Curves & Selected Elbow Epsilon", fontsize=13, y=1.02)
    fig.tight_layout()

    out_file = figures_dir / "k_distance_diagnostics.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_fcm_curves(results_dir: Path, figures_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Mall FCM Search
    df_mall = pd.read_csv(results_dir / "mall_fcm_c_search.csv")
    for m in [1.25, 1.5, 2.0]:
        sub = df_mall[df_mall["m"] == m]
        ax1.plot(sub["c"], sub["silhouette"], marker="o", label=f"m={m}")
    ax1.axvline(6, color="gray", linestyle="--", label="Selected c=6 (m=1.25)")
    ax1.set_title("Mall Customer: FCM Silhouette vs c", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (c)", fontsize=10)
    ax1.set_ylabel("Silhouette Score", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Penguins FCM Search
    df_pen = pd.read_csv(results_dir / "penguins_fcm_c_search.csv")
    for m in [1.25, 1.5, 2.0]:
        sub = df_pen[df_pen["m"] == m]
        ax2.plot(sub["c"], sub["silhouette"], marker="o", label=f"m={m}")
    ax2.axvline(2, color="gray", linestyle="--", label="Selected c=2 (m=1.25)")
    ax2.set_title("Palmer Penguins: FCM Silhouette vs c", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (c)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle("FCM Hyperparameter Search: Silhouette Score Across Cluster Counts and Fuzzifiers", fontsize=13, y=1.02)
    fig.tight_layout()

    out_file = figures_dir / "fcm_tuning_curves.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def main():
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_2d_comparisons("mall", "Annual Income (k$)", "Spending Score (1-100)", results_dir, figures_dir)
    plot_2d_comparisons("penguins", "Flipper Length (mm)", "Body Mass (g)", results_dir, figures_dir)
    plot_k_distance_diagnostics(figures_dir)
    plot_fcm_curves(results_dir, figures_dir)
    print("All figures successfully created in Week-10/figures/")


if __name__ == "__main__":
    main()
