"""Shared evaluation; None means N/A, never a fabricated score."""
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.cluster import contingency_matrix


def purity_score(y_true, labels):
    """Sum of largest true-class count in each cluster / number of samples."""
    if len(y_true) != len(labels) or len(labels) == 0:
        raise ValueError("Purity requires nonempty aligned labels")
    table = contingency_matrix(y_true, labels)
    return float(table.max(axis=0).sum() / table.sum())


def evaluate_clustering(
    X,
    labels,
    external_labels=None,
    noise_label=-1,
    membership_matrix=None,
    partition_coefficient=None,
    partition_entropy=None,
):
    """Evaluate clustering results with internal, external, and fuzzy validation metrics.

    Parameters
    ----------
    X : array-like of shape (N, D)
        Feature matrix.
    labels : array-like of shape (N,)
        Predicted cluster labels (with noise denoted by noise_label).
    external_labels : array-like of shape (N,), optional
        Ground truth labels for external evaluation.
    noise_label : int or float, default=-1
        Label value that designates noise/outlier points.
    membership_matrix : array-like of shape (N, c), optional
        Fuzzy membership matrix U for computing PC and PE if not explicitly passed.
    partition_coefficient : float, optional
        Precomputed Partition Coefficient (PC) for fuzzy clustering.
    partition_entropy : float, optional
        Precomputed Partition Entropy (PE) for fuzzy clustering.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    if X.ndim != 2 or labels.ndim != 1 or len(X) != len(labels) or not len(labels):
        raise ValueError("Expected nonempty aligned matrix and labels")
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite")

    keep = labels != noise_label
    n = int(keep.sum())
    clusters = len(np.unique(labels[keep]))

    result = {
        "n_samples": len(labels),
        "n_evaluated": n,
        "n_noise": int((~keep).sum()),
        "coverage": n / len(labels),
        "n_clusters": clusters,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "ari": None,
        "nmi": None,
        "purity": None,
        "partition_coefficient": None,
        "partition_entropy": None,
        "internal_na_reason": None if 2 <= clusters < n else "requires 2 <= clusters < non-noise samples",
    }

    # Internal validation metrics (computed strictly on non-noise samples)
    if 2 <= clusters < n:
        for key, func in [
            ("silhouette", silhouette_score),
            ("davies_bouldin", davies_bouldin_score),
            ("calinski_harabasz", calinski_harabasz_score),
        ]:
            result[key] = float(func(X[keep], labels[keep]))

    # External validation metrics
    if external_labels is not None:
        truth = np.asarray(external_labels)
        if len(truth) != len(labels):
            raise ValueError("External labels are not aligned")
        if n:
            result.update(
                ari=float(adjusted_rand_score(truth[keep], labels[keep])),
                nmi=float(normalized_mutual_info_score(truth[keep], labels[keep])),
                purity=purity_score(truth[keep], labels[keep]),
            )

    # Fuzzy validation metrics (PC and PE)
    if partition_coefficient is not None:
        result["partition_coefficient"] = float(partition_coefficient)
    elif membership_matrix is not None:
        u = np.asarray(membership_matrix, dtype=float)
        if u.ndim == 2 and len(u) == len(labels):
            result["partition_coefficient"] = float(np.sum(u ** 2) / len(u))

    if partition_entropy is not None:
        result["partition_entropy"] = float(partition_entropy)
    elif membership_matrix is not None:
        u = np.asarray(membership_matrix, dtype=float)
        if u.ndim == 2 and len(u) == len(labels):
            clipped = np.clip(u, 1e-12, 1.0)
            result["partition_entropy"] = float(-np.sum(u * np.log(clipped)) / len(u))

    return result
