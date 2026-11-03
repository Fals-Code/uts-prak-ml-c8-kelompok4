"""Shared evaluation; None means N/A, never a fabricated score."""
import numpy as np
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score)
from sklearn.metrics.cluster import contingency_matrix

def purity_score(y_true, labels):
    """Sum of largest true-class count in each cluster / number of samples."""
    if len(y_true) != len(labels) or len(labels) == 0:
        raise ValueError("Purity requires nonempty aligned labels")
    table = contingency_matrix(y_true, labels)
    return float(table.max(axis=0).sum() / table.sum())

def evaluate_clustering(X, labels, external_labels=None, noise_label=-1):
    X, labels = np.asarray(X), np.asarray(labels)
    if X.ndim != 2 or labels.ndim != 1 or len(X) != len(labels) or not len(labels):
        raise ValueError("Expected nonempty aligned matrix and labels")
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite")
    keep = labels != noise_label
    n = int(keep.sum())
    clusters = len(np.unique(labels[keep]))
    result = {"n_samples": len(labels), "n_evaluated": n, "n_noise": int((~keep).sum()),
              "coverage": n / len(labels), "n_clusters": clusters,
              "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
              "ari": None, "nmi": None, "purity": None,
              "internal_na_reason": None if 2 <= clusters < n else "requires 2 <= clusters < non-noise samples"}
    if 2 <= clusters < n:
        for key, func in [("silhouette", silhouette_score), ("davies_bouldin", davies_bouldin_score),
                          ("calinski_harabasz", calinski_harabasz_score)]:
            result[key] = float(func(X[keep], labels[keep]))
    if external_labels is not None:
        truth = np.asarray(external_labels)
        if len(truth) != len(labels):
            raise ValueError("External labels are not aligned")
        if n:
            result.update(ari=float(adjusted_rand_score(truth[keep], labels[keep])),
                          nmi=float(normalized_mutual_info_score(truth[keep], labels[keep])),
                          purity=purity_score(truth[keep], labels[keep]))
    return result
