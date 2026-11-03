"""DBSCAN runner and k-distance computation module.

Provides a clean wrapper around sklearn.cluster.DBSCAN with noise statistics,
and nearest-neighbors k-distance computation for elbow/eps diagnostics.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


@dataclass
class DBSCANResult:
    labels: np.ndarray           # cluster labels with noise marked as -1
    n_clusters: int              # number of clusters formed (excluding noise)
    n_noise: int                 # number of noise points
    noise_ratio: float           # proportion of noise points: n_noise / n_samples
    coverage: float              # proportion of non-noise points: n_evaluated / n_samples
    eps: float                   # eps parameter used
    min_samples: int             # min_samples parameter used
    metric: str                  # distance metric used
    model: DBSCAN                # fitted DBSCAN instance


def run_dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    metric: str = "euclidean",
) -> DBSCANResult:
    """Execute DBSCAN clustering and extract noise and coverage statistics.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Input feature matrix.
    eps : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    Returns
    -------
    DBSCANResult
        Dataclass containing labels, cluster count, noise count, noise ratio, coverage, and the fitted model.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("X cannot be empty")
    if eps <= 0:
        raise ValueError(f"eps must be strictly positive, got {eps}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1, got {min_samples}")

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)

    is_noise = labels == -1
    n_noise = int(np.sum(is_noise))
    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)

    noise_ratio = float(n_noise / n_samples)
    coverage = float(1.0 - noise_ratio)

    return DBSCANResult(
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_ratio=noise_ratio,
        coverage=coverage,
        eps=float(eps),
        min_samples=int(min_samples),
        metric=metric,
        model=model,
    )


def compute_k_distance(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute sorted k-nearest neighbor distances for all points in X.

    Used to construct a k-distance graph to help determine candidate eps values
    by looking for the elbow / knee point.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Input feature matrix.
    k : int
        The k-th nearest neighbor (typically k = min_samples or dimension + 1).
    metric : str, default='euclidean'
        Distance metric.

    Returns
    -------
    np.ndarray of shape (N,)
        Sorted distances to the k-th nearest neighbor in ascending order.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_samples = len(X)
    if k < 1 or k >= n_samples:
        raise ValueError(f"k must satisfy 1 <= k < n_samples ({n_samples}), got {k}")

    # n_neighbors = k + 1 because the point itself is included as the 0-th neighbor
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Distances to the k-th neighbor (index k)
    k_distances = distances[:, k]
    # Return sorted distances in ascending order
    return np.sort(k_distances)
