"""Standalone Fuzzy C-Means (FCM) clustering implementation.

Designed for NumPy 2.x and Python 3.14+ compatibility without external fuzzy dependencies.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class FCMResult:
    centers: np.ndarray          # shape (c, D)
    u: np.ndarray                # shape (N, c) - membership matrix
    hard_labels: np.ndarray      # shape (N,) - argmax(u, axis=1)
    n_iter: int                  # number of iterations executed
    fpc: float                   # Partition Coefficient (PC)
    pe: float                    # Partition Entropy (PE)
    m: float                     # fuzzifier
    c: int                       # number of clusters


def compute_partition_coefficient(u: np.ndarray) -> float:
    """Compute Partition Coefficient (PC).

    PC = (1 / N) * sum_{i=1}^N sum_{j=1}^c (u_ij ^ 2)
    Ranges from 1/c (maximal fuzziness) to 1.0 (hard partition).
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[0] == 0:
        raise ValueError("Membership matrix U must be a non-empty 2D array")
    return float(np.sum(u ** 2) / u.shape[0])


def compute_partition_entropy(u: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Partition Entropy (PE).

    PE = - (1 / N) * sum_{i=1}^N sum_{j=1}^c (u_ij * ln(u_ij + eps))
    Ranges from 0.0 (hard partition) to ln(c) (maximal fuzziness).
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[0] == 0:
        raise ValueError("Membership matrix U must be a non-empty 2D array")
    clipped = np.clip(u, eps, 1.0)
    return float(-np.sum(u * np.log(clipped)) / u.shape[0])


def run_fcm(
    X: np.ndarray,
    c: int,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    random_state: int | None = 42,
) -> FCMResult:
    """Fit Fuzzy C-Means clustering to data matrix X.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Input feature matrix.
    c : int
        Number of clusters (2 <= c < N).
    m : float, default=2.0
        Fuzzification exponent m > 1.0.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-5
        Convergence tolerance for the Frobenius norm between consecutive membership matrices: ||U^(t) - U^(t-1)||.
    random_state : int or None, default=42
        Seed for pseudo-random number generator for deterministic initialization.

    Returns
    -------
    FCMResult
        Dataclass containing centers, membership matrix U, hard labels, n_iter, PC, and PE.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_samples, n_features = X.shape

    if not isinstance(c, (int, np.integer)) or c < 2 or c >= n_samples:
        raise ValueError(f"c must be an integer satisfying 2 <= c < n_samples ({n_samples}), got {c}")
    if m <= 1.0:
        raise ValueError(f"Fuzzifier m must be strictly greater than 1.0, got {m}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")

    rng = np.random.RandomState(random_state)

    # Initialize membership matrix U randomly such that sum_j u_ij = 1 for each sample i
    raw_u = rng.uniform(low=1e-3, high=1.0, size=(n_samples, c))
    u = raw_u / np.sum(raw_u, axis=1, keepdims=True)

    power = 2.0 / (m - 1.0)
    eps_dist = 1e-12

    n_iter_done = 0
    for iteration in range(1, max_iter + 1):
        n_iter_done = iteration

        # 1. Update cluster centers v_j: shape (c, D)
        # v_j = sum_i (u_ij^m * x_i) / sum_i (u_ij^m)
        u_m = u ** m  # shape (N, c)
        denom_v = np.sum(u_m, axis=0, keepdims=True).T  # shape (c, 1)
        # Avoid division by zero in degenerate cases
        denom_v = np.maximum(denom_v, eps_dist)
        centers = (u_m.T @ X) / denom_v  # shape (c, D)

        # 2. Compute Euclidean distances d_ij = ||x_i - v_j||: shape (N, c)
        # X: (N, 1, D), centers: (1, c, D) -> diff: (N, c, D)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)  # shape (N, c)

        # Handle exact zero distance to one or more cluster centers
        zero_mask = dist < eps_dist
        has_zeros = np.any(zero_mask, axis=1)

        u_new = np.empty_like(u)

        # For samples with no zero distance: standard update formula
        non_zero_idx = ~has_zeros
        if np.any(non_zero_idx):
            inv_dist = (1.0 / np.maximum(dist[non_zero_idx], eps_dist)) ** power  # shape (n_non_zero, c)
            denom_u = np.sum(inv_dist, axis=1, keepdims=True)
            u_new[non_zero_idx] = inv_dist / denom_u

        # For samples coinciding exactly with cluster centers:
        # Assign 1.0 divided equally among coinciding centers and 0 to others
        if np.any(has_zeros):
            for i in np.where(has_zeros)[0]:
                u_new[i, :] = 0.0
                z_indices = np.where(zero_mask[i])[0]
                u_new[i, z_indices] = 1.0 / len(z_indices)

        # 3. Check convergence using Frobenius norm: ||U^(t) - U^(t-1)||
        diff_u = np.linalg.norm(u_new - u)
        u = u_new
        if diff_u < tol:
            break

    # Hard cluster labels
    hard_labels = np.argmax(u, axis=1)
    fpc = compute_partition_coefficient(u)
    pe = compute_partition_entropy(u)

    return FCMResult(
        centers=centers,
        u=u,
        hard_labels=hard_labels,
        n_iter=n_iter_done,
        fpc=fpc,
        pe=pe,
        m=float(m),
        c=int(c),
    )
