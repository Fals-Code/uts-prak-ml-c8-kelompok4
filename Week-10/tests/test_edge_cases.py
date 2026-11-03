"""Unit tests for edge cases in DBSCAN and the shared evaluation module.

Tests edge scenarios:
1. 100% noise (all points labeled -1) -> n_clusters=0, coverage=0.0, internal metrics=None
2. Single cluster (all points labeled 0) -> n_clusters=1, internal metrics=None
3. Cluster combination with noise -> internal metrics on non-noise, external metrics on all
4. k-distance computation and DBSCAN runner parameter validations
"""
import unittest
import numpy as np
from src.evaluation import evaluate_clustering
from src.dbscan_runner import run_dbscan, compute_k_distance


class EdgeCasesTests(unittest.TestCase):
    def setUp(self):
        # 2D sample data
        self.X = np.array([
            [0.0, 0.0],
            [0.0, 0.2],
            [5.0, 5.0],
            [5.0, 5.2],
            [100.0, 100.0]
        ])

    def test_all_noise_scenario(self):
        """When 100% of samples are noise (-1), internal metrics must be None without throwing errors."""
        labels = np.array([-1, -1, -1, -1, -1])
        res = evaluate_clustering(self.X, labels)

        self.assertEqual(res["n_samples"], 5)
        self.assertEqual(res["n_evaluated"], 0)
        self.assertEqual(res["n_noise"], 5)
        self.assertEqual(res["coverage"], 0.0)
        self.assertEqual(res["n_clusters"], 0)
        self.assertIsNone(res["silhouette"])
        self.assertIsNone(res["davies_bouldin"])
        self.assertIsNone(res["calinski_harabasz"])
        self.assertIsNotNone(res["internal_na_reason"])

    def test_single_cluster_scenario(self):
        """When only 1 cluster is formed, internal metrics must be None (scikit-learn requirement: >= 2 clusters)."""
        labels = np.array([0, 0, 0, 0, 0])
        res = evaluate_clustering(self.X, labels)

        self.assertEqual(res["n_samples"], 5)
        self.assertEqual(res["n_evaluated"], 5)
        self.assertEqual(res["n_noise"], 0)
        self.assertEqual(res["coverage"], 1.0)
        self.assertEqual(res["n_clusters"], 1)
        self.assertIsNone(res["silhouette"])
        self.assertIsNone(res["davies_bouldin"])
        self.assertIsNone(res["calinski_harabasz"])
        self.assertIsNotNone(res["internal_na_reason"])

    def test_noise_plus_clusters_evaluation(self):
        """Internal metrics must be computed strictly on non-noise samples."""
        labels = np.array([0, 0, 1, 1, -1])
        truth = np.array(["A", "A", "B", "B", "C"])
        res = evaluate_clustering(self.X, labels, external_labels=truth)

        self.assertEqual(res["n_noise"], 1)
        self.assertEqual(res["n_evaluated"], 4)
        self.assertEqual(res["coverage"], 0.8)
        self.assertEqual(res["n_clusters"], 2)
        # 2 clusters with 4 evaluated samples allows internal validation
        self.assertIsNotNone(res["silhouette"])
        self.assertTrue(np.isfinite(res["silhouette"]))
        self.assertIsNotNone(res["davies_bouldin"])
        self.assertIsNotNone(res["calinski_harabasz"])
        # External metrics
        self.assertIsNotNone(res["ari"])
        self.assertEqual(res["purity"], 1.0)

    def test_dbscan_runner_execution(self):
        """DBSCAN runner should cleanly identify clusters and noise."""
        # 2 dense pairs plus 1 distant outlier at [100, 100]
        res = run_dbscan(self.X, eps=0.5, min_samples=2)

        self.assertEqual(res.n_clusters, 2)
        self.assertEqual(res.n_noise, 1)
        self.assertAlmostEqual(res.coverage, 0.8)
        self.assertAlmostEqual(res.noise_ratio, 0.2)
        self.assertEqual(res.labels[-1], -1)

    def test_dbscan_runner_invalid_inputs(self):
        """DBSCAN runner should validate parameters."""
        with self.assertRaises(ValueError):
            run_dbscan(np.array([]), eps=0.5, min_samples=2)
        with self.assertRaises(ValueError):
            run_dbscan(self.X, eps=0.0, min_samples=2)
        with self.assertRaises(ValueError):
            run_dbscan(self.X, eps=-1.0, min_samples=2)
        with self.assertRaises(ValueError):
            run_dbscan(self.X, eps=0.5, min_samples=0)

    def test_compute_k_distance(self):
        """compute_k_distance should return sorted non-negative distances."""
        kd = compute_k_distance(self.X, k=2)
        self.assertEqual(len(kd), len(self.X))
        self.assertTrue(np.all(kd >= 0.0))
        # Ensure it is monotonically non-decreasing
        self.assertTrue(np.all(np.diff(kd) >= 0.0))

        with self.assertRaises(ValueError):
            compute_k_distance(self.X, k=0)
        with self.assertRaises(ValueError):
            compute_k_distance(self.X, k=len(self.X))


if __name__ == "__main__":
    unittest.main()
