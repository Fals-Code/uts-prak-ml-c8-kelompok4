"""Unit tests for Fuzzy C-Means (FCM) clustering module.

Validates membership matrix constraints, PC/PE metric bounds, reproducibility,
and clustering correctness using Python's standard unittest framework.
"""
import unittest
import numpy as np
from src.fcm import (
    run_fcm,
    compute_partition_coefficient,
    compute_partition_entropy,
)


class FCMTests(unittest.TestCase):
    def setUp(self):
        # Two well-separated 2D synthetic Gaussian clusters
        rng = np.random.RandomState(42)
        cluster1 = rng.normal(loc=[-5.0, -5.0], scale=0.4, size=(25, 2))
        cluster2 = rng.normal(loc=[5.0, 5.0], scale=0.4, size=(25, 2))
        self.X_synthetic = np.vstack([cluster1, cluster2])
        self.y_synthetic = np.array([0] * 25 + [1] * 25)

    def test_membership_matrix_properties(self):
        """Membership u_ij must be in [0, 1] and sum to 1 across clusters for each sample."""
        for c in [2, 3, 5]:
            with self.subTest(c=c):
                res = run_fcm(self.X_synthetic, c=c, m=2.0, random_state=42)
                self.assertEqual(res.u.shape, (len(self.X_synthetic), c))
                # Membership in [0, 1]
                self.assertTrue(np.all(res.u >= 0.0))
                self.assertTrue(np.all(res.u <= 1.0))
                # Sum across clusters == 1.0
                row_sums = np.sum(res.u, axis=1)
                np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_metric_bounds(self):
        """PC must be in [1/c, 1.0] and PE must be >= 0.0 and <= ln(c)."""
        for c in [2, 4]:
            with self.subTest(c=c):
                res = run_fcm(self.X_synthetic, c=c, m=2.0, random_state=42)
                # PC bounds: 1/c <= PC <= 1.0
                self.assertGreaterEqual(res.fpc, (1.0 / c) - 1e-10)
                self.assertLessEqual(res.fpc, 1.0 + 1e-10)

                # PE bounds: 0.0 <= PE <= ln(c)
                self.assertGreaterEqual(res.pe, 0.0 - 1e-10)
                self.assertLessEqual(res.pe, np.log(c) + 1e-10)

    def test_deterministic_reproducibility(self):
        """FCM with the same random_state must yield identical results."""
        run1 = run_fcm(self.X_synthetic, c=2, m=2.0, random_state=42)
        run2 = run_fcm(self.X_synthetic, c=2, m=2.0, random_state=42)

        np.testing.assert_array_equal(run1.hard_labels, run2.hard_labels)
        np.testing.assert_allclose(run1.u, run2.u, atol=1e-12)
        np.testing.assert_allclose(run1.centers, run2.centers, atol=1e-12)
        self.assertEqual(run1.fpc, run2.fpc)
        self.assertEqual(run1.pe, run2.pe)
        self.assertEqual(run1.n_iter, run2.n_iter)

    def test_clustering_accuracy_on_separated_clusters(self):
        """FCM should cleanly separate two clearly distinct synthetic clusters."""
        res = run_fcm(self.X_synthetic, c=2, m=2.0, random_state=42)
        labels = res.hard_labels

        # Since cluster indices may be swapped (0 vs 1), check alignment in either order
        is_exact = np.array_equal(labels, self.y_synthetic)
        is_flipped = np.array_equal(1 - labels, self.y_synthetic)
        self.assertTrue(is_exact or is_flipped, "FCM failed to cleanly separate two distant clusters")

        # In well-separated clusters, PC should be high (close to 1) and PE low (close to 0)
        self.assertGreater(res.fpc, 0.95)
        self.assertLess(res.pe, 0.1)

    def test_invalid_parameters_raise(self):
        """FCM should validate input arguments and raise ValueError accordingly."""
        # 1D array
        with self.assertRaises(ValueError):
            run_fcm(np.array([1.0, 2.0, 3.0]), c=2)
        # c < 2
        with self.assertRaises(ValueError):
            run_fcm(self.X_synthetic, c=1)
        # c >= n_samples
        with self.assertRaises(ValueError):
            run_fcm(self.X_synthetic[:3], c=3)
        # m <= 1.0
        with self.assertRaises(ValueError):
            run_fcm(self.X_synthetic, c=2, m=1.0)
        with self.assertRaises(ValueError):
            run_fcm(self.X_synthetic, c=2, m=0.5)
        # max_iter < 1
        with self.assertRaises(ValueError):
            run_fcm(self.X_synthetic, c=2, max_iter=0)

    def test_standalone_metric_functions(self):
        """Direct computation of PC and PE on edge case membership matrices."""
        # Hard partition: U = [[1, 0], [0, 1]]
        u_hard = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(compute_partition_coefficient(u_hard), 1.0)
        self.assertAlmostEqual(compute_partition_entropy(u_hard), 0.0, places=5)

        # Uniform fuzzy partition: U = [[0.5, 0.5], [0.5, 0.5]] (c=2)
        u_fuzzy = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.assertAlmostEqual(compute_partition_coefficient(u_fuzzy), 0.5)
        self.assertAlmostEqual(compute_partition_entropy(u_fuzzy), np.log(2), places=5)

        with self.assertRaises(ValueError):
            compute_partition_coefficient(np.array([]))
        with self.assertRaises(ValueError):
            compute_partition_entropy(np.array([]))


if __name__ == "__main__":
    unittest.main()
