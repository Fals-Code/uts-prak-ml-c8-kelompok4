import hashlib
import unittest
import numpy as np
import pandas as pd
from src.preprocessing import prepare_dataset, ROOT
from src.pipeline import run_clustering
from src.evaluation import evaluate_clustering, purity_score

class PipelineTests(unittest.TestCase):
    def test_preprocessing_and_alignment(self):
        for name, shape in [("mall", (200, 3)), ("penguins", (342, 4))]:
            with self.subTest(dataset=name):
                data = prepare_dataset(name)
                self.assertEqual(data.X.shape, shape)
                self.assertTrue(np.isfinite(data.X).all())
                self.assertTrue(np.issubdtype(data.X.dtype, np.number))
                self.assertNotIn("Species", data.features.columns)
                self.assertNotIn("CustomerID", data.features.columns)
                np.testing.assert_allclose(data.X.mean(axis=0), 0, atol=1e-12)
                if name == "penguins":
                    raw = pd.read_csv(ROOT / "data/raw/Penguins.csv")
                    np.testing.assert_array_equal(data.external_labels, raw.iloc[data.source_rows]["Species"])
                    self.assertEqual(data.metadata["rows_removed"], 2)

    def test_reproducibility_and_metrics(self):
        for name in ["mall", "penguins"]:
            a = run_clustering(name, k_values=[2, 3])
            b = run_clustering(name, k_values=[2, 3])
            np.testing.assert_array_equal(a["labels"], b["labels"])
            self.assertEqual(len(a["labels"]), len(a["prepared"].X))
            self.assertEqual(a["metrics"], b["metrics"])
            for metric in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
                self.assertTrue(np.isfinite(a["metrics"][metric]))
            self.assertEqual(a["metrics"]["k"], max(a["candidates"], key=lambda r: (r["silhouette"], -r["k"]))["k"])

    def test_purity(self):
        self.assertAlmostEqual(purity_score([0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]), 4/6)
        self.assertEqual(purity_score([0, 0, 1, 1], [1, 1, 0, 0]), 1)
        with self.assertRaises(ValueError): purity_score([], [])

    def test_noise_and_na(self):
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11], [100, 100]])
        result = evaluate_clustering(X, [0, 0, 1, 1, -1], [0, 0, 1, 1, 0])
        self.assertEqual(result["n_noise"], 1)
        self.assertEqual(result["coverage"], .8)
        self.assertTrue(np.isfinite(result["silhouette"]))
        self.assertEqual(result["purity"], 1)
        for labels in [[-1]*5, [0]*5, list(range(5))]:
            self.assertIsNone(evaluate_clustering(X, labels)["silhouette"])

    def test_raw_hashes_after_model(self):
        run_clustering("mall", k_values=[2])
        run_clustering("penguins", k_values=[2])
        for filename, expected in {
            "Mall_Customers.csv": "416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77",
            "Penguins.csv": "5ea908ae6fd5363b6d649f9eecdd63c010caaa389ee9498483e9ec30647dee2d"
        }.items():
            self.assertEqual(hashlib.sha256((ROOT / "data/raw" / filename).read_bytes()).hexdigest(), expected)

if __name__ == "__main__":
    unittest.main()
