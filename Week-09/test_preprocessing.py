"""Uji kontrak output dan kasus tepi preprocessing; tidak menjalankan clustering."""

import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

BASE = Path(__file__).resolve().parent
MALL_AVAILABLE = (BASE / "data/raw/Mall_Customers.csv").is_file()


class PreprocessingTests(unittest.TestCase):
    @unittest.skipUnless(MALL_AVAILABLE, "Sediakan Mall_Customers.csv lokal untuk pengujian ini")
    def test_pipeline_preserves_rows_and_repeats_exactly(self):
        with tempfile.TemporaryDirectory(dir=BASE, prefix=".test-", ignore_cleanup_errors=True) as folder:
            destination = Path(folder)
            raw_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (BASE / "data/raw").glob("*.csv")}
            command = [sys.executable, str(BASE / "preprocessing.py"),
                       "--output-dir", str(destination)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            first = {p.relative_to(destination): p.read_bytes()
                     for p in destination.rglob("*") if p.is_file()}
            self.assertTrue(first)
            self.assertEqual(len(list((destination / "figures").glob("*.png"))), 4)
            for name, source, id_column, columns in [
                ("mall", "Mall_Customers.csv", "CustomerID", 4),
                ("telco", "Telco-Customer-Churn.csv", "customerID", 12),
            ]:
                raw = pd.read_csv(BASE / "data/raw" / source)
                features = pd.read_csv(destination / "processed" / f"{name}_kmodes.csv",
                                       dtype="string")
                reference = pd.read_csv(destination / "processed" / f"{name}_reference.csv")
                self.assertEqual(features.shape, (len(raw), columns))
                self.assertFalse(features.isna().any().any())
                self.assertFalse(features.apply(lambda s: s.str.strip().eq("")).any().any())
                self.assertTrue(set(features.columns).isdisjoint(
                    {"CustomerID", "customerID", "Churn", "source_row"}))
                self.assertEqual(reference[id_column].tolist(), raw[id_column].tolist())
                self.assertEqual(reference.source_row.tolist(), list(range(len(raw))))
                if name == "telco":
                    self.assertEqual(reference.Churn.tolist(), raw.Churn.tolist())
            run_again = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run_again.returncode, 0, run_again.stderr)
            second = {p.relative_to(destination): p.read_bytes()
                      for p in destination.rglob("*") if p.is_file()}
            self.assertEqual(first, second, "Semua CSV, JSON, teks, dan PNG harus identik")
            self.assertEqual(raw_hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (BASE / "data/raw").glob("*.csv")})

    def test_binary_status_is_not_a_continuous_outlier(self):
        spec = importlib.util.spec_from_file_location("preprocessing", BASE / "preprocessing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = pd.read_csv(BASE / "data/raw/Telco-Customer-Churn.csv")
        _, summary = module.inspect_data(raw, "customerID")
        self.assertNotIn("SeniorCitizen", summary["iqr_flags_not_deleted"])

    def test_binning_ties_constant_and_missing(self):
        spec = importlib.util.spec_from_file_location("preprocessing", BASE / "preprocessing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for values in [[5, 5, 5], [0, 0, 0, 0, 1, 2], [1, 2, 3, 4, 5, 6]]:
            result, metadata = module.quantile_categories(pd.Series(values))
            self.assertFalse(result.isna().any())
            self.assertEqual(sum(metadata["counts"].values()), len(values))
            self.assertTrue(all(n > 0 for n in metadata["counts"].values()))
            if len(set(values)) > 1:
                rebuilt = pd.cut(pd.Series(values), bins=metadata["edges"],
                                 labels=metadata["labels"], include_lowest=True, right=True)
                self.assertEqual(result.astype(str).tolist(), rebuilt.astype(str).tolist())
        with self.assertRaises(ValueError):
            module.quantile_categories(pd.Series([1, None, 3]))

    @unittest.skipUnless(MALL_AVAILABLE, "Sediakan Mall_Customers.csv lokal untuk pengujian ini")
    def test_rejects_invalid_input_without_silent_imputation(self):
        spec = importlib.util.spec_from_file_location("preprocessing", BASE / "preprocessing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mall = pd.read_csv(BASE / "data/raw/Mall_Customers.csv")
        for column, value in [("Age", -1), ("Gender", "Unknown"), ("CustomerID", None)]:
            damaged = mall.copy()
            damaged.loc[0, column] = value
            with self.assertRaises(ValueError):
                module.prepare_mall(damaged)
        telco = pd.read_csv(BASE / "data/raw/Telco-Customer-Churn.csv")
        damaged = telco.copy()
        damaged.loc[0, "Contract"] = "Unknown"
        with self.assertRaises(ValueError):
            module.prepare_telco(damaged)
        damaged = telco.copy()
        damaged.loc[0, "PhoneService"] = "Yes"
        with self.assertRaises(ValueError):
            module.prepare_telco(damaged)
        trimmed = telco.copy()
        trimmed["Contract"] = " " + trimmed["Contract"] + " "
        features, _, _ = module.prepare_telco(trimmed)
        self.assertFalse(features.Contract.str.startswith(" ").any())


if __name__ == "__main__":
    unittest.main()
