import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

WEEK_DIR = Path(__file__).resolve().parents[1]
if str(WEEK_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK_DIR))

SPLIT_DIR = WEEK_DIR / "outputs" / "split"


def _read(name):
    df = pd.read_csv(SPLIT_DIR / name)
    if df.columns[0] in ["index", "Unnamed: 0"] or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


def _scores(y_true, y_pred):
    return {
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


class JamalSvmDtTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train = _read("X_train.csv")
        cls.X_test = _read("X_test.csv")
        cls.y_train = _read("y_train.csv").iloc[:, 0]
        cls.y_test = _read("y_test.csv").iloc[:, 0]

    def test_split_ukuran_dan_distribusi(self):
        self.assertEqual(self.X_train.shape, (1176, 20))
        self.assertEqual(self.X_test.shape, (294, 20))
        self.assertEqual(self.y_train.value_counts().to_dict(), {0: 986, 1: 190})
        self.assertEqual(self.y_test.value_counts().to_dict(), {0: 247, 1: 47})
        self.assertEqual(list(self.X_train.columns), list(self.X_test.columns))

    def test_svm_hasil_sesuai_metrics(self):
        model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42).fit(
            self.X_train, self.y_train
        )
        s = _scores(self.y_test, model.predict(self.X_test))
        self.assertEqual(s["cm"], [[242, 5], [40, 7]])
        self.assertAlmostEqual(s["accuracy"], 0.8469, delta=1e-4)
        self.assertAlmostEqual(s["precision"], 0.5833, delta=1e-4)
        self.assertAlmostEqual(s["recall"], 0.1489, delta=1e-4)
        self.assertAlmostEqual(s["f1"], 0.2373, delta=1e-4)

    def test_decision_tree_reproduksi_uts(self):
        model = DecisionTreeClassifier(random_state=42).fit(self.X_train, self.y_train)
        s = _scores(self.y_test, model.predict(self.X_test))
        self.assertEqual(s["cm"], [[208, 39], [25, 22]])
        self.assertAlmostEqual(s["accuracy"], 0.7823, delta=1e-4)
        self.assertAlmostEqual(s["precision"], 0.3607, delta=1e-4)
        self.assertAlmostEqual(s["recall"], 0.4681, delta=1e-4)
        self.assertAlmostEqual(s["f1"], 0.4074, delta=1e-4)

    def test_baseline_selalu_no(self):
        self.assertAlmostEqual(
            float((self.y_test == 0).mean()),
            247 / 294,
            delta=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
