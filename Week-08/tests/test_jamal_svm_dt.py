from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SPLIT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "split"


def _read(name):
    df = pd.read_csv(SPLIT_DIR / name)
    if df.columns[0] in ["index", "Unnamed: 0"] or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


@pytest.fixture(scope="module")
def split():
    return (
        _read("X_train.csv"),
        _read("X_test.csv"),
        _read("y_train.csv").iloc[:, 0],
        _read("y_test.csv").iloc[:, 0],
    )


def _scores(y_true, y_pred):
    return {
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def test_split_ukuran_dan_distribusi(split):
    X_train, X_test, y_train, y_test = split
    assert X_train.shape == (1176, 20)
    assert X_test.shape == (294, 20)
    assert y_train.value_counts().to_dict() == {0: 986, 1: 190}
    assert y_test.value_counts().to_dict() == {0: 247, 1: 47}
    assert list(X_train.columns) == list(X_test.columns)


def test_svm_hasil_sesuai_metrics(split):
    X_train, X_test, y_train, y_test = split
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42).fit(X_train, y_train)
    s = _scores(y_test, model.predict(X_test))
    assert s["cm"] == [[242, 5], [40, 7]]
    assert s["accuracy"] == pytest.approx(0.8469, abs=1e-4)
    assert s["precision"] == pytest.approx(0.5833, abs=1e-4)
    assert s["recall"] == pytest.approx(0.1489, abs=1e-4)
    assert s["f1"] == pytest.approx(0.2373, abs=1e-4)


def test_decision_tree_reproduksi_uts(split):
    X_train, X_test, y_train, y_test = split
    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    s = _scores(y_test, model.predict(X_test))
    assert s["cm"] == [[208, 39], [25, 22]]
    assert s["accuracy"] == pytest.approx(0.7823, abs=1e-4)
    assert s["precision"] == pytest.approx(0.3607, abs=1e-4)
    assert s["recall"] == pytest.approx(0.4681, abs=1e-4)
    assert s["f1"] == pytest.approx(0.4074, abs=1e-4)


def test_baseline_selalu_no(split):
    _, _, _, y_test = split
    assert (y_test == 0).mean() == pytest.approx(247 / 294)
