"""Metric definitions are checked against hand-calculated fixture values."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import METRIC_NAMES, evaluate_regression


@pytest.fixture
def metrics():
    return evaluate_regression([100, 200, 300], [110, 180, 330])


def test_mae(metrics):
    assert metrics["MAE"] == pytest.approx(20.0)


def test_mape_percent(metrics):
    assert metrics["MAPE"] == pytest.approx(10.0)


def test_mse(metrics):
    assert metrics["MSE"] == pytest.approx(1400 / 3)


def test_rmse(metrics):
    assert metrics["RMSE"] == pytest.approx(math.sqrt(1400 / 3))


def test_r2(metrics):
    assert metrics["R2"] == pytest.approx(0.93)


def test_metric_schema(metrics):
    assert tuple(metrics) == METRIC_NAMES
    assert all(isinstance(value, float) for value in metrics.values())


@pytest.mark.parametrize("true,pred", [
    ([1, 2], [1]),
    ([], []),
    ([1], [1]),
    ([1, np.nan], [1, 2]),
    ([1, 2], [1, np.inf]),
    ([0, 2], [1, 2]),
    ([2, 2], [1, 2]),
    ([[1], [2]], [1, 2]),
    ([1, 2], [[1], [2]]),
    (["invalid", 2], [1, 2]),
])
def test_invalid_inputs_rejected(true, pred):
    with pytest.raises(ValueError):
        evaluate_regression(true, pred)


def test_perfect_prediction():
    values = evaluate_regression([100, 200], [100, 200])
    assert values == {"MAE": 0.0, "MAPE": 0.0, "MSE": 0.0, "RMSE": 0.0, "R2": 1.0}
