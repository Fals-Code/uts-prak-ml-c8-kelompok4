"""Shared TM11 regression metrics. MAPE is expressed in percent."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

METRIC_NAMES = ("MAE", "MAPE", "MSE", "RMSE", "R2")


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Return five scalar metrics; reject undefined/invalid evaluation inputs.

    MAPE = sklearn MAPE * 100. Zero targets are rejected, not silently clipped.
    R2 requires at least two observations and a nonconstant target here.
    MAE/RMSE are in grams; MSE is in grams squared for the TM11 target.
    """
    try:
        true = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Metric inputs must be numeric") from exc
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional")
    if len(true) != len(pred) or len(true) < 2:
        raise ValueError("Metric inputs require equal lengths and at least two rows")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must be finite with no NaN")
    if np.any(true == 0):
        raise ValueError("Zero targets are not allowed by the MAPE policy")
    if np.all(true == true[0]):
        raise ValueError("R2 is undefined for a constant target")
    mse = float(mean_squared_error(true, pred))
    values = {
        "MAE": float(mean_absolute_error(true, pred)),
        "MAPE": float(mean_absolute_percentage_error(true, pred) * 100.0),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(true, pred, force_finite=False)),
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Metric calculation produced non-finite values")
    return values
