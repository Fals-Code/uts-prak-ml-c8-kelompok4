"""Gold-specific LightGBM experiment; same split, CV, metrics, and baseline as XGBoost."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .evaluation import evaluate_regression
from .gold_data_pipeline import (
    MANIFEST_PATH, WEEK_DIR, FEATURES, EXPECTED_SHA256,
    replay_gold_manifest, make_time_series_cv, validate_cv_folds,
)
from .train_xgboost_gold import (          # helper bersama -> perilaku identik dengan XGBoost
    SEED, EXPECTED_SPLIT, PRICE_UNIT, sha256,
    persistence_predict, selected_model_cv, save_model,
)

SEARCH_SPACE = {                            # DIBEKUKAN sebelum run pertama (lihat Step 6)
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__num_leaves": [4, 8, 15],
    "model__max_depth": [-1, 3, 5],
    "model__min_child_samples": [10, 20, 40],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__reg_lambda": [0.0, 1.0, 5.0],
}
ARTIFACTS = (
    "models/gold/lightgbm_pipeline.joblib",
    "results/gold/lightgbm_metrics.json", "results/gold/lightgbm_metrics.csv",
    "results/gold/lightgbm_cv_results.csv", "results/gold/lightgbm_metadata.json",
    "results/gold/lightgbm_test_predictions.csv", "results/gold/lightgbm_execution_log.json",
    "figures/gold/lightgbm_actual_vs_predicted.png",
    "figures/gold/lightgbm_residuals.png", "figures/gold/lightgbm_feature_importance.png",
)


def make_gold_model() -> Pipeline:
    return Pipeline([("model", LGBMRegressor(
        objective="regression", random_state=SEED, n_jobs=1,
        deterministic=True, force_row_wise=True,
        subsample_freq=1,                   # tanpa ini `subsample` tidak berpengaruh di LightGBM
        importance_type="gain", verbose=-1,
    ))])


def make_gold_search() -> RandomizedSearchCV:
    return RandomizedSearchCV(
        make_gold_model(), SEARCH_SPACE, n_iter=12, random_state=SEED,
        scoring="neg_root_mean_squared_error", cv=make_time_series_cv(),
        n_jobs=1, refit=True, error_score="raise", return_train_score=True,
    )


def fit_training_search(X_train: pd.DataFrame, y_train: pd.Series):
    """Training-only API deliberately accepts no test data."""
    if tuple(X_train.columns) != FEATURES or not X_train.index.equals(y_train.index):
        raise ValueError("Training feature schema or target alignment mismatch")
    search = make_gold_search()
    search.fit(X_train, y_train)
    return search


def prediction_frame(split, predicted, baseline) -> pd.DataFrame:
    predicted = np.asarray(predicted, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    actual = split.y_test.to_numpy(dtype=np.float64)
    if predicted.shape != actual.shape or baseline.shape != actual.shape:
        raise ValueError("Prediction shapes do not match test targets")
    if not np.isfinite(predicted).all() or not np.isfinite(baseline).all():
        raise ValueError("Nonfinite predictions")
    timeline = split.samples.timeline.loc[split.test_indices]
    return pd.DataFrame({
        "source_row": split.test_indices,
        "target_row": timeline.target_row.to_numpy(),
        "source_date": timeline.source_date.dt.strftime("%Y-%m-%d").to_numpy(),
        "target_date": timeline.target_date.dt.strftime("%Y-%m-%d").to_numpy(),
        "actual": actual,
        "lightgbm_prediction": predicted,
        "lightgbm_residual": actual - predicted,
        "persistence_prediction": baseline,
        "persistence_residual": actual - baseline,
    })
