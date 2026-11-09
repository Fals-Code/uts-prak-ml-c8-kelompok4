"""Tiny synthetic fixtures validate software behavior, not research metrics."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import FEATURES
from src.train_xgboost import make_search, make_xgboost_pipeline, save_and_verify_model, sha256


def test_pipeline_contains_unfitted_preprocessing_and_model():
    pipeline = make_xgboost_pipeline()
    assert list(pipeline.named_steps) == ["preprocessing", "model"]
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps["preprocessing"])
    params = pipeline.named_steps["model"].get_params()
    assert params["objective"] == "reg:squarederror"
    assert params["random_state"] == 42 and params["n_jobs"] == 1
    assert params["tree_method"] == "hist"


def test_search_configuration_and_clone_remain_unfitted():
    search = make_search()
    assert search.n_iter == 12 and search.n_jobs == 1 and search.refit is True
    assert search.error_score == "raise"
    assert search.scoring == "neg_root_mean_squared_error"
    assert search.cv.n_splits == 5 and search.cv.shuffle and search.cv.random_state == 42
    assert all(name.startswith("model__") for name in search.param_distributions)
    copied = clone(search.estimator)
    assert not hasattr(copied.named_steps["preprocessing"], "transformers_")
    assert not hasattr(search, "best_estimator_")


def test_model_roundtrip_and_no_overwrite(tmp_path):
    # Only a test fixture; never saved as an experiment result.
    X = pd.DataFrame([
        [35, 16, 180, "A", "Dream", "MALE"],
        [36, 17, 181, "A", "Dream", "FEMALE"],
        [37, 18, 182, "B", "Biscoe", np.nan],
        [38, 19, 183, "B", "Biscoe", "FEMALE"],
    ], columns=list(FEATURES))
    pipeline = make_xgboost_pipeline().set_params(model__n_estimators=3, model__max_depth=2)
    pipeline.fit(X, [3000, 3200, 3500, 3700])
    predictions = pipeline.predict(X)
    path = tmp_path / "fixture_pipeline.joblib"
    digest = save_and_verify_model(pipeline, path, X, predictions)
    assert digest == sha256(path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict(X), predictions)
    assert hasattr(loaded.named_steps["preprocessing"], "transformers_")
    with pytest.raises(FileExistsError):
        save_and_verify_model(pipeline, path, X, predictions)


def test_prediction_csv_preserves_float32_values_and_residuals(tmp_path):
    from src.train_xgboost import make_test_predictions
    actual = pd.Series([3500.0, 4000.0], index=[23, 91])
    predicted = np.asarray([3315.0166015625, 4383.5751953125], dtype=np.float32)
    frame = make_test_predictions(actual, predicted)
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    restored = pd.read_csv(path)
    np.testing.assert_array_equal(restored["predicted_g"].to_numpy(), predicted.astype(float))
    np.testing.assert_array_equal(restored["residual_g"], restored["actual_g"]-restored["predicted_g"])
    assert restored["source_row"].tolist() == [23, 91]
