"""LightGBM trainer contract tests; no randomized tuning is run by unit tests."""
import inspect
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train_lightgbm_gold as trainer
from src.evaluation import evaluate_regression
from src.gold_data_pipeline import FEATURES, MANIFEST_PATH, WEEK_DIR, replay_gold_manifest


def test_bounded_search_and_unfitted_model():
    search = trainer.make_gold_search()
    assert search.n_iter == 12 and search.random_state == 42
    assert search.n_jobs == 1 and search.refit is True and search.error_score == "raise"
    assert search.scoring == "neg_root_mean_squared_error"
    assert search.cv.n_splits == 5 and search.cv.gap == 1
    assert set(search.param_distributions) == set(trainer.SEARCH_SPACE)
    params = search.estimator.named_steps["model"].get_params()
    assert params["n_jobs"] == 1
    assert params["deterministic"] is True
    assert params["subsample_freq"] == 1
    with pytest.raises(NotFittedError):
        check_is_fitted(search.estimator)


def test_fit_training_search_signature():
    assert len(list(inspect.signature(trainer.fit_training_search).parameters)) == 2


def test_search_receives_only_explicit_training_inputs(monkeypatch):
    split = replay_gold_manifest()
    seen = []

    class FakeSearch:
        def fit(self, X, y):
            seen.append((X.index.tolist(), y.index.tolist()))

    fake = FakeSearch()
    monkeypatch.setattr(trainer, "make_gold_search", lambda: fake)
    assert trainer.fit_training_search(split.X_train, split.y_train) is fake
    assert seen == [(split.train_indices, split.train_indices)]
    assert not set(seen[0][0]) & set(split.test_indices)


def test_fixture_pipeline_roundtrip(tmp_path):
    X = pd.DataFrame(np.arange(44, dtype=float).reshape(4, 11) + 10, columns=FEATURES)
    y = np.array([100.0, 101.0, 103.0, 104.0])
    model = trainer.make_gold_model().set_params(
        model__n_estimators=2,
        model__min_child_samples=1,
        model__num_leaves=2,
    )
    model.fit(X, y)  # Small unit-test fixture only, not a reported market experiment.
    predicted = model.predict(X)
    p = tmp_path / "fixture.joblib"
    assert trainer.save_model(model, p, X, predicted) == trainer.sha256(p)
    np.testing.assert_array_equal(joblib.load(p).predict(X), predicted)


def test_prediction_frame_alignment_and_float_precision():
    split = replay_gold_manifest()
    baseline = trainer.persistence_predict(split.X_test)
    predicted = baseline.astype(np.float32)
    frame = trainer.prediction_frame(split, predicted, baseline)
    assert frame.source_row.tolist() == split.test_indices
    assert frame.target_row.tolist() == split.samples.timeline.loc[split.test_indices, "target_row"].tolist()
    assert frame.lightgbm_prediction.dtype == np.float64
    np.testing.assert_array_equal(frame.lightgbm_residual, frame.actual - frame.lightgbm_prediction)
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()


def test_artifact_names_in_gold_namespace():
    assert all("/gold/" in name for name in trainer.ARTIFACTS)
    for name in trainer.ARTIFACTS:
        filename = Path(name).name
        assert filename.startswith("lightgbm_")


def test_default_run_refuses_overwriting_without_training(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Training must not be reached")

    monkeypatch.setattr(trainer, "fit_training_search", fail)

    temp_root = WEEK_DIR / "results" / "_pytest_lgbm"
    dummy_file = temp_root / trainer.ARTIFACTS[0]
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_file.write_text("dummy", encoding="utf-8")
    try:
        with pytest.raises(FileExistsError):
            trainer.run_gold_experiment(temp_root)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


def test_output_namespace_must_remain_inside_week11():
    with pytest.raises(ValueError, match="inside Week-11"):
        trainer.run_gold_experiment(WEEK_DIR.parent)


def test_trainer_does_not_reference_test_data_in_search():
    source = inspect.getsource(trainer.fit_training_search)
    assert "X_test" not in source
    assert "y_test" not in source


def test_importance_normalization():
    raw = np.array([10.0, 30.0, 60.0])
    normalized = trainer.normalize_gain(raw)
    assert normalized.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(normalized, [0.1, 0.3, 0.6])

    with pytest.raises(ValueError, match="positive and finite"):
        trainer.normalize_gain(np.zeros(5))

    with pytest.raises(ValueError, match="positive and finite"):
        trainer.normalize_gain(np.array([-1.0, -2.0]))


def test_saved_gold_artifacts_readonly():
    metadata_path = WEEK_DIR / "results/gold/lightgbm_metadata.json"
    if not metadata_path.exists():
        pytest.skip("Run the Gold LightGBM experiment before artifact integration audit")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads((WEEK_DIR / "results/gold/lightgbm_metrics.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(WEEK_DIR / "results/gold/lightgbm_test_predictions.csv")
    modelpath = WEEK_DIR / "models/gold/lightgbm_pipeline.joblib"

    # 1. sha256(models/gold/lightgbm_pipeline.joblib) == metadata["model_sha256"]
    assert trainer.sha256(modelpath) == metadata["model_sha256"]

    # 2. metadata["split_fingerprint"] == EXPECTED_SPLIT; manifest_file_sha256 == sha256(MANIFEST_PATH); feature_schema == list(FEATURES)
    assert metadata["split_fingerprint"] == trainer.EXPECTED_SPLIT
    assert metadata["manifest_file_sha256"] == trainer.sha256(MANIFEST_PATH)
    assert metadata["feature_schema"] == list(FEATURES)

    # 3. sha256(results/gold/lightgbm_training_source.py.txt) == metadata["trainer_source_sha256"] dan byte sama dengan src/train_lightgbm_gold.py
    source_archive = WEEK_DIR / "results/gold/lightgbm_training_source.py.txt"
    trainer_file = WEEK_DIR / "src/train_lightgbm_gold.py"
    assert trainer.sha256(source_archive) == metadata["trainer_source_sha256"]
    assert source_archive.read_bytes() == trainer_file.read_bytes()

    # 4. Prediksi CSV: 342 baris; source_row == split.test_indices; target_row cocok dengan timeline; semua numerik finite
    split = replay_gold_manifest()
    assert len(frame) == 342 and frame.source_row.tolist() == split.test_indices
    assert frame.target_row.tolist() == split.samples.timeline.loc[split.test_indices, "target_row"].tolist()
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()

    # 5. joblib.load(model).predict(split.X_test) ≈ lightgbm_prediction (atol=1e-10); residual = actual − prediksi; persistence_prediction == split.X_test.Close
    loaded = joblib.load(modelpath)
    np.testing.assert_allclose(loaded.predict(split.X_test), frame.lightgbm_prediction, rtol=0, atol=1e-10)
    np.testing.assert_allclose(frame.lightgbm_residual, frame.actual - frame.lightgbm_prediction, rtol=0, atol=1e-10)
    np.testing.assert_allclose(frame.persistence_prediction, split.X_test.Close, rtol=0, atol=1e-10)

    # 6. loaded.get_params() cocok dengan metadata["best_parameters"]
    for key, value in metadata["best_parameters"].items():
        assert loaded.get_params()[key] == value

    # 7. metrics.json == metadata["metrics"]; metrik yang dihitung ulang dari CSV prediksi dengan evaluate_regression = metrics["lightgbm_testing"] dan ["persistence_testing"]
    assert metrics == metadata["metrics"]
    recomputed_lgbm = evaluate_regression(frame.actual, frame.lightgbm_prediction)
    recomputed_pers = evaluate_regression(frame.actual, frame.persistence_prediction)
    assert recomputed_lgbm == pytest.approx(metrics["lightgbm_testing"])
    assert recomputed_pers == pytest.approx(metrics["persistence_testing"])

    # 8. cv_results.csv 12 baris, skor fold finite, -mean_test_score.max() == metadata["best_mean_cv_rmse"]
    cv = pd.read_csv(WEEK_DIR / "results/gold/lightgbm_cv_results.csv")
    assert len(cv) == 12
    assert np.isfinite(cv[[f"split{i}_test_score" for i in range(5)]].to_numpy()).all()
    assert float(-cv["mean_test_score"].max()) == pytest.approx(metadata["best_mean_cv_rmse"])

    # 9. Target identik dengan XGBoost: lightgbm_test_predictions.csv dan xgboost_test_predictions.csv punya source_row, target_row, actual, dan persistence_prediction yang sama persis
    xgb_frame = pd.read_csv(WEEK_DIR / "results/gold/xgboost_test_predictions.csv")
    assert frame.source_row.tolist() == xgb_frame.source_row.tolist()
    assert frame.target_row.tolist() == xgb_frame.target_row.tolist()
    np.testing.assert_array_equal(frame.actual, xgb_frame.actual)
    np.testing.assert_array_equal(frame.persistence_prediction, xgb_frame.persistence_prediction)
