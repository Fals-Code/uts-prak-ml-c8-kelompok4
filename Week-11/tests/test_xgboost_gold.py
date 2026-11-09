"""Gold trainer contract tests; no randomized tuning is run by unit tests."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import train_xgboost_gold as trainer
from src.gold_data_pipeline import (
    FEATURES, DATASET_PATH, MANIFEST_PATH, WEEK_DIR,
    replay_gold_manifest, validate_cv_folds,
)
from src.evaluation import evaluate_regression


def test_dataset_and_manifest():
    assert hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()==(
        "f1a366b4edfbdf074136727f3653058be6d91456e0128a462220797400ae6087")
    m=json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert m["split_sha256"]==trainer.EXPECTED_SPLIT
    split=replay_gold_manifest()
    assert (len(split.X_train),len(split.X_test))==(1367,342)
    assert tuple(split.X_train.columns)==FEATURES
    assert split.samples.timeline.loc[split.train_indices,"target_date"].max()<split.samples.timeline.loc[split.test_indices,"source_date"].min()
    assert not split.samples.timeline.target_row.duplicated().any()


def test_cv_training_only_and_gap():
    split=replay_gold_manifest()
    for fold in validate_cv_folds(split):
        assert fold["train_target_end"]<fold["validation_source_start"]
        assert len(fold["gap_source_indices"])==1
        assert not set(fold["validation_source_indices"])&set(split.test_indices)


def test_bounded_search_and_unfitted_model():
    search=trainer.make_gold_search()
    assert search.n_iter==12 and search.random_state==42
    assert search.n_jobs==1 and search.refit is True and search.error_score=="raise"
    assert search.scoring=="neg_root_mean_squared_error"
    assert search.cv.n_splits==5 and search.cv.gap==1
    assert search.param_distributions["model__n_estimators"]==[100,200,300]
    assert set(search.param_distributions)==set(trainer.SEARCH_SPACE)
    assert len(list(inspect.signature(trainer.fit_training_search).parameters))==2
    params=search.estimator.named_steps["model"].get_params()
    assert params["device"]=="cpu" and params["n_jobs"]==1
    with pytest.raises(NotFittedError): check_is_fitted(search.estimator)


def test_search_receives_only_explicit_training_inputs(monkeypatch):
    split=replay_gold_manifest()
    seen=[]
    class FakeSearch:
        def fit(self,X,y):
            seen.append((X.index.tolist(),y.index.tolist()))
    fake=FakeSearch()
    monkeypatch.setattr(trainer,"make_gold_search",lambda:fake)
    assert trainer.fit_training_search(split.X_train,split.y_train) is fake
    assert seen==[(split.train_indices,split.train_indices)]
    assert not set(seen[0][0])&set(split.test_indices)


def test_metric_units_and_persistence():
    result=evaluate_regression([100,200,300],[110,180,330])
    assert result["MAE"]==pytest.approx(20)
    assert result["MAPE"]==pytest.approx(10)
    assert result["MSE"]==pytest.approx(1400/3)
    assert result["RMSE"]==pytest.approx(np.sqrt(1400/3))
    assert result["R2"]==pytest.approx(0.93)
    X=pd.DataFrame({"Close":[101.,202.]})
    np.testing.assert_array_equal(trainer.persistence_predict(X),[101.,202.])


def test_fixture_pipeline_roundtrip(tmp_path):
    X=pd.DataFrame(np.arange(44,dtype=float).reshape(4,11)+10,columns=FEATURES)
    y=np.array([100.,101.,103.,104.])
    model=trainer.make_gold_model().set_params(model__n_estimators=2,model__max_depth=2)
    model.fit(X,y)  # Small unit-test fixture only, not a reported market experiment.
    predicted=model.predict(X)
    p=tmp_path/"fixture.joblib"
    assert trainer.save_model(model,p,X,predicted)==trainer.sha256(p)
    np.testing.assert_array_equal(joblib.load(p).predict(X),predicted)


def test_prediction_frame_alignment_and_float_precision():
    split=replay_gold_manifest()
    baseline=trainer.persistence_predict(split.X_test)
    predicted=baseline.astype(np.float32)
    frame=trainer.prediction_frame(split,predicted,baseline)
    assert frame.source_row.tolist()==split.test_indices
    assert frame.target_row.tolist()==split.samples.timeline.loc[split.test_indices,"target_row"].tolist()
    assert frame.xgboost_prediction.dtype==np.float64
    np.testing.assert_array_equal(frame.xgboost_residual,frame.actual-frame.xgboost_prediction)
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()


def test_experiment_source_provenance_and_gold_namespace():
    metadata=json.loads((WEEK_DIR/"results/gold/xgboost_metadata.json").read_text(encoding="utf-8"))
    source=WEEK_DIR/"results/gold/xgboost_training_source.py.txt"
    assert hashlib.sha256(source.read_bytes()).hexdigest()==metadata["trainer_source_sha256"]
    assert all("/gold/" in name for name in trainer.ARTIFACTS)


def test_default_run_refuses_overwriting_without_training(monkeypatch):
    def fail(*args,**kwargs):
        raise AssertionError("Training must not be reached")
    monkeypatch.setattr(trainer,"fit_training_search",fail)
    with pytest.raises(FileExistsError):
        trainer.run_gold_experiment()


def test_output_namespace_must_remain_inside_week11():
    with pytest.raises(ValueError,match="inside Week-11"):
        trainer.run_gold_experiment(WEEK_DIR.parent)


def test_saved_gold_artifacts_readonly():
    metadata_path=WEEK_DIR/"results/gold/xgboost_metadata.json"
    if not metadata_path.exists():
        pytest.skip("Run the separately authorized Gold experiment before artifact integration audit")
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics=json.loads((WEEK_DIR/"results/gold/xgboost_metrics.json").read_text(encoding="utf-8"))
    frame=pd.read_csv(WEEK_DIR/"results/gold/xgboost_test_predictions.csv")
    modelpath=WEEK_DIR/"models/gold/xgboost_pipeline.joblib"
    assert trainer.sha256(modelpath)==metadata["model_sha256"]
    assert metadata["split_fingerprint"]==trainer.EXPECTED_SPLIT
    assert metadata["manifest_file_sha256"]==trainer.sha256(MANIFEST_PATH)
    assert metadata["feature_schema"]==list(FEATURES)
    split=replay_gold_manifest()
    assert len(frame)==342 and frame.source_row.tolist()==split.test_indices
    assert frame.target_row.tolist()==split.samples.timeline.loc[split.test_indices,"target_row"].tolist()
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
    loaded=joblib.load(modelpath)
    np.testing.assert_allclose(loaded.predict(split.X_test),frame.xgboost_prediction,rtol=0,atol=1e-10)
    np.testing.assert_allclose(frame.xgboost_residual,frame.actual-frame.xgboost_prediction,rtol=0,atol=1e-10)
    np.testing.assert_allclose(frame.persistence_prediction,split.X_test.Close,rtol=0,atol=1e-10)
    for key,value in metadata["best_parameters"].items():
        assert loaded.get_params()[key]==value
    assert metrics==metadata["metrics"]
    cv=pd.read_csv(WEEK_DIR/"results/gold/xgboost_cv_results.csv")
    assert len(cv)==12
    assert np.isfinite(cv[[f"split{i}_test_score" for i in range(5)]].to_numpy()).all()
    assert -cv.mean_test_score.max()==pytest.approx(metadata["best_mean_cv_rmse"])
