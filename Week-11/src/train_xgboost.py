"""Training-only randomized CV and one final held-out XGBoost evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .data_pipeline import (
    DEFAULT_MANIFEST, EXPECTED_SHA256, FEATURES, TARGET, WEEK_DIR,
    load_dataset, make_preprocessor, split_from_manifest,
)
from .evaluation import evaluate_regression

SEED = 42
EXPECTED_SPLIT_SHA256 = "fd37f6636ccce38aa9f7f6b6ab9b3ccd53f269a0e83ad60e032efc471ee0b319"
SEARCH_SPACE = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.05, 0.10],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__reg_lambda": [1.0, 5.0],
}
OUTPUT_FILES = (
    "results/xgboost_metrics.json", "results/xgboost_metrics.csv",
    "results/xgboost_cv_results.csv", "results/xgboost_metadata.json",
    "results/xgboost_test_predictions.csv", "models/xgboost_pipeline.joblib",
    "figures/xgboost_actual_vs_predicted.png", "figures/xgboost_residuals.png",
    "figures/xgboost_feature_importance.png",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_xgboost_pipeline() -> Pipeline:
    """Fresh unfitted preprocessing and estimator; no preprocessing outside CV."""
    return Pipeline([
        ("preprocessing", make_preprocessor()),
        ("model", XGBRegressor(
            objective="reg:squarederror", random_state=SEED,
            n_jobs=1, tree_method="hist", importance_type="gain",
        )),
    ])


def make_search() -> RandomizedSearchCV:
    return RandomizedSearchCV(
        make_xgboost_pipeline(), param_distributions=SEARCH_SPACE,
        n_iter=12, scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=1, refit=True, error_score="raise",
        return_train_score=True,
    )


def save_and_verify_model(pipeline, path: Path, X, expected_predictions) -> str:
    """Persist the whole fitted pipeline and check reload without refitting."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(X), expected_predictions, rtol=0, atol=0)
    return sha256(path)


def make_test_predictions(y_true: pd.Series, predictions) -> pd.DataFrame:
    """Use float64 CSV columns to preserve original float32 predictions exactly."""
    predicted = np.asarray(predictions, dtype=np.float64)
    actual = y_true.to_numpy(dtype=np.float64)
    if predicted.ndim != 1 or len(predicted) != len(actual):
        raise ValueError("Prediction length or shape mismatch")
    return pd.DataFrame({
        "source_row": y_true.index.to_numpy(),
        "actual_g": actual,
        "predicted_g": predicted,
        "residual_g": actual - predicted,
    })


def write_figures(pipeline, actual, predicted, root: Path) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "figure.dpi": 140}):
        fig, ax = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
        ax.scatter(actual, predicted, s=30, alpha=0.75, color="#176b87")
        limits = [float(min(min(actual), min(predicted))), float(max(max(actual), max(predicted)))]
        ax.plot(limits, limits, "--", color="#bd4b30", label="Ideal: y = x")
        ax.set(title="XGBoost: Actual vs Predicted — Test Set",
               xlabel="Actual body mass (g)", ylabel="Predicted body mass (g)")
        ax.legend()
        fig.savefig(figure_dir / "xgboost_actual_vs_predicted.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
        ax.scatter(predicted, np.asarray(actual) - predicted,
                   s=30, alpha=0.75, color="#176b87")
        ax.axhline(0, linestyle="--", color="#bd4b30")
        ax.set(title="XGBoost: Residuals — Test Set",
               xlabel="Predicted body mass (g)", ylabel="Residual: actual − predicted (g)")
        fig.savefig(figure_dir / "xgboost_residuals.png")
        plt.close(fig)

        names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        importance = pipeline.named_steps["model"].feature_importances_
        if len(names) != len(importance):
            raise ValueError("Transformed feature names and importance lengths differ")
        order = np.argsort(importance)
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        ax.barh(np.asarray(names)[order], importance[order], color="#176b87")
        ax.set(title="XGBoost: Encoded Feature Importance",
               xlabel="Normalized gain (predictive contribution; not causation)")
        fig.savefig(figure_dir / "xgboost_feature_importance.png")
        plt.close(fig)


def run_experiment(output_root: Path = WEEK_DIR) -> dict:
    root = Path(output_root).resolve()
    if not root.is_relative_to(WEEK_DIR.resolve()):
        raise ValueError("All output must remain inside Week-11")
    for relative in OUTPUT_FILES:
        if (root / relative).exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {root / relative}")
    started = datetime.now(timezone.utc).isoformat()
    manifest_file_hash = sha256(DEFAULT_MANIFEST)
    preprocessing_source_hash = sha256(WEEK_DIR / "src/data_pipeline.py")
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    if manifest["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Split differs from approved Phase 3A manifest")
    dataset = load_dataset()
    split = split_from_manifest(dataset)
    if (len(split.X_train), len(split.X_test)) != (273, 69):
        raise ValueError("Unexpected train/test counts")
    if tuple(split.X_train.columns) != FEATURES or tuple(split.X_test.columns) != FEATURES:
        raise ValueError("Unexpected feature order")

    search = make_search()
    print("Training-only search: 12 candidates × 5 folds = 60 CV fits; 1 final refit.",
          flush=True)
    before = time.perf_counter()
    search.fit(split.X_train, split.y_train)  # The held-out test set never enters CV.
    elapsed = time.perf_counter() - before
    cv_results = pd.DataFrame(search.cv_results_)
    fold_columns = [f"split{i}_test_score" for i in range(5)]
    scores = cv_results[fold_columns].to_numpy(dtype=float)
    if scores.shape != (12, 5) or not np.isfinite(scores).all():
        raise ValueError("Expected 60 finite validation scores")
    best = search.best_estimator_  # Already refitted on the full training set.
    train_prediction = best.predict(split.X_train)
    test_prediction = best.predict(split.X_test)  # Single final held-out evaluation.
    metrics = {
        "training": evaluate_regression(split.y_train, train_prediction),
        "testing": evaluate_regression(split.y_test, test_prediction),
    }
    for directory in ("results", "models", "figures"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    model_hash = save_and_verify_model(
        best, root / "models/xgboost_pipeline.joblib", split.X_test, test_prediction
    )
    write_figures(best, split.y_test.to_numpy(), test_prediction, root)
    predictions = make_test_predictions(split.y_test, test_prediction)
    if predictions["source_row"].tolist() != manifest["test_indices"]:
        raise ValueError("Prediction indices differ from manifest")
    predictions.to_csv(root / "results/xgboost_test_predictions.csv", index=False)
    cv_results.to_csv(root / "results/xgboost_cv_results.csv", index=False)
    pd.DataFrame([{"subset": subset, **values} for subset, values in metrics.items()]
                 ).to_csv(root / "results/xgboost_metrics.csv", index=False)
    (root / "results/xgboost_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + chr(10), encoding="utf-8"
    )
    if (sha256(DEFAULT_MANIFEST) != manifest_file_hash
            or sha256(dataset.source_path) != EXPECTED_SHA256
            or sha256(WEEK_DIR / "src/data_pipeline.py") != preprocessing_source_hash):
        raise RuntimeError("Dataset, manifest, or shared preprocessing changed during run")
    metadata = {
        "model": "XGBRegressor", "started_utc": started,
        "dataset_sha256": dataset.sha256, "split_sha256": manifest["split_sha256"],
        "manifest_file_sha256": manifest_file_hash,
        "preprocessing_source_sha256": preprocessing_source_hash,
        "n_train": len(split.X_train), "n_test": len(split.X_test),
        "features": list(FEATURES), "target": TARGET, "random_state": SEED,
        "base_model_parameters": {
            "objective": "reg:squarederror", "random_state": SEED,
            "n_jobs": 1, "tree_method": "hist", "importance_type": "gain",
        },
        "cv": {"class": "KFold", "n_splits": 5, "shuffle": True, "random_state": SEED},
        "search": {"class": "RandomizedSearchCV", "n_iter": 12, "random_state": SEED,
                   "n_jobs": 1, "refit": True, "error_score": "raise",
                   "scoring": "neg_root_mean_squared_error"},
        "search_space": SEARCH_SPACE, "best_parameters": search.best_params_,
        "best_candidate_index": int(search.best_index_),
        "best_cv_rmse_g": float(-search.best_score_),
        "best_cv_rmse_std_g": float(search.cv_results_["std_test_score"][search.best_index_]),
        "candidate_count": len(cv_results), "successful_cv_fits": int(scores.size),
        "successful_final_refits": 1, "search_and_refit_seconds": elapsed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {name: version(name) for name in [
            "xgboost", "scikit-learn", "pandas", "numpy", "matplotlib", "joblib"]},
        "metrics": metrics, "metric_units": {
            "MAE": "g", "MAPE": "%", "MSE": "g^2", "RMSE": "g", "R2": "unitless"},
        "mape_zero_policy": "Reject zero targets; MAPE = sklearn MAPE * 100.",
        "model_sha256": model_hash, "reload_prediction_check": "exact equality PASS",
        "reproducibility": {
            "split_source": "results/split_manifest.json; ordered source row indices",
            "preprocessing": "Inside pipeline; fitted separately in each training fold",
            "tuning_data": "Training only; no early stopping or test eval_set",
            "test_evaluation": "One final metric evaluation after training-only selection",
            "test_predict_calls": "Final prediction plus reload consistency check; no retuning",
            "seeds": SEED, "threads": 1, "device": "CPU (hist)",
            "note": "Seeds/versions recorded; bitwise portability across platforms is not guaranteed.",
        },
        "limitations": [
            "Small dataset; random-row test split estimates have sampling uncertainty.",
            "Repeated-individual leakage remains unverified; see preprocessing README.",
            "Feature importance is predictive normalized gain, not causal evidence.",
        ],
    }
    (root / "results/xgboost_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + chr(10), encoding="utf-8"
    )
    print(json.dumps({
        "best_parameters": search.best_params_, "best_cv_rmse_g": -search.best_score_,
        "successful_cv_fits": int(scores.size), "successful_final_refits": 1,
        "metrics": metrics, "model_sha256": model_hash,
        "reload_verification": "PASS", "output_root": str(root),
    }, indent=2, allow_nan=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=WEEK_DIR)
    args = parser.parse_args()
    run_experiment(args.output_root)


if __name__ == "__main__":
    main()
