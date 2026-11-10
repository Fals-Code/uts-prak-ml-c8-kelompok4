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
    EXPECTED_SHA256,
    FEATURES,
    MANIFEST_PATH,
    WEEK_DIR,
    make_time_series_cv,
    replay_gold_manifest,
    validate_cv_folds,
)
from .train_xgboost_gold import (
    EXPECTED_SPLIT,
    PRICE_UNIT,
    SEED,
    persistence_predict,
    save_model,
    selected_model_cv,
    sha256,
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


def normalize_gain(raw) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    total = arr.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Feature importance sum must be positive and finite")
    return arr / total


def _write_json(path: Path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + chr(10))


def _figures(actual, predicted, importance, output_root=WEEK_DIR):
    folder = output_root / "figures/gold"
    with plt.rc_context({"font.size": 10, "figure.dpi": 150}):
        fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
        ax.scatter(actual, predicted, s=15, alpha=0.55, color="#176b87")
        low = min(float(np.min(actual)), float(np.min(predicted)))
        high = max(float(np.max(actual)), float(np.max(predicted)))
        ax.plot([low, high], [low, high], "--", color="#b44a35", label="Ideal y=x")
        ax.set(title="Gold LightGBM: Actual vs Predicted (test)",
               xlabel="Actual GLD adjusted close (dataset price units)",
               ylabel="Predicted GLD adjusted close (dataset price units)")
        ax.legend()
        fig.savefig(folder / "lightgbm_actual_vs_predicted.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
        ax.scatter(predicted, np.asarray(actual) - predicted, s=15, alpha=0.55, color="#176b87")
        ax.axhline(0, linestyle="--", color="#b44a35")
        ax.set(title="Gold LightGBM: Residuals (test)",
               xlabel="Predicted GLD adjusted close (dataset price units)",
               ylabel="Residual: actual - predicted (dataset price units)")
        fig.savefig(folder / "lightgbm_residuals.png")
        plt.close(fig)

        ordered = sorted(importance.items(), key=lambda item: item[1])
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        ax.barh([p[0] for p in ordered], [p[1] for p in ordered], color="#176b87")
        ax.set(title="Gold LightGBM: Feature importance",
               xlabel="Normalized gain (predictive contribution; not causation)")
        fig.savefig(folder / "lightgbm_feature_importance.png")
        plt.close(fig)


def _protected_snapshot(output_root: Path = WEEK_DIR) -> dict:
    """Protect active project files; omit only the current run's output namespace."""
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "Week-11"],
        cwd=WEEK_DIR.parent, text=True).splitlines()
    generated = {(output_root / name).resolve() for name in ARTIFACTS}
    selected = {}
    for name in sorted(set(names)):
        path = (WEEK_DIR.parent / name).resolve()
        if not path.is_file() or path in generated:
            continue
        if output_root != WEEK_DIR and path.is_relative_to(output_root):
            continue
        selected[name] = sha256(path)
    return selected


def run_gold_experiment(output_root: Path = WEEK_DIR) -> dict:
    output_root = Path(output_root).resolve()
    if not output_root.is_relative_to(WEEK_DIR.resolve()):
        raise ValueError("Output must remain inside Week-11")
    for name in ARTIFACTS:
        if (output_root / name).exists():
            raise FileExistsError(f"Inspect existing artifact before rerunning: {name}")
    protected = _protected_snapshot(output_root)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["split_sha256"] != EXPECTED_SPLIT:
        raise ValueError("Gold split fingerprint differs from approved design")
    split = replay_gold_manifest()
    fold_audit = validate_cv_folds(split)
    if (len(split.X_train), len(split.X_test)) != (1367, 342):
        raise ValueError("Unexpected Gold train/test counts")
    if split.samples.dataset.sha256 != EXPECTED_SHA256:
        raise ValueError("Dataset mismatch")
    if split.samples.timeline.target_row.duplicated().any():
        raise ValueError("Duplicate target observations")
    log = {"started_utc": datetime.now(timezone.utc).isoformat(),
           "command": [sys.executable, "-B", "-m", "src.train_lightgbm_gold"],
           "events": [], "protected_existing_file_sha256": protected}
    print("Gold search: 12 candidates x 5 folds = 60 fits, plus 1 training refit.", flush=True)
    print("Selected-parameter CV: 5 additional training-only fits for all metrics.", flush=True)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        search = fit_training_search(split.X_train, split.y_train)
        cv_table = pd.DataFrame(search.cv_results_)
        cv_folds = cv_table[[f"split{i}_test_score" for i in range(5)]].to_numpy(dtype=float)
        if cv_folds.shape != (12, 5) or not np.isfinite(cv_folds).all():
            raise ValueError("Search must yield 60 finite fold scores")
        best = search.best_estimator_
        selected_cv = selected_model_cv(best, split.X_train, split.y_train)
        np.testing.assert_allclose(
            [v["RMSE"] for v in selected_cv["fold_metrics"]],
            -cv_folds[search.best_index_], rtol=1e-10, atol=1e-10)
        log["events"].append("Training-only hyperparameter selection and descriptive CV complete.")
        train_prediction = best.predict(split.X_train)
        test_prediction = best.predict(split.X_test)
        baseline = persistence_predict(split.X_test)
        metrics = {
            "lightgbm_training": evaluate_regression(split.y_train, train_prediction),
            "lightgbm_cv": selected_cv,
            "lightgbm_testing": evaluate_regression(split.y_test, test_prediction),
            "persistence_testing": evaluate_regression(split.y_test, baseline),
        }
        log["events"].append("Final test metrics calculated once per model/baseline; no retuning.")
    warning_messages = [{"category": w.category.__name__, "message": str(w.message)} for w in recorded]
    elapsed = time.perf_counter() - start
    rows = prediction_frame(split, test_prediction, baseline)
    if rows.source_row.tolist() != manifest["test_source_indices"]:
        raise ValueError("Test predictions misaligned with manifest")
    for folder in ["results/gold", "models/gold", "figures/gold"]:
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    model_hash = save_model(best, output_root / "models/gold/lightgbm_pipeline.joblib",
                            split.X_test, test_prediction)
    rows.to_csv(output_root / "results/gold/lightgbm_test_predictions.csv", index=False)
    cv_table.to_csv(output_root / "results/gold/lightgbm_cv_results.csv", index=False)
    metric_rows = [
        {"model": "LightGBM", "subset": "training", **metrics["lightgbm_training"]},
        {"model": "LightGBM", "subset": "cross_validation_mean", **selected_cv["mean"]},
        {"model": "LightGBM", "subset": "testing", **metrics["lightgbm_testing"]},
        {"model": "Persistence", "subset": "testing", **metrics["persistence_testing"]},
    ]
    pd.DataFrame(metric_rows).to_csv(output_root / "results/gold/lightgbm_metrics.csv", index=False)
    _write_json(output_root / "results/gold/lightgbm_metrics.json", metrics)
    raw_gain = best.named_steps["model"].feature_importances_
    importance = {name: float(val) for name, val in zip(FEATURES, normalize_gain(raw_gain), strict=True)}
    _figures(split.y_test.to_numpy(), test_prediction, importance, output_root)
    if _protected_snapshot(output_root) != protected:
        raise RuntimeError("An existing protected file changed")
    model_error = metrics["lightgbm_testing"]["RMSE"]
    persistence_error = metrics["persistence_testing"]["RMSE"]
    metadata = {
        "dataset_sha256": split.samples.dataset.sha256,
        "split_fingerprint": manifest["split_sha256"],
        "manifest_file_sha256": sha256(MANIFEST_PATH),
        "gold_pipeline_source_sha256": sha256(WEEK_DIR / "src/gold_data_pipeline.py"),
        "trainer_source_sha256": sha256(Path(__file__)),
        "target": "GLD ETF adjusted closing price at next recorded observation",
        "price_units": PRICE_UNIT, "feature_schema": list(FEATURES),
        "n_train": len(split.X_train), "n_test": len(split.X_test),
        "purged_indices": split.purged_indices, "date_ranges": manifest["date_ranges"],
        "cv_configuration": {"class": "TimeSeriesSplit", "n_splits": 5, "gap": 1},
        "cv_fold_audit": fold_audit,
        "search_configuration": {"class": "RandomizedSearchCV", "n_iter": 12,
                                "scoring": "neg_root_mean_squared_error", "random_state": 42,
                                "n_jobs": 1, "refit": True, "error_score": "raise"},
        "search_space": SEARCH_SPACE, "base_estimator_parameters": {
            "objective": "regression", "random_state": 42,
            "n_jobs": 1, "deterministic": True, "force_row_wise": True,
            "subsample_freq": 1, "importance_type": "gain", "verbose": -1},
        "best_parameters": search.best_params_, "best_candidate_index": int(search.best_index_),
        "best_mean_cv_rmse": float(-search.best_score_),
        "best_std_cv_rmse_ddof0": float(cv_table.loc[search.best_index_, "std_test_score"]),
        "best_fold_cv_rmse": (-cv_folds[search.best_index_]).tolist(),
        "fit_counts": {"search_candidates": 12, "search_cv_fits": 60,
                       "selected_parameter_metric_cv_fits": 5, "total_cv_fits": 65, "final_refits": 1},
        "metrics": metrics,
        "feature_importance_normalized_gain": importance,
        "rmse_generalization_gap_test_minus_train":
            float(model_error - metrics["lightgbm_training"]["RMSE"]),
        "test_rmse_improvement_vs_persistence_percent":
            float(100 * (persistence_error - model_error) / persistence_error),
        "model_sha256": model_hash, "reload_prediction_check": "exact equality PASS",
        "python": platform.python_version(),
        "libraries": {n: version(n) for n in [
            "lightgbm", "scikit-learn", "numpy", "pandas", "matplotlib", "joblib"]},
        "elapsed_seconds": elapsed, "warnings": warning_messages,
        "selection_policy": "Training CV RMSE only; persistence/test metrics never affect selection.",
        "cv_metrics_policy": "Five extra fixed-parameter CV fits on training before test; no extra search.",
        "reproducibility": "Approved manifest replay; fixed seeds; CPU/single thread; no learned preprocessing.",
        "limitations": [
            "Next recorded observation is not a guaranteed consecutive exchange session.",
            "Data end in 2018; no claim about current market performance.",
            "Adjusted-price point-in-time provenance, currency, and license remain unverified.",
            "Feature importance indicates predictive contribution, not causality.",
            "Perbandingan dengan XGBoost dilakukan pada Tahap 3 oleh Azzam",
        ],
    }
    _write_json(output_root / "results/gold/lightgbm_metadata.json", metadata)
    hashes = {name: sha256(output_root / name) for name in ARTIFACTS if not name.endswith("execution_log.json")}
    log.update({"status": "COMPLETE", "fit_counts": metadata["fit_counts"],
                "best_parameters": search.best_params_, "metrics": metrics,
                "warnings": warning_messages, "artifact_sha256": hashes,
                "existing_files_preserved": True, "elapsed_seconds": elapsed})
    log["events"].append("Pipeline saved/reloaded exactly; artifacts saved; existing files preserved.")
    summary = {k: metadata[k] for k in [
        "best_parameters", "best_mean_cv_rmse", "best_std_cv_rmse_ddof0",
        "best_fold_cv_rmse", "fit_counts", "metrics", "rmse_generalization_gap_test_minus_train",
        "test_rmse_improvement_vs_persistence_percent", "model_sha256", "warnings"]}
    rendered = json.dumps(summary, indent=2, allow_nan=False)
    log["raw_summary_output"] = rendered
    _write_json(output_root / "results/gold/lightgbm_execution_log.json", log)
    print(rendered, flush=True)
    print("Reload equality: PASS; dataset/manifest/existing-file preservation: PASS.", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=WEEK_DIR,
                        help="New output namespace inside Week-11; existing outputs are never overwritten.")
    args = parser.parse_args()
    run_gold_experiment(args.output_root)


if __name__ == "__main__":
    main()
