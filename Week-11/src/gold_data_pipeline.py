"""Gold GLD data preparation only: no fitted preprocessing or estimator."""
from __future__ import annotations
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

WEEK_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = WEEK_DIR / "data/raw/FINAL_USO.csv"
MANIFEST_PATH = WEEK_DIR / "results/gold/split_manifest.json"
EXPECTED_SHA256 = "f1a366b4edfbdf074136727f3653058be6d91456e0128a462220797400ae6087"
REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume")
FEATURES = ("Open", "High", "Low", "Close", "Volume", "close_lag_1",
            "close_lag_5", "close_mean_prev_5", "close_std_prev_5",
            "close_return_1", "close_return_5")
CONFIG = {
    "target_source": "Adj Close", "target_name": "next_adj_close",
    "horizon": "next recorded observation",
    "information_cutoff": "End of source observation t; no t+1 predictors.",
    "close_lags": [1, 5], "rolling_window": 5, "rolling_shift": 1,
    "rolling_min_periods": 5, "rolling_std_ddof": 1,
    "returns": [1, 5], "return_definition": "Close(t) / Close(t-k) - 1; no fill.",
    "max_source_target_gap_calendar_days": 6,
    "test_fraction": 0.20, "test_count_rule": "ceil(n_eligible * test_fraction)",
    "boundary_purge_samples": 1,
    "cv": {"class": "TimeSeriesSplit", "n_splits": 5, "gap": 1},
    "preprocessing": "No fitted transformation, scaling, or imputation.",
}

@dataclass
class GoldData:
    frame: pd.DataFrame
    path: Path
    sha256: str

@dataclass
class GoldSamples:
    X: pd.DataFrame
    y: pd.Series
    timeline: pd.DataFrame
    exclusions: list[dict]
    dataset: GoldData

@dataclass
class GoldSplit:
    samples: GoldSamples
    train_indices: list[int]
    test_indices: list[int]
    purged_indices: list[int]

    @property
    def X_train(self):
        return self.samples.X.loc[self.train_indices].copy()

    @property
    def X_test(self):
        return self.samples.X.loc[self.test_indices].copy()

    @property
    def y_train(self):
        return self.samples.y.loc[self.train_indices].copy()

    @property
    def y_test(self):
        return self.samples.y.loc[self.test_indices].copy()


def load_gold_data(path: str | Path = DATASET_PATH, *,
                   expected_sha256: str = EXPECTED_SHA256,
                   expected_shape: tuple[int, int] | None = (1718, 81)) -> GoldData:
    """Validate bytes, original row order, dates, and numeric values read-only."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Dataset SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names")
    frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    absent = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    if expected_shape is not None and frame.shape != expected_shape:
        raise ValueError(f"Unexpected dataset shape: {frame.shape}")
    if len(frame) < 2:
        raise ValueError("At least two observations required")
    if not frame.Date.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False).all():
        raise ValueError("Date must use YYYY-MM-DD")
    try:
        dates = pd.to_datetime(frame.Date, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid Date") from exc
    if dates.duplicated().any():
        raise ValueError("Duplicate dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be strictly ascending")
    for column in frame.columns.drop("Date"):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric column: {column}") from exc
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite or missing values: {column}")
    if (frame[["Close", "Adj Close"]] <= 0).any().any():
        raise ValueError("Close and target prices must be positive")
    frame["Date"] = dates
    frame.index.name = "source_row"
    return GoldData(frame, path, digest)


def make_gold_samples(dataset: GoldData) -> GoldSamples:
    """Predict next recorded observation; lag/rolling features never look forward."""
    frame = dataset.frame
    close = frame.Close
    X = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    for lag in CONFIG["close_lags"]:
        X[f"close_lag_{lag}"] = close.shift(lag)
    previous = close.shift(1)
    X["close_mean_prev_5"] = previous.rolling(5, min_periods=5, center=False).mean()
    X["close_std_prev_5"] = previous.rolling(5, min_periods=5, center=False).std(ddof=1)
    for lag in CONFIG["returns"]:
        X[f"close_return_{lag}"] = close / close.shift(lag) - 1.0
    X = X.loc[:, list(FEATURES)]
    # Forward shifts are labels/audit metadata only, never predictors.
    target = frame["Adj Close"].shift(-1).rename("next_adj_close")
    target_dates = frame.Date.shift(-1)
    target_rows = pd.Series(frame.index, index=frame.index).shift(-1)
    gaps = (target_dates - frame.Date).dt.days
    exclusions, retained = [], []
    for row in frame.index:
        reasons = []
        if X.loc[row].isna().any():
            reasons.append("leading_history_unavailable")
        if pd.isna(target_dates.loc[row]):
            reasons.append("no_next_recorded_observation")
        elif gaps.loc[row] > 6:
            reasons.append("source_target_gap_exceeds_6_calendar_days")
        if reasons:
            exclusions.append({
                "source_row": int(row), "source_date": frame.loc[row,"Date"].date().isoformat(),
                "target_row": None if pd.isna(target_rows.loc[row]) else int(target_rows.loc[row]),
                "target_date": None if pd.isna(target_dates.loc[row]) else target_dates.loc[row].date().isoformat(),
                "gap_calendar_days": None if pd.isna(gaps.loc[row]) else int(gaps.loc[row]),
                "reasons": reasons,
            })
        else:
            retained.append(int(row))
    X, y = X.loc[retained].copy(), target.loc[retained].copy()
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Unexpected nonfinite engineered values")
    timeline = pd.DataFrame({
        "source_date": frame.loc[retained,"Date"], "target_date": target_dates.loc[retained],
        "target_row": target_rows.loc[retained].astype(int),
        "gap_calendar_days": gaps.loc[retained].astype(int),
    })
    if not (timeline.target_date > timeline.source_date).all():
        raise ValueError("Target must follow information cutoff")
    return GoldSamples(X, y, timeline, exclusions, dataset)


def _validate_partition(samples, train, test, purge):
    combined = train + purge + test
    if not train or not test or len(purge) != 1:
        raise ValueError("Nonempty train/test and exactly one purge required")
    if len(set(combined)) != len(combined) or set(combined) != set(samples.X.index):
        raise ValueError("Partition overlap or incomplete coverage")
    if combined != samples.X.index.tolist():
        raise ValueError("Partition must preserve chronological sample order")
    times = samples.timeline
    if times.loc[train,"target_date"].max() >= times.loc[test,"source_date"].min():
        raise ValueError("Training targets violate test information boundary")


def chronological_split(samples: GoldSamples) -> GoldSplit:
    """Latest ceil(20%) held out; purge one eligible sample before test."""
    count = len(samples.X)
    cut = count - int(np.ceil(count * CONFIG["test_fraction"]))
    if cut <= 1:
        raise ValueError("Not enough samples for train, purge, and test")
    indices = samples.X.index.tolist()
    train, purge, test = indices[:cut-1], indices[cut-1:cut], indices[cut:]
    _validate_partition(samples, train, test, purge)
    return GoldSplit(samples, train, test, purge)


def make_time_series_cv() -> TimeSeriesSplit:
    """Configuration for the training subset only; no fitting."""
    return TimeSeriesSplit(n_splits=5, gap=1)


def validate_cv_folds(split: GoldSplit) -> list[dict]:
    X = split.X_train
    timeline = split.samples.timeline
    records = []
    for number, (tr, va) in enumerate(make_time_series_cv().split(X), 1):
        train, valid = X.index[tr].tolist(), X.index[va].tolist()
        if not train or not valid or set(train) & set(valid):
            raise ValueError("Invalid CV sizes or overlap")
        if tr[-1] + 1 >= va[0]:
            raise ValueError("CV gap missing")
        if timeline.loc[train,"source_date"].max() >= timeline.loc[valid,"source_date"].min():
            raise ValueError("CV source timestamps out of order")
        if timeline.loc[train,"target_date"].max() >= timeline.loc[valid,"source_date"].min():
            raise ValueError("CV training target violates validation cutoff")
        records.append({
            "fold": number, "n_train": len(train), "n_validation": len(valid),
            "train_source_indices": train, "validation_source_indices": valid,
            "gap_source_indices": X.index[tr[-1]+1:va[0]].tolist(),
            "train_source_start": timeline.loc[train,"source_date"].min().date().isoformat(),
            "train_source_end": timeline.loc[train,"source_date"].max().date().isoformat(),
            "train_target_end": timeline.loc[train,"target_date"].max().date().isoformat(),
            "validation_source_start": timeline.loc[valid,"source_date"].min().date().isoformat(),
            "validation_source_end": timeline.loc[valid,"source_date"].max().date().isoformat(),
            "validation_target_end": timeline.loc[valid,"target_date"].max().date().isoformat(),
        })
    return records


def _fingerprint(content: dict) -> str:
    value = {k:v for k,v in content.items() if k != "split_sha256"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def make_gold_manifest(split: GoldSplit) -> dict:
    samples = split.samples
    _validate_partition(samples, split.train_indices, split.test_indices, split.purged_indices)
    ranges = {}
    for name, indices in [("training",split.train_indices),("testing",split.test_indices)]:
        times = samples.timeline.loc[indices]
        ranges[name] = {
            "source_start": times.source_date.min().date().isoformat(),
            "source_end": times.source_date.max().date().isoformat(),
            "target_start": times.target_date.min().date().isoformat(),
            "target_end": times.target_date.max().date().isoformat(),
        }
    content = {
        "schema_version": 1, "dataset_path": "Week-11/data/raw/FINAL_USO.csv",
        "dataset_sha256": samples.dataset.sha256,
        "source_row_count": len(samples.dataset.frame),
        "source_column_count": len(samples.dataset.frame.columns),
        "target_definition": "y(t) = Adj Close at the next recorded observation t+1",
        "feature_allowlist": list(FEATURES), "config": CONFIG,
        "source_index_convention": "Zero-based original CSV data row, excluding header.",
        "eligible_count": len(samples.X), "excluded_count": len(samples.exclusions),
        "excluded_rows": samples.exclusions,
        "train_source_indices": split.train_indices,
        "test_source_indices": split.test_indices, "purged_indices": split.purged_indices,
        "purge_reason": "One-sample temporal information-boundary purge before test.",
        "n_train": len(split.train_indices), "n_test": len(split.test_indices),
        "n_purged": len(split.purged_indices), "date_ranges": ranges,
        "cv_folds": validate_cv_folds(split),
        "calendar_note": "Recorded observation steps; exchange-session completeness unverified.",
    }
    content["split_sha256"] = _fingerprint(content)
    return content


def replay_gold_manifest(path: str | Path = MANIFEST_PATH,
                         dataset: GoldData | None = None) -> GoldSplit:
    """Rebuild deterministic features and replay stored indices; no resplitting."""
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if content.get("split_sha256") != _fingerprint(content):
        raise ValueError("Manifest fingerprint mismatch")
    dataset = load_gold_data() if dataset is None else dataset
    if content["dataset_sha256"] != dataset.sha256:
        raise ValueError("Manifest dataset hash mismatch")
    if content["config"] != CONFIG or content["feature_allowlist"] != list(FEATURES):
        raise ValueError("Manifest configuration mismatch")
    samples = make_gold_samples(dataset)
    split = GoldSplit(samples, content["train_source_indices"],
                      content["test_source_indices"], content["purged_indices"])
    if make_gold_manifest(split) != content:
        raise ValueError("Manifest differs from deterministic reconstruction")
    return split


def main() -> None:
    dataset = load_gold_data()
    split = chronological_split(make_gold_samples(dataset))
    content = make_gold_manifest(split)
    if make_gold_manifest(chronological_split(make_gold_samples(dataset))) != content:
        raise RuntimeError("Non-deterministic manifest")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        if json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) != content:
            raise FileExistsError("Refusing to overwrite a different Gold manifest")
    else:
        with MANIFEST_PATH.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(content, indent=2, allow_nan=False) + chr(10))
    replay = replay_gold_manifest()
    pd.testing.assert_frame_equal(split.X_train, replay.X_train)
    pd.testing.assert_frame_equal(split.X_test, replay.X_test)
    print(json.dumps({k:content[k] for k in [
        "dataset_sha256","source_row_count","eligible_count","excluded_count",
        "n_train","n_test","n_purged","purged_indices","date_ranges",
        "excluded_rows","split_sha256"]}, indent=2))
    print("CV fold boundaries:")
    for fold in content["cv_folds"]:
        print(json.dumps({k:v for k,v in fold.items() if not k.endswith("_indices")}))
    print("Manifest replay and deterministic reconstruction: PASS")
    print("Manifest file SHA256:", hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest())
    print("No preprocessing fit, model training, or model artifact generation.")


if __name__ == "__main__":
    main()
