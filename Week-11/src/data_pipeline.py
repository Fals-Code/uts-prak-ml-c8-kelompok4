"""Read-only dataset loading, unfitted preprocessing, and auditable row splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

WEEK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = WEEK_DIR.parent / "Week-10/data/raw/Penguins.csv"
DEFAULT_MANIFEST = WEEK_DIR / "results/split_manifest.json"
EXPECTED_SHA256 = "5ea908ae6fd5363b6d649f9eecdd63c010caaa389ee9498483e9ec30647dee2d"
TARGET = "Body Mass (g)"
NUMERIC_FEATURES = ("Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)")
CATEGORICAL_FEATURES = ("Species", "Island", "Sex")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MISSING_POLICY = {
    "target": "Exclude missing target rows; never impute target.",
    "numeric": "Training-fold median via SimpleImputer; keep empty features.",
    "categorical": "Strip strings; blanks are missing; Sex uppercase and '.' missing.",
    "categorical_imputation": "Constant Unknown via SimpleImputer inside pipeline.",
    "encoding": "OneHotEncoder(handle_unknown='ignore', sparse_output=False).",
    "fitting": "Return unfitted preprocessor; fit only within each training fold.",
}


@dataclass
class Dataset:
    """X/y retain zero-based source row indices, excluding the CSV header."""
    X: pd.DataFrame
    y: pd.Series
    source_path: Path
    sha256: str
    original_shape: tuple[int, int]
    excluded_indices: list[int]
    raw_missing: dict[str, int]
    normalized_missing: dict[str, int]
    sex_dot_count: int


@dataclass
class DataSplit:
    """Raw, untransformed subsets. Source row indices are never model features."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(
    path: str | Path = DEFAULT_DATASET, *, expected_sha256: str = EXPECTED_SHA256
) -> Dataset:
    """Verify bytes before parsing, validate schema, normalize deterministically."""
    source = Path(path).resolve()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Dataset SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    # Inspect original header before pandas can silently suffix duplicate names.
    header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), [])
    normalized_header = [name.strip() for name in header]
    if len(set(normalized_header)) != len(normalized_header):
        raise ValueError("Duplicate column names after strip()")
    frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    missing_columns = sorted(set((*FEATURES, TARGET)) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    raw_missing = {c: int(frame[c].isna().sum()) for c in (*FEATURES, TARGET)}
    for column in (*NUMERIC_FEATURES, TARGET):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite numeric value in {column}")
    sex_dot_count = int(frame["Sex"].astype("string").str.strip().eq(".").sum())
    for column in CATEGORICAL_FEATURES:
        values = frame[column].astype("string").str.strip()
        if column == "Sex":
            values = values.str.upper().replace(".", pd.NA)
        values = values.replace("", pd.NA)
        # sklearn imputers receive numpy NaN, never pandas.NA.
        frame[column] = values.astype(object).where(values.notna(), np.nan)
    valid_target = frame[TARGET].notna()
    return Dataset(
        X=frame.loc[valid_target, list(FEATURES)].copy(),
        y=frame.loc[valid_target, TARGET].copy(),
        source_path=source,
        sha256=digest,
        original_shape=frame.shape,
        excluded_indices=frame.index[~valid_target].astype(int).tolist(),
        raw_missing=raw_missing,
        normalized_missing={c: int(frame.loc[valid_target, c].isna().sum()) for c in FEATURES},
        sex_dot_count=sex_dot_count,
    )


def make_preprocessor() -> ColumnTransformer:
    """Create a fresh UNFITTED transformer for a downstream model Pipeline."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown",
                                  keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_FEATURES)),
         ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def split_dataset(dataset: Dataset, *, test_size: float = 0.20,
                  random_state: int = 42) -> DataSplit:
    """Split raw features and targets together; learn no preprocessing statistics."""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=test_size,
        random_state=random_state, shuffle=True,
    )
    return DataSplit(X_train, X_test, y_train, y_test)


def split_from_manifest(dataset: Dataset, path: str | Path = DEFAULT_MANIFEST) -> DataSplit:
    """Replay exact ordered source indices without refitting or resplitting."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (manifest["dataset_sha256"] != dataset.sha256
            or manifest["target"] != TARGET
            or manifest["numeric_features"] != list(NUMERIC_FEATURES)
            or manifest["categorical_features"] != list(CATEGORICAL_FEATURES)):
        raise ValueError("Manifest dataset or feature schema mismatch")
    train, test = manifest["train_indices"], manifest["test_indices"]
    combined = train + test
    if (len(combined) != len(set(combined))
            or set(combined) != set(dataset.X.index)
            or len(train) != manifest["n_train"] or len(test) != manifest["n_test"]):
        raise ValueError("Manifest indices overlap, are incomplete, or counts mismatch")
    payload = json.dumps({"train": train, "test": test}, separators=(",", ":"))
    if hashlib.sha256(payload.encode()).hexdigest() != manifest["split_sha256"]:
        raise ValueError("Manifest split hash mismatch")
    return DataSplit(dataset.X.loc[train].copy(), dataset.X.loc[test].copy(),
                     dataset.y.loc[train].copy(), dataset.y.loc[test].copy())


def build_manifest(dataset: Dataset, split: DataSplit, *, test_size: float = 0.20,
                   random_state: int = 42) -> dict:
    """Build actual counts and ordered index lists from the executed split."""
    train = split.X_train.index.astype(int).tolist()
    test = split.X_test.index.astype(int).tolist()
    if (len(train + test) != len(set(train + test))
            or set(train + test) != set(dataset.X.index)):
        raise ValueError("Split must cover every eligible row exactly once")
    if not (split.X_train.index.equals(split.y_train.index)
            and split.X_test.index.equals(split.y_test.index)):
        raise ValueError("Feature and target indices are misaligned")
    payload = json.dumps({"train": train, "test": test}, separators=(",", ":"))
    return {
        "schema_version": 1,
        "source_file": dataset.source_path.name,
        "source_repository_path": "Week-10/data/raw/Penguins.csv",
        "dataset_sha256": dataset.sha256,
        "target": TARGET,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "feature_order": list(FEATURES),
        "random_state": random_state, "test_size": test_size, "shuffle": True,
        "original_shape": list(dataset.original_shape),
        "n_original": dataset.original_shape[0],
        "n_target_valid": len(dataset.y),
        "n_excluded": len(dataset.excluded_indices),
        "excluded_target_indices": dataset.excluded_indices,
        "n_train": len(train), "n_test": len(test),
        "index_convention": "Zero-based CSV data row position, excluding header.",
        "train_indices": train, "test_indices": test,
        "split_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "raw_missing": dataset.raw_missing,
        "normalized_feature_missing_target_valid": dataset.normalized_missing,
        "sex_dot_count": dataset.sex_dot_count,
        "missing_policy": MISSING_POLICY,
        "preprocessor_fitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    dataset = load_dataset()
    split = split_dataset(dataset)
    manifest = build_manifest(dataset, split)
    assert manifest == build_manifest(dataset, split_dataset(dataset))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + chr(10), encoding="utf-8")
    replay = split_from_manifest(dataset, args.manifest)
    assert replay.X_train.equals(split.X_train) and replay.X_test.equals(split.X_test)
    print(json.dumps({k: manifest[k] for k in [
        "dataset_sha256", "original_shape", "n_target_valid", "n_excluded",
        "n_train", "n_test", "split_sha256", "raw_missing",
        "normalized_feature_missing_target_valid", "sex_dot_count"]}, indent=2))
    print("Reproducibility: PASS; manifest replay: PASS; preprocessor remains unfitted.")
    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
