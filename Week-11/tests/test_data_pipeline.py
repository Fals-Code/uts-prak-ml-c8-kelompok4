"""Synthetic fixtures test function behavior only, never model performance."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
    build_manifest, load_dataset, make_preprocessor, split_dataset, split_from_manifest,
)


@pytest.fixture
def csv_fixture(tmp_path):
    def write(*, columns=None, rows=None):
        columns = columns or [*FEATURES, TARGET, "Individual ID"]
        if rows is None:
            rows = [
                [35 + i, 15 + i / 10, 180 + i, "Adelie", "Dream",
                 ["MALE", "FEMALE", None, ".", " male ", ""][i % 6],
                 None if i == 3 else 3000 + 50 * i, f"ID{i}"]
                for i in range(12)
            ]
        path = tmp_path / "fixture.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    return write


def load_fixture(csv_fixture, **kwargs):
    path, digest = csv_fixture(**kwargs)
    return load_dataset(path, expected_sha256=digest)


def test_bom_and_strip_headers(csv_fixture):
    data = load_fixture(csv_fixture, columns=[f" {c} " for c in [*FEATURES, TARGET, "Individual ID"]])
    assert data.original_shape == (12, 8)
    assert list(data.X.columns) == list(FEATURES)


def test_duplicate_normalized_header_rejected(csv_fixture):
    path, digest = csv_fixture(columns=[*FEATURES, TARGET, " Species "])
    with pytest.raises(ValueError, match="Duplicate column"):
        load_dataset(path, expected_sha256=digest)


def test_missing_required_column_rejected(csv_fixture):
    columns = [*FEATURES, TARGET, "Individual ID"]
    columns[0] = "Wrong column"
    with pytest.raises(ValueError, match="Missing required"):
        load_fixture(csv_fixture, columns=columns)


def test_wrong_hash_rejected(csv_fixture):
    path, _ = csv_fixture()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_dataset(path, expected_sha256="0" * 64)


def test_missing_target_removed_without_imputation(csv_fixture):
    data = load_fixture(csv_fixture)
    assert data.excluded_indices == [3]
    assert len(data.y) == 11 and not data.y.isna().any()
    assert data.X.index.equals(data.y.index)


def test_sex_normalized_and_rows_retained(csv_fixture):
    data = load_fixture(csv_fixture)
    assert 2 in data.X.index and pd.isna(data.X.loc[2, "Sex"])
    assert 9 in data.X.index and pd.isna(data.X.loc[9, "Sex"])
    assert pd.isna(data.X.loc[5, "Sex"])
    assert data.X.loc[4, "Sex"] == "MALE"
    assert data.sex_dot_count == 2


def test_feature_allowlist_excludes_target_and_identifiers(csv_fixture):
    data = load_fixture(csv_fixture)
    assert tuple(data.X.columns) == FEATURES
    assert TARGET not in data.X and "Individual ID" not in data.X


def test_split_reproducible_disjoint_complete_and_aligned(csv_fixture):
    data = load_fixture(csv_fixture)
    first, second = split_dataset(data), split_dataset(data)
    assert first.X_train.equals(second.X_train) and first.X_test.equals(second.X_test)
    train, test = set(first.X_train.index), set(first.X_test.index)
    assert not train & test
    assert train | test == set(data.X.index)
    assert len(first.X_train) + len(first.X_test) == len(data.y)
    assert first.X_train.index.equals(first.y_train.index)
    assert first.X_test.index.equals(first.y_test.index)


def test_preprocessor_is_unfitted_and_fresh():
    first, second = make_preprocessor(), make_preprocessor()
    assert first is not second
    with pytest.raises(NotFittedError):
        check_is_fitted(first)
    assert not hasattr(first, "transformers_")


def feature_frame():
    return pd.DataFrame({
        NUMERIC_FEATURES[0]: [1.0, np.nan, 3.0],
        NUMERIC_FEATURES[1]: [2.0, 4.0, 6.0],
        NUMERIC_FEATURES[2]: [10.0, 20.0, 30.0],
        "Species": ["A", "A", "B"],
        "Island": ["Dream", "Dream", "Dream"],
        "Sex": ["MALE", np.nan, "FEMALE"],
    })[list(FEATURES)]


def test_unknown_category_ignored_and_unknown_sex_encoded():
    train = feature_frame()
    prep = make_preprocessor()
    transformed = prep.fit_transform(train)
    unknown = train.iloc[[0]].copy()
    unknown["Species"] = "UNSEEN"
    unknown["Sex"] = np.nan
    result = prep.transform(unknown)
    assert result.shape[1] == transformed.shape[1]
    assert np.isfinite(result).all()
    names = prep.get_feature_names_out().tolist()
    assert "categorical__Sex_Unknown" in names
    assert result[0, names.index("categorical__Sex_Unknown")] == 1
    species_cols = [i for i, name in enumerate(names) if name.startswith("categorical__Species_")]
    assert result[0, species_cols].sum() == 0


def test_median_is_training_only_and_transform_does_not_refit():
    train = feature_frame()
    test = train.iloc[[0]].copy()
    test[NUMERIC_FEATURES[0]] = 10000.0
    prep = make_preprocessor()
    prep.fit(train)
    imputer = prep.named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == 2.0
    before = imputer.statistics_.copy()
    prep.transform(test)
    np.testing.assert_array_equal(imputer.statistics_, before)
    test[NUMERIC_FEATURES[0]] = np.nan
    assert prep.transform(test)[0, 0] == 2.0


def test_manifest_replays_ordered_split(csv_fixture, tmp_path):
    data = load_fixture(csv_fixture)
    split = split_dataset(data)
    manifest = build_manifest(data, split)
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    replay = split_from_manifest(data, path)
    pd.testing.assert_frame_equal(replay.X_train, split.X_train)
    pd.testing.assert_frame_equal(replay.X_test, split.X_test)
    pd.testing.assert_series_equal(replay.y_train, split.y_train)
    pd.testing.assert_series_equal(replay.y_test, split.y_test)
    assert manifest["n_train"] + manifest["n_test"] == manifest["n_target_valid"]


def test_manifest_overlap_rejected(csv_fixture, tmp_path):
    data = load_fixture(csv_fixture)
    manifest = build_manifest(data, split_dataset(data))
    manifest["test_indices"][0] = manifest["train_indices"][0]
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="indices overlap"):
        split_from_manifest(data, path)


def test_non_numeric_feature_rejected(csv_fixture):
    rows = [["invalid", 18, 190, "A", "Dream", "MALE", 3500, "ID1"]]
    with pytest.raises(ValueError):
        load_fixture(csv_fixture, rows=rows)


def test_all_missing_training_numeric_column_preserves_feature():
    train = feature_frame()
    train[NUMERIC_FEATURES[0]] = np.nan
    result = make_preprocessor().fit_transform(train)
    assert np.isfinite(result).all()
    assert np.all(result[:, 0] == 0)
