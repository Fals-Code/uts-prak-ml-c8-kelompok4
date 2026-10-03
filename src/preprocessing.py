"""Dataset loading and preprocessing for UTS Praktikum ML - Kelompok 4 C8.

PIC: Ahmad Mathlaul Falah

Scope:
- load IBM HR Analytics Employee Attrition dataset
- validate target and feature types
- detect/handle missing values
- detect/remove duplicate rows
- detect/handle numeric outliers with IQR capping
- remove identifier/constant columns
- encode categorical features and target

This module intentionally stops before train/test split, Min-Max scaling,
feature selection, modeling, and evaluation because those are handled by
the other group members.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DATASET_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
TARGET_COLUMN = "Attrition"

# Column that only identifies a row/employee and should not be used as a predictor.
IDENTIFIER_COLUMNS = ("EmployeeNumber",)

# Numeric columns with only a few possible values represent ordinal/categorical
# scores rather than continuous measurements. We do not apply IQR capping to them.
LOW_CARDINALITY_THRESHOLD = 10


@dataclass
class PreprocessingResult:
    """Container returned by :func:`preprocess_dataset`."""

    X: pd.DataFrame
    y: pd.Series
    cleaned_data: pd.DataFrame
    missing_before: pd.Series
    duplicate_count_before: int
    outlier_summary: pd.DataFrame
    dropped_columns: list[str]
    categorical_columns: list[str]


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical/text columns without relying on deprecated dtype aliases.

    This helper is compatible with pandas string, object, and category dtypes and
    avoids the Pandas4Warning raised by select_dtypes(include=["object", ...])
    under pandas 3.x.
    """

    columns: list[str] = []

    for column in df.columns:
        dtype = df[column].dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        ):
            columns.append(column)

    return columns


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the CSV dataset and validate the minimum required structure."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di '{path}'. "
            "Download WA_Fn-UseC_-HR-Employee-Attrition.csv dari Kaggle "
            "lalu simpan ke folder data/."
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom target '{TARGET_COLUMN}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("Dataset kosong.")

    return df


def dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    """Return basic information useful for console output and the report."""

    categorical_columns = get_categorical_columns(df)
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "target_distribution": df[TARGET_COLUMN].value_counts(dropna=False).to_dict(),
    }


def detect_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return missing-value counts for every column."""

    return df.isna().sum().sort_values(ascending=False)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median and categorical values with mode.

    The original Kaggle dataset is typically complete, but this function keeps
    the pipeline robust and provides an explicit handling step if missing values
    are encountered.
    """

    result = df.copy()

    numeric_columns = result.select_dtypes(include=np.number).columns
    categorical_columns = get_categorical_columns(result)

    for column in numeric_columns:
        if result[column].isna().any():
            median_value = result[column].median()
            result[column] = result[column].fillna(median_value)

    for column in categorical_columns:
        if result[column].isna().any():
            mode = result[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            result[column] = result[column].fillna(fill_value)

    return result


def detect_duplicates(df: pd.DataFrame) -> int:
    """Count fully duplicated rows."""

    return int(df.duplicated().sum())


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows and reset the index."""

    return df.drop_duplicates().reset_index(drop=True)


def drop_non_informative_columns(
    df: pd.DataFrame,
    identifier_columns: Iterable[str] = IDENTIFIER_COLUMNS,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop identifier columns and constant columns.

    Constant columns contain no variation and cannot help classification.
    Identifier columns such as EmployeeNumber are unique row identifiers rather
    than meaningful predictors.
    """

    result = df.copy()
    identifier_columns = [column for column in identifier_columns if column in result.columns]
    constant_columns = [
        column
        for column in result.columns
        if column != TARGET_COLUMN and result[column].nunique(dropna=False) <= 1
    ]

    columns_to_drop = list(dict.fromkeys([*identifier_columns, *constant_columns]))
    result = result.drop(columns=columns_to_drop, errors="ignore")

    return result, columns_to_drop


def continuous_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Choose numeric columns appropriate for IQR-based outlier detection.

    Low-cardinality numeric columns are usually ordinal survey/category codes
    in this dataset, so they are excluded from IQR capping.
    """

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    return [
        column
        for column in numeric_columns
        if column != TARGET_COLUMN
        and column not in IDENTIFIER_COLUMNS
        and df[column].nunique(dropna=True) > LOW_CARDINALITY_THRESHOLD
    ]


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Detect outliers using the 1.5*IQR rule and return a summary table."""

    selected_columns = list(columns) if columns is not None else continuous_numeric_columns(df)
    rows: list[dict[str, float | int | str]] = []

    for column in selected_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            lower_bound = q1
            upper_bound = q3
            outlier_count = 0
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_count = int(outlier_mask.sum())

        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "q1",
                "q3",
                "iqr",
                "lower_bound",
                "upper_bound",
                "outlier_count",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["outlier_count", "column"], ascending=[False, True]
    ).reset_index(drop=True)


def cap_outliers_iqr(
    df: pd.DataFrame,
    outlier_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Cap outliers to their IQR lower/upper bounds instead of deleting rows."""

    result = df.copy()

    for row in outlier_summary.itertuples(index=False):
        if int(row.outlier_count) == 0:
            continue

        result[row.column] = result[row.column].clip(
            lower=float(row.lower_bound),
            upper=float(row.upper_bound),
        )

    return result


def encode_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """One-hot encode categorical predictors and map Attrition Yes/No to 1/0."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak tersedia.")

    target = df[TARGET_COLUMN].copy()
    valid_target_values = set(target.dropna().unique())

    if not valid_target_values.issubset({"Yes", "No", 1, 0}):
        raise ValueError(
            f"Nilai target tidak sesuai. Ditemukan: {sorted(map(str, valid_target_values))}"
        )

    if pd.api.types.is_numeric_dtype(target):
        y = target.astype(int)
    else:
        y = target.map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Target Attrition mengandung nilai kosong/tidak dikenali.")

    X_raw = df.drop(columns=[TARGET_COLUMN])
    categorical_columns = get_categorical_columns(X_raw)

    X = pd.get_dummies(
        X_raw,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    return X, y.rename(TARGET_COLUMN), categorical_columns


def preprocess_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> PreprocessingResult:
    """Run all preprocessing steps owned by Falah.

    Order:
    load -> missing values -> duplicates -> remove non-informative columns
    -> detect/cap outliers -> encode categorical features and target.
    """

    raw_df = load_dataset(path)

    missing_before = detect_missing_values(raw_df)
    duplicate_count_before = detect_duplicates(raw_df)

    cleaned_df = handle_missing_values(raw_df)
    cleaned_df = remove_duplicates(cleaned_df)
    cleaned_df, dropped_columns = drop_non_informative_columns(cleaned_df)

    outlier_summary = detect_outliers_iqr(cleaned_df)
    cleaned_df = cap_outliers_iqr(cleaned_df, outlier_summary)

    X, y, categorical_columns = encode_features_and_target(cleaned_df)

    return PreprocessingResult(
        X=X,
        y=y,
        cleaned_data=cleaned_df,
        missing_before=missing_before,
        duplicate_count_before=duplicate_count_before,
        outlier_summary=outlier_summary,
        dropped_columns=dropped_columns,
        categorical_columns=categorical_columns,
    )


def print_preprocessing_report(result: PreprocessingResult) -> None:
    """Print concise outputs that can be captured for the UTS report."""

    print("=" * 70)
    print("HASIL PREPROCESSING - KELOMPOK 4 C8")
    print("=" * 70)

    print(f"Jumlah data setelah preprocessing : {len(result.cleaned_data)}")
    print(f"Jumlah fitur sebelum encoding     : {result.cleaned_data.shape[1] - 1}")
    print(f"Jumlah fitur setelah encoding      : {result.X.shape[1]}")
    print(f"Jumlah duplikasi sebelum handling  : {result.duplicate_count_before}")
    print(f"Total missing value sebelum handling: {int(result.missing_before.sum())}")

    print("\nKolom yang dihapus (ID/konstan):")
    print(result.dropped_columns if result.dropped_columns else "-")

    print("\nKolom kategorikal yang di-encoding:")
    print(result.categorical_columns if result.categorical_columns else "-")

    print("\nRingkasan outlier IQR:")
    if result.outlier_summary.empty:
        print("Tidak ada kolom numerik yang memenuhi kriteria pengecekan IQR.")
    else:
        print(result.outlier_summary.to_string(index=False))

    print("\nDistribusi target Attrition (0=No, 1=Yes):")
    print(result.y.value_counts().sort_index().to_string())


def main() -> None:
    """Run preprocessing from the command line."""

    result = preprocess_dataset()
    print_preprocessing_report(result)


if __name__ == "__main__":
    main()
