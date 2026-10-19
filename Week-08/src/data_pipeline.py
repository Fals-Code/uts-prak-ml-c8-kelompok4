"""Shared data pipeline untuk seluruh model Week 08.

Pipeline mengikuti UTS: data hasil preprocessing UTS, split terstratifikasi
80:20, Min-Max scaling, dan seleksi 20 fitur dengan Mutual Information.
Seluruh transformer hanya di-fit pada data training untuk mencegah leakage.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

TARGET = "Attrition"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MI_TOP_K = 20

WEEK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WEEK_DIR.parent
DEFAULT_DATA_PATH = REPO_ROOT / "UTS" / "data" / "employee_attrition_preprocessed.csv"


@dataclass
class PipelineData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_test_original: pd.DataFrame
    scaler: MinMaxScaler
    feature_columns: list[str]
    selected_features: list[str]


def load_preprocessed(path: Path | str = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Membaca output preprocessing UTS tanpa mengubah arsip UTS."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Data preprocessing UTS tidak ditemukan: {source}. "
            "Jalankan preprocessing UTS atau letakkan output pada lokasi tersebut."
        )

    df = pd.read_csv(source)
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan pada {source}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Output preprocessing UTS masih memiliki missing value")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target Attrition harus sudah dikodekan sebagai 0 dan 1")

    return X, y


def prepare_pipeline(path: Path | str = DEFAULT_DATA_PATH) -> PipelineData:
    """Menyiapkan satu split dan transformasi yang dipakai semua model."""
    source = Path(path)
    X, y = load_preprocessed(source)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X.columns,
        index=X_train_raw.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X.columns,
        index=X_test_raw.index,
    )

    binary_mask = [
        set(X_train_scaled[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
        for col in X_train_scaled.columns
    ]
    scores = mutual_info_classif(
        X_train_scaled,
        y_train,
        discrete_features=binary_mask,
        random_state=RANDOM_STATE,
    )
    mi_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    selected_features = mi_scores.head(MI_TOP_K).index.tolist()

    return PipelineData(
        X_train=X_train_scaled[selected_features],
        X_test=X_test_scaled[selected_features],
        y_train=y_train,
        y_test=y_test,
        X_test_original=X_test_raw,
        scaler=scaler,
        feature_columns=X.columns.tolist(),
        selected_features=selected_features,
    )


def transform_for_prediction(data: pd.DataFrame, pipeline: PipelineData) -> pd.DataFrame:
    """Menerapkan scaler dan fitur terpilih yang sama pada input prediksi."""
    missing = [col for col in pipeline.feature_columns if col not in data.columns]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Input kurang {len(missing)} kolom. Contoh: {preview}")

    ordered = data[pipeline.feature_columns].copy()
    scaled = pd.DataFrame(
        pipeline.scaler.transform(ordered),
        columns=pipeline.feature_columns,
        index=ordered.index,
    )
    return scaled[pipeline.selected_features]


