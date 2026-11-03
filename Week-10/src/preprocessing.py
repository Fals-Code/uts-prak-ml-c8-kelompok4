"""Dataset adapters; raw CSV files are never modified."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "mall": ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
    "penguins": ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"],
}
FILES = {"mall": "Mall_Customers.csv", "penguins": "Penguins.csv"}

@dataclass
class PreparedDataset:
    name: str
    X: np.ndarray
    features: pd.DataFrame
    source_rows: np.ndarray
    external_labels: np.ndarray | None
    scaler: StandardScaler
    metadata: dict

def prepare_dataset(dataset_name):
    """Return scaled numeric features and aligned source row IDs (zero based)."""
    if dataset_name not in FEATURES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    path = ROOT / "data/raw" / FILES[dataset_name]
    raw = pd.read_csv(path)
    columns = FEATURES[dataset_name]
    required = columns + (["Species"] if dataset_name == "penguins" else [])
    missing_columns = set(required) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    numeric = raw[columns].apply(pd.to_numeric, errors="coerce")
    invalid = raw[columns].notna() & numeric.isna()
    if invalid.any().any():
        raise ValueError(f"Invalid numeric values: {invalid.sum().to_dict()}")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite feature values")
    missing = numeric.isna().sum().to_dict()
    if dataset_name == "mall" and numeric.isna().any().any():
        raise ValueError("Mall missing values require explicit policy; refusing silent imputation")
    keep = numeric.notna().all(axis=1)
    features = numeric.loc[keep].copy()
    if len(features) < 3:
        raise ValueError("Too few complete samples")
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    labels = raw.loc[keep, "Species"].to_numpy() if dataset_name == "penguins" else None
    metadata = {"dataset": dataset_name, "rows_before": len(raw), "rows_after": len(features),
                "rows_removed": int((~keep).sum()), "removed_source_rows": raw.index[~keep].tolist(),
                "removal_reason": "missing at least one selected feature" if (~keep).any() else None,
                "features": columns, "scaling": "StandardScaler", "missing_before": missing,
                "missing_policy": "complete-case" if dataset_name == "penguins" else "reject missing",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
    return PreparedDataset(dataset_name, X, features, raw.index[keep].to_numpy(), labels, scaler, metadata)
