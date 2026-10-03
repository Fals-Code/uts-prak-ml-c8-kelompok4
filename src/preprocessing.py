import numpy as np
import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
TARGET = "Attrition"


def get_categorical_columns(df):
    cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            cols.append(col)
    return cols


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError("Kolom target Attrition tidak ditemukan")

    return df


def handle_missing_value(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = get_categorical_columns(df)

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def handle_outlier_iqr(df):
    df = df.copy()
    hasil = []

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        # kolom dengan sedikit nilai unik dianggap skor/kategori ordinal
        if df[col].nunique() <= 10:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier = ((df[col] < lower) | (df[col] > upper)).sum()

        hasil.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outlier),
        })

        if outlier > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)

    hasil = pd.DataFrame(hasil)
    if not hasil.empty:
        hasil = hasil.sort_values(
            ["outlier_count", "column"], ascending=[False, True]
        ).reset_index(drop=True)

    return df, hasil


def preprocess_data(path=DATA_PATH):
    df = load_data(path)

    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    # missing value
    df = handle_missing_value(df)

    # duplicate
    df = df.drop_duplicates().reset_index(drop=True)

    # hapus identifier dan kolom yang nilainya konstan
    drop_cols = ["EmployeeNumber"]
    constant_cols = [
        col for col in df.columns
        if col != TARGET and df[col].nunique() <= 1
    ]
    drop_cols = list(dict.fromkeys(drop_cols + constant_cols))
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols)

    # outlier dengan metode IQR
    df, outlier_summary = handle_outlier_iqr(df)

    # target: No = 0, Yes = 1
    y = df[TARGET].map({"No": 0, "Yes": 1})

    # encoding fitur kategorikal
    X = df.drop(columns=[TARGET])
    categorical_cols = get_categorical_columns(X)
    X = pd.get_dummies(X, columns=categorical_cols, dtype=int)

    info = {
        "jumlah_data": len(df),
        "fitur_sebelum_encoding": df.shape[1] - 1,
        "fitur_setelah_encoding": X.shape[1],
        "missing": missing_count,
        "duplicate": duplicate_count,
        "dropped_columns": drop_cols,
        "categorical_columns": categorical_cols,
        "outlier_summary": outlier_summary,
    }

    return X, y, info


def print_result(X, y, info):
    print("=" * 70)
    print("HASIL PREPROCESSING - KELOMPOK 4 C8")
    print("=" * 70)
    print(f"Jumlah data setelah preprocessing : {info['jumlah_data']}")
    print(f"Jumlah fitur sebelum encoding     : {info['fitur_sebelum_encoding']}")
    print(f"Jumlah fitur setelah encoding     : {info['fitur_setelah_encoding']}")
    print(f"Jumlah duplikasi sebelum handling : {info['duplicate']}")
    print(f"Total missing value sebelum handling: {info['missing']}")

    print("\nKolom yang dihapus (ID/konstan):")
    print(info["dropped_columns"])

    print("\nKolom kategorikal yang di-encoding:")
    print(info["categorical_columns"])

    print("\nRingkasan outlier IQR:")
    print(info["outlier_summary"].to_string(index=False))

    print("\nDistribusi target Attrition (0=No, 1=Yes):")
    print(y.value_counts().sort_index().to_string())


if __name__ == "__main__":
    X, y, info = preprocess_data()
    print_result(X, y, info)
