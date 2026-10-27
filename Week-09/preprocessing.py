"""Pemeriksaan, EDA, dan persiapan dua dataset kategorikal Week 09.

Penanggung jawab: Ahmad Mathlaul Falah (434241123). Tidak melatih model.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MALL_COLUMNS = ["CustomerID", "Gender", "Age", "Annual Income (k$)",
                "Spending Score (1-100)"]
TELCO_CATEGORIES = {
    "gender": ["Female", "Male"], "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"], "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: ["Yes", "No", "No internet service"] for c in
       ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"]},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
    "Churn": ["Yes", "No"],
}
TELCO_SELECTED = ["SeniorCitizen", "Partner", "Dependents", "MultipleLines",
                  "InternetService", "OnlineSecurity", "TechSupport", "Contract",
                  "PaperlessBilling", "PaymentMethod"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_and_check(frame, columns, identity, domains, ranges):
    if set(frame.columns) != set(columns) or frame.empty:
        raise ValueError(f"Skema atau jumlah baris tidak sesuai untuk {identity}.")
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object", "str", "string"]):
        cleaned[column] = cleaned[column].astype("string").str.strip().replace("", pd.NA)
    if cleaned[identity].isna().any() or cleaned[identity].duplicated().any():
        raise ValueError(f"Identitas {identity} kosong atau tidak unik.")
    for column, allowed in domains.items():
        if not cleaned[column].isin(allowed).all():
            raise ValueError(f"Kategori kosong/tidak dikenal pada {column}.")
    for column, (lower, upper) in ranges.items():
        values = pd.to_numeric(cleaned[column], errors="raise")
        if not np.isfinite(values).all() or not values.between(lower, upper).all():
            raise ValueError(f"Nilai {column} kosong atau di luar rentang pemeriksaan.")
        cleaned[column] = values
    return cleaned


def quantile_categories(values):
    """Tertil deskriptif: nilai sama tetap bersama, batas kosong digabung."""
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.empty or not np.isfinite(numeric).all():
        raise ValueError("Diskretisasi memerlukan angka terhingga tanpa missing value.")
    edges = np.unique(numeric.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy())
    if len(edges) == 1:
        result = pd.Series("tunggal", index=values.index, dtype="string")
        labels = ["tunggal"]
    else:
        # Pada sampel sangat kecil, interpolasi kuantil dapat membuat interval kosong.
        while True:
            cut = pd.cut(numeric, edges, include_lowest=True, right=True)
            counts = cut.value_counts(sort=False)
            empty = np.flatnonzero(counts.to_numpy() == 0)
            if not len(empty):
                break
            boundary = min(int(empty[0]) + 1, len(edges) - 2)
            edges = np.delete(edges, boundary)
        labels = {1: ["tunggal"], 2: ["rendah", "tinggi"],
                  3: ["rendah", "sedang", "tinggi"]}[len(edges) - 1]
        result = pd.cut(numeric, edges, labels=labels, include_lowest=True,
                        right=True).astype("string")
    metadata = {
        "method": "tertil; quantile linear; batas sama/interval kosong digabung",
        "edges": edges.tolist(), "labels": labels, "right": True,
        "include_lowest": True,
        "counts": {label: int(result.eq(label).sum()) for label in labels},
        "constant": len(edges) == 1,
    }
    return result, metadata


def prepare_mall(raw):
    cleaned = clean_and_check(
        raw, MALL_COLUMNS, "CustomerID", {"Gender": ["Female", "Male"]},
        {"Age": (0, 120), "Annual Income (k$)": (0, np.inf),
         "Spending Score (1-100)": (1, 100)})
    features = cleaned[["Gender"]].copy()
    bins = {}
    for original, target in [("Age", "AgeCategory"),
                             ("Annual Income (k$)", "IncomeCategory"),
                             ("Spending Score (1-100)", "SpendingCategory")]:
        features[target], bins[target] = quantile_categories(cleaned[original])
        bins[target]["source_column"] = original
    reference = raw[["CustomerID"]].copy()
    reference.insert(0, "source_row", np.arange(len(raw)))
    return features.astype("string"), reference, bins


def prepare_telco(raw):
    columns = ["customerID", *TELCO_CATEGORIES, "tenure", "MonthlyCharges", "TotalCharges"]
    cleaned = clean_and_check(raw, columns, "customerID", TELCO_CATEGORIES,
                              {"tenure": (0, 1200), "MonthlyCharges": (0, np.inf)})
    if not cleaned.tenure.eq(cleaned.tenure.round()).all():
        raise ValueError("tenure harus berupa jumlah bulan bulat.")
    phone_mismatch = cleaned.PhoneService.eq("No") != cleaned.MultipleLines.eq("No phone service")
    if phone_mismatch.any():
        raise ValueError("PhoneService tidak konsisten dengan MultipleLines.")
    for column in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies"]:
        if (cleaned.InternetService.eq("No") != cleaned[column].eq("No internet service")).any():
            raise ValueError(f"InternetService tidak konsisten dengan {column}.")
    features = cleaned[TELCO_SELECTED].copy()
    features["SeniorCitizen"] = cleaned.SeniorCitizen.map({0: "No", 1: "Yes"})
    bins = {}
    for original, target in [("tenure", "TenureCategory"),
                             ("MonthlyCharges", "MonthlyChargesCategory")]:
        features[target], bins[target] = quantile_categories(cleaned[original])
        bins[target]["source_column"] = original
    reference = raw[["customerID", "Churn"]].copy()
    reference.insert(0, "source_row", np.arange(len(raw)))
    return features.astype("string"), reference, bins


def inspect_data(frame, identity):
    buffer = io.StringIO()
    frame.info(buf=buffer)
    numeric = frame.select_dtypes(include="number").drop(columns=[identity], errors="ignore").copy()
    anomalies = {}
    if "TotalCharges" in frame:
        numeric["TotalCharges"] = pd.to_numeric(frame.TotalCharges.str.strip(), errors="coerce")
        anomalies["TotalCharges_blank"] = int(frame.TotalCharges.str.strip().eq("").sum())
        anomalies["TotalCharges_nonblank_invalid"] = int(
            (numeric.TotalCharges.isna() & frame.TotalCharges.str.strip().ne("")).sum())
        anomalies["TotalCharges_blank_with_zero_tenure"] = int(
            (numeric.TotalCharges.isna() & frame.tenure.eq(0)).sum())
    iqr_flags = {}
    for column in numeric:
        # Kode status biner bukan besaran kontinu untuk pemeriksaan outlier IQR.
        if column == "SeniorCitizen":
            continue
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        spread = q3 - q1
        iqr_flags[column] = int(((numeric[column] < q1 - 1.5 * spread) |
                                 (numeric[column] > q3 + 1.5 * spread)).sum())
    summary = {
        "shape": list(frame.shape), "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_pandas": frame.isna().sum().astype(int).to_dict(),
        "blank_strings": {c: int(frame[c].astype("string").str.strip().eq("").sum())
                          for c in frame.select_dtypes(exclude="number")},
        "duplicates_full": int(frame.duplicated().sum()),
        "duplicates_without_id": int(frame.drop(columns=identity).duplicated().sum()),
        "unique_ids": int(frame[identity].nunique()), "anomalies": anomalies,
        "iqr_flags_not_deleted": iqr_flags,
        "numeric_summary": json.loads(numeric.describe().to_json()),
    }
    text = [f"SHAPE: {frame.shape}", "HEAD (5):", frame.head().to_string(index=False),
            "INFO:", buffer.getvalue(), "DESCRIBE (raw, semua atribut):",
            frame.describe(include="all").to_string(),
            "DESCRIBE NUMERIK (TotalCharges dikonversi hanya untuk pemeriksaan):",
            numeric.describe().to_string(), "MISSING:", frame.isna().sum().to_string(),
            "RINGKASAN KUALITAS:", json.dumps(summary, ensure_ascii=False, indent=2),
            "DISTRIBUSI NILAI (ID dikecualikan; numerik: 10 nilai terbanyak):"]
    for column in frame.columns.drop(identity):
        counts = frame[column].value_counts(dropna=False)
        text.extend([column, (counts.head(10) if counts.size > 30 else counts).to_string()])
    return "\n\n".join(text) + "\n", summary


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Week-09 preprocessing"})
    plt.close(figure)


def bar_chart(axis, values, title):
    counts = values.value_counts().sort_index()
    counts.plot.barh(ax=axis, color="#287b8e")
    axis.set_title(title)
    axis.set_xlabel("Jumlah pelanggan")
    axis.set_ylabel("")
    axis.bar_label(axis.containers[0], padding=3, fontsize=8)
    axis.set_xlim(0, counts.max() * 1.22)


def make_eda(mall, telco, figures):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    bar_chart(axes[0, 0], mall.Gender, "Mall: distribusi gender")
    for axis, column, label in zip(axes.flat[1:],
                                  ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
                                  ["Usia (tahun)", "Pendapatan tahunan (ribu $)", "Spending score (1–100)"]):
        axis.hist(mall[column], bins=12, color="#287b8e", edgecolor="white")
        axis.set(title=f"Mall: {label}", xlabel=label, ylabel="Jumlah pelanggan")
    save_figure(figure, figures / "mall_distributions.png")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(mall["Annual Income (k$)"], mall["Spending Score (1-100)"],
                 alpha=0.7, color="#287b8e", s=22)
    axis.set(title="Mall: pendapatan dan spending score (data asli)",
             xlabel="Pendapatan tahunan (ribu $)", ylabel="Spending score (1–100)")
    save_figure(figure, figures / "mall_income_spending.png")
    # Dua kolom menjaga label layanan tetap terbaca ketika gambar masuk A4.
    figure, axes = plt.subplots(3, 2, figsize=(10, 10))
    for axis, column in zip(axes.flat, ["SeniorCitizen", "Partner", "Dependents",
                                       "InternetService", "Contract", "PaymentMethod"]):
        values = telco[column].map({0: "No", 1: "Yes"}) if column == "SeniorCitizen" else telco[column]
        bar_chart(axis, values, f"Telco: {column}")
    save_figure(figure, figures / "telco_characteristics.png")
    figure, axes = plt.subplots(3, 2, figsize=(10, 8.5))
    for index, (column, label) in enumerate([
            ("tenure", "Masa berlangganan (bulan)"), ("MonthlyCharges", "Biaya bulanan ($)"),
            ("TotalCharges", "Biaya kumulatif ($)")]):
        values = pd.to_numeric(telco[column], errors="coerce").dropna()
        axes[index, 0].hist(values, bins=18, color="#287b8e", edgecolor="white")
        axes[index, 0].set(title=f"Telco: {column}", xlabel=label, ylabel="Jumlah pelanggan")
        if column == "TotalCharges":
            axes[index, 0].set_title("Telco: TotalCharges\n7.032 nilai valid; 11 kosong")
        axes[index, 1].boxplot(values, orientation="horizontal")
        axes[index, 1].set(title=f"Boxplot: {column}", xlabel=label, yticks=[])
    save_figure(figure, figures / "telco_numeric.png")


def validate_output(raw, features, reference, identity, saved_path):
    if len(features) != len(raw) or not features.index.equals(raw.index):
        raise ValueError("Jumlah/urutan baris berubah.")
    if not reference[identity].equals(raw[identity]):
        raise ValueError("Identitas tidak selaras dengan sumber.")
    if features.isna().any().any() or features.apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("Fitur kosong belum ditangani.")
    if not set(features.columns).isdisjoint({"CustomerID", "customerID", "Churn", "source_row"}):
        raise ValueError("Identitas/label bocor ke fitur.")
    if not all(pd.api.types.is_string_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("Fitur final harus berisi label kategori string.")
    restored = pd.read_csv(saved_path, dtype="string")
    pd.testing.assert_frame_equal(features.reset_index(drop=True), restored)
    return {"shape": list(features.shape), "features": list(features.columns),
            "missing": 0, "rows_and_order_preserved": True,
            "categorical_strings": True, "csv_roundtrip_equal": True,
            "repeated_profiles_retained": int(features.duplicated().sum())}


def run(output_dir):
    raw_dir = BASE / "data/raw"
    if output_dir.resolve().is_relative_to(raw_dir.resolve()):
        raise ValueError("Output tidak boleh disimpan dalam folder data asli.")
    processed = (BASE / "data/processed" if output_dir == BASE / "outputs"
                 else output_dir / "processed")
    figures = output_dir / "figures"
    for folder in [processed, figures]:
        folder.mkdir(parents=True, exist_ok=True)
    paths = {"mall": raw_dir / "Mall_Customers.csv", "telco": raw_dir / "Telco-Customer-Churn.csv"}
    before_hashes = {name: digest(path) for name, path in paths.items()}
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    summaries, binning, validation = {}, {}, {}
    for name, prepare, identity in [("mall", prepare_mall, "CustomerID"),
                                    ("telco", prepare_telco, "customerID")]:
        raw = frames[name]
        inspection, summaries[name] = inspect_data(raw, identity)
        features, reference, binning[name] = prepare(raw)
        path = processed / f"{name}_kmodes.csv"
        features.to_csv(path, index=False, lineterminator="\n")
        reference.to_csv(processed / f"{name}_reference.csv", index=False, lineterminator="\n")
        validation[name] = validate_output(raw, features, reference, identity, path)
        (output_dir / f"{name}_inspection.txt").write_text(inspection, encoding="utf-8")
        records = [{"feature": column, "category": category, "count": int(count),
                    "percent": round(count / len(features) * 100, 4)}
                   for column in features
                   for category, count in features[column].value_counts().sort_index().items()]
        pd.DataFrame(records).to_csv(output_dir / f"{name}_category_counts.csv", index=False,
                                     lineterminator="\n")
    make_eda(frames["mall"], frames["telco"], figures)
    after_hashes = {name: digest(path) for name, path in paths.items()}
    if before_hashes != after_hashes:
        raise ValueError("Dataset asli berubah selama program berjalan.")
    metadata = {"raw_sha256": before_hashes, "inspection": summaries,
                "binning": binning, "validation": validation,
                "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                                "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    (output_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Week 09: pemeriksaan, EDA, dan preprocessing selesai")
    for name, result in validation.items():
        print(f"{name}: {summaries[name]['shape']} -> {result['shape']}; "
              f"missing final=0; baris/urutan tetap; CSV dibaca ulang identik")
    print("Telco TotalCharges kosong:", summaries["telco"]["anomalies"]["TotalCharges_blank"])
    print("4 gambar EDA disimpan. Data asli tidak berubah (SHA-256 diperiksa).")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs",
                        help="Folder alternatif; fitur disimpan pada subfolder processed.")
    arguments = parser.parse_args()
    try:
        run(arguments.output_dir.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Preprocessing gagal: {error}\n")
