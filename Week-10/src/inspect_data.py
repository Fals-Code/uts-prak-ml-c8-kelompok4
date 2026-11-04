"""Pemeriksaan data mentah untuk screenshot Gambar 3.1-3.3 (tidak mengubah CSV).

Jalankan dari folder Week-10/:  python -m src.inspect_data
"""
import pandas as pd
from .preprocessing import FEATURES, FILES, ROOT, prepare_dataset

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

for name, title in [("mall", "MALL CUSTOMER"), ("penguins", "PALMER PENGUINS")]:
    raw = pd.read_csv(ROOT / "data/raw" / FILES[name])
    print(f"\n===== {title}: pemeriksaan data mentah =====")
    print("Bentuk (baris, kolom):", raw.shape)
    print("\nTipe data:\n", raw.dtypes)
    print("\nMissing value per kolom:\n", raw.isna().sum()[lambda s: s > 0] if raw.isna().any().any() else "tidak ada")
    print("\nBaris duplikat penuh:", int(raw.duplicated().sum()))
    print("\nStatistik fitur terpilih:\n", raw[FEATURES[name]].describe().round(3))
    if name == "penguins":
        print("\nJumlah sampel per Species (344 baris mentah):\n", raw["Species"].value_counts())

print("\n===== PREPROCESSING dan VALIDASI MATRIKS FITUR =====")
for name in ["mall", "penguins"]:
    d = prepare_dataset(name)
    print(f"\n[{name}] X shape={d.X.shape} | baris dibuang={d.metadata['rows_removed']} "
          f"{d.metadata['removed_source_rows']} | scaler={d.metadata['scaling']}")
    print("  fitur:", d.metadata["features"])
    print("  rata-rata kolom X (harus ~0):", d.X.mean(axis=0).round(6))
    print("  simpangan baku kolom X (harus ~1):", d.X.std(axis=0).round(6))
    print("  Species/CustomerID ada di fitur?:",
          any(c in ("Species", "CustomerID") for c in d.features.columns))
    print("  SHA-256:", d.metadata["sha256"])