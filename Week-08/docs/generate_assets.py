"""Membuat aset laporan dari kode dan output eksperimen yang benar-benar dijalankan."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from PIL import Image

WEEK_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = WEEK_DIR / "docs" / "assets"
OUTPUT_DIR = WEEK_DIR / "outputs"
sys.path.insert(0, str(WEEK_DIR))

from src.comparison import run_experiment
from src.data_pipeline import transform_for_prediction


def ui_preview(destination: Path):
    experiment = run_experiment()
    pipeline = experiment["pipeline"]
    model_name = experiment["best_model_name"]
    model = experiment["best_model"]
    sample = pipeline.X_test_original.iloc[[0]]
    prediction = int(model.predict(transform_for_prediction(sample, pipeline))[0])
    actual = int(pipeline.y_test.iloc[0])
    accuracy = experiment["metrics"][model_name]["accuracy"]

    fig, ax = plt.subplots(figsize=(12.5, 7), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    ax.text(0.04, 0.92, "Prediksi Employee Attrition", fontsize=24, fontweight="bold", color="#0f172a")
    ax.text(0.04, 0.86, "Week 08 � Praktikum Machine Learning � Kelompok 4 C8", fontsize=11, color="#64748b")
    cards = [
        ("Model UI", model_name),
        ("Accuracy testing", f"{accuracy:.2%}"),
        ("Split data", "80% / 20%"),
    ]
    for index, (label, value) in enumerate(cards):
        x = 0.04 + index * 0.31
        ax.add_patch(plt.Rectangle((x, 0.66), 0.27, 0.13, facecolor="white", edgecolor="#cbd5e1"))
        ax.text(x + 0.02, 0.75, label, fontsize=10, color="#64748b")
        ax.text(x + 0.02, 0.69, value, fontsize=17, fontweight="bold", color="#0f172a")
    ax.add_patch(plt.Rectangle((0.04, 0.48), 0.89, 0.11, facecolor="#eff6ff", edgecolor="#93c5fd"))
    ax.text(0.06, 0.535, "Contoh prediksi pada satu baris data testing", fontsize=12, color="#1e3a8a", va="center")
    predicted_label = "Yes" if prediction == 1 else "No"
    actual_label = "Yes" if actual == 1 else "No"
    ax.text(0.05, 0.36, f"Prediksi Attrition: {predicted_label}", fontsize=20, fontweight="bold", color="#166534")
    ax.text(0.05, 0.29, f"Nilai aktual: {actual_label}", fontsize=13, color="#334155")
    ax.text(0.05, 0.16, "Preview dibuat dari fungsi prediksi yang sama dengan app.py.", fontsize=10, color="#64748b")
    fig.savefig(
        destination,
        dpi=90,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pil_kwargs={"quality": 68, "optimize": True},
    )
    plt.close(fig)


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "confusion_matrix_naive_bayes.png",
        "confusion_matrix_svm.png",
        "confusion_matrix_decision_tree.png",
        "model_comparison.png",
    ]:
        source = Image.open(OUTPUT_DIR / name).convert("RGB")
        source.thumbnail((1000, 800))
        destination = ASSET_DIR / name.replace(".png", ".jpg")
        source.save(destination, quality=72, optimize=True)
    ui_preview(ASSET_DIR / "ui_prediction_preview.jpg")


if __name__ == "__main__":
    main()


