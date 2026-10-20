"""Simpan model SVM dan Decision Tree serta heatmap Confusion Matrix.

Jalankan dari folder Week-08:  python save_svm_dt_artifacts.py
Memakai split yang disimpan Falah (outputs/split) tanpa mengubahnya.
"""
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.decision_tree import predict_decision_tree, train_decision_tree
from src.svm_model import predict_svm, train_svm

SPLIT_DIR = BASE / "outputs" / "split"
MODELS_DIR = BASE / "outputs" / "models"
FIG_DIR = BASE / "outputs" / "figures"


def read_csv(name):
    df = pd.read_csv(SPLIT_DIR / name)
    if df.columns[0] in ["index", "Unnamed: 0"] or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


def save_cm_png(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["No", "Yes"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    X_train, X_test = read_csv("X_train.csv"), read_csv("X_test.csv")
    y_train, y_test = read_csv("y_train.csv").iloc[:, 0], read_csv("y_test.csv").iloc[:, 0]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = {
        "svm": ("SVM (RBF, C=1)", train_svm, predict_svm),
        "decision_tree": ("Decision Tree", train_decision_tree, predict_decision_tree),
    }
    for key, (label, train_fn, predict_fn) in jobs.items():
        model = train_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)
        joblib.dump(model, MODELS_DIR / f"{key}.joblib")
        save_cm_png(
            y_test,
            y_pred,
            f"Confusion Matrix {label}",
            FIG_DIR / f"confusion_matrix_{key}.png",
        )
        print(f"{label}: model dan gambar tersimpan ({int(y_pred.sum())} prediksi Yes)")


if __name__ == "__main__":
    main()
