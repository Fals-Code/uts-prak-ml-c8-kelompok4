"""Evaluasi konsisten untuk seluruh model klasifikasi."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(result: dict, model_name: str, output_path: Path | str) -> Path:
    cm = result["confusion_matrix"]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title=f"Confusion Matrix - {model_name}",
        xlabel="Prediksi",
        ylabel="Aktual",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    threshold = max(max(row) for row in cm) / 2
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(cm[row][col]),
                ha="center",
                va="center",
                color="white" if cm[row][col] > threshold else "black",
                fontsize=14,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


