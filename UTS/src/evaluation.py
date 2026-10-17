from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(y_test, y_pred):
    """Menghitung metrik evaluasi model klasifikasi."""
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def print_evaluation_result(result, model_name):
    """Menampilkan hasil evaluasi model."""
    print(f"\n=== Evaluasi {model_name} ===")
    print("\nConfusion Matrix:")
    print(result["confusion_matrix"])
    print(f"\nAccuracy : {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall   : {result['recall']:.4f}")
    print(f"F1-Score : {result['f1_score']:.4f}")
