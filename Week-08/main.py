"""Menjalankan eksperimen klasifikasi Week 08 dari awal sampai akhir."""

from pathlib import Path

import joblib

from src.comparison import MODEL_ORDER, comparison_frame, run_experiment, save_results

WEEK_DIR = Path(__file__).resolve().parent


def main():
    experiment = run_experiment()
    pipeline = experiment["pipeline"]
    metrics = experiment["metrics"]
    output_paths = save_results(experiment, WEEK_DIR / "outputs")

    split_dir = WEEK_DIR / "outputs" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    pipeline.X_train.to_csv(split_dir / "X_train.csv", index_label="index")
    pipeline.X_test.to_csv(split_dir / "X_test.csv", index_label="index")
    pipeline.y_train.to_frame(name="Attrition").to_csv(
        split_dir / "y_train.csv", index_label="index"
    )
    pipeline.y_test.to_frame(name="Attrition").to_csv(
        split_dir / "y_test.csv", index_label="index"
    )

    model_dir = WEEK_DIR / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "naive_bayes.joblib"
    joblib.dump(experiment["models"]["Naive Bayes"], model_path)

    output_paths.update(
        {
            "X_train": split_dir / "X_train.csv",
            "X_test": split_dir / "X_test.csv",
            "y_train": split_dir / "y_train.csv",
            "y_test": split_dir / "y_test.csv",
            "model_naive_bayes": model_path,
        }
    )

    print("=" * 72)
    print("HASIL KLASIFIKASI WEEK 08 - KELOMPOK 4 C8")
    print("=" * 72)
    print("Sumber data       : UTS/data/employee_attrition_preprocessed.csv")
    print(f"Split             : {len(pipeline.y_train)} training / {len(pipeline.y_test)} testing")
    print("Random state      : 42")
    print(f"Fitur             : {len(pipeline.feature_columns)} awal / {len(pipeline.selected_features)} terpilih (MI)")

    for name in MODEL_ORDER:
        result = metrics[name]
        print(f"\n{name}")
        print(f"Confusion Matrix  : {result['confusion_matrix']}")
        print(f"Accuracy          : {result['accuracy']:.4f}")
        print(f"Precision         : {result['precision']:.4f}")
        print(f"Recall            : {result['recall']:.4f}")
        print(f"F1-Score          : {result['f1_score']:.4f}")

    print("\nTabel perbandingan")
    print(comparison_frame(metrics).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nModel accuracy tertinggi: {experiment['best_model_name']}")
    print("\nOutput tersimpan:")
    for label, path in output_paths.items():
        print(f"- {label}: {path.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()


