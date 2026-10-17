"""Integrasi dan perbandingan Naive Bayes, SVM, dan Decision Tree UTS."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .data_pipeline import PipelineData, prepare_pipeline
from .evaluation import evaluate_predictions, save_confusion_matrix
from .naive_bayes import predict_naive_bayes, train_naive_bayes
from .svm_model import predict_svm, train_svm

MODEL_ORDER = ["Naive Bayes", "SVM", "Decision Tree"]


def run_experiment(data_path=None) -> dict:
    pipeline = prepare_pipeline() if data_path is None else prepare_pipeline(data_path)

    models = {
        "Naive Bayes": train_naive_bayes(pipeline.X_train, pipeline.y_train),
        "SVM": train_svm(pipeline.X_train, pipeline.y_train),
        "Decision Tree": DecisionTreeClassifier(random_state=42).fit(
            pipeline.X_train, pipeline.y_train
        ),
    }

    metrics = {}
    predictions = {}
    for name in MODEL_ORDER:
        y_pred = models[name].predict(pipeline.X_test)
        predictions[name] = y_pred
        metrics[name] = evaluate_predictions(pipeline.y_test, y_pred)

    best_model_name = max(MODEL_ORDER, key=lambda name: metrics[name]["accuracy"])
    return {
        "pipeline": pipeline,
        "models": models,
        "predictions": predictions,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model": models[best_model_name],
    }


def comparison_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        result = metrics[name]
        rows.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-Score": result["f1_score"],
            }
        )
    return pd.DataFrame(rows)


def save_results(experiment: dict, output_dir: Path | str) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = experiment["metrics"]
    pipeline: PipelineData = experiment["pipeline"]

    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = comparison_frame(metrics)
    comparison_path = output / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    pipeline_summary = {
        "source": "UTS/data/employee_attrition_preprocessed.csv",
        "total_rows": int(len(pipeline.y_train) + len(pipeline.y_test)),
        "train_rows": int(len(pipeline.y_train)),
        "test_rows": int(len(pipeline.y_test)),
        "initial_features": int(len(pipeline.feature_columns)),
        "selected_features": pipeline.selected_features,
        "random_state": 42,
        "test_size": 0.2,
    }
    pipeline_path = output / "pipeline_summary.json"
    pipeline_path.write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")

    paths = {
        "metrics": metrics_path,
        "comparison": comparison_path,
        "pipeline": pipeline_path,
    }
    slug = {
        "Naive Bayes": "naive_bayes",
        "SVM": "svm",
        "Decision Tree": "decision_tree",
    }
    for name in MODEL_ORDER:
        paths[f"cm_{slug[name]}"] = save_confusion_matrix(
            metrics[name], name, output / f"confusion_matrix_{slug[name]}.png"
        )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plotted = comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
    plotted.plot(kind="bar", ax=ax, ylim=(0, 1), width=0.78)
    ax.set_title("Perbandingan Evaluasi Model Week 08")
    ax.set_xlabel("")
    ax.set_ylabel("Skor")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    comparison_chart = output / "model_comparison.png"
    fig.savefig(comparison_chart, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["comparison_chart"] = comparison_chart
    return paths


