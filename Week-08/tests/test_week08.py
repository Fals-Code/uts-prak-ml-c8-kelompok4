import sys
import unittest
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK_DIR))

from src.comparison import MODEL_ORDER, run_experiment
from src.data_pipeline import transform_for_prediction


class Week08IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = run_experiment()

    def test_shared_split_and_selected_features(self):
        pipeline = self.experiment["pipeline"]
        self.assertEqual(len(pipeline.y_train), 1176)
        self.assertEqual(len(pipeline.y_test), 294)
        self.assertEqual(pipeline.X_train.shape[1], 20)
        self.assertEqual(pipeline.X_test.shape[1], 20)
        self.assertEqual(set(pipeline.y_train.unique()), {0, 1})

    def test_metrics_and_confusion_matrix_are_valid(self):
        for name in MODEL_ORDER:
            result = self.experiment["metrics"][name]
            self.assertEqual(sum(sum(row) for row in result["confusion_matrix"]), 294)
            for metric in ["accuracy", "precision", "recall", "f1_score"]:
                self.assertGreaterEqual(result[metric], 0)
                self.assertLessEqual(result[metric], 1)

    def test_ui_prediction_uses_training_pipeline(self):
        pipeline = self.experiment["pipeline"]
        sample = pipeline.X_test_original.iloc[[0]]
        transformed = transform_for_prediction(sample, pipeline)
        prediction = self.experiment["best_model"].predict(transformed)
        self.assertEqual(transformed.columns.tolist(), pipeline.selected_features)
        self.assertIn(int(prediction[0]), [0, 1])

    def test_best_model_is_selected_by_accuracy(self):
        metrics = self.experiment["metrics"]
        expected = max(MODEL_ORDER, key=lambda name: metrics[name]["accuracy"])
        self.assertEqual(self.experiment["best_model_name"], expected)


if __name__ == "__main__":
    unittest.main()

