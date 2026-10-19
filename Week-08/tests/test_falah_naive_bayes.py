import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

WEEK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK_DIR))

from src.data_pipeline import DEFAULT_DATA_PATH, RANDOM_STATE, TEST_SIZE, load_preprocessed, prepare_pipeline
from src.evaluation import evaluate_predictions
from src.naive_bayes import predict_naive_bayes, train_naive_bayes


class FalahNaiveBayesTest(unittest.TestCase):
    def test_training_only_transformations_and_split(self):
        pipeline = prepare_pipeline()
        source = pd.read_csv(DEFAULT_DATA_PATH)
        X, y = load_preprocessed()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        self.assertEqual(len(pipeline.X_train), 1176)
        self.assertEqual(len(pipeline.X_test), 294)
        self.assertEqual(pipeline.X_train.index.tolist(), X_train.index.tolist())
        self.assertEqual(pipeline.X_test.index.tolist(), X_test.index.tolist())
        self.assertEqual(pipeline.y_train.index.tolist(), y_train.index.tolist())
        self.assertEqual(pipeline.y_test.index.tolist(), y_test.index.tolist())
        self.assertTrue((pipeline.scaler.data_min_ == X_train.min().to_numpy()).all())
        X_train_scaled = pd.DataFrame(
            pipeline.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        binary_mask = [
            set(X_train_scaled[col].unique()).issubset({0, 1})
            for col in X_train_scaled.columns
        ]
        scores = mutual_info_classif(
            X_train_scaled,
            y_train,
            discrete_features=binary_mask,
            random_state=RANDOM_STATE,
        )
        expected_features = pd.Series(scores, index=X_train.columns).sort_values(
            ascending=False
        ).head(20).index.tolist()
        self.assertEqual(pipeline.selected_features, expected_features)
        self.assertTrue(DEFAULT_DATA_PATH.exists())
        self.assertEqual(source.shape, (1470, 52))
        self.assertFalse(source.isna().any().any())
        self.assertFalse(source.duplicated().any())
        self.assertEqual(source["Attrition"].value_counts().to_dict(), {0: 1233, 1: 237})

    def test_naive_bayes_prediction_and_metrics(self):
        pipeline = prepare_pipeline()
        model = train_naive_bayes(pipeline.X_train, pipeline.y_train)
        prediction = predict_naive_bayes(model, pipeline.X_test)
        result = evaluate_predictions(pipeline.y_test, prediction)

        self.assertEqual(len(prediction), len(pipeline.y_test))
        self.assertEqual(model.class_count_.tolist(), [986.0, 190.0])
        self.assertEqual(result["confusion_matrix"], [[160, 87], [14, 33]])
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), len(pipeline.y_test))
        self.assertAlmostEqual(result["accuracy"], 0.6564625850340136)
        self.assertAlmostEqual(result["precision"], 0.275)
        self.assertAlmostEqual(result["recall"], 0.7021276595744681)
        self.assertAlmostEqual(result["f1_score"], 0.39520958083832336)

    def test_pipeline_and_model_are_reproducible(self):
        first = prepare_pipeline()
        second = prepare_pipeline()
        self.assertEqual(first.selected_features, second.selected_features)
        pd.testing.assert_frame_equal(first.X_train, second.X_train)
        pd.testing.assert_frame_equal(first.X_test, second.X_test)

        first_model = train_naive_bayes(first.X_train, first.y_train)
        second_model = train_naive_bayes(second.X_train, second.y_train)
        self.assertEqual(
            predict_naive_bayes(first_model, first.X_test).tolist(),
            predict_naive_bayes(second_model, second.X_test).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
