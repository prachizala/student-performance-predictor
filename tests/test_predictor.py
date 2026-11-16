"""Automated unit test suite for Student Performance Predictor.

Tests model predictions, mathematical bounds, grade letter allocations,
and dataset file validity using Python standard library.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_engine import (
    load_dataset,
    compute_metrics,
    predict_final_score,
    get_feature_importances,
    benchmark_models,
    DATA_PATH
)


class TestStudentPredictor(unittest.TestCase):
    def test_dataset_exists_and_valid(self):
        self.assertTrue(os.path.exists(DATA_PATH), f"Dataset missing at {DATA_PATH}")
        features, X, y = load_dataset()
        self.assertGreaterEqual(len(X), 100, "Dataset should contain at least 100 records")
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(features), 6)

    def test_prediction_output_bounds(self):
        # Test low boundary inputs
        low_res = predict_final_score(
            study_hours=2.0,
            attendance_pct=50.0,
            past_score=40.0,
            sleep_hours=5.0,
            assignments_pct=40.0,
            extracurricular=0
        )
        self.assertGreaterEqual(low_res["predicted_score"], 0.0)
        self.assertLessEqual(low_res["predicted_score"], 100.0)
        self.assertIn(low_res["grade"], ["C", "F"])

        # Test high boundary inputs
        high_res = predict_final_score(
            study_hours=35.0,
            attendance_pct=98.0,
            past_score=95.0,
            sleep_hours=8.0,
            assignments_pct=98.0,
            extracurricular=1
        )
        self.assertGreaterEqual(high_res["predicted_score"], 80.0)
        self.assertLessEqual(high_res["predicted_score"], 100.0)
        self.assertIn(high_res["grade"], ["A", "A+"])

    def test_monotonicity_of_study_hours(self):
        """More study hours should strictly increase or maintain the predicted score."""
        score_low = predict_final_score(10.0, 80.0, 70.0, 7.0, 80.0, 0)["predicted_score"]
        score_high = predict_final_score(25.0, 80.0, 70.0, 7.0, 80.0, 0)["predicted_score"]
        self.assertGreater(score_high, score_low)

    def test_metric_computations(self):
        y_true = [80.0, 85.0, 90.0, 95.0]
        y_pred = [82.0, 84.0, 89.0, 96.0]
        metrics = compute_metrics(y_true, y_pred)
        self.assertGreater(metrics["r2"], 0.90)
        self.assertLess(metrics["mae"], 3.0)
        self.assertLess(metrics["rmse"], 3.0)

    def test_feature_importances_total_100(self):
        imps = get_feature_importances()
        total_pct = sum(item["importance_pct"] for item in imps)
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

    def test_benchmark_models_comparison(self):
        benchmarks = benchmark_models()
        self.assertEqual(len(benchmarks), 3)
        model_names = [b["model"] for b in benchmarks]
        self.assertIn("OLS Linear Regression", model_names)
        self.assertIn("Random Forest Regressor", model_names)
        self.assertIn("Decision Tree Regressor", model_names)

        for b in benchmarks:
            self.assertIn("r2", b)
            self.assertIn("mae", b)
            self.assertIn("rmse", b)
            self.assertGreater(b["r2"], 0.5)
            self.assertGreater(b["mae"], 0.0)
            self.assertGreater(b["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
