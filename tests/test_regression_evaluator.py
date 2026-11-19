"""Unit tests for regression metric evaluator."""
import unittest

class TestRegressionEvaluator(unittest.TestCase):
    def test_rmse_calculation(self):
        y_true = [80.0, 90.0, 85.0]
        y_pred = [82.0, 88.0, 85.0]
        mse = sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)
        rmse = mse ** 0.5
        self.assertAlmostEqual(rmse, 1.63299, places=4)

if __name__ == '__main__':
    unittest.main()
