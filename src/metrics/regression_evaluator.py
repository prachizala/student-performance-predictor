"""Regression performance metric evaluator computing MAE, RMSE, and coefficient of determination (R2)."""

import math


class RegressionMetricEvaluator:
    """Computes academic regression prediction accuracy benchmarks."""

    @staticmethod
    def compute_metrics(actual: list[float], predicted: list[float]) -> dict:
        """Computes MAE, RMSE, and R-squared for continuous grade predictions."""
        n = len(actual)
        if n == 0 or len(predicted) != n:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

        mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
        mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
        rmse = math.sqrt(mse)

        mean_actual = sum(actual) / n
        ss_tot = sum((a - mean_actual) ** 2 for a in actual)
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        r2 = (1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
        }
