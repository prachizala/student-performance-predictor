"""Sensitivity and marginal impact analyzer for predictive grade factors."""


class GradeSensitivityAnalyzer:
    """Computes marginal grade improvement per unit increase of each predictor variable."""

    @staticmethod
    def calculate_marginal_gains(weights: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
        """Calculates expected grade point increase for specific student lifestyle interventions."""
        impacts = {}
        for feature, delta in deltas.items():
            coef = weights.get(feature, 0.0)
            impacts[feature] = round(coef * delta, 2)
        impacts["total_expected_gain"] = round(sum(impacts.values()), 2)
        return impacts
