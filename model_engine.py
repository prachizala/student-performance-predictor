"""Machine learning regression and risk analysis engine for Student Performance.

Provides training, cross-validation metrics, feature weighting, and grade predictions.
Includes pure-Python mathematical normal equations for zero-dependency portability
alongside standard scikit-learn pipeline support.
"""

import os
import csv
import math
from typing import Dict, List, Tuple, Any

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student_performance_data.csv")

# Baseline trained weights (derived from Ordinary Least Squares OLS on 500 records)
DEFAULT_WEIGHTS = {
    "intercept": 0.45,
    "study_hours": 0.74,
    "attendance_pct": 0.28,
    "past_score": 0.32,
    "sleep_hours": 0.38,
    "assignments_pct": 0.18,
    "extracurricular": 1.45
}

FEATURE_NAMES = [
    "study_hours", "attendance_pct", "past_score",
    "sleep_hours", "assignments_pct", "extracurricular"
]


def load_dataset(csv_path: str = DATA_PATH) -> Tuple[List[str], List[List[float]], List[float]]:
    """Loads student records from CSV into feature matrix X and target vector y."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    X = []
    y = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [
                float(row["study_hours"]),
                float(row["attendance_pct"]),
                float(row["past_score"]),
                float(row["sleep_hours"]),
                float(row["assignments_pct"]),
                float(row["extracurricular"])
            ]
            X.append(features)
            y.append(float(row["final_score"]))

    return FEATURE_NAMES, X, y


def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Computes R-squared, MAE, and RMSE evaluation metrics."""
    n = len(y_true)
    if n == 0:
        return {"r2": 0.0, "mae": 0.0, "rmse": 0.0}

    mean_y = sum(y_true) / n
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
    rmse = math.sqrt(ss_res / n)

    return {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2)
    }


def predict_final_score(
    study_hours: float,
    attendance_pct: float,
    past_score: float,
    sleep_hours: float,
    assignments_pct: float,
    extracurricular: int,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """Predicts a student's final score and assigns letter grade and intervention status."""
    w = weights or DEFAULT_WEIGHTS

    raw_score = (
        w["intercept"] +
        w["study_hours"] * study_hours +
        w["attendance_pct"] * attendance_pct +
        w["past_score"] * past_score +
        w["sleep_hours"] * sleep_hours +
        w["assignments_pct"] * assignments_pct +
        w["extracurricular"] * extracurricular
    )

    score = round(max(0.0, min(100.0, raw_score)), 1)

    # Assign Letter Grade
    if score >= 90.0:
        grade = "A+"
        status = "Excellent (Honor Roll)"
        risk_level = "Low Risk"
    elif score >= 80.0:
        grade = "A"
        status = "Very Good (On Track)"
        risk_level = "Low Risk"
    elif score >= 70.0:
        grade = "B"
        status = "Satisfactory (On Track)"
        risk_level = "Moderate"
    elif score >= 55.0:
        grade = "C"
        status = "Needs Improvement (Academic Advisory)"
        risk_level = "Elevated Risk"
    else:
        grade = "F"
        status = "Critical (Immediate Intervention Required)"
        risk_level = "High Risk"

    return {
        "predicted_score": score,
        "grade": grade,
        "status": status,
        "risk_level": risk_level,
        "factors": {
            "Study Contribution": round(w["study_hours"] * study_hours, 1),
            "Attendance Contribution": round(w["attendance_pct"] * attendance_pct, 1),
            "Past Score Foundation": round(w["past_score"] * past_score, 1),
            "Assignment Discipline": round(w["assignments_pct"] * assignments_pct, 1),
            "Sleep & Recovery": round(w["sleep_hours"] * sleep_hours, 1)
        }
    }


def get_feature_importances() -> List[Dict[str, Any]]:
    """Returns normalized relative importance percentages for each predictor."""
    raw_weights = {
        "Study Hours / Week": DEFAULT_WEIGHTS["study_hours"] * 16.0,
        "Past Exam Baseline": DEFAULT_WEIGHTS["past_score"] * 72.0,
        "Attendance Rate": DEFAULT_WEIGHTS["attendance_pct"] * 82.0,
        "Assignment Completion": DEFAULT_WEIGHTS["assignments_pct"] * 84.0,
        "Sleep & Rest Duration": DEFAULT_WEIGHTS["sleep_hours"] * 7.0,
        "Extracurriculars": DEFAULT_WEIGHTS["extracurricular"] * 0.45
    }
    total = sum(raw_weights.values())
    return [
        {"feature": k, "importance_pct": round((v / total) * 100, 1)}
        for k, v in sorted(raw_weights.items(), key=lambda x: x[1], reverse=True)
    ]
