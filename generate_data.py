"""Curated academic dataset generator for Student Performance Predictor.

Generates 500 realistic student records reflecting study habits,
attendance rates, sleep duration, and assignment completion using Python stdlib.
"""

import os
import random
import csv
import math

random.seed(42)

def generate_student_dataset(num_records: int = 500):
    rows = []
    headers = [
        "student_id", "study_hours", "attendance_pct", "past_score",
        "sleep_hours", "assignments_pct", "extracurricular", "final_score"
    ]

    for i in range(num_records):
        sid = f"STU-{1000 + i}"
        study_hours = round(max(2.0, min(38.0, random.gauss(16.0, 6.0))), 1)
        attendance_pct = round(max(45.0, min(100.0, random.gauss(82.0, 11.0))), 1)
        past_score = round(max(35.0, min(99.0, random.gauss(72.0, 12.0))), 1)
        sleep_hours = round(max(4.5, min(9.5, random.gauss(7.0, 1.1))), 1)
        assignments_pct = round(max(40.0, min(100.0, random.gauss(84.0, 12.0))), 1)
        extracurricular = 1 if random.random() < 0.45 else 0

        # Academic weighted formula + realistic variance
        noise = random.gauss(0.0, 3.5)
        raw_final = (
            0.32 * past_score +
            0.28 * attendance_pct +
            0.75 * study_hours +
            0.18 * assignments_pct +
            0.40 * sleep_hours +
            1.50 * extracurricular +
            noise
        )
        final_score = round(max(20.0, min(100.0, raw_final)), 1)

        rows.append([
            sid, study_hours, attendance_pct, past_score,
            sleep_hours, assignments_pct, extracurricular, final_score
        ])

    return headers, rows

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "student_performance_data.csv")

    headers, rows = generate_student_dataset(500)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated {len(rows)} student records at {out_path}")
