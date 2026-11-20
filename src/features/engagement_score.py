"""Feature engineering module computing weighted student engagement index."""


class AcademicEngagementFeature:
    """Transforms raw attendance, assignment submission, and study hours into a standardized index."""

    @staticmethod
    def compute_engagement_index(
        study_hours_weekly: float,
        attendance_percentage: float,
        assignment_completion_rate: float,
    ) -> float:
        """Computes composite academic engagement index on a normalized 0-100 scale."""
        norm_study = min(study_hours_weekly / 40.0, 1.0) * 100.0
        norm_attendance = min(max(attendance_percentage, 0.0), 100.0)
        norm_assignments = min(max(assignment_completion_rate * 100.0, 0.0), 100.0)

        composite = (0.40 * norm_study) + (0.35 * norm_attendance) + (0.25 * norm_assignments)
        return round(composite, 2)
