"""
grade_engine.py
Member 2 — Performance Analytics Lead

Converts a BPS score (0-100, from scoring_engine.py) into a letter grade.
Kept separate from scoring_engine.py to match the team's planned file split.
"""

# Thresholds are a design decision — documented here so teammates/graders
# can see exactly how grades map to scores, and adjust easily if needed.
GRADE_THRESHOLDS = [
    (90, "A+"),
    (80, "A"),
    (70, "B+"),
    (60, "B"),
    (50, "C+"),
    (40, "C"),
    (0, "D"),
]


def compute_grade(bps_score: float) -> dict:
    """
    Maps a BPS score (0-100) to a letter grade using fixed thresholds.
    Returns both the letter grade and the numeric score it was based on,
    so the dashboard can show "B (67.4)" style displays.
    """
    for threshold, letter in GRADE_THRESHOLDS:
        if bps_score >= threshold:
            return {"grade": letter, "numeric_score": round(bps_score, 1)}
    return {"grade": "D", "numeric_score": round(bps_score, 1)}