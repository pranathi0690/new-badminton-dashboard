"""
scoring_engine.py
Member 2 — Performance Analytics Lead

Reads features.csv (per-frame engineered data) and computes the
Match Overview metrics + Movement Efficiency scores + BPS score.

Every function here takes a pandas DataFrame (features) and returns
plain Python numbers/dicts — no Streamlit code lives in this file.
This keeps it testable on its own and reusable by performance_section.py.
"""

import numpy as np
import pandas as pd

# fps used by the original pipeline (from analytics.json: 60.002)
# kept as a constant here so distance/speed-over-time math is consistent
FPS = 60.002


# ---------------------------------------------------------------------------
# MATCH OVERVIEW METRICS
# ---------------------------------------------------------------------------

def compute_match_overview(features: pd.DataFrame) -> dict:
    """
    Returns the raw Match Overview numbers:
    total_distance, avg_speed, peak_speed, recovery_time_total, recovery_time_pct
    Note: 'speed' and 'distance' in features.csv are in NORMALIZED (0-1) units
    per frame, not real-world meters/pixels. We report them as-is (relative
    units) and flag this clearly in the dashboard rather than inventing a
    fake real-world conversion.
    """
    total_distance = float(features["distance"].sum())
    avg_speed = float(features["speed"].mean())
    peak_speed = float(features["speed"].max())

    total_frames = len(features)
    frames_moving = int((features["recovery_time"] > 0).sum())
    recovery_time_pct = round(100 * frames_moving / total_frames, 2) if total_frames else 0.0
    longest_movement_streak_frames = int(features["recovery_time"].max())
    longest_movement_streak_seconds = round(longest_movement_streak_frames / FPS, 2)

    return {
        "total_distance": round(total_distance, 4),
        "avg_speed": round(avg_speed, 6),
        "peak_speed": round(peak_speed, 6),
        "frames_moving_pct": recovery_time_pct,
        "longest_movement_streak_seconds": longest_movement_streak_seconds,
        "units_note": "distance/speed are in normalized (0-1) coordinate units per frame, not real-world meters",
    }


# ---------------------------------------------------------------------------
# MOVEMENT EFFICIENCY SCORES (all returned 0-100 for easy display/comparison)
# ---------------------------------------------------------------------------

def _normalize_0_100(value, low, high):
    """Clamp + scale a raw value into a 0-100 score given an expected [low, high] range."""
    if high <= low:
        return 50.0
    pct = (value - low) / (high - low)
    pct = max(0.0, min(1.0, pct))
    return round(pct * 100, 1)


def compute_agility_score(features: pd.DataFrame) -> float:
    """
    Agility = how often and how sharply the player changes direction.
    Built from direction_change (already computed in features.csv) — higher
    change rate = more agile footwork, within a sane range.
    """
    change_rate = features["direction_change"].mean()
    return _normalize_0_100(change_rate, low=0.0, high=0.40)


def compute_recovery_score(features: pd.DataFrame) -> float:
    """
    Recovery = how quickly the player returns toward their baseline/ready
    position (recovery_distance) after moving. Lower average recovery_distance
    relative to its own max = better recovery discipline.
    """
    rd = features["recovery_distance"]
    if rd.max() == 0:
        return 100.0
    avg_recovery_ratio = rd.mean() / rd.max()
    score = (1 - avg_recovery_ratio) * 100
    return round(max(0.0, min(100.0, score)), 1)


def compute_explosiveness_score(features: pd.DataFrame) -> float:
    """
    Explosiveness = magnitude of acceleration bursts (sudden speed changes).
    Uses the 95th percentile of absolute acceleration to capture peak bursts
    without letting one single outlier frame dominate the score.
    """
    abs_accel = features["acceleration"].abs()
    burst_value = abs_accel.quantile(0.95)
    max_observed = abs_accel.max()
    return _normalize_0_100(burst_value, low=0.0, high=max_observed if max_observed > 0 else 1.0)


def compute_consistency_score(features: pd.DataFrame) -> float:
    """
    Consistency = inverse of speed variability (rolling_speed_std).
    Lower variability in movement speed = more consistent footwork rhythm.
    """
    avg_std = features["rolling_speed_std"].mean()
    max_std = features["rolling_speed_std"].max()
    if max_std == 0:
        return 100.0
    ratio = avg_std / max_std
    score = (1 - ratio) * 100
    return round(max(0.0, min(100.0, score)), 1)


def compute_path_efficiency_score(features: pd.DataFrame) -> float:
    """
    path_efficiency is already a 0-1 ratio per frame in features.csv
    (straight-line distance / cumulative distance traveled). We report the
    mean across all frames, scaled to 0-100.
    """
    mean_eff = features["path_efficiency"].mean()
    return round(mean_eff * 100, 1)


def compute_efficiency_scores(features: pd.DataFrame) -> dict:
    """Bundles all 5 Movement Efficiency scores into one dict."""
    return {
        "agility_score": compute_agility_score(features),
        "recovery_score": compute_recovery_score(features),
        "explosiveness_score": compute_explosiveness_score(features),
        "consistency_score": compute_consistency_score(features),
        "path_efficiency": compute_path_efficiency_score(features),
    }


# ---------------------------------------------------------------------------
# BPS (Badminton Performance Score) — overall + breakdown
# ---------------------------------------------------------------------------

BPS_WEIGHTS = {
    "agility_score": 0.25,
    "recovery_score": 0.25,
    "explosiveness_score": 0.20,
    "consistency_score": 0.15,
    "path_efficiency": 0.15,
}


def compute_bps(efficiency_scores: dict) -> dict:
    """
    BPS = weighted average of the 5 efficiency scores, each already 0-100.
    Returns both the overall score and the per-component contribution,
    so the dashboard can show a breakdown, not just one number.
    """
    breakdown = {}
    overall = 0.0
    for key, weight in BPS_WEIGHTS.items():
        component_value = efficiency_scores[key]
        contribution = round(component_value * weight, 2)
        breakdown[key] = {
            "score": component_value,
            "weight": weight,
            "contribution": contribution,
        }
        overall += contribution

    return {
        "bps_score": round(overall, 1),
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# TOP-LEVEL ENTRY POINT
# ---------------------------------------------------------------------------

def run_full_performance_analysis(features: pd.DataFrame) -> dict:
    """
    The single function performance_section.py calls.
    Returns everything Member 2 owns EXCEPT the grade, which lives in
    grade_engine.py and gets merged in by performance_section.py.
    """
    match_overview = compute_match_overview(features)
    efficiency = compute_efficiency_scores(features)
    bps = compute_bps(efficiency)

    return {
        "match_overview": match_overview,
        "efficiency_scores": efficiency,
        "bps_score": bps["bps_score"],
        "bps_breakdown": bps["breakdown"],
    }