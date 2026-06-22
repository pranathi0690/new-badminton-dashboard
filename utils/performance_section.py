"""
performance_section.py
Member 2 — Performance Analytics Lead

Each metric is its own small render_* function, so app.py can show
exactly the metrics the user has clicked — individually, additively,
remembered per main-topic across navigation.

This file calls scoring_engine.py + grade_engine.py to get the numbers,
then renders them. No scoring math lives here — only display logic.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.scoring_engine import run_full_performance_analysis
from utils.grade_engine import compute_grade


def _run_analysis(features: pd.DataFrame) -> dict:
    """Shared logic: runs scoring + grading on an already-loaded DataFrame."""
    result = run_full_performance_analysis(features)
    grade_info = compute_grade(result["bps_score"])
    result["grade"] = grade_info["grade"]
    result["numeric_score"] = grade_info["numeric_score"]
    return result


@st.cache_data
def get_performance_data(features_csv_path: str) -> dict:
    """Loads features.csv from a PATH and runs the full performance analysis."""
    features = pd.read_csv(features_csv_path)
    return _run_analysis(features)


def get_performance_data_from_df(features: pd.DataFrame) -> dict:
    """Same as get_performance_data(), but takes an already-loaded DataFrame."""
    return _run_analysis(features)


def get_analytics_export(perf_data: dict) -> dict:
    """
    Returns ONLY the fields Member 3 (and the shared analytics.json) need,
    using the exact field names specified in the team plan.
    """
    eff = perf_data["efficiency_scores"]
    return {
        "bps_score": perf_data["bps_score"],
        "grade": perf_data["grade"],
        "agility_score": eff["agility_score"],
        "recovery_score": eff["recovery_score"],
        "explosiveness_score": eff["explosiveness_score"],
        "consistency_score": eff["consistency_score"],
        "path_efficiency": eff["path_efficiency"],
    }


def save_to_analytics_json(perf_data: dict, analytics_json_path: str) -> dict:
    """
    Merges Member 2's exported fields into the shared analytics.json file.
    Reads the existing file first and only updates/adds keys — never
    overwrites the whole file.
    """
    import json
    import os

    if os.path.exists(analytics_json_path):
        with open(analytics_json_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    my_fields_raw = get_analytics_export(perf_data)
    my_fields = {
        key: (float(value) if isinstance(value, (int, float)) else value)
        for key, value in my_fields_raw.items()
    }
    existing_data.update(my_fields)

    with open(analytics_json_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    return existing_data


# ---------------------------------------------------------------------------
# MATCH OVERVIEW — 6 individual metrics
# ---------------------------------------------------------------------------

def render_total_distance(perf_data: dict):
    mo = perf_data["match_overview"]
    st.metric("Total Distance (norm. units)", f"{mo['total_distance']:.3f}")
    st.caption("⚠️ Normalized (0-1) coordinate units per frame, not real-world meters.")


def render_avg_speed(perf_data: dict):
    mo = perf_data["match_overview"]
    st.metric("Avg Speed (norm. units/frame)", f"{mo['avg_speed']:.4f}")


def render_peak_speed(perf_data: dict):
    mo = perf_data["match_overview"]
    st.metric("Peak Speed (norm. units/frame)", f"{mo['peak_speed']:.4f}")


def render_recovery_time(perf_data: dict):
    mo = perf_data["match_overview"]
    col1, col2 = st.columns(2)
    col1.metric("Longest Movement Streak", f"{mo['longest_movement_streak_seconds']:.2f} s")
    col2.metric("% Frames Moving", f"{mo['frames_moving_pct']:.1f}%")


def render_court_coverage(perf_data: dict, court_coverage: dict | None = None):
    if court_coverage:
        st.metric("Court Coverage %", f"{court_coverage.get('court_coverage', 0):.1f}%")
    else:
        st.metric("Court Coverage %", "⏳ pending (Member 1)")


def render_bps_inputs(perf_data: dict):
    eff = perf_data["efficiency_scores"]
    st.caption("These 5 scores are combined (weighted) to compute the overall BPS score.")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Agility", f"{eff['agility_score']:.0f}")
    col2.metric("Recovery", f"{eff['recovery_score']:.0f}")
    col3.metric("Explosiveness", f"{eff['explosiveness_score']:.0f}")
    col4.metric("Consistency", f"{eff['consistency_score']:.0f}")
    col5.metric("Path Efficiency", f"{eff['path_efficiency']:.0f}")


# ---------------------------------------------------------------------------
# MOVEMENT EFFICIENCY — 5 individual metrics
# ---------------------------------------------------------------------------

def render_agility_score(perf_data: dict):
    eff = perf_data["efficiency_scores"]
    st.metric("Agility Score", f"{eff['agility_score']:.0f}")


def render_recovery_score(perf_data: dict):
    eff = perf_data["efficiency_scores"]
    st.metric("Recovery Score", f"{eff['recovery_score']:.0f}")


def render_explosiveness_score(perf_data: dict):
    eff = perf_data["efficiency_scores"]
    st.metric("Explosiveness Score", f"{eff['explosiveness_score']:.0f}")


def render_consistency_score(perf_data: dict):
    eff = perf_data["efficiency_scores"]
    st.metric("Consistency Score", f"{eff['consistency_score']:.0f}")


def render_path_efficiency_score(perf_data: dict):
    eff = perf_data["efficiency_scores"]
    st.metric("Path Efficiency", f"{eff['path_efficiency']:.0f}")


def render_efficiency_radar(perf_data: dict):
    """The radar chart showing all 5 efficiency dimensions together (bonus visual)."""
    eff = perf_data["efficiency_scores"]
    categories = ["Agility", "Recovery", "Explosiveness", "Consistency", "Path Efficiency"]
    values = [
        eff["agility_score"], eff["recovery_score"], eff["explosiveness_score"],
        eff["consistency_score"], eff["path_efficiency"],
    ]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(r=values_closed, theta=categories_closed, fill="toself", name="Efficiency")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, margin=dict(l=40, r=40, t=20, b=20), height=400,
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# BPS SCORE — 2 individual metrics
# ---------------------------------------------------------------------------

def render_overall_score(perf_data: dict):
    bps_score = perf_data["bps_score"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=bps_score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#bbf7d0"},
                ],
            },
            title={"text": "Overall BPS"},
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, width="stretch")


def render_breakdown(perf_data: dict):
    breakdown = perf_data["bps_breakdown"]
    st.caption("Score breakdown (component × weight = contribution)")
    labels = list(breakdown.keys())
    contributions = [breakdown[k]["contribution"] for k in labels]
    fig2 = go.Figure(go.Bar(x=contributions, y=labels, orientation="h", marker_color="#2563eb"))
    fig2.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Contribution to BPS")
    st.plotly_chart(fig2, width="stretch")


# ---------------------------------------------------------------------------
# SESSION GRADE — 2 individual metrics
# ---------------------------------------------------------------------------

def render_grade(perf_data: dict):
    grade = perf_data["grade"]
    grade_colors = {
        "A+": "#16a34a", "A": "#22c55e", "B+": "#84cc16", "B": "#eab308",
        "C+": "#f97316", "C": "#ef4444", "D": "#991b1b",
    }
    color = grade_colors.get(grade, "#64748b")
    st.markdown(
        f'<div style="font-size:64px;font-weight:800;color:{color};">{grade}</div>',
        unsafe_allow_html=True,
    )


def render_numeric_score(perf_data: dict):
    numeric_score = perf_data["numeric_score"]
    st.metric("Numeric Score", f"{numeric_score:.1f} / 100")


# ---------------------------------------------------------------------------
# Registry: maps (main_topic, sub_metric_label) -> render function
# Used by app.py to look up which function to call for each clicked button.
# ---------------------------------------------------------------------------

MATCH_OVERVIEW_METRICS = {
    "Total Distance": render_total_distance,
    "Average Speed": render_avg_speed,
    "Peak Speed": render_peak_speed,
    "Recovery Time": render_recovery_time,
    "Court Coverage %": render_court_coverage,
    "BPS Inputs": render_bps_inputs,
}

MOVEMENT_EFFICIENCY_METRICS = {
    "Agility Score": render_agility_score,
    "Recovery Score": render_recovery_score,
    "Explosiveness Score": render_explosiveness_score,
    "Consistency Score": render_consistency_score,
    "Path Efficiency": render_path_efficiency_score,
}

BPS_SCORE_METRICS = {
    "Overall Score": render_overall_score,
    "Breakdown": render_breakdown,
}

SESSION_GRADE_METRICS = {
    "Grade": render_grade,
    "Numeric Score": render_numeric_score,
}

ALL_METRICS_BY_TOPIC = {
    "Match Overview": MATCH_OVERVIEW_METRICS,
    "Movement Efficiency": MOVEMENT_EFFICIENCY_METRICS,
    "BPS Score": BPS_SCORE_METRICS,
    "Session Grade": SESSION_GRADE_METRICS,
}