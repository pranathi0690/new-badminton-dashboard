"""
app.py — Badminton AI Performance Dashboard
Unified: Member 1 (Video/Court/Upload) + Member 2 (Performance) + Member 3 (Identity/Alerts/Tactical/Report)

Folder structure expected:
    app.py
    styling.py
    data/
        analytics.json, landmarks.csv, predictions.csv, features.csv, rf.pkl
    video/
        annotated_video.mp4
    pipeline/
        __init__.py
        run_pipeline.py
    utils/
        __init__.py
        visualization_utils.py
        scoring_engine.py
        grade_engine.py
        video_section.py
        court_section.py
        upload_section.py
        alerts.py
        identity_section.py
        tactical_section.py
        coach_report.py
        performance_section.py
        styling.py

session_state key namespacing:
    DEMO track  (loaded once at startup, never overwritten):
        demo_landmarks, demo_predictions, demo_analytics, demo_features
    UPLOAD track (set only after a user processes their own video):
        upload_result, upload_landmarks, upload_predictions
    SHARED computed:
        perf_data       — Member 2's analysis dict (from demo or upload)
        analytics       — analytics.json-shaped dict Member 3 reads (from demo or upload)
        predictions     — list[dict] Member 3's tactical/coach sections read
        court_coverage_result — returned by section_court(), passed to render_court_coverage()
"""

import os
import json
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from styling import apply_custom_theme

from utils.upload_section import render_upload_section, render_upload_results
from utils.video_section import section_video
from utils.court_section import section_court

from utils.alerts import render_alerts_section
from utils.identity_section import render_identity_section
from utils.tactical_section import render_tactical_section
from utils.coach_report import render_coach_report_section

from utils.performance_section import (
    get_performance_data,
    get_performance_data_from_df,
    get_analytics_export,
    render_total_distance,
    render_avg_speed,
    render_peak_speed,
    render_recovery_time,
    render_court_coverage,
    render_bps_inputs,
    render_agility_score,
    render_recovery_score,
    render_explosiveness_score,
    render_consistency_score,
    render_path_efficiency_score,
    render_efficiency_radar,
    render_overall_score,
    render_breakdown,
    render_grade,
    render_numeric_score,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Badminton AI Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()


# ---------------------------------------------------------------------------
# Load DEMO data once into session_state at startup — namespaced so Upload
# can never collide with or overwrite this data.
# ---------------------------------------------------------------------------
def load_demo_data():
    if "demo_landmarks" in st.session_state:
        return  # already loaded this session

    for key, path, loader in [
        ("demo_landmarks",   "data/landmarks.csv",   lambda p: pd.read_csv(p)),
        ("demo_predictions", "data/predictions.csv", lambda p: pd.read_csv(p)),
        ("demo_features",    "data/features.csv",    lambda p: pd.read_csv(p)),
        ("demo_analytics",   "data/analytics.json",  lambda p: json.load(open(p))),
    ]:
        if os.path.exists(path):
            st.session_state[key] = loader(path)
        else:
            st.session_state[key] = {} if key == "demo_analytics" else None


load_demo_data()


# ---------------------------------------------------------------------------
# Bridge: converts run_full_pipeline() output → Member 2 perf_data dict
# and → analytics.json-shaped dict Member 3's files read from session_state.
# ---------------------------------------------------------------------------
def compute_analytics_from_upload(upload_result: dict) -> tuple[dict, dict]:
    features_df = upload_result["features_df"]
    perf_data = get_performance_data_from_df(features_df)
    member2_fields = get_analytics_export(perf_data)

    # Start from demo analytics so Member 3's fields that aren't in the upload
    # (court_zone_coverage, top_strengths, etc.) still have fallback values.
    analytics_for_member3 = dict(st.session_state.get("demo_analytics") or {})
    analytics_for_member3["bps"] = member2_fields["bps_score"]
    analytics_for_member3["session_grade"] = member2_fields["grade"]
    analytics_for_member3["agility_score"] = member2_fields["agility_score"]
    analytics_for_member3["recovery_score"] = member2_fields["recovery_score"]
    analytics_for_member3["explosiveness_score"] = member2_fields["explosiveness_score"]
    analytics_for_member3["consistency_score"] = member2_fields["consistency_score"]
    analytics_for_member3["avg_path_efficiency"] = member2_fields["path_efficiency"] / 100

    return perf_data, analytics_for_member3


def predictions_df_to_records(predictions_df: pd.DataFrame) -> list[dict]:
    """Convert predictions_df into the list-of-dicts Member 3's sections expect."""
    return predictions_df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏸 Dashboard")
    selected = option_menu(
        menu_title=None,
        options=[
            "Video", "Upload", "Court",
            "Performance",
            "Identity", "Alerts", "Tactical", "Report",
        ],
        icons=[
            "camera-video", "cloud-upload", "geo-alt",
            "bar-chart-line",
            "person-badge", "exclamation-triangle", "diagram-3", "robot",
        ],
        menu_icon="cast",
        default_index=0,
    )

st.title("🏸 Badminton AI Performance Dashboard")

# ---------------------------------------------------------------------------
# Section router
# ---------------------------------------------------------------------------

# ── Upload ──────────────────────────────────────────────────────────────────
if selected == "Upload":
    upload_result = render_upload_section()
    if upload_result:
        # Store under upload_* keys — NEVER touch demo_* keys here.
        st.session_state["upload_result"] = upload_result
        st.session_state["upload_landmarks"] = upload_result["landmarks_df"]
        st.session_state["upload_predictions"] = upload_result["predictions_df"]

        # Bridge into Member 2 + Member 3 data shapes
        perf_data, fresh_analytics = compute_analytics_from_upload(upload_result)
        st.session_state["perf_data"] = perf_data
        st.session_state["analytics"] = fresh_analytics
        st.session_state["predictions"] = predictions_df_to_records(upload_result["predictions_df"])

    if "upload_result" in st.session_state:
        render_upload_results(st.session_state["upload_result"])

# ── Video ────────────────────────────────────────────────────────────────────
elif selected == "Video":
    section_video()

# ── Court ────────────────────────────────────────────────────────────────────
elif selected == "Court":
    court_result = section_court()
    if court_result:
        st.session_state["court_coverage_result"] = court_result

# ── Performance (Member 2) ───────────────────────────────────────────────────
elif selected == "Performance":
    st.markdown("## 📈 Performance Analytics")

    # Use upload's perf_data if available, else compute from demo features.csv
    if "perf_data" in st.session_state:
        perf_data = st.session_state["perf_data"]
    elif os.path.exists("data/features.csv"):
        perf_data = get_performance_data("data/features.csv")
        st.session_state["perf_data"] = perf_data

        # Also seed the analytics dict for Member 3 sections using demo data
        if "analytics" not in st.session_state:
            st.session_state["analytics"] = st.session_state.get("demo_analytics") or {}
    else:
        perf_data = None
        st.error(
            "data/features.csv not found and no video has been processed yet. "
            "Go to the Upload tab to process a video, or add features.csv to data/."
        )

    if perf_data is not None:
        court_coverage_data = st.session_state.get("court_coverage_result")

        st.markdown("### Match Overview")
        mo1, mo2, mo3 = st.columns(3)
        with mo1:
            render_total_distance(perf_data)
        with mo2:
            render_avg_speed(perf_data)
        with mo3:
            render_peak_speed(perf_data)
        render_recovery_time(perf_data)
        render_court_coverage(perf_data, court_coverage_data)
        render_bps_inputs(perf_data)

        st.divider()
        st.markdown("### Movement Efficiency")
        ef1, ef2, ef3, ef4, ef5 = st.columns(5)
        with ef1:
            render_agility_score(perf_data)
        with ef2:
            render_recovery_score(perf_data)
        with ef3:
            render_explosiveness_score(perf_data)
        with ef4:
            render_consistency_score(perf_data)
        with ef5:
            render_path_efficiency_score(perf_data)
        render_efficiency_radar(perf_data)

        st.divider()
        st.markdown("### BPS Score")
        bps_col1, bps_col2 = st.columns(2)
        with bps_col1:
            render_overall_score(perf_data)
        with bps_col2:
            render_breakdown(perf_data)

        st.divider()
        st.markdown("### Session Grade")
        grade_col1, grade_col2 = st.columns(2)
        with grade_col1:
            render_grade(perf_data)
        with grade_col2:
            render_numeric_score(perf_data)

# ── Identity (Member 3) ──────────────────────────────────────────────────────
elif selected == "Identity":
    # Fallback: if no upload has been processed, seed analytics from demo file
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = st.session_state.get("demo_analytics") or {}
    render_identity_section()

# ── Alerts (Member 3) ────────────────────────────────────────────────────────
elif selected == "Alerts":
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = st.session_state.get("demo_analytics") or {}
    render_alerts_section()

# ── Tactical (Member 3) ──────────────────────────────────────────────────────
elif selected == "Tactical":
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = st.session_state.get("demo_analytics") or {}
    # predictions: use upload track if available, else load from demo CSV
    if "predictions" not in st.session_state:
        if os.path.exists("data/predictions.csv"):
            demo_preds = pd.read_csv("data/predictions.csv")
            st.session_state["predictions"] = demo_preds.to_dict(orient="records")
    render_tactical_section()

# ── AI Coach Report (Member 3) ───────────────────────────────────────────────
elif selected == "Report":
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = st.session_state.get("demo_analytics") or {}
    if "predictions" not in st.session_state:
        if os.path.exists("data/predictions.csv"):
            demo_preds = pd.read_csv("data/predictions.csv")
            st.session_state["predictions"] = demo_preds.to_dict(orient="records")
    render_coach_report_section()
