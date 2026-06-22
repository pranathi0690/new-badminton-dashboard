"""
alerts.py
Member 3 - Tactical & AI Coach Lead

Owns: Coach Alert Box (Alerts, Notifications)

Reads from analytics.json (written by Member 1 + Member 2).
Generates rule-based, named alerts when metrics cross thresholds, and
renders them with st.error() / st.warning() / st.success() per spec.

generate_alerts() is a pure function (no I/O, no Streamlit) so it can be
unit-tested or reused headlessly by other sections (e.g. coach_report.py).
"""

import json


# ---------------------------------------------------------------------------
# Thresholds - tune these as a team once real session data comes in.
# Values are chosen against the live analytics.json fields:
#   forehand_usage / backhand_usage   -> 0-100 percentages
#   avg_recovery_distance             -> 0-1 normalized
#   court_zone_coverage               -> dict of zone -> % time
#   avg_path_efficiency               -> 0-1 normalized
#   avg_stability_index               -> 0-1 normalized (used as proxy for agility)
#   bps                               -> 0-100 composite score
#   pose_detection_rate               -> 0-1
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "forehand_dependency_pct": 65.0,    # forehand_usage above this -> dependency alert
    "backhand_dependency_pct": 65.0,    # backhand_usage above this -> dependency alert
    "poor_recovery_distance": 0.30,     # avg_recovery_distance above this -> poor recovery
    "low_court_coverage_zones": 2,      # fewer than this many zones used -> low coverage
    "excellent_agility_stability": 95.0,# avg_stability_index*100 above this -> excellent agility
    "low_path_efficiency": 40.0,
    "low_avg_speed": 200.0,
    "low_bps": 50.0,
    "low_pose_detection": 0.85,
}


def load_analytics(path: str = "data/analytics.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def generate_alerts(analytics: dict) -> list[dict]:
    """
    Returns a list of alert dicts:
        {"type": str, "level": "critical"|"warning"|"success"|"info", "message": str}
    `type` matches the named alerts from the build spec, for easy filtering/testing.
    """
    alerts = []

    forehand = analytics.get("forehand_usage", 0)
    backhand = analytics.get("backhand_usage", 0)
    recovery_dist = analytics.get("avg_recovery_distance", 0)
    zones = analytics.get("court_zone_coverage", {})
    stability = analytics.get("avg_stability_index", 0) * 100
    path_eff = analytics.get("avg_path_efficiency", 0) * 100
    avg_speed = analytics.get("avg_speed", 0)
    bps = analytics.get("bps", 0)
    pose_rate = analytics.get("pose_detection_rate", 1.0)

    # --- High Forehand / Backhand Dependency -------------------------------
    if forehand >= THRESHOLDS["forehand_dependency_pct"]:
        alerts.append({
            "type": "forehand_dependency",
            "level": "warning",
            "message": f"High forehand dependency — {forehand:.1f}% of shots were forehand. "
                       f"Opponents may target the backhand side to limit options.",
        })
    elif backhand >= THRESHOLDS["backhand_dependency_pct"]:
        alerts.append({
            "type": "backhand_dependency",
            "level": "warning",
            "message": f"High backhand dependency — {backhand:.1f}% of shots were backhand. "
                       f"Opponents may target the forehand side to limit options.",
        })

    # --- Poor Recovery Alert ------------------------------------------------
    if recovery_dist > THRESHOLDS["poor_recovery_distance"]:
        alerts.append({
            "type": "poor_recovery",
            "level": "critical",
            "message": f"Poor recovery — average recovery distance is {recovery_dist:.2f}, "
                       f"above the {THRESHOLDS['poor_recovery_distance']} target. "
                       f"Player is not consistently returning to base position after shots.",
        })

    # --- Low Court Coverage Alert -------------------------------------------
    zones_used = len(zones)
    if zones_used <= THRESHOLDS["low_court_coverage_zones"]:
        zone_list = ", ".join(zones.keys()) if zones else "none recorded"
        alerts.append({
            "type": "low_court_coverage",
            "level": "warning",
            "message": f"Low court coverage — only {zones_used} court zone(s) used "
                       f"({zone_list}). Movement may be too restricted to one area of the court.",
        })

    # --- Excellent Agility Alert (positive signal) --------------------------
    if stability >= THRESHOLDS["excellent_agility_stability"]:
        alerts.append({
            "type": "excellent_agility",
            "level": "success",
            "message": f"Excellent agility & stability — stability index of {stability:.1f}% "
                       f"is well above average. Strong balance and control under movement.",
        })

    # --- Path Efficiency ------------------------------------------------------
    if path_eff < THRESHOLDS["low_path_efficiency"]:
        alerts.append({
            "type": "low_path_efficiency",
            "level": "warning",
            "message": f"Path efficiency is low ({path_eff:.1f}%). Movement on court is "
                       f"taking longer, less direct routes than necessary.",
        })

    # --- Average Speed ----------------------------------------------------
    if avg_speed < THRESHOLDS["low_avg_speed"]:
        alerts.append({
            "type": "low_avg_speed",
            "level": "warning",
            "message": f"Average speed ({avg_speed:.0f}) is below the healthy range. "
                       f"Consider footwork and conditioning drills.",
        })

    # --- Overall BPS ----------------------------------------------------
    if bps < THRESHOLDS["low_bps"]:
        alerts.append({
            "type": "low_bps",
            "level": "critical",
            "message": f"Overall BPS score ({bps:.1f}) is below target. Session indicates "
                       f"multiple performance gaps that compound across metrics.",
        })
    elif bps >= 80:
        alerts.append({
            "type": "high_bps",
            "level": "success",
            "message": f"Strong overall BPS score ({bps:.1f}). Performance is well above "
                       f"the target threshold this session.",
        })

    # --- Data Quality ----------------------------------------------------
    if pose_rate < THRESHOLDS["low_pose_detection"]:
        alerts.append({
            "type": "low_pose_detection",
            "level": "info",
            "message": f"Pose detection rate was only {pose_rate*100:.0f}%. Some metrics "
                       f"in this report may be less reliable due to tracking gaps.",
        })

    if not alerts:
        alerts.append({
            "type": "all_clear",
            "level": "success",
            "message": "No critical issues detected this session. Solid, balanced performance overall.",
        })

    return alerts


def render_alerts_section(analytics: dict | None = None, json_path: str = "data/analytics.json"):
    """Streamlit rendering entrypoint. Import and call from app.py."""
    import streamlit as st

    if analytics is None:
        analytics = st.session_state.get("analytics") or load_analytics(json_path)

    st.markdown("## 🚨 Coach Alert Box")

    alerts = generate_alerts(analytics)

    critical = [a for a in alerts if a["level"] == "critical"]
    warning = [a for a in alerts if a["level"] == "warning"]
    success = [a for a in alerts if a["level"] == "success"]
    info = [a for a in alerts if a["level"] == "info"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical", len(critical))
    c2.metric("Warnings", len(warning))
    c3.metric("Positive", len(success))
    c4.metric("Info", len(info))

    for a in critical:
        st.error(a["message"], icon="🔴")
    for a in warning:
        st.warning(a["message"], icon="🟠")
    for a in success:
        st.success(a["message"], icon="🟢")
    for a in info:
        st.info(a["message"], icon="🔵")


# Allow standalone run for quick testing: `python alerts.py`
if __name__ == "__main__":
    data = load_analytics("data/analytics.json")
    for a in generate_alerts(data):
        print(f"[{a['level'].upper()}] ({a['type']}) {a['message']}")
