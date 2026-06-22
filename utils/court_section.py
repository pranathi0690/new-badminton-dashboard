import streamlit as st
from utils.visualization_utils import (
    get_hip_center,
    build_heatmap,
    build_trajectory,
    compute_zone_distribution,
    compute_court_coverage,
)


def section_court():
    st.subheader("📍 Court Intelligence")

    # THE FIX: read from demo_landmarks / demo_analytics (the namespaced keys
    # that app.py writes at startup) instead of the old bare "landmarks" /
    # "analytics" keys which no longer exist in session_state.
    if "demo_landmarks" not in st.session_state:
        st.error(
            "Demo landmarks not found in session_state under 'demo_landmarks'. "
            "Make sure app.py has finished loading data/landmarks.csv at startup."
        )
        return

    if "demo_analytics" not in st.session_state:
        st.error(
            "Demo analytics not found in session_state under 'demo_analytics'. "
            "Make sure app.py has finished loading data/analytics.json at startup."
        )
        return

    landmarks = st.session_state["demo_landmarks"]
    analytics = st.session_state["demo_analytics"]

    hip_x, hip_y = get_hip_center(landmarks)

    # ---------- Computed fresh from landmarks.csv ----------
    coverage = compute_court_coverage(hip_x, hip_y)
    zones = compute_zone_distribution(hip_y)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Court Coverage % (from landmarks)", f"{coverage}%")
    col2.metric("Front Court %", f"{zones['front_zone_pct']}%")
    col3.metric("Mid Court %", f"{zones['mid_zone_pct']}%")
    col4.metric("Back Court %", f"{zones['back_zone_pct']}%")

    st.divider()

    # ---------- Heatmap + Trajectory ----------
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(build_heatmap(hip_x, hip_y), use_container_width=True)
    with col_b:
        st.plotly_chart(build_trajectory(hip_x, hip_y), use_container_width=True)

    # ---------- Court Occupancy (frame counts per row-zone) ----------
    st.markdown("**Court Occupancy (frame counts)**")
    total_frames = len(hip_y)
    occ_col1, occ_col2, occ_col3 = st.columns(3)
    occ_col1.metric("Front Frames", int(round(zones["front_zone_pct"] / 100 * total_frames)))
    occ_col2.metric("Mid Frames", int(round(zones["mid_zone_pct"] / 100 * total_frames)))
    occ_col3.metric("Back Frames", int(round(zones["back_zone_pct"] / 100 * total_frames)))

    st.divider()

    # ---------- Reference: existing 9-zone grid from analytics.json ----------
    st.markdown("**9-Zone Court Grid (from analytics.json — teammate's RF-based zone classifier)**")
    zone_data = analytics.get("court_zone_coverage", {})
    if zone_data:
        zone_cols = st.columns(len(zone_data))
        zone_labels = {
            "1": "Front-Left", "2": "Front-Center", "3": "Front-Right",
            "4": "Mid-Left", "5": "Mid-Center", "6": "Mid-Right",
            "7": "Back-Left", "8": "Back-Center", "9": "Back-Right",
        }
        for col, (zone_id, pct) in zip(zone_cols, zone_data.items()):
            label = zone_labels.get(zone_id, f"Zone {zone_id}")
            col.metric(label, f"{pct}%")
    else:
        st.info("No court_zone_coverage data found in analytics.json.")

    return {
        "court_coverage": coverage,
        "front_zone_pct": zones["front_zone_pct"],
        "mid_zone_pct": zones["mid_zone_pct"],
        "back_zone_pct": zones["back_zone_pct"],
    }
