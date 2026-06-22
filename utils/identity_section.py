"""
identity_section.py
Member 3 - Tactical & AI Coach Lead

Owns: Player Identity Card
    - Archetype (rule-based playstyle classification)
    - Dominant Side (forehand vs backhand % comparison)
    - Court Preference (front/mid/back tendency)
    - Strongest Area / Improvement Area
    - Session Summary
    - Style Tags (extra flavor descriptors)
    - Court Split visual (st.progress)
"""

import json


def load_analytics(path: str = "data/analytics.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Zone label mapping. court_zone_coverage keys are zone IDs (strings) -> %
# time spent. NOTE: this mapping is a placeholder based on a 3x3 court grid
# assumption - confirm the real zone numbering with whoever owns
# court_section.py (Member 1) before relying on it for accuracy.
# ---------------------------------------------------------------------------
ZONE_LABELS = {
    "1": "Front Left", "2": "Front Center", "3": "Front Right",
    "4": "Mid Left", "5": "Mid Center", "6": "Mid Right",
    "7": "Back Left", "8": "Back Center", "9": "Back Right",
}

ZONE_DEPTH = {
    "1": "front", "2": "front", "3": "front",
    "4": "mid", "5": "mid", "6": "mid",
    "7": "back", "8": "back", "9": "back",
}


def determine_archetype(analytics: dict) -> str:
    """Classify playing style from movement + shot-mix signals."""
    path_eff = analytics.get("avg_path_efficiency", 0) * 100
    avg_speed = analytics.get("avg_speed", 0)
    stability = analytics.get("avg_stability_index", 0) * 100
    forehand = analytics.get("forehand_usage", 0)
    backhand = analytics.get("backhand_usage", 0)

    if stability > 90 and path_eff < 40:
        return "Defensive Anchor"
    if avg_speed > 400 and path_eff > 60:
        return "Aggressive Attacker"
    if abs(forehand - backhand) > 30:
        return "One-Sided Specialist"
    if stability > 85 and avg_speed > 250:
        return "All-Court Controller"
    return "Developing Player"


def archetype_description(archetype: str) -> str:
    """One-line flavor text per archetype, shown under the badge."""
    descriptions = {
        "Defensive Anchor": "Prioritizes balance and control over aggressive shot-making. "
                             "Wins through consistency and forcing opponent errors.",
        "Aggressive Attacker": "High-tempo, direct movement with fast attacking shots. "
                                "Looks to dominate rallies early.",
        "One-Sided Specialist": "Heavily reliant on one wing. Effective when that side is "
                                 "in form, but predictable under pressure.",
        "All-Court Controller": "Strong movement and positional play across the whole court. "
                                 "Well-rounded foundation to build tactics on.",
        "Developing Player": "Fundamentals are still forming. Focus on movement consistency "
                              "before adding tactical complexity.",
    }
    return descriptions.get(archetype, "")


def determine_dominant_side(analytics: dict) -> dict:
    forehand = analytics.get("forehand_usage", 0)
    backhand = analytics.get("backhand_usage", 0)
    if abs(forehand - backhand) < 5:
        side = "Balanced"
    else:
        side = "Backhand" if backhand > forehand else "Forehand"
    return {"side": side, "forehand_pct": forehand, "backhand_pct": backhand}


def determine_court_preference(analytics: dict) -> dict:
    zones = analytics.get("court_zone_coverage", {})
    if not zones:
        return {"label": "Unknown", "top_zone_pct": 0, "depth_breakdown": {}}

    top_zone = max(zones, key=zones.get)
    label = ZONE_LABELS.get(top_zone, f"Zone {top_zone}")

    # Aggregate by depth (front/mid/back) regardless of left/center/right
    depth_breakdown = {"front": 0.0, "mid": 0.0, "back": 0.0}
    for zone_id, pct in zones.items():
        depth = ZONE_DEPTH.get(zone_id)
        if depth:
            depth_breakdown[depth] += pct

    return {
        "label": label,
        "top_zone_pct": zones[top_zone],
        "depth_breakdown": depth_breakdown,
    }


def determine_strongest_area(analytics: dict) -> dict:
    strengths = analytics.get("top_strengths", [])
    if not strengths:
        return {"metric": "N/A", "score": 0}
    return strengths[0]


def determine_improvement_area(analytics: dict) -> dict:
    weaknesses = analytics.get("top_weaknesses", [])
    if not weaknesses:
        return {"metric": "N/A", "score": 0}
    return min(weaknesses, key=lambda w: w["score"])


def determine_style_tags(analytics: dict) -> list[str]:
    """Extra descriptive chips for the profile card - more texture, more info."""
    tags = []
    stability = analytics.get("avg_stability_index", 0) * 100
    path_eff = analytics.get("avg_path_efficiency", 0) * 100
    avg_speed = analytics.get("avg_speed", 0)
    recovery_dist = analytics.get("avg_recovery_distance", 0)
    stance_width = analytics.get("avg_stance_width", 0)

    if stability >= 95:
        tags.append("Rock-Solid Balance")
    if path_eff < 35:
        tags.append("Indirect Movement")
    elif path_eff > 65:
        tags.append("Efficient Mover")
    if avg_speed > 350:
        tags.append("High Tempo")
    elif avg_speed < 200:
        tags.append("Measured Pace")
    if recovery_dist < 0.15:
        tags.append("Fast Recovery")
    elif recovery_dist > 0.30:
        tags.append("Slow to Reset")
    if stance_width > 0.08:
        tags.append("Wide Base Stance")

    return tags or ["Standard Profile"]


def build_session_summary(analytics: dict) -> str:
    duration = analytics.get("duration_seconds", 0)
    grade = analytics.get("session_grade", "N/A")
    bps = analytics.get("bps", 0)
    dominant = determine_dominant_side(analytics)
    strongest = determine_strongest_area(analytics)

    return (
        f"A {duration:.1f}s session graded {grade} (BPS {bps:.1f}). "
        f"Play leaned {dominant['side'].lower()}-dominant "
        f"({dominant['forehand_pct']:.0f}% forehand / {dominant['backhand_pct']:.0f}% backhand) "
        f"with {strongest['metric'].replace('_', ' ')} as the standout strength "
        f"at a score of {strongest['score']:.1f}."
    )


def build_identity_card(analytics: dict) -> dict:
    return {
        "archetype": determine_archetype(analytics),
        "archetype_description": archetype_description(determine_archetype(analytics)),
        "dominant_side": determine_dominant_side(analytics),
        "court_preference": determine_court_preference(analytics),
        "strongest_area": determine_strongest_area(analytics),
        "improvement_area": determine_improvement_area(analytics),
        "style_tags": determine_style_tags(analytics),
        "session_summary": build_session_summary(analytics),
    }


def render_identity_section(analytics: dict | None = None, json_path: str = "data/analytics.json"):
    import streamlit as st

    if analytics is None:
        analytics = st.session_state.get("analytics") or load_analytics(json_path)

    card = build_identity_card(analytics)

    st.markdown("## 🧍 Player Identity Card")

    # --- Archetype badge ---------------------------------------------------
    st.markdown(f"### {card['archetype']}")
    st.caption(card["archetype_description"])

    # --- Style tags ----------------------------------------------------
    st.markdown(" ".join(f"`{tag}`" for tag in card["style_tags"]))

    st.markdown("")  # spacing

    # --- Core metrics row ----------------------------------------------------
    col1, col2, col3 = st.columns(3)
    dom = card["dominant_side"]
    col1.metric("Dominant Side", dom["side"])
    col2.metric("Strongest Area", card["strongest_area"]["metric"].replace("_", " ").title(),
                f"{card['strongest_area']['score']:.1f}")
    col3.metric("Improvement Area", card["improvement_area"]["metric"].replace("_", " ").title(),
                f"{card['improvement_area']['score']:.1f}", delta_color="inverse")

    # --- Forehand / Backhand split (st.progress) ----------------------------
    st.markdown("**Shot Side Split**")
    st.progress(dom["forehand_pct"] / 100, text=f"Forehand {dom['forehand_pct']:.0f}%")
    st.progress(dom["backhand_pct"] / 100, text=f"Backhand {dom['backhand_pct']:.0f}%")

    # --- Court preference / depth split ----------------------------
    st.markdown("**Court Preference**")
    cp = card["court_preference"]
    st.markdown(f"Primary zone: **{cp['label']}** ({cp['top_zone_pct']:.0f}% of time)")
    depth = cp["depth_breakdown"]
    if depth:
        d1, d2, d3 = st.columns(3)
        d1.metric("Front Court", f"{depth.get('front', 0):.0f}%")
        d2.metric("Mid Court", f"{depth.get('mid', 0):.0f}%")
        d3.metric("Back Court", f"{depth.get('back', 0):.0f}%")

    st.markdown(f"**Session Summary:** {card['session_summary']}")


if __name__ == "__main__":
    data = load_analytics("data/analytics.json")
    card = build_identity_card(data)
    import json as _json
    print(_json.dumps(card, indent=2))
