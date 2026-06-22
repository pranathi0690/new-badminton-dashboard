"""
coach_report.py
Member 3 - Tactical & AI Coach Lead

Owns: AI Coach Report — full narrative writeup, broken into expandable sections:
    - Session Summary
    - Strength Analysis
    - Weakness Analysis
    - Tactical Insights
    - Recovery Recommendations
    - Training Suggestions

Rule-based templating (no external LLM API call) — deterministic, free,
no API key needed for the demo. If you later wire up Groq or Ollama,
build_report_inputs() is already the exact payload to send as context;
just swap generate_report_text() for an API call using the same dict.
"""

import json

from utils.identity_section import load_analytics, build_identity_card
from utils.tactical_section import load_predictions, build_tactical_data


def build_report_inputs(analytics: dict, predictions: list[dict] | None = None) -> dict:
    """Single payload combining everything the report needs."""
    identity = build_identity_card(analytics)
    tactical = build_tactical_data(analytics, predictions)

    return {
        "identity": identity,
        "tactical": tactical,
        "grade": analytics.get("session_grade", "N/A"),
        "bps": analytics.get("bps", 0),
        "duration": analytics.get("duration_seconds", 0),
        "pose_detection_rate": analytics.get("pose_detection_rate", 1.0),
        "avg_recovery_distance": analytics.get("avg_recovery_distance", 0),
        "avg_stance_width": analytics.get("avg_stance_width", 0),
    }


# ---------------------------------------------------------------------------
# Section generators — each returns a markdown string. Split out so they can
# be rendered as separate st.expander() blocks, per the build spec.
# ---------------------------------------------------------------------------
def generate_session_summary(inputs: dict) -> str:
    identity = inputs["identity"]
    grade = inputs["grade"]
    bps = inputs["bps"]
    duration = inputs["duration"]
    dom = identity["dominant_side"]

    return (
        f"This session ran **{duration:.1f} seconds** and earned an overall grade of "
        f"**{grade}** with a Badminton Performance Score (BPS) of **{bps:.1f} / 100**. "
        f"The player's profile is best described as a **{identity['archetype']}** — "
        f"{identity['archetype_description'].lower()}\n\n"
        f"Shot distribution leaned **{dom['side'].lower()}-dominant**, with "
        f"{dom['forehand_pct']:.0f}% forehand and {dom['backhand_pct']:.0f}% backhand usage. "
        f"Court positioning was centered around **{identity['court_preference']['label']}**, "
        f"covering {identity['court_preference']['top_zone_pct']:.0f}% of tracked time in that zone. "
        f"Pose detection confidence for this session was {inputs['pose_detection_rate']*100:.0f}%, "
        f"{'which is reliable for these readings.' if inputs['pose_detection_rate'] >= 0.85 else 'so some metrics here should be treated as approximate.'}"
    )


def generate_strength_analysis(inputs: dict) -> str:
    strengths = inputs["tactical"]["top_strengths"]
    if not strengths:
        return "No strength data available for this session."

    lines = ["The session's top three strengths, ranked by score:\n"]
    for i, s in enumerate(strengths, start=1):
        lines.append(
            f"{i}. **{s['metric'].replace('_', ' ').title()}** ({s['score']:.1f}/100) — {s['blurb']}"
        )

    lines.append(
        f"\nThe standout metric, **{strengths[0]['metric'].replace('_', ' ')}**, "
        f"is a strong foundation to build the rest of the player's game around. "
        f"Reinforcing this in training will help it hold up under match pressure."
    )
    return "\n".join(lines)


def generate_weakness_analysis(inputs: dict) -> str:
    weaknesses = inputs["tactical"]["top_weaknesses"]
    if not weaknesses:
        return "No weakness data available for this session."

    lines = ["The session's three lowest-scoring areas:\n"]
    for i, w in enumerate(weaknesses, start=1):
        lines.append(
            f"{i}. **{w['metric'].replace('_', ' ').title()}** ({w['score']:.1f}/100) — {w['blurb']}"
        )

    worst = weaknesses[0]
    lines.append(
        f"\n**{worst['metric'].replace('_', ' ').title()}** is the most urgent gap this session "
        f"(score: {worst['score']:.1f}). Left unaddressed, this metric is likely to be the first "
        f"thing exploited by a more experienced opponent."
    )
    return "\n".join(lines)


def generate_tactical_insights(inputs: dict) -> str:
    t = inputs["tactical"]["tactical"]
    identity = inputs["identity"]

    lines = [
        f"{t['tendency_text']}\n",
        f"Shot-side balance: **{t['forehand_pct']:.0f}% forehand** vs "
        f"**{t['backhand_pct']:.0f}% backhand**. "
    ]

    imbalance = abs(t["forehand_pct"] - t["backhand_pct"])
    if imbalance > 20:
        weaker = "forehand" if t["backhand_pct"] > t["forehand_pct"] else "backhand"
        lines.append(
            f"This {imbalance:.0f}-point gap suggests a tactical vulnerability — opponents "
            f"who consistently target the **{weaker}** side may be able to disrupt rhythm."
        )
    else:
        lines.append("Shot-side usage is reasonably balanced, giving fewer predictable patterns to exploit.")

    depth = t["depth_breakdown"]
    dominant_depth = max(depth, key=depth.get) if any(depth.values()) else None
    if dominant_depth:
        lines.append(
            f"\nCourt depth usage is concentrated in the **{dominant_depth} court** "
            f"({depth[dominant_depth]:.0f}%). Mixing in more variation across front/mid/back "
            f"would make positioning less predictable to opponents."
        )

    lines.append(f"\nPlay style tags this session: {', '.join(identity['style_tags'])}.")

    return "\n".join(lines)


def generate_recovery_recommendations(inputs: dict) -> str:
    recovery_dist = inputs["avg_recovery_distance"]
    stance_width = inputs["avg_stance_width"]

    lines = []
    if recovery_dist > 0.30:
        lines.append(
            f"Recovery distance averaged **{recovery_dist:.2f}**, above the healthy target of 0.30. "
            f"This means the player is consistently failing to return to base position between shots, "
            f"which compounds fatigue and opens up the court for the opponent's next shot."
        )
        lines.append(
            "**Recommendation:** Drill explicit split-step and recovery-step timing — practice "
            "returning to a marked base position after every shot in shadow footwork sessions."
        )
    else:
        lines.append(
            f"Recovery distance averaged **{recovery_dist:.2f}**, within a healthy range. "
            f"The player is generally resetting to base position effectively between shots."
        )
        lines.append("**Recommendation:** Maintain current recovery habits; consider raising the "
                      "intensity of recovery drills to build resilience under match fatigue.")

    if stance_width:
        lines.append(
            f"\nAverage stance width was **{stance_width:.3f}** (normalized). "
            f"{'A wider base generally supports better lateral stability.' if stance_width > 0.07 else 'A narrower stance can limit lateral reach — consider stance-width cues in footwork drills.'}"
        )

    return "\n".join(lines)


def generate_training_suggestions(inputs: dict) -> str:
    priorities = inputs["tactical"]["training_priorities"]
    if not priorities:
        return "No training priority data available."

    lines = ["Recommended training plan for the next session, ranked by urgency:\n"]
    for p in priorities:
        lines.append(
            f"**Priority {p['rank']}: {p['focus']}** (urgency {p['urgency']:.0f}/100)  \n"
            f"Drill: {p['suggestion']}"
        )

    lines.append(
        "\nSuggested structure: spend the first third of the next session on Priority 1 drills "
        "while movement is fresh, then rotate through Priority 2 and 3 as conditioning work."
    )
    return "\n\n".join(lines)


def generate_report_text(report_inputs: dict) -> str:
    """Full flat-text version (used for standalone/CLI testing and copy-paste export)."""
    identity = report_inputs["identity"]
    grade = report_inputs["grade"]
    bps = report_inputs["bps"]

    sections = [
        f"## AI Coach Report\n",
        f"**Grade {grade} · BPS {bps:.1f} · {identity['archetype']}**\n",
        "### Session Summary",
        generate_session_summary(report_inputs),
        "\n### Strength Analysis",
        generate_strength_analysis(report_inputs),
        "\n### Weakness Analysis",
        generate_weakness_analysis(report_inputs),
        "\n### Tactical Insights",
        generate_tactical_insights(report_inputs),
        "\n### Recovery Recommendations",
        generate_recovery_recommendations(report_inputs),
        "\n### Training Suggestions",
        generate_training_suggestions(report_inputs),
    ]
    return "\n\n".join(sections)


def render_coach_report_section(
    analytics: dict | None = None,
    predictions: list[dict] | None = None,
    json_path: str = "data/analytics.json",
    csv_path: str = "data/predictions.csv",
):
    import streamlit as st

    if analytics is None:
        analytics = st.session_state.get("analytics") or load_analytics(json_path)
    if predictions is None:
        predictions = st.session_state.get("predictions") or load_predictions(csv_path)

    inputs = build_report_inputs(analytics, predictions)

    st.markdown("## 🤖 AI Coach Report")
    st.caption(f"Grade {inputs['grade']} · BPS {inputs['bps']:.1f} · {inputs['identity']['archetype']}")

    with st.expander("📋 Session Summary", expanded=True):
        st.markdown(generate_session_summary(inputs))

    with st.expander("💪 Strength Analysis"):
        st.markdown(generate_strength_analysis(inputs))

    with st.expander("⚠️ Weakness Analysis"):
        st.markdown(generate_weakness_analysis(inputs))

    with st.expander("🎯 Tactical Insights"):
        st.markdown(generate_tactical_insights(inputs))

    with st.expander("🔄 Recovery Recommendations"):
        st.markdown(generate_recovery_recommendations(inputs))

    with st.expander("🏋️ Training Suggestions"):
        st.markdown(generate_training_suggestions(inputs))

    with st.expander("📄 Full Report (copy-paste text)"):
        st.text_area("Full report", generate_report_text(inputs), height=400, label_visibility="collapsed")


if __name__ == "__main__":
    analytics = load_analytics("data/analytics.json")
    predictions = load_predictions("data/predictions.csv")
    inputs = build_report_inputs(analytics, predictions)
    print(generate_report_text(inputs))
