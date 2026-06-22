import streamlit as st


def apply_custom_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0C1419;
        --card: #141C24;
        --border: #223240;
        --text: #EDF1F3;
        --muted: #7D8C95;
        --coral: #FF6B4A;
        --teal: #4FD1C5;
        --good: #34C9A3;
        --warn: #E8A33D;
        --poor: #E0533D;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #080D11;
        border-right: 1px solid var(--border);
    }

    h1 {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 1.85rem !important;
        color: var(--text);
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }

    h2 {
        font-family: 'Oswald', sans-serif;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted) !important;
        border: none !important;
        background: none !important;
        padding: 0 0 8px 0 !important;
        margin-top: 2.2rem !important;
        margin-bottom: 0.8rem !important;
        border-bottom: 1px solid var(--border) !important;
    }

    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 18px 12px 18px;
        transition: border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--teal);
    }

    [data-testid="stMetricValue"] {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        color: var(--coral) !important;
        font-size: 1.9rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--warn);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    [data-testid="stPlotlyChart"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 8px;
    }

    .nav-link {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.84rem !important;
        color: var(--muted) !important;
        border-radius: 6px !important;
    }
    .nav-link-selected {
        background: var(--card) !important;
        color: var(--teal) !important;
        border-left: 3px solid var(--teal) !important;
        font-weight: 700 !important;
    }

    hr {
        border-color: var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    .stProgress > div > div > div {
        background: var(--teal) !important;
    }

    .stButton button {
        background: var(--coral);
        color: #0C1419;
        border: none;
        font-weight: 600;
        border-radius: 6px;
    }
    .stButton button:hover {
        background: #ff8569;
    }
    </style>
    """, unsafe_allow_html=True)


def signal_color(score, good_threshold=70, mid_threshold=40):
    if score >= good_threshold:
        return "#34C9A3"
    elif score >= mid_threshold:
        return "#E8A33D"
    return "#E0533D"


def render_hero_stat(score, label, sublabel=""):
    """Signature hero stat — use ONCE (Match Overview / BPS section)."""
    color = signal_color(score)
    st.markdown(f"""
    <div style="background:#141C24; border:1px solid #223240; border-radius:10px;
                padding:26px 24px; border-top:3px solid #4FD1C5;">
        <div style="font-family:'Inter'; font-size:0.7rem; text-transform:uppercase;
                    letter-spacing:0.12em; color:#7D8C95; font-weight:600; margin-bottom:6px;">
            {label}
        </div>
        <div style="font-family:'Oswald'; font-weight:600; font-size:3.4rem;
                    color:{color}; line-height:1;">
            {score}
        </div>
        <div style="font-family:'Inter'; font-size:0.85rem; color:#7D8C95; margin-top:4px;">
            {sublabel}
        </div>
    </div>
    """, unsafe_allow_html=True)