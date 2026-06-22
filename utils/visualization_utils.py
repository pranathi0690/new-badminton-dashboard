import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_hip_center(landmarks_df):
    """Average left/right hip x,y into a single hip_center per frame."""
    hip_x = (landmarks_df["left_hip_x"] + landmarks_df["right_hip_x"]) / 2
    hip_y = (landmarks_df["left_hip_y"] + landmarks_df["right_hip_y"]) / 2
    return hip_x, hip_y


def build_heatmap(hip_x, hip_y):
    fig = px.density_heatmap(
        x=hip_x, y=hip_y,
        nbinsx=30, nbinsy=30,
        color_continuous_scale="Viridis",
        labels={"x": "Court X", "y": "Court Y"},
    )
    fig.update_yaxes(autorange="reversed")  # court Y typically top-down
    fig.update_layout(title="Court Occupancy Heatmap", height=450)
    return fig


def build_trajectory(hip_x, hip_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hip_x, y=hip_y,
        mode="lines+markers",
        line=dict(width=1, color="cyan"),
        marker=dict(size=3),
        name="Movement Path"
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Movement Trajectory", height=450)
    return fig


def compute_zone_distribution(hip_y):
    """Bucket hip_y into Front / Mid / Back based on tercile split of the Y range."""
    y_min, y_max = hip_y.min(), hip_y.max()
    third = (y_max - y_min) / 3

    front_bound = y_min + third
    mid_bound = y_min + 2 * third

    front = (hip_y <= front_bound).sum()
    mid = ((hip_y > front_bound) & (hip_y <= mid_bound)).sum()
    back = (hip_y > mid_bound).sum()

    total = front + mid + back
    return {
        "front_zone_pct": round(100 * front / total, 1),
        "mid_zone_pct": round(100 * mid / total, 1),
        "back_zone_pct": round(100 * back / total, 1),
    }


def compute_court_coverage(hip_x, hip_y):
    """Bounding box area covered, normalized to a 0-100 coverage score."""
    x_range = hip_x.max() - hip_x.min()
    y_range = hip_y.max() - hip_y.min()
    # Normalize against the full possible court range (assumes 0-1 normalized coords from MediaPipe)
    coverage_pct = round(100 * (x_range * y_range), 1)
    return min(coverage_pct, 100.0)