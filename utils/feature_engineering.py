"""
feature_engineering.py  (Step 4)

Reads new_landmarks.csv (named joint columns: left_shoulder_x, left_shoulder_y, ...)
and computes the exact 53 features the trained RandomForest model expects
(model.feature_names_in_), in the correct order, writing features.csv.

Run:
    python feature_engineering.py
"""

import numpy as np
import pandas as pd
from safe_write import safe_write_csv

LANDMARKS_PATH = "new_landmarks.csv"
OUTPUT_PATH = "features.csv"

ROLLING_WINDOW = 5  # frames, for rolling mean/std smoothing
FPS = 60.002

# Source video resolution -- inferred from raw MediaPipe pixel coordinates
# (max x ~1854, max y ~960), best guess confirmed by user. Update if known
# precisely; this scales all distance/speed/acceleration outputs.
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Court boundaries used for normalizing court_zone / centroid -- adjust if
# your MediaPipe coordinates aren't already normalized 0-1.
COURT_ZONES_X = 3  # left / center / right
COURT_ZONES_Y = 3  # near / mid / far -> zone = row*3+col, 0-8 (matches court_zone 0-8 seen earlier)


def compute_features(landmarks: pd.DataFrame) -> pd.DataFrame:
    df = landmarks.sort_values("frame", kind="stable").reset_index(drop=True).copy()

    # IMPORTANT: the trained model expects NORMALIZED (0-1) coordinates --
    # feeding it pixel-scale values pushes inputs out of its training
    # distribution and causes degenerate, overconfident single-class
    # predictions (verified: pixel-scale input -> 100% Forehand_Front @ 0.89
    # confidence on every frame, which is not real signal).
    # So: keep all MODEL features in normalized space. Pixel conversion is
    # used only for the separate, display-only speed_px_per_sec column below.
    f = pd.DataFrame()
    f["frame"] = df["frame"]

    # --- Raw joint passthrough (needed directly by the model) ---
    joint_cols = [
        "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y",
        "left_hip_x", "left_hip_y", "right_hip_x", "right_hip_y",
        "left_knee_x", "left_knee_y", "right_knee_x", "right_knee_y",
        "left_ankle_x", "left_ankle_y", "right_ankle_x", "right_ankle_y",
        "left_foot_x", "left_foot_y", "right_foot_x", "right_foot_y",
    ]
    for c in joint_cols:
        f[c] = df[c]

    # --- Centroid (average of all 10 body points) ---
    x_cols = [c for c in joint_cols if c.endswith("_x")]
    y_cols = [c for c in joint_cols if c.endswith("_y")]
    f["centroid_x"] = df[x_cols].mean(axis=1)
    f["centroid_y"] = df[y_cols].mean(axis=1)

    # --- Hip center ---
    f["hip_center_x"] = (df["left_hip_x"] + df["right_hip_x"]) / 2
    f["hip_center_y"] = (df["left_hip_y"] + df["right_hip_y"]) / 2

    # --- Stance width (distance between ankles) ---
    f["stance_width"] = np.sqrt(
        (df["left_ankle_x"] - df["right_ankle_x"]) ** 2
        + (df["left_ankle_y"] - df["right_ankle_y"]) ** 2
    )

    # --- Frame-to-frame displacement of centroid ---
    f["dx"] = f["centroid_x"].diff().fillna(0)
    f["dy"] = f["centroid_y"].diff().fillna(0)

    # --- Distance & speed (per-frame displacement) ---
    f["distance"] = np.sqrt(f["dx"] ** 2 + f["dy"] ** 2)
    f["speed"] = f["distance"]  # normalized distance/frame -- what the model was trained on

    # --- Display-only pixel-space speed (NOT fed to the model) ---
    centroid_x_px = f["centroid_x"] * VIDEO_WIDTH
    centroid_y_px = f["centroid_y"] * VIDEO_HEIGHT
    dx_px = centroid_x_px.diff().fillna(0)
    dy_px = centroid_y_px.diff().fillna(0)
    distance_px = np.sqrt(dx_px ** 2 + dy_px ** 2)
    f["speed_px_per_sec"] = distance_px * FPS  # pixels/sec, readable real-unit speed for dashboard

    # --- Smoothed centroid (rolling mean, reduces jitter) ---
    f["centroid_x_smooth"] = f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1, center=True).mean()
    f["centroid_y_smooth"] = f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1, center=True).mean()

    # --- Rolling speed stats ---
    f["rolling_speed_mean"] = f["speed"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    f["rolling_speed_std"] = f["speed"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    f["speed_smoothed"] = f["rolling_speed_mean"]

    # --- Acceleration (change in speed) ---
    f["acceleration"] = f["speed"].diff().fillna(0)

    # --- Cumulative / straight-line / path distances ---
    f["cumulative_distance"] = f["distance"].cumsum()
    f["path_length"] = f["cumulative_distance"]  # alias used by model
    f["straight_distance"] = np.sqrt(
        (f["centroid_x"] - f["centroid_x"].iloc[0]) ** 2
        + (f["centroid_y"] - f["centroid_y"].iloc[0]) ** 2
    )
    # path efficiency: straight-line distance from start / total distance traveled so far
    f["path_efficiency"] = np.where(
        f["cumulative_distance"] > 0,
        f["straight_distance"] / f["cumulative_distance"],
        0.0,
    )

    # --- Direction classification (8-way -> collapsed to 5 categories used by model) ---
    angle = np.degrees(np.arctan2(f["dy"], f["dx"]))

    def classify_direction(row_dx, row_dy, row_angle):
        speed = np.sqrt(row_dx ** 2 + row_dy ** 2)
        if speed < 1e-4:
            return "STILL"
        if -45 <= row_angle <= 45:
            return "RIGHT"
        elif 45 < row_angle <= 135:
            return "FORWARD"
        elif row_angle > 135 or row_angle <= -135:
            return "LEFT"
        else:
            return "BACKWARD"

    directions = [
        classify_direction(dx_, dy_, ang_)
        for dx_, dy_, ang_ in zip(f["dx"], f["dy"], angle)
    ]
    f["direction"] = directions

    direction_map = {"STILL": 0, "FORWARD": 1, "BACKWARD": 2, "LEFT": 3, "RIGHT": 4}
    f["direction_code"] = f["direction"].map(direction_map)

    # one-hot direction columns (model expects these exact names)
    for d in ["BACKWARD", "FORWARD", "LEFT", "RIGHT", "STILL"]:
        f[f"direction_{d}"] = (f["direction"] == d).astype(int)

    # --- Direction change (1 if direction differs from previous frame) ---
    f["direction_change"] = (f["direction"] != f["direction"].shift(1)).astype(int)
    f.loc[0, "direction_change"] = 0

    # --- Trajectory angle ---
    f["trajectory_angle"] = angle.fillna(0)

    # --- Movement range (rolling max-min over window, captures local spread) ---
    f["movement_range_x"] = (
        f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1).max()
        - f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1).min()
    )
    f["movement_range_y"] = (
        f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1).max()
        - f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1).min()
    )

    # --- Court zone (0-8 grid based on centroid position, normalized space) ---
    x_norm = f["centroid_x"].clip(0, 1)
    y_norm = f["centroid_y"].clip(0, 1)
    col = np.floor(x_norm * COURT_ZONES_X).clip(0, COURT_ZONES_X - 1).astype(int)
    row = np.floor(y_norm * COURT_ZONES_Y).clip(0, COURT_ZONES_Y - 1).astype(int)
    f["court_zone"] = row * COURT_ZONES_X + col

    # --- Stability index (inverse of rolling std of centroid position) ---
    pos_std = (
        f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
        + f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    )
    f["stability_index"] = 1 / (1 + pos_std)

    # --- Recovery distance & time (distance/time back to a resting hip-center baseline) ---
    baseline_x = f["hip_center_x"].median()
    baseline_y = f["hip_center_y"].median()
    f["recovery_distance"] = np.sqrt(
        (f["hip_center_x"] - baseline_x) ** 2 + (f["hip_center_y"] - baseline_y) ** 2
    )
    # recovery_time: consecutive frames spent above a "moving" speed threshold
    moving = f["speed"] > f["speed"].median()
    recovery_time = []
    counter = 0
    for is_moving in moving:
        counter = counter + 1 if is_moving else 0
        recovery_time.append(counter)
    f["recovery_time"] = recovery_time

    return f


def main():
    landmarks = pd.read_csv(LANDMARKS_PATH)
    print(f"Loaded {len(landmarks)} rows from {LANDMARKS_PATH}")

    features = compute_features(landmarks)

    # drop the human-readable 'direction' column before saving --
    # model only needs direction_code + one-hot columns
    features_for_model = features.drop(columns=["direction"])

    print(f"Computed {len(features_for_model.columns) - 1} feature columns (+ frame)")
    safe_write_csv(features_for_model, OUTPUT_PATH)


if __name__ == "__main__":
    main()
