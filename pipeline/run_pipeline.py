import os
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib
import subprocess
import imageio_ffmpeg

mp_pose = mp.solutions.pose

# --- THE FIX ---
# MediaPipe's PoseLandmark enum gives the REAL index for each named joint.
# Previously the code did `lm[i]` using enumerate() position (0,1,2,3...),
# which silently grabbed the WRONG points (e.g. "left_hip" was actually
# pulling the left eye's coordinates). This is now built directly from
# mp_pose.PoseLandmark so every name maps to its correct landmark index.
LANDMARK_INDEX = {
    "nose": mp_pose.PoseLandmark.NOSE.value,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST.value,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP.value,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP.value,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE.value,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE.value,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    "left_foot": mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
    "right_foot": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
}

# Joints the model's 53 features were engineered from (matches the offline
# feature_engineering.py used to train it)
MODEL_JOINTS = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_foot", "right_foot",
]

ROLLING_WINDOW = 5
COURT_ZONES_X = 3
COURT_ZONES_Y = 3

# Skeleton connections for drawing lines between joints
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

RF_MODEL_PATH = "data/rf.pkl"


def load_model_and_encoder():
    model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    encoder = None
    return model, encoder


def _build_model_features(landmarks_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Computes the FULL feature set the model was trained on (53 columns),
    matching feature_engineering.py from the offline pipeline. Needs the
    whole video's landmarks at once because of rolling windows / cumulative
    sums -- this is why prediction has to happen in a second pass, not
    frame-by-frame during video reading.
    """
    df = landmarks_df.sort_values("frame", kind="stable").reset_index(drop=True)
    f = pd.DataFrame()
    f["frame"] = df["frame"]

    x_cols = [f"{j}_x" for j in MODEL_JOINTS]
    y_cols = [f"{j}_y" for j in MODEL_JOINTS]
    for c in x_cols + y_cols:
        f[c] = df[c] if c in df.columns else np.nan

    for c in x_cols + y_cols:
        f[c] = f[c].ffill().bfill().fillna(0.5)  # hold last known position through brief detection gaps

    f["centroid_x"] = f[x_cols].mean(axis=1)
    f["centroid_y"] = f[y_cols].mean(axis=1)
    f["hip_center_x"] = (f["left_hip_x"] + f["right_hip_x"]) / 2
    f["hip_center_y"] = (f["left_hip_y"] + f["right_hip_y"]) / 2
    f["stance_width"] = np.sqrt((f["left_ankle_x"] - f["right_ankle_x"])**2 + (f["left_ankle_y"] - f["right_ankle_y"])**2)

    f["dx"] = f["centroid_x"].diff().fillna(0)
    f["dy"] = f["centroid_y"].diff().fillna(0)
    f["distance"] = np.sqrt(f["dx"]**2 + f["dy"]**2)
    f["speed"] = f["distance"]

    f["centroid_x_smooth"] = f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1, center=True).mean()
    f["centroid_y_smooth"] = f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1, center=True).mean()
    f["rolling_speed_mean"] = f["speed"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    f["rolling_speed_std"] = f["speed"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    f["speed_smoothed"] = f["rolling_speed_mean"]
    f["acceleration"] = f["speed"].diff().fillna(0)

    f["cumulative_distance"] = f["distance"].cumsum()
    f["path_length"] = f["cumulative_distance"]
    f["straight_distance"] = np.sqrt((f["centroid_x"] - f["centroid_x"].iloc[0])**2 + (f["centroid_y"] - f["centroid_y"].iloc[0])**2)
    f["path_efficiency"] = np.where(f["cumulative_distance"] > 0, f["straight_distance"] / f["cumulative_distance"], 0.0)

    angle = np.degrees(np.arctan2(f["dy"], f["dx"]))

    def classify_direction(dx_, dy_, ang_):
        speed = np.sqrt(dx_**2 + dy_**2)
        if speed < 1e-4:
            return "STILL"
        if -45 <= ang_ <= 45:
            return "RIGHT"
        elif 45 < ang_ <= 135:
            return "FORWARD"
        elif ang_ > 135 or ang_ <= -135:
            return "LEFT"
        return "BACKWARD"

    directions = [classify_direction(dx_, dy_, ang_) for dx_, dy_, ang_ in zip(f["dx"], f["dy"], angle)]
    direction_map = {"STILL": 0, "FORWARD": 1, "BACKWARD": 2, "LEFT": 3, "RIGHT": 4}
    f["direction_code"] = [direction_map[d] for d in directions]
    for d in ["BACKWARD", "FORWARD", "LEFT", "RIGHT", "STILL"]:
        f[f"direction_{d}"] = [1 if dd == d else 0 for dd in directions]
    f["direction_change"] = (pd.Series(directions) != pd.Series(directions).shift(1)).astype(int)
    f.loc[0, "direction_change"] = 0
    f["trajectory_angle"] = angle.fillna(0)

    f["movement_range_x"] = f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1).max() - f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1).min()
    f["movement_range_y"] = f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1).max() - f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1).min()

    x_norm = f["centroid_x"].clip(0, 1)
    y_norm = f["centroid_y"].clip(0, 1)
    col = np.floor(x_norm * COURT_ZONES_X).clip(0, COURT_ZONES_X - 1).astype(int)
    row = np.floor(y_norm * COURT_ZONES_Y).clip(0, COURT_ZONES_Y - 1).astype(int)
    f["court_zone"] = row * COURT_ZONES_X + col

    pos_std = f["centroid_x"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0) + f["centroid_y"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    f["stability_index"] = 1 / (1 + pos_std)

    baseline_x, baseline_y = f["hip_center_x"].median(), f["hip_center_y"].median()
    f["recovery_distance"] = np.sqrt((f["hip_center_x"] - baseline_x)**2 + (f["hip_center_y"] - baseline_y)**2)
    moving = f["speed"] > f["speed"].median()
    recovery_time, counter = [], 0
    for is_moving in moving:
        counter = counter + 1 if is_moving else 0
        recovery_time.append(counter)
    f["recovery_time"] = recovery_time

    return f


def run_full_pipeline(video_path, model, encoder, progress_callback=None):
    # --- THE FIX: unique filename per run, so Streamlit/browser can never
    # serve a stale cached video from a previous run at the same path ---
    run_id = int(time.time() * 1000)
    output_path = f"video/uploaded_annotated_{run_id}.mp4"
    os.makedirs("video", exist_ok=True)
    temp_output = output_path.replace(".mp4", "_temp.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    skeleton_video_path = temp_output.replace("_temp.mp4", "_skeleton.mp4")
    writer = cv2.VideoWriter(skeleton_video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open for {skeleton_video_path}. "
            f"This silently produces an empty/broken video with NO overlays "
            f"if not checked. Try a different fourcc, e.g. 'avc1' or 'XVID'."
        )

    landmarks_rows = []
    detected_frame_count = 0

    # --- PASS 1: detect pose, draw skeleton + bbox, collect raw landmarks ---
    # (No label drawn yet -- the model needs rolling-window features computed
    # across the WHOLE video, which we can't know frame-by-frame on the fly.)
    with mp_pose.Pose(model_complexity=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            target_long_side = 720
            scale = target_long_side / max(w, h)
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            row = {"frame": frame_idx}
            xs, ys = [], []
            joint_px = {}

            if result.pose_landmarks:
                detected_frame_count += 1
                lm = result.pose_landmarks.landmark
                for name, idx in LANDMARK_INDEX.items():
                    point = lm[idx]
                    row[f"{name}_x"] = point.x
                    row[f"{name}_y"] = point.y
                    px, py = int(point.x * w), int(point.y * h)
                    joint_px[name] = (px, py)
                    xs.append(px)
                    ys.append(py)

                for a, b in SKELETON_EDGES:
                    if a in joint_px and b in joint_px:
                        cv2.line(frame, joint_px[a], joint_px[b], (255, 255, 255), 3, cv2.LINE_AA)
                for px, py in joint_px.values():
                    cv2.circle(frame, (px, py), 6, (0, 200, 255), -1, cv2.LINE_AA)

            landmarks_rows.append(row)

            if xs and ys:
                cv2.rectangle(frame, (min(xs) - 15, min(ys) - 15), (max(xs) + 15, max(ys) + 15),
                              (0, 0, 255), 3, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
            if progress_callback:
                progress_callback("Detecting pose", frame_idx, total_frames)

    cap.release()
    writer.release()

    if detected_frame_count == 0:
        raise RuntimeError(
            "MediaPipe detected ZERO poses in this entire video. The overlay "
            "code is correct but had nothing to draw. Check: is a person "
            "clearly visible/well-lit in frame? Try lowering "
            "min_detection_confidence further or check the input video isn't corrupted."
        )

    landmarks_df = pd.DataFrame(landmarks_rows)

    # --- Compute the model's full feature set in batch (needs rolling windows) ---
    if progress_callback:
        progress_callback("Computing movement features", 1, 1)
    features_df = _build_model_features(landmarks_df, fps)

    # --- Run predictions for every frame at once ---
    predictions_rows = []
    expected_features = list(getattr(model, "feature_names_in_", [])) if model else []
    if model is not None and expected_features:
        missing = [c for c in expected_features if c not in features_df.columns]
        if missing:
            raise RuntimeError(f"Computed features are missing columns the model needs: {missing}")
        X = features_df[expected_features]
        probs_all = model.predict_proba(X)
        pred_labels = model.classes_[probs_all.argmax(axis=1)]
        confidences = probs_all.max(axis=1)
        for i, frame_num in enumerate(features_df["frame"]):
            predictions_rows.append({
                "frame": int(frame_num),
                "prediction": pred_labels[i],
                "confidence": float(confidences[i]),
            })
    else:
        for frame_num in features_df["frame"]:
            predictions_rows.append({"frame": int(frame_num), "prediction": "No Model", "confidence": 0.0})

    predictions_df = pd.DataFrame(predictions_rows)
    pred_by_frame = predictions_df.set_index("frame")

    if progress_callback:
        progress_callback("Distribution check", 1, 1)
    unique_label_count = predictions_df["prediction"].nunique()
    if model is not None and unique_label_count == 1 and len(predictions_df) > 30:
        # Don't fail hard -- could occasionally be a genuinely repetitive
        # clip -- but make this loud and visible rather than silent.
        print(
            f"WARNING: model predicted the SAME class for all "
            f"{len(predictions_df)} frames ('{predictions_df['prediction'].iloc[0]}'). "
            f"This usually means input features are still out-of-distribution -- "
            f"verify the model was trained on NORMALIZED (0-1) coordinates, "
            f"matching what's being fed here."
        )

    # --- PASS 2: re-read skeleton video, burn in the now-correct label per frame ---
    cap2 = cv2.VideoCapture(skeleton_video_path)
    writer2 = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))
    if not writer2.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {temp_output} (pass 2).")

    frame_idx = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        if frame_idx in pred_by_frame.index:
            pred_label = pred_by_frame.loc[frame_idx, "prediction"]
            confidence = pred_by_frame.loc[frame_idx, "confidence"]
        else:
            pred_label, confidence = "No Pose", 0.0

        label_text = f"{pred_label} ({confidence*100:.0f}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (5, 5), (15 + tw, 15 + th + 15), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (10, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        writer2.write(frame)
        frame_idx += 1
        if progress_callback:
            progress_callback("Burning in labels", frame_idx, total_frames)

    cap2.release()
    writer2.release()
    if os.path.exists(skeleton_video_path):
        os.remove(skeleton_video_path)

    if progress_callback:
        progress_callback("Re-encoding for browser", 1, 1)
    _reencode_for_browser(temp_output, output_path)
    if os.path.exists(temp_output):
        os.remove(temp_output)

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise RuntimeError(
            f"Final output video at {output_path} is missing or suspiciously "
            f"small -- the writer or ffmpeg re-encode silently failed."
        )

    label_counts = predictions_df["prediction"].value_counts().to_dict()
    label_confidence = predictions_df.groupby("prediction")["confidence"].mean().to_dict()

    return {
        "annotated_video_path": output_path,
        "total_frames": len(landmarks_df),
        "detected_frame_count": detected_frame_count,
        "label_counts": label_counts,
        "label_confidence": label_confidence,
        "landmarks_df": landmarks_df,
        "predictions_df": predictions_df,
        "features_df": features_df,
    }


def _reencode_for_browser(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_exe, "-y", "-i", input_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path],
        capture_output=True
    )
    if result.returncode != 0:
        # surface the real ffmpeg error instead of silently falling back to
        # an unplayable mp4v file
        raise RuntimeError(f"ffmpeg re-encode failed: {result.stderr.decode(errors='ignore')[-800:]}")
