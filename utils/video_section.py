import streamlit as st
import os
import cv2
import numpy as np


def section_video():
    st.subheader("🎥 Video Analysis Preview")

    # --- THE FIX ---
    # Use namespaced keys (demo_landmarks / demo_predictions) instead of the
    # shared "landmarks" / "predictions" keys. Those shared keys get
    # overwritten by upload_section.py after a user processes their own
    # video, which silently swapped this DEMO card's data out from under it
    # -- explaining the squished/mismatched skeleton (uploaded video's
    # coordinates drawn onto the demo video's frames).
    if "demo_landmarks" not in st.session_state or "demo_predictions" not in st.session_state:
        st.error(
            "Demo landmarks/predictions not found in session_state under "
            "'demo_landmarks' / 'demo_predictions'. Make sure app.py loads "
            "the FIXED demo CSVs into these specific keys at startup, "
            "separate from whatever keys the upload pipeline writes to."
        )
        return

    predictions = st.session_state["demo_predictions"]
    landmarks = st.session_state["demo_landmarks"]
    video_path = "video/annotated_video.mp4"

    col1, col2 = st.columns([1.3, 1])

    with col1:
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.warning("annotated_video.mp4 not found in /video — showing summary only.")

    with col2:
        st.markdown("**Processing Summary**")
        total_frames = len(predictions)
        st.metric("Frames Processed", total_frames)
        st.metric("Predicted Classes", predictions["prediction"].nunique())

        st.markdown("**Pipeline Status**")
        st.progress(1.0, text="Frames → Landmarks → Features → RF Predictions ✅")

    st.divider()
    _frame_inspector(landmarks, predictions, video_path)


def _frame_inspector(landmarks, predictions, video_path):
    st.markdown("**🔍 Frame Inspector — Skeleton + Bounding Box + RF Prediction**")

    # --- THE FIX: bound the slider by the ACTUAL video's frame count too,
    # not just the landmarks dataframe length. If they ever disagree again,
    # show a warning instead of silently seeking past the end of the video. ---
    video_frame_count = _get_video_frame_count(video_path)
    landmarks_frame_count = len(landmarks)

    if video_frame_count is not None and video_frame_count != landmarks_frame_count:
        st.warning(
            f"⚠️ Data mismatch: video has {video_frame_count} frames but "
            f"landmarks data has {landmarks_frame_count} rows. These should "
            f"always match -- showing the smaller of the two to avoid "
            f"out-of-range seeks."
        )

    max_frame = min(
        landmarks_frame_count,
        video_frame_count if video_frame_count is not None else landmarks_frame_count,
    ) - 1
    max_frame = max(max_frame, 0)

    frame_idx = st.slider("Select frame", 0, max_frame, 0)

    row = landmarks.iloc[frame_idx]
    pred_row = predictions.iloc[frame_idx] if frame_idx < len(predictions) else None

    img, frame_read_ok = _get_raw_frame(video_path, frame_idx)
    if not frame_read_ok:
        st.error(f"Could not read frame {frame_idx} from {video_path} — showing placeholder instead of garbage overlay.")
        return

    img = _draw_skeleton(img, row)
    img = _draw_bounding_box(img, row)

    if pred_row is not None:
        label = pred_row["prediction"]
        cv2.putText(img, f"Prediction: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    img_col, _ = st.columns([1, 1])
    with img_col:
        st.image(img, channels="BGR", width=420)


def _get_video_frame_count(video_path):
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count > 0 else None


def _get_raw_frame(video_path, frame_idx):
    """Returns (frame, success_bool). Previously silently returned a gray
    placeholder on failure with no signal to the caller, which let mismatched
    skeleton coordinates get drawn onto a blank/wrong-sized canvas without
    any visible warning."""
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame, True
    return np.full((480, 640, 3), 30, dtype=np.uint8), False


def _draw_skeleton(img, row):
    h, w = img.shape[:2]
    x_cols = [c for c in row.index if c.endswith("_x")]
    for x_col in x_cols:
        y_col = x_col[:-2] + "_y"
        if y_col in row.index:
            x_px = int(row[x_col] * w)
            y_px = int(row[y_col] * h)
            cv2.circle(img, (x_px, y_px), 4, (0, 200, 255), -1)
    return img


def _draw_bounding_box(img, row):
    h, w = img.shape[:2]
    x_cols = [c for c in row.index if c.endswith("_x")]
    y_cols = [c for c in row.index if c.endswith("_y")]

    if not x_cols or not y_cols:
        return img

    x_min = row[x_cols].min() * w
    x_max = row[x_cols].max() * w
    y_min = row[y_cols].min() * h
    y_max = row[y_cols].max() * h

    cv2.rectangle(
        img,
        (int(x_min), int(y_min)),
        (int(x_max), int(y_max)),
        (0, 0, 255), 2
    )
    return img
