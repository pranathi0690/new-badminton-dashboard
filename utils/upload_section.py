import os
import tempfile
import traceback
import streamlit as st
from pipeline.run_pipeline import load_model_and_encoder, run_full_pipeline


@st.cache_resource
def get_model_and_encoder():
    return load_model_and_encoder()


def render_upload_section():
    st.markdown("### 🎬 Upload a Match Video")
    st.caption("Upload a raw .mp4 clip to run pose detection → footwork classification → annotated video.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is None:
        return None

    process_clicked = st.button("▶️ Process Video", type="primary")
    if not process_clicked:
        st.info("Click 'Process Video' to run the pipeline.")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        temp_video_path = tmp.name

    model, encoder = get_model_and_encoder()
    if model is None:
        st.warning("rf.pkl not found in data/ — video will be annotated with skeleton/bbox only, no footwork predictions.")

    progress_bar = st.progress(0, text="Starting...")

    def update_progress(stage, current, total):
        pct = current / total if total else 0
        progress_bar.progress(min(pct, 1.0), text=f"{stage}: {current}/{total}")

    try:
        with st.spinner("Processing video — this may take a minute..."):
            result = run_full_pipeline(temp_video_path, model, encoder, progress_callback=update_progress)
        st.success(
            f"✅ Processing complete! Pose detected in "
            f"{result['detected_frame_count']}/{result['total_frames']} frames."
        )
        if result["detected_frame_count"] < result["total_frames"] * 0.5:
            st.warning(
                "⚠️ Pose was only detected in fewer than half the frames. "
                "Overlays will be missing on undetected frames. Check lighting/framing in the source video."
            )
        return result
    except Exception as e:
        # THE FIX: show the real error and full traceback in an expander
        # instead of a generic one-liner that hides what actually broke.
        st.error(f"❌ Pipeline failed: {e}")
        with st.expander("Full error details (for debugging)"):
            st.code(traceback.format_exc())
        return None
    finally:
        os.unlink(temp_video_path)


def render_upload_results(result: dict):
    if result is None:
        return

    st.subheader("🎥 Uploaded Video — Analysis Results")

    # THE FIX: cache-bust the video src so Streamlit/browser can never show
    # a stale cached video from a previous run, even if paths were ever
    # reused. run_pipeline.py now generates a unique filename per run too.
    video_path = result["annotated_video_path"]
    if not os.path.exists(video_path):
        st.error(f"Annotated video file not found at {video_path}")
        return

    col1, col2 = st.columns([1.3, 1])
    with col1:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)  # bytes, not path -- avoids any path/URL caching entirely

    with col2:
        st.markdown("**Processing Summary**")
        st.metric("Total Frames Processed", result["total_frames"])
        st.metric("Frames With Pose Detected", result["detected_frame_count"])
        st.metric("Footwork Labels Detected", len(result["label_counts"]))

        st.markdown("**Footwork Label Breakdown**")
        for label, count in result["label_counts"].items():
            pct = round(100 * count / result["total_frames"], 1)
            avg_conf = result["label_confidence"].get(label, 0.0)
            st.write(f"**{label}** — {count} frames ({pct}%) · avg confidence {avg_conf*100:.0f}%")
