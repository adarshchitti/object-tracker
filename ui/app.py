"""Streamlit UI for the Object Identification service."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time

import requests
import streamlit as st

from ui.api_client import ApiError, get_task_result, get_task_status, upload_video

DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 300


def init_session_state() -> None:
    defaults = {
        "task_id": None,
        "task_status": None,
        "task_result": None,
        "task_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_task_state() -> None:
    st.session_state.task_id = None
    st.session_state.task_status = None
    st.session_state.task_result = None
    st.session_state.task_error = None


def render_sidebar() -> str:
    st.sidebar.header("Configuration")
    base_url = st.sidebar.text_input(
        "API base URL",
        value=DEFAULT_API_BASE_URL,
        help="The FastAPI server. Defaults to localhost or $API_BASE_URL.",
    )
    if st.sidebar.button("Reset"):
        reset_task_state()
        st.rerun()

    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=2)
        if r.status_code == 200:
            st.sidebar.success("API reachable")
        else:
            st.sidebar.error(f"API responded {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"API unreachable: {type(e).__name__}")

    return base_url


def render_upload(base_url: str) -> None:
    st.subheader("Upload a video")
    uploaded = st.file_uploader(
        "Video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Max 100 MB. Supported: mp4, mov, avi, mkv.",
    )

    if uploaded is not None and st.button("Analyze", type="primary"):
        with st.spinner("Uploading..."):
            try:
                handle = upload_video(
                    base_url=base_url,
                    file=uploaded,
                    filename=uploaded.name,
                    content_type=uploaded.type or "video/mp4",
                )
                st.session_state.task_id = handle.task_id
                st.session_state.task_status = handle.status
                st.rerun()
            except ApiError as e:
                st.error(f"Upload failed: {e.message}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


def render_processing(base_url: str) -> None:
    task_id = st.session_state.task_id
    st.info(f"Task `{task_id}` is **{st.session_state.task_status}**")

    progress = st.progress(0, text="Processing...")
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > POLL_TIMEOUT_SECONDS:
            st.error("Polling timed out. Check API logs.")
            return

        try:
            status_data = get_task_status(base_url, task_id)
        except requests.exceptions.Timeout:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        except ApiError as e:
            st.session_state.task_error = e.message
            return

        st.session_state.task_status = status_data["status"]
        pct = min(int((elapsed / 30.0) * 95), 95)
        progress.progress(pct, text=f"Status: {status_data['status']} ({int(elapsed)}s)")

        if status_data["status"] == "completed":
            try:
                result = get_task_result(base_url, task_id)
                st.session_state.task_result = result
                progress.progress(100, text="Done")
                st.rerun()
                return
            except ApiError as e:
                st.session_state.task_error = e.message
                return

        if status_data["status"] == "failed":
            st.session_state.task_error = status_data.get("error_message", "Unknown error")
            return

        time.sleep(POLL_INTERVAL_SECONDS)


def render_results(base_url: str) -> None:
    result = st.session_state.task_result
    if not result:
        st.warning("No result available.")
        return

    st.subheader("Analysis result")

    objects = result.get("objectsDetected", [])
    total_interactions = sum(len(o.get("interactions", [])) for o in objects)
    vm = result.get("videoMetadata", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Objects detected", len(objects))
    col2.metric("Total interactions", total_interactions)
    col3.metric("Duration", f"{vm.get('durationSeconds', 0):.1f}s")

    if objects:
        st.markdown("### Objects")
        for obj in objects:
            n_motion = len(obj.get("motion_history", []))
            n_inter = len(obj.get("interactions", []))
            with st.expander(
                f"Object {obj['object_id']} — {obj['class']} "
                f"({n_motion} motion intervals, {n_inter} interactions)"
            ):
                st.json(obj)
    else:
        st.info("No objects detected in this video.")

    keyframes = result.get("keyFrames") or []
    if keyframes:
        st.markdown("### Keyframes")
        st.caption(
            f"{len(keyframes)} extracted: motion transitions and "
            "peak-interaction moments."
        )
        cols = st.columns(min(len(keyframes), 4))
        task_id = st.session_state.task_id
        for i, fname in enumerate(keyframes):
            col = cols[i % 4]
            url = f"{base_url.rstrip('/')}/tasks/{task_id}/keyframes/{fname}"
            col.image(url, caption=fname, use_container_width=True)

    with st.expander("Raw JSON response"):
        st.json(result)


def render_error() -> None:
    st.error(f"Task failed: {st.session_state.task_error}")
    if st.button("Try another video"):
        reset_task_state()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Object Identification", layout="wide")
    st.title("Object Identification in Video")
    st.caption(
        "Upload a video to detect objects, classify their motion, "
        "and identify human interactions."
    )

    init_session_state()
    base_url = render_sidebar()

    if st.session_state.task_error:
        render_error()
    elif st.session_state.task_result is not None:
        render_results(base_url)
        if st.button("Analyze another video"):
            reset_task_state()
            st.rerun()
    elif st.session_state.task_id is not None:
        render_processing(base_url)
    else:
        render_upload(base_url)


if __name__ == "__main__":
    main()
