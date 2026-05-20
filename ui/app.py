"""Streamlit UI shell for the Object Identification service."""

import os

import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.title("Object Identification in Video")
st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
st.info("Upload a video to begin analysis")
