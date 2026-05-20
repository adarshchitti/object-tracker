"""Filesystem layout for uploaded videos and task artefacts."""

from __future__ import annotations

import os
from pathlib import Path

STORAGE_ROOT = Path("storage")
TASKS_ROOT = STORAGE_ROOT / "tasks"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def ensure_storage_root() -> None:
    """Create storage/tasks/ if missing. Called on app startup."""
    TASKS_ROOT.mkdir(parents=True, exist_ok=True)


def task_directory(task_id: str) -> Path:
    """Returns the directory for a task. Does NOT create it."""
    return TASKS_ROOT / task_id


def create_task_directory(task_id: str) -> Path:
    """Creates and returns the directory for a new task."""
    path = task_directory(task_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def input_video_path(task_id: str, extension: str) -> Path:
    """Returns the expected path to the task's input video.

    extension includes the leading dot (e.g., '.mp4').
    """
    return task_directory(task_id) / f"input{extension}"


def is_allowed_extension(filename: str | None) -> bool:
    """Returns True if the filename has an allowed video extension."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS
