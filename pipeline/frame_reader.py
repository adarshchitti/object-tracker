from __future__ import annotations

"""Frame extraction from video files using OpenCV."""

import os
from typing import Iterator

import cv2
import numpy as np


class FrameReader:
    """Iterates frames from a video file using OpenCV.

    Supports stride-based subsampling for performance.
    Yields (frame_index, frame_bgr) tuples where frame_index
    is the ORIGINAL index in the source video (not the post-stride index).
    """

    def __init__(self, video_path: str, stride: int = 1) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.stride = stride

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        self._metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
        }

    @property
    def metadata(self) -> dict:
        """Returns {width, height, fps, frame_count, duration_seconds}."""
        return self._metadata

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yields (original_frame_index, bgr_frame) honoring stride."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame_index % self.stride == 0:
                yield frame_index, frame
            frame_index += 1

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()
