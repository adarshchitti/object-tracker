"""Frame extraction from video files using OpenCV.

Iterates video frames with a configurable stride to reduce compute load.
Yields (frame_index, numpy_array) tuples in BGR format.
"""

from typing import Generator

import numpy as np


class FrameReader:
    """Read frames from a video file with optional stride (frame-skip)."""

    def __init__(self, video_path: str, stride: int = 1) -> None:
        """
        Args:
            video_path: Absolute path to the video file.
            stride: Yield every Nth frame. stride=1 means every frame.
        """
        self.video_path = video_path
        self.stride = stride

    def __iter__(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_bgr) for each sampled frame."""
        raise NotImplementedError

    def metadata(self) -> dict:
        """Return dict with duration_seconds, frame_count, width, height, fps."""
        raise NotImplementedError
