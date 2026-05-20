"""YOLOv8 object detection with ByteTrack multi-object tracking.

Uses ultralytics.YOLO("yolov8n.pt").track(persist=True, tracker="bytetrack.yaml")
to produce per-frame detections with stable track IDs across frames.
Weights are downloaded automatically on first use (~6 MB for yolov8n).
"""

from typing import Any

import numpy as np


class Detection:
    """Single object detection with tracking ID."""

    track_id: int
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class YoloDetector:
    """Wrap YOLOv8n + ByteTrack to yield detections per frame."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.3) -> None:
        """
        Args:
            model_path: Local path or ultralytics model name (auto-downloads).
            conf: Minimum confidence threshold.
        """
        self.model_path = model_path
        self.conf = conf
        self._model: Any = None  # ultralytics.YOLO instance, lazy-loaded

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run tracking on a single BGR frame and return detections.

        Calls model.track(frame, persist=True, conf=self.conf, tracker='bytetrack.yaml').
        """
        raise NotImplementedError
