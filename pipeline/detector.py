from __future__ import annotations

"""YOLOv8 object detection with ByteTrack multi-object tracking."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detection:
    track_id: int           # persistent ID from ByteTrack
    class_id: int           # COCO class index
    class_name: str         # COCO class name
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


class YoloDetector:
    """Wraps Ultralytics YOLOv8 with built-in ByteTrack tracking.

    Uses yolov8n.pt by default (nano, fast on CPU). Weights download
    on first instantiation to ~/.cache or project root.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model: Any = None  # ultralytics.YOLO instance, lazy-loaded

    def _load_model(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection + tracking on a single frame.

        Calls model.track(frame, persist=True, tracker='bytetrack.yaml').
        Returns list of Detection objects with stable track_ids across calls.
        Filters by confidence_threshold.
        """
        if self._model is None:
            self._load_model()

        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            conf=self.confidence_threshold,
            device=self.device,
        )

        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        names = results[0].names

        for i in range(len(boxes)):
            # Ultralytics sometimes returns None track IDs in the first frame
            # or when a track is lost — skip those detections entirely.
            if boxes.id is None:
                continue
            track_id_val = boxes.id[i]
            if track_id_val is None:
                continue

            conf = float(boxes.conf[i])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(boxes.cls[i])
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])

            detections.append(
                Detection(
                    track_id=int(track_id_val),
                    class_id=cls_id,
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections

    def reset_tracker(self) -> None:
        """Reset tracking state. Call between different videos."""
        if self._model is not None:
            # Re-instantiate to clear ByteTrack's internal state
            self._load_model()
