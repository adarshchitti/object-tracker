"""Hand landmark detection using MediaPipe Hands.

Detects hand bounding boxes from MediaPipe 21-landmark output.
Each detected hand is represented as a bounding box in pixel coordinates,
used downstream for hand-object proximity/interaction detection.
"""

import numpy as np


class HandDetector:
    """Wrap MediaPipe Hands to extract hand bounding boxes per frame."""

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5) -> None:
        """
        Args:
            max_hands: Maximum number of hands to detect per frame.
            min_detection_confidence: MediaPipe confidence threshold.
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self._hands = None  # mediapipe.solutions.hands.Hands instance, lazy-loaded

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Return list of (x1, y1, x2, y2) bboxes for each detected hand in the BGR frame.

        Converts MediaPipe normalized landmarks to absolute pixel coordinates,
        then computes the bounding box as min/max of landmark x/y values.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release MediaPipe resources."""
        raise NotImplementedError
