from __future__ import annotations

"""Hand landmark detection using MediaPipe Hands.

Each frame yields 0..N HandObservation objects.  MediaPipe is initialised
once in __init__ (lazy import) and reused across calls; callers should use
the context manager to release resources cleanly.

Tradeoff: MediaPipe expects RGB input, so each frame is converted from BGR
inside detect().  The overhead is negligible compared with landmark inference.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HandObservation:
    """A detected hand in a single frame.

    bbox is the axis-aligned bounding box of all 21 landmarks in pixel coords.
    center is the wrist landmark (landmark 0), the most stable anchor point.
    handedness is 'Left' or 'Right' as reported by MediaPipe (may be inaccurate
    in selfie vs. non-selfie context; we report it but don't rely on it for
    interaction logic).
    """

    frame_index: int
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    center: tuple[float, float]                # wrist landmark in pixels
    confidence: float                          # detection confidence
    handedness: str                            # "Left" or "Right"


class HandDetector:
    """Wraps MediaPipe Hands for per-frame hand landmark detection.

    Each detect() call processes a single BGR frame and returns 0..N hand
    observations.  MediaPipe is stateless across calls in our usage; we
    initialise once and reuse the solution object.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Lazy import: keeps module-level import fast and avoids loading
        # MediaPipe when only HandObservation is needed (e.g. in tests).
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,  # video mode: faster & more temporally stable
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_index: int, frame_bgr: np.ndarray) -> list[HandObservation]:
        """Run hand detection on a single BGR frame.

        Returns 0..max_num_hands HandObservation objects.
        Empty list when no hands are detected.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if results.multi_hand_landmarks is None:
            return []

        observations: list[HandObservation] = []
        for hand_lm, handedness_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            lm = hand_lm.landmark
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]

            bbox = (min(xs), min(ys), max(xs), max(ys))
            # Wrist (landmark 0) is the most stable anchor for proximity checks
            center = (lm[0].x * w, lm[0].y * h)

            label = handedness_info.classification[0].label
            score = float(handedness_info.classification[0].score)

            observations.append(
                HandObservation(
                    frame_index=frame_index,
                    bbox=bbox,
                    center=center,
                    confidence=score,
                    handedness=label,
                )
            )

        return observations

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
