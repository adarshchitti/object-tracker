from __future__ import annotations

"""Hand landmark detection using MediaPipe Hands (Tasks API, v0.10+).

MediaPipe 0.10 removed the legacy solutions API.  HandDetector now uses
mediapipe.tasks.python.vision.HandLandmarker.  The model file
(hand_landmarker.task, ~7 MB) is downloaded automatically to
~/.cache/mediapipe/ on first instantiation.

Tradeoff: MediaPipe expects RGB input, so each frame is converted from BGR
inside detect().  The overhead is negligible compared with landmark inference.
"""

from dataclasses import dataclass

import cv2
import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"


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


def _ensure_model() -> str:
    """Return the local path to hand_landmarker.task, downloading if absent."""
    import os
    import urllib.request

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, _MODEL_FILENAME)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


class HandDetector:
    """Wraps MediaPipe HandLandmarker (Tasks API) for per-frame hand detection.

    Each detect() call processes a single BGR frame and returns 0..N hand
    observations.  We use IMAGE running mode (stateless per frame).
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Lazy imports: keep module-level import fast; also avoids loading
        # MediaPipe when only HandObservation is needed (e.g. unit tests).
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_index: int, frame_bgr: np.ndarray) -> list[HandObservation]:
        """Run hand detection on a single BGR frame.

        Returns 0..max_num_hands HandObservation objects.
        Empty list when no hands are detected.
        """
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        observations: list[HandObservation] = []
        for hand_lm, handedness_cats in zip(result.hand_landmarks, result.handedness):
            xs = [lm.x * w for lm in hand_lm]
            ys = [lm.y * h for lm in hand_lm]

            bbox = (min(xs), min(ys), max(xs), max(ys))
            # Wrist (landmark 0) is the most stable anchor for proximity checks
            center = (hand_lm[0].x * w, hand_lm[0].y * h)
            label = handedness_cats[0].category_name
            score = float(handedness_cats[0].score)

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
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
