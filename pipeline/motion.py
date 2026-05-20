"""Motion classification using centroid displacement over a sliding window.

Computes per-track motion state ("moving" | "stationary") by comparing
Euclidean centroid displacement across a configurable window of frames
against a pixel-distance threshold.
"""

from api.schemas import MotionInterval


def centroid(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (cx, cy) from (x1, y1, x2, y2)."""
    raise NotImplementedError


def classify_motion(
    centroid_history: list[tuple[float, float]],
    window: int = 15,
    threshold: float = 10.0,
) -> list[MotionInterval]:
    """Classify motion state over a list of centroids using a sliding window.

    Args:
        centroid_history: Ordered (cx, cy) per sampled frame for one track.
        window: Number of frames in each sliding window segment.
        threshold: Pixel displacement below which the object is "stationary".

    Returns:
        List of MotionInterval with merged consecutive segments of equal state.
    """
    raise NotImplementedError
