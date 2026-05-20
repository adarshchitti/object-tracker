"""Hand-object interaction detection via bounding-box proximity.

An interaction is recorded when the Euclidean distance between a hand bbox
centroid and an object bbox centroid falls below distance_threshold pixels
for at least one frame. Person ID is derived from hand track index.
"""

from api.schemas import Interaction


def bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (cx, cy) from (x1, y1, x2, y2)."""
    raise NotImplementedError


def detect_interactions(
    hand_bboxes_per_frame: list[list[tuple[float, float, float, float]]],
    object_tracks: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    distance_threshold: float = 50.0,
) -> list[Interaction]:
    """Detect hand-object interactions across all frames.

    Args:
        hand_bboxes_per_frame: List of hand bbox lists, indexed by frame number.
        object_tracks: Mapping of track_id → [(frame_index, bbox_xyxy), ...].
        distance_threshold: Max centroid distance (pixels) to count as interaction.

    Returns:
        List of Interaction records with person index and frame range.
    """
    raise NotImplementedError
