"""Aggregate per-frame detection and interaction data into the final AnalysisResult schema.

Collects centroid histories, runs motion classification, merges interactions,
and assembles VideoMetadata + DetectedObject lists into AnalysisResult.
"""

from api.schemas import AnalysisResult


def build_result(
    metadata: dict,
    object_tracks: dict,
    hand_bboxes_per_frame: list,
    keyframe_paths: list[str] | None = None,
) -> AnalysisResult:
    """Build the final AnalysisResult from aggregated pipeline outputs.

    Args:
        metadata: Dict from FrameReader.metadata() (duration, fps, etc.).
        object_tracks: Mapping track_id → list of (frame_index, bbox, class_name).
        hand_bboxes_per_frame: Output from HandDetector per frame.
        keyframe_paths: Optional list of saved keyframe file paths.

    Returns:
        Fully populated AnalysisResult ready for JSON serialisation.
    """
    raise NotImplementedError
