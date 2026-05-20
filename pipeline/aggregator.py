from __future__ import annotations

"""Aggregate per-frame detection data into the final AnalysisResult schema.

Pure module: no I/O, no model calls. Takes already-collected data and
assembles the schema objects.
"""

import math
from collections import defaultdict

from pipeline.detector import Detection
from pipeline.interaction import Interaction as PipelineInteraction
from pipeline.motion import (
    CentroidObservation,
    MotionInterval as PipelineMotionInterval,
    bbox_to_centroid,
    classify_motion,
)
from api.schemas import (
    AnalysisResult,
    DetectedObject,
    Interaction as SchemaInteraction,
    MotionInterval as SchemaMotionInterval,
    VideoMetadata,
)


def build_centroid_trails(
    per_frame_objects: dict[int, list[Detection]],
) -> dict[int, list[CentroidObservation]]:
    """For each non-person track_id, build an ordered list of CentroidObservations.

    Iterates frames in sorted order so the resulting lists are sorted by
    frame_index. Persons are skipped — their motion is not part of the schema.
    """
    trails: dict[int, list[CentroidObservation]] = defaultdict(list)
    for frame_index in sorted(per_frame_objects.keys()):
        for det in per_frame_objects[frame_index]:
            if det.class_name == "person":
                continue
            cx, cy = bbox_to_centroid(det.bbox)
            trails[det.track_id].append(CentroidObservation(frame_index, cx, cy))
    return dict(trails)


def filter_ephemeral_tracks(
    centroid_trails: dict[int, list[CentroidObservation]],
    per_frame_objects: dict[int, list[Detection]],
    min_frames: int = 3,
) -> tuple[dict[int, list[CentroidObservation]], dict[int, list[Detection]]]:
    """Drop object tracks that appear in fewer than min_frames sampled frames.

    Suppresses single-frame false positives. Persons are kept in
    per_frame_objects regardless (needed by interaction detection downstream).
    Returns (filtered_trails, filtered_objects) with consistent track_id sets.
    """
    valid_ids = {
        tid for tid, trail in centroid_trails.items() if len(trail) >= min_frames
    }
    filtered_trails = {tid: trail for tid, trail in centroid_trails.items() if tid in valid_ids}

    filtered_objects: dict[int, list[Detection]] = {}
    for fi, dets in per_frame_objects.items():
        kept = [
            det for det in dets if det.class_name == "person" or det.track_id in valid_ids
        ]
        if kept:
            filtered_objects[fi] = kept

    return filtered_trails, filtered_objects


def resolve_class_name(
    track_id: int,
    per_frame_objects: dict[int, list[Detection]],
) -> str:
    """Return the most frequent class_name for track_id across all frames.

    Handles occasional label flips on low-confidence frames by majority vote.
    Returns 'unknown' if the track_id is not found.
    """
    counts: dict[str, int] = defaultdict(int)
    for dets in per_frame_objects.values():
        for det in dets:
            if det.track_id == track_id:
                counts[det.class_name] += 1
    if not counts:
        return "unknown"
    return max(counts, key=lambda k: counts[k])


def compute_motion_intervals_by_object(
    centroid_trails: dict[int, list[CentroidObservation]],
    frame_diagonal: float,
    motion_window: int = 5,
    motion_threshold_fraction: float = 0.02,
) -> dict[int, list[PipelineMotionInterval]]:
    """Run motion classification for every tracked object.

    motion_threshold_fraction is multiplied by frame_diagonal inside
    classify_motion to give a resolution-independent pixel threshold.

    Shared by build_detected_objects (for the schema response) and the
    orchestrator's keyframe extraction step (for motion-transition frames).
    """
    return {
        track_id: classify_motion(
            trail,
            window=motion_window,
            threshold_pixels=motion_threshold_fraction,
            frame_diagonal=frame_diagonal,
        )
        for track_id, trail in centroid_trails.items()
    }


def build_detected_objects(
    centroid_trails: dict[int, list[CentroidObservation]],
    per_frame_objects: dict[int, list[Detection]],
    interactions_by_object: dict[int, list[PipelineInteraction]],
    frame_diagonal: float,
    motion_window: int = 5,
    motion_threshold_fraction: float = 0.02,
) -> list[DetectedObject]:
    """Build list[DetectedObject] from centroid trails and interactions.

    Sorted by track_id for stable output.
    """
    motion_intervals_by_object = compute_motion_intervals_by_object(
        centroid_trails, frame_diagonal, motion_window, motion_threshold_fraction
    )

    result: list[DetectedObject] = []
    for track_id in sorted(centroid_trails.keys()):
        class_name = resolve_class_name(track_id, per_frame_objects)

        motion_intervals = motion_intervals_by_object.get(track_id, [])
        schema_motion = [
            SchemaMotionInterval(frame_range=iv.frame_range, state=iv.state)
            for iv in motion_intervals
        ]

        raw_interactions = interactions_by_object.get(track_id, [])
        schema_interactions = [
            SchemaInteraction(
                interacted_by_person=inter.interacted_by_person,
                frame_start=inter.frame_start,
                frame_end=inter.frame_end,
            )
            for inter in raw_interactions
        ]

        result.append(
            DetectedObject(
                object_id=track_id,
                class_=class_name,
                motion_history=schema_motion,
                interactions=schema_interactions,
            )
        )
    return result


def build_analysis_result(
    video_metadata_dict: dict,
    per_frame_objects: dict[int, list[Detection]],
    per_frame_hands: dict[int, list],
    interactions_by_object: dict[int, list[PipelineInteraction]],
    keyframe_filenames: list[str] | None = None,
    min_track_frames: int = 5,
    motion_window: int = 5,
    motion_threshold_fraction: float = 0.02,
) -> AnalysisResult:
    """Top-level aggregator: build the final AnalysisResult from collected data.

    video_metadata_dict comes directly from FrameReader.metadata.
    Ephemeral tracks (appearing in fewer than min_track_frames frames) are
    dropped before motion classification to suppress false positives.
    """
    width: int = video_metadata_dict["width"]
    height: int = video_metadata_dict["height"]
    frame_diagonal = math.sqrt(width**2 + height**2)

    video_metadata = VideoMetadata(
        duration_seconds=video_metadata_dict["duration_seconds"],
        frame_count=video_metadata_dict["frame_count"],
        width=width,
        height=height,
        fps=video_metadata_dict["fps"],
    )

    centroid_trails = build_centroid_trails(per_frame_objects)
    # Diagnostic data showed phantom tracks (e.g. car, bottle) at ~3 frames vs
    # real tracks at 21+; threshold of 5 sits in the gap and confidence didn't
    # separate the classes.
    centroid_trails, filtered_objects = filter_ephemeral_tracks(
        centroid_trails, per_frame_objects, min_track_frames
    )

    objects_detected = build_detected_objects(
        centroid_trails=centroid_trails,
        per_frame_objects=filtered_objects,
        interactions_by_object=interactions_by_object,
        frame_diagonal=frame_diagonal,
        motion_window=motion_window,
        motion_threshold_fraction=motion_threshold_fraction,
    )

    result = AnalysisResult(
        video_metadata=video_metadata,
        objects_detected=objects_detected,
    )
    if keyframe_filenames:
        result.key_frames = keyframe_filenames
    return result
