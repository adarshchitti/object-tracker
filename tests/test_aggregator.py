from __future__ import annotations

import json

import pytest

from api.schemas import AnalysisResult, DetectedObject
from pipeline.aggregator import (
    build_analysis_result,
    build_centroid_trails,
    build_detected_objects,
    filter_ephemeral_tracks,
    resolve_class_name,
)
from pipeline.detector import Detection
from pipeline.interaction import Interaction as PipelineInteraction
from pipeline.motion import CentroidObservation

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

METADATA = {
    "width": 640,
    "height": 480,
    "fps": 30.0,
    "frame_count": 60,
    "duration_seconds": 2.0,
}


def _det(
    track_id: int,
    class_name: str = "laptop",
    bbox: tuple = (0.0, 0.0, 100.0, 100.0),
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        track_id=track_id,
        class_id=63,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
    )


def _person(track_id: int = 99) -> Detection:
    return Detection(
        track_id=track_id, class_id=0, class_name="person",
        confidence=0.9, bbox=(300.0, 100.0, 400.0, 300.0),
    )


def _frames(track_id: int, n: int, class_name: str = "laptop") -> dict[int, list[Detection]]:
    """n frames, all with the same stationary detection for track_id."""
    return {i: [_det(track_id, class_name)] for i in range(n)}


# ---------------------------------------------------------------------------
# build_centroid_trails
# ---------------------------------------------------------------------------


def test_build_centroid_trails_groups_by_track_id() -> None:
    per_frame = {
        0: [_det(1, bbox=(0.0, 0.0, 100.0, 100.0)), _det(2, bbox=(200.0, 200.0, 300.0, 300.0))],
        1: [_det(1, bbox=(10.0, 0.0, 110.0, 100.0)), _det(2, bbox=(200.0, 200.0, 300.0, 300.0))],
        2: [_det(1, bbox=(20.0, 0.0, 120.0, 100.0))],
    }
    trails = build_centroid_trails(per_frame)

    assert set(trails.keys()) == {1, 2}
    assert len(trails[1]) == 3
    assert len(trails[2]) == 2
    # Sorted by frame_index
    assert trails[1][0].frame_index == 0
    assert trails[1][2].frame_index == 2


def test_build_centroid_trails_skips_person() -> None:
    per_frame = {
        0: [_det(1), _person(99)],
        1: [_det(1), _person(99)],
    }
    trails = build_centroid_trails(per_frame)
    assert 99 not in trails
    assert 1 in trails


def test_build_centroid_trails_empty_input() -> None:
    assert build_centroid_trails({}) == {}


def test_build_centroid_trails_centroid_values() -> None:
    per_frame = {0: [_det(1, bbox=(0.0, 0.0, 100.0, 80.0))]}
    trails = build_centroid_trails(per_frame)
    obs = trails[1][0]
    assert obs.cx == pytest.approx(50.0)
    assert obs.cy == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# filter_ephemeral_tracks
# ---------------------------------------------------------------------------


def test_filter_ephemeral_tracks_drops_short_runs() -> None:
    # Track 1: 1 frame; Track 2: 10 frames; min_frames=3 → only track 2 survives
    per_frame = {**_frames(2, 10), 0: [_det(2), _det(1)]}
    trails = build_centroid_trails(per_frame)

    filtered_trails, _ = filter_ephemeral_tracks(trails, per_frame, min_frames=3)

    assert 2 in filtered_trails
    assert 1 not in filtered_trails


def test_filter_ephemeral_tracks_filters_both_dicts() -> None:
    # Frame 0: both track 1 and track 2. Frames 1-9: only track 1.
    per_frame: dict[int, list[Detection]] = {i: [_det(1)] for i in range(10)}
    per_frame[0] = [_det(1), _det(2, class_name="cup")]  # track 2 in one frame only

    trails = build_centroid_trails(per_frame)
    filtered_trails, filtered_objects = filter_ephemeral_tracks(trails, per_frame, min_frames=3)

    # Track 2 dropped from both
    assert 2 not in filtered_trails
    assert all(det.track_id != 2 for det in filtered_objects.get(0, []))
    # Track 1 kept in both
    assert 1 in filtered_trails
    assert any(det.track_id == 1 for det in filtered_objects[0])


def test_filter_ephemeral_tracks_keeps_persons() -> None:
    # Persons are not in centroid_trails but must survive in filtered_objects
    per_frame = {i: [_det(1), _person()] for i in range(10)}
    per_frame[0] = [_det(1), _person(), _det(2, "cup")]  # track 2: ephemeral

    trails = build_centroid_trails(per_frame)
    _, filtered_objects = filter_ephemeral_tracks(trails, per_frame, min_frames=3)

    # Person survives even though it's not a tracked non-person object
    assert any(det.class_name == "person" for det in filtered_objects[0])


# ---------------------------------------------------------------------------
# resolve_class_name
# ---------------------------------------------------------------------------


def test_resolve_class_name_picks_most_frequent() -> None:
    per_frame = {
        0: [_det(1, "laptop")],
        1: [_det(1, "laptop")],
        2: [_det(1, "tv")],
        3: [_det(1, "laptop")],
    }
    assert resolve_class_name(1, per_frame) == "laptop"


def test_resolve_class_name_handles_single_appearance() -> None:
    per_frame = {0: [_det(1, "cup")]}
    assert resolve_class_name(1, per_frame) == "cup"


def test_resolve_class_name_returns_unknown_for_missing_track() -> None:
    assert resolve_class_name(999, {0: [_det(1)]}) == "unknown"


# ---------------------------------------------------------------------------
# build_detected_objects
# ---------------------------------------------------------------------------


def test_build_detected_objects_produces_schema_shape() -> None:
    trails = {1: [CentroidObservation(i, 50.0, 50.0) for i in range(10)]}
    per_frame = _frames(1, 10)

    result = build_detected_objects(trails, per_frame, {}, frame_diagonal=800.0)

    assert isinstance(result, list)
    assert len(result) == 1
    obj = result[0]
    assert isinstance(obj, DetectedObject)
    assert obj.object_id == 1
    assert obj.class_ == "laptop"
    assert isinstance(obj.motion_history, list)
    assert isinstance(obj.interactions, list)


def test_build_detected_objects_includes_interactions() -> None:
    trails = {1: [CentroidObservation(i, 50.0, 50.0) for i in range(10)]}
    per_frame = _frames(1, 10)
    interactions = {1: [PipelineInteraction(interacted_by_person=99, frame_start=3, frame_end=7)]}

    result = build_detected_objects(trails, per_frame, interactions, frame_diagonal=800.0)

    assert len(result[0].interactions) == 1
    inter = result[0].interactions[0]
    assert inter.interacted_by_person == 99
    assert inter.frame_start == 3
    assert inter.frame_end == 7


def test_build_detected_objects_empty_interactions_when_none() -> None:
    trails = {1: [CentroidObservation(i, 50.0, 50.0) for i in range(10)]}
    per_frame = _frames(1, 10)

    result = build_detected_objects(trails, per_frame, {}, frame_diagonal=800.0)
    assert result[0].interactions == []


def test_build_detected_objects_sorted_by_track_id() -> None:
    trails = {
        5: [CentroidObservation(i, 50.0, 50.0) for i in range(5)],
        1: [CentroidObservation(i, 50.0, 50.0) for i in range(5)],
        3: [CentroidObservation(i, 50.0, 50.0) for i in range(5)],
    }
    per_frame = {
        i: [_det(1), _det(3), _det(5)] for i in range(5)
    }

    result = build_detected_objects(trails, per_frame, {}, frame_diagonal=800.0)
    assert [obj.object_id for obj in result] == [1, 3, 5]


# ---------------------------------------------------------------------------
# build_analysis_result
# ---------------------------------------------------------------------------


def test_build_analysis_result_produces_video_metadata() -> None:
    result = build_analysis_result(METADATA, {}, {}, {})
    vm = result.video_metadata
    assert vm.width == 640
    assert vm.height == 480
    assert vm.fps == pytest.approx(30.0)
    assert vm.frame_count == 60
    assert vm.duration_seconds == pytest.approx(2.0)


def test_build_analysis_result_empty_video() -> None:
    result = build_analysis_result(METADATA, {}, {}, {})
    assert isinstance(result, AnalysisResult)
    assert result.objects_detected == []


def test_build_analysis_result_handles_person_only() -> None:
    per_frame = {i: [_person()] for i in range(10)}
    result = build_analysis_result(METADATA, per_frame, {}, {})
    assert result.objects_detected == []


def test_build_analysis_result_filters_ephemeral_tracks() -> None:
    # Track 1: 1 frame (ephemeral). Track 2: 10 frames (kept). min_track_frames=3
    per_frame = {i: [_det(2)] for i in range(10)}
    per_frame[0] = [_det(2), _det(1, "cup")]  # track 1 only in frame 0

    result = build_analysis_result(METADATA, per_frame, {}, {}, min_track_frames=3)

    track_ids = [obj.object_id for obj in result.objects_detected]
    assert 2 in track_ids
    assert 1 not in track_ids


def test_build_analysis_result_serializes_with_camelcase_aliases() -> None:
    per_frame = {i: [_det(1)] for i in range(10)}
    result = build_analysis_result(METADATA, per_frame, {}, {}, min_track_frames=3)

    data = json.loads(result.model_dump_json(by_alias=True))

    assert "videoMetadata" in data
    assert "objectsDetected" in data
    assert "video_metadata" not in data

    if data["objectsDetected"]:
        obj = data["objectsDetected"][0]
        assert "class" in obj
        assert "class_" not in obj


def test_video_metadata_serializes_camelcase() -> None:
    per_frame = {i: [_det(1)] for i in range(10)}
    result = build_analysis_result(METADATA, per_frame, {}, {}, min_track_frames=3)

    data = json.loads(result.model_dump_json(by_alias=True))
    vm = data["videoMetadata"]
    assert "durationSeconds" in vm
    assert "frameCount" in vm
    assert "duration_seconds" not in vm
    assert "frame_count" not in vm


def test_build_analysis_result_includes_keyframe_filenames() -> None:
    per_frame = {i: [_det(1)] for i in range(10)}
    filenames = ["obj1_motion_transition_frame5.jpg", "obj1_peak_interaction_person2_frame8.jpg"]

    result = build_analysis_result(
        METADATA, per_frame, {}, {},
        keyframe_filenames=filenames,
        min_track_frames=3,
    )

    assert result.key_frames == filenames

    data = json.loads(result.model_dump_json(by_alias=True))
    assert data["keyFrames"] == filenames
