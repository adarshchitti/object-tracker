from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipeline.interaction import Interaction
from pipeline.keyframes import (
    Keyframe,
    extract_keyframes,
    find_motion_transition_frames,
    find_peak_interaction_frames,
)
from pipeline.motion import MotionInterval

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "sample_short.mp4"


# ---------------------------------------------------------------------------
# find_motion_transition_frames
# ---------------------------------------------------------------------------


def test_find_motion_transition_frames_basic() -> None:
    intervals = {
        2: [
            MotionInterval(frame_range=(0, 50), state="stationary"),
            MotionInterval(frame_range=(51, 100), state="moving"),
        ]
    }
    keyframes = find_motion_transition_frames(intervals)
    assert len(keyframes) == 1
    kf = keyframes[0]
    assert kf.frame_index == 51
    assert kf.object_id == 2
    assert kf.kind == "motion_transition"
    assert kf.filename == "obj2_motion_transition_frame51.jpg"


def test_find_motion_transition_frames_skips_initial_moving() -> None:
    intervals = {
        2: [
            MotionInterval(frame_range=(0, 50), state="moving"),
            MotionInterval(frame_range=(51, 100), state="stationary"),
        ]
    }
    assert find_motion_transition_frames(intervals) == []


def test_find_motion_transition_frames_multiple_intervals() -> None:
    intervals = {
        2: [
            MotionInterval(frame_range=(0, 20), state="stationary"),
            MotionInterval(frame_range=(21, 40), state="moving"),
            MotionInterval(frame_range=(41, 60), state="stationary"),
            MotionInterval(frame_range=(61, 80), state="moving"),
        ]
    }
    keyframes = find_motion_transition_frames(intervals)
    assert [kf.frame_index for kf in keyframes] == [21, 61]


def test_find_motion_transition_frames_empty_input() -> None:
    assert find_motion_transition_frames({}) == []


# ---------------------------------------------------------------------------
# find_peak_interaction_frames
# ---------------------------------------------------------------------------


def test_find_peak_interaction_frames_picks_smallest_distance() -> None:
    interactions = {
        2: [Interaction(interacted_by_person=99, frame_start=10, frame_end=20)]
    }
    distances = {
        10: {2: 50.0},
        15: {2: 5.0},
        20: {2: 30.0},
    }
    keyframes = find_peak_interaction_frames(interactions, distances)
    assert len(keyframes) == 1
    kf = keyframes[0]
    assert kf.frame_index == 15
    assert kf.object_id == 2
    assert kf.kind == "peak_interaction"
    assert kf.filename == "obj2_peak_interaction_person99_frame15.jpg"


def test_find_peak_interaction_frames_no_distances_in_range() -> None:
    interactions = {
        2: [Interaction(interacted_by_person=99, frame_start=10, frame_end=20)]
    }
    # Distance data exists, but for unrelated frames.
    distances = {0: {2: 1.0}, 100: {2: 2.0}}
    assert find_peak_interaction_frames(interactions, distances) == []


def test_find_peak_interaction_frames_multiple_intervals() -> None:
    interactions = {
        2: [
            Interaction(interacted_by_person=99, frame_start=10, frame_end=15),
            Interaction(interacted_by_person=99, frame_start=30, frame_end=35),
        ]
    }
    distances = {
        10: {2: 100.0},
        12: {2: 10.0},
        30: {2: 80.0},
        33: {2: 5.0},
    }
    keyframes = find_peak_interaction_frames(interactions, distances)
    assert [kf.frame_index for kf in keyframes] == [12, 33]


# ---------------------------------------------------------------------------
# extract_keyframes
# ---------------------------------------------------------------------------


def test_extract_keyframes_empty_list(tmp_path) -> None:
    out = tmp_path / "kfs"
    result = extract_keyframes("nonexistent.mp4", out, [])
    assert result == []
    assert out.is_dir()
    assert list(out.iterdir()) == []


@pytest.mark.slow
def test_extract_keyframes_writes_jpgs(tmp_path) -> None:
    if not FIXTURE_VIDEO.exists():
        pytest.skip("Fixture video missing")
    out = tmp_path / "kfs"
    keyframes = [
        Keyframe(
            frame_index=0,
            object_id=1,
            kind="motion_transition",
            filename="obj1_motion_transition_frame0.jpg",
        ),
        Keyframe(
            frame_index=10,
            object_id=1,
            kind="peak_interaction",
            filename="obj1_peak_interaction_person2_frame10.jpg",
        ),
    ]
    written = extract_keyframes(str(FIXTURE_VIDEO), out, keyframes)
    assert len(written) == 2
    for fname in written:
        path = out / fname
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.slow
def test_extract_keyframes_skips_invalid_frame_indices(tmp_path) -> None:
    if not FIXTURE_VIDEO.exists():
        pytest.skip("Fixture video missing")
    out = tmp_path / "kfs"
    keyframes = [
        Keyframe(
            frame_index=9999,
            object_id=1,
            kind="motion_transition",
            filename="obj1_motion_transition_frame9999.jpg",
        ),
    ]
    written = extract_keyframes(str(FIXTURE_VIDEO), out, keyframes)
    assert written == [] or all(
        not (out / fname).exists() or (out / fname).stat().st_size == 0
        for fname in written
    )
