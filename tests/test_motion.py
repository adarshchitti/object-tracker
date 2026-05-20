from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.motion import (
    CentroidObservation,
    MotionInterval,
    bbox_to_centroid,
    classify_motion,
    classify_per_observation,
    collapse_to_intervals,
    compute_displacements,
    smooth_states,
)


# ---------------------------------------------------------------------------
# bbox_to_centroid
# ---------------------------------------------------------------------------


def test_bbox_to_centroid() -> None:
    cx, cy = bbox_to_centroid((0.0, 0.0, 100.0, 200.0))
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(100.0)


def test_bbox_to_centroid_non_origin() -> None:
    cx, cy = bbox_to_centroid((10.0, 20.0, 30.0, 60.0))
    assert cx == pytest.approx(20.0)
    assert cy == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# compute_displacements
# ---------------------------------------------------------------------------


def _obs(frame_index: int, cx: float, cy: float) -> CentroidObservation:
    return CentroidObservation(frame_index, cx, cy)


def test_compute_displacements_zero_for_initial_window() -> None:
    observations = [_obs(i, float(i * 100), 0.0) for i in range(10)]
    window = 3
    result = compute_displacements(observations, window=window)
    assert result.shape == (10,)
    assert np.all(result[:window] == 0.0)


def test_compute_displacements_static_object() -> None:
    observations = [_obs(i, 100.0, 100.0) for i in range(15)]
    result = compute_displacements(observations, window=5)
    assert np.allclose(result, 0.0)


def test_compute_displacements_linear_motion() -> None:
    # Centroids at (0,0), (10,0), (20,0), (30,0) — window=2
    observations = [_obs(i, float(i * 10), 0.0) for i in range(4)]
    result = compute_displacements(observations, window=2)
    # First two are 0; index 2: ||(20,0)-(0,0)|| = 20; index 3: ||(30,0)-(10,0)|| = 20
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(20.0)
    assert result[3] == pytest.approx(20.0)


def test_compute_displacements_window_larger_than_sequence() -> None:
    observations = [_obs(i, float(i * 50), 0.0) for i in range(3)]
    result = compute_displacements(observations, window=5)
    assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# classify_per_observation
# ---------------------------------------------------------------------------


def test_classify_per_observation_threshold() -> None:
    displacements = np.array([0.0, 0.0, 5.0, 20.0, 30.0])
    result = classify_per_observation(displacements, threshold=10.0)
    assert result == ["stationary", "stationary", "stationary", "moving", "moving"]


def test_classify_per_observation_exactly_at_threshold_is_stationary() -> None:
    # Boundary: > threshold is moving, == threshold is stationary
    result = classify_per_observation(np.array([10.0, 10.001]), threshold=10.0)
    assert result == ["stationary", "moving"]


# ---------------------------------------------------------------------------
# smooth_states
# ---------------------------------------------------------------------------


def test_smooth_states_removes_single_flicker() -> None:
    states = ["stationary"] * 3 + ["moving"] + ["stationary"] * 3
    result = smooth_states(states, min_run_length=3)
    assert result == ["stationary"] * 7


def test_smooth_states_removes_two_frame_flicker() -> None:
    states = ["stationary"] * 4 + ["moving"] * 2 + ["stationary"] * 4
    result = smooth_states(states, min_run_length=3)
    assert result == ["stationary"] * 10


def test_smooth_states_keeps_real_transition() -> None:
    states = ["stationary"] * 5 + ["moving"] * 5
    result = smooth_states(states, min_run_length=3)
    assert result == states  # unchanged — no middle run


def test_smooth_states_leaves_edge_runs_alone() -> None:
    # Single 'moving' at the start — no left neighbor, must not be flipped
    states = ["moving"] + ["stationary"] * 10
    result = smooth_states(states, min_run_length=3)
    assert result == states


def test_smooth_states_empty() -> None:
    assert smooth_states([]) == []


def test_smooth_states_single_element() -> None:
    assert smooth_states(["moving"]) == ["moving"]


# ---------------------------------------------------------------------------
# collapse_to_intervals
# ---------------------------------------------------------------------------


def test_collapse_to_intervals_basic() -> None:
    observations = [_obs(0, 0, 0), _obs(1, 0, 0), _obs(2, 10, 0), _obs(3, 20, 0)]
    states = ["stationary", "stationary", "moving", "moving"]
    result = collapse_to_intervals(observations, states)
    assert len(result) == 2
    assert result[0] == MotionInterval(frame_range=(0, 1), state="stationary")
    assert result[1] == MotionInterval(frame_range=(2, 3), state="moving")


def test_collapse_to_intervals_uses_original_frame_indices() -> None:
    # stride=5: frames 0, 5, 10, 15 — all stationary
    observations = [_obs(i * 5, 0.0, 0.0) for i in range(4)]
    states = ["stationary"] * 4
    result = collapse_to_intervals(observations, states)
    assert len(result) == 1
    assert result[0].frame_range == (0, 15)
    assert result[0].state == "stationary"


def test_collapse_to_intervals_empty() -> None:
    assert collapse_to_intervals([], []) == []


def test_collapse_to_intervals_single_observation() -> None:
    result = collapse_to_intervals([_obs(7, 0, 0)], ["moving"])
    assert result == [MotionInterval(frame_range=(7, 7), state="moving")]


# ---------------------------------------------------------------------------
# classify_motion (end-to-end)
# ---------------------------------------------------------------------------


def test_classify_motion_empty_observations() -> None:
    assert classify_motion([]) == []


def test_classify_motion_too_few_observations() -> None:
    # 2 observations, window=5 → insufficient history → single 'stationary'
    observations = [_obs(0, 0.0, 0.0), _obs(1, 100.0, 0.0)]
    result = classify_motion(observations, window=5)
    assert len(result) == 1
    assert result[0].state == "stationary"
    assert result[0].frame_range == (0, 1)


def test_classify_motion_end_to_end_stationary() -> None:
    observations = [_obs(i, 100.0, 100.0) for i in range(30)]
    result = classify_motion(observations, window=5, threshold_pixels=15.0)
    assert len(result) == 1
    assert result[0].state == "stationary"
    assert result[0].frame_range == (0, 29)


def test_classify_motion_end_to_end_moving() -> None:
    # 20-pixel step per frame — well above threshold=15
    observations = [_obs(i, i * 20.0, 0.0) for i in range(30)]
    result = classify_motion(observations, window=5, threshold_pixels=15.0)
    states = [iv.state for iv in result]
    assert "moving" in states
    assert result[-1].state == "moving"


def test_classify_motion_transition() -> None:
    # 15 stationary frames then 15 clearly moving frames → exactly 2 intervals
    obs_still = [_obs(i, 0.0, 0.0) for i in range(15)]
    obs_move = [_obs(i + 15, (i + 1) * 50.0, 0.0) for i in range(15)]
    observations = obs_still + obs_move
    result = classify_motion(observations, window=5, threshold_pixels=15.0)
    assert len(result) == 2
    assert result[0].state == "stationary"
    assert result[1].state == "moving"


def test_classify_motion_resolution_normalization() -> None:
    """Same fractional motion relative to frame diagonal produces same intervals."""
    diag_small = math.sqrt(640**2 + 480**2)   # ~800
    diag_large = math.sqrt(1920**2 + 1080**2)  # ~2203

    frac_threshold = 0.02   # 2% of diagonal
    frac_step = 0.03        # 3% of diagonal per observation step (above threshold)

    obs_small = [_obs(i, i * frac_step * diag_small, 0.0) for i in range(20)]
    obs_large = [_obs(i, i * frac_step * diag_large, 0.0) for i in range(20)]

    result_small = classify_motion(
        obs_small, window=5, threshold_pixels=frac_threshold, frame_diagonal=diag_small
    )
    result_large = classify_motion(
        obs_large, window=5, threshold_pixels=frac_threshold, frame_diagonal=diag_large
    )

    assert len(result_small) == len(result_large)
    for a, b in zip(result_small, result_large):
        assert a.state == b.state
