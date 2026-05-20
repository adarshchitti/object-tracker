from __future__ import annotations

"""Motion classification using centroid displacement over a sliding window.

Pure functions: no I/O, no global state. The main entry point is
`classify_motion`; everything else is a composable helper.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentroidObservation:
    """A single observation of an object's centroid at a given frame."""

    frame_index: int
    cx: float
    cy: float


@dataclass(frozen=True)
class MotionInterval:
    """A continuous range of frames where an object had the same motion state.

    frame_range is (inclusive_start, inclusive_end) using ORIGINAL source frame
    indices (as returned by FrameReader), not post-stride observation indices.
    With stride > 1, consecutive intervals may have gaps in the source timeline.
    """

    frame_range: tuple[int, int]
    state: str  # "moving" or "stationary"


def bbox_to_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Convert (x1, y1, x2, y2) to (cx, cy)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def compute_displacements(
    observations: list[CentroidObservation],
    window: int = 5,
) -> np.ndarray:
    """Return per-observation Euclidean displacement over a sliding window.

    For observation i, the displacement is the L2 distance between
    observations[i] and observations[i - window]. The first `window` values
    are 0.0 because there is insufficient history.

    Window is measured in observation indices, not source frame indices.
    With stride-based sampling (e.g. stride=3, window=5), the window spans
    5 sampled frames = 15 source frames. This makes the classifier invariant
    to stride choice: the same real-world motion produces the same displacement
    regardless of how densely the video is sampled.
    """
    n = len(observations)
    displacements = np.zeros(n, dtype=float)
    if n <= window:
        return displacements

    centroids = np.array([(obs.cx, obs.cy) for obs in observations], dtype=float)
    deltas = centroids[window:] - centroids[: n - window]
    displacements[window:] = np.linalg.norm(deltas, axis=1)
    return displacements


def classify_per_observation(
    displacements: np.ndarray,
    threshold: float,
) -> list[str]:
    """Map each displacement to 'moving' or 'stationary'.

    'moving' when displacement > threshold, otherwise 'stationary'.
    """
    return ["moving" if d > threshold else "stationary" for d in displacements]


def smooth_states(states: list[str], min_run_length: int = 3) -> list[str]:
    """Suppress short flicker runs (single pass).

    A run of identical states shorter than min_run_length that is sandwiched
    between two runs of the same opposite state is flipped to that state.
    Runs at the start or end of the sequence are left untouched because there
    is no neighbor on both sides to confirm they are noise.

    Example (min_run_length=3):
        ['s','s','s','m','s','s','s'] -> ['s','s','s','s','s','s','s']
    """
    if not states:
        return []

    # Build runs: list of (state, inclusive_start, inclusive_end)
    runs: list[tuple[str, int, int]] = []
    run_start = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            runs.append((states[run_start], run_start, i - 1))
            run_start = i
    runs.append((states[run_start], run_start, len(states) - 1))

    if len(runs) < 3:
        return list(states)

    result = list(states)
    for i in range(1, len(runs) - 1):
        run_state, start, end = runs[i]
        run_len = end - start + 1
        prev_state = runs[i - 1][0]
        next_state = runs[i + 1][0]
        if run_len < min_run_length and prev_state == next_state:
            for j in range(start, end + 1):
                result[j] = prev_state

    return result


def collapse_to_intervals(
    observations: list[CentroidObservation],
    states: list[str],
) -> list[MotionInterval]:
    """Collapse parallel observation/state lists into contiguous MotionIntervals.

    frame_range uses observations[i].frame_index (original source frame index),
    not the list index i. With stride > 1, intervals span gaps in the source
    timeline. The convention: frame_range[1] is the last SAMPLED frame index
    carrying that state; the gap to the next interval's start is unassigned.
    """
    if not observations:
        return []

    intervals: list[MotionInterval] = []
    start_obs = observations[0]
    current_state = states[0]

    for i in range(1, len(observations)):
        if states[i] != current_state:
            intervals.append(
                MotionInterval(
                    frame_range=(start_obs.frame_index, observations[i - 1].frame_index),
                    state=current_state,
                )
            )
            start_obs = observations[i]
            current_state = states[i]

    intervals.append(
        MotionInterval(
            frame_range=(start_obs.frame_index, observations[-1].frame_index),
            state=current_state,
        )
    )
    return intervals


def classify_motion(
    observations: list[CentroidObservation],
    window: int = 5,
    threshold_pixels: float = 15.0,
    frame_diagonal: float | None = None,
    min_run_length: int = 3,
) -> list[MotionInterval]:
    """High-level entry: classify an object's centroid trail into motion intervals.

    If frame_diagonal is provided (e.g. sqrt(w^2 + h^2)), threshold_pixels is
    treated as a fraction of that diagonal, making the threshold
    resolution-independent. Otherwise it is a raw pixel distance.

    Returns [] for empty observations.
    Returns a single 'stationary' interval when len(observations) < window + 1
    (insufficient history to compute any real displacements).

    Pipeline: displacements -> per-obs classification -> smoothing -> intervals.
    """
    if not observations:
        return []

    actual_threshold = (
        threshold_pixels * frame_diagonal
        if frame_diagonal is not None
        else threshold_pixels
    )

    if len(observations) < window + 1:
        return [
            MotionInterval(
                frame_range=(observations[0].frame_index, observations[-1].frame_index),
                state="stationary",
            )
        ]

    displacements = compute_displacements(observations, window)
    states = classify_per_observation(displacements, actual_threshold)
    states = smooth_states(states, min_run_length)
    return collapse_to_intervals(observations, states)
