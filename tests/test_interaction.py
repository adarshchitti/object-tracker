from __future__ import annotations

import pytest

from pipeline.detector import Detection
from pipeline.hands import HandObservation
from pipeline.interaction import (
    Interaction,
    detect_interactions,
    distance_point_to_bbox,
    find_single_person_id,
    per_frame_interaction_distances,
    per_frame_interactions,
)

# ---------------------------------------------------------------------------
# Shared test geometry
# ---------------------------------------------------------------------------
# Frame diagonal for a 640×480 frame: ~800.  proximity_fraction=0.05 → 40 px.
DIAG = 800.0
PROX = 0.05  # threshold = 40 pixels

PERSON_ID = 99
LAPTOP_ID = 1
CUP_ID = 2


def _person(track_id: int = PERSON_ID) -> Detection:
    return Detection(
        track_id=track_id, class_id=0, class_name="person",
        confidence=0.9, bbox=(300.0, 100.0, 400.0, 300.0),
    )


def _laptop(track_id: int = LAPTOP_ID) -> Detection:
    return Detection(
        track_id=track_id, class_id=63, class_name="laptop",
        confidence=0.9, bbox=(0.0, 0.0, 100.0, 100.0),
    )


def _cup(track_id: int = CUP_ID) -> Detection:
    return Detection(
        track_id=track_id, class_id=41, class_name="cup",
        confidence=0.9, bbox=(500.0, 500.0, 600.0, 600.0),
    )


def _hand_near_laptop(frame_index: int = 0) -> HandObservation:
    # Wrist at (50, 50): inside laptop bbox (0,0,100,100) → distance = 0
    return HandObservation(
        frame_index=frame_index, bbox=(40.0, 40.0, 80.0, 80.0),
        center=(50.0, 50.0), confidence=0.9, handedness="Right",
    )


def _hand_near_cup(frame_index: int = 0) -> HandObservation:
    # Wrist at (550, 550): inside cup bbox (500,500,600,600) → distance = 0
    return HandObservation(
        frame_index=frame_index, bbox=(540.0, 540.0, 580.0, 580.0),
        center=(550.0, 550.0), confidence=0.9, handedness="Right",
    )


def _hand_far(frame_index: int = 0) -> HandObservation:
    # Wrist at (720, 420): far from both laptop and cup bboxes
    return HandObservation(
        frame_index=frame_index, bbox=(710.0, 410.0, 750.0, 450.0),
        center=(720.0, 420.0), confidence=0.9, handedness="Left",
    )


# ---------------------------------------------------------------------------
# distance_point_to_bbox
# ---------------------------------------------------------------------------


def test_distance_point_to_bbox_inside_returns_zero() -> None:
    assert distance_point_to_bbox((50.0, 50.0), (0.0, 0.0, 100.0, 100.0)) == pytest.approx(0.0)


def test_distance_point_to_bbox_on_edge_returns_zero() -> None:
    assert distance_point_to_bbox((0.0, 50.0), (0.0, 0.0, 100.0, 100.0)) == pytest.approx(0.0)


def test_distance_point_to_bbox_left_of_bbox() -> None:
    # Nearest point on bbox: (0, 50), distance = 10
    assert distance_point_to_bbox((-10.0, 50.0), (0.0, 0.0, 100.0, 100.0)) == pytest.approx(10.0)


def test_distance_point_to_bbox_corner() -> None:
    # Point (-3, -4); nearest corner (0, 0); 3-4-5 triangle → distance = 5
    assert distance_point_to_bbox((-3.0, -4.0), (0.0, 0.0, 100.0, 100.0)) == pytest.approx(5.0)


def test_distance_point_to_bbox_above_bbox() -> None:
    # Point (50, -20); nearest point (50, 0); distance = 20
    assert distance_point_to_bbox((50.0, -20.0), (0.0, 0.0, 100.0, 100.0)) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# per_frame_interactions
# ---------------------------------------------------------------------------


def test_per_frame_interactions_skips_person_class() -> None:
    # Hand wrist at (350, 200): inside person bbox (300,100,400,300)
    hand = HandObservation(
        frame_index=0, bbox=(340.0, 190.0, 360.0, 210.0),
        center=(350.0, 200.0), confidence=0.9, handedness="Right",
    )
    result = per_frame_interactions([hand], [_person()], DIAG, PROX)
    assert result == set()


def test_per_frame_interactions_close_hand_counts() -> None:
    result = per_frame_interactions(
        [_hand_near_laptop()], [_person(), _laptop()], DIAG, PROX
    )
    assert result == {LAPTOP_ID}


def test_per_frame_interactions_far_hand_excluded() -> None:
    result = per_frame_interactions(
        [_hand_far()], [_person(), _laptop()], DIAG, PROX
    )
    assert result == set()


def test_per_frame_interactions_no_hands() -> None:
    result = per_frame_interactions([], [_laptop()], DIAG, PROX)
    assert result == set()


def test_per_frame_interactions_no_objects() -> None:
    result = per_frame_interactions([_hand_near_laptop()], [], DIAG, PROX)
    assert result == set()


# ---------------------------------------------------------------------------
# find_single_person_id
# ---------------------------------------------------------------------------


def test_find_single_person_id_picks_most_frequent() -> None:
    per_frame = {
        0: [_person(1), _laptop()],
        1: [_person(1), _laptop()],
        2: [_person(1), _laptop()],
        3: [_person(5), _laptop()],
        4: [_person(5), _laptop()],
    }
    assert find_single_person_id(per_frame) == 1


def test_find_single_person_id_returns_none_when_no_person() -> None:
    per_frame = {0: [_laptop()], 1: [_laptop()]}
    assert find_single_person_id(per_frame) is None


def test_find_single_person_id_empty_frames() -> None:
    assert find_single_person_id({}) is None


# ---------------------------------------------------------------------------
# detect_interactions — end-to-end
# ---------------------------------------------------------------------------


def _build_frames(
    n: int,
    near_frames: set[int],
    near_hand_fn=_hand_near_laptop,
) -> tuple[dict, dict]:
    """Build per_frame_hands and per_frame_objects for n frames 0..n-1."""
    per_frame_hands = {
        i: [near_hand_fn(i) if i in near_frames else _hand_far(i)]
        for i in range(n)
    }
    per_frame_objects = {i: [_person(), _laptop()] for i in range(n)}
    return per_frame_hands, per_frame_objects


def test_detect_interactions_basic_run() -> None:
    hands, objects = _build_frames(10, near_frames=set(range(3, 8)))
    result = detect_interactions(hands, objects, DIAG, PROX, min_run_length=3)

    assert LAPTOP_ID in result
    assert len(result[LAPTOP_ID]) == 1
    inter = result[LAPTOP_ID][0]
    assert inter.interacted_by_person == PERSON_ID
    assert inter.frame_start == 3
    assert inter.frame_end == 7


def test_detect_interactions_drops_short_runs() -> None:
    # Only frame 5 has the hand near the laptop (1 frame < min_run_length=3)
    hands, objects = _build_frames(10, near_frames={5})
    result = detect_interactions(hands, objects, DIAG, PROX, min_run_length=3)
    assert result == {} or result.get(LAPTOP_ID, []) == []


def test_detect_interactions_multiple_runs() -> None:
    # Frames 0..4 near, 5..9 far, 10..14 near
    near = set(range(0, 5)) | set(range(10, 15))
    hands, objects = _build_frames(15, near_frames=near)
    result = detect_interactions(hands, objects, DIAG, PROX, min_run_length=3)

    assert LAPTOP_ID in result
    runs = result[LAPTOP_ID]
    assert len(runs) == 2
    assert runs[0].frame_start == 0 and runs[0].frame_end == 4
    assert runs[1].frame_start == 10 and runs[1].frame_end == 14


def test_detect_interactions_multiple_objects() -> None:
    # Frames 0..5: hand near laptop; frames 6..9: far; frames 10..15: near cup
    per_frame_hands = {}
    per_frame_objects = {}
    for i in range(16):
        if i <= 5:
            per_frame_hands[i] = [_hand_near_laptop(i)]
        elif i >= 10:
            per_frame_hands[i] = [_hand_near_cup(i)]
        else:
            per_frame_hands[i] = [_hand_far(i)]
        per_frame_objects[i] = [_person(), _laptop(), _cup()]

    result = detect_interactions(per_frame_hands, per_frame_objects, DIAG, PROX, min_run_length=3)

    assert LAPTOP_ID in result
    assert CUP_ID in result
    assert len(result[LAPTOP_ID]) == 1
    assert len(result[CUP_ID]) == 1
    assert result[LAPTOP_ID][0].frame_start == 0
    assert result[CUP_ID][0].frame_start == 10


def test_detect_interactions_no_person_returns_empty() -> None:
    per_frame_hands = {i: [_hand_near_laptop(i)] for i in range(5)}
    per_frame_objects = {i: [_laptop()] for i in range(5)}  # no person
    result = detect_interactions(per_frame_hands, per_frame_objects, DIAG, PROX)
    assert result == {}


def test_detect_interactions_uses_original_frame_indices() -> None:
    # Stride=5: observation keys are 0, 5, 10, 15
    # Hand near laptop at frames 5, 10, 15 → 3 consecutive observations
    per_frame_hands = {
        0: [_hand_far(0)],
        5: [_hand_near_laptop(5)],
        10: [_hand_near_laptop(10)],
        15: [_hand_near_laptop(15)],
    }
    per_frame_objects = {fi: [_person(), _laptop()] for fi in (0, 5, 10, 15)}

    result = detect_interactions(per_frame_hands, per_frame_objects, DIAG, PROX, min_run_length=3)

    assert LAPTOP_ID in result
    assert len(result[LAPTOP_ID]) == 1
    inter = result[LAPTOP_ID][0]
    assert inter.frame_start == 5
    assert inter.frame_end == 15


def test_detect_interactions_handles_missing_frames() -> None:
    # per_frame_hands has frame 10, per_frame_objects does not → no KeyError
    per_frame_hands = {
        0: [_hand_near_laptop(0)],
        5: [_hand_near_laptop(5)],
        10: [_hand_near_laptop(10)],
    }
    per_frame_objects = {
        0: [_person(), _laptop()],
        5: [_person(), _laptop()],
        # frame 10 intentionally absent
    }
    # Frame 10: objects=[] → no interaction → run breaks at length 2 < 3
    result = detect_interactions(per_frame_hands, per_frame_objects, DIAG, PROX, min_run_length=3)
    assert isinstance(result, dict)  # no KeyError


# ---------------------------------------------------------------------------
# per_frame_interaction_distances
# ---------------------------------------------------------------------------


def _hand_at(point: tuple[float, float], frame_index: int = 0) -> HandObservation:
    return HandObservation(
        frame_index=frame_index,
        bbox=(point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10),
        center=point,
        confidence=0.9,
        handedness="Right",
    )


def test_per_frame_interaction_distances_basic() -> None:
    # Hand at (100, 100); laptop bbox at (200, 200, 300, 300).
    # Nearest corner is (200, 200). Distance = sqrt(100^2 + 100^2) ≈ 141.42.
    laptop = Detection(
        track_id=LAPTOP_ID, class_id=63, class_name="laptop",
        confidence=0.9, bbox=(200.0, 200.0, 300.0, 300.0),
    )
    per_frame_hands = {0: [_hand_at((100.0, 100.0))]}
    per_frame_objects = {0: [laptop]}

    result = per_frame_interaction_distances(per_frame_hands, per_frame_objects)

    assert 0 in result
    assert LAPTOP_ID in result[0]
    assert result[0][LAPTOP_ID] == pytest.approx(141.421356, rel=1e-4)


def test_per_frame_interaction_distances_skips_person() -> None:
    per_frame_hands = {0: [_hand_at((50.0, 50.0))]}
    per_frame_objects = {0: [_person(), _laptop()]}

    result = per_frame_interaction_distances(per_frame_hands, per_frame_objects)

    assert PERSON_ID not in result[0]
    assert LAPTOP_ID in result[0]


def test_per_frame_interaction_distances_min_across_hands() -> None:
    laptop = Detection(
        track_id=LAPTOP_ID, class_id=63, class_name="laptop",
        confidence=0.9, bbox=(0.0, 0.0, 100.0, 100.0),
    )
    far_hand = _hand_at((500.0, 500.0))   # distance to laptop corner ~ 565
    close_hand = _hand_at((50.0, 50.0))   # inside bbox -> distance 0

    result = per_frame_interaction_distances(
        {0: [far_hand, close_hand]}, {0: [laptop]}
    )

    assert result[0][LAPTOP_ID] == pytest.approx(0.0)
