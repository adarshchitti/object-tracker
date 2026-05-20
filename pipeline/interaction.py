from __future__ import annotations

"""Hand-object interaction detection via wrist-to-bbox proximity.

An interaction is a continuous run of frames where a hand (identified by
wrist position) is within a threshold distance of a non-person object's
bounding box.

Design notes
------------
* Why hand center, not hand bbox?
  Hand bboxes are small and noisy frame-to-frame.  The wrist landmark is
  the single most stable point; wrist-to-bbox distance is a cleaner signal.

* Proximity fraction.
  0.05 of a 1280×720 diagonal (~73 px) is a reasonable starting threshold
  for over-equipment lab work.  Pass a different value to tune.

* Single-person assumption.
  find_single_person_id returns the most frequent person track_id.  Multi-
  person attribution (nearest person to each hand) is a documented follow-up.

* min_run_length suppresses noise.
  A single-frame "interaction" is usually a hand briefly passing an object.
  Real interactions persist across multiple sampled frames.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

from pipeline.detector import Detection
from pipeline.hands import HandObservation


@dataclass(frozen=True)
class Interaction:
    """An interaction interval between a person and an object.

    frame_start and frame_end are inclusive and use ORIGINAL source frame
    indices (as yielded by FrameReader), not sampled-frame list indices.
    """

    interacted_by_person: int   # person's track_id
    frame_start: int
    frame_end: int


def distance_point_to_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> float:
    """Euclidean distance from a point to the nearest edge of an axis-aligned bbox.

    Returns 0.0 if the point is inside or on the boundary of the bbox.
    """
    px, py = point
    x1, y1, x2, y2 = bbox
    nearest_x = max(x1, min(px, x2))
    nearest_y = max(y1, min(py, y2))
    return math.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)


def per_frame_interactions(
    hands_in_frame: list[HandObservation],
    objects_in_frame: list[Detection],
    frame_diagonal: float,
    proximity_fraction: float = 0.05,
) -> set[int]:
    """Return the set of object track_ids interacting with any hand in this frame.

    proximity_fraction * frame_diagonal gives the pixel threshold.
    Person detections are skipped — a person cannot interact with themselves.
    """
    threshold = proximity_fraction * frame_diagonal
    interacting: set[int] = set()
    for hand in hands_in_frame:
        for det in objects_in_frame:
            if det.class_name == "person":
                continue
            if distance_point_to_bbox(hand.center, det.bbox) <= threshold:
                interacting.add(det.track_id)
    return interacting


def find_single_person_id(per_frame_objects: dict[int, list[Detection]]) -> int | None:
    """Return the most frequent person track_id across all frames, or None.

    Frequency-based: robust to occasional missed detections.  For the
    target single-person lab scenario this is sufficient.
    """
    counts: dict[int, int] = defaultdict(int)
    for detections in per_frame_objects.values():
        for det in detections:
            if det.class_name == "person":
                counts[det.track_id] += 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def detect_interactions(
    per_frame_hands: dict[int, list[HandObservation]],
    per_frame_objects: dict[int, list[Detection]],
    frame_diagonal: float,
    proximity_fraction: float = 0.05,
    min_run_length: int = 3,
) -> dict[int, list[Interaction]]:
    """Return a mapping of object_track_id -> list[Interaction].

    Algorithm:
    1. Resolve the person track_id (single-person assumption).
    2. For each sampled frame (from per_frame_hands keys, sorted), compute
       the set of object track_ids whose bbox is within proximity of any hand.
    3. Per object, find maximal consecutive runs in the sorted frame list
       where it was interacting.  "Consecutive" means adjacent in the sorted
       list of sampled frame indices — gaps from stride sampling do not break
       a run.
    4. Drop runs shorter than min_run_length (noise suppression).
    5. Emit one Interaction per surviving run.

    Frames absent from per_frame_objects are treated as having no detections.
    Returns {} if no person track_id is found.
    """
    person_id = find_single_person_id(per_frame_objects)
    if person_id is None:
        return {}

    sorted_frames = sorted(per_frame_hands.keys())

    # Build per-frame interaction sets
    frame_interacting: dict[int, set[int]] = {
        fi: per_frame_interactions(
            per_frame_hands[fi],
            per_frame_objects.get(fi, []),
            frame_diagonal,
            proximity_fraction,
        )
        for fi in sorted_frames
    }

    # Collect all non-person object track_ids seen across all frames
    all_object_ids: set[int] = {
        det.track_id
        for detections in per_frame_objects.values()
        for det in detections
        if det.class_name != "person"
    }

    result: dict[int, list[Interaction]] = {}

    for obj_id in all_object_ids:
        interactions: list[Interaction] = []
        in_run = False
        run_start: int = 0
        run_end: int = 0
        run_count = 0

        for fi in sorted_frames:
            if obj_id in frame_interacting[fi]:
                if not in_run:
                    in_run = True
                    run_start = fi
                    run_count = 1
                else:
                    run_count += 1
                run_end = fi
            else:
                if in_run:
                    if run_count >= min_run_length:
                        interactions.append(
                            Interaction(person_id, run_start, run_end)
                        )
                    in_run = False
                    run_count = 0

        # Close any run still open at end of sequence
        if in_run and run_count >= min_run_length:
            interactions.append(Interaction(person_id, run_start, run_end))

        if interactions:
            result[obj_id] = interactions

    return result
