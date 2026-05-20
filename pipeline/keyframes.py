"""Keyframe extraction: motion transitions and peak-interaction snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from pipeline.interaction import Interaction
from pipeline.motion import MotionInterval


@dataclass(frozen=True)
class Keyframe:
    """A keyframe to be extracted. `filename` is relative to the task's
    keyframes/ directory.
    """

    frame_index: int
    object_id: int
    kind: str  # "motion_transition" or "peak_interaction"
    filename: str


def find_motion_transition_frames(
    motion_intervals_by_object: dict[int, list[MotionInterval]],
) -> list[Keyframe]:
    """First frame of any 'moving' interval that follows a 'stationary' interval.

    Initial 'moving' intervals (with no preceding stationary) are skipped.
    Filename: obj{tid}_motion_transition_frame{idx}.jpg
    """
    keyframes: list[Keyframe] = []
    for object_id, intervals in motion_intervals_by_object.items():
        for prev, curr in zip(intervals, intervals[1:]):
            if prev.state == "stationary" and curr.state == "moving":
                frame_index = curr.frame_range[0]
                keyframes.append(
                    Keyframe(
                        frame_index=frame_index,
                        object_id=object_id,
                        kind="motion_transition",
                        filename=f"obj{object_id}_motion_transition_frame{frame_index}.jpg",
                    )
                )
    return keyframes


def find_peak_interaction_frames(
    interactions_by_object: dict[int, list[Interaction]],
    per_frame_distances: dict[int, dict[int, float]],
) -> list[Keyframe]:
    """For each Interaction, find the frame in [frame_start, frame_end] with
    the smallest hand-to-object distance.

    Silently skips intervals with no distance data in the range.
    Filename: obj{tid}_peak_interaction_person{pid}_frame{idx}.jpg
    """
    keyframes: list[Keyframe] = []
    for object_id, intervals in interactions_by_object.items():
        for inter in intervals:
            best_frame: int | None = None
            best_dist: float = float("inf")
            for fi in range(inter.frame_start, inter.frame_end + 1):
                dist = per_frame_distances.get(fi, {}).get(object_id)
                if dist is None:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_frame = fi
            if best_frame is None:
                continue
            person_id = inter.interacted_by_person
            keyframes.append(
                Keyframe(
                    frame_index=best_frame,
                    object_id=object_id,
                    kind="peak_interaction",
                    filename=(
                        f"obj{object_id}_peak_interaction_"
                        f"person{person_id}_frame{best_frame}.jpg"
                    ),
                )
            )
    return keyframes


def extract_keyframes(
    video_path: str,
    output_dir: Path,
    keyframes: list[Keyframe],
) -> list[str]:
    """Seek to each keyframe's frame_index in the source video and write JPG.

    Returns the list of filenames successfully written. Frames that can't be
    read (e.g., out-of-range index or codec quirks with CAP_PROP_POS_FRAMES,
    which is approximate for some codecs) are silently skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not keyframes:
        return []

    written: list[str] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return []
        for kf in sorted(keyframes, key=lambda k: k.frame_index):
            cap.set(cv2.CAP_PROP_POS_FRAMES, kf.frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_path = output_dir / kf.filename
            success = cv2.imwrite(
                str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if success:
                written.append(kf.filename)
    finally:
        cap.release()
    return written
