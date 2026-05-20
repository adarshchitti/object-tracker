from __future__ import annotations

"""Top-level pipeline orchestrator.

Called by FastAPI BackgroundTasks after a video upload. Manages its own DB
session (background thread, separate from the request session), updates
Task.status at each stage, and persists the serialised AnalysisResult on
success.  Never raises — all exceptions are caught and recorded as failure.
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from api.db import SessionLocal
from api.models import Task
from pipeline.aggregator import build_analysis_result
from pipeline.detector import YoloDetector
from pipeline.frame_reader import FrameReader
from pipeline.hands import HandDetector
from pipeline.interaction import detect_interactions

logger = logging.getLogger(__name__)


def process_video(
    task_id: str,
    video_path: str,
    stride: int = 5,
    min_track_frames: int = 3,
    motion_window: int = 5,
    motion_threshold_fraction: float = 0.02,
    interaction_proximity_fraction: float = 0.05,
    interaction_min_run_length: int = 3,
) -> None:
    """Run the full pipeline and persist the result to the database.

    Status transitions:
        pending    -> processing  (on entry)
        processing -> completed   (on success, result_json populated)
        processing -> failed      (on any exception, error_message populated)

    Manages its own DB session because this runs in a background thread
    separate from the FastAPI request that scheduled it.
    """
    session: Session = SessionLocal()
    try:
        _set_status(session, task_id, "processing")
        result = _run_pipeline(
            video_path=video_path,
            stride=stride,
            min_track_frames=min_track_frames,
            motion_window=motion_window,
            motion_threshold_fraction=motion_threshold_fraction,
            interaction_proximity_fraction=interaction_proximity_fraction,
            interaction_min_run_length=interaction_min_run_length,
        )
        result_json = result.model_dump_json(by_alias=True)
        _set_completed(session, task_id, result_json)
    except Exception as e:
        logger.exception("Pipeline failed for task %s", task_id)
        _set_failed(session, task_id, str(e))
    finally:
        session.close()


def _run_pipeline(
    video_path: str,
    stride: int,
    min_track_frames: int,
    motion_window: int,
    motion_threshold_fraction: float,
    interaction_proximity_fraction: float,
    interaction_min_run_length: int,
):
    """Open video, run models, return AnalysisResult.

    Separated from process_video so integration tests can call it directly
    without a DB session.
    """
    detector = YoloDetector()
    per_frame_objects: dict[int, list] = {}
    per_frame_hands: dict[int, list] = {}

    with FrameReader(video_path, stride=stride) as reader, HandDetector() as hand_detector:
        metadata = reader.metadata
        width = metadata["width"]
        height = metadata["height"]
        frame_diagonal = (width**2 + height**2) ** 0.5

        for frame_idx, frame in reader:
            detections = detector.detect(frame)
            hands = hand_detector.detect(frame_idx, frame)
            if detections:
                per_frame_objects[frame_idx] = detections
            if hands:
                per_frame_hands[frame_idx] = hands

    interactions_by_object = detect_interactions(
        per_frame_hands=per_frame_hands,
        per_frame_objects=per_frame_objects,
        frame_diagonal=frame_diagonal,
        proximity_fraction=interaction_proximity_fraction,
        min_run_length=interaction_min_run_length,
    )

    return build_analysis_result(
        video_metadata_dict=metadata,
        per_frame_objects=per_frame_objects,
        per_frame_hands=per_frame_hands,
        interactions_by_object=interactions_by_object,
        min_track_frames=min_track_frames,
        motion_window=motion_window,
        motion_threshold_fraction=motion_threshold_fraction,
    )


def _set_status(session: Session, task_id: str, status: str) -> None:
    task = session.query(Task).filter(Task.id == task_id).one()
    task.status = status
    task.updated_at = datetime.utcnow()
    session.commit()


def _set_completed(session: Session, task_id: str, result_json: str) -> None:
    task = session.query(Task).filter(Task.id == task_id).one()
    task.status = "completed"
    task.result_json = result_json
    task.updated_at = datetime.utcnow()
    session.commit()


def _set_failed(session: Session, task_id: str, error_message: str) -> None:
    task = session.query(Task).filter(Task.id == task_id).one()
    task.status = "failed"
    task.error_message = error_message
    task.updated_at = datetime.utcnow()
    session.commit()
