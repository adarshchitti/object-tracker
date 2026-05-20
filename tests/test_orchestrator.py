from __future__ import annotations

import json
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.db import Base
from api.models import Task
from api.schemas import AnalysisResult

FIXTURE_VIDEO = os.path.join(os.path.dirname(__file__), "fixtures", "sample_short.mp4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_factory(tmp_path):
    """Return a sessionmaker pointed at a fresh temp SQLite DB."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def _insert_task(SessionFactory, task_id: str, video_path: str = "/fake.mp4") -> None:
    session = SessionFactory()
    session.add(Task(id=task_id, status="pending", video_path=video_path))
    session.commit()
    session.close()


def _get_task(SessionFactory, task_id: str) -> Task:
    session = SessionFactory()
    task = session.get(Task, task_id)
    session.expunge(task)
    session.close()
    return task


# ---------------------------------------------------------------------------
# Fast: DB helper unit tests
# ---------------------------------------------------------------------------


def test_set_status_helpers_update_db(tmp_path) -> None:
    from pipeline.orchestrator import _set_completed, _set_failed, _set_status

    TempSession = _make_session_factory(tmp_path)
    _insert_task(TempSession, "t1")

    session = TempSession()
    try:
        _set_status(session, "t1", "processing")
        assert session.get(Task, "t1").status == "processing"

        _set_completed(session, "t1", '{"data": "ok"}')
        t = session.get(Task, "t1")
        assert t.status == "completed"
        assert t.result_json == '{"data": "ok"}'

        _set_failed(session, "t1", "something broke")
        t = session.get(Task, "t1")
        assert t.status == "failed"
        assert t.error_message == "something broke"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Slow: integration tests (download YOLO weights, run full pipeline)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_pipeline_on_fixture() -> None:
    """Call _run_pipeline directly; assert it returns a valid AnalysisResult."""
    if not os.path.exists(FIXTURE_VIDEO):
        pytest.skip("Fixture video not found — run tests/fixtures/generate_fixture.py")

    from pipeline.orchestrator import _run_pipeline

    result = _run_pipeline(
        video_path=FIXTURE_VIDEO,
        stride=5,
        min_track_frames=1,
        motion_window=5,
        motion_threshold_fraction=0.02,
        interaction_proximity_fraction=0.05,
        interaction_min_run_length=3,
    )

    assert isinstance(result, AnalysisResult)
    data = json.loads(result.model_dump_json(by_alias=True))
    assert "videoMetadata" in data
    assert "objectsDetected" in data
    vm = data["videoMetadata"]
    assert vm["width"] == 640
    assert vm["height"] == 480


@pytest.mark.slow
def test_process_video_writes_completed_status(tmp_path, monkeypatch) -> None:
    """End-to-end: process_video updates DB to completed with valid JSON."""
    if not os.path.exists(FIXTURE_VIDEO):
        pytest.skip("Fixture video not found — run tests/fixtures/generate_fixture.py")

    TempSession = _make_session_factory(tmp_path)
    monkeypatch.setattr("pipeline.orchestrator.SessionLocal", TempSession)
    _insert_task(TempSession, "task-e2e", video_path=FIXTURE_VIDEO)

    from pipeline.orchestrator import process_video

    process_video("task-e2e", FIXTURE_VIDEO, stride=5, min_track_frames=1)

    task = _get_task(TempSession, "task-e2e")
    assert task.status == "completed"
    assert task.result_json is not None
    data = json.loads(task.result_json)
    assert "videoMetadata" in data
    assert "objectsDetected" in data


@pytest.mark.slow
def test_process_video_records_failure_on_bad_path(tmp_path, monkeypatch) -> None:
    """process_video with a nonexistent path must not raise; DB must show failed."""
    TempSession = _make_session_factory(tmp_path)
    monkeypatch.setattr("pipeline.orchestrator.SessionLocal", TempSession)
    _insert_task(TempSession, "task-bad", video_path="/nonexistent.mp4")

    from pipeline.orchestrator import process_video

    # Must not raise
    process_video("task-bad", "/nonexistent/video.mp4", stride=5, min_track_frames=1)

    task = _get_task(TempSession, "task-bad")
    assert task.status == "failed"
    assert task.error_message is not None
    assert len(task.error_message) > 0
