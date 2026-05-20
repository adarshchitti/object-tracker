"""HTTP-level tests for the task endpoints using FastAPI TestClient."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api import storage
from api.db import Base, get_db
from api.main import app
from api.models import Task

# ---------------------------------------------------------------------------
# Health / OpenAPI
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema_includes_all_endpoints(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    assert "/tasks" in paths
    assert "post" in paths["/tasks"]
    assert "/tasks/{task_id}" in paths
    assert "get" in paths["/tasks/{task_id}"]
    assert "/tasks/{task_id}/result" in paths
    assert "get" in paths["/tasks/{task_id}/result"]


# ---------------------------------------------------------------------------
# POST /tasks
# ---------------------------------------------------------------------------


def test_create_task_with_valid_video(client: TestClient, sample_video_bytes: bytes) -> None:
    response = client.post(
        "/tasks",
        files={"file": ("clip.mp4", sample_video_bytes, "video/mp4")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert len(body["task_id"]) == 32
    int(body["task_id"], 16)  # uuid4().hex is 32 hex chars


def test_create_task_with_bad_extension(client: TestClient) -> None:
    response = client.post(
        "/tasks",
        files={"file": ("malicious.exe", b"\x00\x01\x02", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_create_task_with_no_extension(client: TestClient) -> None:
    response = client.post(
        "/tasks",
        files={"file": ("noextension", b"abc", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_create_task_persists_to_db(
    client: TestClient, sample_video_bytes: bytes, db_session
) -> None:
    response = client.post(
        "/tasks",
        files={"file": ("clip.mp4", sample_video_bytes, "video/mp4")},
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    row = db_session.get(Task, task_id)
    assert row is not None
    assert row.status == "pending"
    assert row.video_path.endswith(f"{task_id}{os.sep}input.mp4") or row.video_path.endswith(
        f"{task_id}/input.mp4"
    )


def test_create_task_saves_video_to_disk(
    client: TestClient, sample_video_bytes: bytes, storage_root
) -> None:
    response = client.post(
        "/tasks",
        files={"file": ("clip.mp4", sample_video_bytes, "video/mp4")},
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    video_path = storage_root / "tasks" / task_id / "input.mp4"
    assert video_path.exists()
    assert video_path.read_bytes() == sample_video_bytes


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}
# ---------------------------------------------------------------------------


def test_get_task_status_for_existing_task(client: TestClient, db_session) -> None:
    db_session.add(
        Task(id="abc123", status="processing", video_path="/fake/input.mp4")
    )
    db_session.commit()

    response = client.get("/tasks/abc123")
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "abc123"
    assert body["status"] == "processing"


def test_get_task_status_404_for_missing_task(client: TestClient) -> None:
    response = client.get("/tasks/nonexistent-id")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/result
# ---------------------------------------------------------------------------


_SAMPLE_RESULT = {
    "videoMetadata": {
        "durationSeconds": 2.0,
        "frameCount": 60,
        "width": 640,
        "height": 480,
        "fps": 30.0,
    },
    "objectsDetected": [
        {
            "object_id": 1,
            "class": "laptop",
            "motion_history": [{"frame_range": [0, 30], "state": "stationary"}],
            "interactions": [],
        }
    ],
    "keyFrames": None,
}


def test_get_task_result_returns_completed_result(client: TestClient, db_session) -> None:
    db_session.add(
        Task(
            id="done1",
            status="completed",
            video_path="/fake/input.mp4",
            result_json=json.dumps(_SAMPLE_RESULT),
        )
    )
    db_session.commit()

    response = client.get("/tasks/done1/result")
    assert response.status_code == 200
    body = response.json()
    assert "videoMetadata" in body
    assert "objectsDetected" in body
    assert body["videoMetadata"]["durationSeconds"] == 2.0
    assert body["videoMetadata"]["frameCount"] == 60


def test_get_task_result_409_when_pending(client: TestClient, db_session) -> None:
    db_session.add(Task(id="p1", status="pending", video_path="/fake/input.mp4"))
    db_session.commit()

    response = client.get("/tasks/p1/result")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["status"] == "pending"


def test_get_task_result_409_when_processing(client: TestClient, db_session) -> None:
    db_session.add(Task(id="proc1", status="processing", video_path="/fake/input.mp4"))
    db_session.commit()

    response = client.get("/tasks/proc1/result")
    assert response.status_code == 409
    assert response.json()["detail"]["status"] == "processing"


def test_get_task_result_409_when_failed(client: TestClient, db_session) -> None:
    db_session.add(
        Task(
            id="f1",
            status="failed",
            video_path="/fake/input.mp4",
            error_message="something broke",
        )
    )
    db_session.commit()

    response = client.get("/tasks/f1/result")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["status"] == "failed"
    assert detail["error_message"] == "something broke"


def test_get_task_result_404_for_missing_task(client: TestClient) -> None:
    response = client.get("/tasks/nope/result")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Slow: full upload -> completed -> result flow with the real pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/keyframes/{filename}
# ---------------------------------------------------------------------------


def test_get_keyframe_returns_image(client, storage_root, db_session) -> None:
    db_session.add(Task(id="kf1", status="completed", video_path="/fake.mp4"))
    db_session.commit()
    kf_dir = storage_root / "tasks" / "kf1" / "keyframes"
    kf_dir.mkdir(parents=True)
    fname = "obj2_motion_transition_frame50.jpg"
    (kf_dir / fname).write_bytes(b"\xff\xd8\xff\xe0fakejpgbytes")

    response = client.get(f"/tasks/kf1/keyframes/{fname}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_get_keyframe_404_for_missing_task(client) -> None:
    response = client.get("/tasks/nope/keyframes/obj1_motion_transition_frame1.jpg")
    assert response.status_code == 404


def test_get_keyframe_404_for_missing_file(client, db_session) -> None:
    db_session.add(Task(id="kf2", status="completed", video_path="/fake.mp4"))
    db_session.commit()
    response = client.get("/tasks/kf2/keyframes/obj1_motion_transition_frame1.jpg")
    assert response.status_code == 404


def test_get_keyframe_400_for_invalid_filename(client, db_session) -> None:
    db_session.add(Task(id="kf3", status="completed", video_path="/fake.mp4"))
    db_session.commit()

    # Wrong shape (no obj prefix, wrong ext)
    r = client.get("/tasks/kf3/keyframes/malicious.exe")
    assert r.status_code == 400

    # URL-encoded path traversal — the filename arrives as ../../etc/passwd
    # which the regex rejects before any path operation.
    r = client.get("/tasks/kf3/keyframes/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404)  # Starlette may normalize before route match


def test_get_keyframe_400_for_non_jpg_extension(client, db_session) -> None:
    db_session.add(Task(id="kf4", status="completed", video_path="/fake.mp4"))
    db_session.commit()
    r = client.get("/tasks/kf4/keyframes/obj1_motion_transition_frame1.png")
    assert r.status_code == 400


@pytest.mark.slow
def test_full_upload_to_result_flow(tmp_path, monkeypatch) -> None:
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_short.mp4")
    if not os.path.exists(fixture):
        pytest.skip("Fixture video missing")

    # Redirect storage and DB to tmp paths.
    storage_root = tmp_path / "storage"
    tasks_root = storage_root / "tasks"
    tasks_root.mkdir(parents=True)
    monkeypatch.setattr(storage, "STORAGE_ROOT", storage_root)
    monkeypatch.setattr(storage, "TASKS_ROOT", tasks_root)

    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def _override_get_db():
        s = TestSession()
        try:
            yield s
        finally:
            s.close()

    # process_video opens its own session via SessionLocal — point it at the test DB.
    monkeypatch.setattr("pipeline.orchestrator.SessionLocal", TestSession)
    app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(app) as client:
            with open(fixture, "rb") as fh:
                video_bytes = fh.read()
            response = client.post(
                "/tasks",
                files={"file": ("sample.mp4", video_bytes, "video/mp4")},
            )
            assert response.status_code == 202
            task_id = response.json()["task_id"]

            # FastAPI BackgroundTasks in TestClient run synchronously before the
            # response returns, so the task should already be completed/failed.
            deadline = time.time() + 60
            status = None
            while time.time() < deadline:
                r = client.get(f"/tasks/{task_id}")
                assert r.status_code == 200
                status = r.json()["status"]
                if status in {"completed", "failed"}:
                    break
                time.sleep(0.2)

            assert status == "completed", f"task ended with status={status}"

            r = client.get(f"/tasks/{task_id}/result")
            assert r.status_code == 200
            body = r.json()
            assert "videoMetadata" in body
            assert "objectsDetected" in body
            assert "durationSeconds" in body["videoMetadata"]
            assert "frameCount" in body["videoMetadata"]
    finally:
        app.dependency_overrides.pop(get_db, None)
