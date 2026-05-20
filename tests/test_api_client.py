"""Unit tests for ui.api_client using requests-mock."""

from __future__ import annotations

import io

import pytest

from ui.api_client import (
    ApiError,
    TaskHandle,
    get_task_result,
    get_task_status,
    upload_video,
)

BASE_URL = "http://test-api"


def test_upload_video_success(requests_mock) -> None:
    requests_mock.post(
        f"{BASE_URL}/tasks",
        status_code=202,
        json={
            "task_id": "abc123def",
            "status": "pending",
            "created_at": "2026-05-20T00:00:00",
            "updated_at": "2026-05-20T00:00:00",
            "error_message": None,
        },
    )

    handle = upload_video(
        base_url=BASE_URL,
        file=io.BytesIO(b"fakebytes"),
        filename="clip.mp4",
    )

    assert isinstance(handle, TaskHandle)
    assert handle.task_id == "abc123def"
    assert handle.status == "pending"


def test_upload_video_400_raises_api_error(requests_mock) -> None:
    requests_mock.post(
        f"{BASE_URL}/tasks",
        status_code=400,
        json={"detail": "Unsupported file extension"},
    )

    with pytest.raises(ApiError) as exc_info:
        upload_video(
            base_url=BASE_URL,
            file=io.BytesIO(b"x"),
            filename="bad.exe",
        )
    assert exc_info.value.status_code == 400
    assert "Unsupported" in exc_info.value.message


def test_get_task_status_success(requests_mock) -> None:
    payload = {
        "task_id": "t1",
        "status": "processing",
        "created_at": "2026-05-20T00:00:00",
        "updated_at": "2026-05-20T00:00:01",
        "error_message": None,
    }
    requests_mock.get(f"{BASE_URL}/tasks/t1", status_code=200, json=payload)

    result = get_task_status(BASE_URL, "t1")
    assert result == payload


def test_get_task_status_404_raises(requests_mock) -> None:
    requests_mock.get(
        f"{BASE_URL}/tasks/missing",
        status_code=404,
        json={"detail": "Task missing not found"},
    )

    with pytest.raises(ApiError) as exc_info:
        get_task_status(BASE_URL, "missing")
    assert exc_info.value.status_code == 404


def test_get_task_result_completed(requests_mock) -> None:
    payload = {
        "videoMetadata": {
            "durationSeconds": 2.0,
            "frameCount": 60,
            "width": 640,
            "height": 480,
            "fps": 30.0,
        },
        "objectsDetected": [],
        "keyFrames": None,
    }
    requests_mock.get(f"{BASE_URL}/tasks/done/result", status_code=200, json=payload)

    result = get_task_result(BASE_URL, "done")
    assert result == payload


def test_get_task_result_409_returns_none(requests_mock) -> None:
    requests_mock.get(
        f"{BASE_URL}/tasks/pending1/result",
        status_code=409,
        json={"detail": {"task_id": "pending1", "status": "pending", "error_message": None}},
    )

    assert get_task_result(BASE_URL, "pending1") is None


def test_get_task_result_500_raises(requests_mock) -> None:
    requests_mock.get(
        f"{BASE_URL}/tasks/broken/result",
        status_code=500,
        json={"detail": "Internal Server Error"},
    )

    with pytest.raises(ApiError) as exc_info:
        get_task_result(BASE_URL, "broken")
    assert exc_info.value.status_code == 500
