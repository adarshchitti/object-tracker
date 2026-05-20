"""Thin HTTP wrapper around the FastAPI service. No Streamlit imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import requests


@dataclass(frozen=True)
class TaskHandle:
    """Returned from upload_video. Used to query status and result."""

    task_id: str
    status: str


class ApiError(Exception):
    """Raised on non-2xx responses (except 409 on result, which is normal)."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"{status_code}: {message}")


def _extract_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason
    detail = body.get("detail") if isinstance(body, dict) else None
    if detail is None:
        return response.text
    if isinstance(detail, str):
        return detail
    return str(detail)


def upload_video(
    base_url: str,
    file: BinaryIO,
    filename: str,
    content_type: str = "video/mp4",
    timeout: float = 30.0,
) -> TaskHandle:
    """POST /tasks with a multipart upload. Returns the new TaskHandle.

    Raises ApiError on failure.
    """
    url = f"{base_url.rstrip('/')}/tasks"
    response = requests.post(
        url,
        files={"file": (filename, file, content_type)},
        timeout=timeout,
    )
    if response.status_code != 202:
        raise ApiError(response.status_code, _extract_detail(response))
    body = response.json()
    return TaskHandle(task_id=body["task_id"], status=body["status"])


def get_task_status(base_url: str, task_id: str, timeout: float = 5.0) -> dict:
    """GET /tasks/{task_id}. Returns the parsed JSON.

    Raises ApiError on 4xx/5xx.
    """
    url = f"{base_url.rstrip('/')}/tasks/{task_id}"
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise ApiError(response.status_code, _extract_detail(response))
    return response.json()


def get_task_result(base_url: str, task_id: str, timeout: float = 10.0) -> dict | None:
    """GET /tasks/{task_id}/result.

    Returns the parsed JSON on 200. Returns None on 409 (still processing or
    failed — caller decides what to do). Raises ApiError on any other failure.
    """
    url = f"{base_url.rstrip('/')}/tasks/{task_id}/result"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 200:
        return response.json()
    if response.status_code == 409:
        return None
    raise ApiError(response.status_code, _extract_detail(response))
