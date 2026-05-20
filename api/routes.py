"""FastAPI route definitions for task submission and result retrieval."""

from __future__ import annotations

import os
import re
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api import storage
from api.db import get_db
from api.models import Task
from api.schemas import AnalysisResult, TaskStatus
from pipeline.orchestrator import process_video

router = APIRouter(prefix="/tasks", tags=["tasks"])

_CHUNK_SIZE = 1024 * 1024  # 1 MB
KEYFRAME_FILENAME_PATTERN = re.compile(
    r"^obj\d+_[a-z_]+_(?:person\d+_)?frame\d+\.jpg$"
)


@router.post(
    "",
    response_model=TaskStatus,
    status_code=202,
    responses={400: {"description": "Invalid file: bad extension, empty, or too large"}},
)
async def create_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> TaskStatus:
    """Upload a video for analysis.

    The video is saved to storage/tasks/{task_id}/input.<ext>, a Task row is
    created with status='pending', and process_video is scheduled to run in
    the background. Returns the task status immediately (202 Accepted).
    """
    if not storage.is_allowed_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension. Allowed: {sorted(storage.ALLOWED_VIDEO_EXTENSIONS)}",
        )

    ext = os.path.splitext(file.filename)[1].lower()
    task_id = uuid.uuid4().hex
    storage.create_task_directory(task_id)
    video_path = storage.input_video_path(task_id, ext)

    total = 0
    try:
        with open(video_path, "wb") as fh:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > storage.MAX_UPLOAD_BYTES:
                    fh.close()
                    video_path.unlink(missing_ok=True)
                    storage.task_directory(task_id).rmdir()
                    raise HTTPException(
                        status_code=400,
                        detail=f"File exceeds maximum size of {storage.MAX_UPLOAD_BYTES} bytes",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception:
        video_path.unlink(missing_ok=True)
        raise

    if total == 0:
        video_path.unlink(missing_ok=True)
        storage.task_directory(task_id).rmdir()
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    task = Task(id=task_id, status="pending", video_path=str(video_path))
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(process_video, task_id, str(video_path))

    return TaskStatus(
        task_id=task.id,
        status=task.status,
        created_at=task.created_at,
        updated_at=task.updated_at,
        error_message=task.error_message,
    )


@router.get(
    "/{task_id}",
    response_model=TaskStatus,
    responses={404: {"description": "Task not found"}},
)
def get_task_status(task_id: str, db: Session = Depends(get_db)) -> TaskStatus:
    """Return the current status of a task."""
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskStatus(
        task_id=task.id,
        status=task.status,
        created_at=task.created_at,
        updated_at=task.updated_at,
        error_message=task.error_message,
    )


@router.get(
    "/{task_id}/result",
    response_model=AnalysisResult,
    responses={
        404: {"description": "Task not found"},
        409: {"description": "Task is not yet completed"},
    },
)
def get_task_result(task_id: str, db: Session = Depends(get_db)) -> AnalysisResult:
    """Return the analysis result for a completed task."""
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "task_id": task_id,
                "status": task.status,
                "error_message": task.error_message,
            },
        )

    return AnalysisResult.model_validate_json(task.result_json)


@router.get(
    "/{task_id}/keyframes/{filename}",
    responses={
        400: {"description": "Invalid filename"},
        404: {"description": "Task or keyframe not found"},
    },
)
def get_keyframe(
    task_id: str,
    filename: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream a keyframe JPG. Filename must match the strict naming pattern;
    rejected before any filesystem access to prevent path traversal."""
    if not KEYFRAME_FILENAME_PATTERN.match(filename):
        raise HTTPException(status_code=400, detail="Invalid keyframe filename")

    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    path = storage.task_directory(task_id) / "keyframes" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Keyframe not found")

    return FileResponse(str(path), media_type="image/jpeg")
