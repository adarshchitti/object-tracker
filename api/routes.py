"""FastAPI route definitions for task submission and result retrieval."""

from fastapi import APIRouter, HTTPException, UploadFile

from api.schemas import AnalysisResult, TaskStatus

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", status_code=202)
async def create_task(file: UploadFile) -> dict:
    """Accept a video file upload, persist it, create a Task row with status=pending,
    enqueue processing via BackgroundTasks, and return the new task_id."""
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str) -> TaskStatus:
    """Return current status (pending / processing / completed / failed) for the given task_id."""
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{task_id}/result", response_model=AnalysisResult)
async def get_task_result(task_id: str) -> AnalysisResult:
    """Return the full AnalysisResult JSON for a completed task.
    Raises 404 if task not found, 409 if still processing."""
    raise HTTPException(status_code=501, detail="Not implemented")
