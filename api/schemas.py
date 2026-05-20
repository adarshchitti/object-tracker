"""Pydantic v2 response models matching the required output schema."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    model_config = {"populate_by_name": True}

    duration_seconds: float = Field(alias="durationSeconds")
    frame_count: int = Field(alias="frameCount")
    width: int
    height: int
    fps: float


class MotionInterval(BaseModel):
    frame_range: tuple[int, int]
    state: Literal["moving", "stationary"]


class Interaction(BaseModel):
    interacted_by_person: int
    frame_start: int
    frame_end: int


class DetectedObject(BaseModel):
    model_config = {"populate_by_name": True}

    object_id: int
    class_: str = Field(alias="class")
    motion_history: list[MotionInterval]
    interactions: list[Interaction]


class AnalysisResult(BaseModel):
    model_config = {"populate_by_name": True}

    video_metadata: VideoMetadata = Field(alias="videoMetadata")
    objects_detected: list[DetectedObject] = Field(alias="objectsDetected")
    key_frames: list[str] | None = Field(default=None, alias="keyFrames")


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
