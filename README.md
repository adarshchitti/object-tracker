# Object Identification in Video

## Overview

An asynchronous video analysis service that accepts video uploads, runs object detection and multi-object tracking (YOLOv8 + ByteTrack), classifies per-object motion states, detects hand-object interactions via MediaPipe, and returns structured JSON results through a REST API. A Streamlit UI provides a browser interface for uploads and result display.

## Architecture

```
┌─────────────────────┐        ┌───────────────────────────────────────┐
│   Streamlit UI       │──POST─▶│  FastAPI  (port 8000)                 │
│   (port 8501)        │◀─JSON──│  /tasks  /tasks/{id}  /tasks/{id}/result│
└─────────────────────┘        └──────────────┬────────────────────────┘
                                               │ BackgroundTasks
                                               ▼
                                ┌──────────────────────────┐
                                │  Pipeline                │
                                │  FrameReader             │
                                │  YoloDetector (ByteTrack)│
                                │  HandDetector (MediaPipe)│
                                │  motion / interaction    │
                                │  aggregator / keyframes  │
                                └──────────────┬───────────┘
                                               │
                          ┌────────────────────┴──────────────────┐
                          │  storage/                              │
                          │  ├── tasks.db  (SQLite)               │
                          │  └── tasks/{id}/keyframes/*.jpg       │
                          └───────────────────────────────────────┘
```

## Tech stack

- Python 3.11+
- FastAPI + Uvicorn
- Streamlit
- Ultralytics YOLOv8 + ByteTrack
- MediaPipe Hands
- OpenCV
- NumPy
- Pydantic v2
- SQLAlchemy + SQLite
- pytest + FastAPI TestClient
- `uv` package manager

## Setup

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. Install dependencies:
   ```
   uv sync
   ```
3. Run both services:
   ```
   uv run bash scripts/run_local.sh
   ```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/tasks` | Upload video, create analysis task |
| GET | `/tasks/{task_id}` | Poll task status |
| GET | `/tasks/{task_id}/result` | Retrieve completed analysis JSON |
| GET | `/docs` | Interactive Swagger UI |

## Project structure

```
object-tracker/
├── api/            FastAPI app, routes, schemas, DB
├── pipeline/       Detection, tracking, motion, interaction stubs
├── ui/             Streamlit frontend
├── storage/        Runtime: SQLite DB + keyframe JPEGs (gitignored)
├── tests/          Smoke tests
└── scripts/        Local dev launcher
```

## Testing

```
uv run pytest -v
```

## Assumptions and tradeoffs

- TODO: YOLOv8 uses COCO 80-class labels — evaluate whether domain-specific fine-tuning is needed
- TODO: `BackgroundTasks` is single-process; Celery + Redis would be needed for multi-worker or priority queues
- TODO: Frame stride (default 1) processes every frame — tune for speed vs. accuracy on long videos
- TODO: SQLite is fine for a single-node prototype; Postgres required for concurrent writers

## Time spent

_TODO: fill in after completion_
