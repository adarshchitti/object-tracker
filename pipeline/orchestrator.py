"""Top-level pipeline orchestrator.

Called by FastAPI BackgroundTasks after a video upload. Updates Task.status
in the DB at each stage (processing → completed | failed) and writes
the serialised AnalysisResult to Task.result_json on success.

Stage order:
  1. FrameReader  — extract frames at configured stride
  2. YoloDetector — detect + track objects per frame
  3. HandDetector — detect hands per frame
  4. interaction   — match hands to objects
  5. aggregator    — build AnalysisResult
  6. keyframes     — (optional) save representative frames
"""


def process_video(task_id: str, video_path: str) -> None:
    """Run the full analysis pipeline for a given task.

    Args:
        task_id: UUID string matching the Task.id row in the DB.
        video_path: Absolute path to the uploaded video file.

    Side effects:
        Updates Task.status to "processing", then "completed" or "failed".
        Writes serialised AnalysisResult JSON to Task.result_json on success.
    """
    raise NotImplementedError
