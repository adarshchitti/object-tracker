"""Keyframe extraction and JPEG persistence.

Selects representative frames (e.g., on scene change or fixed interval)
and saves them as JPEGs under storage/tasks/{task_id}/keyframes/.
Returns relative file paths for inclusion in AnalysisResult.keyFrames.
"""

import numpy as np


def save_keyframes(
    frames: list[tuple[int, np.ndarray]],
    task_id: str,
    storage_root: str = "storage/tasks",
) -> list[str]:
    """Save selected frames as JPEG files and return their relative paths.

    Args:
        frames: List of (frame_index, bgr_array) to persist.
        task_id: Used to build the output directory path.
        storage_root: Base directory for task artefacts.

    Returns:
        List of relative file paths, e.g. ["storage/tasks/<id>/keyframes/0042.jpg"].
    """
    raise NotImplementedError
