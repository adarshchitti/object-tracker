from __future__ import annotations

import os

import numpy as np
import pytest

FIXTURE_VIDEO = os.path.join(os.path.dirname(__file__), "fixtures", "sample_short.mp4")


@pytest.fixture(scope="module")
def sample_frame() -> np.ndarray:
    """Read the first frame from the fixture video."""
    if not os.path.exists(FIXTURE_VIDEO):
        pytest.skip("Fixture video not found — run tests/fixtures/generate_fixture.py")

    from pipeline.frame_reader import FrameReader

    with FrameReader(FIXTURE_VIDEO) as reader:
        for _, frame in reader:
            return frame

    pytest.skip("Could not read frame from fixture video")


def test_detection_dataclass_shape() -> None:
    """Verify the Detection dataclass has all required fields."""
    from pipeline.detector import Detection

    det = Detection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=(10.0, 20.0, 100.0, 200.0),
    )
    assert det.track_id == 1
    assert det.class_id == 0
    assert det.class_name == "person"
    assert abs(det.confidence - 0.85) < 1e-6
    assert det.bbox == (10.0, 20.0, 100.0, 200.0)
    assert isinstance(det.bbox, tuple) and len(det.bbox) == 4


@pytest.mark.slow
def test_detector_returns_detections(sample_frame: np.ndarray) -> None:
    """Run detector on a fixture frame; assert it returns a list[Detection].

    The synthetic fixture has no real COCO objects so the list may be empty —
    we only verify the call succeeds and returns the correct type.
    If YOLO weights can't download (offline), this test will fail with a
    network error; that's expected and acceptable in offline CI.
    """
    from pipeline.detector import Detection, YoloDetector

    detector = YoloDetector()
    result = detector.detect(sample_frame)

    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, Detection)
