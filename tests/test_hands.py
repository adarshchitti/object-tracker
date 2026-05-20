from __future__ import annotations

import os

import pytest

from pipeline.hands import HandObservation

FIXTURE_VIDEO = os.path.join(os.path.dirname(__file__), "fixtures", "sample_short.mp4")


def test_hand_observation_dataclass() -> None:
    obs = HandObservation(
        frame_index=5,
        bbox=(10.0, 20.0, 80.0, 90.0),
        center=(15.0, 25.0),
        confidence=0.92,
        handedness="Right",
    )
    assert obs.frame_index == 5
    assert obs.bbox == (10.0, 20.0, 80.0, 90.0)
    assert obs.center == (15.0, 25.0)
    assert abs(obs.confidence - 0.92) < 1e-6
    assert obs.handedness == "Right"


@pytest.mark.slow
def test_hand_detector_runs_on_fixture() -> None:
    """MediaPipe smoke test on the synthetic fixture.

    The fixture has no real hands, so the result list will likely be empty.
    We only assert: no exception is raised and the return type is correct.
    """
    if not os.path.exists(FIXTURE_VIDEO):
        pytest.skip("Fixture video not found — run tests/fixtures/generate_fixture.py")

    from pipeline.frame_reader import FrameReader
    from pipeline.hands import HandDetector

    with FrameReader(FIXTURE_VIDEO) as reader:
        for frame_index, frame_bgr in reader:
            first_frame = frame_bgr
            first_index = frame_index
            break

    with HandDetector() as detector:
        result = detector.detect(first_index, first_frame)

    assert isinstance(result, list)
    for obs in result:
        assert isinstance(obs, HandObservation)
