from __future__ import annotations

import os

import pytest

FIXTURE_VIDEO = os.path.join(os.path.dirname(__file__), "fixtures", "sample_short.mp4")


@pytest.fixture(scope="module")
def fixture_path() -> str:
    if not os.path.exists(FIXTURE_VIDEO):
        pytest.skip("Fixture video not found — run tests/fixtures/generate_fixture.py")
    return FIXTURE_VIDEO


def test_metadata(fixture_path: str) -> None:
    from pipeline.frame_reader import FrameReader

    with FrameReader(fixture_path) as reader:
        meta = reader.metadata
    assert meta["frame_count"] == 60
    assert abs(meta["fps"] - 30.0) < 1.0
    assert meta["width"] == 640
    assert meta["height"] == 480


def test_iteration_stride_1(fixture_path: str) -> None:
    from pipeline.frame_reader import FrameReader

    with FrameReader(fixture_path, stride=1) as reader:
        frames = [(idx, frame) for idx, frame in reader]

    assert len(frames) == 60
    indices = [idx for idx, _ in frames]
    assert indices == list(range(60))


def test_iteration_stride_3(fixture_path: str) -> None:
    from pipeline.frame_reader import FrameReader

    with FrameReader(fixture_path, stride=3) as reader:
        frames = [(idx, frame) for idx, frame in reader]

    assert len(frames) == 20
    indices = [idx for idx, _ in frames]
    assert indices == list(range(0, 60, 3))
