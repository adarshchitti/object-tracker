from __future__ import annotations

"""Generate a synthetic 2-second test video for pipeline tests.

Creates tests/fixtures/sample_short.mp4:
  - 640x480, 30 fps, 60 frames
  - A colored rectangle moving left-to-right (simulates a moving object)
  - A small stationary rectangle in the top-left corner
"""

import os
import sys

import cv2
import numpy as np

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample_short.mp4")

WIDTH, HEIGHT = 640, 480
FPS = 30
FRAME_COUNT = 60


def generate() -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        print(f"ERROR: cannot open VideoWriter for {OUTPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    for i in range(FRAME_COUNT):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Moving rectangle: travels from x=0 to x=WIDTH-100 over 60 frames
        x_offset = int(i * (WIDTH - 100) / (FRAME_COUNT - 1))
        cv2.rectangle(frame, (x_offset, 180), (x_offset + 100, 300), (0, 200, 255), -1)

        # Stationary rectangle in the top-left corner
        cv2.rectangle(frame, (10, 10), (60, 60), (0, 255, 0), -1)

        writer.write(frame)

    writer.release()
    print(f"Generated {OUTPUT_PATH}  ({FRAME_COUNT} frames, {FPS} fps)")


if __name__ == "__main__":
    generate()
