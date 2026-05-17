"""Shared-frame bridge for the native spec3 virtual camera extension."""

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path

import cv2
import numpy as np


APP_SUPPORT_NAME = "spec3 correction"
FRAME_MAGIC = 0x53335033  # "SP3" with a stable little-endian marker.
FRAME_VERSION = 1
FRAME_HEADER_SIZE = 64
FRAME_FORMAT_BGRA = 1
FRAME_HEADER_STRUCT = struct.Struct("<IIIIIIIIQQ")


def default_frame_path() -> Path:
    support_dir = os.environ.get("SPEC3_SUPPORT_DIR")
    if support_dir:
        base = Path(support_dir)
    else:
        base = Path.home() / "Library" / "Application Support" / APP_SUPPORT_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base / "virtual_camera_frame.bgra"


class VirtualCameraFrameSink:
    """Write the latest corrected frame into a memory-mapped BGRA file.

    A future CoreMediaIO camera extension can read this file without touching the
    Python/OpenCV process. The sequence number is written odd while a frame is in
    progress and even after the frame is complete, so readers can ignore torn
    frames.
    """

    def __init__(self, path: Path | None = None, enabled: bool | None = None):
        if enabled is None:
            enabled = os.environ.get("SPEC3_VIRTUAL_CAMERA_ENABLED", "1") != "0"
        self.enabled = enabled
        self.path = path or default_frame_path()
        self._mmap: mmap.mmap | None = None
        self._size = 0
        self._width = 0
        self._height = 0
        self._stride = 0
        self._sequence = 0

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if not self.enabled or frame_bgr is None or frame_bgr.size == 0:
            return
        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return

        frame_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        stride = int(frame_bgra.strides[0])
        frame_size = stride * height
        total_size = FRAME_HEADER_SIZE + frame_size
        self._ensure_mapping(width, height, stride, total_size)
        if self._mmap is None:
            return

        self._sequence += 1
        if self._sequence % 2 == 0:
            self._sequence += 1

        timestamp_ns = time.monotonic_ns()
        self._write_header(self._sequence, width, height, stride, timestamp_ns)
        self._mmap[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + frame_size] = frame_bgra.reshape(-1).tobytes()
        self._sequence += 1
        self._write_header(self._sequence, width, height, stride, timestamp_ns)

    def _ensure_mapping(self, width: int, height: int, stride: int, total_size: int) -> None:
        if (
            self._mmap is not None
            and self._size == total_size
            and self._width == width
            and self._height == height
            and self._stride == stride
        ):
            return

        self.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            f.truncate(total_size)
        fd = os.open(self.path, os.O_RDWR)
        try:
            self._mmap = mmap.mmap(fd, total_size)
        finally:
            os.close(fd)
        self._size = total_size
        self._width = width
        self._height = height
        self._stride = stride

    def _write_header(
        self,
        sequence: int,
        width: int,
        height: int,
        stride: int,
        timestamp_ns: int,
    ) -> None:
        if self._mmap is None:
            return
        header = FRAME_HEADER_STRUCT.pack(
            FRAME_MAGIC,
            FRAME_VERSION,
            FRAME_HEADER_SIZE,
            width,
            height,
            stride,
            FRAME_FORMAT_BGRA,
            0,
            sequence,
            timestamp_ns,
        )
        self._mmap[: len(header)] = header
