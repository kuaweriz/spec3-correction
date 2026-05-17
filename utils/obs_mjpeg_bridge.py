"""Local MJPEG bridge for OBS Virtual Camera.

OBS already ships a signed macOS virtual camera. This bridge lets spec3 feed a
clean corrected preview into OBS through a local browser source, avoiding a
custom system extension for day-to-day use.
"""

from __future__ import annotations

import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 29339


class ObsMjpegBridge:
    """Serve the latest corrected frame to OBS as an MJPEG stream."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        enabled: bool | None = None,
        fps: float = 30.0,
    ):
        if enabled is None:
            enabled = os.environ.get("SPEC3_OBS_BRIDGE_ENABLED", "1") != "0"
        self.enabled = enabled
        self.host = host
        self.port = port
        self.fps = max(5.0, min(float(fps), 30.0))
        self.url = f"http://{self.host}:{self.port}/"
        self._lock = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._sequence = 0
        self._last_encode = 0.0
        self._stream_clients = 0
        self._last_frame_request = 0.0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def has_clients(self) -> bool:
        with self._lock:
            return self._stream_clients > 0 or time.monotonic() - self._last_frame_request < 2.0

    def start(self) -> None:
        if not self.enabled or self._server is not None:
            return

        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path in ("/", "/index.html"):
                    self._send_html()
                elif self.path.startswith("/stream.mjpg"):
                    self._send_stream()
                elif self.path.startswith("/frame.jpg"):
                    self._send_frame()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def _send_html(self) -> None:
                html = b"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #05070a; overflow: hidden; }
    img { width: 100vw; height: 100vh; object-fit: contain; display: block; background: #05070a; }
  </style>
</head>
<body><img src="/stream.mjpg" alt="spec3 correction"></body>
</html>
"""
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def _send_frame(self) -> None:
                with bridge._lock:
                    bridge._last_frame_request = time.monotonic()
                    frame = bridge._latest_jpeg
                if frame is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame yet")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)

            def _send_stream(self) -> None:
                boundary = "spec3frame"
                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()

                with bridge._lock:
                    bridge._stream_clients += 1
                    last_sequence = -1
                try:
                    while bridge.enabled:
                        with bridge._lock:
                            bridge._lock.wait_for(
                                lambda: bridge._latest_jpeg is not None and bridge._sequence != last_sequence,
                                timeout=2.0,
                            )
                            frame = bridge._latest_jpeg
                            last_sequence = bridge._sequence
                        if frame is None:
                            continue
                        part_header = (
                            f"--{boundary}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(frame)}\r\n\r\n"
                        ).encode("ascii")
                        self.wfile.write(part_header)
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    pass
                finally:
                    with bridge._lock:
                        bridge._stream_clients = max(0, bridge._stream_clients - 1)

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            self.enabled = False
            return
        self._thread = threading.Thread(target=self._server.serve_forever, name="spec3-obs-mjpeg", daemon=True)
        self._thread.start()

    def update(self, frame_bgr: np.ndarray) -> None:
        if not self.enabled or frame_bgr is None or frame_bgr.size == 0:
            return
        if not self.has_clients:
            return

        now = time.monotonic()
        min_interval = 1.0 / self.fps
        if now - self._last_encode < min_interval:
            return
        self._last_encode = now

        encode_frame = frame_bgr
        height, width = encode_frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280 / max(width, 1), 720 / max(height, 1))
            encode_frame = cv2.resize(
                encode_frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(".jpg", encode_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = encoded.tobytes()
            self._sequence += 1
            self._lock.notify_all()

    def close(self) -> None:
        self.enabled = False
        with self._lock:
            self._lock.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=0.8)
            self._thread = None
