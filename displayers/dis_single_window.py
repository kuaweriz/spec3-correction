"""
Single Window Gaze Corrector

A simplified gaze correction application that:
- Uses a single window (no socket communication)
- Allows real-time toggle of gaze correction with 'g' key
- Supports injectable face predictor backends
- Decoupled architecture: FacePredictor -> GazeCorrector
- Calibration mode for camera offset adjustment
"""

from __future__ import annotations

import json
import os
import math
import subprocess
import threading
import time
import cv2
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - OpenCV fallback for minimal installs
    Image = None
    ImageDraw = None
    ImageFont = None

from utils.logger import Logger
from utils.camera_selection import (
    CameraInfo,
    camera_name,
    is_virtual_camera_name,
    list_macos_cameras,
    load_saved_camera_name,
    physical_camera_options,
    save_camera_id,
)
from displayers.face_predictor import (
    FacePredictor,
    EyeExtractionConfig,
    create_face_predictor,
)
from model_managers.gaze_corrector_v1 import GazeCorrector


################################################################################
# Configuration
################################################################################


UI_FONT_FAMILIES = {
    "rounded": ("/System/Library/Fonts/SFNSRounded.ttf", "/System/Library/Fonts/SFNSRounded.ttf"),
    "system": ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/SFNS.ttf"),
    "compact": ("/System/Library/Fonts/SFCompact.ttf", "/System/Library/Fonts/SFCompact.ttf"),
    "mono": ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/SFNSMono.ttf"),
    "serif": ("/System/Library/Fonts/NewYork.ttf", "/System/Library/Fonts/NewYork.ttf"),
}


UI_STYLE_PALETTES = {
    "orange": {"accent": (68, 148, 255), "live": (92, 178, 232), "glow": (33, 46, 66)},
    "blue": {"accent": (255, 132, 43), "live": (210, 168, 86), "glow": (54, 56, 87)},
    "mint": {"accent": (132, 200, 45), "live": (205, 174, 80), "glow": (38, 66, 52)},
    "violet": {"accent": (255, 105, 155), "live": (116, 190, 250), "glow": (62, 48, 78)},
    "graphite": {"accent": (176, 166, 150), "live": (126, 182, 216), "glow": (56, 58, 64)},
}


@lru_cache(maxsize=96)
def _load_ui_font(size: int, bold: bool = False, font_style: str = "rounded"):
    if ImageFont is None:
        return None
    regular_path, bold_path = UI_FONT_FAMILIES.get(font_style, UI_FONT_FAMILIES["rounded"])
    font_path = bold_path if bold else regular_path
    try:
        return ImageFont.truetype(font_path, size=size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=2048)
def _measure_ui_text(text: str, size: int, bold: bool = False, font_style: str = "rounded") -> tuple[int, int]:
    font = _load_ui_font(size, bold, font_style)
    if font is not None and Image is not None and ImageDraw is not None:
        try:
            scratch = Image.new("RGBA", (1, 1))
            draw = ImageDraw.Draw(scratch)
            bbox = draw.textbbox((0, 0), text, font=font, anchor="ls")
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            pass
    scale = size / 34.0
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]


@lru_cache(maxsize=1024)
def _render_ui_text_patch(
    text: str,
    size: int,
    color: tuple[int, int, int],
    bold: bool = False,
    font_style: str = "rounded",
) -> tuple[np.ndarray, int, int] | None:
    font = _load_ui_font(size, bold, font_style)
    if font is None or Image is None or ImageDraw is None:
        return None

    try:
        scratch = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(scratch)
        bbox = draw.textbbox((0, 0), text, font=font, anchor="ls")
    except Exception:
        return None

    pad = 3
    patch_w = max(1, bbox[2] - bbox[0] + pad * 2)
    patch_h = max(1, bbox[3] - bbox[1] + pad * 2)
    patch = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    rgb = (int(color[2]), int(color[1]), int(color[0]), 255)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=rgb, anchor="ls")
    return np.asarray(patch, dtype=np.float32), bbox[0] - pad, bbox[1] - pad


@dataclass
class DisplayConfig:
    """Configuration for the display application."""

    video_size: tuple[int, int] = (640, 480)
    face_detect_size: tuple[int, int] = (320, 240)
    processing_max_size: tuple[int, int] = (1280, 720)
    window_name: str = "spec3 correction"

    @property
    def x_ratio(self) -> float:
        return self.video_size[0] / self.face_detect_size[0]

    @property
    def y_ratio(self) -> float:
        return self.video_size[1] / self.face_detect_size[1]


@dataclass
class CalibrationConfig:
    """Configuration for calibration mode."""

    step_xy: float = 0.5      # cm per key press for X/Y
    step_z: float = 0.5       # cm per key press for Z
    step_focal: float = 10.0  # pixels per key press for focal length


class CameraFrameStream:
    """Continuously read the latest camera frame so processing never builds a backlog."""

    def __init__(self, cap, logger: Logger):
        self.cap = cap
        self.logger = logger
        self._lock = threading.Lock()
        self._running = True
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._last_frame_time = 0.0
        self._failed_reads = 0
        self._thread = threading.Thread(target=self._reader, name="spec3-camera-stream", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            now = time.monotonic()
            with self._lock:
                if ret and frame is not None:
                    self._frame = frame
                    self._frame_id += 1
                    self._last_frame_time = now
                    self._failed_reads = 0
                else:
                    self._failed_reads += 1
            if not ret:
                time.sleep(0.01)

    def snapshot(self) -> tuple[int, np.ndarray | None, float, int]:
        with self._lock:
            frame = self._frame
            frame_id = self._frame_id
            last_frame_time = self._last_frame_time
            failed_reads = self._failed_reads

        age = time.monotonic() - last_frame_time if last_frame_time > 0.0 else 999.0
        return frame_id, frame, age, failed_reads

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=0.5)
        self.cap.release()


################################################################################
# Single Window Gaze Corrector
################################################################################


class SingleWindowGazeCorrector:
    """
    Single-window gaze correction application with real-time toggle.

    This class orchestrates:
    - FacePredictor: Detects faces and extracts eye data
    - GazeCorrector: Applies gaze correction model

    Controls:
        - 'g': Toggle gaze correction on/off
        - 'c': Toggle calibration mode
        - 'q': Quit

    Calibration Mode Controls:
        - Arrow keys: Adjust X/Y offset
        - '+'/'-': Adjust Z offset
        - '['/']': Adjust focal length
        - 'r': Reset to default
    """

    # Arrow key codes (platform-dependent)
    # KEY_UP = 82
    # KEY_DOWN = 84
    # KEY_LEFT = 81
    # KEY_RIGHT = 83
    KEY_UP = 0
    KEY_DOWN = 1
    KEY_LEFT = 2
    KEY_RIGHT = 3

    def __init__(
        self,
        face_predictor: Optional[FacePredictor] = None,
        gaze_corrector: Optional[GazeCorrector] = None,
        display_config: Optional[DisplayConfig] = None,
        calibration_config: Optional[CalibrationConfig] = None,
        camera_id: int = 0,
        config_path: str = "./model_managers/gaze_corrector_v1_01.yaml",
    ):
        self.logger = Logger(self.__class__.__name__)
        self.display_cfg = display_config or DisplayConfig()
        self.calib_cfg = calibration_config or CalibrationConfig()

        # Initialize face predictor (injectable)
        self.face_predictor = face_predictor or create_face_predictor("dlib")
        self.logger.log(f"Using face predictor: {self.face_predictor.get_name()}")

        # Initialize gaze corrector (injectable)
        self.gaze_corrector = gaze_corrector or GazeCorrector(config_path=config_path)

        # Eye extraction config (matches model requirements)
        self.eye_config = EyeExtractionConfig()

        # State
        self.gaze_correction_enabled = True
        self.calibration_mode = False
        self.camera_id = camera_id
        self.settings_window_name = self.display_cfg.window_name
        self.should_quit = False
        self._dragging_control: Optional[str] = None
        self._slider_regions: dict[str, tuple[int, int, int, float, float]] = {}
        self._button_regions: dict[str, tuple[int, int, int, int]] = {}
        self.control_panel_width = 430
        self.preview_max_size = (1280, 720)
        self._panel_origin_x = 0
        self._last_canvas_size: tuple[int, int] | None = None
        self._pending_window_resize_size: tuple[int, int] | None = None
        self._window_resize_retries = 0
        self.window_visible = True
        self._window_created = False
        self.control_file_path = os.environ.get("GAZE_CONTROL_FILE")
        self.native_request_file_path = os.environ.get("SPEC3_NATIVE_REQUEST_FILE")
        self.log_file_path = os.environ.get("SPEC3_LOG_FILE") or os.path.expanduser(
            "~/Library/Logs/spec3 correction.log"
        )
        self.preferences_file_path = os.environ.get("SPEC3_PREFERENCES_FILE") or os.path.expanduser(
            "~/Library/Application Support/spec3 correction/preferences.json"
        )
        self._preferences_mtime = 0.0
        self._ui_theme = "dark"
        self._ui_language = "en"
        self._ui_style = "orange"
        self._ui_font_style = "rounded"
        self._last_control_mtime = 0.0
        self._pending_hide_window = False
        self._pending_show_window = False
        self._preview_region: tuple[int, int, int, int] | None = None
        self._dragging_preview_aim = False
        self._aim_target_offsets: tuple[float, float] | None = None
        self._aim_applied_offsets: tuple[float, float] | None = None
        self._aim_save_when_settled = False
        self._last_aim_update = time.monotonic()
        self._aim_visible_until = 0.0
        self._pending_camera_id: Optional[int] = None
        self._pending_camera_label = ""
        self.camera_dropdown_open = False
        self.camera_options: list[CameraInfo] = []
        self._all_camera_options: list[CameraInfo] = []
        saved_camera_label = load_saved_camera_name()
        if saved_camera_label and is_virtual_camera_name(saved_camera_label):
            saved_camera_label = ""
        self.camera_label = saved_camera_label or f"Camera {self.camera_id}"
        self._camera_label_overrides: dict[int, str] = {}
        if saved_camera_label:
            self._camera_label_overrides[self.camera_id] = saved_camera_label
        self._camera_status_message = "Camera ready"
        self._camera_status_until = 0.0
        self._camera_switching = False
        self._camera_refresh_lock = threading.Lock()
        self._camera_refreshing = False
        self._camera_options_refreshed_at = 0.0
        self._camera_refresh_interval = 8.0
        self.refresh_camera_options(async_refresh=True)

        # Store default values for reset
        self.default_camera_offset = self.gaze_corrector.get_camera_offset()
        self.default_focal_length = self.gaze_corrector.get_focal_length()

    def create_settings_panel(self) -> None:
        """Create the single application window with camera and controls."""
        cv2.namedWindow(self.display_cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.display_cfg.window_name, self.handle_settings_mouse)
        self.window_visible = True
        self._window_created = True

    def apply_settings_panel(self) -> None:
        """Kept for compatibility with older app loops."""

    def draw_settings_panel(self) -> None:
        """Kept for compatibility; the panel is now drawn into the main canvas."""

    def request_show_window(self) -> None:
        self._pending_show_window = True
        self._pending_hide_window = False

    def request_hide_window(self) -> None:
        self._pending_hide_window = True
        self._pending_show_window = False

    def apply_pending_window_actions(self) -> None:
        """Run window create/destroy outside OpenCV mouse callbacks."""
        if self._pending_hide_window:
            self._pending_hide_window = False
            self.hide_window()
        if self._pending_show_window:
            self._pending_show_window = False
            self.show_window()

    def show_window(self) -> None:
        """Show or recreate the camera window while keeping the camera process alive."""
        if not self._window_created:
            self.create_settings_panel()
        self.window_visible = True
        self._last_canvas_size = None

    def hide_window(self, already_closed: bool = False) -> None:
        """Hide the camera window but keep capture/model processing alive."""
        self.window_visible = False
        self._window_created = False
        self._last_canvas_size = None
        self._dragging_control = None
        self._dragging_preview_aim = False
        self._aim_target_offsets = None
        self._aim_applied_offsets = None
        self._aim_save_when_settled = False
        self._slider_regions.clear()
        self._button_regions.clear()
        if not already_closed:
            try:
                cv2.destroyWindow(self.display_cfg.window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

    def _apply_camera_options(
        self,
        cameras: list[CameraInfo],
        *,
        refresh_finished: bool = False,
    ) -> None:
        """Update the camera list without blocking the UI thread."""
        with self._camera_refresh_lock:
            active_id = self._pending_camera_id if self._pending_camera_id is not None else self.camera_id
            if cameras:
                self._all_camera_options = list(cameras)
                self.camera_options = self._camera_options_for_display(cameras, active_id)
            elif not self.camera_options:
                self.camera_options = [CameraInfo(self.camera_id, f"Camera {self.camera_id}")]

            active_name = camera_name(active_id, cameras) if cameras else f"Camera {active_id}"
            has_physical = any(not is_virtual_camera_name(camera.name) for camera in cameras)
            active_is_hidden_virtual = has_physical and is_virtual_camera_name(active_name)
            if all(camera.id != active_id for camera in self.camera_options) and not active_is_hidden_virtual:
                self.camera_options.append(CameraInfo(active_id, f"Camera {active_id}"))
                self.camera_options = self._deduplicate_camera_options(self.camera_options, active_id)

            if active_is_hidden_virtual and self.camera_options:
                physical = self.camera_options[0]
                if self._pending_camera_id is None:
                    self._pending_camera_id = physical.id
                    self._pending_camera_label = physical.name
                    self._camera_switching = True
                active_id = physical.id

            self.camera_label = self._camera_label_overrides.get(
                active_id,
                camera_name(active_id, self.camera_options),
            )
            self._camera_options_refreshed_at = time.monotonic()
            if refresh_finished:
                self._camera_refreshing = False

    def _start_camera_refresh(self, force: bool = False) -> None:
        with self._camera_refresh_lock:
            if self._camera_refreshing:
                return
            self._camera_refreshing = True

        def worker() -> None:
            cameras = list_macos_cameras(force_refresh=force)
            self._apply_camera_options(cameras, refresh_finished=True)

        threading.Thread(target=worker, name="spec3-camera-refresh", daemon=True).start()

    @staticmethod
    def _camera_label_key(label: str) -> str:
        return label.replace("\xa0", " ").strip().casefold()

    def _camera_options_for_display(
        self,
        cameras: list[CameraInfo],
        active_id: int,
    ) -> list[CameraInfo]:
        cameras = physical_camera_options(cameras)
        options = [
            CameraInfo(camera.id, self._camera_label_overrides.get(camera.id, camera.name))
            for camera in cameras
        ]
        return self._deduplicate_camera_options(options, active_id)

    def _deduplicate_camera_options(
        self,
        options: list[CameraInfo],
        active_id: int,
    ) -> list[CameraInfo]:
        """Hide stale native slots after OpenCV remaps a physical camera."""
        active_label = self._camera_label_overrides.get(active_id)
        if not active_label:
            return options

        active_key = self._camera_label_key(active_label)
        cleaned: list[CameraInfo] = []
        skipped_duplicate = False
        for camera in options:
            is_duplicate = (
                camera.id != active_id
                and self._camera_label_key(camera.name) == active_key
            )
            if is_duplicate:
                skipped_duplicate = True
                continue
            cleaned.append(camera)

        if skipped_duplicate and all(camera.id != active_id for camera in cleaned):
            cleaned.insert(0, CameraInfo(active_id, active_label))
        return cleaned

    def refresh_camera_options(self, force: bool = False, async_refresh: bool = False) -> None:
        """Refresh camera names shown in the in-app selector."""
        with self._camera_refresh_lock:
            is_fresh = (
                bool(self.camera_options)
                and time.monotonic() - self._camera_options_refreshed_at < self._camera_refresh_interval
            )
        if not force and is_fresh:
            return

        if async_refresh:
            self._start_camera_refresh(force=force)
            return

        self._apply_camera_options(list_macos_cameras(force_refresh=force))

    def toggle_camera_dropdown(self) -> None:
        self.camera_dropdown_open = not self.camera_dropdown_open
        if self.camera_dropdown_open:
            self.refresh_camera_options(force=True, async_refresh=True)
            self.logger.log("Camera dropdown opened")

    def request_camera_select(self, camera_id: int, label: str) -> None:
        """Queue a camera switch from the UI; applied in the main loop."""
        self.camera_dropdown_open = False
        self.camera_label = label
        self._set_camera_status(f"Switching to {self._short_label(label, 28)}...", 2.5)
        if camera_id != self.camera_id:
            self._pending_camera_id = camera_id
            self._pending_camera_label = label
            self._camera_switching = True
        else:
            save_camera_id(self.camera_id, label)
            self._camera_label_overrides[self.camera_id] = label
            self._set_camera_status("Already selected", 1.4)

    def _short_camera_label(self, max_chars: int = 25) -> str:
        label = self.camera_label.replace("\xa0", " ")
        return self._short_label(label, max_chars)

    def _short_label(self, label: str, max_chars: int) -> str:
        return label if len(label) <= max_chars else label[: max_chars - 3].rstrip() + "..."

    def _set_camera_status(self, message: str, ttl: float = 2.0) -> None:
        self._camera_status_message = message
        self._camera_status_until = time.monotonic() + ttl

    def _camera_status_text(self) -> str:
        if self._camera_switching:
            return "Opening camera..."
        if self._camera_refreshing:
            return "Scanning cameras..."
        if time.monotonic() < self._camera_status_until:
            return self._camera_status_message
        return f"Selected slot #{self.camera_id}"

    def _remember_camera_label(self, camera_id: int, label: str) -> None:
        clean_label = label.strip() or f"Camera {camera_id}"
        self._camera_label_overrides[camera_id] = clean_label
        with self._camera_refresh_lock:
            replaced = False
            updated_options: list[CameraInfo] = []
            for camera in self.camera_options:
                if camera.id == camera_id:
                    updated_options.append(CameraInfo(camera.id, clean_label))
                    replaced = True
                else:
                    updated_options.append(camera)
            if not replaced:
                updated_options.append(CameraInfo(camera_id, clean_label))
            self.camera_options = self._deduplicate_camera_options(updated_options, camera_id)

    def _fit_text(self, text: str, max_width: int, size: int, bold: bool = False) -> str:
        if self._measure_text(text, size, bold)[0] <= max_width:
            return text
        trimmed = text
        while len(trimmed) > 4:
            trimmed = trimmed[:-1].rstrip()
            candidate = trimmed + "..."
            if self._measure_text(candidate, size, bold)[0] <= max_width:
                return candidate
        return "..."

    def _measure_text(self, text: str, size: int, bold: bool = False) -> tuple[int, int]:
        return _measure_ui_text(text, size, bold, self._ui_font_style)

    def _draw_text(
        self,
        image: np.ndarray,
        text: str,
        org: tuple[int, int],
        size: int,
        color: tuple[int, int, int],
        bold: bool = False,
    ) -> None:
        """Draw sharper macOS text with a safe OpenCV fallback."""
        rendered = _render_ui_text_patch(text, size, tuple(color), bold, self._ui_font_style)
        if rendered is None:
            scale = size / 34.0
            thickness = 2 if bold else 1
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            return

        x, y = org
        patch_np, offset_x, offset_y = rendered
        left = x + offset_x
        top = y + offset_y
        right = left + patch_np.shape[1]
        bottom = top + patch_np.shape[0]

        dst_left = max(left, 0)
        dst_top = max(top, 0)
        dst_right = min(right, image.shape[1])
        dst_bottom = min(bottom, image.shape[0])
        if dst_right <= dst_left or dst_bottom <= dst_top:
            return

        src_x1 = dst_left - left
        src_y1 = dst_top - top
        src_x2 = src_x1 + (dst_right - dst_left)
        src_y2 = src_y1 + (dst_bottom - dst_top)
        src = patch_np[src_y1:src_y2, src_x1:src_x2]
        alpha = src[:, :, 3:4] / 255.0
        color_bgr = src[:, :, :3][:, :, ::-1]
        roi = image[dst_top:dst_bottom, dst_left:dst_right].astype(np.float32)
        image[dst_top:dst_bottom, dst_left:dst_right] = np.clip(
            color_bgr * alpha + roi * (1.0 - alpha), 0, 255
        ).astype(np.uint8)

    def _draw_round_rect(
        self,
        image: np.ndarray,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        radius: int = 7,
        border_color: tuple[int, int, int] | None = None,
        border: int = 1,
    ) -> None:
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            return
        radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        if radius == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)

        if border_color is not None and border > 0:
            if radius == 0:
                cv2.rectangle(image, (x1, y1), (x2, y2), border_color, border, cv2.LINE_AA)
                return
            cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), border_color, border, cv2.LINE_AA)
            cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), border_color, border, cv2.LINE_AA)
            cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), border_color, border, cv2.LINE_AA)
            cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), border_color, border, cv2.LINE_AA)
            cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border_color, border, cv2.LINE_AA)
            cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border_color, border, cv2.LINE_AA)
            cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border_color, border, cv2.LINE_AA)
            cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border_color, border, cv2.LINE_AA)

    def set_correction_enabled(self, enabled: bool) -> None:
        self.gaze_correction_enabled = enabled
        self.gaze_corrector.set_tuning(enabled=enabled)

    def process_control_command(self) -> None:
        """Handle menu-bar commands written by the native macOS launcher."""
        if not self.control_file_path:
            return
        try:
            mtime = os.path.getmtime(self.control_file_path)
        except OSError:
            return
        if mtime <= self._last_control_mtime:
            return
        self._last_control_mtime = mtime

        try:
            with open(self.control_file_path, "r", encoding="utf-8") as f:
                command = f.readline().strip().lower()
        except OSError:
            return

        if command == "show":
            self.request_show_window()
        elif command == "hide":
            self.request_hide_window()
        elif command == "enable":
            self.set_correction_enabled(True)
        elif command == "disable":
            self.set_correction_enabled(False)
        elif command == "toggle":
            self.toggle_correction()
        elif command == "logs":
            self.open_logs()
        elif command == "training":
            self.open_training_window()
        elif command == "settings":
            self.open_settings_window()
        elif command == "record_read":
            self.set_ai_training_label("read")
        elif command == "record_live":
            self.set_ai_training_label("live")
        elif command == "record_glance":
            self.set_ai_training_label("glance")
        elif command == "record_stop":
            self.set_ai_training_label(None)
        elif command == "train_ai":
            self.train_personal_ai()
        elif command == "reset_training":
            self.reset_personal_ai_samples()
        elif command == "quit":
            self.request_quit()

    def refresh_preferences(self) -> None:
        try:
            mtime = os.path.getmtime(self.preferences_file_path)
        except OSError:
            return
        if mtime <= self._preferences_mtime:
            return
        self._preferences_mtime = mtime
        try:
            with open(self.preferences_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        theme = str(data.get("theme", "dark")).lower()
        language = str(data.get("language", "en")).lower()
        style = str(data.get("style", "orange")).lower()
        font_style = str(data.get("font", "rounded")).lower()
        self._ui_theme = "light" if theme == "light" else "dark"
        self._ui_language = "ru" if language == "ru" else "en"
        self._ui_style = style if style in UI_STYLE_PALETTES else "orange"
        self._ui_font_style = font_style if font_style in UI_FONT_FAMILIES else "rounded"

    def tr(self, en: str, ru: str) -> str:
        return ru if self._ui_language == "ru" else en

    def _ui_palette(self) -> dict[str, tuple[int, int, int]]:
        return UI_STYLE_PALETTES.get(self._ui_style, UI_STYLE_PALETTES["orange"])

    def compose_app_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compose the camera preview and the control panel into one window."""
        frame_h, frame_w = frame.shape[:2]
        max_w, max_h = self.preview_max_size
        scale = min(max_w / max(frame_w, 1), max_h / max(frame_h, 1))
        scale = max(0.25, min(scale, 2.0))
        preview_w = max(1, int(frame_w * scale))
        preview_h = max(1, int(frame_h * scale))

        if (preview_w, preview_h) != (frame_w, frame_h):
            interpolation = cv2.INTER_AREA if preview_w < frame_w or preview_h < frame_h else cv2.INTER_LINEAR
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=interpolation)
        else:
            preview = frame

        canvas_h = max(preview_h, 720)
        canvas_w = preview_w + self.control_panel_width
        canvas = np.full((canvas_h, canvas_w, 3), (16, 18, 23), dtype=np.uint8)

        preview_y = (canvas_h - preview_h) // 2
        canvas[preview_y:preview_y + preview_h, 0:preview_w] = preview
        self._preview_region = (0, preview_y, preview_w, preview_y + preview_h)
        cv2.rectangle(canvas, (0, preview_y), (preview_w - 1, preview_y + preview_h - 1), (44, 49, 58), 1)

        self._panel_origin_x = preview_w
        cv2.line(canvas, (preview_w, 0), (preview_w, canvas_h), (54, 60, 72), 1, cv2.LINE_AA)
        self.draw_control_panel(canvas, preview_w, canvas_h)

        canvas_size = (canvas_w, canvas_h)
        if self._last_canvas_size != canvas_size:
            cv2.resizeWindow(self.display_cfg.window_name, canvas_w, canvas_h)
            self._last_canvas_size = canvas_size
            self._pending_window_resize_size = canvas_size
            self._window_resize_retries = 12
            self.logger.log(f"Window canvas size set to {canvas_w}x{canvas_h}")

        return canvas

    def _apply_pending_window_resize(self) -> None:
        if self._pending_window_resize_size is None or self._window_resize_retries <= 0:
            return
        width, height = self._pending_window_resize_size
        try:
            cv2.resizeWindow(self.display_cfg.window_name, width, height)
        except cv2.error:
            return
        self._window_resize_retries -= 1
        if self._window_resize_retries <= 0:
            self._pending_window_resize_size = None

    def draw_control_panel(self, canvas: np.ndarray, x0: int, height: int) -> None:
        """Draw the integrated control panel on the right side of the app."""
        self.refresh_preferences()
        tuning = self.gaze_corrector.get_tuning()
        reading_active, reading_score, effective_hold = self.gaze_corrector.get_reading_state()
        personal_ai = self.gaze_corrector.get_personal_ai_state()
        panel_w = self.control_panel_width
        x1 = x0 + panel_w
        compact = height <= 760
        margin = 24 if compact else 28
        left = x0 + margin
        right = x1 - margin
        self._slider_regions.clear()
        self._button_regions.clear()

        light = self._ui_theme == "light"
        panel_bg = (238, 241, 245) if light else (18, 19, 23)
        header_bg = (249, 250, 252) if light else (25, 25, 31)
        card_bg = (255, 255, 255) if light else (31, 31, 38)
        card_line = (205, 211, 220) if light else (66, 66, 78)
        muted = (95, 101, 112) if light else (158, 164, 176)
        text = (28, 31, 36) if light else (242, 245, 250)
        palette = self._ui_palette()
        accent = palette["accent"]
        live_accent = palette["live"]

        header_h = 92 if compact else 100
        header_line_y = header_h - 1
        cv2.rectangle(canvas, (x0 + 1, 0), (x1, height), panel_bg, -1)
        cv2.rectangle(canvas, (x0 + 1, 0), (x1, header_h), header_bg, -1)
        line_color = (210, 215, 224) if light else (47, 51, 60)
        cv2.line(canvas, (x0 + 1, header_line_y), (x1, header_line_y), line_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (left, header_h - 16), (right, header_h - 16), line_color, 1, cv2.LINE_AA)

        title_x = left
        title_font = 22 if compact else 28
        subtitle_font = 12 if compact else 13
        title_dot_y = 32 if compact else 36
        cv2.circle(canvas, (title_x + 6, title_dot_y), 5, accent, -1, cv2.LINE_AA)
        cv2.circle(canvas, (title_x + 6, title_dot_y), 9, palette["glow"], 1, cv2.LINE_AA)
        title = self._fit_text("spec3 correction", right - title_x - (116 if compact else 150), title_font, bold=True)
        self._draw_text(canvas, title, (title_x + 22, 39 if compact else 43), title_font, text, bold=True)
        self._draw_text(canvas, self.tr("Reading stabilizer", "Стабилизация чтения"), (title_x + 24, 62 if compact else 70), subtitle_font, muted)

        on_label = "ON" if tuning.enabled and self.gaze_correction_enabled else "OFF"
        self._draw_button(
            canvas,
            "toggle",
            (right - (92 if compact else 108), 22 if compact else 24, right, 56 if compact else 62),
            on_label,
            tuning.enabled and self.gaze_correction_enabled,
        )

        camera_y1, camera_y2 = (108, 158) if compact else (116, 174)
        self._draw_round_rect(canvas, (left, camera_y1, right, camera_y2), card_bg, 5, card_line)
        cv2.rectangle(canvas, (left, camera_y1 + 1), (left + 5, camera_y2 - 1), (92, 86, 82) if not light else (190, 174, 160), -1)
        self._draw_text(canvas, self.tr("Camera", "Камера"), (left + 16, camera_y1 + 22), 13 if compact else 14, muted, bold=True)
        name_x = left + (108 if compact else 142)
        camera_text = self._fit_text(self._short_camera_label(32), right - name_x - 34, 14 if compact else 15, bold=True)
        self._draw_text(canvas, camera_text, (name_x, camera_y1 + 22), 14 if compact else 15, text, bold=True)
        self._draw_text(canvas, self._camera_status_text(), (name_x, camera_y1 + 40), 10 if compact else 11, (150, 158, 170))
        chevron = "^" if self.camera_dropdown_open else "v"
        self._draw_text(canvas, chevron, (right - 20, camera_y1 + 23), 14, muted, bold=True)
        self._button_regions["camera_toggle"] = (left, camera_y1, right, camera_y2)

        mode_label = "READ" if reading_active else "LIVE"
        mode_color = accent if reading_active else live_accent
        mode_y1, mode_y2 = (170, 224) if compact else (188, 252)
        self._draw_round_rect(canvas, (left, mode_y1, right, mode_y2), card_bg, 5, card_line)
        cv2.rectangle(canvas, (left, mode_y1 + 1), (left + 5, mode_y2 - 1), mode_color, -1)
        mode_title = self._fit_text(self.tr("Reading detector", "Детектор чтения"), right - left - 150, 15)
        self._draw_text(canvas, mode_title, (left + 16, mode_y1 + 23), 14 if compact else 15, text)
        self._draw_text(canvas, mode_label, (right - 82, mode_y1 + 24), 16 if compact else 18, mode_color, bold=True)
        self._draw_meter(canvas, left + 16, mode_y1 + 43, left + 188, reading_score, "read")
        self._draw_meter(canvas, left + 208, mode_y1 + 43, right - 16, effective_hold, "hold")

        sliders = [
            ("strength", self.tr("Stabilizer Power", "Сила стабилизации"), tuning.strength * 100.0, 0.0, 150.0, "%"),
            ("stabilizer", self.tr("Reading Hold", "Удержание чтения"), tuning.reading_stabilizer * 100.0, 0.0, 100.0, "%"),
            ("smooth", self.tr("Smoothness", "Плавность"), tuning.smoothing * 100.0, 0.0, 100.0, "%"),
            ("live", self.tr("Eye Life", "Живость глаз"), tuning.natural_motion * 100.0, 0.0, 100.0, "%"),
        ]

        row1_y = height - (82 if compact else 88)
        row2_y = row1_y + (40 if compact else 44)
        ai_y1 = row1_y - (116 if compact else 100)
        ai_y2 = row1_y - 12

        track_x1 = left + 4
        track_x2 = right - (104 if compact else 112)
        y = 258 if compact else 292
        slider_step = 54 if compact else 64
        for key, label, value, min_value, max_value, suffix in sliders:
            self._draw_slider(canvas, key, label, value, min_value, max_value, suffix, y, track_x1, track_x2)
            y += slider_step

        self._draw_personal_ai_panel(canvas, left, right, ai_y1, ai_y2, personal_ai)

        gap = 10
        col_w = (right - left - gap * 2) // 3
        button_h = 34 if compact else 36
        self._draw_button(canvas, "app_quit", (left, row1_y, left + col_w, row1_y + button_h), self.tr("Quit", "Выход"), False, danger=True)
        self._draw_button(canvas, "logs", (left + col_w + gap, row1_y, left + col_w * 2 + gap, row1_y + button_h), self.tr("Logs", "Логи"), False)
        self._draw_button(canvas, "settings", (left + col_w * 2 + gap * 2, row1_y, right, row1_y + button_h), self.tr("Settings", "Настройки"), False)
        half_w = (right - left - gap) // 2
        self._draw_button(canvas, "reset", (left, row2_y, left + half_w, row2_y + 32), self.tr("Reset", "Сброс"), False)
        self._draw_button(canvas, "hide", (left + half_w + gap, row2_y, right, row2_y + 32), self.tr("Hide", "Скрыть"), False)
        if self.camera_dropdown_open:
            self._draw_camera_dropdown(canvas, left, right, camera_y2 + 8)

    def _draw_personal_ai_panel(
        self,
        canvas: np.ndarray,
        left: int,
        right: int,
        y1: int,
        y2: int,
        state: dict[str, object],
    ) -> None:
        light = self._ui_theme == "light"
        card_bg = (255, 255, 255) if light else (31, 31, 38)
        card_line = (205, 211, 220) if light else (66, 66, 78)
        primary = (35, 39, 46) if light else (230, 234, 242)
        secondary = (92, 99, 112) if light else (164, 171, 184)
        muted = (112, 120, 134) if light else (138, 147, 160)
        accent = self._ui_palette()["accent"]
        recording_label = state.get("recording_label")
        samples = state.get("samples") if isinstance(state.get("samples"), dict) else {}
        training_pool = state.get("training_pool") if isinstance(state.get("training_pool"), dict) else {}
        has_model = bool(state.get("has_model"))
        probability = state.get("personal_probability")
        model_samples = int(state.get("model_samples") or 0)
        model_accuracy = float(state.get("model_accuracy") or 0.0)
        status = str(state.get("last_status") or "")
        if len(status) > 46:
            status = status[:43] + "..."

        model_text = self.tr("model ON", "модель готова") if has_model else self.tr("collect data", "соберите данные")
        if has_model and model_accuracy > 0:
            model_text = f"AI {model_accuracy * 100:.0f}%"
        if isinstance(probability, (int, float)) and not has_model:
            model_text = f"AI {float(probability) * 100:.0f}%"
        read_count = int(samples.get("read", 0)) if isinstance(samples, dict) else 0
        live_count = int(samples.get("live", 0)) if isinstance(samples, dict) else 0
        look_count = int(samples.get("glance", 0)) if isinstance(samples, dict) else 0
        pool_total = 0
        if isinstance(training_pool, dict):
            pool_total = (
                int(training_pool.get("read", 0))
                + int(training_pool.get("live", 0))
                + int(training_pool.get("glance", 0))
            )

        self._draw_round_rect(canvas, (left, y1, right, y2), card_bg, 5, card_line)
        cv2.rectangle(canvas, (left, y1 + 1), (left + 5, y2 - 1), accent if has_model else (84, 88, 102), -1)

        model_text = self._fit_text(model_text, 150, 13, bold=True)
        model_width = self._measure_text(model_text, 13, bold=True)[0]
        self._draw_text(canvas, "Personal AI", (left + 18, y1 + 24), 16, primary, bold=True)
        self._draw_text(canvas, model_text, (right - 18 - model_width, y1 + 24), 13, accent, bold=True)
        target = int(state.get("target_per_label") or 300)
        sample_text = self.tr(
            f"new read {read_count}/{target}  live {live_count}/{target}  look {look_count}/{target}",
            f"новые: чтение {read_count}/{target} live {live_count}/{target} взгляд {look_count}/{target}",
        )
        if pool_total > 0 and model_samples > 0:
            status = self.tr(
                f"History {pool_total} samples, model {model_samples}",
                f"История {pool_total}, модель {model_samples}",
            )
        sample_text = self._fit_text(sample_text, right - left - 142, 11)
        status = self._fit_text(status, right - left - 142, 11)
        self._draw_text(canvas, sample_text, (left + 18, y1 + 46), 11, secondary)
        self._draw_text(canvas, status, (left + 18, y1 + 66), 11, muted)

        self._draw_button(canvas, "ai_training", (right - 114, y1 + 44, right - 14, y1 + 70), self.tr("Train", "Обучение"), bool(recording_label))

    def _draw_camera_dropdown(self, canvas: np.ndarray, left: int, right: int, y: int) -> None:
        """Draw the camera dropdown over the rest of the control panel."""
        with self._camera_refresh_lock:
            camera_options = list(self.camera_options)
            camera_refreshing = self._camera_refreshing

        x1 = left
        x2 = right
        row_h = 34
        max_rows = min(len(camera_options), 5) if camera_options else 1
        dropdown_h = max(1, max_rows) * row_h + 10
        bottom = y + dropdown_h

        light = self._ui_theme == "light"
        dropdown_bg = (249, 250, 252) if light else (24, 24, 30)
        dropdown_line = (198, 205, 217) if light else (82, 86, 98)
        self._draw_round_rect(canvas, (x1, y, x2, bottom), dropdown_bg, 5, dropdown_line)

        if not camera_options:
            label = "Scanning cameras..." if camera_refreshing else "No cameras found"
            color = (70, 76, 88) if light else (218, 222, 230)
            self._draw_text(canvas, label, (x1 + 18, y + 28), 14, color)
            return

        for index, camera in enumerate(camera_options[:max_rows]):
            row_y1 = y + 5 + index * row_h
            row_y2 = row_y1 + row_h
            active = camera.id == self.camera_id
            hover_bg = (237, 240, 245) if light else (39, 37, 34)
            active_bg = (230, 235, 245) if light else (48, 41, 35)
            self._draw_round_rect(canvas, (x1 + 6, row_y1, x2 - 6, row_y2), active_bg if active else hover_bg, 4)
            if active:
                cv2.circle(canvas, (x1 + 20, row_y1 + 18), 5, self._ui_palette()["accent"], -1, cv2.LINE_AA)

            name = self._fit_text(camera.name, x2 - x1 - 108, 14, bold=active)
            name_color = (35, 39, 46) if light else ((248, 248, 250) if active else (218, 222, 230))
            self._draw_text(canvas, name, (x1 + 34, row_y1 + 22), 14, name_color, bold=active)
            id_text = f"slot #{camera.id}"
            self._draw_text(canvas, id_text, (x2 - 62, row_y1 + 22), 12, (146, 152, 164))
            self._button_regions[f"camera_option:{camera.id}"] = (x1 + 6, row_y1, x2 - 6, row_y2)

    def _clamp_to_preview(self, x: int, y: int) -> tuple[int, int] | None:
        if self._preview_region is None:
            return None

        x1, y1, x2, y2 = self._preview_region
        if x2 <= x1 or y2 <= y1:
            return None
        return max(x1, min(x, x2 - 1)), max(y1, min(y, y2 - 1))

    def _is_inside_preview(self, x: int, y: int) -> bool:
        if self._preview_region is None:
            return False
        x1, y1, x2, y2 = self._preview_region
        return x1 <= x < x2 and y1 <= y < y2

    def _preview_point_to_offsets(self, x: int, y: int) -> tuple[float, float] | None:
        point = self._clamp_to_preview(x, y)
        if point is None or self._preview_region is None:
            return None

        px, py = point
        x1, y1, x2, y2 = self._preview_region
        nx = ((px - x1) / max(1, x2 - x1 - 1) - 0.5) * 2.0
        ny = ((py - y1) / max(1, y2 - y1 - 1) - 0.5) * 2.0
        nx = math.copysign(abs(nx) ** 1.18, nx)
        ny = math.copysign(abs(ny) ** 1.18, ny)
        horizontal = max(-45.0, min(nx * 45.0, 45.0))
        vertical = max(-90.0, min(-ny * 90.0, 90.0))
        return vertical, horizontal

    def _offsets_to_preview_point(self, offsets: tuple[float, float] | None = None) -> tuple[int, int] | None:
        if self._preview_region is None:
            return None

        if offsets is None:
            tuning = self.gaze_corrector.get_tuning()
            offsets = (tuning.vertical_offset, tuning.horizontal_offset)
        x1, y1, x2, y2 = self._preview_region
        nx = max(-1.0, min(offsets[1] / 45.0, 1.0))
        ny = max(-1.0, min(-offsets[0] / 90.0, 1.0))
        nx = math.copysign(abs(nx) ** (1.0 / 1.18), nx)
        ny = math.copysign(abs(ny) ** (1.0 / 1.18), ny)
        x = int(x1 + (nx * 0.5 + 0.5) * max(1, x2 - x1 - 1))
        y = int(y1 + (ny * 0.5 + 0.5) * max(1, y2 - y1 - 1))
        return x, y

    def _set_gaze_from_preview_point(self, x: int, y: int, save: bool) -> None:
        offsets = self._preview_point_to_offsets(x, y)
        if offsets is None:
            return

        self._aim_target_offsets = offsets
        self._aim_visible_until = time.monotonic() + 1.2
        if self._aim_applied_offsets is None:
            tuning = self.gaze_corrector.get_tuning()
            self._aim_applied_offsets = (tuning.vertical_offset, tuning.horizontal_offset)

        if save:
            self._aim_save_when_settled = True
            self._update_preview_aim_motion(force=True, save=True)
        elif self._dragging_preview_aim:
            self._update_preview_aim_motion(force=False, save=False)

    def _update_preview_aim_motion(self, force: bool = False, save: bool = False) -> None:
        if self._aim_target_offsets is None:
            return

        now = time.monotonic()
        dt = max(1.0 / 120.0, min(now - self._last_aim_update, 0.10))
        self._last_aim_update = now

        if self._aim_applied_offsets is None:
            tuning = self.gaze_corrector.get_tuning()
            self._aim_applied_offsets = (tuning.vertical_offset, tuning.horizontal_offset)

        current = self._aim_applied_offsets
        target = self._aim_target_offsets
        if force:
            next_offsets = target
        else:
            response = 0.018 if self._dragging_preview_aim else 0.055
            alpha = 1.0 - math.exp(-dt / response)
            raw_next = (
                current[0] * (1.0 - alpha) + target[0] * alpha,
                current[1] * (1.0 - alpha) + target[1] * alpha,
            )
            speed_scale = max(0.5, min(dt * 60.0, 1.7))
            max_vertical_step = (9.5 if self._dragging_preview_aim else 4.4) * speed_scale
            max_horizontal_step = (6.8 if self._dragging_preview_aim else 3.2) * speed_scale
            next_offsets = (
                current[0] + max(-max_vertical_step, min(raw_next[0] - current[0], max_vertical_step)),
                current[1] + max(-max_horizontal_step, min(raw_next[1] - current[1], max_horizontal_step)),
            )

        if (
            not force
            and abs(next_offsets[0] - current[0]) < 0.025
            and abs(next_offsets[1] - current[1]) < 0.025
        ):
            return

        self._aim_applied_offsets = next_offsets
        if self._dragging_preview_aim or save:
            self.gaze_corrector.hold_manual_aim(0.24)
        self.gaze_corrector.set_tuning(
            vertical_offset=next_offsets[0],
            horizontal_offset=next_offsets[1],
            save=save,
            reset_tracking=False,
        )

        if save or (
            not self._dragging_preview_aim
            and abs(next_offsets[0] - target[0]) < 0.08
            and abs(next_offsets[1] - target[1]) < 0.08
        ):
            if save:
                self._aim_save_when_settled = False
            elif self._aim_save_when_settled:
                self.gaze_corrector.save_tuning_settings()
                self._aim_save_when_settled = False
            self._aim_target_offsets = None
            self._aim_applied_offsets = None

    def _draw_preview_aim_overlay(self, canvas: np.ndarray) -> None:
        if not self._dragging_preview_aim and time.monotonic() > self._aim_visible_until:
            return

        point = self._offsets_to_preview_point()
        if point is None or self._preview_region is None:
            return

        x, y = point
        palette = self._ui_palette()
        color = palette["accent"] if self._dragging_preview_aim else palette["live"]
        shadow = (12, 13, 16)

        if self._aim_target_offsets is not None:
            target_point = self._offsets_to_preview_point(self._aim_target_offsets)
            if target_point is not None:
                tx, ty = target_point
                cv2.circle(canvas, (tx, ty), 4, shadow, -1, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 3, color, -1, cv2.LINE_AA)

        cv2.circle(canvas, (x, y), 7, shadow, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 10, color, 1, cv2.LINE_AA)

    def _draw_meter(self, panel: np.ndarray, x1: int, y: int, x2: int, value: float, label: str) -> None:
        value = max(0.0, min(value, 1.0))
        self._draw_text(panel, label, (x1, y - 8), 12, (148, 157, 170))
        cv2.line(panel, (x1 + 44, y - 11), (x2, y - 11), (64, 70, 84), 5, cv2.LINE_AA)
        fill_x = int((x1 + 44) + value * (x2 - (x1 + 44)))
        cv2.line(panel, (x1 + 44, y - 11), (fill_x, y - 11), self._ui_palette()["accent"], 5, cv2.LINE_AA)

    def reset_settings_panel(self) -> None:
        """Reset tuning sliders and saved tuning values."""
        self.gaze_corrector.reset_tuning()
        self.gaze_correction_enabled = True

    def toggle_correction(self) -> None:
        self.gaze_correction_enabled = not self.gaze_correction_enabled
        self.gaze_corrector.set_tuning(enabled=self.gaze_correction_enabled)

    def request_quit(self) -> None:
        self.should_quit = True

    def open_logs(self) -> None:
        log_path = os.path.expanduser(self.log_file_path)
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            if not os.path.exists(log_path):
                with open(log_path, "a", encoding="utf-8"):
                    pass
            if self._send_native_request("logs"):
                self.logger.log("Requested native log viewer")
                return
            subprocess.run(
                ["open", "-b", "local.spec3-correction", "spec3correction://logs"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            self.logger.log("Requested native log viewer")
        except OSError as exc:
            self.logger.log(f"Could not open logs: {exc}")
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            subprocess.Popen(["open", log_path])
            self.logger.log(f"Opened raw log fallback: {log_path}")

    def open_training_window(self) -> None:
        if self._send_native_request("training"):
            self.logger.log("Requested native training window")
            return
        try:
            subprocess.run(
                ["open", "-b", "local.spec3-correction", "spec3correction://training"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            self.logger.log("Requested native training window")
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
            self.logger.log(f"Could not open training window: {exc}")

    def open_settings_window(self) -> None:
        if self._send_native_request("settings"):
            self.logger.log("Requested native settings window")
            return
        try:
            subprocess.run(
                ["open", "-b", "local.spec3-correction", "spec3correction://settings"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            self.logger.log("Requested native settings window")
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
            self.logger.log(f"Could not open settings window: {exc}")

    def _send_native_request(self, command: str) -> bool:
        if not self.native_request_file_path:
            return False
        try:
            os.makedirs(os.path.dirname(self.native_request_file_path), exist_ok=True)
            tmp_path = f"{self.native_request_file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(f"{command}\n{time.time():.6f}\n")
            os.replace(tmp_path, self.native_request_file_path)
            return True
        except OSError as exc:
            self.logger.log(f"Could not send native request {command}: {exc}")
            return False

    def request_app_quit(self) -> None:
        self.should_quit = True
        try:
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    'tell application id "local.spec3-correction" to quit',
                ]
            )
        except OSError as exc:
            self.logger.log(f"Could not ask native app to quit: {exc}")

    def set_ai_training_label(self, label: str | None) -> None:
        message = self.gaze_corrector.set_personal_training_label(label)
        self.logger.log(message)

    def train_personal_ai(self) -> None:
        message = self.gaze_corrector.train_personal_reading_model()
        self.logger.log(message)

    def reset_personal_ai_samples(self) -> None:
        message = self.gaze_corrector.reset_personal_training_samples()
        self.logger.log(message)

    def _draw_button(
        self,
        panel: np.ndarray,
        key: str,
        rect: tuple[int, int, int, int],
        label: str,
        active: bool,
        danger: bool = False,
    ) -> None:
        x1, y1, x2, y2 = rect
        light = self._ui_theme == "light"
        color = (226, 230, 238) if light else (58, 63, 74)
        border = (191, 198, 211) if light else (88, 96, 110)
        text_color = (30, 34, 42) if light else (243, 246, 250)
        if active:
            color = self._ui_palette()["accent"]
            border = tuple(min(c + 36, 255) for c in color)
            text_color = (255, 255, 255)
        if danger:
            color = (248, 221, 224) if light else (82, 66, 72)
            border = (214, 106, 116) if light else (152, 104, 113)
            text_color = (108, 32, 42) if light else (250, 239, 242)

        self._draw_round_rect(panel, (x1, y1, x2, y2), color, 5, border)
        cv2.line(panel, (x1 + 7, y1 + 1), (x2 - 7, y1 + 1), tuple(min(c + 18, 255) for c in color), 1, cv2.LINE_AA)
        font_size = 16
        while font_size > 12:
            text_size = self._measure_text(label, font_size)
            if text_size[0] <= x2 - x1 - 14:
                break
            font_size -= 1
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        self._draw_text(panel, label, (tx, ty), font_size, text_color, bold=active)
        self._button_regions[key] = rect

    def _draw_slider(
        self,
        panel: np.ndarray,
        key: str,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        suffix: str,
        y: int,
        x1: int = 30,
        x2: int = 380,
    ) -> None:
        light = self._ui_theme == "light"
        track_y = y + 24
        value = max(min_value, min(value, max_value))
        ratio = 0.0 if max_value == min_value else (value - min_value) / (max_value - min_value)
        knob_x = int(x1 + ratio * (x2 - x1))

        label_color = (35, 39, 46) if light else (224, 229, 238)
        track_color = (205, 211, 220) if light else (58, 62, 72)
        zero_color = (155, 162, 174) if light else (88, 94, 108)
        self._draw_text(panel, label, (x1, y), 14, label_color)
        value_text = f"{value:+.0f}{suffix}" if min_value < 0 else f"{value:.0f}{suffix}"
        fill_color = self._ui_palette()["accent"]
        self._draw_text(panel, value_text, (x2 + 16, y), 14, fill_color, bold=True)
        cv2.line(panel, (x1, track_y), (x2, track_y), track_color, 5, cv2.LINE_AA)
        if min_value < 0 < max_value:
            zero_ratio = (0.0 - min_value) / (max_value - min_value)
            zero_x = int(x1 + zero_ratio * (x2 - x1))
            cv2.line(panel, (zero_x, track_y - 9), (zero_x, track_y + 9), zero_color, 1, cv2.LINE_AA)
            cv2.line(panel, (zero_x, track_y), (knob_x, track_y), fill_color, 5, cv2.LINE_AA)
        else:
            cv2.line(panel, (x1, track_y), (knob_x, track_y), fill_color, 5, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 10, (12, 13, 16), -1, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 7, (238, 243, 240), -1, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 9, fill_color, 2, cv2.LINE_AA)
        self._slider_regions[key] = (x1, track_y, x2, min_value, max_value)

    def handle_settings_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        """Handle clicks and drags in the custom settings panel."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, rect in self._button_regions.items():
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._handle_button(key)
                    return

            if self.camera_dropdown_open:
                self.camera_dropdown_open = False
                return

            for key, region in self._slider_regions.items():
                x1, track_y, x2, _min_value, _max_value = region
                if x1 - 16 <= x <= x2 + 16 and track_y - 18 <= y <= track_y + 18:
                    self._dragging_control = key
                    self._set_slider_from_x(key, x, save=False)
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging_preview_aim:
                self._set_gaze_from_preview_point(x, y, save=False)
            elif self._dragging_control:
                self._set_slider_from_x(self._dragging_control, x, save=False)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging_preview_aim:
                self._set_gaze_from_preview_point(x, y, save=True)
            elif self._dragging_control:
                self._set_slider_from_x(self._dragging_control, x, save=True)
            self._dragging_preview_aim = False
            self._dragging_control = None

    def _handle_button(self, key: str) -> None:
        if key == "toggle":
            self.gaze_correction_enabled = not self.gaze_correction_enabled
            self.gaze_corrector.set_tuning(enabled=self.gaze_correction_enabled)
        elif key == "camera_toggle":
            self.toggle_camera_dropdown()
            return
        elif key.startswith("camera_option:"):
            try:
                camera_id = int(key.split(":", 1)[1])
            except ValueError:
                return
            camera = next((item for item in self.camera_options if item.id == camera_id), None)
            if camera is not None:
                self.request_camera_select(camera.id, camera.name)
            return
        elif key == "hide":
            self.request_hide_window()
            return
        elif key == "app_quit":
            self.request_app_quit()
            return
        elif key == "logs":
            self.open_logs()
            return
        elif key == "settings":
            self.open_settings_window()
            return
        elif key == "ai_training":
            self.open_training_window()
            return
        elif key == "reset":
            self.reset_settings_panel()
            return

    def _set_slider_from_x(self, key: str, x: int, save: bool) -> None:
        x1, _track_y, x2, min_value, max_value = self._slider_regions[key]
        ratio = max(0.0, min((x - x1) / (x2 - x1), 1.0))
        value = min_value + ratio * (max_value - min_value)

        kwargs = {"save": save, "reset_tracking": save}
        if key == "strength":
            kwargs["strength"] = value / 100.0
        elif key == "vertical":
            kwargs["vertical_offset"] = value
        elif key == "horizontal":
            kwargs["horizontal_offset"] = value
        elif key == "smooth":
            kwargs["smoothing"] = value / 100.0
        elif key == "stabilizer":
            kwargs["reading_stabilizer"] = value / 100.0
        elif key == "live":
            kwargs["natural_motion"] = value / 100.0

        if key in ("vertical", "horizontal"):
            self.gaze_corrector.hold_manual_aim(0.24)

        self.gaze_corrector.set_tuning(**kwargs)

    def draw_status(self, frame) -> None:
        """Draw status overlay on frame."""
        tuning = self.gaze_corrector.get_tuning()
        reading_active, reading_score, effective_hold = self.gaze_corrector.get_reading_state()
        status = "GAZE ON" if self.gaze_correction_enabled and tuning.enabled else "GAZE OFF"
        color = (0, 255, 0) if self.gaze_correction_enabled and tuning.enabled else (0, 0, 255)

        mode = "READ" if reading_active else "LIVE"
        mode_color = (88, 220, 130) if reading_active else (120, 190, 255)

        cv2.rectangle(frame, (10, 10), (315, 138), (0, 0, 0), -1)
        cv2.putText(
            frame, status, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"[{self.face_predictor.get_name()}]", (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Power {tuning.strength * 100:.0f}%  Smooth {tuning.smoothing * 100:.0f}%",
            (20, 73),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Read hold {tuning.reading_stabilizer * 100:.0f}%  Eye life {tuning.natural_motion * 100:.0f}%",
            (20, 93),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Mode {mode}  read {reading_score * 100:.0f}%  hold {effective_hold * 100:.0f}%",
            (20, 111),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, mode_color, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Reading Stabilizer | R reset",
            (20, 129),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA
        )

    def draw_calibration_overlay(self, frame) -> None:
        """Draw calibration mode overlay with camera offset visualization."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 260, 10), (w - 10, 245), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame, "CALIBRATION MODE", (w - 250, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )

        # Current camera offset
        offset = self.gaze_corrector.get_camera_offset()
        cv2.putText(
            frame, f"Camera Offset:", (w - 250, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  X: {offset[0]:+.1f} cm", (w - 250, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Y: {offset[1]:+.1f} cm", (w - 250, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Z: {offset[2]:+.1f} cm", (w - 250, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 100), 1, cv2.LINE_AA
        )

        # Eye position (estimated)
        eye_pos = self.gaze_corrector.get_last_eye_position()
        cv2.putText(
            frame, f"Eye Position:", (w - 250, 145),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  X: {eye_pos[0]:+.1f} cm", (w - 250, 165),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Y: {eye_pos[1]:+.1f} cm", (w - 250, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Z: {eye_pos[2]:+.1f} cm", (w - 250, 195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 150), 1, cv2.LINE_AA
        )

        # Focal length
        focal = self.gaze_corrector.get_focal_length()
        cv2.putText(
            frame, f"Focal Length: {focal:.0f} px", (w - 250, 215),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 255), 1, cv2.LINE_AA
        )

        # Controls hint
        cv2.putText(
            frame, "[Arrows:XY] [+/-:Z] [[/]:F] [R:Reset]", (w - 250, 235),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA
        )

        # Draw camera position diagram (bottom left)
        self._draw_camera_diagram(frame, offset, eye_pos)

    def _draw_camera_diagram(
        self, frame, camera_offset: tuple, eye_pos: list
    ) -> None:
        """Draw a simple diagram showing camera and eye positions."""
        h, w = frame.shape[:2]

        # Diagram area
        diagram_x, diagram_y = 10, h - 160
        diagram_w, diagram_h = 150, 150

        # Background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (diagram_x, diagram_y),
            (diagram_x + diagram_w, diagram_y + diagram_h),
            (30, 30, 30), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame, "Top View (X-Z)", (diagram_x + 10, diagram_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Center point (screen center, Z=0)
        center_x = diagram_x + diagram_w // 2
        center_y = diagram_y + diagram_h - 30

        # Scale: 1cm = 2 pixels
        scale = 2

        # Draw screen line
        cv2.line(
            frame,
            (diagram_x + 10, center_y),
            (diagram_x + diagram_w - 10, center_y),
            (100, 100, 100), 2
        )
        cv2.putText(
            frame, "Screen", (diagram_x + 50, center_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA
        )

        # Draw camera position (offset from screen center)
        cam_px = center_x + int(camera_offset[0] * scale)
        cam_py = center_y + int(camera_offset[2] * scale)  # Z goes up in diagram
        cv2.circle(frame, (cam_px, cam_py), 6, (0, 255, 255), -1)
        cv2.putText(
            frame, "Cam", (cam_px - 12, cam_py - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA
        )

        # Draw eye position (estimated)
        eye_px = center_x + int(eye_pos[0] * scale)
        eye_py = center_y + int(eye_pos[2] * scale)
        cv2.circle(frame, (eye_px, eye_py), 5, (255, 100, 100), -1)
        cv2.putText(
            frame, "Eye", (eye_px - 10, eye_py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1, cv2.LINE_AA
        )

        # Draw gaze line from eye to camera
        cv2.line(frame, (eye_px, eye_py), (cam_px, cam_py), (100, 255, 100), 1)

    def handle_calibration_key(self, key: int) -> bool:
        """
        Handle calibration mode key presses.

        Args:
            key: Key code

        Returns:
            True if key was handled
        """
        step_xy = self.calib_cfg.step_xy
        step_z = self.calib_cfg.step_z

        if key == self.KEY_LEFT:
            self.gaze_corrector.adjust_camera_offset(dx=-step_xy)
            return True
        elif key == self.KEY_RIGHT:
            self.gaze_corrector.adjust_camera_offset(dx=step_xy)
            return True
        elif key == self.KEY_UP:
            self.gaze_corrector.adjust_camera_offset(dy=-step_xy)
            return True
        elif key == self.KEY_DOWN:
            self.gaze_corrector.adjust_camera_offset(dy=step_xy)
            return True
        elif key == ord("+") or key == ord("="):
            self.gaze_corrector.adjust_camera_offset(dz=step_z)
            return True
        elif key == ord("-") or key == ord("_"):
            self.gaze_corrector.adjust_camera_offset(dz=-step_z)
            return True
        elif key == ord("["):
            self.gaze_corrector.adjust_focal_length(-self.calib_cfg.step_focal)
            return True
        elif key == ord("]"):
            self.gaze_corrector.adjust_focal_length(self.calib_cfg.step_focal)
            return True
        elif key == ord("r"):
            x, y, z = self.default_camera_offset
            self.gaze_corrector.set_camera_offset(x, y, z)
            self.gaze_corrector.set_focal_length(self.default_focal_length)
            self.logger.log("Camera offset and focal length reset to default")
            return True

        return False

    def process_frame(self, frame):
        """
        Process a single frame with gaze correction.

        Args:
            frame: BGR video frame

        Returns:
            Processed frame with gaze correction applied
        """
        display_frame = frame.copy()

        # Get eye data for all detected faces
        face_data_list = self.face_predictor.list_eye_data(frame, self.eye_config)
        if not face_data_list:
            self.gaze_corrector.reset_tracking()
            return display_frame

        # Process first detected face
        for face_data in face_data_list:
            try:
                # Apply gaze correction (pass video_size)
                display_frame = self.gaze_corrector.apply_correction(
                    display_frame, face_data, self.display_cfg.video_size
                )
            except Exception as e:
                self.logger.log(f"Error: {e}")
            break  # Only process first face

        return display_frame

    def _prepare_pipeline_frame(self, frame: np.ndarray) -> np.ndarray:
        max_w, max_h = self.display_cfg.processing_max_size
        frame_h, frame_w = frame.shape[:2]
        scale = min(max_w / max(frame_w, 1), max_h / max(frame_h, 1), 1.0)
        if scale >= 0.999:
            return frame

        resized_w = max(1, int(frame_w * scale))
        resized_h = max(1, int(frame_h * scale))
        return cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    def _frame_live_metrics(self, frame: np.ndarray) -> tuple[bool, float, float, float, float]:
        """Return a conservative live-camera quality estimate for fallback selection."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        sample = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        mean = float(sample.mean())
        std = float(sample.std())
        bright_ratio = float(np.mean(sample > 28))
        score = mean * 0.55 + std * 1.35 + bright_ratio * 42.0
        usable = (
            (mean >= 12.0 and std >= 7.0)
            or (std >= 26.0 and bright_ratio >= 0.06)
            or bright_ratio >= 0.18
        )
        return usable, score, mean, std, bright_ratio

    def _camera_candidate_ids(self, preferred_id: int, requested_label: str) -> list[int]:
        candidates: list[int] = []
        with self._camera_refresh_lock:
            camera_options = list(self._all_camera_options or self.camera_options)
        virtual_ids = {
            camera.id for camera in camera_options if is_virtual_camera_name(camera.name)
        }
        logged_virtual_skips: set[int] = set()
        prefer_physical = bool(requested_label) and not is_virtual_camera_name(requested_label)

        def add(candidate_id: int) -> None:
            if candidate_id < 0 or candidate_id in candidates:
                return
            if prefer_physical and candidate_id in virtual_ids:
                if candidate_id not in logged_virtual_skips:
                    self.logger.log(
                        f"Skipping virtual camera slot {candidate_id} for {requested_label}"
                    )
                    logged_virtual_skips.add(candidate_id)
                return
            candidates.append(candidate_id)

        add(preferred_id)

        physical_first = sorted(
            camera_options,
            key=lambda camera: (is_virtual_camera_name(camera.name), camera.id),
        )
        for camera in physical_first:
            add(camera.id)

        # OpenCV can expose AVFoundation devices in a different order from the
        # native device list. Probe only known slots when macOS gave us a list,
        # otherwise keep a small blind fallback for unusual systems.
        probe_limit = max((camera.id for camera in camera_options), default=5) + 1
        if not camera_options:
            probe_limit = 6
        for camera_id in range(probe_limit):
            add(camera_id)

        return candidates

    def _open_camera_capture_direct(self, camera_id: int, require_live_frame: bool = True):
        """Open one concrete OpenCV camera slot and verify it returns a live frame."""
        camera_id = self.camera_id if camera_id is None else camera_id
        backends: list[tuple[str, int | None]] = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
        backends.append(("default", None))

        for backend_name, backend in backends:
            cap = (
                cv2.VideoCapture(camera_id, backend)
                if backend is not None
                else cv2.VideoCapture(camera_id)
            )
            if not cap.isOpened():
                cap.release()
                self.logger.log(f"Camera {camera_id} did not open with {backend_name}")
                continue

            requested_w = max(1280, min(self.display_cfg.video_size[0], self.display_cfg.processing_max_size[0]))
            requested_h = max(720, min(self.display_cfg.video_size[1], self.display_cfg.processing_max_size[1]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            has_frame = False
            best_metrics: tuple[bool, float, float, float, float] | None = None
            for _ in range(10):
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    has_frame = True
                    metrics = self._frame_live_metrics(test_frame)
                    if best_metrics is None or metrics[1] > best_metrics[1]:
                        best_metrics = metrics
                time.sleep(0.025)
            if not has_frame:
                cap.release()
                self.logger.log(f"Camera {camera_id} opened with {backend_name} but returned no frames")
                continue

            if best_metrics is not None:
                usable, score, mean, std, bright_ratio = best_metrics
                if require_live_frame and not usable:
                    cap.release()
                    self.logger.log(
                        "Camera "
                        f"{camera_id} returned a blank/inactive image "
                        f"(score={score:.1f}, mean={mean:.1f}, std={std:.1f}, bright={bright_ratio:.2f})"
                    )
                    continue

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.log(f"Camera {camera_id} opened with {backend_name} at {actual_w}x{actual_h}")
            return cap

        return None

    def _open_camera_capture(
        self,
        camera_id: Optional[int] = None,
        requested_label: str = "",
        allow_remap: bool = True,
    ):
        """Open the requested camera, remapping to a live slot if macOS names are out of order."""
        preferred_id = self.camera_id if camera_id is None else camera_id
        label = requested_label or self._camera_label_overrides.get(preferred_id, "")
        allow_fallback = allow_remap and (not label or not is_virtual_camera_name(label))
        candidates = self._camera_candidate_ids(preferred_id, label) if allow_fallback else [preferred_id]

        for candidate_id in candidates:
            candidate_label = self._camera_label_overrides.get(
                candidate_id,
                camera_name(candidate_id, self._all_camera_options or self.camera_options),
            )
            if label and not is_virtual_camera_name(label) and is_virtual_camera_name(candidate_label):
                self.logger.log(
                    f"Skipping virtual camera slot {candidate_id} ({candidate_label}) "
                    f"for requested physical camera {label}"
                )
                continue
            cap = self._open_camera_capture_direct(candidate_id, require_live_frame=True)
            if cap is not None:
                if candidate_id != preferred_id:
                    pretty_label = self._short_label(label or f"Camera {preferred_id}", 28)
                    self._set_camera_status(f"{pretty_label} opened on slot #{candidate_id}", 3.0)
                    self.logger.log(
                        f"Camera slot remapped: requested {preferred_id}, opened {candidate_id}"
                    )
                return cap, candidate_id

        preferred_label = self._camera_label_overrides.get(
            preferred_id,
            camera_name(preferred_id, self._all_camera_options or self.camera_options),
        )
        if label and not is_virtual_camera_name(label) and is_virtual_camera_name(preferred_label):
            self.logger.log(
                f"Not falling back to virtual camera slot {preferred_id} "
                f"({preferred_label}) for requested physical camera {label}"
            )
            return None, preferred_id

        cap = self._open_camera_capture_direct(preferred_id, require_live_frame=False)
        if cap is not None:
            self._set_camera_status("Camera opened, but image looks dark", 3.0)
            return cap, preferred_id

        return None, preferred_id

    def _update_video_size_from_frame(self, frame: np.ndarray) -> None:
        frame_h, frame_w = frame.shape[:2]
        video_size = (frame_w, frame_h)
        if self.display_cfg.video_size == video_size:
            return

        self.display_cfg.video_size = video_size
        self.display_cfg.face_detect_size = (max(1, frame_w // 2), max(1, frame_h // 2))
        self._last_canvas_size = None
        self.logger.log(f"Video size updated to {frame_w}x{frame_h}")

    def _apply_pending_camera_switch(self, stream: CameraFrameStream):
        if self._pending_camera_id is None:
            return stream

        next_camera_id = self._pending_camera_id
        requested_label = self._pending_camera_label
        self._pending_camera_id = None
        self._pending_camera_label = ""
        if next_camera_id == self.camera_id:
            self._camera_switching = False
            return stream

        self.logger.log(f"Switching camera {self.camera_id} -> {next_camera_id}")
        next_cap, actual_camera_id = self._open_camera_capture(
            next_camera_id,
            requested_label,
            allow_remap=True,
        )
        if next_cap is None:
            self.logger.log(f"Could not switch to camera {next_camera_id}")
            self.camera_label = self._camera_label_overrides.get(
                self.camera_id,
                camera_name(self.camera_id, self._all_camera_options or self.camera_options),
            )
            self._camera_switching = False
            self._set_camera_status("Camera did not respond", 3.0)
            self.refresh_camera_options(force=True, async_refresh=True)
            return stream

        stream.release()
        self.camera_id = actual_camera_id
        self.camera_label = requested_label or camera_name(
            self.camera_id,
            self._all_camera_options or self.camera_options,
        )
        self._remember_camera_label(self.camera_id, self.camera_label)
        if not is_virtual_camera_name(self.camera_label):
            save_camera_id(self.camera_id, self.camera_label)
        self._camera_switching = False
        self._set_camera_status(f"Selected slot #{self.camera_id}", 2.0)
        self.refresh_camera_options(force=False, async_refresh=True)
        self.gaze_corrector.reset_tracking()
        return CameraFrameStream(next_cap, self.logger)

    def _recover_camera_stream(self, stream: CameraFrameStream, reason: str):
        """Try the physical stream again instead of falling into virtual camera slots."""
        self.logger.log(f"{reason}; looking for another live camera slot")
        stream.release()
        with self._camera_refresh_lock:
            camera_options = list(self._all_camera_options or self.camera_options)

        prefer_physical = not is_virtual_camera_name(self.camera_label)
        candidates = self._camera_candidate_ids(self.camera_id, self.camera_label)
        for camera in sorted(camera_options, key=lambda item: (is_virtual_camera_name(item.name), item.id)):
            if prefer_physical and is_virtual_camera_name(camera.name):
                self.logger.log(
                    f"Skipping virtual recovery slot {camera.id} ({camera.name})"
                )
                continue
            if camera.id not in candidates:
                candidates.append(camera.id)

        for candidate_id in candidates:
            recovered_label = self._camera_label_overrides.get(
                candidate_id,
                camera_name(candidate_id, self._all_camera_options or self.camera_options),
            )
            if prefer_physical and is_virtual_camera_name(recovered_label):
                self.logger.log(
                    f"Skipping virtual recovery slot {candidate_id} ({recovered_label})"
                )
                continue
            cap = self._open_camera_capture_direct(candidate_id, require_live_frame=True)
            if cap is None:
                continue

            previous_id = self.camera_id
            self.camera_id = candidate_id
            self.camera_label = recovered_label
            self._remember_camera_label(candidate_id, self.camera_label)
            if not is_virtual_camera_name(self.camera_label):
                save_camera_id(candidate_id, self.camera_label)
            self._set_camera_status(f"Recovered camera on slot #{candidate_id}", 3.0)
            self.logger.log(f"Recovered camera stream: {previous_id} -> {candidate_id}")
            self.gaze_corrector.reset_tracking()
            return CameraFrameStream(cap, self.logger)

        self.logger.log("Could not recover a live camera stream")
        return None

    def run(self):
        """Main application loop."""
        self.logger.log(f"Starting camera {self.camera_id}...")
        cap, actual_camera_id = self._open_camera_capture(self.camera_id, self.camera_label)
        if cap is None:
            self.logger.log(f"Could not open camera {self.camera_id}")
            return
        if actual_camera_id != self.camera_id:
            self.camera_id = actual_camera_id
        self._remember_camera_label(self.camera_id, self.camera_label)
        if not is_virtual_camera_name(self.camera_label):
            save_camera_id(self.camera_id, self.camera_label)
        stream = CameraFrameStream(cap, self.logger)

        self.create_settings_panel()
        self.logger.log("Press 'g' to toggle gaze, 'c' for calibration, 'q' to hide")

        failed_reads = 0
        last_frame_id = 0
        while True:
            self.process_control_command()
            self.apply_pending_window_actions()
            stream = self._apply_pending_camera_switch(stream)
            if self.should_quit:
                break

            frame_id, frame, frame_age, stream_failed_reads = stream.snapshot()
            if frame is None:
                failed_reads += 1
                if failed_reads <= 240:
                    if failed_reads == 1:
                        self.logger.log("Camera returned an empty frame; waiting briefly")
                    time.sleep(0.01)
                    continue
                self.logger.log("Failed to read frame after retries")
                recovered = self._recover_camera_stream(stream, "Failed to read frame after retries")
                if recovered is not None:
                    stream = recovered
                    failed_reads = 0
                    last_frame_id = 0
                    continue
                break
            if frame_id == last_frame_id:
                if stream_failed_reads > 180 and frame_age > 4.0:
                    self.logger.log("Camera stream stopped returning fresh frames")
                    recovered = self._recover_camera_stream(stream, "Camera stream stopped returning fresh frames")
                    if recovered is not None:
                        stream = recovered
                        failed_reads = 0
                        last_frame_id = 0
                        continue
                    break
                time.sleep(0.003)
                continue
            last_frame_id = frame_id
            failed_reads = 0
            frame = self._prepare_pipeline_frame(frame)
            self._update_video_size_from_frame(frame)

            if self.gaze_correction_enabled:
                display_frame = self.process_frame(frame)
            else:
                display_frame = frame.copy()

            # Draw calibration overlay if enabled
            if self.calibration_mode:
                self.draw_calibration_overlay(display_frame)

            key = -1
            if self.window_visible:
                if not self._window_created:
                    self.create_settings_panel()
                app_frame = self.compose_app_frame(display_frame)
                cv2.imshow(self.display_cfg.window_name, app_frame)
                self._apply_pending_window_resize()
                try:
                    if cv2.getWindowProperty(self.display_cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self.logger.log("Main window hidden")
                        self.hide_window(already_closed=True)
                        continue
                except cv2.error:
                    self.hide_window(already_closed=True)
                    continue
                key = cv2.waitKeyEx(1)

            key_ascii = key & 0xFF
            if key_ascii == 27:
                break
            elif key_ascii in (ord("q"), ord("Q")):
                self.request_hide_window()
            elif key_ascii in (ord("g"), ord("G")):
                self.toggle_correction()
                self.logger.log(
                    f"Gaze correction: {'enabled' if self.gaze_correction_enabled else 'disabled'}"
                )
            elif key_ascii in (ord("c"), ord("C")):
                self.calibration_mode = not self.calibration_mode
                self.logger.log(
                    f"Calibration mode: {'enabled' if self.calibration_mode else 'disabled'}"
                )
            elif key_ascii in (ord("r"), ord("R")):
                self.reset_settings_panel()
                self.logger.log("Gaze tuning reset to default")
            elif key_ascii == ord("1"):
                self.set_ai_training_label("read")
            elif key_ascii == ord("2"):
                self.set_ai_training_label("live")
            elif key_ascii == ord("3"):
                self.set_ai_training_label("glance")
            elif key_ascii == ord("0"):
                self.set_ai_training_label(None)
            elif key_ascii in (ord("t"), ord("T")):
                self.train_personal_ai()
            elif self.calibration_mode:
                self.handle_calibration_key(key)

        # Cleanup
        self.gaze_corrector.save_tuning_settings()
        stream.release()
        cv2.destroyAllWindows()
        self.gaze_corrector.close()
        self.logger.log("Shutdown complete")
