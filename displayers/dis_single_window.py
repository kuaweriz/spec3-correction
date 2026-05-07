"""
Single Window Gaze Corrector

A simplified gaze correction application that:
- Uses a single window (no socket communication)
- Allows real-time toggle of gaze correction with 'g' key
- Supports injectable face predictor backends
- Decoupled architecture: FacePredictor -> GazeCorrector
- Calibration mode for camera offset adjustment
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from utils.logger import Logger
from displayers.face_predictor import (
    FacePredictor,
    EyeExtractionConfig,
    create_face_predictor,
)
from model_managers.gaze_corrector_v1 import GazeCorrector


################################################################################
# Configuration
################################################################################


@dataclass
class DisplayConfig:
    """Configuration for the display application."""

    video_size: tuple[int, int] = (640, 480)
    face_detect_size: tuple[int, int] = (320, 240)
    window_name: str = "Gaze Correction"

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
        self.window_visible = True
        self._window_created = False
        self.control_file_path = os.environ.get("GAZE_CONTROL_FILE")
        self._last_control_mtime = 0.0
        self._pending_hide_window = False
        self._pending_show_window = False

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
        self._slider_regions.clear()
        self._button_regions.clear()
        if not already_closed:
            try:
                cv2.destroyWindow(self.display_cfg.window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

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
        elif command == "quit":
            self.request_quit()

    def compose_app_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compose the camera preview and the control panel into one window."""
        frame_h, frame_w = frame.shape[:2]
        max_w, max_h = self.preview_max_size
        scale = min(max_w / max(frame_w, 1), max_h / max(frame_h, 1))
        scale = max(0.25, min(scale, 2.0))
        preview_w = max(1, int(frame_w * scale))
        preview_h = max(1, int(frame_h * scale))

        if (preview_w, preview_h) != (frame_w, frame_h):
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
        else:
            preview = frame

        canvas_h = max(preview_h, 720)
        canvas_w = preview_w + self.control_panel_width
        canvas = np.full((canvas_h, canvas_w, 3), (16, 18, 23), dtype=np.uint8)

        preview_y = (canvas_h - preview_h) // 2
        canvas[preview_y:preview_y + preview_h, 0:preview_w] = preview
        cv2.rectangle(canvas, (0, preview_y), (preview_w - 1, preview_y + preview_h - 1), (44, 49, 58), 1)

        self._panel_origin_x = preview_w
        cv2.line(canvas, (preview_w, 0), (preview_w, canvas_h), (54, 60, 72), 1, cv2.LINE_AA)
        self.draw_control_panel(canvas, preview_w, canvas_h)

        canvas_size = (canvas_w, canvas_h)
        if self._last_canvas_size != canvas_size:
            cv2.resizeWindow(self.display_cfg.window_name, canvas_w, canvas_h)
            self._last_canvas_size = canvas_size

        return canvas

    def draw_control_panel(self, canvas: np.ndarray, x0: int, height: int) -> None:
        """Draw the integrated control panel on the right side of the app."""
        tuning = self.gaze_corrector.get_tuning()
        reading_active, reading_score, effective_hold = self.gaze_corrector.get_reading_state()
        panel_w = self.control_panel_width
        x1 = x0 + panel_w
        self._slider_regions.clear()
        self._button_regions.clear()

        panel_bg = (18, 19, 23)
        header_bg = (25, 25, 31)
        card_bg = (31, 31, 38)
        card_line = (66, 66, 78)
        muted = (158, 164, 176)
        text = (242, 245, 250)
        accent = (102, 203, 135)
        live_accent = (224, 157, 88)

        cv2.rectangle(canvas, (x0 + 1, 0), (x1, height), panel_bg, -1)
        cv2.rectangle(canvas, (x0 + 1, 0), (x1, 96), header_bg, -1)
        cv2.line(canvas, (x0 + 1, 95), (x1, 95), (47, 51, 60), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0 + 28, 82), (x0 + 392, 82), (43, 47, 56), 1, cv2.LINE_AA)

        title_x = x0 + 28
        cv2.circle(canvas, (title_x + 6, 36), 5, accent, -1, cv2.LINE_AA)
        cv2.circle(canvas, (title_x + 6, 36), 9, (42, 72, 54), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Gaze Studio", (title_x + 22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, text, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Live gaze control", (title_x + 24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.43, muted, 1, cv2.LINE_AA)

        on_label = "ON" if tuning.enabled and self.gaze_correction_enabled else "OFF"
        self._draw_button(canvas, "toggle", (x0 + 292, 24, x0 + 392, 62), on_label, tuning.enabled and self.gaze_correction_enabled)

        mode_label = "READ" if reading_active else "LIVE"
        mode_color = accent if reading_active else live_accent
        cv2.rectangle(canvas, (x0 + 28, 112), (x0 + 392, 178), card_bg, -1)
        cv2.rectangle(canvas, (x0 + 28, 112), (x0 + 392, 178), card_line, 1)
        cv2.rectangle(canvas, (x0 + 28, 112), (x0 + 33, 178), mode_color, -1)
        cv2.putText(canvas, "Adaptive mode", (x0 + 46, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (218, 223, 232), 1, cv2.LINE_AA)
        cv2.putText(canvas, mode_label, (x0 + 298, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 2, cv2.LINE_AA)
        self._draw_meter(canvas, x0 + 46, 158, x0 + 216, reading_score, "read")
        self._draw_meter(canvas, x0 + 230, 158, x0 + 374, effective_hold, "hold")

        sliders = [
            ("strength", "Strength", tuning.strength * 100.0, 0.0, 150.0, "%"),
            ("vertical", "Eyes Up / Down", tuning.vertical_offset, -90.0, 90.0, " deg"),
            ("horizontal", "Eyes Left / Right", tuning.horizontal_offset, -45.0, 45.0, " deg"),
            ("smooth", "Smooth", tuning.smoothing * 100.0, 0.0, 100.0, "%"),
            ("stabilizer", "Pupil Hold", tuning.reading_stabilizer * 100.0, 0.0, 100.0, "%"),
            ("live", "Live Look", tuning.natural_motion * 100.0, 0.0, 100.0, "%"),
        ]

        track_x1 = x0 + 34
        track_x2 = x0 + 286
        y = 220
        for key, label, value, min_value, max_value, suffix in sliders:
            self._draw_slider(canvas, key, label, value, min_value, max_value, suffix, y, track_x1, track_x2)
            y += 58

        button_y = min(height - 94, 608)
        self._draw_button(canvas, "reading", (x0 + 28, button_y, x0 + 186, button_y + 42), "Reading Preset", reading_active)
        self._draw_button(canvas, "reset", (x0 + 202, button_y, x0 + 292, button_y + 42), "Reset", False)
        self._draw_button(canvas, "quit", (x0 + 308, button_y, x0 + 392, button_y + 42), "Hide", False, danger=True)

        footer_y = min(height - 28, button_y + 78)
        cv2.putText(canvas, f"Menu bar controls window and correction", (x0 + 28, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (138, 147, 160), 1, cv2.LINE_AA)

    def _draw_meter(self, panel: np.ndarray, x1: int, y: int, x2: int, value: float, label: str) -> None:
        value = max(0.0, min(value, 1.0))
        cv2.putText(panel, label, (x1, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (148, 157, 170), 1, cv2.LINE_AA)
        cv2.line(panel, (x1 + 44, y - 11), (x2, y - 11), (64, 70, 84), 5, cv2.LINE_AA)
        fill_x = int((x1 + 44) + value * (x2 - (x1 + 44)))
        cv2.line(panel, (x1 + 44, y - 11), (fill_x, y - 11), (84, 178, 130), 5, cv2.LINE_AA)

    def reset_settings_panel(self) -> None:
        """Reset tuning sliders and saved tuning values."""
        self.gaze_corrector.reset_tuning()
        self.gaze_correction_enabled = True

    def toggle_correction(self) -> None:
        self.gaze_correction_enabled = not self.gaze_correction_enabled
        self.gaze_corrector.set_tuning(enabled=self.gaze_correction_enabled)

    def request_quit(self) -> None:
        self.should_quit = True

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
        color = (58, 63, 74)
        border = (88, 96, 110)
        text_color = (243, 246, 250)
        if active:
            color = (67, 163, 102)
            border = (118, 215, 145)
        if danger:
            color = (82, 66, 72)
            border = (152, 104, 113)
            text_color = (250, 239, 242)

        cv2.rectangle(panel, (x1 + 2, y1 + 3), (x2 + 2, y2 + 3), (11, 12, 15), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
        cv2.line(panel, (x1 + 1, y1), (x2 - 1, y1), tuple(min(c + 24, 255) for c in color), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (x1, y1), (x2, y2), border, 1)
        scale = 0.55
        while scale > 0.34:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
            if text_size[0] <= x2 - x1 - 14:
                break
            scale -= 0.04
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)
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
        track_y = y + 28
        value = max(min_value, min(value, max_value))
        ratio = 0.0 if max_value == min_value else (value - min_value) / (max_value - min_value)
        knob_x = int(x1 + ratio * (x2 - x1))

        cv2.putText(panel, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (224, 229, 238), 1, cv2.LINE_AA)
        value_text = f"{value:+.0f}{suffix}" if min_value < 0 else f"{value:.0f}{suffix}"
        cv2.putText(panel, value_text, (x2 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (142, 224, 182), 1, cv2.LINE_AA)
        cv2.line(panel, (x1, track_y), (x2, track_y), (58, 62, 72), 8, cv2.LINE_AA)
        fill_color = (94, 191, 128)
        if min_value < 0 < max_value:
            zero_ratio = (0.0 - min_value) / (max_value - min_value)
            zero_x = int(x1 + zero_ratio * (x2 - x1))
            cv2.line(panel, (zero_x, track_y - 10), (zero_x, track_y + 10), (88, 94, 108), 1, cv2.LINE_AA)
            cv2.line(panel, (zero_x, track_y), (knob_x, track_y), fill_color, 8, cv2.LINE_AA)
        else:
            cv2.line(panel, (x1, track_y), (knob_x, track_y), fill_color, 8, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 13, (12, 13, 16), -1, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 10, (238, 243, 240), -1, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 12, fill_color, 2, cv2.LINE_AA)
        self._slider_regions[key] = (x1, track_y, x2, min_value, max_value)

    def handle_settings_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        """Handle clicks and drags in the custom settings panel."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, rect in self._button_regions.items():
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._handle_button(key)
                    return

            for key, region in self._slider_regions.items():
                x1, track_y, x2, _min_value, _max_value = region
                if x1 - 16 <= x <= x2 + 16 and track_y - 18 <= y <= track_y + 18:
                    self._dragging_control = key
                    self._set_slider_from_x(key, x, save=False)
                    return

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging_control:
            self._set_slider_from_x(self._dragging_control, x, save=False)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging_control:
                self._set_slider_from_x(self._dragging_control, x, save=True)
            self._dragging_control = None

    def _handle_button(self, key: str) -> None:
        if key == "toggle":
            self.gaze_correction_enabled = not self.gaze_correction_enabled
            self.gaze_corrector.set_tuning(enabled=self.gaze_correction_enabled)
        elif key == "quit":
            self.request_hide_window()
            return
        elif key == "reading":
            self.gaze_corrector.set_tuning(
                smoothing=0.88,
                reading_stabilizer=1.0,
                natural_motion=0.18,
            )
        elif key == "reset":
            self.reset_settings_panel()
            return

    def _set_slider_from_x(self, key: str, x: int, save: bool) -> None:
        x1, _track_y, x2, min_value, max_value = self._slider_regions[key]
        ratio = max(0.0, min((x - x1) / (x2 - x1), 1.0))
        value = min_value + ratio * (max_value - min_value)

        kwargs = {"save": save}
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
            frame, f"Strength {tuning.strength * 100:.0f}%  Smooth {tuning.smoothing * 100:.0f}%",
            (20, 73),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"V {tuning.vertical_offset:+.1f} deg  H {tuning.horizontal_offset:+.1f} deg  Stable {tuning.reading_stabilizer * 100:.0f}%",
            (20, 93),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Mode {mode}  read {reading_score * 100:.0f}%  hold {effective_hold * 100:.0f}%",
            (20, 111),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, mode_color, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Use Gaze Settings sliders | R reset",
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

    def run(self):
        """Main application loop."""
        self.logger.log(f"Starting camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_cfg.video_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_cfg.video_size[1])

        self.create_settings_panel()
        self.logger.log("Press 'g' to toggle gaze, 'c' for calibration, 'q' to hide")

        while True:
            self.process_control_command()
            self.apply_pending_window_actions()
            if self.should_quit:
                break

            ret, frame = cap.read()
            if not ret:
                self.logger.log("Failed to read frame")
                break

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
            elif self.calibration_mode:
                self.handle_calibration_key(key)

        # Cleanup
        self.gaze_corrector.save_tuning_settings()
        cap.release()
        cv2.destroyAllWindows()
        self.gaze_corrector.close()
        self.logger.log("Shutdown complete")
