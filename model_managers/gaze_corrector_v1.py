"""
Gaze Corrector V1 Module

This module provides the gaze correction model wrapper and inference logic,
with YAML-based configuration and database-backed user settings.
"""

from __future__ import annotations

import math
import time
import yaml
import numpy as np
import tensorflow as tf
import cv2
from dataclasses import dataclass
from collections import deque

from tf_models.gaze_corrector_v1 import gaze_warp_model
from utils.logger import Logger
from model_managers.user_settings_db import UserSettingsDB


################################################################################
# Configuration Classes
################################################################################


@dataclass
class GazeWarpModelConfig:
    """Hyperparameters for the gaze warp model."""
    height: int = 48
    width: int = 64
    encoded_angle_dim: int = 16


@dataclass
class GazeModelConfig:
    """Configuration for the gaze correction model."""
    
    model_dir: str = "./weights/warping_model/flx/12/"
    eye_input_size: tuple[int, int] = (48, 64)  # (height, width)
    ef_dim: int = 12
    channel: int = 3
    gaze_warp_model: GazeWarpModelConfig = None
    
    def __post_init__(self):
        if self.gaze_warp_model is None:
            self.gaze_warp_model = GazeWarpModelConfig()
        elif isinstance(self.gaze_warp_model, dict):
            self.gaze_warp_model = GazeWarpModelConfig(**self.gaze_warp_model)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GazeModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert eye_input_size from list to tuple
        if 'eye_input_size' in data and isinstance(data['eye_input_size'], list):
            data['eye_input_size'] = tuple(data['eye_input_size'])
        
        return cls(**data)


@dataclass
class CameraUserSetting:
    """User-adjustable camera and screen geometry settings."""
    
    focal_length: float = 650.0
    ipd: float = 6.3  # Inter-pupillary distance in cm
    camera_offset: tuple[float, float, float] = (0, -21, -1)  # relative to screen center
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'focal_length': self.focal_length,
            'ipd': self.ipd,
            'camera_offset': list(self.camera_offset),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CameraUserSetting":
        """Load from dictionary."""
        if 'camera_offset' in data and isinstance(data['camera_offset'], list):
            data['camera_offset'] = tuple(data['camera_offset'])
        return cls(**data)


@dataclass
class GazeTuningSetting:
    """User-facing controls for manual gaze tuning."""

    enabled: bool = True
    strength: float = 1.0
    vertical_offset: float = 0.0
    horizontal_offset: float = 0.0
    smoothing: float = 0.84
    reading_stabilizer: float = 0.90
    natural_motion: float = 0.06

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "strength": self.strength,
            "vertical_offset": self.vertical_offset,
            "horizontal_offset": self.horizontal_offset,
            "smoothing": self.smoothing,
            "reading_stabilizer": self.reading_stabilizer,
            "natural_motion": self.natural_motion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GazeTuningSetting":
        data.pop("pupil_lock", None)
        if "reading_hold" in data and "reading_stabilizer" not in data:
            data["reading_stabilizer"] = data.pop("reading_hold")
        else:
            data.pop("reading_hold", None)
        data.pop("reading_x_gain", None)
        data.pop("reading_y_gain", None)
        data.setdefault("reading_stabilizer", 0.90)
        return cls(**data)


@dataclass
class GazeStabilizationConfig:
    """Temporal smoothing settings for stable, natural-looking gaze correction."""

    enabled: bool = True
    min_cutoff: float = 0.9
    beta: float = 0.08
    derivative_cutoff: float = 1.0
    reading_deadband_degrees: float = 0.45
    reading_lock_strength: float = 0.82
    max_step_degrees: float = 1.35
    natural_motion_strength: float = 0.08


class OneEuroVectorFilter:
    """Adaptive low-pass filter that removes landmark jitter while tracking motion."""

    def __init__(self, config: GazeStabilizationConfig):
        self.cfg = config
        self.prev_value: np.ndarray | None = None
        self.prev_derivative: np.ndarray | None = None
        self.prev_time: float | None = None
        self.started_at = time.monotonic()

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        """Clear filter history, used when tracking is lost."""
        self.prev_value = None
        self.prev_derivative = None
        self.prev_time = None
        self.started_at = time.monotonic()

    def apply(self, value: list[float], natural_scale: float = 1.0) -> list[float]:
        """Return a smoothed gaze angle vector [vertical, horizontal]."""
        now = time.monotonic()
        value_np = np.asarray(value, dtype=np.float32)

        if self.prev_value is None or self.prev_time is None:
            self.prev_value = value_np
            self.prev_derivative = np.zeros_like(value_np)
            self.prev_time = now
            return self._add_natural_motion(value_np, natural_scale).tolist()

        dt = max(1.0 / 120.0, min(now - self.prev_time, 0.1))
        raw_delta = value_np - self.prev_value

        stable_axes = np.abs(raw_delta) < self.cfg.reading_deadband_degrees
        locked_target = value_np.copy()
        locked_target[stable_axes] = (
            self.prev_value[stable_axes] * self.cfg.reading_lock_strength
            + value_np[stable_axes] * (1.0 - self.cfg.reading_lock_strength)
        )

        derivative = (locked_target - self.prev_value) / dt
        derivative_alpha = self._alpha(self.cfg.derivative_cutoff, dt)
        derivative_hat = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.prev_derivative
        )

        cutoff = self.cfg.min_cutoff + self.cfg.beta * np.abs(derivative_hat)
        value_alpha = self._alpha(cutoff, dt)
        filtered = value_alpha * locked_target + (1.0 - value_alpha) * self.prev_value

        max_step = self.cfg.max_step_degrees * max(0.5, min(dt * 60.0, 2.0))
        filtered = self.prev_value + np.clip(
            filtered - self.prev_value, -max_step, max_step
        )

        self.prev_value = filtered
        self.prev_derivative = derivative_hat
        self.prev_time = now

        return self._add_natural_motion(filtered, natural_scale).tolist()

    def _add_natural_motion(self, value: np.ndarray, natural_scale: float = 1.0) -> np.ndarray:
        if self.cfg.natural_motion_strength <= 0 or natural_scale <= 0:
            return value

        t = time.monotonic() - self.started_at
        vertical = math.sin(t * 1.7) * 0.55 + math.sin(t * 0.53 + 1.2) * 0.45
        horizontal = math.sin(t * 1.3 + 2.1) * 0.6 + math.sin(t * 0.47) * 0.4
        natural = np.asarray([vertical, horizontal], dtype=np.float32)
        return value + natural * self.cfg.natural_motion_strength * natural_scale


class PupilHoldFilter:
    """Tracks a stable iris position and returns only the reading/saccade drift."""

    def __init__(self):
        self.stable_offsets: dict[str, np.ndarray] = {}
        self.smoothed_offsets: dict[str, np.ndarray] = {}
        self.release_scores: dict[str, float] = {}

    def reset(self) -> None:
        self.stable_offsets.clear()
        self.smoothed_offsets.clear()
        self.release_scores.clear()

    def reset_eye(self, eye_side: str) -> None:
        self.stable_offsets.pop(eye_side, None)
        self.smoothed_offsets.pop(eye_side, None)
        self.release_scores.pop(eye_side, None)

    def apply(
        self,
        eye_side: str,
        raw_offset: tuple[float, float],
        stabilizer_strength: float,
    ) -> np.ndarray:
        raw = np.asarray(raw_offset, dtype=np.float32)
        if stabilizer_strength <= 0:
            self.stable_offsets[eye_side] = raw
            self.smoothed_offsets[eye_side] = raw
            self.release_scores[eye_side] = 0.0
            return np.zeros(2, dtype=np.float32)

        if eye_side not in self.stable_offsets:
            self.stable_offsets[eye_side] = raw
            self.smoothed_offsets[eye_side] = raw
            self.release_scores[eye_side] = 0.0
            return np.zeros(2, dtype=np.float32)

        previous_raw = self.smoothed_offsets[eye_side]
        raw_step = float(np.linalg.norm(raw - previous_raw))
        raw_smooth_alpha = 0.22 + 0.24 * (1.0 - stabilizer_strength)
        raw_smoothed = previous_raw * (1.0 - raw_smooth_alpha) + raw * raw_smooth_alpha

        stable = self.stable_offsets[eye_side]
        distance = float(np.linalg.norm(raw_smoothed - stable))

        # A real gaze change should not be dragged back like reading jitter.
        if raw_step > 0.11 and distance > 0.54:
            self.smoothed_offsets[eye_side] = raw_smoothed
            self.stable_offsets[eye_side] = raw_smoothed
            self.release_scores[eye_side] = 1.0
            return np.zeros(2, dtype=np.float32)

        release_score = self.release_scores.get(eye_side, 0.0) * 0.86
        release_distance = 0.24 + 0.08 * stabilizer_strength
        if raw_step > 0.070 and distance > 0.36:
            release_score += 0.24
        elif distance > release_distance:
            release_score += min(0.08 + (distance - release_distance) * 0.8, 0.22)
        else:
            release_score -= 0.04
        release_score = float(np.clip(release_score, 0.0, 1.0))

        # Strong stabilizer: ignore fast reading saccades, but follow sustained gaze shifts.
        base_follow = 0.004 + ((1.0 - stabilizer_strength) ** 2) * 0.18
        sustained_follow = max(0.0, (release_score - 0.55) / 0.45) * 0.32
        hard_follow = 0.0
        if distance > 0.55:
            hard_follow = min((distance - 0.55) * 0.45, 0.28)
        follow_alpha = float(np.clip(base_follow + sustained_follow + hard_follow, 0.004, 0.42))

        new_stable = stable * (1.0 - follow_alpha) + raw_smoothed * follow_alpha
        delta = raw_smoothed - new_stable
        deadband = 0.006 + 0.012 * (1.0 - stabilizer_strength)
        delta[np.abs(delta) < deadband] = 0.0
        delta = np.clip(delta, -0.70, 0.70)

        self.smoothed_offsets[eye_side] = raw_smoothed
        self.stable_offsets[eye_side] = new_stable
        self.release_scores[eye_side] = release_score
        return delta


class EyeOutputHoldFilter:
    """Temporally holds iris detail without freezing the whole eye crop."""

    def __init__(self):
        self.previous: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.previous.clear()

    def reset_eye(self, eye_side: str) -> None:
        self.previous.pop(eye_side, None)

    def _iris_weight(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        center = (int(w * 0.50), int(h * 0.53))
        axes = (max(2, int(w * 0.25)), max(2, int(h * 0.30)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        blur = max(5, int(min(h, w) * 0.18) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

        luma = image.mean(axis=2)
        dark = np.clip((0.48 - luma) / 0.32, 0.0, 1.0)
        return np.clip(mask * (0.35 + 0.65 * dark), 0.0, 1.0)

    def apply(self, eye_side: str, image: np.ndarray, hold_strength: float) -> np.ndarray:
        current = image.astype(np.float32)
        previous = self.previous.get(eye_side)

        if hold_strength <= 0.02:
            if previous is None or previous.shape != current.shape:
                self.previous[eye_side] = current
                return image

            diff = float(np.mean(np.abs(current - previous)))
            if diff < 0.075:
                alpha = 0.42
            elif diff < 0.135:
                alpha = 0.62
            else:
                self.previous[eye_side] = current
                return image

            smoothed = previous * (1.0 - alpha) + current * alpha
            self.previous[eye_side] = smoothed
            return smoothed

        if previous is None or previous.shape != current.shape:
            self.previous[eye_side] = current
            return current

        diff = float(np.mean(np.abs(current - previous)))

        if diff > 0.34 and hold_strength < 0.74:
            self.previous[eye_side] = current
            return image

        # High reading hold should stabilize the iris, not paste an old eyelid/skin crop.
        alpha = 0.008 + 0.34 * ((1.0 - hold_strength) ** 2.0)
        if diff > 0.24:
            alpha = max(alpha, 0.22)
        elif diff > 0.16:
            alpha = max(alpha, 0.06)

        held = previous * (1.0 - alpha) + current * alpha
        iris_weight = self._iris_weight(current)[:, :, None] * np.clip(hold_strength, 0.0, 1.0)
        output = held * iris_weight + current * (1.0 - iris_weight)
        self.previous[eye_side] = output
        return output


class ReadingActivityDetector:
    """Classifies reading from short-term iris motion patterns."""

    def __init__(self):
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=90)
        self.score = 0.0
        self.active = False
        self.active_until = 0.0
        self.release_until = 0.0
        self.last_release = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.score = 0.0
        self.active = False
        self.active_until = 0.0
        self.release_until = 0.0
        self.last_release = 0.0

    def update(
        self,
        left_offset: tuple[float, float],
        right_offset: tuple[float, float],
        eyes_closed: bool,
    ) -> tuple[bool, float]:
        now = time.monotonic()
        if eyes_closed:
            self.score *= 0.76
            self.active = self.score > 0.24 and now < self.active_until
            return self.active, self.score

        point = (
            np.asarray(left_offset, dtype=np.float32)
            + np.asarray(right_offset, dtype=np.float32)
        ) * 0.5
        self.history.append((now, point))

        while self.history and now - self.history[0][0] > 1.25:
            self.history.popleft()

        evidence, release = self._reading_evidence(now)
        self.last_release = release
        if release > 0:
            self.release_until = max(self.release_until, now + 0.20 + 0.08 * release)
            self.active_until = 0.0
            self.score = min(self.score * (1.0 - 0.92 * release), 0.08)
            if self.history:
                latest = self.history[-1]
                self.history.clear()
                self.history.append(latest)
        if now < self.release_until:
            evidence *= 0.04

        if evidence > 0.38:
            self.active_until = max(self.active_until, now + 0.46)

        if evidence > 0.56:
            rate = 0.86
        elif evidence > 0.38:
            rate = 0.74
        else:
            rate = 0.58 if evidence >= self.score else (0.18 if self.active else 0.38)
        self.score = float(np.clip(self.score * (1.0 - rate) + evidence * rate, 0.0, 1.0))

        if now < self.release_until:
            self.active = False
        elif self.active:
            self.active = self.score > 0.13 or now < self.active_until
        else:
            self.active = self.score > 0.22 or evidence > 0.46
        return self.active, self.score

    def _reading_evidence(self, now: float) -> tuple[float, float]:
        if len(self.history) < 4:
            return 0.0, 0.0

        window = [(t, point) for t, point in self.history if now - t <= 0.72]
        if len(window) < 4:
            window = list(self.history)

        times = np.asarray([t for t, _point in window], dtype=np.float32)
        points = np.asarray([point for _t, point in window], dtype=np.float32)
        diffs = np.diff(points, axis=0)
        dx = diffs[:, 0]
        dy = diffs[:, 1]

        abs_dx = np.abs(dx)
        horizontal_motion = float(np.mean(abs_dx))
        vertical_motion = float(np.mean(np.abs(dy)))
        total_horizontal_motion = float(np.sum(abs_dx))
        x_range = float(np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10))
        y_range = float(np.percentile(points[:, 1], 90) - np.percentile(points[:, 1], 10))
        mean_x_abs = float(abs(np.mean(points[:, 0])))
        end_x_abs = float(abs(points[-1, 0]))
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        active_horizontal_steps = abs_dx > 0.0022
        small_horizontal_steps = np.logical_and(abs_dx > 0.0022, abs_dx < 0.046)
        large_horizontal_steps = abs_dx > 0.070
        active_step_count = int(np.sum(active_horizontal_steps))
        small_step_count = int(np.sum(small_horizontal_steps))
        active_ratio = active_step_count / max(len(abs_dx), 1)
        active_span = 0.0
        active_indices = np.flatnonzero(active_horizontal_steps)
        if len(active_indices) >= 2:
            active_span = float(times[active_indices[-1] + 1] - times[active_indices[0]])

        significant = dx[np.abs(dx) > 0.003]
        sign_changes = 0
        if len(significant) > 2:
            signs = np.sign(significant)
            sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
        positive_motion = float(np.sum(significant[significant > 0.0])) if len(significant) else 0.0
        negative_motion = float(np.sum(np.abs(significant[significant < 0.0]))) if len(significant) else 0.0
        bidirectional_motion = min(positive_motion, negative_motion)
        bidirectional_ratio = bidirectional_motion / max(total_horizontal_motion, 1e-6)

        recent = dx[-min(7, len(dx)) :]
        recent_abs = float(np.sum(np.abs(recent)))
        recent_net = float(abs(np.sum(recent)))
        direct_ratio = recent_net / max(recent_abs, 1e-6)
        recent_significant = recent[np.abs(recent) > 0.004]
        recent_changes = 0
        if len(recent_significant) > 2:
            signs = np.sign(recent_significant)
            recent_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
        fresh = dx[-min(3, len(dx)) :]
        fresh_abs = float(np.sum(np.abs(fresh)))
        fresh_net = float(abs(np.sum(fresh)))
        fresh_direct_ratio = fresh_net / max(fresh_abs, 1e-6)
        fresh_max_step = float(np.max(np.abs(fresh))) if len(fresh) else 0.0
        fresh_significant = fresh[np.abs(fresh) > 0.004]
        fresh_changes = 0
        if len(fresh_significant) > 2:
            signs = np.sign(fresh_significant)
            fresh_changes = int(np.sum(signs[1:] * signs[:-1] < 0))

        max_step = float(np.max(abs_dx)) if len(abs_dx) else 0.0
        median_active_step = (
            float(np.median(abs_dx[active_horizontal_steps]))
            if np.any(active_horizontal_steps)
            else 0.0
        )
        net_horizontal_motion = float(abs(points[-1, 0] - points[0, 0]))
        path_direct_ratio = net_horizontal_motion / max(total_horizontal_motion, 1e-6)
        recent_max_step = float(np.max(np.abs(recent))) if len(recent) else 0.0

        side_fixation = (
            end_x_abs > 0.140
            and mean_x_abs > 0.112
            and x_range < 0.082
            and total_horizontal_motion < 0.185
            and max_step < 0.060
        )
        side_noise = (
            end_x_abs > 0.165
            and mean_x_abs > 0.125
            and x_range < 0.110
            and active_span > 0.20
            and total_horizontal_motion < 0.240
            and bidirectional_ratio > 0.16
        )
        side_zone = end_x_abs > 0.155 or mean_x_abs > 0.135
        center_zone = end_x_abs < 0.155 and mean_x_abs < 0.125

        direct_jump = (
            max_step > 0.070
            and path_direct_ratio > 0.64
            and sign_changes <= 1
        )
        single_direction_glance = (
            total_horizontal_motion > 0.045
            and path_direct_ratio > 0.82
            and bidirectional_ratio < 0.10
            and sign_changes <= 1
            and (small_step_count <= 3 or max_step > 0.050)
        )
        slow_direct_glance = (
            total_horizontal_motion > 0.036
            and path_direct_ratio > 0.82
            and bidirectional_ratio < 0.10
            and sign_changes <= 1
            and active_span < 0.72
            and small_step_count <= 4
        )
        recent_direct_release = (
            recent_abs > 0.050
            and recent_max_step > 0.026
            and direct_ratio > 0.78
            and recent_changes == 0
            and small_step_count <= 4
        ) or (
            fresh_abs > 0.045
            and fresh_max_step > 0.026
            and fresh_direct_ratio > 0.84
            and fresh_changes == 0
            and small_step_count <= 4
        )

        directed_glance = (
            direct_jump
            or single_direction_glance
            or slow_direct_glance
            or recent_direct_release
            or side_fixation
            or side_noise
            or (
                side_zone
                and not (x_range > 0.115 and bidirectional_ratio > 0.20)
                and total_horizontal_motion < 0.260
            )
        )
        if directed_glance:
            return 0.0, 1.0

        meaningful_scan_range = x_range > 0.010 and total_horizontal_motion > 0.018
        stable_vertical = y_range < 0.150 and vertical_motion < 0.026
        central_micro_read = (
            center_zone
            and small_step_count >= 3
            and active_step_count >= 3
            and active_ratio > 0.30
            and active_span > 0.10
            and total_horizontal_motion > 0.012
            and horizontal_motion > 0.0016
            and max_step < 0.042
            and median_active_step < 0.026
            and stable_vertical
            and path_direct_ratio < 0.86
        )
        progressive_reading_scan = (
            small_step_count >= 3
            and active_step_count >= 3
            and active_ratio > 0.34
            and active_span > 0.13
            and total_horizontal_motion > 0.018
            and horizontal_motion > 0.0022
            and max_step < 0.055
            and median_active_step < 0.030
            and stable_vertical
            and meaningful_scan_range
            and (center_zone or x_range > 0.055)
        )
        alternating_motion = (
            sign_changes >= 1
            and bidirectional_ratio > 0.10
            and small_step_count >= 3
            and total_horizontal_motion > 0.018
            and stable_vertical
            and (center_zone or x_range > 0.055)
        ) or (
            recent_changes >= 1
            and bidirectional_ratio > 0.10
            and small_step_count >= 3
            and stable_vertical
            and center_zone
        )
        long_micro_scan = (
            small_step_count >= 5
            and active_span > 0.22
            and max_step < 0.040
            and total_horizontal_motion > 0.024
            and stable_vertical
            and center_zone
        )
        read_like = central_micro_read or progressive_reading_scan or alternating_motion or long_micro_scan
        if not read_like:
            return 0.0, 0.0

        step_score = np.clip((small_step_count - 2.0) / 5.0, 0.0, 1.0)
        density_score = np.clip((active_ratio - 0.28) / 0.42, 0.0, 1.0)
        span_score = np.clip((active_span - 0.10) / 0.30, 0.0, 1.0)
        horizontal_score = np.clip((total_horizontal_motion - 0.018) / 0.095, 0.0, 1.0)
        range_score = np.clip((x_range - 0.012) / 0.105, 0.0, 1.0)
        return_score = np.clip(sign_changes / 3.0, 0.0, 1.0)
        bidirectional_score = np.clip((bidirectional_ratio - 0.07) / 0.22, 0.0, 1.0)
        dominance = horizontal_motion / max(vertical_motion, 0.003)
        dominance_score = np.clip((dominance - 1.1) / 3.0, 0.0, 1.0)
        vertical_penalty = np.clip((vertical_motion - 0.014) / 0.055, 0.0, 1.0)
        drift_penalty = np.clip((y_range - 0.18) / 0.25, 0.0, 1.0)
        big_jump_penalty = np.clip((int(np.sum(large_horizontal_steps)) - 0.2) / 1.6, 0.0, 1.0)

        evidence = (
            step_score * 0.26
            + density_score * 0.18
            + span_score * 0.17
            + horizontal_score * 0.16
            + range_score * 0.08
            + return_score * 0.08
            + bidirectional_score * 0.06
            + dominance_score * 0.05
            - vertical_penalty * 0.18
            - drift_penalty * 0.10
            - big_jump_penalty * 0.22
        )
        if progressive_reading_scan:
            evidence = max(evidence, 0.50)
        if central_micro_read:
            evidence = max(evidence, 0.50)
        if alternating_motion:
            evidence = max(evidence, 0.52 + min(sign_changes, 3) * 0.04)
        if long_micro_scan:
            evidence = max(evidence, 0.48)
        if progressive_reading_scan:
            evidence += min(max(duration - 0.18, 0.0) * 0.20, 0.06)

        return float(np.clip(evidence, 0.0, 1.0)), 0.0


################################################################################
# Gaze Correction Model
################################################################################


class GazeModel:
    """TensorFlow model wrapper for eye gaze correction."""

    def __init__(self, config: GazeModelConfig):
        self.cfg = config
        self.logger = Logger("GazeModel")
        self._load_models()

    def _load_models(self):
        """Load left and right eye models."""
        # Build ModelConfig for gaze_warp_model
        model_cfg = gaze_warp_model.ModelConfig(
            height=self.cfg.gaze_warp_model.height,
            width=self.cfg.gaze_warp_model.width,
            encoded_angle_dim=self.cfg.gaze_warp_model.encoded_angle_dim,
        )

        # Left eye model
        self.logger.log("Loading left eye model...")
        with tf.Graph().as_default() as g_left:
            with tf.name_scope("inputs"):
                self.le_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.le_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.le_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.le_pred, _, _ = gaze_warp_model.build_inference_graph(
                self.le_img, self.le_fp, self.le_ang, False, model_cfg
            )
            self.l_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_left,
            )
            self._restore_checkpoint(self.l_sess, self.cfg.model_dir + "L/")

        # Right eye model
        self.logger.log("Loading right eye model...")
        with tf.Graph().as_default() as g_right:
            with tf.name_scope("inputs"):
                self.re_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.re_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.re_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.re_pred, _, _ = gaze_warp_model.build_inference_graph(
                self.re_img, self.re_fp, self.re_ang, False, model_cfg
            )
            self.r_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_right,
            )
            self._restore_checkpoint(self.r_sess, self.cfg.model_dir + "R/")

        self.logger.log("Models loaded successfully")

    def _restore_checkpoint(self, sess, model_dir: str):
        """Restore model from checkpoint."""
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            self.logger.log(f"Warning: No checkpoint found in {model_dir}")

    def infer_eye(
        self, eye: str, img: np.ndarray, anchor_map: np.ndarray, angle: list
    ) -> np.ndarray:
        """
        Run inference for a single eye.

        Args:
            eye: "L" or "R"
            img: Eye image normalized to [0, 1], shape (H, W, 3)
            anchor_map: Feature point map, shape (H, W, ef_dim)
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image, shape (H, W, 3)
        """
        if eye == "L":
            result = self.l_sess.run(
                self.le_pred,
                feed_dict={
                    self.le_img: np.expand_dims(img, axis=0),
                    self.le_fp: np.expand_dims(anchor_map, axis=0),
                    self.le_ang: np.expand_dims(angle, axis=0),
                },
            )
        else:
            result = self.r_sess.run(
                self.re_pred,
                feed_dict={
                    self.re_img: np.expand_dims(img, axis=0),
                    self.re_fp: np.expand_dims(anchor_map, axis=0),
                    self.re_ang: np.expand_dims(angle, axis=0),
                },
            )
        return result.reshape(self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], 3)

    def close(self):
        """Close TensorFlow sessions."""
        self.l_sess.close()
        self.r_sess.close()


################################################################################
# Gaze Corrector
################################################################################


class GazeCorrector:
    """
    High-level gaze correction interface with database-backed user settings.
    
    Takes FaceData from a FacePredictor and applies gaze correction.
    video_size is passed from outside via apply_correction.
    """

    def __init__(
        self,
        config_path: str = "./model_managers/gaze_corrector_v1_01.yaml",
        db_path: str = "./user_settings.db",
        setting_name: str = "camera_default",
    ):
        """
        Initialize gaze corrector.
        
        Args:
            config_path: Path to YAML configuration file
            db_path: Path to SQLite database
            setting_name: Name of camera setting to load from database
        """
        self.logger = Logger("GazeCorrector")
        
        # Load model configuration from YAML
        self.model_cfg = GazeModelConfig.from_yaml(config_path)
        self.logger.log(f"Loaded model config from: {config_path}")
        
        # Initialize database
        self.db = UserSettingsDB(db_path)
        self.setting_name = setting_name
        
        # Load camera settings from database or use defaults
        self.camera_settings = self._load_camera_settings()
        self.tuning_settings = self._load_tuning_settings()
        
        # Initialize model
        self.model = GazeModel(self.model_cfg)

        # Pixel border to cut when replacing eyes (reduces edge artifacts)
        self.pixel_cut = (3, 4)

        # Stabilize detector/model jitter while preserving small natural eye motion
        self.stabilization_cfg = GazeStabilizationConfig()
        self.gaze_filter = OneEuroVectorFilter(self.stabilization_cfg)
        self.pupil_hold_filter = PupilHoldFilter()
        self.eye_output_hold_filter = EyeOutputHoldFilter()
        self.reading_detector = ReadingActivityDetector()
        self._apply_tuning_to_filter()

        # Last estimated eye position (for visualization)
        self.last_eye_position: list[float] = [0, 0, -60]
        self.last_reading_active = False
        self.last_reading_score = 0.0
        self.last_effective_hold = 0.0
        self.hold_filter_value = 0.0
        self.hold_filter_time = time.monotonic()
        self.prev_face_motion: tuple[float, float, float, float] | None = None
        self.motion_guard_until = 0.0
        self.eye_closed_previous = {"L": False, "R": False}
        self.eye_reopen_time = {"L": 0.0, "R": 0.0}

    def _load_camera_settings(self) -> CameraUserSetting:
        """Load camera settings from database or return defaults."""
        saved = self.db.get_setting(self.setting_name)
        if saved:
            self.logger.log(f"Loaded camera settings from database: {self.setting_name}")
            return CameraUserSetting.from_dict(saved)
        else:
            self.logger.log("Using default camera settings")
            settings = CameraUserSetting()
            # Save defaults to database
            self.db.save_setting(self.setting_name, settings.to_dict())
            return settings

    def _load_tuning_settings(self) -> GazeTuningSetting:
        """Load gaze tuning controls from database or return defaults."""
        saved = self.db.get_setting("gaze_tuning")
        if saved:
            self.logger.log("Loaded gaze tuning from database")
            return GazeTuningSetting.from_dict(saved)

        settings = GazeTuningSetting()
        self.db.save_setting("gaze_tuning", settings.to_dict())
        return settings

    def save_camera_settings(self):
        """Save current camera settings to database."""
        self.db.save_setting(self.setting_name, self.camera_settings.to_dict())
        self.logger.log(f"Saved camera settings to database: {self.setting_name}")

    def save_tuning_settings(self):
        """Save current gaze tuning controls to database."""
        self.db.save_setting("gaze_tuning", self.tuning_settings.to_dict())

    ############################################################################
    # Camera Offset Adjustment API
    ############################################################################

    def get_camera_offset(self) -> tuple[float, float, float]:
        """Get current camera offset (x, y, z) in cm."""
        return self.camera_settings.camera_offset

    def set_camera_offset(self, x: float, y: float, z: float) -> None:
        """
        Set camera offset relative to screen center.

        Args:
            x: Horizontal offset in cm (positive = right)
            y: Vertical offset in cm (positive = down)
            z: Depth offset in cm (negative = behind screen)
        """
        self.camera_settings.camera_offset = (x, y, z)
        self.save_camera_settings()
        self.logger.log(f"Camera offset set to: ({x:.1f}, {y:.1f}, {z:.1f})")

    def adjust_camera_offset(self, dx: float = 0, dy: float = 0, dz: float = 0) -> tuple[float, float, float]:
        """
        Adjust camera offset by delta values.

        Args:
            dx: Change in X (horizontal)
            dy: Change in Y (vertical)
            dz: Change in Z (depth)

        Returns:
            New camera offset tuple
        """
        x, y, z = self.camera_settings.camera_offset
        self.camera_settings.camera_offset = (x + dx, y + dy, z + dz)
        self.save_camera_settings()
        return self.camera_settings.camera_offset

    def get_last_eye_position(self) -> list[float]:
        """Get the last estimated eye position [x, y, z] in cm."""
        return self.last_eye_position

    ############################################################################
    # Focal Length Adjustment API
    ############################################################################

    def get_focal_length(self) -> float:
        """Get current focal length in pixels."""
        return self.camera_settings.focal_length

    def set_focal_length(self, focal_length: float) -> None:
        """
        Set focal length.

        Args:
            focal_length: Focal length in pixels (typically 500-1000)
        """
        self.camera_settings.focal_length = focal_length
        self.save_camera_settings()
        self.logger.log(f"Focal length set to: {focal_length:.1f}")

    def adjust_focal_length(self, delta: float) -> float:
        """
        Adjust focal length by delta value.

        Args:
            delta: Change in focal length (pixels)

        Returns:
            New focal length
        """
        self.camera_settings.focal_length += delta
        self.save_camera_settings()
        return self.camera_settings.focal_length

    ############################################################################
    # IPD Adjustment API
    ############################################################################

    def get_ipd(self) -> float:
        """Get inter-pupillary distance in cm."""
        return self.camera_settings.ipd

    def set_ipd(self, ipd: float) -> None:
        """
        Set inter-pupillary distance.

        Args:
            ipd: IPD in cm (typically 5.5-7.0)
        """
        self.camera_settings.ipd = ipd
        self.save_camera_settings()
        self.logger.log(f"IPD set to: {ipd:.1f} cm")

    ############################################################################
    # Gaze Estimation and Correction
    ############################################################################

    def reset_tracking(self) -> None:
        """Reset temporal state after face/eye tracking is lost."""
        self.gaze_filter.reset()
        self.pupil_hold_filter.reset()
        self.eye_output_hold_filter.reset()
        self.reading_detector.reset()
        self.last_reading_active = False
        self.last_reading_score = 0.0
        self.last_effective_hold = 0.0
        self.hold_filter_value = 0.0
        self.hold_filter_time = time.monotonic()
        self.prev_face_motion = None
        self.motion_guard_until = 0.0
        self.eye_closed_previous = {"L": False, "R": False}
        self.eye_reopen_time = {"L": 0.0, "R": 0.0}

    def set_stabilization_enabled(self, enabled: bool) -> None:
        """Enable or disable gaze stabilization at runtime."""
        self.stabilization_cfg.enabled = enabled
        if not enabled:
            self.gaze_filter.reset()

    def get_tuning(self) -> GazeTuningSetting:
        """Get current manual gaze tuning settings."""
        return self.tuning_settings

    def get_reading_state(self) -> tuple[bool, float, float]:
        """Return adaptive reading state for UI display."""
        return self.last_reading_active, self.last_reading_score, self.last_effective_hold

    def set_tuning(
        self,
        *,
        enabled: bool | None = None,
        strength: float | None = None,
        vertical_offset: float | None = None,
        horizontal_offset: float | None = None,
        smoothing: float | None = None,
        reading_stabilizer: float | None = None,
        natural_motion: float | None = None,
        save: bool = True,
        reset_tracking: bool = True,
    ) -> None:
        """Update manual gaze tuning controls."""
        should_reset_temporal_state = reset_tracking and any(
            value is not None
            for value in (
                enabled,
                strength,
                smoothing,
                reading_stabilizer,
                natural_motion,
            )
        )

        if enabled is not None:
            self.tuning_settings.enabled = enabled
        if strength is not None:
            self.tuning_settings.strength = max(0.0, min(strength, 1.5))
        if vertical_offset is not None:
            self.tuning_settings.vertical_offset = max(-90.0, min(vertical_offset, 90.0))
        if horizontal_offset is not None:
            self.tuning_settings.horizontal_offset = max(-45.0, min(horizontal_offset, 45.0))
        if smoothing is not None:
            self.tuning_settings.smoothing = max(0.0, min(smoothing, 1.0))
        if reading_stabilizer is not None:
            self.tuning_settings.reading_stabilizer = max(0.0, min(reading_stabilizer, 1.0))
        if natural_motion is not None:
            self.tuning_settings.natural_motion = max(0.0, min(natural_motion, 1.0))

        self._apply_tuning_to_filter()
        if should_reset_temporal_state:
            self.reset_tracking()
        if save:
            self.save_tuning_settings()

    def reset_tuning(self) -> None:
        """Restore manual gaze tuning controls to defaults."""
        self.tuning_settings = GazeTuningSetting()
        self._apply_tuning_to_filter()
        self.save_tuning_settings()
        self.reset_tracking()

    def _apply_tuning_to_filter(self) -> None:
        """Map simple UI smoothing controls to filter parameters."""
        smoothing = self.tuning_settings.smoothing
        self.stabilization_cfg.min_cutoff = 2.4 - (2.0 * smoothing)
        self.stabilization_cfg.beta = 0.16 - (0.12 * smoothing)
        self.stabilization_cfg.reading_deadband_degrees = 0.15 + (0.55 * smoothing)
        self.stabilization_cfg.reading_lock_strength = 0.45 + (0.45 * smoothing)
        self.stabilization_cfg.max_step_degrees = 2.4 - (1.35 * smoothing)
        self.stabilization_cfg.natural_motion_strength = 0.28 * self.tuning_settings.natural_motion

    def estimate_gaze_angle(
        self, 
        le_center: tuple[float, float], 
        re_center: tuple[float, float],
        video_size: tuple[int, int],
        natural_scale: float = 1.0,
    ) -> tuple[list[float], list[float]]:
        """
        Estimate gaze redirection angles based on eye positions.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels
            video_size: (width, height) of video frame

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        settings = self.camera_settings

        # Estimate eye depth from inter-pupillary distance
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(settings.focal_length * settings.ipd) / ipd_pixels

        # Estimate eye position in 3D (camera coordinates, cm)
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - video_size[0])
            / (2 * settings.focal_length)
            + settings.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - video_size[1])
            / (2 * settings.focal_length)
            + settings.camera_offset[1]
        )

        eye_position = [eye_x, eye_y, eye_z]

        # Store for visualization
        self.last_eye_position = eye_position

        # Target gaze point (looking at camera)
        target = (0, 0, 0)

        # Calculate angles
        a_v = math.degrees(math.atan((target[1] - eye_y) / (target[2] - eye_z)))
        a_h = math.degrees(math.atan((target[0] - eye_x) / (target[2] - eye_z)))

        # Add camera offset angles
        a_v += math.degrees(
            math.atan((eye_y - settings.camera_offset[1]) / (settings.camera_offset[2] - eye_z))
        )
        a_h += math.degrees(
            math.atan((eye_x - settings.camera_offset[0]) / (settings.camera_offset[2] - eye_z))
        )

        tuning = self.tuning_settings
        tuned_alpha = [
            a_v * tuning.strength + tuning.vertical_offset,
            a_h * tuning.strength + tuning.horizontal_offset,
        ]

        if self.stabilization_cfg.enabled:
            return self.gaze_filter.apply(tuned_alpha, natural_scale), eye_position

        return tuned_alpha, eye_position

    def _adaptive_reading_hold(self, le, re, le_closed: bool, re_closed: bool) -> float:
        active, score = self.reading_detector.update(
            le.pupil_offset,
            re.pupil_offset,
            le_closed or re_closed,
        )
        now = time.monotonic()
        release_active = now < self.reading_detector.release_until

        if release_active:
            target_hold = 0.0
            active = False
            if self.hold_filter_value > 0.02:
                self.pupil_hold_filter.reset()
                self.eye_output_hold_filter.reset()
            self.hold_filter_value = 0.0
        elif active:
            intensity = np.clip((score - 0.045) / 0.255, 0.0, 1.0)
            target_hold = float(self.tuning_settings.reading_stabilizer * intensity)
        else:
            target_hold = 0.0

        dt = max(1.0 / 120.0, min(now - self.hold_filter_time, 0.12))
        self.hold_filter_time = now
        tau = 0.022 if target_hold >= self.hold_filter_value else 0.052
        alpha = 1.0 - math.exp(-dt / tau)
        hold = self.hold_filter_value * (1.0 - alpha) + target_hold * alpha
        self.hold_filter_value = float(np.clip(hold, 0.0, 1.0))

        hold = self.hold_filter_value
        if hold < 0.025:
            hold = 0.0
            self.hold_filter_value = 0.0
            if target_hold <= 0.001:
                self.pupil_hold_filter.reset()
                self.eye_output_hold_filter.reset()
        self.last_reading_active = active
        self.last_reading_score = score
        self.last_effective_hold = hold
        return hold

    def _face_motion_guard_active(self, le, re) -> bool:
        """Suppress reading hold during quick head/face movement."""
        now = time.monotonic()
        center_x = (le.center[0] + re.center[0]) * 0.5
        center_y = (le.center[1] + re.center[1]) * 0.5
        ipd = max(
            float(np.hypot(le.center[0] - re.center[0], le.center[1] - re.center[1])),
            1.0,
        )

        previous = self.prev_face_motion
        self.prev_face_motion = (center_x, center_y, ipd, now)
        if previous is None:
            return False

        prev_x, prev_y, prev_ipd, _prev_time = previous
        motion = float(np.hypot(center_x - prev_x, center_y - prev_y)) / ipd
        scale_motion = abs(ipd - prev_ipd) / max(prev_ipd, 1.0)

        if motion > 0.160 or scale_motion > 0.120:
            self.motion_guard_until = now + 0.25
            self.pupil_hold_filter.reset()
            self.eye_output_hold_filter.reset()
            self.reading_detector.history.clear()
            self.reading_detector.score = 0.0
            self.reading_detector.active = False
            self.reading_detector.active_until = 0.0
            self.reading_detector.release_until = max(self.reading_detector.release_until, now + 0.25)

        return now < self.motion_guard_until

    def _get_pupil_delta(
        self, eye_data, eye_side: str, stabilizer: float
    ) -> np.ndarray:
        if stabilizer <= 0.01:
            return np.zeros(2, dtype=np.float32)

        return self.pupil_hold_filter.apply(
            eye_side, eye_data.pupil_offset, stabilizer
        )

    def _paired_pupil_deltas(
        self, le, re, stabilizer: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use one binocular stabilization signal so both pupils stay aligned."""
        le_delta = self._get_pupil_delta(le, "L", stabilizer)
        re_delta = self._get_pupil_delta(re, "R", stabilizer)
        if stabilizer <= 0.01:
            return le_delta, re_delta

        paired_delta = (le_delta + re_delta) * 0.5
        pair_lock = stabilizer ** 1.8
        locked_le = le_delta * (1.0 - pair_lock) + paired_delta * pair_lock
        locked_re = re_delta * (1.0 - pair_lock) + paired_delta * pair_lock
        return (
            self._limit_pupil_delta(locked_le, stabilizer),
            self._limit_pupil_delta(locked_re, stabilizer),
        )

    def _limit_pupil_delta(self, delta: np.ndarray, stabilizer: float) -> np.ndarray:
        if stabilizer <= 0.01:
            return delta

        max_x = 0.105 + 0.055 * (1.0 - stabilizer)
        max_y = 0.060 + 0.035 * (1.0 - stabilizer)
        limited = np.asarray(
            [
                np.clip(delta[0], -max_x, max_x),
                np.clip(delta[1], -max_y, max_y),
            ],
            dtype=np.float32,
        )
        max_mag = 0.125 + 0.070 * (1.0 - stabilizer)
        mag = float(np.linalg.norm(limited))
        if mag > max_mag:
            limited *= max_mag / mag
        return limited

    def _safe_pupil_compensation(
        self, delta: np.ndarray, stabilizer: float
    ) -> tuple[float, float]:
        gain = stabilizer ** 1.35
        vertical = -float(delta[1]) * 82.0 * gain
        horizontal = -float(delta[0]) * 104.0 * gain
        max_vertical = 4.8 + 2.4 * (1.0 - stabilizer)
        max_horizontal = 7.2 + 3.2 * (1.0 - stabilizer)
        return (
            float(np.clip(vertical, -max_vertical, max_vertical)),
            float(np.clip(horizontal, -max_horizontal, max_horizontal)),
        )

    def _safe_final_angle(self, angle: list[float], stabilizer: float) -> list[float]:
        if stabilizer <= 0.01:
            return angle

        max_vertical = 42.0 + 18.0 * (1.0 - stabilizer)
        max_horizontal = 32.0 + 12.0 * (1.0 - stabilizer)
        return [
            float(np.clip(angle[0], -max_vertical, max_vertical)),
            float(np.clip(angle[1], -max_horizontal, max_horizontal)),
        ]

    def _angle_with_pupil_delta(
        self, angle: list[float], delta: np.ndarray, stabilizer: float
    ) -> list[float]:
        if stabilizer <= 0.01:
            return angle

        vertical_comp, horizontal_comp = self._safe_pupil_compensation(delta, stabilizer)
        return self._safe_final_angle(
            [
                angle[0] + vertical_comp,
                angle[1] + horizontal_comp,
            ],
            stabilizer,
        )

    def _eye_is_closed(self, eye_data) -> bool:
        """Detect blink/closed eye; DeepWarp must not draw an open pupil then."""
        return float(getattr(eye_data, "openness", 1.0)) < 0.19

    def _update_blink_state(self, eye_side: str, closed: bool) -> None:
        now = time.monotonic()
        was_closed = self.eye_closed_previous.get(eye_side, False)
        if closed:
            self.eye_closed_previous[eye_side] = True
            self.eye_reopen_time[eye_side] = 0.0
            return

        if was_closed:
            self.eye_reopen_time[eye_side] = now
            self.eye_closed_previous[eye_side] = False

    def _reopen_blend_strength(self, eye_side: str) -> float:
        reopened_at = self.eye_reopen_time.get(eye_side, 0.0)
        if reopened_at <= 0.0:
            return 1.0

        age = time.monotonic() - reopened_at
        if age < 0.045:
            return 0.0
        if age >= 0.220:
            return 1.0

        t = np.clip((age - 0.045) / 0.175, 0.0, 1.0)
        return float(t * t * (3.0 - 2.0 * t))

    def _reset_eye_temporal_state(self, eye_side: str) -> None:
        self.pupil_hold_filter.reset_eye(eye_side)
        self.eye_output_hold_filter.reset_eye(eye_side)

    def _shift_corrected_eye(
        self, image: np.ndarray, delta: np.ndarray, stabilizer: float
    ) -> np.ndarray:
        """Move the generated eye as one piece, preserving pupil/iris shape."""
        if stabilizer <= 0.01 or stabilizer >= 0.05:
            return image

        img = image.astype(np.float32)
        h, w = img.shape[:2]
        shift_gain = stabilizer ** 1.45
        shift_x = float(np.clip(-delta[0] * 8.0 * shift_gain, -1.2, 1.2))
        shift_y = float(np.clip(-delta[1] * 6.0 * shift_gain, -0.8, 0.8))
        if abs(shift_x) < 0.08 and abs(shift_y) < 0.08:
            return image

        matrix = np.asarray([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
        shifted = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return np.clip(shifted, 0.0, 1.0)

    def correct_eye(
        self,
        eye_data,
        eye_side: str,
        angle: list[float],
        hold_strength: float,
        pupil_delta: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply gaze correction to a single eye.

        Args:
            eye_data: Eye extraction data (EyeData from face_predictor)
            eye_side: "L" or "R"
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image resized to original size
        """
        result = self.model.infer_eye(
            eye_side, eye_data.image, eye_data.anchor_map, angle
        )
        if pupil_delta is not None:
            result = self._shift_corrected_eye(result, pupil_delta, hold_strength)
        result = self.eye_output_hold_filter.apply(
            eye_side, result, hold_strength
        )
        # Resize back to original size
        return cv2.resize(result, (eye_data.original_size[1], eye_data.original_size[0]))

    def _eye_alpha_mask(
        self, eye_data, output_size: tuple[int, int], hold_strength: float = 0.0
    ) -> np.ndarray:
        """Create a feathered eye-only mask to avoid rectangular DeepWarp artifacts."""
        h, w = output_size
        mask = np.zeros((h, w), dtype=np.float32)

        if eye_data.mask_points:
            input_h, input_w = self.model_cfg.eye_input_size
            points = np.asarray(
                [
                    [
                        float(x) * w / input_w,
                        float(y) * h / input_h,
                    ]
                    for x, y in eye_data.mask_points
                ],
                dtype=np.float32,
            )
            if len(points) >= 3:
                hull = cv2.convexHull(points.astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1.0)
        else:
            center = (w // 2, h // 2)
            axes = (max(2, int(w * 0.30)), max(2, int(h * 0.20)))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        if mask.max() <= 0:
            return mask

        base = mask.copy()
        if hold_strength < 0.12:
            dilate = max(1, int(min(h, w) * 0.026))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)
            )
            base = cv2.dilate(base, kernel, iterations=1)

        blur = max(7, int(min(h, w) * 0.12) | 1)
        soft = cv2.GaussianBlur(base, (blur, blur), 0)
        if hold_strength >= 0.12:
            soft = np.minimum(soft, mask)

        max_alpha = 0.84 - 0.12 * np.clip(hold_strength, 0.0, 1.0)
        return np.clip(soft * max_alpha, 0.0, max_alpha)

    def _pupil_safe_mask(
        self, eye_data, output_size: tuple[int, int], hold_strength: float
    ) -> np.ndarray:
        h, w = output_size
        mask = np.zeros((h, w), dtype=np.float32)
        input_h, input_w = self.model_cfg.eye_input_size
        if getattr(eye_data, "pupil_center", None) is not None:
            cx = float(eye_data.pupil_center[0]) * w / input_w
            cy = float(eye_data.pupil_center[1]) * h / input_h
        else:
            cx = w * 0.50
            cy = h * 0.53

        axes_x = max(2, int(w * (0.25 - 0.04 * hold_strength)))
        axes_y = max(2, int(h * (0.30 - 0.05 * hold_strength)))
        cv2.ellipse(
            mask,
            (int(cx), int(cy)),
            (axes_x, axes_y),
            0,
            0,
            360,
            1.0,
            -1,
        )
        blur = max(5, int(min(h, w) * 0.12) | 1)
        return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)

    def _sharpen_corrected_eye(self, image: np.ndarray) -> np.ndarray:
        """Recover a little crispness from the low-resolution generated eye crop."""
        img = image.astype(np.float32)
        blurred = cv2.GaussianBlur(img, (0, 0), 0.85)
        sharpened = cv2.addWeighted(img, 1.28, blurred, -0.28, 0)
        return np.clip(sharpened, 0.0, 1.0)

    def _match_eye_tone(
        self,
        corrected: np.ndarray,
        original: np.ndarray,
        alpha: np.ndarray,
        hold_strength: float,
    ) -> np.ndarray:
        region = alpha[:, :, 0] > 0.20
        if int(np.sum(region)) < 12:
            return corrected

        corrected_matched = corrected.copy()
        strength = 0.26 + 0.16 * np.clip(hold_strength, 0.0, 1.0)
        for channel in range(3):
            c_values = corrected[:, :, channel][region]
            o_values = original[:, :, channel][region]
            c_mean = float(np.mean(c_values))
            o_mean = float(np.mean(o_values))
            corrected_matched[:, :, channel] += (o_mean - c_mean) * strength
        return np.clip(corrected_matched, 0.0, 1.0)

    def _blend_eye_region(
        self,
        frame: np.ndarray,
        eye_data,
        corrected_eye: np.ndarray,
        hold_strength: float = 0.0,
        correction_alpha: float = 1.0,
    ) -> None:
        """Blend the corrected eye through a soft eye contour mask."""
        row, col = eye_data.top_left
        roi_h, roi_w = eye_data.original_size
        frame_h, frame_w = frame.shape[:2]

        dst_top = max(row, 0)
        dst_left = max(col, 0)
        dst_bottom = min(row + roi_h, frame_h)
        dst_right = min(col + roi_w, frame_w)
        if dst_bottom <= dst_top or dst_right <= dst_left:
            return

        src_top = dst_top - row
        src_left = dst_left - col
        src_bottom = src_top + (dst_bottom - dst_top)
        src_right = src_left + (dst_right - dst_left)

        corrected = self._sharpen_corrected_eye(corrected_eye)
        mask = self._eye_alpha_mask(eye_data, eye_data.original_size, hold_strength)
        pupil_safe = self._pupil_safe_mask(eye_data, eye_data.original_size, hold_strength)

        corrected_crop = corrected[src_top:src_bottom, src_left:src_right]
        alpha = mask[src_top:src_bottom, src_left:src_right, None]
        alpha = alpha * np.clip(correction_alpha, 0.0, 1.0)
        original = frame[dst_top:dst_bottom, dst_left:dst_right].astype(np.float32) / 255.0
        corrected_crop = self._match_eye_tone(
            corrected_crop,
            original,
            alpha,
            hold_strength,
        )

        corrected_luma = corrected_crop.mean(axis=2)
        original_luma = original.mean(axis=2)
        safe_crop = pupil_safe[src_top:src_bottom, src_left:src_right]
        pupil_spill = np.logical_and(
            hold_strength > 0.10,
            np.logical_and(corrected_luma < 0.30, safe_crop < 0.18),
        ).astype(np.float32)
        if pupil_spill.max() > 0:
            pupil_spill = cv2.GaussianBlur(pupil_spill, (7, 7), 0)
            alpha = alpha * (1.0 - pupil_spill[:, :, None] * (0.35 + 0.45 * hold_strength))

        dark_artifact = np.logical_and(
            corrected_luma < 0.035,
            original_luma - corrected_luma > 0.16,
        ).astype(np.float32)
        if dark_artifact.max() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dark_artifact = cv2.dilate(dark_artifact, kernel, iterations=1)
            dark_artifact = cv2.GaussianBlur(dark_artifact, (7, 7), 0)
            alpha = alpha * (1.0 - dark_artifact[:, :, None])

        blended = corrected_crop * alpha + original * (1.0 - alpha)
        frame[dst_top:dst_bottom, dst_left:dst_right] = np.clip(
            blended * 255.0, 0, 255
        ).astype(np.uint8)

    def apply_correction(self, frame: np.ndarray, face_data, video_size: tuple[int, int]) -> np.ndarray:
        """
        Apply gaze correction to a frame using extracted face data.

        Args:
            frame: BGR video frame to modify
            face_data: Extracted face/eye data from FacePredictor (FaceData)
            video_size: (width, height) of video frame

        Returns:
            Frame with corrected gaze
        """
        if face_data.left_eye is None or face_data.right_eye is None:
            return frame
        if not self.tuning_settings.enabled:
            return frame

        le = face_data.left_eye
        re = face_data.right_eye
        le_closed = self._eye_is_closed(le)
        re_closed = self._eye_is_closed(re)

        self._update_blink_state("L", le_closed)
        self._update_blink_state("R", re_closed)
        if le_closed:
            self._reset_eye_temporal_state("L")
        if re_closed:
            self._reset_eye_temporal_state("R")
        if le_closed and re_closed:
            self._adaptive_reading_hold(le, re, le_closed, re_closed)
            return frame

        motion_guard = self._face_motion_guard_active(le, re)
        hold_strength = self._adaptive_reading_hold(le, re, le_closed, re_closed)
        if motion_guard:
            hold_strength = 0.0
            self.last_reading_active = False
            self.last_effective_hold = 0.0
        natural_scale = 1.0 - min(hold_strength, 1.0) * 0.96

        # Estimate gaze angle (video_size passed from outside)
        alpha, _ = self.estimate_gaze_angle(
            le.center, re.center, video_size, natural_scale=natural_scale
        )
        if not le_closed and not re_closed:
            le_delta, re_delta = self._paired_pupil_deltas(le, re, hold_strength)
        else:
            le_delta = (
                self._get_pupil_delta(le, "L", hold_strength)
                if not le_closed
                else np.zeros(2, dtype=np.float32)
            )
            re_delta = (
                self._get_pupil_delta(re, "R", hold_strength)
                if not re_closed
                else np.zeros(2, dtype=np.float32)
            )

        if not le_closed:
            le_blend = self._reopen_blend_strength("L")
            le_alpha = self._angle_with_pupil_delta(alpha, le_delta, hold_strength)
            le_hold = hold_strength * le_blend
            if le_blend > 0.01:
                le_corrected = self.correct_eye(le, "L", le_alpha, le_hold, le_delta)
                self._blend_eye_region(frame, le, le_corrected, le_hold, le_blend)
        if not re_closed:
            re_blend = self._reopen_blend_strength("R")
            re_alpha = self._angle_with_pupil_delta(alpha, re_delta, hold_strength)
            re_hold = hold_strength * re_blend
            if re_blend > 0.01:
                re_corrected = self.correct_eye(re, "R", re_alpha, re_hold, re_delta)
                self._blend_eye_region(frame, re, re_corrected, re_hold, re_blend)

        return frame

    def close(self):
        """Release model resources."""
        self.model.close()
