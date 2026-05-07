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
    smoothing: float = 0.86
    reading_stabilizer: float = 0.95
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
        data.setdefault("reading_stabilizer", 0.95)
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

    def apply(self, value: list[float]) -> list[float]:
        """Return a smoothed gaze angle vector [vertical, horizontal]."""
        now = time.monotonic()
        value_np = np.asarray(value, dtype=np.float32)

        if self.prev_value is None or self.prev_time is None:
            self.prev_value = value_np
            self.prev_derivative = np.zeros_like(value_np)
            self.prev_time = now
            return self._add_natural_motion(value_np).tolist()

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

        return self._add_natural_motion(filtered).tolist()

    def _add_natural_motion(self, value: np.ndarray) -> np.ndarray:
        if self.cfg.natural_motion_strength <= 0:
            return value

        t = time.monotonic() - self.started_at
        vertical = math.sin(t * 1.7) * 0.55 + math.sin(t * 0.53 + 1.2) * 0.45
        horizontal = math.sin(t * 1.3 + 2.1) * 0.6 + math.sin(t * 0.47) * 0.4
        natural = np.asarray([vertical, horizontal], dtype=np.float32)
        return value + natural * self.cfg.natural_motion_strength


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
        raw_smooth_alpha = 0.22 + 0.24 * (1.0 - stabilizer_strength)
        raw_smoothed = previous_raw * (1.0 - raw_smooth_alpha) + raw * raw_smooth_alpha

        stable = self.stable_offsets[eye_side]
        distance = float(np.linalg.norm(raw_smoothed - stable))

        release_score = self.release_scores.get(eye_side, 0.0) * 0.86
        release_distance = 0.24 + 0.08 * stabilizer_strength
        if distance > release_distance:
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
        self._apply_tuning_to_filter()

        # Last estimated eye position (for visualization)
        self.last_eye_position: list[float] = [0, 0, -60]

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

    def set_stabilization_enabled(self, enabled: bool) -> None:
        """Enable or disable gaze stabilization at runtime."""
        self.stabilization_cfg.enabled = enabled
        if not enabled:
            self.gaze_filter.reset()

    def get_tuning(self) -> GazeTuningSetting:
        """Get current manual gaze tuning settings."""
        return self.tuning_settings

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
    ) -> None:
        """Update manual gaze tuning controls."""
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
        if save:
            self.save_tuning_settings()

    def reset_tuning(self) -> None:
        """Restore manual gaze tuning controls to defaults."""
        self.tuning_settings = GazeTuningSetting()
        self._apply_tuning_to_filter()
        self.save_tuning_settings()
        self.gaze_filter.reset()

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
            return self.gaze_filter.apply(tuned_alpha), eye_position

        return tuned_alpha, eye_position

    def _get_pupil_delta(self, eye_data, eye_side: str) -> np.ndarray:
        stabilizer = self.tuning_settings.reading_stabilizer
        if stabilizer <= 0:
            return np.zeros(2, dtype=np.float32)

        return self.pupil_hold_filter.apply(
            eye_side, eye_data.pupil_offset, stabilizer
        )

    def _angle_with_pupil_delta(
        self, angle: list[float], delta: np.ndarray
    ) -> list[float]:
        stabilizer = self.tuning_settings.reading_stabilizer
        if stabilizer <= 0:
            return angle

        gain = stabilizer ** 1.35
        return [
            angle[0] - float(delta[1]) * 75.0 * gain,
            angle[1] - float(delta[0]) * 100.0 * gain,
        ]

    def _warp_eye_around_pupil(
        self,
        image: np.ndarray,
        pupil_center: tuple[float, float],
        delta: np.ndarray,
        strength_multiplier: float,
    ) -> np.ndarray:
        """Counter iris movement with a smooth local warp of the current eye image."""
        stabilizer = self.tuning_settings.reading_stabilizer
        if stabilizer <= 0:
            return image

        img = image.astype(np.float32)
        h, w = img.shape[:2]
        hold_gain = (0.35 + stabilizer * 0.85) * strength_multiplier
        eye_span_px = w * 2.0 / 3.0
        shift_x = float(np.clip(delta[0] * eye_span_px * hold_gain, -18.0, 18.0))
        shift_y = float(np.clip(delta[1] * eye_span_px * hold_gain, -14.0, 14.0))
        if abs(shift_x) < 0.15 and abs(shift_y) < 0.15:
            return img

        current_x, current_y = pupil_center
        target_x = float(np.clip(current_x - shift_x, 0, w - 1))
        target_y = float(np.clip(current_y - shift_y, 0, h - 1))

        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        sigma_x = max(w * 0.23, 6.0)
        sigma_y = max(h * 0.20, 5.0)
        weight = np.exp(
            -(
                ((grid_x - target_x) ** 2) / (2.0 * sigma_x * sigma_x)
                + ((grid_y - target_y) ** 2) / (2.0 * sigma_y * sigma_y)
            )
        ).astype(np.float32)

        map_x = grid_x + shift_x * weight
        map_y = grid_y + shift_y * weight
        stabilized = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return np.clip(stabilized, 0.0, 1.0)

    def _stabilize_eye_input(self, eye_data, delta: np.ndarray) -> np.ndarray:
        """Move the current iris toward the held target before model inference."""
        if eye_data.pupil_center is None:
            return eye_data.image

        return self._warp_eye_around_pupil(
            eye_data.image, eye_data.pupil_center, delta, strength_multiplier=1.0
        )

    def correct_eye(
        self,
        eye_data,
        eye_side: str,
        angle: list[float],
        input_image: np.ndarray | None = None,
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
        img = input_image if input_image is not None else eye_data.image
        result = self.model.infer_eye(
            eye_side, img, eye_data.anchor_map, angle
        )
        if pupil_delta is not None and eye_data.pupil_center is not None:
            result = self._warp_eye_around_pupil(
                result, eye_data.pupil_center, pupil_delta, strength_multiplier=0.38
            )
        # Resize back to original size
        return cv2.resize(result, (eye_data.original_size[1], eye_data.original_size[0]))

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

        # Estimate gaze angle (video_size passed from outside)
        alpha, _ = self.estimate_gaze_angle(le.center, re.center, video_size)
        le_delta = self._get_pupil_delta(le, "L")
        re_delta = self._get_pupil_delta(re, "R")
        le_alpha = self._angle_with_pupil_delta(alpha, le_delta)
        re_alpha = self._angle_with_pupil_delta(alpha, re_delta)
        le_input = self._stabilize_eye_input(le, le_delta)
        re_input = self._stabilize_eye_input(re, re_delta)

        # Correct both eyes
        le_corrected = self.correct_eye(le, "L", le_alpha, le_input, le_delta)
        re_corrected = self.correct_eye(re, "R", re_alpha, re_input, re_delta)

        # Replace eye regions in frame (with border cropping)
        pc = self.pixel_cut
        frame[
            le.top_left[0] + pc[0] : le.top_left[0] + le.original_size[0] - pc[0],
            le.top_left[1] + pc[1] : le.top_left[1] + le.original_size[1] - pc[1],
        ] = (le_corrected[pc[0] : -pc[0], pc[1] : -pc[1]] * 255)

        frame[
            re.top_left[0] + pc[0] : re.top_left[0] + re.original_size[0] - pc[0],
            re.top_left[1] + pc[1] : re.top_left[1] + re.original_size[1] - pc[1],
        ] = (re_corrected[pc[0] : -pc[0], pc[1] : -pc[1]] * 255)

        return frame

    def close(self):
        """Release model resources."""
        self.model.close()
