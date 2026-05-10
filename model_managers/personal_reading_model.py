"""Personal reading-state model trained from local user samples."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from utils.logger import Logger


FEATURE_NAMES = [
    "duration",
    "sample_count",
    "horizontal_motion",
    "vertical_motion",
    "total_horizontal_motion",
    "x_range",
    "y_range",
    "mean_x_abs",
    "end_x_abs",
    "active_ratio",
    "active_span",
    "small_step_count",
    "active_step_count",
    "sign_changes",
    "bidirectional_ratio",
    "max_step",
    "median_active_step",
    "path_direct_ratio",
    "recent_abs",
    "recent_direct_ratio",
    "recent_changes",
    "fresh_abs",
    "fresh_direct_ratio",
    "fresh_changes",
    "side_score",
    "center_score",
    "dominance",
]

POSITIVE_LABELS = {"read", "reading"}
NEGATIVE_LABELS = {"live", "glance", "look"}
VALID_LABELS = POSITIVE_LABELS | NEGATIVE_LABELS
TARGET_SAMPLES_PER_LABEL = 300
MIN_SAMPLES_PER_GROUP = 20
RECORDING_WARMUP_SECONDS = 0.65
MAX_HISTORY_FILES = 24


@dataclass
class TrainingResult:
    success: bool
    message: str
    samples: int = 0
    positives: int = 0
    negatives: int = 0
    accuracy: float = 0.0
    threshold: float = 0.5
    history_files: int = 0


class PersonalReadingModel:
    """Tiny standardized logistic model with no external dependencies."""

    def __init__(
        self,
        feature_names: list[str],
        mean: list[float],
        scale: list[float],
        weights: list[float],
        bias: float,
        threshold: float = 0.5,
        samples: int = 0,
        positives: int = 0,
        negatives: int = 0,
        training_accuracy: float = 0.0,
    ):
        self.feature_names = feature_names
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.samples = int(samples)
        self.positives = int(positives)
        self.negatives = int(negatives)
        self.training_accuracy = float(training_accuracy)

    @classmethod
    def load(cls, path: Path) -> "PersonalReadingModel | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mean = [float(value) for value in data["mean"]]
            scale = [float(value) for value in data["scale"]]
            weights = [float(value) for value in data["weights"]]
            bias = float(data["bias"])
            threshold = float(data.get("threshold", 0.5))
            samples = int(data.get("samples", 0) or 0)
            positives = int(data.get("positives", 0) or 0)
            negatives = int(data.get("negatives", 0) or 0)
            training_accuracy = float(data.get("training_accuracy", 0.0) or 0.0)
            values = [*mean, *scale, *weights, bias, threshold]
            if any(not math.isfinite(value) for value in values):
                return None
            return cls(
                feature_names=list(data["feature_names"]),
                mean=mean,
                scale=scale,
                weights=weights,
                bias=bias,
                threshold=threshold,
                samples=samples,
                positives=positives,
                negatives=negatives,
                training_accuracy=training_accuracy,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def predict(self, features: Mapping[str, float]) -> float:
        values = np.asarray(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )
        normalized = (values - self.mean) / self.scale
        logit = float(np.dot(normalized, self.weights) + self.bias)
        if not math.isfinite(logit):
            return 0.5
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_value = math.exp(logit)
        return exp_value / (1.0 + exp_value)


class PersonalReadingLearner:
    """Records labeled gaze-motion samples and trains the personal model."""

    def __init__(self, logger: Logger | None = None, support_dir: str | Path | None = None):
        self.logger = logger or Logger("PersonalReadingAI")
        self.support_dir = Path(support_dir) if support_dir else self._default_support_dir()
        self.support_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.support_dir / "reading_samples.csv"
        self.model_path = self.support_dir / "reading_model.json"
        self.state_path = Path(
            os.environ.get("SPEC3_TRAINING_STATE_FILE")
            or self.support_dir / "training_state.json"
        )
        self.model = PersonalReadingModel.load(self.model_path)
        self.recording_label: str | None = None
        self.recording_started_at = 0.0
        self.last_sample_time = 0.0
        self.last_prediction: float | None = None
        self.last_status = "Personal AI ready" if self.model else "No personal AI model yet"
        if self.model:
            self.logger.log(
                f"Loaded personal AI model from {self.model_path} "
                f"(threshold {self.model.threshold:.2f})"
            )
        self.sample_counts = self._count_samples()
        self._last_state_write_time = 0.0
        self._write_state()

    @staticmethod
    def _default_support_dir() -> Path:
        env_dir = os.environ.get("SPEC3_SUPPORT_DIR")
        if env_dir:
            return Path(env_dir)
        return Path.home() / "Library" / "Application Support" / "spec3 correction"

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def set_recording_label(self, label: str | None) -> str:
        normalized = label.lower() if label else None
        if normalized in {"stop", "none", ""}:
            normalized = None
        if normalized is not None and normalized not in VALID_LABELS:
            return f"Unknown training label: {label}"

        if normalized == "reading":
            normalized = "read"
        if normalized == "look":
            normalized = "glance"
        self.recording_label = normalized
        self.recording_started_at = time.monotonic()
        self.last_sample_time = 0.0
        if self.recording_label:
            self.last_status = (
                f"Recording {self.recording_label.upper()} samples "
                f"(warm-up {RECORDING_WARMUP_SECONDS:.1f}s)"
            )
        else:
            self.last_status = "Recording stopped"
        self.logger.log(self.last_status)
        self._write_state()
        return self.last_status

    def predict(self, features: Mapping[str, float]) -> float | None:
        if self.model is None:
            self.last_prediction = None
            return None
        probability = self.model.predict(features)
        self.last_prediction = probability
        if time.monotonic() - self._last_state_write_time > 0.50:
            self._write_state()
        return probability

    def record_if_needed(self, features: Mapping[str, float], now: float) -> None:
        if not self.recording_label:
            return
        if now - self.recording_started_at < RECORDING_WARMUP_SECONDS:
            return
        if now - self.last_sample_time < 0.10:
            return
        if not self._sample_is_clean(self.recording_label, features):
            return
        self.last_sample_time = now
        self._append_sample(self.recording_label, features)
        self.sample_counts[self.recording_label] = self.sample_counts.get(self.recording_label, 0) + 1
        total = sum(self.sample_counts.values())
        self.last_status = f"Recording {self.recording_label.upper()} samples: {total}"
        self._write_state()

    def train(self) -> TrainingResult:
        result = train_personal_reading_model(self.samples_path, self.model_path)
        self.last_status = result.message
        if result.success:
            self.model = PersonalReadingModel.load(self.model_path)
            self._archive_and_reset_samples(result)
            self.sample_counts = self._count_samples()
            self.recording_label = None
            self.last_status = (
                f"{result.message}. Current samples reset; saved history stays for future training."
            )
        else:
            self.sample_counts = self._count_samples()
        self.logger.log(result.message)
        self._write_state()
        return result

    def reset_samples(self) -> str:
        """Clear the current recording session without deleting the trained model."""
        self.recording_label = None
        if self.samples_path.exists():
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            archive_path = self.support_dir / f"reading_samples_reset_{timestamp}.csv"
            try:
                self.samples_path.replace(archive_path)
                self.last_status = "Training samples reset"
                self.logger.log(f"Archived reset samples to {archive_path}")
            except OSError as exc:
                self.last_status = f"Could not reset samples: {exc}"
                self.logger.log(self.last_status)
        else:
            self.last_status = "Training samples already empty"
        self.sample_counts = self._count_samples()
        self._write_state()
        return self.last_status

    def state(self) -> dict[str, object]:
        training_pool = self._count_training_pool()
        model_samples = self.model.samples if self.model else 0
        model_accuracy = self.model.training_accuracy if self.model else 0.0
        return {
            "has_model": self.has_model,
            "recording_label": self.recording_label,
            "samples": dict(self.sample_counts),
            "training_pool": dict(training_pool),
            "history_files": len(self._history_sample_paths()),
            "target_per_label": TARGET_SAMPLES_PER_LABEL,
            "min_read_samples": MIN_SAMPLES_PER_GROUP,
            "min_non_read_samples": MIN_SAMPLES_PER_GROUP,
            "can_train": self._can_train(),
            "last_prediction": self.last_prediction,
            "last_status": self.last_status,
            "recording_warmup_seconds": RECORDING_WARMUP_SECONDS,
            "model_samples": model_samples,
            "model_accuracy": model_accuracy,
            "model_threshold": self.model.threshold if self.model else 0.0,
            "samples_path": str(self.samples_path),
            "model_path": str(self.model_path),
            "state_path": str(self.state_path),
        }

    def _append_sample(self, label: str, features: Mapping[str, float]) -> None:
        needs_header = not self.samples_path.exists() or self.samples_path.stat().st_size == 0
        row = {
            "timestamp": f"{time.time():.6f}",
            "label": label,
        }
        for name in FEATURE_NAMES:
            row[name] = f"{float(features.get(name, 0.0)):.8f}"

        with self.samples_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "label", *FEATURE_NAMES])
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def _count_samples(self) -> dict[str, int]:
        return self._count_rows([self.samples_path])

    def _count_training_pool(self) -> dict[str, int]:
        return self._count_rows([self.samples_path, *self._history_sample_paths()])

    def _count_rows(self, paths: list[Path]) -> dict[str, int]:
        counts = {"read": 0, "live": 0, "glance": 0}
        for path in paths:
            if not path.exists():
                continue
            try:
                with path.open("r", newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        label = (row.get("label") or "").lower()
                        if label == "reading":
                            label = "read"
                        if label == "look":
                            label = "glance"
                        if label in counts:
                            counts[label] += 1
            except OSError:
                pass
        return counts

    def _can_train(self) -> bool:
        pool = self._count_training_pool()
        read = pool.get("read", 0)
        non_read = pool.get("live", 0) + pool.get("glance", 0)
        return read >= MIN_SAMPLES_PER_GROUP and non_read >= MIN_SAMPLES_PER_GROUP

    def _history_sample_paths(self) -> list[Path]:
        paths = sorted(self.support_dir.glob("reading_samples_trained_*.csv"))
        return paths[-MAX_HISTORY_FILES:]

    def _sample_is_clean(self, label: str, features: Mapping[str, float]) -> bool:
        duration = float(features.get("duration", 0.0))
        sample_count = float(features.get("sample_count", 0.0))
        total_motion = float(features.get("total_horizontal_motion", 0.0))
        max_step = float(features.get("max_step", 0.0))
        side_score = float(features.get("side_score", 0.0))
        center_score = float(features.get("center_score", 0.0))
        bidirectional = float(features.get("bidirectional_ratio", 0.0))
        sign_changes = float(features.get("sign_changes", 0.0))
        small_steps = float(features.get("small_step_count", 0.0))
        active_steps = float(features.get("active_step_count", 0.0))
        path_direct = float(features.get("path_direct_ratio", 0.0))
        recent_direct = float(features.get("recent_direct_ratio", 0.0))
        y_range = float(features.get("y_range", 0.0))
        vertical_motion = float(features.get("vertical_motion", 0.0))

        if duration < 0.18 or sample_count < 5 or max_step > 0.115:
            return False
        if label == "read":
            too_direct = total_motion > 0.030 and max(path_direct, recent_direct) > 0.88
            if side_score > 0.62 or y_range > 0.230 or vertical_motion > 0.052:
                return False
            return (
                total_motion > 0.007
                and active_steps >= 2
                and small_steps >= 2
                and not (too_direct and bidirectional < 0.08 and sign_changes <= 1)
            )
        if label == "live":
            reading_like = (
                center_score > 0.36
                and small_steps >= 4
                and active_steps >= 4
                and total_motion > 0.020
                and (bidirectional > 0.10 or sign_changes >= 1)
                and y_range < 0.170
            )
            return not reading_like
        if label == "glance":
            return total_motion > 0.010 and (
                side_score > 0.14
                or path_direct > 0.48
                or recent_direct > 0.52
                or max_step > 0.020
            )
        return True

    def _write_state(self) -> None:
        payload = self.state()
        tmp_path = self.state_path.with_suffix(".json.tmp")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.state_path)
            self._last_state_write_time = time.monotonic()
        except OSError as exc:
            self.logger.log(f"Could not write training state: {exc}")

    def _archive_and_reset_samples(self, result: TrainingResult) -> None:
        if not self.samples_path.exists():
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        archive_path = self.support_dir / f"reading_samples_trained_{timestamp}.csv"
        try:
            self.samples_path.replace(archive_path)
            archived_counts = self._count_rows([archive_path])
            archived_total = sum(archived_counts.values())
            self.logger.log(f"Archived {archived_total} new training samples to {archive_path}")
        except OSError as exc:
            self.logger.log(f"Could not reset training samples: {exc}")


def train_personal_reading_model(samples_path: Path, model_path: Path) -> TrainingResult:
    source_paths = _training_source_paths(samples_path)
    history_file_count = sum(1 for path in source_paths if path.name.startswith("reading_samples_trained_"))
    rows = _load_training_rows(source_paths)
    if not rows:
        return TrainingResult(False, "No training samples yet")

    x = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1000.0, neginf=-1000.0)
    x = np.clip(x, -1000.0, 1000.0)
    y = np.asarray([row["target"] for row in rows], dtype=np.float32)
    positives = int(np.sum(y == 1.0))
    negatives = int(np.sum(y == 0.0))
    if positives < MIN_SAMPLES_PER_GROUP or negatives < MIN_SAMPLES_PER_GROUP:
        return TrainingResult(
            False,
            (
                f"Need at least {MIN_SAMPLES_PER_GROUP} READ and "
                f"{MIN_SAMPLES_PER_GROUP} non-read samples "
                f"(now read={positives}, live/look={negatives})"
            ),
            samples=len(rows),
            positives=positives,
            negatives=negatives,
            history_files=history_file_count,
        )

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-5, 1.0, scale)
    x_norm = np.clip((x - mean) / scale, -8.0, 8.0)

    weights = np.zeros(x_norm.shape[1], dtype=np.float64)
    bias = 0.0
    sample_weights = np.where(y > 0.5, 0.5 / positives, 0.5 / negatives).astype(np.float64) * len(y)

    for step in range(1200):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = x_norm @ weights + bias
        logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
        probs = _sigmoid(logits)
        error = (probs - y) * sample_weights
        error = np.clip(error, -4.0, 4.0)
        lr = 0.035 * (0.997 ** step)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            grad_w = (x_norm.T @ error) / len(y) + 0.004 * weights
        grad_w = np.nan_to_num(grad_w, nan=0.0, posinf=3.0, neginf=-3.0)
        grad_w = np.clip(grad_w, -3.0, 3.0)
        weights -= lr * grad_w
        weights = np.clip(weights, -4.0, 4.0)
        bias -= lr * float(np.clip(np.mean(error), -3.0, 3.0))
        bias = float(np.clip(bias, -4.0, 4.0))

    if not np.all(np.isfinite(weights)) or not math.isfinite(bias):
        return TrainingResult(
            False,
            "Training became unstable; samples were kept. Record cleaner examples and train again.",
            samples=len(rows),
            positives=positives,
            negatives=negatives,
            history_files=history_file_count,
        )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = x_norm @ weights + bias
    logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
    probabilities = _sigmoid(logits)
    if not np.all(np.isfinite(probabilities)):
        return TrainingResult(
            False,
            "Training produced invalid probabilities; samples were kept.",
            samples=len(rows),
            positives=positives,
            negatives=negatives,
            history_files=history_file_count,
        )
    threshold, accuracy = _best_threshold(probabilities, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "trained_at": time.time(),
        "feature_names": FEATURE_NAMES,
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "bias": float(bias),
        "threshold": float(threshold),
        "samples": len(rows),
        "positives": positives,
        "negatives": negatives,
        "training_accuracy": float(accuracy),
    }
    tmp_path = model_path.with_suffix(".json.tmp")
    if model_path.exists():
        backup_path = model_path.with_name("reading_model_previous.json")
        try:
            backup_path.write_text(model_path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass
    tmp_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp_path.replace(model_path)

    return TrainingResult(
        True,
        (
            f"Personal AI trained: {len(rows)} total samples "
            f"({history_file_count} saved sessions), "
            f"accuracy {accuracy * 100:.0f}%"
        ),
        samples=len(rows),
        positives=positives,
        negatives=negatives,
        accuracy=float(accuracy),
        threshold=float(threshold),
        history_files=history_file_count,
    )


def _training_source_paths(samples_path: Path) -> list[Path]:
    support_dir = samples_path.parent
    history_paths = sorted(support_dir.glob("reading_samples_trained_*.csv"))[-MAX_HISTORY_FILES:]
    paths: list[Path] = []
    if samples_path.exists():
        paths.append(samples_path)
    paths.extend(path for path in history_paths if path != samples_path)
    return paths


def _load_training_rows(paths: list[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                label = (raw.get("label") or "").lower()
                if label == "reading":
                    label = "read"
                if label == "look":
                    label = "glance"
                if label not in VALID_LABELS:
                    continue
                try:
                    row = {name: float(raw.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}
                except ValueError:
                    continue
                if any(not math.isfinite(value) for value in row.values()):
                    continue
                identity = (str(path), raw.get("timestamp") or str(len(rows)))
                if identity in seen:
                    continue
                seen.add(identity)
                row["target"] = 1.0 if label in POSITIVE_LABELS else 0.0
                rows.append(row)
    return rows


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _best_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.35, 0.70, 36):
        pred = probabilities >= threshold
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        tpr = float(np.mean(pred[pos_mask])) if np.any(pos_mask) else 0.0
        tnr = float(np.mean(~pred[neg_mask])) if np.any(neg_mask) else 0.0
        balanced = (tpr + tnr) * 0.5
        if balanced > best_score:
            best_score = balanced
            best_threshold = float(threshold)
    return best_threshold, best_score
