"""Camera discovery and selection helpers for the macOS app."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


APP_SUPPORT_NAME = "spec3 correction"
VIRTUAL_CAMERA_TOKENS = ("virtual", "gazeat", "casablanca", "obs", "sample", "snap")
_CAMERA_CACHE_TTL_SECONDS = 8.0
_camera_cache: list["CameraInfo"] | None = None
_camera_cache_time = 0.0


@dataclass(frozen=True)
class CameraInfo:
    id: int
    name: str


def _settings_path() -> Path:
    support_dir = os.environ.get("SPEC3_SUPPORT_DIR")
    if support_dir:
        base = Path(support_dir)
    else:
        base = Path.home() / "Library" / "Application Support" / APP_SUPPORT_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base / "camera.json"


def load_saved_camera_id() -> int | None:
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    camera_id = data.get("camera_id")
    return camera_id if isinstance(camera_id, int) and camera_id >= 0 else None


def load_saved_camera_name() -> str | None:
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    camera_name = data.get("camera_name")
    return camera_name if isinstance(camera_name, str) and camera_name else None


def save_camera_id(camera_id: int, camera_name: str | None = None) -> None:
    data = {"camera_id": int(camera_id)}
    if camera_name:
        data["camera_name"] = camera_name
    try:
        _settings_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _normalise_camera_name(name: str) -> str:
    return name.replace("\xa0", " ").strip()


def _list_cameras_from_json_profiler() -> list[CameraInfo]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    cameras: list[CameraInfo] = []
    for item in data.get("SPCameraDataType", []):
        if not isinstance(item, dict):
            continue
        name = _normalise_camera_name(str(item.get("_name", "")))
        if name and name.lower() != "camera":
            cameras.append(CameraInfo(len(cameras), name))
    return cameras


def _list_cameras_from_avfoundation_file() -> list[CameraInfo]:
    camera_list_file = os.environ.get("SPEC3_CAMERA_LIST_FILE")
    if not camera_list_file:
        return []

    try:
        data = json.loads(Path(camera_list_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    cameras: list[CameraInfo] = []
    if not isinstance(data, list):
        return cameras

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        camera_id = item.get("id", index)
        name = _normalise_camera_name(str(item.get("name", "")))
        if isinstance(camera_id, int) and name:
            cameras.append(CameraInfo(camera_id, name))
    return cameras


def _list_cameras_from_text_profiler() -> list[CameraInfo]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    cameras: list[CameraInfo] = []
    for line in result.stdout.splitlines():
        if not line.startswith("    ") or line.startswith("      "):
            continue
        name = line.strip()
        if not name.endswith(":"):
            continue
        name = _normalise_camera_name(name[:-1])
        if name and name.lower() != "camera":
            cameras.append(CameraInfo(len(cameras), name))
    return cameras


def list_macos_cameras(force_refresh: bool = False) -> list[CameraInfo]:
    """Return camera names in the order OpenCV/AVFoundation usually uses."""
    global _camera_cache, _camera_cache_time

    now = time.monotonic()
    if (
        not force_refresh
        and _camera_cache is not None
        and now - _camera_cache_time < _CAMERA_CACHE_TTL_SECONDS
    ):
        return list(_camera_cache)

    cameras = (
        _list_cameras_from_avfoundation_file()
        or _list_cameras_from_json_profiler()
        or _list_cameras_from_text_profiler()
    )
    _camera_cache = cameras
    _camera_cache_time = now
    return list(cameras)


def camera_name(camera_id: int, cameras: list[CameraInfo] | None = None) -> str:
    cameras = cameras if cameras is not None else list_macos_cameras()
    for camera in cameras:
        if camera.id == camera_id:
            return camera.name
    return f"Camera {camera_id}"


def is_virtual_camera_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in VIRTUAL_CAMERA_TOKENS)


def choose_camera_id(requested_camera_id: int) -> int:
    """Choose the physical webcam by default, avoiding virtual feedback cameras."""
    env_camera = os.environ.get("SPEC3_CAMERA_ID")
    if env_camera:
        try:
            camera_id = int(env_camera)
            print(f"Using camera {camera_id} from SPEC3_CAMERA_ID")
            return camera_id
        except ValueError:
            print(f"Ignoring invalid SPEC3_CAMERA_ID={env_camera!r}")

    if requested_camera_id >= 0:
        return requested_camera_id

    cameras = list_macos_cameras()
    saved_camera_id = load_saved_camera_id()
    saved_camera_name = load_saved_camera_name()
    if saved_camera_name and not is_virtual_camera_name(saved_camera_name):
        saved_by_name = next((camera for camera in cameras if camera.name == saved_camera_name), None)
        if saved_by_name is not None:
            print(f"Using saved camera {saved_by_name.id}: {saved_by_name.name}")
            return saved_by_name.id

    saved_camera = next((camera for camera in cameras if camera.id == saved_camera_id), None)
    if saved_camera is not None and not is_virtual_camera_name(saved_camera.name):
        print(f"Using saved camera {saved_camera.id}: {saved_camera.name}")
        return saved_camera_id

    if not cameras:
        print("Could not list macOS cameras; using camera 0")
        return 0

    print("Detected cameras:")
    for camera in cameras:
        print(f"  {camera.id}: {camera.name}")

    preferred_tokens = ("macbook", "facetime", "built-in", "camera macbook")
    best_camera = cameras[0]
    best_score = -999
    for camera in cameras:
        lower = camera.name.lower()
        score = 0
        if any(token in lower for token in preferred_tokens):
            score += 100
        if is_virtual_camera_name(camera.name):
            score -= 100
        if score > best_score:
            best_camera = camera
            best_score = score

    if best_score < 0:
        print("Only virtual/unknown cameras found; using camera 0")
        return 0

    print(f"Selected camera {best_camera.id}: {best_camera.name}")
    return best_camera.id
