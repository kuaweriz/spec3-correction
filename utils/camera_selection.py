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
CONTINUITY_CAMERA_TOKENS = ("iphone", "ipad", "continuity")
_CAMERA_CACHE_TTL_SECONDS = 8.0
_camera_cache: list["CameraInfo"] | None = None
_camera_cache_time = 0.0


@dataclass(frozen=True)
class CameraInfo:
    id: int
    name: str
    unique_id: str = ""
    model_id: str = ""

    def with_name(self, name: str) -> "CameraInfo":
        return CameraInfo(self.id, name, self.unique_id, self.model_id)


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


def load_saved_camera_unique_id() -> str | None:
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    unique_id = data.get("camera_unique_id")
    return unique_id if isinstance(unique_id, str) and unique_id else None


def load_saved_camera_open_id() -> int | None:
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    camera_open_id = data.get("camera_open_id")
    return camera_open_id if isinstance(camera_open_id, int) and camera_open_id >= 0 else None


def save_camera_id(
    camera_id: int,
    camera_name: str | None = None,
    camera_unique_id: str | None = None,
    camera_open_id: int | None = None,
) -> None:
    try:
        previous = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
    data = {"camera_id": int(camera_id)}
    if camera_name:
        data["camera_name"] = camera_name
    if camera_unique_id:
        data["camera_unique_id"] = camera_unique_id
    if camera_open_id is not None and camera_open_id >= 0:
        data["camera_open_id"] = int(camera_open_id)
    elif isinstance(previous.get("camera_open_id"), int) and previous["camera_open_id"] >= 0:
        data["camera_open_id"] = int(previous["camera_open_id"])
    try:
        _settings_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _normalise_camera_name(name: str) -> str:
    return name.replace("\xa0", " ").strip()


def _camera_names_match(left: str, right: str) -> bool:
    return _normalise_camera_name(left).casefold() == _normalise_camera_name(right).casefold()


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
            unique_id = _normalise_camera_name(str(item.get("spcamera_unique-id", "")))
            model_id = _normalise_camera_name(str(item.get("spcamera_model-id", "")))
            cameras.append(CameraInfo(len(cameras), name, unique_id, model_id))
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
        unique_id = _normalise_camera_name(str(item.get("unique_id", "")))
        model_id = _normalise_camera_name(str(item.get("model_id", "")))
        if isinstance(camera_id, int) and name:
            cameras.append(CameraInfo(camera_id, name, unique_id, model_id))
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
    """Return camera names in the order OpenCV usually exposes on macOS."""
    global _camera_cache, _camera_cache_time

    now = time.monotonic()
    if (
        not force_refresh
        and _camera_cache is not None
        and now - _camera_cache_time < _CAMERA_CACHE_TTL_SECONDS
    ):
        return list(_camera_cache)

    avfoundation_cameras = _list_cameras_from_avfoundation_file()
    if avfoundation_cameras:
        cameras = avfoundation_cameras
    else:
        cameras = _list_cameras_from_json_profiler() or _list_cameras_from_text_profiler()
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
    lower = _normalise_camera_name(name).lower()
    return any(token in lower for token in VIRTUAL_CAMERA_TOKENS)


def is_virtual_camera(camera: CameraInfo) -> bool:
    haystack = " ".join(
        _normalise_camera_name(value).lower()
        for value in (camera.name, camera.model_id, camera.unique_id)
        if value
    )
    return any(token in haystack for token in VIRTUAL_CAMERA_TOKENS)


def is_continuity_camera(camera: CameraInfo) -> bool:
    haystack = " ".join(
        _normalise_camera_name(value).lower()
        for value in (camera.name, camera.model_id, camera.unique_id)
        if value
    )
    return any(token in haystack for token in CONTINUITY_CAMERA_TOKENS)


def physical_camera_options(cameras: list[CameraInfo]) -> list[CameraInfo]:
    """Return stable real cameras for user-facing selection.

    Continuity Camera can appear and disappear as the iPhone wakes/sleeps. If a
    built-in or USB camera is present, keep the iPhone out of the default list so
    the app does not jump to it while the user is trying to select MacBook camera.
    """
    physical = [camera for camera in cameras if not is_virtual_camera(camera)]
    non_continuity = [camera for camera in physical if not is_continuity_camera(camera)]
    if non_continuity:
        physical = non_continuity
    cleaned: list[CameraInfo] = []
    seen: set[str] = set()
    for camera in physical:
        identity = camera.unique_id or _normalise_camera_name(camera.name).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        cleaned.append(camera)
    return cleaned


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
    saved_camera_unique_id = load_saved_camera_unique_id()
    saved_camera = next((camera for camera in cameras if camera.id == saved_camera_id), None)
    visible_cameras = physical_camera_options(cameras)

    if saved_camera_unique_id:
        saved_by_unique_id = next(
            (
                camera
                for camera in visible_cameras
                if _normalise_camera_name(camera.unique_id)
                == _normalise_camera_name(saved_camera_unique_id)
            ),
            None,
        )
        if saved_by_unique_id is not None:
            print(f"Using saved camera by unique id {saved_by_unique_id.id}: {saved_by_unique_id.name}")
            if saved_by_unique_id.id != saved_camera_id:
                save_camera_id(
                    saved_by_unique_id.id,
                    saved_by_unique_id.name,
                    saved_by_unique_id.unique_id,
                )
            return saved_by_unique_id.id

    if saved_camera_name and not is_virtual_camera_name(saved_camera_name):
        if (
            saved_camera is not None
            and not is_virtual_camera(saved_camera)
            and _camera_names_match(saved_camera.name, saved_camera_name)
        ):
            print(f"Using saved camera slot {saved_camera.id}: {saved_camera.name}")
            if saved_camera.unique_id and saved_camera_unique_id != saved_camera.unique_id:
                save_camera_id(saved_camera.id, saved_camera.name, saved_camera.unique_id)
            return saved_camera.id

        saved_by_name = next(
            (camera for camera in visible_cameras if _camera_names_match(camera.name, saved_camera_name)),
            None,
        )
        if saved_by_name is not None:
            print(f"Using saved camera by name {saved_by_name.id}: {saved_by_name.name}")
            if saved_by_name.id != saved_camera_id:
                save_camera_id(saved_by_name.id, saved_by_name.name, saved_by_name.unique_id)
            return saved_by_name.id

        if saved_camera is not None and not is_virtual_camera(saved_camera):
            print(
                "Saved camera name was not found in the current list; "
                f"falling back to slot {saved_camera.id}: {saved_camera.name}"
            )
            if saved_camera.unique_id:
                save_camera_id(saved_camera.id, saved_camera.name, saved_camera.unique_id)
            return saved_camera.id

        if saved_camera_id is not None and saved_camera is not None and is_virtual_camera(saved_camera):
            print(
                "Ignoring stale saved camera slot "
                f"{saved_camera_id}: it was saved as {saved_camera_name}, "
                f"but macOS now reports {saved_camera.name}"
            )

    if saved_camera_name and is_virtual_camera_name(saved_camera_name):
        print(f"Ignoring saved virtual camera: {saved_camera_name}")

    if saved_camera is not None and not is_virtual_camera(saved_camera):
        print(f"Using saved camera {saved_camera.id}: {saved_camera.name}")
        if saved_camera.unique_id:
            save_camera_id(saved_camera.id, saved_camera.name, saved_camera.unique_id)
        return saved_camera_id

    if not cameras:
        print("Could not list macOS cameras; using camera 0")
        return 0

    if not visible_cameras:
        print("No physical cameras found in the macOS list; using camera 0")
        return 0

    print("Detected cameras:")
    for camera in cameras:
        print(f"  {camera.id}: {camera.name}")

    preferred_tokens = ("macbook", "facetime", "built-in", "camera macbook")
    best_camera = visible_cameras[0]
    best_score = -999
    for camera in visible_cameras:
        lower = camera.name.lower()
        score = 0
        if any(token in lower for token in preferred_tokens):
            score += 100
        if is_virtual_camera(camera):
            score -= 100
        if score > best_score:
            best_camera = camera
            best_score = score

    if best_score < 0:
        print("Only virtual/unknown cameras found; using camera 0")
        return 0

    print(f"Selected camera {best_camera.id}: {best_camera.name}")
    save_camera_id(best_camera.id, best_camera.name, best_camera.unique_id)
    return best_camera.id
