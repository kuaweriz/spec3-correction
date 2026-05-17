#!/usr/bin/env python3
"""Create an OBS scene collection that publishes the spec3 MJPEG bridge."""

from __future__ import annotations

import configparser
import json
import uuid
from pathlib import Path


OBS_SUPPORT = Path.home() / "Library" / "Application Support" / "obs-studio"
PROFILE_NAME = "spec3 correction"
SCENE_FILE = "spec3 correction.json"
BRIDGE_URL = "http://127.0.0.1:29339/"


def source_base(name: str, source_id: str) -> dict:
    return {
        "prev_ver": 536936450,
        "name": name,
        "uuid": str(uuid.uuid4()),
        "id": source_id,
        "versioned_id": source_id,
        "settings": {},
        "mixers": 0,
        "sync": 0,
        "flags": 0,
        "volume": 1.0,
        "balance": 0.5,
        "enabled": True,
        "muted": False,
        "push-to-mute": False,
        "push-to-mute-delay": 0,
        "push-to-talk": False,
        "push-to-talk-delay": 0,
        "hotkeys": {},
        "deinterlace_mode": 0,
        "deinterlace_field_order": 0,
        "monitoring_type": 0,
        "private_settings": {},
    }


def write_profile() -> None:
    profile_dir = OBS_SUPPORT / "basic" / "profiles" / PROFILE_NAME
    profile_dir.mkdir(parents=True, exist_ok=True)

    basic = configparser.ConfigParser()
    basic.optionxform = str
    basic["General"] = {"Name": PROFILE_NAME}
    basic["Video"] = {
        "BaseCX": "1280",
        "BaseCY": "720",
        "OutputCX": "1280",
        "OutputCY": "720",
        "FPSCommon": "30",
        "ScaleType": "bicubic",
    }
    basic["Output"] = {"Mode": "Simple"}
    with (profile_dir / "basic.ini").open("w", encoding="utf-8") as f:
        basic.write(f, space_around_delimiters=False)


def write_scene_collection() -> None:
    scenes_dir = OBS_SUPPORT / "basic" / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    browser = source_base("spec3 corrected video", "browser_source")
    browser["settings"] = {
        "url": BRIDGE_URL,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "shutdown": False,
        "restart_when_active": True,
        "reroute_audio": False,
        "css": "html, body { margin: 0; background: #05070a; overflow: hidden; }",
    }

    scene = source_base(PROFILE_NAME, "scene")
    scene["settings"] = {
        "id_counter": 1,
        "custom_size": False,
        "items": [
            {
                "name": browser["name"],
                "source_uuid": browser["uuid"],
                "visible": True,
                "locked": True,
                "id": 1,
                "pos": {"x": 0.0, "y": 0.0},
                "rot": 0.0,
                "scale": {"x": 1.0, "y": 1.0},
                "align": 5,
                "bounds_type": 0,
                "bounds_align": 0,
                "bounds": {"x": 0.0, "y": 0.0},
                "crop_left": 0,
                "crop_top": 0,
                "crop_right": 0,
                "crop_bottom": 0,
                "scale_filter": "disable",
                "blend_method": "default",
                "blend_type": "normal",
                "show_transition": {"duration": 0},
                "hide_transition": {"duration": 0},
                "private_settings": {},
            }
        ],
    }
    scene["hotkeys"] = {"OBSBasic.SelectScene": []}
    scene["canvas_uuid"] = "6c69626f-6273-4c00-9d88-c5136d61696e"

    collection = {
        "name": PROFILE_NAME,
        "sources": [scene, browser],
        "groups": [],
        "scene_order": [{"name": PROFILE_NAME}],
        "current_scene": PROFILE_NAME,
        "current_program_scene": PROFILE_NAME,
        "canvases": [],
        "current_transition": "Fade",
        "transition_duration": 300,
        "transitions": [],
        "quick_transitions": [],
        "saved_projectors": [],
        "preview_locked": False,
        "scaling_enabled": False,
        "scaling_level": 0,
        "scaling_off_x": 0.0,
        "scaling_off_y": 0.0,
        "virtual-camera": {"type2": 3},
        "modules": {
            "scripts-tool": [],
            "output-timer": {
                "streamTimerHours": 0,
                "streamTimerMinutes": 0,
                "streamTimerSeconds": 30,
                "recordTimerHours": 0,
                "recordTimerMinutes": 0,
                "recordTimerSeconds": 30,
                "autoStartStreamTimer": False,
                "autoStartRecordTimer": False,
                "pauseRecordTimer": True,
            },
            "auto-scene-switcher": {
                "interval": 300,
                "non_matching_scene": "",
                "switch_if_not_matching": False,
                "active": False,
                "switches": [],
            },
        },
        "version": 2,
    }
    (scenes_dir / SCENE_FILE).write_text(json.dumps(collection, indent=4, ensure_ascii=False), encoding="utf-8")


def update_user_ini() -> None:
    user_ini = OBS_SUPPORT / "user.ini"
    user_ini.parent.mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(user_ini, encoding="utf-8")
    if "Basic" not in config:
        config["Basic"] = {}
    config["Basic"]["Profile"] = PROFILE_NAME
    config["Basic"]["ProfileDir"] = PROFILE_NAME
    config["Basic"]["SceneCollection"] = PROFILE_NAME
    config["Basic"]["SceneCollectionFile"] = SCENE_FILE
    with user_ini.open("w", encoding="utf-8") as f:
        config.write(f, space_around_delimiters=False)


def main() -> None:
    write_profile()
    write_scene_collection()
    update_user_ini()
    print(f"OBS scene ready: {PROFILE_NAME} -> {BRIDGE_URL}")


if __name__ == "__main__":
    main()
