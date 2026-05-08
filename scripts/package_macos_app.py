#!/usr/bin/env python3
"""Create a lightweight macOS .app launcher on the user's Desktop."""

from __future__ import annotations

import plistlib
import shutil
import stat
import subprocess
from pathlib import Path


APP_NAME = "spec3 correction"
BUNDLE_ID = "local.spec3-correction"
LEGACY_APP_NAMES = ("Gaze Correction Camera", "SpecTree")
LEGACY_STOP_APP_NAMES = ("Stop Gaze Correction Camera", "Stop SpecTree", "Stop spec3 correction")


def make_icon_png(path: Path, size: int) -> None:
    """Write a simple PPM icon image; sips converts it to PNG later."""
    cx = cy = size / 2
    radius = size * 0.38
    pixels: list[str] = []

    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > radius:
                r, g, b = 18, 22, 28
            else:
                shade = max(0.0, 1.0 - dist / radius)
                r = int(34 + 42 * shade)
                g = int(126 + 82 * shade)
                b = int(178 + 58 * shade)

            eye_w = size * 0.19
            eye_h = size * 0.095
            for eye_cx in (size * 0.38, size * 0.62):
                eye_cy = size * 0.49
                eye = ((x - eye_cx) / eye_w) ** 2 + ((y - eye_cy) / eye_h) ** 2
                if eye <= 1.0:
                    r, g, b = 238, 248, 252
                pupil = ((x - eye_cx) / (eye_w * 0.34)) ** 2 + (
                    (y - eye_cy) / (eye_h * 0.72)
                ) ** 2
                if pupil <= 1.0:
                    r, g, b = 15, 30, 38

            if abs(y - (size * 0.72 + (x - size * 0.5) * 0.10)) < size * 0.018:
                if size * 0.34 < x < size * 0.66:
                    r, g, b = 190, 238, 215

            pixels.append(f"{r} {g} {b}")

    path.write_text(f"P3\n{size} {size}\n255\n" + "\n".join(pixels) + "\n")


def create_icon(resources_dir: Path) -> None:
    iconset = resources_dir / "AppIcon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)

    icon_sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }

    for name, size in icon_sizes.items():
        ppm_path = iconset / f"{name}.ppm"
        png_path = iconset / name
        make_icon_png(ppm_path, size)
        subprocess.run(["sips", "-s", "format", "png", str(ppm_path), "--out", str(png_path)], check=True)
        ppm_path.unlink()

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(resources_dir / "AppIcon.icns")],
        check=True,
    )
    shutil.rmtree(iconset)


def write_launcher(macos_dir: Path, repo_dir: Path) -> None:
    source = repo_dir / "scripts" / "native_launcher.m"
    launcher = macos_dir / APP_NAME
    subprocess.run(
        [
            "clang",
            "-fobjc-arc",
            str(source),
            "-framework",
            "Cocoa",
            "-framework",
            "AVFoundation",
            "-o",
            str(launcher),
        ],
        check=True,
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_info_plist(contents_dir: Path) -> None:
    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": "AppIcon",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "LSArchitecturePriority": ["arm64"],
        "LSRequiresNativeExecution": True,
        "NSCameraUsageDescription": "Camera access is needed for spec3 correction to correct gaze in real time.",
        "NSDesktopFolderUsageDescription": "Desktop access is only needed when loading this local app bundle.",
    }
    with (contents_dir / "Info.plist").open("wb") as f:
        plistlib.dump(info, f)


def copy_project(repo_dir: Path, resources_dir: Path) -> None:
    project_dir = resources_dir / "project"
    if project_dir.exists():
        shutil.rmtree(project_dir)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "build",
            "dist",
        }
        return {name for name in names if name in ignored or name.endswith(".app")}

    shutil.copytree(repo_dir, project_dir, symlinks=True, ignore=ignore)


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    desktop_dir = Path.home() / "Desktop"
    app_dir = desktop_dir / f"{APP_NAME}.app"
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"

    if app_dir.exists():
        shutil.rmtree(app_dir)

    for legacy_name in LEGACY_APP_NAMES:
        legacy_dir = desktop_dir / f"{legacy_name}.app"
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)

    for legacy_name in LEGACY_STOP_APP_NAMES:
        legacy_dir = desktop_dir / f"{legacy_name}.app"
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)

    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)

    write_launcher(macos_dir, repo_dir)
    write_info_plist(contents_dir)
    copy_project(repo_dir, resources_dir)
    create_icon(resources_dir)

    print(app_dir)


if __name__ == "__main__":
    main()
