#!/usr/bin/env python3
"""Create a lightweight macOS .app launcher on the user's Desktop."""

from __future__ import annotations

import plistlib
import shutil
import stat
import subprocess
import math
from pathlib import Path


APP_NAME = "spec3 correction"
BUNDLE_ID = "local.spec3-correction"
CAMERA_EXTENSION_BUNDLE_ID = "local.spec3-correction.camera-extension"
CAMERA_EXTENSION_NAME = "spec3 correction Camera Extension"
CAMERA_EXTENSION_EXECUTABLE = "Spec3CameraExtension"
APP_VERSION = "1.2.0"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _mix(left: tuple[float, float, float], right: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    amount = _clamp01(amount)
    return tuple(left[i] * (1.0 - amount) + right[i] * amount for i in range(3))


def _blend(base: tuple[float, float, float], color: tuple[float, float, float], alpha: float) -> tuple[float, float, float]:
    return _mix(base, color, alpha)


def _soft_rect(x: float, y: float, cx: float, cy: float, w: float, h: float, softness: float) -> float:
    dx = abs(x - cx) - w * 0.5
    dy = abs(y - cy) - h * 0.5
    distance = max(dx, dy)
    return 1.0 - _smoothstep(0.0, softness, distance)


def _soft_circle(x: float, y: float, cx: float, cy: float, radius: float, softness: float) -> float:
    distance = math.hypot(x - cx, y - cy) - radius
    return 1.0 - _smoothstep(0.0, softness, distance)


def _rounded_square_mask(x: float, y: float) -> tuple[float, float]:
    half_size = 0.86
    radius = 0.20
    qx = abs(x) - half_size + radius
    qy = abs(y) - half_size + radius
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    signed_distance = outside + inside - radius
    fill = 1.0 - _smoothstep(0.0, 0.026, signed_distance)
    edge = 1.0 - _smoothstep(0.0, 0.020, abs(signed_distance) - 0.010)
    return fill, edge


def _eye_alpha(x: float, y: float) -> float:
    eye_w = 0.60
    eye_h = 0.22
    horizontal = abs(x) / eye_w
    if horizontal >= 1.0:
        return 0.0
    lid = eye_h * (1.0 - horizontal**1.65) ** 0.58
    return 1.0 - _smoothstep(0.0, 0.022, abs(y + 0.015) - lid)


def make_icon_png(path: Path, size: int) -> None:
    """Write a polished PPM icon image; sips converts it to PNG later."""
    pixels: list[str] = []
    dark = (6.0, 8.0, 13.0)
    graphite = (24.0, 27.0, 36.0)
    violet = (124.0, 86.0, 255.0)
    cyan = (68.0, 218.0, 234.0)
    amber = (255.0, 149.0, 65.0)

    for y in range(size):
        v = ((y + 0.5) / size) * 2.0 - 1.0
        for x in range(size):
            u = ((x + 0.5) / size) * 2.0 - 1.0

            fill, edge = _rounded_square_mask(u, v)
            base_gradient = _mix(dark, graphite, 0.48 + 0.24 * u + 0.28 * v)
            color = _blend((4.0, 5.0, 8.0), base_gradient, fill)

            glow_violet = math.exp(-(((u + 0.58) / 0.68) ** 2 + ((v + 0.62) / 0.62) ** 2))
            glow_cyan = math.exp(-(((u - 0.58) / 0.72) ** 2 + ((v - 0.54) / 0.66) ** 2))
            glow_amber = math.exp(-(((u - 0.34) / 0.54) ** 2 + ((v + 0.52) / 0.42) ** 2))
            color = _blend(color, violet, fill * glow_violet * 0.46)
            color = _blend(color, cyan, fill * glow_cyan * 0.34)
            color = _blend(color, amber, fill * glow_amber * 0.22)

            inner_shadow = 1.0 - _rounded_square_mask(u * 1.08, v * 1.08)[0]
            color = _blend(color, (0.0, 0.0, 0.0), fill * inner_shadow * 0.35)
            color = _blend(color, (198.0, 220.0, 236.0), edge * 0.30)

            # Stabilization brackets.
            for side, bracket_color in ((-1.0, violet), (1.0, cyan)):
                vertical = _soft_rect(u, v, side * 0.56, -0.015, 0.038, 0.42, 0.020)
                top = _soft_rect(u, v, side * 0.48, -0.235, 0.18, 0.035, 0.020)
                bottom = _soft_rect(u, v, side * 0.48, 0.205, 0.18, 0.035, 0.020)
                if side < 0:
                    top *= 1.0 - _smoothstep(-0.48, -0.40, u)
                    bottom *= 1.0 - _smoothstep(-0.48, -0.40, u)
                else:
                    top *= _smoothstep(0.40, 0.48, u)
                    bottom *= _smoothstep(0.40, 0.48, u)
                color = _blend(color, bracket_color, fill * max(vertical, top, bottom) * 0.86)

            # Eye body and lens ring.
            ring_distance = abs(math.hypot(u, v + 0.02) - 0.39)
            ring = 1.0 - _smoothstep(0.010, 0.034, ring_distance)
            color = _blend(color, (170.0, 206.0, 226.0), fill * ring * 0.30)

            eye = _eye_alpha(u, v)
            color = _blend(color, (236.0, 247.0, 251.0), fill * eye * 0.98)
            lid_shadow = eye * _smoothstep(0.02, 0.22, abs(v + 0.015))
            color = _blend(color, (176.0, 196.0, 210.0), fill * lid_shadow * 0.24)

            iris_radius = math.hypot(u, v + 0.01)
            iris = _soft_circle(u, v, 0.0, -0.01, 0.155, 0.016) * eye
            iris_color = _mix((26.0, 90.0, 130.0), (91.0, 225.0, 235.0), 1.0 - _clamp01(iris_radius / 0.155))
            color = _blend(color, iris_color, fill * iris)

            pupil = _soft_circle(u, v, 0.0, -0.01, 0.070, 0.010) * eye
            color = _blend(color, (5.0, 13.0, 19.0), fill * pupil)

            highlight = _soft_circle(u, v, -0.050, -0.075, 0.034, 0.012) * eye
            color = _blend(color, (255.0, 255.0, 255.0), fill * highlight * 0.82)

            # Three small marks for spec3.
            for index, dot_color in enumerate((violet, cyan, amber)):
                dot_x = -0.13 + index * 0.13
                dot = _soft_circle(u, v, dot_x, 0.52, 0.025, 0.010)
                color = _blend(color, dot_color, fill * dot * 0.95)

            r, g, b = (max(0, min(255, int(round(channel)))) for channel in color)
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
            "-framework",
            "SystemExtensions",
            "-framework",
            "Security",
            "-o",
            str(launcher),
        ],
        check=True,
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_camera_extension_info_plist(contents_dir: Path) -> None:
    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": CAMERA_EXTENSION_NAME,
        "CFBundleExecutable": CAMERA_EXTENSION_EXECUTABLE,
        "CFBundleIdentifier": CAMERA_EXTENSION_BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": CAMERA_EXTENSION_NAME,
        "CFBundlePackageType": "SYSX",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": "12",
        "LSMinimumSystemVersion": "14.0",
        "NSSystemExtensionUsageDescription": "spec3 correction installs a virtual camera so video conferencing apps can use the corrected gaze feed.",
        "CMIOExtension": {
            "CMIOExtensionMachServiceName": CAMERA_EXTENSION_BUNDLE_ID,
        },
    }
    with (contents_dir / "Info.plist").open("wb") as f:
        plistlib.dump(info, f)


def build_camera_extension(system_extensions_dir: Path, repo_dir: Path) -> None:
    extension_dir = system_extensions_dir / f"{CAMERA_EXTENSION_BUNDLE_ID}.systemextension"
    contents_dir = extension_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    macos_dir.mkdir(parents=True, exist_ok=True)
    write_camera_extension_info_plist(contents_dir)

    source_dir = repo_dir / "virtual_camera" / "Spec3CameraExtension"
    subprocess.run(
        [
            "swiftc",
            "-target",
            "arm64-apple-macosx14.0",
            "-framework",
            "CoreMediaIO",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
            str(source_dir / "main.swift"),
            str(source_dir / "Spec3Provider.swift"),
            "-o",
            str(macos_dir / CAMERA_EXTENSION_EXECUTABLE),
        ],
        check=True,
    )


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
        "CFBundleURLTypes": [
            {
                "CFBundleURLName": BUNDLE_ID,
                "CFBundleURLSchemes": ["spec3correction"],
            }
        ],
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": "12",
        "LSMinimumSystemVersion": "14.0",
        "LSArchitecturePriority": ["arm64"],
        "LSRequiresNativeExecution": True,
        "NSCameraUsageDescription": "Camera access is needed for spec3 correction to stabilize the camera preview in real time.",
        "NSDesktopFolderUsageDescription": "Desktop access is only needed when loading this local app bundle.",
        "NSSystemExtensionUsageDescription": "spec3 correction installs a virtual camera so video conferencing apps can select the corrected gaze feed.",
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
            "user_settings.db",
        }
        return {
            name
            for name in names
            if name in ignored or name.endswith(".app") or name.endswith(".log")
        }

    shutil.copytree(repo_dir, project_dir, symlinks=True, ignore=ignore)
    sanitize_copied_project(project_dir)


def sanitize_copied_project(project_dir: Path) -> None:
    """Keep the app bundle self-contained enough for reliable local signing."""
    for link in project_dir.rglob("*"):
        if link.is_symlink():
            link.unlink()


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    desktop_dir = Path.home() / "Desktop"
    app_dir = desktop_dir / f"{APP_NAME}.app"
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    system_extensions_dir = contents_dir / "Library" / "SystemExtensions"

    if app_dir.exists():
        shutil.rmtree(app_dir)

    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)
    system_extensions_dir.mkdir(parents=True)

    write_launcher(macos_dir, repo_dir)
    build_camera_extension(system_extensions_dir, repo_dir)
    write_info_plist(contents_dir)
    copy_project(repo_dir, resources_dir)
    create_icon(resources_dir)

    print(app_dir)


if __name__ == "__main__":
    main()
