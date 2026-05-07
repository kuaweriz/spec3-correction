# Gaze Correction Camera

**Make natural eye contact on every video call.**

Gaze Correction Camera is a macOS virtual camera extension that automatically adjusts your eye direction in real-time — so you always look directly at the people you're talking to, even when you're reading notes or looking at your own video tile.

<img src="https://github.com/user-attachments/assets/66e2355a-20d7-4ac5-b711-cb1b2ff653d7" style="width: 160px; display: block;">

## Demo

<a href="https://www.youtube.com/watch?v=tOobANsNzOQ" target="_blank">
  <img src="https://img.youtube.com/vi/tOobANsNzOQ/0.jpg" style="width: 320px; display: block;">
</a>

## Download

**[⬇️ Download the macOS App](https://drive.google.com/file/d/1E47OZ66YPab1QuTbxN97hL2u3GYwyUbz/view?usp=drive_link)**

Install the `.app`, grant camera access, and you're ready to go.

## Requirements

- macOS 14 (Sonoma) or later
- A connected webcam or built-in camera
- Camera access permission granted to the app

## Getting Started

1. **Download and open** the app from the link above.
2. **Grant camera access** when prompted by macOS.
3. **Select your camera** from the dropdown in the top-right corner of the app window.
4. **Enable gaze correction** using the toggle in the settings panel.
5. **Set this app as your camera** in Zoom, Teams, FaceTime, or any video app — look for **"Gaze Correction Camera"** in the camera device list.

## Controls

| Key | Action |
| --- | ------ |
| `g` | Toggle gaze correction on / off |
| `c` | Toggle calibration panel |
| `q` | Quit |

## Calibration

Press `c` to open the calibration panel and fine-tune the correction for your setup:

| Control | Action |
| ------- | ------ |
| `↑` `↓` `←` `→` | Adjust camera position up/down/left/right |
| `+` / `-` | Adjust camera distance (closer/further) |
| `[` / `]` | Adjust focal length |
| `r` | Reset all values to default |

> **Tip:** Start with the defaults. Only calibrate if the gaze correction looks off for your specific desk setup.

---

## Advanced: Build from Source / Python CLI

If you want to run from source or use the Python CLI directly:

### Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/)
- [CMake](https://cmake.org/download/)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)

### Install

```bash
brew install pkg-config cmake
poetry install
```

### Download model weights

Download from [GitHub Releases](https://github.com/WangWilly/gaze-correction-cam/releases) and place in the correct directories:

- `lm_feat/shape_predictor_68_face_landmarks.dat`
- `weights/warping_model/flx/12/L/` and `.../R/` — checkpoint + weight files
- *(Optional)* `models/face_landmarker.task` — for MediaPipe backend

### Run

```bash
# Default (dlib backend)
poetry run python bin_single_window.py

# MediaPipe backend
poetry run python bin_single_window.py --backend mediapipe

# Specific camera
poetry run python bin_single_window.py --camera 1
```

---

## How It Works

1. Captures your webcam feed in real-time
2. Detects your face and eye positions using computer vision
3. Calculates where your eyes need to point to look at the camera
4. Applies a learned neural network warp to redirect your gaze
5. Outputs the corrected video as a virtual camera device

---

*Looking for architecture details or module documentation? See [docs/architecture.md](docs/architecture.md).*
