# spec3 correction

**Reading stabilization for a more natural camera look.**

spec3 correction is a macOS camera app focused on one practical job: making your eyes look calmer and more natural when you read from the screen. It reduces obvious pupil movement during reading, keeps the look from becoming frozen, and gives you simple controls for tuning the result to your own camera, lighting, and face position.

## What It Does

- Stabilizes visible eye movement while reading text on screen.
- Keeps subtle natural motion so the eyes do not look locked or robotic.
- Lets you choose the input camera directly inside the app.
- Supports manual vertical eye direction tuning for your monitor setup.
- Includes Personal AI training for READ, LIVE, and LOOK behavior.
- Runs locally on your Mac.

## Requirements

- macOS 14 Sonoma or later.
- Built-in MacBook camera or an external webcam.
- Camera permission granted to the app.

## Getting Started

1. Open `spec3 correction.app`.
2. Allow camera access when macOS asks.
3. Choose the camera in the app panel.
4. Turn the app `ON`.
5. Adjust stabilization strength, vertical direction, hold, smoothness, and eye life until the look feels natural.
6. Open the training window if you want the READ detector to adapt to your behavior.

## Main Controls

| Control | Purpose |
| --- | --- |
| `ON / OFF` | Enables or disables correction. |
| `Camera` | Selects the camera source. |
| `Reading detector` | Shows whether the app is in `READ` or `LIVE` mode. |
| `Stabilization strength` | Controls how strongly reading motion is reduced. |
| `Eye direction` | Moves the reading target up or down. |
| `Reading hold` | Controls how firmly the eyes stay stable during reading. |
| `Smoothness` | Softens transitions and reduces sudden jumps. |
| `Eye life` | Adds controlled natural movement. |
| `Personal AI` | Opens the training window for custom READ/LIVE/LOOK examples. |
| `Logs` | Opens a readable log window for diagnostics. |
| `Settings` | Opens visual and language preferences. |
| `Hide` | Hides the window while the app keeps running. |
| `Quit` | Fully exits the app. |

## Personal AI Training

The training panel records short examples of three states:

| State | What to record |
| --- | --- |
| `READ` | Read text naturally from your usual screen position. |
| `LIVE` | Look normally at the camera or main point on screen. |
| `LOOK` | Make normal side glances and casual eye movement. |

After enough examples are collected, train the model from the same window. The app keeps the trained model for later launches, so you do not need to retrain every time unless you want to improve it with new samples.

## Troubleshooting

- If the preview is black, choose the correct camera from the camera menu and make sure no other app is holding the camera.
- If the camera list looks stale, quit the app fully and reopen it after connecting or removing cameras.
- If READ mode feels wrong, collect fresh Personal AI examples in the same lighting and screen position you normally use.
- If the eyes look too frozen, lower `Reading hold` or increase `Eye life`.
- If the eyes move too much while reading, increase stabilization strength and `Reading hold`.

## Project Status

This project is now developed as `spec3 correction`. The README intentionally documents only the current app and its workflow.
