# Getting Started

This guide is for the first clean launch of `spec3 correction`.

## 1. Open The App

Open `spec3 correction.app` from the Desktop or Applications folder.

macOS may ask for camera permission. Choose `Allow`. The app needs camera access to process the preview locally.

## 2. Pick A Real Camera

Use the camera picker in the right panel and select the physical webcam you want to use.

Good signs:

- the preview updates within a few seconds;
- the selected camera name matches the device you expect;
- the app does not show a virtual camera such as OBS, Casablanca, or another software camera as the active input.

## 3. Start With Safe Defaults

Recommended first setup:

| Control | Starting Point |
| --- | --- |
| `Stabilizer Power` | `80-100%` |
| `Eye Direction` | close to center, then adjust up/down |
| `Reading Hold` | `80-95%` |
| `Smoothness` | `85-95%` |
| `Eye Life` | low, then raise only if the eyes look too frozen |

## 4. Test READ Mode

Place the text where you normally read from it during a call. Read naturally for 20-30 seconds.

The app should move into `READ` while you read and return to `LIVE` when you stop. If the mode is wrong, train Personal AI with examples from the same lighting and screen position.

## 5. Use In Calls

For Zoom, Meet, Yandex, Telegram, Discord, and similar apps, use the OBS Bridge:

1. Install OBS Studio.
2. Open `Settings` in `spec3 correction`.
3. Press `Start OBS Camera`.
4. Choose `OBS Virtual Camera` in the meeting app.

More detail: [OBS Bridge](obs_bridge.md).
