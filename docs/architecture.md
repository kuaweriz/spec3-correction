# spec3 correction Architecture

Developer reference for the current macOS app. For user-facing setup and usage, see the main README.

## Product Goal

spec3 correction is built around reading stabilization. The app should reduce visible reading motion in the eyes while keeping enough subtle movement that the result does not look frozen or artificial.

The current product priorities are:

- stable READ detection;
- realistic eye-region rendering;
- smooth camera preview performance;
- predictable camera selection;
- simple controls for strength, vertical direction, hold, smoothness, and eye life;
- Personal AI training that persists between app launches.

## Runtime Flow

```text
Camera input
  -> camera selection and capture guard
  -> face and eye tracking
  -> READ / LIVE / LOOK state estimation
  -> eye stabilization target
  -> eye-region model and blend
  -> single-window preview and controls
```

## Main Areas

### App Window

`displayers/dis_single_window.py` owns the combined preview and control panel. It is responsible for:

- the main preview frame;
- control panel layout;
- camera picker;
- settings, logs, training, hide, reset, and quit actions;
- keeping UI actions responsive even while frames are being processed.

### Camera Layer

The camera layer should always prefer real available cameras and avoid stale virtual devices. The picker displays the selected slot so the user can verify what is active.

Important behavior:

- refresh devices before opening the menu;
- keep the selected camera stable by slot and name;
- release the previous capture before opening a new one;
- show a clear inactive preview when the camera cannot be opened;
- avoid leaving a camera session locked after quit or hide.

### Eye Tracking

`displayers/face_predictor.py` extracts face and eye data from each frame. It provides normalized eye crops, landmarks, eye centers, and placement data for compositing.

The rest of the app depends on these values being temporally stable. Any noisy landmark jump can become a visible eye artifact, so smoothing and validity checks belong close to this layer.

### Eye Stabilization

`displayers/gaze_corrector.py` is the existing low-level model wrapper. The product layer treats it as an eye-region transformation module.

The stabilizer should:

- keep the iris inside the eye shape;
- respect eyelids and blink state;
- avoid showing a second iris when the eye opens after a blink;
- clamp extreme directions before they create obvious artifacts;
- blend only inside a reliable eye mask;
- preserve color, contrast, and lighting from the source frame.

### READ Detector

The READ detector decides when reading stabilization should be strong. It combines current motion features, recent history, and the optional Personal AI model.

Expected behavior:

- reading text should enter READ quickly;
- a single side glance should stay LIVE;
- READ should not latch onto a corner after a normal glance;
- transitions should be smooth, not delayed by multiple seconds;
- Personal AI should improve behavior without resetting after every launch.

### Personal AI Training

Training records examples for:

- `READ`: the user reads normal text in the usual screen position;
- `LIVE`: the user looks normally at the camera or central screen area;
- `LOOK`: the user makes natural side glances and casual eye movement.

Training data is temporary for the current session, while the trained model is saved for future app launches. New training should improve the saved model without making the UI unclear.

### Logs

The logs window is for readable diagnostics, not raw noise. It should make startup, camera choice, model loading, training, and errors understandable without forcing the user to inspect terminal output.

## Quality Rules

- The app should never show a black preview silently; it needs an inactive state and a useful log entry.
- Buttons must work regardless of OpenCV focus.
- Camera switching should feel instant and deterministic.
- Training should give countdown and completion signals.
- Extreme eye direction values should degrade gracefully instead of creating obvious overlays.
- UI text should remain readable in both dark and light themes.

## Future Work

The current model path is still based on eye-region transformation. For a more realistic long-term result, the next generation should separate eyelid tracking, iris placement, and eye texture synthesis more explicitly. That would make downward looks, blinks, and head motion more reliable than simply warping the current crop.
