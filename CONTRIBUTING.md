# Contributing

Thanks for helping improve `spec3 correction`.

The project is a macOS camera app, so the most important standard is user-visible reliability: camera startup, smooth preview, READ detection, eye rendering, and clear logs matter more than clever code.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Optional dlib support:

```bash
python -m pip install -e ".[dlib]"
```

## Run The App

```bash
python3 bin_single_window.py --backend mediapipe
```

## Build The macOS App

```bash
./script/build_and_run.sh --verify
```

## Checks Before A Pull Request

Run the lightweight repository checks:

```bash
./script/repository_check.sh
```

If you touched packaging, also run:

```bash
./script/build_and_run.sh --verify
```

## Pull Request Style

Good pull requests are small, testable, and user-facing:

- explain what changed;
- explain why it matters to the app experience;
- include screenshots or short recordings for UI and eye-rendering changes when possible;
- include logs for camera startup or OBS changes;
- avoid committing local settings, `.app` bundles, logs, model weights, or personal recordings.

## Privacy

Do not commit personal camera recordings, screenshots with private content, trained user models, or local databases. The app should stay local-first and respectful of camera data.
