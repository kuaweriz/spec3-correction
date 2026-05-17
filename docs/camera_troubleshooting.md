# Camera Troubleshooting

Camera issues on macOS usually come from permissions, stale device slots, or another app holding the camera.

## Preview Is Black

Try this order:

1. Choose a different physical camera in the app.
2. Quit Zoom, Meet, Telegram, OBS, FaceTime, and other camera apps.
3. Press `Reset` in `spec3 correction`.
4. Fully quit and reopen `spec3 correction`.
5. Reconnect the external camera if you use one.

## Camera List Shows Old Virtual Cameras

macOS can keep virtual-camera devices visible for a while even after the source app is removed.

Useful checks:

```bash
system_profiler SPCameraDataType
```

If stale devices remain, restart the Mac. If they are still listed after restart, remove the old app's camera extension from macOS System Settings or the vendor uninstaller.

## macOS Permission Keeps Appearing

macOS permission prompts are tied to the exact app bundle identity and path. Rebuilding the app many times can make macOS treat it as a new app.

To reset permission cleanly:

```bash
tccutil reset Camera
```

Then open the final app bundle once and approve camera access.

## Another App Is Holding The Camera

Only one app may control some physical cameras at a time. Close apps that may be using the camera, then reopen `spec3 correction`.

Common holders:

- FaceTime;
- Photo Booth;
- Zoom;
- Google Meet in a browser tab;
- Telegram;
- OBS;
- another virtual-camera app.

## Logs

Use the in-app `Logs` button first. For deeper debugging, the runtime log is stored at:

```text
~/Library/Logs/spec3 correction.log
```
