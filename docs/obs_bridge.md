# OBS Bridge

The practical no-paid-developer route for video calls is:

1. spec3 correction processes the real camera and serves the corrected image at:
   `http://127.0.0.1:29339/`
2. OBS loads that local page as a Browser Source in the `spec3 correction` scene.
3. OBS publishes the scene as `OBS Virtual Camera`.
4. Zoom, Meet, Yandex, Telegram, Discord, and similar apps select `OBS Virtual Camera`.

This avoids the Apple Developer system-extension entitlement for everyday use.
The camera name shown in meeting apps is `OBS Virtual Camera`, not `spec3
correction Camera`.

## Start

From the app, open Settings and press `Start OBS Camera`, or run:

```bash
./script/start_obs_bridge.sh
```

The first time, macOS may show OBS as waiting for approval:

```bash
systemextensionsctl list | grep -i obs
```

Approve it in:

`System Settings -> General -> Login Items & Extensions -> Camera Extensions`

After approval, reopen the meeting app and choose `OBS Virtual Camera`.
