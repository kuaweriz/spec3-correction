# Roadmap

`spec3 correction` is focused on one product promise: reading from screen text should look calmer and more natural on camera.

## Now

- Harden camera selection and startup recovery.
- Improve READ detection so normal side glances do not trigger reading mode.
- Reduce eye-rendering artifacts during blinks, head movement, and downward direction.
- Keep the single-window app clear and comfortable.
- Make OBS Bridge easy to start and diagnose.

## Next

- Better Personal AI training guidance with clearer recording sessions.
- Stronger iris retargeting for realistic downward and upward eye direction.
- More robust blink handling so the original iris does not flash during eye reopening.
- Lightweight performance profiling for smoother preview on MacBook cameras.
- Cleaner release packaging and install flow.

## Later

- Native `spec3 correction Camera` through Apple's CoreMediaIO Camera Extension path.
- More polished multilingual UI.
- Better app settings export/import.
- Optional diagnostics bundle for support.

## Design Principle

Every feature should make the app easier to trust during a real call. If a control creates confusion or visible artifacts, it should be simplified, hidden behind advanced settings, or removed.
