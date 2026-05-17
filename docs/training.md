# Personal AI Training

Personal AI helps the READ detector adapt to your face, lighting, camera angle, and screen position.

## What To Record

| State | What You Should Do |
| --- | --- |
| `READ` | Read normal text from the same place where your script or notes usually sit. |
| `LIVE` | Look normally at the camera or a central point without reading. |
| `LOOK` | Make natural side glances and casual eye movement, then return to center. |

## Recommended Amount

The app can train with small batches, but better examples make a better detector.

| Level | Samples Per State | Use Case |
| --- | ---: | --- |
| Quick test | `20-60` | Check that the feature is working. |
| Usable | `100-150` | Better behavior for one lighting setup. |
| Strong | `300+` | Best baseline for daily use. |

## Best Recording Pattern

1. Sit exactly how you sit on calls.
2. Use the same text position you normally read from.
3. Record `READ` while actually reading, not pretending.
4. Record `LIVE` while calmly looking forward.
5. Record `LOOK` with real glances to the side, not long reading-style scanning.
6. Train the model.
7. Test for one minute in the main app.

## When To Retrain

Retrain when one of these changes:

- camera position;
- lighting;
- distance from the screen;
- text position;
- external webcam;
- glasses, cap, or strong face shadow.

## What Good Training Looks Like

Good training does not need perfect stillness. It needs honest examples.

`READ` should contain the motion that happens while you really read. `LOOK` should contain ordinary glances. `LIVE` should contain calm forward presence. The detector becomes stronger when these three states are clearly different.
