# Next Architecture: Iris Retargeting

The next major improvement should focus on realistic iris retargeting instead of stronger sliders. The current transformation path can stabilize reading motion, but it still receives an eye crop where the real iris is already moving. That is why fast reading movement, blinks, and strong downward direction can still create visible artifacts.

## Direction

Split eye rendering into three jobs:

1. Track eyelids and eye shape.
2. Estimate a stable target for READ mode.
3. Rebuild the iris and nearby sclera inside the valid eye area with temporal consistency.

This should make the result less dependent on the original moving iris and less likely to show a visible overlay.

## Requirements

- READ mode hides fast horizontal reading movement.
- LIVE mode keeps natural movement and side glances.
- A blink must never reveal a second iris during eye opening.
- Strong downward direction should remain inside the eyelids.
- Small slow motion should remain so the eyes do not look dead.
- The user should control only simple settings: strength, vertical direction, hold, smoothness, and eye life.

## Candidate Approaches

- Eye-only segmentation with iris replacement and inpainting.
- Small personalized model trained on the user's own eye examples.
- Temporal iris tracker that clamps the output to the visible eye mask.
- Existing transformation path as a fallback for low-confidence frames.
