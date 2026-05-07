# Next Architecture: Stable Reading Gaze

The current app is based on eye-region warping. It can redirect gaze, but it
does not truly control the pupil as a separate object. When the user reads
text, the real iris moves inside the source crop. A warp model can compensate
some of that movement, but it will still leak fast reading saccades.

## Why the Current Approach Jitters

- The input eye crop already contains the moving pupil.
- Landmark and iris tracking noise changes the correction angle every frame.
- DeepWarp generates a corrected eye from the current crop, so reading motion
  can remain visible even after angle smoothing.
- Freezing or blending previous eye patches looks fake because the head and
  eyelids keep moving while the old eye texture stays behind.

## Better Direction

The next version should separate three jobs:

1. Face and eyelid tracking.
2. Stable target gaze estimation.
3. Eye synthesis/inpainting that places the iris at the stable target while
   preserving eyelids, lighting, blink state, and eye shape.

That means the stable-reading version needs a dedicated eye synthesis module,
not just more UI sliders. Good candidates:

- train or fine-tune an eye-only generator for iris relocation;
- use segmentation/inpainting around iris and sclera with temporal consistency;
- use a modern face/eye reenactment model and drive it with a stable gaze target;
- keep the old DeepWarp path only as a fallback.

## Product Requirements

- Reading text should not show fast horizontal saccades.
- Small slow motion should remain, so the gaze does not look dead.
- The user should control only simple settings: strength, eyes up/down,
  stabilizer, smoothness, and live look.
- Exit must work independently of OpenCV window focus.
