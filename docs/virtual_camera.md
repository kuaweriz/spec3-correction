# spec3 correction virtual camera

Goal: make the corrected spec3 video available as a normal macOS camera named
`spec3 correction Camera`, so conferencing apps can select it the same way they
select Casablanca or OBS virtual cameras.

## Architecture

1. `bin_single_window.py` keeps doing the actual camera capture and gaze
   correction.
2. `VirtualCameraFrameSink` writes the latest corrected 1280x720 BGRA frame to:

   `~/Library/Application Support/spec3 correction/virtual_camera_frame.bgra`

3. The native CoreMediaIO camera extension reads that frame file and publishes
   it as a source stream named `spec3 correction Camera`.

This split keeps the heavy Python/TensorFlow/OpenCV pipeline out of the system
extension. The extension only has to read the newest frame and emit
`CMSampleBuffer`s.

## Current state

- Python frame bridge: implemented and enabled by default.
- Swift CoreMediaIO provider source: added in `virtual_camera/Spec3CameraExtension`.
- Verified locally with `swiftc` that the extension source compiles against
  CoreMediaIO.

## Remaining packaging work

macOS virtual cameras are system extensions. To make the camera visible in Zoom,
Meet, FaceTime, or Yandex/Most, the extension must be embedded into the app,
signed with the correct system-extension entitlements, installed through
`OSSystemExtensionRequest`, and approved once by the user in macOS Settings.

Apple's current path for this is CoreMediaIO Camera Extension, not the older DAL
plug-in approach.

Local builds are signed without `com.apple.developer.system-extension.install`
by default because this is a restricted entitlement. Enabling it with the local
development certificate makes macOS reject app launch with:

`Restricted entitlements not validated / No matching profile found`

For a real installable virtual camera build, sign with an Apple Developer
provisioning profile that includes the System Extension install entitlement, then
build with:

`SPEC3_ENABLE_SYSTEM_EXTENSION=1 ./script/build_and_run.sh`
