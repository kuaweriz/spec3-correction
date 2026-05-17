#!/usr/bin/env python3
"""spec3 correction camera application.

Single-window reading stabilization preview with camera controls.

Usage:
    python bin_single_window.py                      # Use MediaPipe backend
    python bin_single_window.py --backend dlib       # Use optional dlib backend
    python bin_single_window.py --camera 1           # Use camera device 1

Controls:
    - 'g': Toggle stabilization on/off
    - 'c': Toggle calibration mode
    - 'q': Quit
"""

import cv2
from displayers.dis_single_window import SingleWindowGazeCorrector, DisplayConfig
from displayers.face_predictor import create_face_predictor
from utils.camera_selection import choose_camera_id


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    """
    Return the initial preview resolution.

    The live window updates itself from the first real frame. Avoid opening the
    camera here: on macOS, quickly opening and closing the same device before
    the main stream starts can leave AVFoundation returning blank frames.
    
    Args:
        camera_id: Camera device ID
        
    Returns:
        Tuple of (width, height) in pixels
    """
    print(f"Initial camera resolution: 1280x720 for camera {camera_id}")
    return (1280, 720)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="spec3 correction camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'g' - Toggle stabilization on/off
  'c' - Toggle calibration mode
  'q' - Quit the application

Examples:
  %(prog)s                         # Use default MediaPipe backend
  %(prog)s --backend mediapipe     # Use MediaPipe for face detection
  %(prog)s --camera 1              # Use camera device 1
        """,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mediapipe",
        choices=["dlib", "mediapipe"],
        help="Face detection backend (default: mediapipe)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=-1,
        help="Camera device ID (default: auto-select physical camera)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./model_managers/gaze_corrector_v1_01.yaml",
        help="Path to gaze corrector config file (default: ./model_managers/gaze_corrector_v1_01.yaml)",
    )
    args = parser.parse_args()

    camera_id = choose_camera_id(args.camera)

    # Detect camera resolution
    video_size = detect_camera_resolution(camera_id)
    
    # Calculate appropriate face detection size (half resolution)
    face_detect_size = (video_size[0] // 2, video_size[1] // 2)
    
    # Create display config with detected resolution
    display_config = DisplayConfig(
        video_size=video_size,
        face_detect_size=face_detect_size,
    )
    
    print(f"Video size: {video_size}, Face detection size: {face_detect_size}")

    # Create face predictor based on selected backend
    predictor = create_face_predictor(args.backend)

    # Create and run the corrector
    corrector = SingleWindowGazeCorrector(
        face_predictor=predictor,
        display_config=display_config,
        camera_id=camera_id,
        config_path=args.config,
    )
    corrector.run()


if __name__ == "__main__":
    main()
