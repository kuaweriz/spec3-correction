#!/usr/bin/env python3
"""
SpecTree Camera Application

SpecTree gaze correction implementation using a single window.

Usage:
    python bin_single_window.py                      # Use dlib backend
    python bin_single_window.py --backend mediapipe  # Use mediapipe backend
    python bin_single_window.py --camera 1           # Use camera device 1

Controls:
    - 'g': Toggle gaze correction on/off
    - 'c': Toggle calibration mode
    - 'q': Quit
"""

import cv2
from displayers.dis_single_window import SingleWindowGazeCorrector, DisplayConfig
from displayers.face_predictor import create_face_predictor
from utils.camera_selection import choose_camera_id


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    """
    Detect the actual resolution of the specified camera.
    
    Args:
        camera_id: Camera device ID
        
    Returns:
        Tuple of (width, height) in pixels
    """
    cap = None
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_id}, using default resolution")
        return (640, 480)

    ret, frame = cap.read()
    if ret and frame is not None:
        height, width = frame.shape[:2]
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Detected camera resolution: {width}x{height}")
    return (width, height)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SpecTree Camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'g' - Toggle gaze correction on/off
  'c' - Toggle calibration mode
  'q' - Quit the application

Examples:
  %(prog)s                         # Use default dlib backend
  %(prog)s --backend mediapipe     # Use MediaPipe for face detection
  %(prog)s --camera 1              # Use camera device 1
        """,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dlib",
        choices=["dlib", "mediapipe"],
        help="Face detection backend (default: dlib)",
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
