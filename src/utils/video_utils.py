"""
Video processing and compilation utilities.
"""

import os
from pathlib import Path
from typing import List, Union, Optional

import cv2
import numpy as np


def frames_to_video(
    frames_or_dir: Union[List[np.ndarray], str, Path],
    output_video_path: Union[str, Path] = "result.mp4",
    fps: int = 24,
    fourcc_str: str = "mp4v",
) -> str:
    """
    Compiles a sequence of image frames or a directory of saved images into a video file.

    Args:
        frames_or_dir: Either a list of numpy BGR frames or a directory path containing image files.
        output_video_path: Destination path for the generated video.
        fps: Frames per second.
        fourcc_str: 4-character codec code (e.g. 'mp4v' for MP4, 'XVID' for AVI).

    Returns:
        Path to output video file.
    """
    frames = []

    if isinstance(frames_or_dir, (str, Path)):
        frames_dir = Path(frames_or_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        # Sort frame filenames naturally (frame_1, frame_2, ... frame_10)
        image_files = sorted(
            [f for f in frames_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
        )

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                frames.append(img)
    else:
        frames = frames_or_dir

    if not frames:
        raise ValueError("No frames provided to compile video.")

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for i, frame in enumerate(frames, start=1):
        out.write(frame)

    out.release()
    print(f"Compiled {len(frames)} frames into video: {output_path}")
    return str(output_path)
