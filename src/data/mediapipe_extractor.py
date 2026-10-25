"""
MediaPipe Holistic Landmark Extractor.

Extracts 543 facial, pose, and hand landmarks from videos or live webcam feed
and exports them to Parquet or pandas DataFrame.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.config import (
    LANDMARK_TYPES,
    LANDMARK_COUNTS,
    TOTAL_HOLISTIC_LANDMARKS,
    SAMPLE_VIDEO_PATH,
    SAMPLE_PARQUET_PATH,
)


class MediaPipeExtractor:
    """Extracts MediaPipe Holistic landmarks from video frames."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.columns = self._create_column_names()

    @staticmethod
    def _create_column_names() -> List[str]:
        """Generates standard column names for all 543 holistic landmarks."""
        columns = []
        for lm_type, count in LANDMARK_COUNTS.items():
            for i in range(count):
                for coord in ['x', 'y', 'z']:
                    columns.append(f"{coord}_{lm_type}_{i}")
        return columns

    def extract_from_video(
        self,
        video_source: Union[str, int, Path],
        output_parquet: Optional[Union[str, Path]] = None,
        save_frames_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Processes a video file or webcam stream and extracts landmark coordinates.

        Args:
            video_source: Path to video file or webcam index (e.g. 0).
            output_parquet: Optional destination path to save extracted Parquet file.
            save_frames_dir: Optional directory to save annotated visual frames.
            max_frames: Optional maximum number of frames to process.

        Returns:
            pd.DataFrame containing frame-by-frame (x, y, z) landmark coordinates.
        """
        cap = cv2.VideoCapture(str(video_source) if isinstance(video_source, (str, Path)) else video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        if save_frames_dir:
            os.makedirs(save_frames_dir, exist_ok=True)

        landmark_data = []
        frame_idx = 1

        with self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as holistic:
            while True:
                success, image = cap.read()
                if not success:
                    break

                if max_frames and frame_idx > max_frames:
                    break

                # Prepare image for holistic tracking
                image.flags.writeable = False
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_image)

                # Initialize flat array for 543 landmarks * 3 coordinates
                frame_landmarks = np.zeros((TOTAL_HOLISTIC_LANDMARKS, 3), dtype=np.float32)

                start_idx = 0
                for lm_type in LANDMARK_TYPES:
                    landmarks = getattr(results, f"{lm_type}_landmarks", None)
                    if landmarks:
                        for idx, lm in enumerate(landmarks.landmark):
                            if idx < LANDMARK_COUNTS[lm_type]:
                                frame_landmarks[start_idx + idx] = [lm.x, lm.y, lm.z]
                    start_idx += LANDMARK_COUNTS[lm_type]

                landmark_data.append(frame_landmarks.flatten())

                # Optionally annotate and save frames
                if save_frames_dir:
                    image.flags.writeable = True
                    annotated_image = image.copy()
                    if results.face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            annotated_image,
                            results.face_landmarks,
                            self.mp_holistic.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                        )
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            annotated_image,
                            results.pose_landmarks,
                            self.mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                        )
                    if results.left_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            annotated_image,
                            results.left_hand_landmarks,
                            self.mp_holistic.HAND_CONNECTIONS,
                        )
                    if results.right_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            annotated_image,
                            results.right_hand_landmarks,
                            self.mp_holistic.HAND_CONNECTIONS,
                        )

                    frame_path = Path(save_frames_dir) / f"frame_{frame_idx}.jpg"
                    cv2.imwrite(str(frame_path), annotated_image)

                frame_idx += 1

        cap.release()

        df = pd.DataFrame(landmark_data, columns=self.columns)
        if output_parquet:
            Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(str(output_parquet), engine="pyarrow")
            print(f"Saved {len(df)} frames of landmarks to {output_parquet}")

        return df


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks from video")
    parser.add_argument("--video", type=str, default=str(SAMPLE_VIDEO_PATH), help="Path to input video")
    parser.add_argument("--output", type=str, default=str(SAMPLE_PARQUET_PATH), help="Path to output parquet")
    parser.add_argument("--frames-dir", type=str, default=None, help="Directory to save annotated frames")
    args = parser.parse_args()

    extractor = MediaPipeExtractor()
    df = extractor.extract_from_video(args.video, output_parquet=args.output, save_frames_dir=args.frames_dir)
    print("Extracted DataFrame shape:", df.shape)


if __name__ == "__main__":
    main()
