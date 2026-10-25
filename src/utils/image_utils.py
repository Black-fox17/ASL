"""
Static image processing with MediaPipe Holistic and background segmentation.
"""

from pathlib import Path
from typing import Union, Optional, Tuple, Any

import cv2
import mediapipe as mp
import numpy as np


def process_static_image(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    bg_color: Tuple[int, int, int] = (192, 192, 192),
) -> Tuple[np.ndarray, Optional[Any]]:
    """
    Runs MediaPipe Holistic on a static image, applies background segmentation,
    and draws face, pose, and hand landmarks.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the annotated output image.
        bg_color: Background color tuple (B, G, R).

    Returns:
        Tuple of (annotated_image, holistic_results).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True,
    ) as holistic:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)

        annotated_image = image.copy()

        # Apply segmentation mask if available
        if results.segmentation_mask is not None:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = bg_color
            annotated_image = np.where(condition, annotated_image, bg_image)

        # Draw Face Mesh
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

        # Draw Pose Connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw Hand Landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), annotated_image)
            print(f"Saved annotated image to {output_path}")

        return annotated_image, results
