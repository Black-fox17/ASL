"""
ASL Fingerspelling Predictor using TensorFlow Lite.
"""

import json
import argparse
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    CHARACTER_MAP_PATH,
    MODEL_TFLITE_PATH,
    FEATURE_COLUMNS,
    SAMPLE_PARQUET_PATH,
    SAMPLE_VIDEO_PATH,
)
from src.data.mediapipe_extractor import MediaPipeExtractor


class ASLPredictor:
    """Performs fingerspelling prediction using the exported TFLite model."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_TFLITE_PATH,
        char_map_path: Union[str, Path] = CHARACTER_MAP_PATH,
    ):
        self.model_path = Path(model_path)
        self.char_map_path = Path(char_map_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Load character mapping
        with open(self.char_map_path, "r", encoding="utf-8") as f:
            self.char_map = json.load(f)
        self.rev_char_map = {idx: ch for ch, idx in self.char_map.items()}

        # Initialize TFLite Interpreter
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")

    def predict_landmarks(self, landmarks_array: np.ndarray) -> str:
        """
        Runs inference on raw (num_frames, 144) landmark array.

        Args:
            landmarks_array: numpy array of shape (T, 144) or compatible.

        Returns:
            Decoded text phrase.
        """
        inputs = landmarks_array.astype(np.float32)
        output = self.prediction_fn(inputs=inputs)
        one_hot = output["outputs"]
        token_indices = np.argmax(one_hot, axis=-1)

        phrase = "".join([self.rev_char_map.get(idx, "") for idx in token_indices])
        return phrase

    def predict_parquet(self, parquet_path: Union[str, Path]) -> str:
        """Extracts FEATURE_COLUMNS from a parquet file and runs prediction."""
        df = pd.read_parquet(parquet_path)
        # Select only required feature columns
        available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        if len(available_cols) < len(FEATURE_COLUMNS):
            raise ValueError(f"Parquet missing required landmark columns. Found {len(available_cols)}/{len(FEATURE_COLUMNS)}")

        landmarks = df[FEATURE_COLUMNS].values
        return self.predict_landmarks(landmarks)

    def predict_video(self, video_path: Union[str, Path]) -> str:
        """Extracts MediaPipe landmarks from video and predicts phrase."""
        extractor = MediaPipeExtractor()
        df = extractor.extract_from_video(video_path)
        available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        if len(available_cols) < len(FEATURE_COLUMNS):
            raise ValueError("Extracted video landmarks do not match required feature columns.")

        landmarks = df[FEATURE_COLUMNS].values
        return self.predict_landmarks(landmarks)


def main():
    parser = argparse.ArgumentParser(description="Predict ASL Fingerspelling from Parquet or Video")
    parser.add_argument("--model", type=str, default=str(MODEL_TFLITE_PATH), help="Path to TFLite model")
    parser.add_argument("--parquet", type=str, default=None, help="Path to landmark Parquet file")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()

    predictor = ASLPredictor(model_path=args.model)

    if args.parquet:
        phrase = predictor.predict_parquet(args.parquet)
        print(f"Predicted Phrase: '{phrase}'")
    elif args.video:
        phrase = predictor.predict_video(args.video)
        print(f"Predicted Phrase: '{phrase}'")
    else:
        # Default test using sample parquet if present
        if SAMPLE_PARQUET_PATH.exists():
            phrase = predictor.predict_parquet(SAMPLE_PARQUET_PATH)
            print(f"Sample Parquet Prediction: '{phrase}'")
        else:
            print("Please specify --parquet or --video to run prediction.")


if __name__ == "__main__":
    main()
