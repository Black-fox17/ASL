"""
Configuration settings and landmark column definitions for ASL Fingerspelling Recognition.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
CHARACTER_MAP_PATH = ASSETS_DIR / "character_to_prediction_index.json"
SAMPLE_VIDEO_PATH = SAMPLES_DIR / "00414.mp4"
SAMPLE_PARQUET_PATH = SAMPLES_DIR / "landmarks.parquet"
MODEL_TFLITE_PATH = PROJECT_ROOT / "model.tflite"

# MediaPipe Landmark Counts
LANDMARK_TYPES = ['face', 'pose', 'left_hand', 'right_hand']
LANDMARK_COUNTS = {'face': 468, 'pose': 33, 'left_hand': 21, 'right_hand': 21}
TOTAL_HOLISTIC_LANDMARKS = sum(LANDMARK_COUNTS.values())  # 543 landmarks

# Selected Pose Landmarks for Fingerspelling (Shoulders, Elbows, Wrists)
# 11: left shoulder, 12: right shoulder, 13: left elbow, 14: right elbow, 15: left wrist, 16: right wrist
POSE_INDICES = [11, 12, 13, 14, 15, 16]

# Feature Coordinate Definitions
X_COLS = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE_INDICES]
Y_COLS = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE_INDICES]
Z_COLS = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE_INDICES]

FEATURE_COLUMNS = X_COLS + Y_COLS + Z_COLS
NUM_FEATURES = len(FEATURE_COLUMNS)  # 144 features (48 landmarks * 3 coordinates)

# Coordinate Index Subsets for Vectorized Operations
X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right_hand" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left_hand" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col.split('_')[-1]) in [12, 14, 16]]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col.split('_')[-1]) in [11, 13, 15]]

# Sequence Dimensions
FRAME_LEN = 128
MAX_PHRASE_LEN = 100

# Special Sequence Tokens
PAD_TOKEN = 'P'
START_TOKEN = '<'
END_TOKEN = '>'

PAD_TOKEN_IDX = 59
START_TOKEN_IDX = 60
END_TOKEN_IDX = 61
NUM_CLASSES = 62  # 59 characters + pad + start + end

# Model Hyperparameters
NUM_HIDDEN = 64
NUM_HEADS = 2
NUM_FEED_FORWARD = 128
NUM_LAYERS_ENC = 4
NUM_LAYERS_DEC = 1
DROPOUT_RATE = 0.1
