# American Sign Language (ASL) Fingerspelling Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-teal.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade deep learning system that translates American Sign Language (ASL) fingerspelling gestures from continuous video into text in real time. 

The system leverages **MediaPipe Holistic** for sub-millisecond landmark extraction, processes hand and upper-body kinematics through a custom **Transformer Encoder-Decoder** architecture, and deploys as a lightweight, quantized **TensorFlow Lite (`.tflite`)** model optimized for on-device mobile and edge inference.

---

## Table of Contents

- [Project Purpose & Overview](#project-purpose--overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Modular Code Breakdown (The Chunks)](#modular-code-breakdown-the-chunks)
- [Installation](#installation)
- [Quickstart & Usage](#quickstart--usage)
  - [1. Extract Landmarks from Video](#1-extract-landmarks-from-video)
  - [2. Preprocess & Generate TFRecords](#2-preprocess--generate-tfrecords)
  - [3. Train the Transformer Model](#3-train-the-transformer-model)
  - [4. Export to TensorFlow Lite](#4-export-to-tensorflow-lite)
  - [5. Run Inference](#5-run-inference)
- [Model Architecture & Theory](#model-architecture--theory)
- [Evaluation Metric](#evaluation-metric)
- [References & Acknowledgments](#references--acknowledgments)

---

## Project Purpose & Overview

Over 70 million deaf individuals worldwide use sign language as their primary mode of communication. Fingerspelling is an indispensable facet of ASL used to spell out names, addresses, technical terms, and words lacking a standardized sign.

### The Challenge
Fingerspelling in continuous video involves rapid, subtle hand articulators, high occlusion, diverse hand orientations, variable signing speeds, and camera noise. Traditional convolutional models struggle to capture the complex temporal dependencies and long-distance letter correlations.

### The Solution
This repository breaks down the solution into decoupled, modular, and maintainable components:
1. **Holistic Landmark Tracking**: Extracts 543 spatial coordinates (face, hands, upper pose) per video frame using MediaPipe Holistic.
2. **Kinematic Feature Selection**: Filters landmarks down to 48 keypoints (both hands + shoulders, elbows, and wrists = 144 features per frame).
3. **Dominant-Hand Normalization**: Automatically detects the active signing hand, mirrors left-handed signers to a canonical right-hand coordinate frame, and applies spatial standardization.
4. **Sequence-to-Sequence Transformer**: Employs 1D convolutional downsampling embeddings, a 4-layer multi-head self-attention encoder, and a causal cross-attention decoder with greedy autoregressive generation.
5. **Mobile-Ready Edge Deployment**: Wraps the full inference graph into a standalone TensorFlow Lite model (`model.tflite`) for sub-30ms latency on smartphones and IoT devices.

---

## System Architecture

```
                                  [ Input Video / Webcam Stream ]
                                                 │
                                                 ▼
                             [ MediaPipe Holistic Landmark Tracking ]
                                (543 3D Keypoints: Face, Hands, Pose)
                                                 │
                                                 ▼
                              [ Feature Slicing & Preprocessing ]
                            - Select Hands + Pose subset (144 floats)
                            - Dominant Hand Detection (NaN counts)
                            - Invariance Flip (1 - x for left hand)
                            - Zero-mean / Unit-variance Normalization
                            - Resize & Pad to 128 frames
                                                 │
                                                 ▼
                                     [ Landmark Embedding ]
                           (3x Conv1D layers with stride 2 downsampling)
                                                 │
                                                 ▼
                                   [ Transformer Encoder ]
                           (4x Multi-Head Self-Attention + FFN Layers)
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │ Memory Context                                  │
                        ▼                                                 ▼
             [ Target Text Tokens ]                              [ Causal Cross-Decoder ]
          (Shifted Right + Pos-Emb)                          (Autoregressive Generation)
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                                     [ Dense Classification ]
                                    (62 Token Classes Vocabulary)
                                                 │
                                                 ▼
                                      [ Output ASL Text Phrase ]
                                      e.g. "3 creekhouse"
```

---

## Repository Structure

```
ASL/
├── README.md                            # Comprehensive project guide and documentation
├── requirements.txt                     # Pinned project dependencies
├── .gitignore                           # Excludes large binaries, frames, and temp artifacts
├── model.tflite                         # Exported production TensorFlow Lite model (~39 MB)
│
├── assets/
│   ├── character_to_prediction_index.json  # 59-character ASL vocabulary mapping
│   ├── architecture_overview.png           # Squeezeformer / Transformer architecture diagram
│   ├── samples/
│   │   ├── 00414.mp4                    # Sample ASL fingerspelling video (phrase: '3 creekhouse')
│   │   └── landmarks.parquet            # Sample extracted 543 holistic landmarks
│   └── videos/                          # Storage for reference demo videos
│
├── src/
│   ├── __init__.py                      # Package entry point
│   ├── config.py                        # Centralized configuration, landmark indices & constants
│   │
│   ├── data/                            # [Chunk: Data Ingestion & Preprocessing]
│   │   ├── __init__.py
│   │   ├── mediapipe_extractor.py       # Holistic landmark extraction from video/webcam
│   │   ├── preprocessing.py             # Coordinate normalization, dominant hand flip & resize
│   │   └── dataset.py                   # TFRecord serialization, parsing & tf.data pipeline
│   │
│   ├── models/                          # [Chunk: Deep Learning Model Architecture]
│   │   ├── __init__.py
│   │   ├── embedding.py                 # Conv1D temporal landmark & token positional embeddings
│   │   ├── encoder.py                   # Multi-Head Attention Transformer Encoder blocks
│   │   ├── decoder.py                   # Causal & Cross-Attention Transformer Decoder blocks
│   │   └── transformer.py               # End-to-end model with Levenshtein edit distance metric
│   │
│   ├── training/                        # [Chunk: Training & Optimization]
│   │   ├── __init__.py
│   │   ├── train.py                     # Training loop, optimizer, loss & weight checkpointing
│   │   └── callbacks.py                 # DisplayOutputs validation prediction visualizer
│   │
│   ├── inference/                       # [Chunk: TFLite Export & Prediction]
│   │   ├── __init__.py
│   │   ├── tflite_export.py             # Export model to self-contained TFLite with serving signature
│   │   └── predict.py                   # Real-time / offline inference engine for Parquet & Video
│   │
│   └── utils/                           # [Chunk: Utilities & Computer Vision Helpers]
│       ├── __init__.py
│       ├── video_utils.py               # Frame-to-video compilation and video writing tools
│       └── image_utils.py               # Static image MediaPipe holistic & background segmentation
│
├── notebooks/
│   └── asl-fingerspelling-recognition-w-tensorflow.ipynb  # Interactive exploratory Kaggle notebook
│
└── benchmarks/
    └── kaggle-1st-place-solution/       # Squeezeformer 1st place reference solution
```

---

## Modular Code Breakdown (The Chunks)

Rather than keeping monolithic 1,500-line notebooks or hardcoded scripts, the codebase is modularized into dedicated, reusable Python components:

### 1. Central Configuration (`src/config.py`)
- Defines spatial landmark indices for the MediaPipe Holistic model (468 face, 33 pose, 21 left hand, 21 right hand).
- Isolate the specific subset required for fingerspelling: Left Hand (21), Right Hand (21), and Pose Keypoints (`[11, 12, 13, 14, 15, 16]` - shoulders, elbows, and wrists).
- Establishes tensor dimensionality: `144` input floats per frame, `128` fixed frame sequence length, and special sequence tokens (`<` start: 60, `>` end: 61, `P` pad: 59).

### 2. MediaPipe Landmark Extractor (`src/data/mediapipe_extractor.py`)
- Interfaces with OpenCV and `mediapipe.solutions.holistic`.
- Processes raw video or live camera streams, tracking bounding contours and 3D joint landmarks.
- Exports structured Parquet tables matching the Google ASL competition specification.

### 3. Data Preprocessing & Dominant Hand Normalization (`src/data/preprocessing.py`)
- **Dominant Hand Logic**: Computes missing coordinate ratios (`NaN` values). Whichever hand exhibits fewer NaNs is designated dominant.
- **Orientation Normalization**: When a left-handed signer is detected, horizontal coordinates are reflected via `1.0 - x`, transforming all gestures into canonical right-hand space.
- **Zero-Mean Unit-Variance Scaling**: Centers hand landmarks around zero to render predictions invariant to physical position within the camera frame.
- **Temporal Resizing**: Pads short clips with zeros and resizes longer gestures to `128` frames.

### 4. Input Pipeline & Tokenization (`src/data/dataset.py`)
- Implements `tf.data.TFRecordDataset` with asynchronous prefetching (`tf.data.AUTOTUNE`).
- Maps raw text phrases into integer tokens using `tf.lookup.StaticHashTable`.

### 5. Neural Network Embeddings (`src/models/embedding.py`)
- `LandmarkEmbedding`: Three cascaded 1D convolutional layers (`kernel_size=11, strides=2, padding='same'`) that contract 128 raw temporal frames down to 16 dense spatial representations of dimension `num_hid=64`.
- `TokenEmbedding`: Embeds output vocabulary characters and injects learnable 1D positional encodings.

### 6. Transformer Encoder (`src/models/encoder.py`)
- 4-layer stacked transformer encoder with `MultiHeadAttention(num_heads=2, key_dim=64)`.
- Position-wise feed-forward sublayers with layer normalizations and residual connections.

### 7. Transformer Decoder (`src/models/decoder.py`)
- Masked causal multi-head attention ensures predictions depend solely on prior tokens.
- Cross-attention integrates the encoded video representation with the generated character sequence.

### 8. End-to-End ASL Transformer (`src/models/transformer.py`)
- Connects encoder and decoder into a unified `keras.Model`.
- Custom `train_step` computes label-smoothed categorical cross-entropy while masking padding tokens (`pad_token_idx=59`).
- Custom `test_step` computes the competition evaluation metric: normalized Levenshtein distance (`tf.edit_distance`).
- `generate()` implements greedy autoregressive decoding for inference.

### 9. Training Engine (`src/training/train.py` & `callbacks.py`)
- Configurable training pipeline supporting learning rate warm-up and Adam optimizer.
- `DisplayOutputs` callback displays ground truth versus predicted text after every N epochs.

### 10. TFLite Export & Inference Engine (`src/inference/`)
- `TFLiteModel`: Packages preprocessing, feature scaling, model generation, and character mapping into a single self-contained TensorFlow graph.
- Exposes standard signature `serving_default(inputs: [None, 144]) -> {'outputs': [seq_len, 59]}`.
- `ASLPredictor`: High-level Python interface to run inference on videos, Parquet files, or live webcam streams.

### 11. Computer Vision Helpers (`src/utils/`)
- `video_utils.py`: Converts raw frame sequences into compressed MP4/AVI videos.
- `image_utils.py`: Runs holistic background segmentation and landmark rendering on static images.

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- (Optional) CUDA-compatible GPU for accelerated model training

### Clone & Install
```bash
# Clone the repository
git clone https://github.com/Black-fox17/ASL.git
cd ASL

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Extract Landmarks from Video
Extract 543 MediaPipe Holistic landmarks from a video file and save to Parquet:
```bash
python -m src.data.mediapipe_extractor --video assets/samples/00414.mp4 --output assets/samples/landmarks.parquet
```

### 2. Preprocess & Generate TFRecords
Convert extracted landmark Parquet files and phrase labels into optimized TFRecords:
```python
from src.data.preprocessing import save_dataset_as_tfrecords
import pandas as pd

# Load dataset metadata (containing file_id, sequence_id, phrase, path)
df = pd.read_csv("train.csv")
save_dataset_as_tfrecords(df, data_dir="train_landmarks", output_dir="preprocessed")
```

### 3. Train the Transformer Model
Train the ASL Transformer model on preprocessed TFRecords:
```bash
python -m src.training.train --records-dir preprocessed --batch-size 64 --epochs 20 --save-weights model.h5
```

### 4. Export to TensorFlow Lite
Convert the trained Keras model into a mobile-ready `.tflite` binary:
```python
from src.training.train import build_model
from src.inference.tflite_export import export_to_tflite

model = build_model()
model.load_weights("model.h5")
export_to_tflite(model, output_path="model.tflite")
```

### 5. Run Inference

#### Predict from a Parquet File:
```bash
python -m src.inference.predict --parquet assets/samples/landmarks.parquet
```

#### Predict Directly from a Video:
```bash
python -m src.inference.predict --video assets/samples/00414.mp4
```

#### Programmatic Python API:
```python
from src.inference.predict import ASLPredictor

predictor = ASLPredictor(model_path="model.tflite")

# Predict on an extracted landmark file
phrase = predictor.predict_parquet("assets/samples/landmarks.parquet")
print("Decoded Phrase:", phrase)
```

---

## Model Architecture & Theory

### Layer Tensor Flow

| Stage | Input Shape | Layer / Operation | Output Shape | Notes |
|---|---|---|---|---|
| **Input** | `(B, T, 144)` | Raw Landmark Coordinates | `(B, T, 144)` | $T = 128$ frames |
| **Embedding** | `(B, 128, 144)` | 3x Conv1D (stride=2, k=11) | `(B, 16, 64)` | Temporal downsampling to 16 steps |
| **Encoder** | `(B, 16, 64)` | 4x TransformerEncoder | `(B, 16, 64)` | 2 Attention Heads, 128 FFN Dim |
| **Target Emb** | `(B, L)` | TokenEmbedding + PosEmb | `(B, L, 64)` | $L \le 100$ characters |
| **Decoder** | `(B, L, 64)` | Causal + Cross-Attention | `(B, L, 64)` | Attends to Encoder `(B, 16, 64)` |
| **Classifier** | `(B, L, 64)` | Dense Linear Layer | `(B, L, 62)` | 62 Character Classes Logits |

---

## Evaluation Metric

The model is evaluated using the **Normalized Total Levenshtein Distance**:

$$\text{Score} = \frac{\sum_{i=1}^N D(P_i, T_i)}{\sum_{i=1}^N |T_i|}$$

where:
- $D(P_i, T_i)$ represents the minimum number of insertions, deletions, and substitutions required to transform prediction $P_i$ into ground truth target $T_i$.
- $|T_i|$ denotes the character length of the target phrase.
- A score closer to $0.0$ signifies near-perfect character sequence reconstruction.

---

## References & Acknowledgments

- **Google - American Sign Language Fingerspelling Recognition**: [Kaggle Competition](https://www.kaggle.com/competitions/asl-fingerspelling)
- **MediaPipe**: Real-time cross-platform holistic perception pipeline by Google ([MediaPipe Research](https://developers.google.com/mediapipe)).
- **Squeezeformer**: An Efficient Transformer for Automatic Speech and Gesture Recognition ([arXiv:2206.00888](https://arxiv.org/abs/2206.00888)).
- **1st Place Solution**: Competition winning solution architecture adapted in `benchmarks/kaggle-1st-place-solution/`.

---

## License

This project is licensed under the [MIT License](LICENSE).
