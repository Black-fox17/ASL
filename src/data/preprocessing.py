"""
Data preprocessing, dominant-hand normalization, and TFRecord creation.
"""

import os
import shutil
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.config import (
    FRAME_LEN,
    FEATURE_COLUMNS,
    RHAND_IDX,
    LHAND_IDX,
    RPOSE_IDX,
    LPOSE_IDX,
)


def resize_pad(x: tf.Tensor) -> tf.Tensor:
    """
    Standardizes sequence length to FRAME_LEN (128).
    Pads shorter sequences with zeros or resizes longer sequences.
    """
    seq_len = tf.shape(x)[0]
    if seq_len < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN - seq_len], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x


def pre_process(x: tf.Tensor) -> tf.Tensor:
    """
    Processes raw landmark coordinates:
    1. Extracts right/left hands and shoulders/elbows/wrists.
    2. Identifies dominant hand via missing value (NaN) counts.
    3. Flips coordinate signs (1 - x) if dominant hand is left to standardize pose.
    4. Normalizes hand coordinates by subtracting mean and dividing by standard deviation.
    5. Concatenates hand and pose features, standardizes length to FRAME_LEN (128).
    6. Replaces remaining NaNs with zeros and returns flattened feature array.
    """
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)

    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)

    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)

    num_hand_coords = len(LHAND_IDX) // 3
    num_pose_coords = len(LPOSE_IDX) // 3

    # Flip orientation if left hand has fewer missing values (dominant)
    if rnans > lnans:
        hand = lhand
        pose = lpose

        hand_x = hand[:, 0 * num_hand_coords : 1 * num_hand_coords]
        hand_y = hand[:, 1 * num_hand_coords : 2 * num_hand_coords]
        hand_z = hand[:, 2 * num_hand_coords : 3 * num_hand_coords]
        hand = tf.concat([1.0 - hand_x, hand_y, hand_z], axis=1)

        pose_x = pose[:, 0 * num_pose_coords : 1 * num_pose_coords]
        pose_y = pose[:, 1 * num_pose_coords : 2 * num_pose_coords]
        pose_z = pose[:, 2 * num_pose_coords : 3 * num_pose_coords]
        pose = tf.concat([1.0 - pose_x, pose_y, pose_z], axis=1)
    else:
        hand = rhand
        pose = rpose

    # Separate coordinates and stack into (time, landmarks, 3)
    hand_x = hand[:, 0 * num_hand_coords : 1 * num_hand_coords]
    hand_y = hand[:, 1 * num_hand_coords : 2 * num_hand_coords]
    hand_z = hand[:, 2 * num_hand_coords : 3 * num_hand_coords]
    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)

    # Standardize hand coordinates
    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    # Avoid zero division
    std = tf.where(tf.math.equal(std, 0.0), tf.ones_like(std), std)
    hand = (hand - mean) / std

    pose_x = pose[:, 0 * num_pose_coords : 1 * num_pose_coords]
    pose_y = pose[:, 1 * num_pose_coords : 2 * num_pose_coords]
    pose_z = pose[:, 2 * num_pose_coords : 3 * num_pose_coords]
    pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)

    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)

    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    return x


def save_dataset_as_tfrecords(
    dataset_df: pd.DataFrame,
    data_dir: Union[str, Path],
    output_dir: Union[str, Path] = "preprocessed",
):
    """
    Reads landmark parquet files referenced in dataset_df and writes serialized TFRecords.

    Args:
        dataset_df: DataFrame containing ['file_id', 'sequence_id', 'phrase', 'path']
        data_dir: Root directory containing landmark Parquet files.
        output_dir: Destination directory for .tfrecord files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir_path = Path(data_dir)

    for file_id in tqdm(dataset_df['file_id'].unique(), desc="Converting to TFRecords"):
        pq_path = data_dir_path / f"{file_id}.parquet"
        if not pq_path.exists():
            continue

        file_df = dataset_df[dataset_df['file_id'] == file_id]
        tfr_path = output_path / f"{file_id}.tfrecord"

        # Load file landmarks
        landmarks_df = pd.read_parquet(pq_path, columns=FEATURE_COLUMNS)

        with tf.io.TFRecordWriter(str(tfr_path)) as writer:
            for seq_id, phrase in zip(file_df['sequence_id'], file_df['phrase']):
                if seq_id not in landmarks_df.index:
                    continue
                features = {}
                seq_landmarks = landmarks_df.loc[seq_id]
                for col in FEATURE_COLUMNS:
                    features[col] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=seq_landmarks[col].values)
                    )
                features["phrase"] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[phrase.encode('utf-8')])
                )
                record = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(record.SerializeToString())
