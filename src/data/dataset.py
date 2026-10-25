"""
TFRecord decoding, character tokenization, and tf.data pipeline creation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import tensorflow as tf

from src.config import (
    CHARACTER_MAP_PATH,
    FEATURE_COLUMNS,
    FRAME_LEN,
    MAX_PHRASE_LEN,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    PAD_TOKEN_IDX,
    START_TOKEN_IDX,
    END_TOKEN_IDX,
)
from src.data.preprocessing import pre_process


def load_character_mapping(
    mapping_path: Union[str, Path] = CHARACTER_MAP_PATH
) -> Tuple[Dict[str, int], Dict[int, str], tf.lookup.StaticHashTable]:
    """
    Loads character-to-index mapping and constructs TensorFlow lookup table
    including pad ('P'), start ('<'), and end ('>') tokens.
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        char_to_num = json.load(f)

    char_to_num[PAD_TOKEN] = PAD_TOKEN_IDX
    char_to_num[START_TOKEN] = START_TOKEN_IDX
    char_to_num[END_TOKEN] = END_TOKEN_IDX

    num_to_char = {idx: ch for ch, idx in char_to_num.items()}

    keys = list(char_to_num.keys())
    values = list(char_to_num.values())

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys=keys, values=values),
        default_value=tf.constant(-1),
        name="character_table",
    )

    return char_to_num, num_to_char, table


def decode_fn(record_bytes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parses a single serialized TFRecord example into landmarks and phrase string.
    """
    schema = {col: tf.io.VarLenFeature(dtype=tf.float32) for col in FEATURE_COLUMNS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)

    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]
    landmarks = [tf.sparse.to_dense(features[col]) for col in FEATURE_COLUMNS]
    landmarks = tf.transpose(landmarks)
    return landmarks, phrase


def build_convert_fn(char_table: tf.lookup.StaticHashTable):
    """Returns a function to preprocess landmarks and tokenize phrase."""
    def convert_fn(landmarks: tf.Tensor, phrase: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Preprocess landmarks (dominant hand standardization, normalization)
        landmarks = pre_process(landmarks)

        # Tokenize phrase: add start ('<') and end ('>') tokens
        phrase = tf.strings.bytes_split(phrase)
        phrase = char_table.lookup(phrase)
        phrase = tf.concat([
            tf.constant([START_TOKEN_IDX], dtype=tf.int32),
            phrase,
            tf.constant([END_TOKEN_IDX], dtype=tf.int32)
        ], axis=0)

        # Pad phrase to MAX_PHRASE_LEN with PAD_TOKEN_IDX
        phrase = tf.pad(
            phrase,
            paddings=[[0, MAX_PHRASE_LEN - tf.shape(phrase)[0]]],
            mode='CONSTANT',
            constant_values=PAD_TOKEN_IDX
        )
        return landmarks, phrase

    return convert_fn


def create_dataset(
    record_paths: List[str],
    batch_size: int = 64,
    is_training: bool = True,
    char_table: Optional[tf.lookup.StaticHashTable] = None,
) -> tf.data.Dataset:
    """
    Creates a prefetching, parallelized tf.data.Dataset from TFRecord files.
    """
    if char_table is None:
        _, _, char_table = load_character_mapping()

    convert_fn = build_convert_fn(char_table)

    ds = tf.data.TFRecordDataset(record_paths, num_parallel_reads=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(buffer_size=1024)

    ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(convert_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
