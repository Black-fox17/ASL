"""
Export ASL Fingerspelling Transformer model to TensorFlow Lite for mobile/edge inference.
"""

import argparse
from pathlib import Path
from typing import Union, Optional

import tensorflow as tf

from src.config import (
    FEATURE_COLUMNS,
    START_TOKEN_IDX,
    END_TOKEN_IDX,
    MODEL_TFLITE_PATH,
)
from src.data.preprocessing import pre_process
from src.models.transformer import Transformer


class TFLiteModel(tf.Module):
    """
    Wraps the trained Transformer model with pre-processing and greedy autoregressive
    decoding logic for seamless deployment via TensorFlow Lite.
    """

    def __init__(self, model: Transformer):
        super().__init__()
        self.target_start_token_idx = START_TOKEN_IDX
        self.target_end_token_idx = END_TOKEN_IDX
        self.model = model

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, len(FEATURE_COLUMNS)], dtype=tf.float32, name="inputs")
        ]
    )
    def __call__(self, inputs: tf.Tensor):
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        # Handle empty sequences
        x = tf.cond(
            tf.shape(x)[1] == 0,
            lambda: tf.zeros((1, 1, len(FEATURE_COLUMNS))),
            lambda: tf.identity(x),
        )
        x = x[0]
        x = pre_process(x)
        x = x[None]

        # Autoregressive generation
        generated = self.model.generate(x, self.target_start_token_idx)
        generated = generated[0]

        # Crop tokens up to end token
        idx = tf.argmax(tf.cast(tf.equal(generated, self.target_end_token_idx), tf.int32))
        idx = tf.where(tf.math.less(idx, 1), tf.constant(2, dtype=tf.int64), idx)
        out_tokens = generated[1:idx]
        one_hot = tf.one_hot(out_tokens, 59)
        return {"outputs": one_hot}


def export_to_tflite(
    model: Transformer,
    output_path: Union[str, Path] = MODEL_TFLITE_PATH,
) -> bytes:
    """
    Converts a Transformer model instance into an optimized TFLite binary.
    """
    tflite_module = TFLiteModel(model)

    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_module)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model_bytes = converter.convert()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model_bytes)

    print(f"Exported TFLite model to {output_path} ({len(tflite_model_bytes) / 1024 / 1024:.2f} MB)")
    return tflite_model_bytes
