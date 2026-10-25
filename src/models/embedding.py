"""
Token and Landmark Embedding Layers with Positional Encodings.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TokenEmbedding(layers.Layer):
    """
    Embeds character tokens and adds learnable 1D positional encodings.
    """

    def __init__(self, num_vocab: int = 1000, maxlen: int = 100, num_hid: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_vocab = num_vocab
        self.maxlen = maxlen
        self.num_hid = num_hid
        self.emb = layers.Embedding(input_dim=num_vocab, output_dim=num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[-1]
        x_emb = self.emb(x)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions_emb = self.pos_emb(positions)
        return x_emb + positions_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_vocab": self.num_vocab,
            "maxlen": self.maxlen,
            "num_hid": self.num_hid,
        })
        return config


class LandmarkEmbedding(layers.Layer):
    """
    Projects continuous landmark sequence coordinates using 1D convolutions
    with strides=2 to downsample temporal length and extract local spatio-temporal features.
    """

    def __init__(self, num_hid: int = 64, maxlen: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.num_hid = num_hid
        self.maxlen = maxlen
        self.conv1 = layers.Conv1D(
            num_hid, kernel_size=11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = layers.Conv1D(
            num_hid, kernel_size=11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = layers.Conv1D(
            num_hid, kernel_size=11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hid": self.num_hid,
            "maxlen": self.maxlen,
        })
        return config
