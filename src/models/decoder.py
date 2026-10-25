"""
Transformer Decoder Layer with Causal Self-Attention and Cross-Attention.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerDecoder(layers.Layer):
    """
    Transformer Decoder block containing:
    1. Causal Multi-Head Self-Attention on target tokens.
    2. Multi-Head Cross-Attention querying encoder landmark representations.
    3. Position-wise Feed-Forward Network.
    4. Layer Normalizations and Dropout with residual skips.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(
        self,
        enc_out: tf.Tensor,
        target: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        # Causal Self-Attention (prevents attending to future tokens)
        causal_mask = self.get_causal_attention_mask(target)
        attn1 = self.self_att(
            query=target,
            value=target,
            key=target,
            attention_mask=causal_mask,
        )
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(target + attn1)

        # Cross-Attention to Encoder outputs
        attn2 = self.enc_att(
            query=out1,
            value=enc_out,
            key=enc_out,
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Position-wise Feed Forward
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.layernorm3(out2 + ffn_out)

    def get_causal_attention_mask(self, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
