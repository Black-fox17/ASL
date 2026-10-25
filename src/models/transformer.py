"""
End-to-End ASL Fingerspelling Transformer Model with Edit Distance Metric.
"""

from typing import List, Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    PAD_TOKEN_IDX,
    START_TOKEN_IDX,
    END_TOKEN_IDX,
    NUM_CLASSES,
    NUM_HIDDEN,
    NUM_HEADS,
    NUM_FEED_FORWARD,
    NUM_LAYERS_ENC,
    NUM_LAYERS_DEC,
    FRAME_LEN,
    MAX_PHRASE_LEN,
)
from src.models.embedding import LandmarkEmbedding, TokenEmbedding
from src.models.encoder import TransformerEncoder
from src.models.decoder import TransformerDecoder


class Transformer(keras.Model):
    """
    Sequence-to-Sequence Transformer translating frame landmarks into text phrases.

    Includes custom train_step and test_step with masked cross-entropy loss
    and normalized Levenshtein edit distance metric evaluation.
    """

    def __init__(
        self,
        num_hid: int = NUM_HIDDEN,
        num_head: int = NUM_HEADS,
        num_feed_forward: int = NUM_FEED_FORWARD,
        source_maxlen: int = FRAME_LEN,
        target_maxlen: int = MAX_PHRASE_LEN,
        num_layers_enc: int = NUM_LAYERS_ENC,
        num_layers_dec: int = NUM_LAYERS_DEC,
        num_classes: int = NUM_CLASSES,
        pad_token_idx: int = PAD_TOKEN_IDX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.Mean(name="edit_dist")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        self.pad_token_idx = pad_token_idx

        # Encoders & Decoders
        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out: tf.Tensor, target: tf.Tensor, training: bool = False) -> tf.Tensor:
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y, training=training)
        return y

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        source, target = inputs[0], inputs[1]
        x = self.encoder(source, training=training)
        y = self.decode(x, target, training=training)
        return self.classifier(y)

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        return [self.loss_metric, self.acc_metric]

    def train_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        source, target = batch[0], batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        with tf.GradientTape() as tape:
            preds = self([source, dec_input], training=True)
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, self.pad_token_idx))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Normalized Levenshtein distance metric
        edit_dist = tf.edit_distance(
            tf.sparse.from_dense(target),
            tf.sparse.from_dense(tf.cast(tf.argmax(preds, axis=-1), tf.int32)),
        )
        edit_dist = tf.reduce_mean(edit_dist)

        self.acc_metric.update_state(edit_dist)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result(), "edit_dist": self.acc_metric.result()}

    def test_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        source, target = batch[0], batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        preds = self([source, dec_input], training=False)
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, self.pad_token_idx))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)

        edit_dist = tf.edit_distance(
            tf.sparse.from_dense(target),
            tf.sparse.from_dense(tf.cast(tf.argmax(preds, axis=-1), tf.int32)),
        )
        edit_dist = tf.reduce_mean(edit_dist)

        self.acc_metric.update_state(edit_dist)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result(), "edit_dist": self.acc_metric.result()}

    def generate(self, source: tf.Tensor, target_start_token_idx: int = START_TOKEN_IDX) -> tf.Tensor:
        """
        Greedy autoregressive decoder generates predicted token indices from landmark input.
        """
        batch_size = tf.shape(source)[0]
        enc = self.encoder(source, training=False)
        dec_input = tf.ones((batch_size, 1), dtype=tf.int32) * target_start_token_idx

        for _ in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input, training=False)
            logits = self.classifier(dec_out)
            predicted_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = predicted_token[:, -1][..., tf.newaxis]
            dec_input = tf.concat([dec_input, last_logit], axis=-1)

        return dec_input
