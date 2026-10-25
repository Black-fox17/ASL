"""
Custom Keras Callbacks for monitoring predictions during training.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import START_TOKEN_IDX, END_TOKEN_IDX, PAD_TOKEN_IDX


class DisplayOutputs(keras.callbacks.Callback):
    """
    Displays sample batch predictions and ground truth phrases every N epochs.
    """

    def __init__(
        self,
        batch: Tuple[tf.Tensor, tf.Tensor],
        idx_to_token: Dict[int, str],
        target_start_token_idx: int = START_TOKEN_IDX,
        target_end_token_idx: int = END_TOKEN_IDX,
        print_every_n_epochs: int = 4,
    ):
        super().__init__()
        self.batch = batch
        self.idx_to_token = idx_to_token
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.print_every_n_epochs = print_every_n_epochs

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if (epoch + 1) % self.print_every_n_epochs != 0:
            return

        source = self.batch[0]
        target = self.batch[1]
        bs = min(tf.shape(source)[0].numpy(), 4)

        preds = self.model.generate(source[:bs], self.target_start_token_idx).numpy()

        print(f"\n--- Epoch {epoch + 1} Predictions ---")
        for i in range(bs):
            target_text = "".join([
                self.idx_to_token.get(idx, "")
                for idx in target[i].numpy()
                if idx not in [self.target_start_token_idx, self.target_end_token_idx, PAD_TOKEN_IDX]
            ])

            pred_indices = []
            for token_idx in preds[i, 1:]:
                if token_idx == self.target_end_token_idx:
                    break
                pred_indices.append(token_idx)

            pred_text = "".join([self.idx_to_token.get(idx, "") for idx in pred_indices])
            print(f"Target    : {target_text}")
            print(f"Prediction: {pred_text}")
        print("---------------------------------\n")
