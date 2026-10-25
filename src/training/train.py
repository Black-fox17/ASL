"""
Training script for the ASL Fingerspelling Recognition Transformer model.
"""

import argparse
from pathlib import Path
from typing import Optional, List

import tensorflow as tf
from tensorflow import keras

from src.config import (
    CHARACTER_MAP_PATH,
    NUM_HIDDEN,
    NUM_HEADS,
    NUM_FEED_FORWARD,
    NUM_LAYERS_ENC,
    NUM_LAYERS_DEC,
    NUM_CLASSES,
    FRAME_LEN,
    MAX_PHRASE_LEN,
)
from src.data.dataset import create_dataset, load_character_mapping
from src.models.transformer import Transformer
from src.training.callbacks import DisplayOutputs


def build_model(
    num_hid: int = NUM_HIDDEN,
    num_head: int = NUM_HEADS,
    num_feed_forward: int = NUM_FEED_FORWARD,
    num_layers_enc: int = NUM_LAYERS_ENC,
    num_layers_dec: int = NUM_LAYERS_DEC,
    num_classes: int = NUM_CLASSES,
) -> Transformer:
    """Builds and compiles the Transformer model."""
    model = Transformer(
        num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        source_maxlen=FRAME_LEN,
        target_maxlen=MAX_PHRASE_LEN,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=num_classes,
    )

    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    )
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss=loss_fn)
    return model


def train_model(
    train_record_paths: List[str],
    valid_record_paths: Optional[List[str]] = None,
    batch_size: int = 64,
    epochs: int = 20,
    save_weights_path: str = "model.h5",
):
    """Executes the full training loop with validation callback."""
    char_to_num, num_to_char, char_table = load_character_mapping(CHARACTER_MAP_PATH)

    train_ds = create_dataset(
        train_record_paths,
        batch_size=batch_size,
        is_training=True,
        char_table=char_table,
    )

    callbacks = []
    if valid_record_paths:
        valid_ds = create_dataset(
            valid_record_paths,
            batch_size=batch_size,
            is_training=False,
            char_table=char_table,
        )
        sample_batch = next(iter(valid_ds))
        callbacks.append(DisplayOutputs(sample_batch, num_to_char, print_every_n_epochs=2))
    else:
        valid_ds = None

    model = build_model()

    print("Beginning ASL Transformer training...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    if save_weights_path:
        model.save_weights(save_weights_path)
        print(f"Model weights successfully saved to {save_weights_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train ASL Fingerspelling Transformer")
    parser.add_argument("--records-dir", type=str, default="preprocessed", help="Path to TFRecord directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save-weights", type=str, default="model.h5", help="Path to save weights")
    args = parser.parse_args()

    record_files = list(Path(args.records_dir).glob("*.tfrecord"))
    record_paths = [str(p) for p in record_files]

    if not record_paths:
        print(f"No TFRecord files found in {args.records_dir}. Run preprocessing first.")
        return

    train_model(
        train_record_paths=record_paths,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_weights_path=args.save_weights,
    )


if __name__ == "__main__":
    main()
