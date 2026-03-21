"""Utilities for downloading and serializing the MNIST dataset.

This module is used by the ``prepare`` Docker service. Its responsibility is
limited to dataset acquisition and conversion into PyTorch tensors that can be
reused by the training script.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
import torch


def load_and_process_data():
    """Load MNIST and convert it into normalized PyTorch tensors.

    Returns:
        tuple: ``((x_train, y_train), (x_test, y_test))`` where image tensors
        are stored in ``NCHW`` format and labels are ``torch.long`` tensors.

    Notes:
        TensorFlow provides MNIST in ``NHW`` layout with integer pixel values.
        The model in this project expects channel-first PyTorch tensors, so we
        add a singleton channel dimension with ``np.expand_dims(..., 1)``.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize raw grayscale values from [0, 255] into [0.0, 1.0].
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert from [N, H, W] to [N, C, H, W] because Conv2d expects channels.
    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)


def save_data(output_dir="src/prepare/dataset/"):
    """Persist processed MNIST tensors to ``.pt`` files.

    Args:
        output_dir: Directory where ``train.pt`` and ``test.pt`` will be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_and_process_data()
    torch.save((x_train, y_train), output_path / "train.pt")
    torch.save((x_test, y_test), output_path / "test.pt")

    print(f"Saved train.pt and test.pt in {output_path}/")


if __name__ == "__main__":
    save_data()
