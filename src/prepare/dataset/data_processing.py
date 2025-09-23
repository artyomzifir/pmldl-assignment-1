# src/dataset/data_processing.py
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_and_process_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)


def save_data(output_dir="src/prepare/dataset/"):

    (x_train, y_train), (x_test, y_test) = load_and_process_data()

    torch.save((x_train, y_train), Path(output_dir) / "train.pt")
    torch.save((x_test, y_test), Path(output_dir) / "test.pt")

    print(f"Saved train.pt and test.pt in {output_dir}/")


if __name__ == "__main__":
    save_data()
