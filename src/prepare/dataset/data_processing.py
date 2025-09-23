import argparse
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds
import cv2
import torch
from sklearn.model_selection import train_test_split

raw_dir = Path("data/raw")
proc_dir = Path("data/processed")
raw_dir.mkdir(parents=True, exist_ok=True)
proc_dir.mkdir(parents=True, exist_ok=True)

raw_file = raw_dir / "mnist_raw.npz"

def download_raw():
    print("Downloading MNIST corrupted (canny_edges)...")
    ds = tfds.load("mnist_corrupted/canny_edges", split="train+test", as_supervised=True, try_gcs=False, download=True, data_dir=str(raw_dir))

    images, labels = [], []
    for img, label in tfds.as_numpy(ds):
        images.append(img.squeeze())
        labels.append(label)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)

    np.savez_compressed(raw_file, images=images, labels=labels)
    print(f"Saved raw dataset at {raw_file} with {images.shape[0]} samples")


def process_and_split():
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file {raw_file} not found. Run with --raw first.")

    raw = np.load(raw_file)
    images, labels = raw["images"], raw["labels"]

    processed = []
    for img in images:
        img = img * 255
        filled = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        processed.append(filled / 255.0)

    processed = np.array(processed, dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )

    x_train = torch.tensor(x_train).unsqueeze(1)  # [N,1,28,28]
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test).unsqueeze(1)
    y_test = torch.tensor(y_test)

    torch.save((x_train, y_train), proc_dir / "train.pt")
    torch.save((x_test, y_test), proc_dir / "test.pt")

    print(f"Saved processed data: train {x_train.shape}, test {x_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MNIST corrupted dataset")
    parser.add_argument("--raw", action="store_true", help="Download and save raw dataset")
    parser.add_argument("--process", action="store_true", help="Process raw dataset and split train/test")
    args = parser.parse_args()

    if args.raw:
        download_raw()
    elif args.process:
        process_and_split()
    else:
        parser.print_help()
