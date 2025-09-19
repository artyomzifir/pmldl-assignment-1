# python -m src.datasets.mnist_data_ops prepare --raw data/raw --processed data/processed --val-size 10000 --seed 42


# code/datasets/mnist_data_ops.py
from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torchvision import datasets, transforms

# ---------- Константы ----------
IMG_SIZE = (28, 28)  # MNIST
NUM_CLASSES = 10

# ---------- Вспомогательные структуры ----------
@dataclass
class SplitInfo:
    name: str
    num_samples: int
    path: str

@dataclass
class DataSummary:
    train: SplitInfo
    val: SplitInfo
    test: SplitInfo
    train_mean: float
    train_std: float
    img_size: Tuple[int, int] = IMG_SIZE
    num_classes: int = NUM_CLASSES

# ---------- Функции ----------
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def download_mnist(raw_dir: Path) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Скачивает/подготавливает сырой MNIST в raw_dir, возвращает train и test датасеты."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root=str(raw_dir), train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(raw_dir), train=False, download=True, transform=transform
    )
    return train_ds, test_ds

def tensor_from_dataset(ds: datasets.MNIST) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Возвращает:
      images: uint8 тензор [N, 28, 28] со значениями 0..255
      labels: long тензор [N]
    """
    # torchvision MNIST хранит .data (uint8 0..255) и .targets
    images_u8 = ds.data.clone()  # [N,28,28], uint8
    labels = ds.targets.clone().long()  # [N]
    return images_u8, labels

def compute_mean_std(images_u8: torch.Tensor) -> Tuple[float, float]:
    """
    Считает mean/std по train в шкале 0..1 (т.е. делим на 255).
    Возвращаем скаляры float.
    """
    x = images_u8.float() / 255.0
    mean = x.mean().item()
    std = x.std(unbiased=False).item()
    return mean, std

def save_split_pt(path: Path, images_u8: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Сохраняем один сплит в .pt как словарь:
      {'images_u8': uint8 [N,28,28], 'labels': long [N]}
    """
    obj = {"images_u8": images_u8.contiguous(), "labels": labels.contiguous()}
    torch.save(obj, path)

def make_val_split(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_size: int,
    seed: int = 42,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Делит train на (new_train, val) по количеству val_size.
    Возвращает (train_images_new, train_labels_new), (val_images, val_labels).
    """
    set_seed(seed)
    n = train_images.shape[0]
    if not (0 < val_size < n):
        raise ValueError(f"val_size must be in (0, {n}), got {val_size}")

    indices = np.arange(n)
    np.random.shuffle(indices)
    val_idx = indices[:val_size]
    trn_idx = indices[val_size:]

    val_images = train_images[val_idx]
    val_labels = train_labels[val_idx]
    trn_images = train_images[trn_idx]
    trn_labels = train_labels[trn_idx]
    return (trn_images, trn_labels), (val_images, val_labels)

def prepare(
    raw_dir: Path,
    processed_dir: Path,
    val_size: int = 10_000,
    seed: int = 42,
    overwrite: bool = False,
) -> DataSummary:
    """
    Полный цикл:
      1) скачиваем MNIST в raw_dir (если нет)
      2) формируем train/val/test сплиты
      3) сохраняем их в processed_dir как .pt
      4) пишем metadata.json со сводкой
    """
    ensure_dirs(raw_dir, processed_dir)

    # 1) Скачать
    train_ds, test_ds = download_mnist(raw_dir)

    # 2) Тензоры изображений/меток из датасетов
    tr_images, tr_labels = tensor_from_dataset(train_ds)
    te_images, te_labels = tensor_from_dataset(test_ds)

    # 2.1) Деление на train/val
    (tr_images, tr_labels), (va_images, va_labels) = make_val_split(
        tr_images, tr_labels, val_size=val_size, seed=seed
    )

    # 3) Пути сплитов
    train_pt = processed_dir / "train.pt"
    val_pt = processed_dir / "val.pt"
    test_pt = processed_dir / "test.pt"

    if not overwrite and all(p.exists() for p in [train_pt, val_pt, test_pt]):
        # Уже готово — просто читаем сводку/пересчитываем
        train_mean, train_std = compute_mean_std(tr_images)
        summary = DataSummary(
            train=SplitInfo("train", tr_images.shape[0], str(train_pt)),
            val=SplitInfo("val", va_images.shape[0], str(val_pt)),
            test=SplitInfo("test", te_images.shape[0], str(test_pt)),
            train_mean=train_mean,
            train_std=train_std,
        )
        # обновим metadata.json на всякий случай
        (processed_dir / "metadata.json").write_text(json.dumps(asdict(summary), indent=2))
        return summary

    # 4) Сохранение .pt
    save_split_pt(train_pt, tr_images, tr_labels)
    save_split_pt(val_pt, va_images, va_labels)
    save_split_pt(test_pt, te_images, te_labels)

    # 5) Метаданные + статистика
    train_mean, train_std = compute_mean_std(tr_images)
    summary = DataSummary(
        train=SplitInfo("train", tr_images.shape[0], str(train_pt)),
        val=SplitInfo("val", va_images.shape[0], str(val_pt)),
        test=SplitInfo("test", te_images.shape[0], str(test_pt)),
        train_mean=train_mean,
        train_std=train_std,
    )
    (processed_dir / "metadata.json").write_text(json.dumps(asdict(summary), indent=2))
    return summary

def stats(processed_dir: Path) -> Dict[str, int]:
    """
    Быстрая сводка по размерам каждого сплита.
    """
    info = {}
    for split in ["train", "val", "test"]:
        p = processed_dir / f"{split}.pt"
        if not p.exists():
            info[split] = 0
            continue
        obj = torch.load(p, map_location="cpu")
        images = obj["images_u8"]
        info[split] = int(images.shape[0])
    # Попробуем метаданные
    meta_path = processed_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            info["mean"] = meta.get("train_mean", None)
            info["std"] = meta.get("train_std", None)
        except Exception:
            pass
    return info

def clean(processed_dir: Path) -> None:
    """
    Удаляет train.pt/val.pt/test.pt и metadata.json из processed_dir.
    """
    for name in ["train.pt", "val.pt", "test.pt", "metadata.json"]:
        p = processed_dir / name
        if p.exists():
            p.unlink()

# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("mnist_data_ops")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="download MNIST, split into train/val/test and save to processed/")
    pp.add_argument("--raw", type=Path, default=Path("data/raw"))
    pp.add_argument("--processed", type=Path, default=Path("data/processed"))
    pp.add_argument("--val-size", type=int, default=10_000, help="how many samples to put into validation split")
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--overwrite", action="store_true")

    ps = sub.add_parser("stats", help="print split sizes and (mean,std)")
    ps.add_argument("--processed", type=Path, default=Path("data/processed"))

    pc = sub.add_parser("clean", help="remove processed splits")
    pc.add_argument("--processed", type=Path, default=Path("data/processed"))
    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        summary = prepare(args.raw, args.processed, args.val_size, args.seed, args.overwrite)
        print(json.dumps(asdict(summary), indent=2))
    elif args.cmd == "stats":
        print(json.dumps(stats(args.processed), indent=2))
    elif args.cmd == "clean":
        clean(args.processed)
        print("processed data removed")

if __name__ == "__main__":
    main()
