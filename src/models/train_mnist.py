# code/models/train_mnist.py
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ===== Dataset из .pt =====
class PTImageDataset(Dataset):
    def __init__(self, pt_path: str, mean: float, std: float):
        obj = torch.load(pt_path, map_location="cpu")
        self.images_u8 = obj["images_u8"]           # [N,28,28], uint8
        self.labels = obj["labels"].long()          # [N]
        self.mean = mean
        self.std = std

    def __len__(self): return self.images_u8.shape[0]

    def __getitem__(self, idx):
        x = self.images_u8[idx].float() / 255.0     # → [0..1]
        # (1) нормализация теми же mean/std, что посчитали на train
        x = (x - self.mean) / (self.std + 1e-8)
        x = x.unsqueeze(0)                          # [1,28,28] — канал
        y = self.labels[idx]
        return x, y

# ===== Модель =====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 7x7
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x): return self.net(x)

# ===== Утилиты =====
@dataclass
class TrainConfig:
    processed_dir: str
    out: str
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2
    device: str = "cpu"

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_meta(processed_dir: Path):
    meta = json.loads((processed_dir / "metadata.json").read_text())
    return meta["train_mean"], meta["train_std"]

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        b = x.size(0)
        total_loss += loss.item() * b
        total_acc  += (logits.argmax(1)==y).float().sum().item()
        n += b
    return total_loss/n, total_acc/n

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        b = x.size(0)
        total_loss += loss.item() * b
        total_acc  += (logits.argmax(1)==y).float().sum().item()
        n += b
    return total_loss/n, total_acc/n

def save_model(out_path: Path, model: nn.Module, meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_class": "SimpleCNN",
        "meta": meta,  # кладём mean/std, num_classes, img_size — нужно для инференса
    }
    torch.save(payload, out_path)

def get_device(name: str):
    if name == "cuda" and torch.cuda.is_available(): return "cuda"
    return "cpu"

# ====== main ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="models/mnist_cnn.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    cfg = TrainConfig(
        processed_dir=args.processed, out=args.out, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        device=get_device(args.device), num_workers=args.num_workers
    )
    print("TrainConfig:", asdict(cfg))

    set_seed(cfg.seed)

    processed_dir = Path(cfg.processed_dir)
    train_pt = processed_dir / "train.pt"
    val_pt   = processed_dir / "val.pt"
    test_pt  = processed_dir / "test.pt"
    mean, std = load_meta(processed_dir)

    train_ds = PTImageDataset(str(train_pt), mean, std)
    val_ds   = PTImageDataset(str(val_pt),   mean, std)
    test_ds  = PTImageDataset(str(test_pt),  mean, std)

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    val_ld   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_ld  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = SimpleCNN().to(cfg.device)
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, cfg.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, crit, opt, cfg.device)
        va_loss, va_acc = eval_epoch(model, val_ld, crit, cfg.device)
        print(f"Epoch {epoch:02d} | train: loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val: loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    # тестируем лучшую модель
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = eval_epoch(model, test_ld, crit, cfg.device)
    print(f"TEST  | loss {te_loss:.4f} acc {te_acc:.4f}")

    # сохраняем
    meta = {
        "train_mean": mean, "train_std": std,
        "img_size": [28,28], "num_classes": 10,
        "test_acc": te_acc
    }
    save_model(Path(cfg.out), model, meta)
    print(f"Saved model to {cfg.out}")

if __name__ == "__main__":
    main()
