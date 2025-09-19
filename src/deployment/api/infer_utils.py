'''
python -m code.models.train_mnist \
  --processed data/processed \
  --out models/mnist_cnn.pt \
  --epochs 3 --batch-size 128 --lr 1e-3 --seed 42
'''

# code/models/infer_utils.py
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
import numpy as np
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def load_model_and_meta(model_path: str, metadata_json: str):
    payload = torch.load(model_path, map_location="cpu")
    meta = payload.get("meta", {})
    model = SimpleCNN(num_classes=meta.get("num_classes", 10))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    # для совместимости с API — mean/std тянем из meta или из metadata.json
    if "train_mean" not in meta or "train_std" not in meta:
        meta_file = json.loads(Path(metadata_json).read_text())
        meta["train_mean"] = meta_file["train_mean"]
        meta["train_std"] = meta_file["train_std"]
    return model, meta

def _preprocess_np(gray_np: np.ndarray, mean: float, std: float):
    """
    gray_np: [H,W] в 0..255 (uint8 или float)
    → тензор [1,1,28,28] с нормализацией
    """
    if gray_np.dtype != np.float32:
        gray_np = gray_np.astype(np.float32)
    gray_np = gray_np / 255.0
    gray_np = (gray_np - mean) / (std + 1e-8)
    t = torch.from_numpy(gray_np)[None, None, :, :]  # [1,1,H,W]
    return t

def predict_pil(model: nn.Module, meta: dict, pil_img: Image.Image):
    """Возвращает (pred_class:int, probs:list[10])."""
    mean, std = meta["train_mean"], meta["train_std"]
    img = pil_img.convert("L").resize((28,28))
    arr = np.array(img, dtype=np.float32)
    x = _preprocess_np(arr, mean, std)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
        pred = int(np.argmax(probs))
    return {"pred": pred, "probs": probs}
