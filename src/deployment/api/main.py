# code/deployment/api/main.py
from __future__ import annotations
import io, os, json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from infer_utils import load_model_and_meta, predict_pil

# --- import our model utils from repo ---
import sys

MODEL_PATH = os.getenv("MODEL_PATH", "/models/mnist_cnn.pt")
METADATA_JSON = os.getenv("METADATA_JSON", "/data/processed/metadata.json")

app = FastAPI(title="MNIST API")

# lazy load model on first request
_model_cache: Optional[torch.nn.Module] = None
_meta_cache: Optional[dict] = None

def get_model():
    global _model_cache, _meta_cache
    if _model_cache is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        if not Path(METADATA_JSON).exists():
            raise RuntimeError(f"Metadata not found at {METADATA_JSON}")
        _model_cache, _meta_cache = load_model_and_meta(MODEL_PATH, METADATA_JSON)
    return _model_cache, _meta_cache

@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "healthy"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает PNG/JPG. Мы приводим к 28x28 grayscale и прогоняем через модель.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28), Image.NEAREST)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image file")

    model, meta = get_model()
    out = predict_pil(model, meta, img)
    return out

class ArrayPayload(BaseModel):
    """Опционально: прямой ввод 28x28 массива (0/1 или 0..255)."""
    array: list

@app.post("/predict_array")
def predict_array(payload: ArrayPayload):
    arr = np.array(payload.array, dtype=np.float32)
    if arr.shape != (28, 28):
        raise HTTPException(status_code=422, detail="array must be 28x28")
    img = Image.fromarray(arr.astype(np.uint8))
    model, meta = get_model()
    out = predict_pil(model, meta, img)
    return out
