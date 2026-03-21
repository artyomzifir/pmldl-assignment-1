"""FastAPI inference service for the trained MNIST classifier."""

from fastapi import FastAPI
from pydantic import BaseModel
import torch

from src.prepare.model.model import SimpleNN

app = FastAPI()

# Select the best available backend for local inference.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = "src/prepare/model/simple_nn.pth"

# The service loads the model once at startup and reuses it for all requests.
model = SimpleNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


class InputData(BaseModel):
    """Prediction request payload.

    Attributes:
        data: A ``28 x 28`` nested list representing a single grayscale digit.
    """

    data: list


@app.get("/health")
async def health_check():
    """Return a simple readiness response for container health checks."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(input_data: InputData):
    """Run inference on a single preprocessed digit image.

    The input arrives as a 2D list. Two ``unsqueeze`` operations are needed to
    transform it into ``[batch, channel, height, width]`` before passing it to
    ``Conv2d``.
    """
    arr = (
        torch.tensor(input_data.data, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out = model(arr)
        pred = out.argmax(1).item()
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy().tolist()[0]

    return {"prediction": pred, "probs": probs}
