from fastapi import FastAPI
from pydantic import BaseModel
import torch
from src.prepare.model.model import SimpleNN

app = FastAPI()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = "src/prepare/model/simple_nn.pth"
model = SimpleNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class InputData(BaseModel):
    data: list

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(input_data: InputData):
    arr = torch.tensor(input_data.data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(arr)
        pred = out.argmax(1).item()
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy().tolist()[0]
    return {"prediction": pred, "probs": probs}
