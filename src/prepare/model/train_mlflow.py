import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mlflow
import mlflow.pytorch

from src.prepare.model.model import SimpleNN

proc_dir = Path("data/processed")
models_dir = Path("model")
models_dir.mkdir(exist_ok=True)

lr = 1e-3
batch_size = 64
epochs = 3

(x_train, y_train) = torch.load(proc_dir / "train.pt")
(x_test, y_test) = torch.load(proc_dir / "test.pt")

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

mlflow.set_experiment("mnist_cnn")

with mlflow.start_run():
    mlflow.log_param("lr", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    accuracy = correct / total
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"Test Accuracy: {accuracy:.4f}")

    mlflow.pytorch.log_model(model, "model")

    torch.save(model.state_dict(), models_dir / "simple_nn.pth")
    print("Model saved at model/simple_nn.pth")
