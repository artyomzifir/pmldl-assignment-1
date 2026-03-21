"""Model definition and training entrypoint for the MNIST classifier."""

from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    """A compact convolutional classifier for grayscale handwritten digits.

    Args:
        num_classes: Number of output classes. For MNIST this is ``10``.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor with shape ``[batch, 1, 28, 28]``.

        Returns:
            torch.Tensor: Logits for each output class.
        """
        return self.net(x)


def load_data_from_files(data_dir: str = "src/prepare/dataset/"):
    """Load serialized train and test splits from disk.

    Args:
        data_dir: Directory that contains ``train.pt`` and ``test.pt``.

    Returns:
        tuple: The train and test dataset tuples saved by ``save_data``.
    """
    train_data = torch.load(Path(data_dir) / "train.pt")
    test_data = torch.load(Path(data_dir) / "test.pt")
    return train_data, test_data


def select_device() -> str:
    """Return the best available compute backend for training or inference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_and_save(
    model_path: str = "src/prepare/model/simple_nn.pth",
    epochs: int = 5,
    batch_size: int = 64,
):
    """Train the classifier on MNIST and save its weights.

    Args:
        model_path: Path where the exported state dict will be stored.
        epochs: Number of full passes over the training split.
        batch_size: Mini-batch size for both train and test loaders.
    """
    device = select_device()
    print(f"Using device: {device}")

    (x_train, y_train), (x_test, y_test) = load_data_from_files()

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SimpleNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    train_and_save()
