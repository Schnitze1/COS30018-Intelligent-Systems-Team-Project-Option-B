# FNN.py
import os
import argparse
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# Use canonical models dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "saved_models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# MNIST stats (same as previous code)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def make_extra_transform():
    """
    Transform to load external crop images (ImageFolder) and make them MNIST-like:
      - convert to grayscale
      - resize to 28x28
      - convert to tensor and normalize
      - invert if background is bright (heuristic)
    """
    def maybe_invert(img):
        # PIL image input
        import numpy as np
        a = np.array(img).astype('float32') / 255.0
        if a.mean() > 0.5:
            # invert
            from PIL import Image
            return Image.fromarray((255 - np.array(img)).astype('uint8'))
        return img

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(maybe_invert),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def get_dataloaders(batch_size: int = 200, extra_dir: Optional[str] = None):
    # standard MNIST train/test loaders
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    # Optionally include extra labeled crop images from extra_dir (expects 0..9 subfolders)
    if extra_dir:
        extra_dir = str(Path(extra_dir).resolve())
        if os.path.isdir(extra_dir):
            extra_ds = ImageFolder(root=extra_dir, transform=make_extra_transform())
            # Concat extra_ds to train_set
            train_set = ConcatDataset([train_set, extra_ds])
            print(f"[FNN] Added extra dataset from {extra_dir}; sizes -> train:{len(train_set)}")
        else:
            print(f"[FNN] Extra dir {extra_dir} not found; skipping extra data.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_and_save(epochs=10, lr=0.05, batch_size=200, extra_dir: Optional[str] = None, save_name="fnn_net.pt", device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[FNN] Using device: {device}")
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, extra_dir=extra_dir)

    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//2), gamma=0.1)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"[FNN] Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)} Loss {loss.item():.6f}")
        scheduler.step()

    # Evaluate
    net.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"[FNN] Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")

    save_path = os.path.join(MODELS_DIR, save_name)
    torch.save(net.state_dict(), save_path)
    print(f"[FNN] Saved model to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--extra-data", type=str, default=None,
                        help="Optional: path to extra labeled data with subfolders 0..9")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train_and_save(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, extra_dir=args.extra_data, device=args.device)