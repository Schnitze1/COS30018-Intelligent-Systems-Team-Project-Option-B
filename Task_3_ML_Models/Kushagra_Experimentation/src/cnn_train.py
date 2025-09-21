"""
Task 3 — CNN (Training)
Author: Kushagra

What this script does (end-to-end):
1) Loads MNIST (28x28 grayscale digits), with standard normalization.
2) Splits training data into train/validation to track generalisation.
3) Defines a *small but effective* CNN (LeNet-ish: 2 conv blocks -> FC).
4) Trains with Adam, early-stops on validation loss, and saves the best model.
5) Evaluates on the official MNIST test set and saves:
   - Confusion matrix image (cm_mnist.png)
   - A 5x5 grid of test images with predicted labels (sample_preds.png)
   - The best model weights (model_best.pth)
   - A simple log file with epoch metrics (logs.txt)

"""

import os
import random
import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ----------------------------
# Reproducibility & Paths
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUT_DIR = "Task3_Kushagra/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {DEVICE}")

# ----------------------------
# Hyperparameters
# Keep these simple & sensible so your tutor can run it quickly.
# ----------------------------
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 12
PATIENCE = 3  # Early stopping patience (epochs without val loss improvement)


# ----------------------------
# Transforms for MNIST
# - ToTensor() converts [0,255] -> [0,1]
# - Normalize with standard MNIST stats
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# ----------------------------
# Datasets & Dataloaders
# ----------------------------
def get_dataloaders(batch_size=BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads (if needed) and prepares MNIST datasets.
    Splits the official train set into train/val (90/10).
    Returns train_loader, val_loader, test_loader.
    """
    train_full = datasets.MNIST(root="~/.pytorch/mnist", train=True, download=True, transform=transform)
    test_set   = datasets.MNIST(root="~/.pytorch/mnist", train=False, download=True, transform=transform)

    # 90/10 split for validation
    val_size = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# ----------------------------
# Model: Small, clear CNN
# - Two conv blocks (Conv -> ReLU -> MaxPool)
# - Flatten -> FC -> ReLU -> Dropout -> FC -> Logits(10)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (N, 1, 28, 28)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # -> (N, 32, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # -> (N, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (N, 64, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # -> (N, 64, 7, 7)
        )
        # Flatten -> (N, 64*7*7) = 3136
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),      # Regularisation to reduce overfitting
            nn.Linear(128, 10)     # 10 digits (0..9)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)     # keep batch dim
        x = self.classifier(x)
        return x


# ----------------------------
# Training utilities
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return avg_loss, acc, all_preds, all_targets


def plot_confusion_matrix(y_true, y_pred, classes=tuple(range(10)), save_path=os.path.join(OUT_DIR, "cm_mnist.png")):
    """
    Saves a simple confusion-matrix heatmap (no seaborn, to keep deps minimal).
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion Matrix (MNIST)')

    # Tick labels 0..9
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Minor text in cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_sample_preds_grid(model, loader, n=25, save_path=os.path.join(OUT_DIR, "sample_preds.png")):
    """
    Creates a grid of N test images with predicted labels.
    Helps examiners visually confirm the model is doing something sensible.
    """
    model.eval()
    images_batch, targets_batch = next(iter(loader))
    images = images_batch[:n].to(DEVICE)
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()

    # De-normalize for pretty display (roughly)
    imgs_disp = images.cpu() * 0.3081 + 0.1307  # invert normalization
    grid = utils.make_grid(imgs_disp, nrow=int(np.sqrt(n)), padding=2)
    npimg = grid.numpy().transpose(1, 2, 0).squeeze()

    plt.figure(figsize=(6,6))
    plt.imshow(npimg, cmap='gray')
    plt.axis('off')
    plt.title("Sample predictions (read row-wise)")
    # Overlay text labels along the top-left of each tile
    # (Keep it simple: just print them to console too)
    print("[Info] Sample predictions:", preds.tolist())
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # 1) Data
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # 2) Model / Loss / Optim
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3) Training with Early Stopping on validation loss
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
        }
        logs.append(log)
        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Check early stopping criterion
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Info] Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                break

    # 4) Save the best model
    if best_state is None:
        # In case no improvement at all (unlikely), save current
        best_state = model.state_dict()
    best_model_path = os.path.join(OUT_DIR, "model_best.pth")
    torch.save(best_state, best_model_path)
    print(f"[Info] Saved best model -> {best_model_path}")

    # Save logs
    with open(os.path.join(OUT_DIR, "logs.txt"), "w") as f:
        for l in logs:
            f.write(json.dumps(l) + "\n")

    # 5) Final evaluation on test set (using the best weights)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion)
    print(f"[Test] loss={test_loss:.4f}, acc={test_acc:.4f}")

    # 6) Save confusion matrix & sample predictions grid
    plot_confusion_matrix(test_targets, test_preds, save_path=os.path.join(OUT_DIR, "cm_mnist.png"))
    save_sample_preds_grid(model, test_loader, n=25, save_path=os.path.join(OUT_DIR, "sample_preds.png"))
    print("[Info] Saved: cm_mnist.png, sample_preds.png")

if __name__ == "__main__":
    main()
