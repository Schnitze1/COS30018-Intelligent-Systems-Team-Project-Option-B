import os
import random
import json
import argparse
from pathlib import Path
from typing import Tuple
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torchvision import datasets, transforms, utils
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Repro seeds
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "saved_models"))
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CNN] Using device: {DEVICE}")

# Hyperparams
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 12
PATIENCE = 3

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def make_extra_transform():
    def maybe_invert(img):
        import numpy as np
        a = np.array(img).astype('float32') / 255.0
        if a.mean() > 0.5:
            from PIL import Image
            return Image.fromarray((255 - np.array(img)).astype('uint8'))
        return img
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(maybe_invert),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

def get_dataloaders(batch_size=BATCH_SIZE, extra_dir: str = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    train_full = datasets.MNIST(root="~/.pytorch/mnist", train=True, download=True, transform=transform)
    test_set   = datasets.MNIST(root="~/.pytorch/mnist", train=False, download=True, transform=transform)

    # 90/10 split for validation
    val_size = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    # Optionally add extra data (ImageFolder expects class subfolders 0..9)
    if extra_dir:
        extra_dir = str(Path(extra_dir).resolve())
        if os.path.isdir(extra_dir):
            extra_ds = ImageFolder(root=extra_dir, transform=make_extra_transform())
            # combine extra_ds with train_set by making a ConcatDataset
            train_set = ConcatDataset([train_set, extra_ds])
            print(f"[CNN] Added extra data from {extra_dir}")
        else:
            print(f"[CNN] Extra dir {extra_dir} not found; skipping.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_main(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, extra_dir: str = None, out_dir: str = None):
    out_dir = out_dir or MODELS_DIR
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, extra_dir=extra_dir)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    logs = []

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, targets in train_loader:
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
        train_loss = running_loss / total
        train_acc = correct / total

        # validate
        model.eval()
        val_loss, val_acc, _, _ = 0.0, 0.0, None, None
        with torch.no_grad():
            v_running_loss, v_correct, v_total = 0.0, 0, 0
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, targets)
                v_running_loss += loss.item() * images.size(0)
                v_correct += (logits.argmax(dim=1) == targets).sum().item()
                v_total += targets.size(0)
            val_loss = v_running_loss / v_total
            val_acc = v_correct / v_total

        log = {"epoch": epoch, "train_loss": round(train_loss,5), "train_acc": round(train_acc,5), "val_loss": round(val_loss,5), "val_acc": round(val_acc,5)}
        logs.append(log)
        print(f"[CNN] Epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[CNN] Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                break

    if best_state is None:
        best_state = model.state_dict()
    # save best model into canonical models folder
    best_model_path = os.path.join(out_dir, "cnn_model_best.pth")
    torch.save(best_state, best_model_path)
    print(f"[CNN] Saved best model -> {best_model_path}")

    # save logs
    with open(os.path.join(out_dir, "cnn_logs.txt"), "w") as f:
        for l in logs:
            f.write(json.dumps(l) + "\n")

    # final test eval
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc, test_preds, test_targets = evaluate_final(model, test_loader, criterion)
    print(f"[CNN] Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # optionally save confusion matrix & sample preds grid
    try:
        plot_confusion_matrix(test_targets, test_preds, save_path=os.path.join(out_dir, "cm_mnist.png"))
        save_sample_preds_grid(model, test_loader, n=25, save_path=os.path.join(out_dir, "sample_preds.png"))
        print("[CNN] Saved: cm_mnist.png, sample_preds.png")
    except Exception:
        pass

    return best_model_path

def evaluate_final(model, loader, criterion):
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

def plot_confusion_matrix(y_true, y_pred, classes=tuple(range(10)), save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion Matrix (MNIST)')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

def save_sample_preds_grid(model, loader, n=25, save_path=None):
    model.eval()
    images_batch, targets_batch = next(iter(loader))
    images = images_batch[:n].to(DEVICE)
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
    imgs_disp = images.cpu() * MNIST_STD + MNIST_MEAN
    grid = utils.make_grid(imgs_disp, nrow=int(np.sqrt(n)), padding=2)
    npimg = grid.numpy().transpose(1, 2, 0).squeeze()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.imshow(npimg, cmap='gray')
    plt.axis('off')
    plt.title("Sample predictions (read row-wise)")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--extra-data", type=str, default=None, help="Optional: path to labeled extra data (0..9 subfolders)")
    parser.add_argument("--out-dir", type=str, default=MODELS_DIR)
    args = parser.parse_args()
    train_main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, extra_dir=args.extra_data, out_dir=args.out_dir)