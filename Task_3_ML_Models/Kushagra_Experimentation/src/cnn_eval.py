"""
Task 3 — CNN (Evaluation / Inference)
Author: Kushagra

What this script does:
1) Loads the saved best CNN model (model_best.pth).
2) Evaluates on the official MNIST test set (optional confirmation).
3) Runs inference on your Task-2 cropped digit images (segmented_*.png).
   - Resizes to 28x28, converts to grayscale if needed.
   - Handles background polarity: if background is white, it inverts so digits are white on black.
   - Applies the same normalization as training.
4) Saves a grid 'sample_preds.png' showing your cropped digits with predicted labels in the title.

Usage:
    python cnn_eval.py --model Task3_Kushagra/outputs/model_best.pth \
                       --crops_dir Task2/outputs/ \
                       --out_dir Task3_Kushagra/outputs
"""

import os
import glob
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# Model (same as in training)
# ----------------------------
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

# Same normalization as training
MNIST_NORM = transforms.Normalize((0.1307,), (0.3081,))

def load_image_as_tensor(path: str):
    """
    Loads an image from disk and returns a normalized 1x28x28 tensor suitable for the CNN.
    - Converts to grayscale
    - Resizes to 28x28
    - If background appears white, inverts to make digits white-on-black (more MNIST-like)
    - Normalizes with MNIST mean/std
    """
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((28, 28), Image.BILINEAR)
    x = np.array(img).astype(np.float32) / 255.0

    # Heuristic: if the mean is high (bright background), invert so digits are white on black.
    # MNIST digits are light strokes on dark-ish background after normalization.
    if x.mean() > 0.5:
        x = 1.0 - x

    # (H, W) -> (1, H, W)
    x = torch.from_numpy(x).unsqueeze(0)
    x = MNIST_NORM(x)  # apply same normalization
    return x  # shape: (1, 28, 28)

def build_crops_batch(crops_dir: str, limit: int = 25):
    """
    Reads up to 'limit' cropped digit images from crops_dir (e.g., Task2/outputs/segmented_*.png)
    and builds a tensor batch (N, 1, 28, 28) for inference. Also returns their file paths
    for display/annotation.
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(crops_dir, ext)))
    # filter to segmented_* first if present (nicer ordering)
    seg_files = sorted([f for f in files if os.path.basename(f).startswith("segmented_")])
    if seg_files:
        files = seg_files
    files = files[:limit]

    tensors = []
    for p in files:
        tensors.append(load_image_as_tensor(p))
    if not tensors:
        return None, []

    batch = torch.stack(tensors, dim=0)  # (N, 1, 28, 28)
    return batch, files

def save_crops_grid_with_preds(images, preds, files, save_path):
    """
    Saves a grid of input crops with predicted labels in the title.
    """
    # De-normalise for display: x * std + mean
    imgs_disp = images.clone()
    imgs_disp = imgs_disp * 0.3081 + 0.1307
    grid = utils.make_grid(imgs_disp, nrow=int(np.ceil(np.sqrt(images.size(0)))), padding=2)
    npimg = grid.numpy().transpose(1, 2, 0).squeeze()

    plt.figure(figsize=(6,6))
    plt.imshow(npimg, cmap='gray')
    plt.axis("off")
    title = "Cropped digits predictions: " + " ".join(str(p) for p in preds.tolist())
    plt.title(title)
    # Optionally also print mapping filename -> pred to console for the report
    for f, p in zip(files, preds.tolist()):
        print(f"[Pred] {os.path.basename(f)} -> {p}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../outputs/model_best.pth",
                        help="Path to trained model weights.")
    parser.add_argument("--crops_dir", type=str, default="../outputs",
                        help="Directory with cropped digit images (segmented_*.png).")
    parser.add_argument("--out_dir", type=str, default="../outputs",
                        help="Where to save outputs.")
    parser.add_argument("--eval_mnist", action="store_true",
                        help="Also evaluate on MNIST test set for confirmation.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 1) Load model
    model = SimpleCNN().to(device)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"[Info] Loaded model weights from {args.model}")

    # 2) (Optional) Evaluate on MNIST test
    if args.eval_mnist:
        from torch.utils.data import DataLoader
        test_set = datasets.MNIST(root="~/.pytorch/mnist", train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), MNIST_NORM]))
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[MNIST Test] accuracy = {correct/total:.4f}")

    # 3) Inference on your Task-2 cropped digits
    batch, files = build_crops_batch(args.crops_dir, limit=25)
    if batch is None:
        print(f"[Warn] No images found in {args.crops_dir}. "
              f"Expected files like 'segmented_00.png' from Task 2.")
        return
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu()

    save_path = os.path.join(args.out_dir, "sample_preds.png")
    save_crops_grid_with_preds(batch.cpu(), preds, files, save_path)
    print(f"[Info] Saved {save_path}")

if __name__ == "__main__":
    main()
