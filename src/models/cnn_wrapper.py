# src/models/cnn_wrapper.py
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

"""


"""

# Recreate SimpleCNN exactly as in your training/eval code
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

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class CNNModel:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SimpleCNN().to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CNN weights not found: {model_path}")
        state = torch.load(model_path, map_location=self.device)
        # state could be state_dict or full model dict; assume state_dict
        if isinstance(state, dict):
            self.model.load_state_dict(state)
        else:
            # unlikely, but try load_state_dict
            self.model.load_state_dict(state)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        ])

    def _preprocess(self, crop_image):
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)
        img = crop_image.convert("L").resize((28,28), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        if arr.mean() > 0.5:
            arr = 1.0 - arr
        # convert back to PIL for torchvision transform convenience
        img2 = Image.fromarray((arr*255).astype("uint8"))
        t = self.transform(img2)  # (1,28,28)
        return t.unsqueeze(0).to(self.device)  # (1,1,28,28)

    def predict(self, crop_image):
        x = self._preprocess(crop_image)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            return int(pred.item()), float(conf.item())

    # inside class CNNModel
    def predict_from_preprocessed(self, cnn_tensor):
        """
        cnn_tensor: torch.Tensor (1,1,28,28) already normalized and on the desired device.
        Returns (label:int, confidence:float)
        """
        with torch.no_grad():
            x = cnn_tensor.to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            return int(pred.item()), float(conf.item())