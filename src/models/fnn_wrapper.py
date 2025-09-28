import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

"""


"""

# Import the exact Net class from your FNN.py if available, otherwise re-declare
try:
    from src.models.FNN import Net  # adjust import path if FNN.py lives elsewhere
except Exception:
    import torch.nn as nn
    import torch.nn.functional as F
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

# torchvision-like normalization used in FNN training
FNN_NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))

class FNNModel:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Net().to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FNN model file not found: {model_path}")
        # assuming saved with torch.save(net.state_dict(), path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # transform pipeline similar to your load_image
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1) if False else transforms.Lambda(lambda x: x),  # we will handle PIL to np
            transforms.ToTensor(),  # expects PIL Image or ndarray
            FNN_NORMALIZE
        ])

    def _preprocess(self, crop_image):
        """
        crop_image: numpy array (H,W) or (H,W,3) in uint8 or PIL Image
        returns: tensor shape (1, 784) on device
        """
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)
        # Convert to tensor then normalize
        t = transforms.ToTensor()(crop_image)  # (C,H,W) with C=1 or 3
        # if 3 channels, convert to grayscale by taking mean
        if t.size(0) == 3:
            t = t.mean(dim=0, keepdim=True)
        t = FNN_NORMALIZE(t)
        t = t.view(1, -1).to(self.device)  # (1,784)
        return t

    def predict(self, crop_image):
        """
        Returns (label:int, confidence:float)
        """
        x = self._preprocess(crop_image)
        with torch.no_grad():
            out = self.model(x)  # log_softmax
            probs = torch.exp(out)  # convert log-probs to probs
            conf, pred = torch.max(probs, dim=1)
            return int(pred.item()), float(conf.item())

    def predict_from_preprocessed(self, fnn_input_np):
        """
        fnn_input_np: numpy float32 (1, 784) already normalized with MNIST mean/std.
        Returns (label:int, confidence:float)
        """
        import torch
        tensor = torch.from_numpy(fnn_input_np).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.exp(out)
            conf, pred = torch.max(probs, dim=1)
            return int(pred.item()), float(conf.item())