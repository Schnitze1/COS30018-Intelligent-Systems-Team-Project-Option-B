import os
import numpy as np
from PIL import Image
import tensorflow as tf

"""


"""

class SeqModel:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Sequential model not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        # The Sequential in your Colab used tf.keras.utils.normalize(..., axis=1)
        self.input_size = (28, 28)

    def _preprocess(self, crop_image):
        # Accept PIL image or numpy array
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)
        img = crop_image.convert("L").resize(self.input_size, Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0  # scale 0..1
        # Heuristic to match training: make digits white-on-black (MNIST-like)
        if arr.mean() > 0.5:
            arr = 1.0 - arr
        # The Colab training applied tf.keras.utils.normalize(x, axis=1)
        arr = tf.keras.utils.normalize(arr, axis=1)
        # model expects shape (batch, 28,28) because Input(shape=(28,28))
        return arr.reshape(1, 28, 28).astype("float32")

    def predict(self, crop_image):
        """
        Returns (label:int, confidence:float)
        """
        x = self._preprocess(crop_image)
        probs = self.model.predict(x)  # shape (1,10)
        pred = int(probs.argmax(axis=1)[0])
        conf = float(probs.max(axis=1)[0])
        return pred, conf

    def predict_from_preprocessed(self, seq_input_np):
        """
        seq_input_np: numpy float32 (1,28,28) already normalized similar to Colab.
        Returns (label:int, confidence:float)
        """
        probs = self.model.predict(seq_input_np)  # (1,10)
        pred = int(probs.argmax(axis=1)[0])
        conf = float(probs.max(axis=1)[0])
        return pred, conf