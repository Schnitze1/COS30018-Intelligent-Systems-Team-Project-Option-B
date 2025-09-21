import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2 
import os

def preprocess_image(img): 
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("image not found or unable to load.")
    else:        
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        img = tf.expand_dims(img, axis=0)
        return img
