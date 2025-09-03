import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import layers
import cv2 
import os

model = tf.keras.models.load_model('ML_Models/SEQ_MNIST_Model/Sequential.keras')

img = cv2.imread("ML_Models/test_images/DionDrawing1.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = 255 - img
img = tf.expand_dims(img, axis=0)

pred = model.predict(img)

plt.imshow(img[0], cmap="gray")
plt.show()
print(f"result: {tf.argmax(pred, axis=1).numpy()[0]}")