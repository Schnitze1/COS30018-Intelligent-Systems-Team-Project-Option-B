import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import layers
import cv2 
import os
from preprocess import preprocess_image

model = tf.keras.models.load_model('ML_Models/SEQUENTIAL_MNIST_MODEL/Sequential.keras')

img = preprocess_image('ML_Models/testing_images/Thicker3.png')

plt.imshow(img[0])
plt.show()
pred = model.predict(img)
print(f"result: {tf.argmax(pred, axis=1).numpy()[0]}")

