# seq_train_save.py  (local)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# canonical path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "saved_models"))
os.makedirs(MODELS_DIR, exist_ok=True)

SAVE_DIR = os.path.join(MODELS_DIR, "Sequential.keras")
print(f"[SEQ] Save dir: {SAVE_DIR}")

mnist = tf.keras.datasets.mnist
(x_train , y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = keras.Sequential([
    layers.Input(shape=(28, 28), name="inputs"),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train.astype('float32'), y_train.astype('int32'), epochs=3)
model.save(SAVE_DIR)

# Optional quick eval
model = tf.keras.models.load_model(SAVE_DIR)
loss, accuracy = model.evaluate(x_train, y_train)
print(f"Accuracy:{accuracy}")
print(f"Loss: {loss}")