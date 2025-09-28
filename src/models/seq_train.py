# seq_train.py
import os
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "saved_models"))
os.makedirs(MODELS_DIR, exist_ok=True)
SAVE_DIR = os.path.join(MODELS_DIR, "Sequential.keras")

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def preprocess_fn(image, label=None):
    # image may be uint8 HxW or 3-ch; produce float32 28x28 scaled 0..1 and invert if needed
    image = tf.image.rgb_to_grayscale(image) if image.shape[-1] == 3 else image
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0
    # invert if background is bright
    mean = tf.reduce_mean(image)
    image = tf.cond(mean > 0.5, lambda: 1.0 - image, lambda: image)
    # the original Colab training normalised with tf.keras.utils.normalize(..., axis=1)
    # We'll apply same operation per-sample on axis=1
    image = tf.keras.utils.normalize(image, axis=1)
    if label is None:
        return image
    return image, label

def build_mnist_dataset(batch_size=128, extra_dir=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train shape (N,28,28) -> add channel dim
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Optionally add extra labeled data from directory (expects subfolders 0..9)
    if extra_dir:
        extra_dir = str(Path(extra_dir).resolve())
        if os.path.isdir(extra_dir):
            extra_ds = tf.keras.preprocessing.image_dataset_from_directory(
                extra_dir,
                labels="inferred",
                label_mode="int",
                color_mode="grayscale",
                batch_size=batch_size,
                image_size=(28,28),
                shuffle=True
            ).map(lambda x,y: (tf.keras.utils.normalize(tf.cast(x/255.0, tf.float32), axis=1), y))
            # concatenate datasets
            train_ds = train_ds.concatenate(extra_ds)
            print(f"[SEQ] Added extra data from {extra_dir}")
        else:
            print(f"[SEQ] Extra dir {extra_dir} not found; skipping extra data.")
    return train_ds, test_ds

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28,28), name="inputs"),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save(epochs=3, batch_size=128, extra_dir=None):
    train_ds, test_ds = build_mnist_dataset(batch_size=batch_size, extra_dir=extra_dir)
    model = build_model()
    model.fit(train_ds, epochs=epochs, validation_data=test_ds)
    model.save(SAVE_DIR)
    print(f"[SEQ] Model saved to {SAVE_DIR}")
    loss, acc = model.evaluate(test_ds)
    print(f"[SEQ] Final test eval -> loss: {loss:.4f}, acc: {acc:.4f}")
    return SAVE_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--extra-data", type=str, default=None, help="Optional: path to extra labeled data with subfolders 0..9")
    args = parser.parse_args()
    train_and_save(epochs=args.epochs, batch_size=args.batch_size, extra_dir=args.extra_data)