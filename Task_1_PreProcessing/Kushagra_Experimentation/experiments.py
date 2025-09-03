"""
COS30018 - Intelligent Systems
Project Assignment Option B
Task 1: Preprocessing Experiments

Author: Kushagra Suryawanshi
-----------------------------------
This script runs three preprocessing experiments for MNIST digit recognition:

1. Raw Pixels + Logistic Regression (baseline)
2. HOG Features + Linear SVM
3. PCA (50 components) + Logistic Regression

It also saves example images:
- digit_original.png  : Raw MNIST digit
- digit_hog.png       : HOG feature visualization
- digit_pca.png       : PCA reconstructed digit

And prints accuracy comparisons for Task 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from skimage import exposure

# ---------------------------
# Load MNIST dataset
# ---------------------------
print("Loading MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0   # normalize to [0,1]
y = y.astype(int)

# Take subset (6000 images for speed)
X_small, _, y_small, _ = train_test_split(X, y, train_size=6000, stratify=y, random_state=42)

# ---------------------------
# Save example images
# ---------------------------
example = X_small[0].reshape(28, 28)

# Original
plt.imshow(example, cmap="gray")
plt.title("Original MNIST Digit")
plt.savefig("digit_original.png")
plt.close()

# HOG Visualization
hog_features, hog_image = hog(example, orientations=9, pixels_per_cell=(8,8),
                              cells_per_block=(2,2), block_norm='L2-Hys',
                              visualize=True, feature_vector=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Visualization")
plt.savefig("digit_hog.png")
plt.close()

# PCA Reconstruction
pca_vis = PCA(n_components=50)
X_pca_vis = pca_vis.fit_transform(X_small[:1000])    # fit on 1k samples
reconstructed = pca_vis.inverse_transform(X_pca_vis[0]).reshape(28,28)
plt.imshow(reconstructed, cmap="gray")
plt.title("PCA Reconstructed (50 comps)")
plt.savefig("digit_pca.png")
plt.close()

print("Saved example images: digit_original.png, digit_hog.png, digit_pca.png")

# ---------------------------
# Experiment 1: Raw Pixels + Logistic Regression
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, stratify=y_small, random_state=42)
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
acc_raw = lr.score(X_test, y_test)
print("Raw Pixels + LR Accuracy:", acc_raw)

# ---------------------------
# Experiment 2: HOG + Linear SVM
# ---------------------------
def extract_hog(images):
    return np.array([
        hog(img.reshape(28,28), orientations=9, pixels_per_cell=(8,8),
            cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
        for img in images
    ])

X_hog = extract_hog(X_small)
X_train, X_test, y_train, y_test = train_test_split(X_hog, y_small, test_size=0.2, stratify=y_small, random_state=42)
svm_clf = svm.LinearSVC(max_iter=5000)
svm_clf.fit(X_train, y_train)
acc_hog = svm_clf.score(X_test, y_test)
print("HOG + SVM Accuracy:", acc_hog)

# ---------------------------
# Experiment 3: PCA + Logistic Regression
# ---------------------------
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_small)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_small, test_size=0.2, stratify=y_small, random_state=42)
pca_clf = LogisticRegression(max_iter=2000)
pca_clf.fit(X_train, y_train)
acc_pca = pca_clf.score(X_test, y_test)
print("PCA (50 comps) + LR Accuracy:", acc_pca)

# ---------------------------
# Summary
# ---------------------------
print("\n=== Task 1 Results ===")
print(f"Raw Pixels + LR : {acc_raw:.4f}")
print(f"HOG + SVM       : {acc_hog:.4f}")
print(f"PCA + LR        : {acc_pca:.4f}")
