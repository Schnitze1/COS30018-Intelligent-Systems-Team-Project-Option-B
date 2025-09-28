import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

"""
Preprocessing utilities for segmentation and model input preparation.

Functions:
 - read_image(path, color=True)
 - to_grayscale(img)
 - denoise(img, method='gaussian')
 - otsu_binarize(img)
 - ensure_white_on_black(img)  # digits white on black bg
 - deskew(img)
 - resize_and_center(img, size=(28,28))
 - prepare_for_fnn(img)  -> numpy float32 shape (1, 784) scaled & normalized (MNIST stats)
 - prepare_for_cnn(img)  -> torch tensor shape (1,1,28,28) normalized (MNIST stats)
 - prepare_for_seq(img)  -> numpy float32 shape (1,28,28) normalized similar to Colab
"""


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def read_image(path: str, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_grayscale(img: np.ndarray):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def denoise(img: np.ndarray, method='gaussian'):
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (3,3), 0)
    elif method == 'median':
        return cv2.medianBlur(img, 3)
    return img

def otsu_binarize(gray_img: np.ndarray):
    if gray_img.dtype != np.uint8:
        gray = (np.clip(gray_img,0,1)*255).astype('uint8')
    else:
        gray = gray_img
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th



def enhance_for_models(img: np.ndarray, use_clahe=True, dilate=True):
    """
    Apply preprocessing enhancements to bring custom digits closer to MNIST style.
    img: uint8 grayscale (28x28, digits white on black).
    """
    out = img.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        out = clahe.apply(out)

    if dilate:
        th = cv2.threshold(out, 10, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        out = cv2.dilate(th, kernel, iterations=1)
        out = cv2.GaussianBlur(out, (3,3), 0)

    return out


def ensure_white_on_black(img: np.ndarray):
    """
    More robust polarity fixer:
    - compute Otsu binarization (both normal and inverted)
    - choose the version whose foreground area is smaller (digits are small)
    - return a uint8 image with digits white (255) on black (0)
    """
    gray = to_grayscale(img)
    if gray.dtype != np.uint8:
        gray8 = (np.clip(gray,0,1)*255).astype('uint8')
    else:
        gray8 = gray.copy()

    # normal Otsu (digits dark -> foreground is dark pixels)
    _, th_normal = cv2.threshold(gray8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # inverted Otsu (digits white -> foreground is white pixels)
    _, th_inv = cv2.threshold(gray8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # compute foreground areas (count nonzero)
    area_normal = int(np.count_nonzero(th_normal))
    area_inv = int(np.count_nonzero(th_inv))

    # pick the binarization where the foreground area is smaller (digits are small)
    # If area_inv is smaller, we want th_inv (digits white-on-black). Otherwise invert th_normal.
    if area_inv <= area_normal:
        chosen = th_inv
    else:
        # invert th_normal so digits become white
        chosen = 255 - th_normal

    # optional small opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    chosen_clean = cv2.morphologyEx(chosen, cv2.MORPH_OPEN, kernel, iterations=1)
    return chosen_clean


def deskew(img: np.ndarray):
    """
    Deskew a grayscale image using image moments.
    Input and output are uint8 grayscale images.
    """
    gray = img if img.dtype == np.uint8 else (img*255).astype('uint8')
    m = cv2.moments(gray)
    if abs(m["mu02"]) < 1e-2:
        return gray
    skew = m["mu11"] / m["mu02"]
    h, w = gray.shape[:2]
    M = np.float32([[1, skew, -0.5*w*skew],[0,1,0]])
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def resize_and_center(img, size=(28,28), pad=4):
    """
    Resize keeping aspect ratio and center into target size.
    img: grayscale numpy array (H,W) with foreground white on black.
    """
    h,w = img.shape[:2]
    # find bbox of nonzero to crop tightly
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = img[y0:y1+1, x0:x1+1]
    # compute new size that fits within size - pad
    target_w, target_h = size
    max_w = target_w - pad*2
    max_h = target_h - pad*2
    ch, cw = crop.shape[:2]
    scale = min(max_w/cw, max_h/ch)
    new_w = max(1, int(cw*scale))
    new_h = max(1, int(ch*scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # place into black canvas
    canvas = np.zeros(size, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def prepare_for_fnn(crop_image):
    """
    Returns numpy float32 shape (1, 784), normalized with MNIST mean/std.
    Accepts PIL Image or numpy array (H,W) or color.
    """
    if not isinstance(crop_image, Image.Image):
        crop_image = Image.fromarray(crop_image)
    img = crop_image.convert("L").resize((28,28), Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0
    # ensure white-on-black
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    # normalize using MNIST stats (same as PyTorch transform)
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return arr.reshape(1, -1).astype('float32')

def prepare_for_cnn(crop_image, device='cpu'):
    """
    Returns torch Tensor shape (1,1,28,28) normalized with MNIST stats.
    """
    if not isinstance(crop_image, Image.Image):
        crop_image = Image.fromarray(crop_image)
    img = crop_image.convert("L").resize((28,28), Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    t = transforms.Compose([
        transforms.ToTensor(),  # to (1,28,28) and scale 0..1
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    # feed t with PIL converted back from arr:
    img2 = Image.fromarray((arr*255).astype('uint8'))
    tensor = t(img2).unsqueeze(0)  # (1,1,28,28)
    return tensor.to(device)

def prepare_for_seq(crop_image):
    """
    Returns numpy float32 shape (1,28,28) normalized similar to Colab usage:
    tf.keras.utils.normalize(arr, axis=1)
    """
    if not isinstance(crop_image, Image.Image):
        crop_image = Image.fromarray(crop_image)
    img = crop_image.convert("L").resize((28,28), Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    import tensorflow as tf
    arr = tf.keras.utils.normalize(arr, axis=1)
    return arr.reshape(1, 28, 28).astype('float32')