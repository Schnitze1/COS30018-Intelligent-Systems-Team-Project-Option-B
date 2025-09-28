import os
from typing import List, Tuple, Optional
import cv2
import numpy as np

# preprocessing helpers from your new module
from src.preprocessing.preprocessing import (
    to_grayscale,
    denoise,
    otsu_binarize,
    ensure_white_on_black,
    deskew,
    resize_and_center,
)

"""
Segmentation module (consolidates binarising.py, boxes.py, crop_digit.py, preview.py)

Functions:
 - segment_image(...) -> segments digits, saves crops, returns boxes, crops, overlay_img
 - save_overlay(...) -> writes overlay image
 - save_crops(...) -> writes cropped digit images as segmented_00.png, ...
 - build_labeled_preview(...) -> overlay image with labels (same as preview_labeled.png)

CLI:
    python -m src.segmentation.segment path/to/image.png
"""


# canonical locations (relative to project root)
DEFAULT_OUT_DIR = "outputs"
DEFAULT_BIN_DIR = "binarized"

def _binarize_image(img_gray: np.ndarray, save_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (binary_inv, cleaned) both uint8 images
    - binary_inv is Otsu thresholded and inverted so digits are white (255) on black (0)
    - cleaned applies a small morphological open to remove speckle
    """
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "binarized.png"), th_inv)
        cv2.imwrite(os.path.join(save_dir, "binarized_clean.png"), cleaned)
    return th_inv, cleaned

def find_digit_contours_from_binary(binary_img: np.ndarray, min_area: int = 50) -> List[Tuple[int,int,int,int]]:
    """
    Given a binary image (white foreground on black background), return bounding boxes
    sorted left-to-right. Filters small components using min_area.
    """
    # ensure binary_img is uint8
    binu = binary_img.copy()
    if binu.dtype != np.uint8:
        binu = (np.clip(binu,0,1)*255).astype('uint8')

    contours, _ = cv2.findContours(binu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = binu.shape[:2]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < min_area:
            continue
        # optional: ignore full-image boxes
        if w >= 0.9 * W and h >= 0.9 * H:
            continue
        boxes.append((x,y,w,h))
    boxes = sorted(boxes, key=lambda b: b[0])  # left -> right
    return boxes

def crop_from_boxes(gray_image: np.ndarray, boxes: List[Tuple[int,int,int,int]], pad: int = 4) -> List[np.ndarray]:
    """
    Crop grayscale image using boxes, add padding, clamp to image bounds.
    Returns list of uint8 crops.
    """
    H, W = gray_image.shape[:2]
    crops = []
    for (x,y,w,h) in boxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        crop = gray_image[y0:y1, x0:x1]
        crops.append(crop)
    return crops

def save_crops(crops: List[np.ndarray], out_dir: str = DEFAULT_OUT_DIR, prefix: str = "segmented"):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, crop in enumerate(crops):
        fname = f"{prefix}_{i:02d}.png"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, crop)
        saved.append(path)
    return saved

def build_overlay_image(color_image: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: Optional[List[str]] = None, pad: int = 0):
    """
    Draw boxes (and optional labels) on a color image (BGR) and return the annotated image.
    """
    vis = color_image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for i, (x,y,w,h) in enumerate(boxes):
        cv2.rectangle(vis, (x-pad, y-pad), (x+w+pad, y+h+pad), (0,255,0), 2)
        if labels is not None:
            lab = str(labels[i]) if i < len(labels) else str(i)
            cv2.putText(vis, lab, (x, max(12, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return vis

def segment_image(
        image_path: str,
        out_dir: str = DEFAULT_OUT_DIR,
        bin_dir: str = DEFAULT_BIN_DIR,
        pad: int = 4,
        min_area: int = 50,
        save_crops_flag: bool = True,
        return_overlay: bool = True,
):
    """
    Full segmentation pipeline:
      1) read grayscale image
      2) binarize (Otsu + clean)
      3) find contours -> bounding boxes
      4) crop boxes from the original grayscale
      5) deskew/center/resize each crop to 28x28 for later processing (but save original crop too)
      6) save segmented_XX.png files in out_dir and binarized files in bin_dir
      7) return dict with boxes, crops (both original and centered 28x28), overlay image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    # Read grayscale original
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image as grayscale: {image_path}")

    # 1) Binarize and save
    th_inv, cleaned = _binarize_image(gray, save_dir=bin_dir)

    # 2) Find boxes (use cleaned binary)
    boxes = find_digit_contours_from_binary(cleaned, min_area=min_area)

    # 3) Crop from original grayscale (better than binary for preserving stroke gradation)
    crops = crop_from_boxes(gray, boxes, pad=pad)

    # 4) Create centered 28x28 versions using preprocessing helpers
    centered_crops = []
    for crop in crops:
        # ensure foreground polarity is digits white-on-black for centering
        bin_crop = ensure_white_on_black(crop)
        desk = deskew(bin_crop)
        centered28 = resize_and_center(desk, size=(28,28), pad=4)
        centered_crops.append(centered28)

    # 5) Save crops (original grayscale crops) as segmented_XX.png if requested
    saved_paths = []
    if save_crops_flag:
        saved_paths = save_crops(crops, out_dir=out_dir, prefix="segmented")

    # 6) Build overlay (labels left blank)
    color_src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_src is None:
        color_src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = build_overlay_image(color_src, boxes, labels=None, pad=pad)

    result = {
        "image_path": image_path,
        "boxes": boxes,
        "crops": crops,                      # list of original grayscale crops (uint8)
        "centered_crops": centered_crops,    # list of 28x28 uint8 ready for model preprocessing
        "saved_paths": saved_paths,
        "overlay": overlay,
        "binary": cleaned
    }
    # optionally save overlay preview image
    preview_path = os.path.join(out_dir, "preview_labeled.png")
    cv2.imwrite(preview_path, build_overlay_image(color_src, boxes, labels=[str(i) for i in range(len(boxes))], pad=pad))
    return result

# ---- CLI / simple test ----
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Segment digits from an input image.")
    parser.add_argument("image", help="input image path (grayscale or color)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory for crops/preview")
    parser.add_argument("--bin", default=DEFAULT_BIN_DIR, help="binarized outputs directory")
    parser.add_argument("--pad", type=int, default=4, help="padding around boxes when cropping")
    parser.add_argument("--min-area", type=int, default=50, help="minimum bbox area to keep")
    args = parser.parse_args()

    res = segment_image(args.image, out_dir=args.out, bin_dir=args.bin, pad=args.pad, min_area=args.min_area)
    print(f"[Segment] Found {len(res['boxes'])} boxes. Saved {len(res['saved_paths'])} crops to {args.out}")
    print("Preview written to:", os.path.join(args.out, "preview_labeled.png"))

if __name__ == "__main__":
    _cli()