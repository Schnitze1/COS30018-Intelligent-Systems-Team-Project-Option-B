import os
import cv2
import numpy as np

"""
This file is the entry point for the digit recognition pipeline. The application utilises a modular design that re-uses
the pre-processing and image segmentation processes for each model. 

"""

# Try to import user segmentation module; if not present, use fallback
try:
    from src.segmentation.segment import find_digit_contours, crop_digits, overlay_boxes
    _HAS_SEG = True
except Exception:
    _HAS_SEG = False
    def find_digit_contours(image, min_area=100):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h,w = gray.shape[:2]
        for cnt in contours:
            x,y,ww,hh = cv2.boundingRect(cnt)
            if ww*hh < min_area:
                continue
            boxes.append((x,y,ww,hh))
        boxes = sorted(boxes, key=lambda b: b[0])
        return boxes

    def crop_digits(image, boxes, pad=6):
        crops = []
        h,w = image.shape[:2]
        for x,y,ww,hh in boxes:
            x0 = max(0, x-pad); y0 = max(0, y-pad)
            x1 = min(w, x+ww+pad); y1 = min(h, y+hh+pad)
            crops.append(image[y0:y1, x0:x1])
        return crops

    def overlay_boxes(image, boxes, labels=None):
        img = image.copy()
        for i,(x,y,ww,hh) in enumerate(boxes):
            cv2.rectangle(img, (x,y), (x+ww, y+hh), (0,255,0), 2)
            txt = str(labels[i]) if labels and i < len(labels) else str(i)
            cv2.putText(img, txt, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return img

# preprocessing helpers
from src.preprocessing.preprocessing import (
    ensure_white_on_black,
    deskew,
    resize_and_center,
    prepare_for_fnn,
    prepare_for_seq,
    prepare_for_cnn
)

# model wrappers
from src.models.fnn_wrapper import FNNModel
from src.models.seq_wrapper import SeqModel
from src.models.cnn_wrapper import CNNModel

def preprocess_crop_pipeline(crop_np_uint8):
    """
    crop_np_uint8: numpy uint8 (H,W) or (H,W,3)
    Returns a tuple of standardized preprocessed representations:
      - fnn_input: numpy float32 (1, 784) normalized (MNIST stats)
      - seq_input: numpy float32 (1,28,28) normalized like your Colab code
      - cnn_input: torch.Tensor (1,1,28,28) normalized (MNIST stats)
    """
    # Step 1: ensure grayscale uint8
    if crop_np_uint8.ndim == 3:
        import cv2
        gray = cv2.cvtColor(crop_np_uint8, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_np_uint8.copy()

    # Step 2: make digits white-on-black and binarize
    whites = ensure_white_on_black(gray)  # uint8 binary image (0 or 255)
    # Step 3: deskew (optional, helpful)
    desk = deskew(whites)
    # Step 4: resize & center into 28x28
    centered = resize_and_center(desk, size=(28,28), pad=4)

    # Now prepare model-specific inputs
    fnn_input = prepare_for_fnn(centered)    # (1,784) numpy float32
    seq_input = prepare_for_seq(centered)    # (1,28,28) numpy float32
    cnn_input = prepare_for_cnn(centered)    # torch tensor (1,1,28,28)

    return fnn_input, seq_input, cnn_input, centered  # return centered for debug/display

def run_pipeline_on_image(image_input, model_paths: dict, device: str = None):
    """
    image_input: path string or numpy array (BGR)
    model_paths: dict with keys 'fnn','seq','cnn' -> paths to saved models
    returns:
      {
        'boxes': [ (x,y,w,h), ... ],
        'predictions': [ {'fnn':{'label', 'conf'}, ... }, ... ],
        'overlay': overlay_image (numpy BGR),
        'crops_centered': [28x28 numpy images]  # aligned crops after deskew/resize
      }
    """
    # Read image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_input}")
    else:
        img = image_input.copy()

    boxes = find_digit_contours(img)
    crops = crop_digits(img, boxes)

    # load models (lazy)
    models = {}
    if 'fnn' in model_paths:
        models['fnn'] = FNNModel(model_paths['fnn'], device=device)
    if 'seq' in model_paths:
        models['seq'] = SeqModel(model_paths['seq'])
    if 'cnn' in model_paths:
        models['cnn'] = CNNModel(model_paths['cnn'], device=device)

    predictions = []
    overlay_labels = []
    crops_centered = []

    for crop in crops:
        # make sure crop is uint8
        if crop.dtype != np.uint8:
            crop = (np.clip(crop,0,1)*255).astype('uint8')

        # pre-process pipeline -> canonical inputs
        fnn_in, seq_in, cnn_in, centered = preprocess_crop_pipeline(crop)
        crops_centered.append(centered)

        crop_preds = {}
        # FNN: accept preprocessed numpy (1,784)
        if 'fnn' in models:
            try:
                # prefer predict_from_preprocessed if available
                if hasattr(models['fnn'], "predict_from_preprocessed"):
                    label, conf = models['fnn'].predict_from_preprocessed(fnn_in)
                else:
                    label, conf = models['fnn'].predict(crop)
            except Exception as e:
                label, conf = None, None
            crop_preds['fnn'] = {'label': label, 'conf': conf}

        # SEQ
        if 'seq' in models:
            try:
                if hasattr(models['seq'], "predict_from_preprocessed"):
                    label, conf = models['seq'].predict_from_preprocessed(seq_in)
                else:
                    label, conf = models['seq'].predict(crop)
            except Exception as e:
                label, conf = None, None
            crop_preds['seq'] = {'label': label, 'conf': conf}

        # CNN
        if 'cnn' in models:
            try:
                if hasattr(models['cnn'], "predict_from_preprocessed"):
                    label, conf = models['cnn'].predict_from_preprocessed(cnn_in)
                else:
                    label, conf = models['cnn'].predict(crop)
            except Exception as e:
                label, conf = None, None
            crop_preds['cnn'] = {'label': label, 'conf': conf}

        # select overlay label priority cnn -> seq -> fnn
        chosen = crop_preds.get('cnn') or crop_preds.get('seq') or crop_preds.get('fnn')
        overlay_labels.append(chosen['label'] if chosen else '')
        predictions.append(crop_preds)

    overlay_img = overlay_boxes(img, boxes, overlay_labels)
    return {
        'boxes': boxes,
        'predictions': predictions,
        'overlay': overlay_img,
        'crops_centered': crops_centered
    }