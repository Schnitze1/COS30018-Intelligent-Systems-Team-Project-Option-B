import os, cv2

BIN_PATH = "binarized/binarized_clean.png"   # binary image (white digits on black)
GRAY_SRC = "inputs/1234.png"                # my original grayscale image
OUT_DIR  = "outputs"
PAD = 2  # pixels of padding around each crop

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(BIN_PATH, cv2.IMREAD_GRAYSCALE)
gray = cv2.imread(GRAY_SRC, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Problem reaching the image")
if gray is None:
    raise FileNotFoundError(f"Problem reaching the grayscaled image")


contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h > 50:  
        boxes.append((x, y, w, h))
boxes.sort(key=lambda b: b[0])


saved = []
H, W = gray.shape[:2]
for i, (x, y, w, h) in enumerate(boxes):
    x0 = max(x - PAD, 0)
    y0 = max(y - PAD, 0)
    x1 = min(x + w + PAD, W)
    y1 = min(y + h + PAD, H)
    crop = gray[y0:y1, x0:x1]
    out_path = os.path.join(OUT_DIR, f"segmented_{i:02d}.png")
    cv2.imwrite(out_path, crop)
    saved.append(os.path.basename(out_path))


print(f"Cropped {len(saved)} digits:", saved)
print(f"Saved to: {OUT_DIR}")
