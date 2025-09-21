import os, cv2

IN_PATH  = "inputs/1234.png"
 
OUT_DIR  = "binarized"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load in grayscale
img = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not read: {IN_PATH}")

# 2) Otsu threshold with inversion (since digits are dark on light background)
#    THRESH_BINARY_INV makes digits -> white (255), background -> black (0)
blur = cv2.GaussianBlur(img, (5,5), 0)
_, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(OUT_DIR, "binarized.png"), th_inv)

# 3) Optional: small morphological open to clean noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
clean = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imwrite(os.path.join(OUT_DIR, "binarized_clean.png"), clean)

print("Saved:",
      os.path.join(OUT_DIR, "binarized.png"),
      "and",
      os.path.join(OUT_DIR, "binarized_clean.png"))
