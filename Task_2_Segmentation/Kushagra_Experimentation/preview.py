import os, cv2

BIN_PATH = "binarized/binarized_clean.png"
GRAY_SRC = "inputs/1234.png"
OUT_DIR  = "outputs"
PAD = 2

os.makedirs(OUT_DIR, exist_ok=True)

# Load
th = cv2.imread(BIN_PATH, cv2.IMREAD_GRAYSCALE)
gray = cv2.imread(GRAY_SRC, cv2.IMREAD_GRAYSCALE)

# Find contours
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w*h > 50:
        boxes.append((x,y,w,h))
boxes.sort(key=lambda b: b[0])

# Make color version for drawing
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for i,(x,y,w,h) in enumerate(boxes):
    cv2.rectangle(vis, (x-PAD,y-PAD), (x+w+PAD, y+h+PAD), (0,255,0), 2)
    cv2.putText(vis, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

out_path = os.path.join(OUT_DIR, "preview_labeled.png")
cv2.imwrite(out_path, vis)
print(f"Saved labeled preview with {len(boxes)} digits -> {out_path}")
