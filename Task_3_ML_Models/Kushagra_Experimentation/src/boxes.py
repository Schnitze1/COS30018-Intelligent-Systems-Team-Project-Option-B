import os, cv2

IN_PATH = "binarized/binarized_clean.png"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("some error finding the file, chck the path ")

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # grayscale → color
boxes = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)  # smallest box around contour
    if w*h > 50:   # ignore tiny spots/noise
        boxes.append((x,y,w,h))
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)  # draw green box


boxes.sort(key=lambda box: box[0])

out_path = os.path.join(OUT_DIR, "preview_boxes.png")
cv2.imwrite(out_path, vis)

print(f"Found {len(boxes)} digits")
print("Saved preview image with bounding boxes at:", out_path)
    