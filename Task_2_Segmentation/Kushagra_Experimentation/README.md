# Task 2 – Image Segmentation (Kushagra)

This folder contains my experiments for Task 2 of the Handwritten Number Recognition System project.  
I implemented a **contour-based segmentation** pipeline:

1. **Binarization** – converted digits to white on black background (Otsu’s method + morphological cleaning).  
2. **Contour Detection** – used OpenCV `findContours` to locate connected components (digit blobs).  
3. **Bounding Boxes** – drew boxes around each digit for visualization.  
4. **Digit Cropping** – cropped each digit as a separate image.  
5. **Labeled Preview** – generated an overlay image with boxes and index labels.

### Example Outputs
- `binarized/binarized_clean.png` – binary version of the input.  
- `outputs/preview_boxes.png` – digits with green boxes.  
- `outputs/segmented_00.png`, `segmented_01.png`, … – individual digit crops.  
- `outputs/preview_labeled.png` – original with boxes + digit labels.

### How to Run
```bash
python binarising.py
python boxes.py
python crop_digits.py
python preview.py
