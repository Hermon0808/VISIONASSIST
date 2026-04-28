"""
Generate minimal test data for metrics.py evaluation tasks.
Creates sample images, labels, and text files.
"""
import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# 1. Detection: create test images with colored rectangles & YOLO labels
# ==========================================================================
img_dir = os.path.join(BASE, "test_images")
lbl_dir = os.path.join(BASE, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

for i in range(3):
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a rectangle representing a "ground truth" object
    x1, y1 = 100 + i * 50, 100 + i * 30
    x2, y2 = 300 + i * 50, 300 + i * 30
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
    fname = f"test_{i:03d}"
    cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), img)

    # YOLO label: class_id x_center y_center width height (normalised)
    cx = ((x1 + x2) / 2) / 640
    cy = ((y1 + y2) / 2) / 640
    w  = (x2 - x1) / 640
    h  = (y2 - y1) / 640
    with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

print(f"[OK] Detection: {3} images + labels created")

# ==========================================================================
# 2. OCR: create images with text rendered on them & ground truth .txt
# ==========================================================================
ocr_img_dir = os.path.join(BASE, "ocr_images")
ocr_txt_dir = os.path.join(BASE, "ocr_labels")
os.makedirs(ocr_img_dir, exist_ok=True)
os.makedirs(ocr_txt_dir, exist_ok=True)

sample_texts = [
    "Hello World",
    "Python 3",
    "OpenCV Test",
]
for i, text in enumerate(sample_texts):
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255  # white background
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    fname = f"ocr_{i:03d}"
    cv2.imwrite(os.path.join(ocr_img_dir, f"{fname}.png"), img)
    with open(os.path.join(ocr_txt_dir, f"{fname}.txt"), "w", encoding="utf-8") as f:
        f.write(text)

print(f"[OK] OCR: {len(sample_texts)} images + ground truth created")

# ==========================================================================
# 3. Summarization: create source documents and reference summaries
# ==========================================================================
docs_dir = os.path.join(BASE, "docs")
refs_dir = os.path.join(BASE, "refs")
os.makedirs(docs_dir, exist_ok=True)
os.makedirs(refs_dir, exist_ok=True)

documents = [
    (
        "Artificial intelligence has transformed many industries. "
        "Machine learning models can now recognize images, translate languages, "
        "and generate human-like text. These advances have led to significant "
        "productivity gains across healthcare, finance, and technology sectors."
    ),
]
references = [
    "AI has transformed industries through advances in machine learning, "
    "improving productivity in healthcare, finance, and technology.",
]
for i, (doc, ref) in enumerate(zip(documents, references)):
    with open(os.path.join(docs_dir, f"doc_{i:03d}.txt"), "w", encoding="utf-8") as f:
        f.write(doc)
    with open(os.path.join(refs_dir, f"doc_{i:03d}.txt"), "w", encoding="utf-8") as f:
        f.write(ref)

print(f"[OK] Summarization: {len(documents)} document-reference pairs created")
print("\nDone! You can now run:")
print("  python metrics.py --task detection --images ./test_images --labels ./labels")
print("  python metrics.py --task ocr --images ./ocr_images --texts ./ocr_labels")
print("  python metrics.py --task summarization --inputs ./docs --references ./refs")
