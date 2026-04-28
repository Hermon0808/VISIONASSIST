# ocr.py — Live camera OCR with automatic paper detection + LLaMA summary
# =========================================================================
#
# Usage (standalone):
#   python ocr.py
#   python ocr.py --image path/to/image.png   (skip camera, use file)
#
# When used as a module by voice_assistant_main.py:
#   from ocr import scan_paper_live, summarize_with_llama
# =========================================================================

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
import requests

LOGGER = logging.getLogger(__name__)

# ── Tesseract settings ──────────────────────────────────────────────────
DEFAULT_TESSERACT_CONFIG = "--oem 3 --psm 6"

# ── Ollama / LLaMA settings (local) ─────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "llama3"

# ── Paper-detection tuning ───────────────────────────────────────────────
MIN_PAPER_AREA_RATIO = 0.15        # paper must fill ≥15 % of the frame
STABLE_FRAMES_REQUIRED = 15        # paper must be steady for N frames
CONTOUR_APPROX_EPSILON = 0.02      # polygon approximation tolerance
CANNY_LOW, CANNY_HIGH = 30, 100    # Canny edge thresholds


# ═════════════════════════════════════════════════════════════════════════
#  Image pre-processing & OCR
# ═════════════════════════════════════════════════════════════════════════

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enlarge + threshold an image for better Tesseract accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(
        resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def extract_text(image: np.ndarray, *, lang: str = "eng",
                 config: str = DEFAULT_TESSERACT_CONFIG) -> str:
    """Run Tesseract on an OpenCV image and return the extracted text."""
    processed = preprocess_for_ocr(image)
    text = pytesseract.image_to_string(processed, lang=lang,
                                       config=config).strip()
    return text


# ═════════════════════════════════════════════════════════════════════════
#  Paper / document detection helpers
# ═════════════════════════════════════════════════════════════════════════

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-warp the region defined by four points into a rectangle."""
    rect = _order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    dst = np.array([
        [0, 0], [max_w - 1, 0],
        [max_w - 1, max_h - 1], [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))


def detect_paper_contour(frame: np.ndarray) -> np.ndarray | None:
    """Find the largest quadrilateral contour that could be a paper."""
    h, w = frame.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * MIN_PAPER_AREA_RATIO:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, CONTOUR_APPROX_EPSILON * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best = approx
            best_area = area

    return best


# ═════════════════════════════════════════════════════════════════════════
#  Live camera scanning
# ═════════════════════════════════════════════════════════════════════════

def scan_paper_live(cap: cv2.VideoCapture | None = None,
                    *,
                    own_camera: bool = False,
                    camera_index: int = 0) -> tuple[str, np.ndarray] | None:
    """
    Show a live camera preview.  When a paper is detected and held
    steady for a short period the frame is captured and OCR is run.

    Returns (extracted_text, warped_image) or None on failure / cancel.

    Press **q** or **Esc** to cancel manually.
    Press **Space** to force-capture even without auto-detection.
    """
    opened_here = False
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            LOGGER.error("Cannot open camera %s", camera_index)
            return None
        opened_here = True
        time.sleep(0.8)   # warm-up

    stable_count = 0
    captured_image: np.ndarray | None = None
    status_msg = "Looking for paper..."
    status_color = (0, 165, 255)   # orange

    LOGGER.info("OCR live preview started. Press [Space] to force-capture, "
                "[q/Esc] to cancel.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                LOGGER.warning("Camera read failed during OCR preview.")
                break

            display = frame.copy()
            contour = detect_paper_contour(frame)

            if contour is not None:
                # Draw green quadrilateral around the detected paper
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 3)
                stable_count += 1

                if stable_count < STABLE_FRAMES_REQUIRED:
                    pct = int(stable_count / STABLE_FRAMES_REQUIRED * 100)
                    status_msg = f"Paper detected — hold steady... {pct}%"
                    status_color = (0, 200, 255)   # yellow-ish
                else:
                    # Auto-capture!
                    status_msg = "Captured! Processing OCR..."
                    status_color = (0, 255, 0)
                    captured_image = _four_point_warp(frame, contour)
            else:
                stable_count = 0
                status_msg = "Looking for paper..."
                status_color = (0, 165, 255)

            # Draw status bar
            cv2.rectangle(display, (0, 0), (display.shape[1], 45),
                          (30, 30, 30), -1)
            cv2.putText(display, status_msg, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

            # Draw helper text at the bottom
            h = display.shape[0]
            cv2.rectangle(display, (0, h - 35), (display.shape[1], h),
                          (30, 30, 30), -1)
            cv2.putText(display, "[Space] Force capture  |  [q/Esc] Cancel",
                        (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1)

            cv2.imshow("OCR Scanner", display)

            if captured_image is not None:
                # Show the captured frame briefly
                cv2.imshow("OCR Scanner", display)
                cv2.waitKey(800)
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):      # q or Esc → cancel
                LOGGER.info("OCR scan cancelled by user.")
                cv2.destroyWindow("OCR Scanner")
                return None
            if key == 32:                   # Space → force capture
                LOGGER.info("Force-capturing current frame.")
                if contour is not None:
                    captured_image = _four_point_warp(frame, contour)
                else:
                    captured_image = frame.copy()
                break

    finally:
        cv2.destroyWindow("OCR Scanner")
        if opened_here:
            cap.release()

    if captured_image is None:
        return None

    # Run OCR on the captured, warped image
    text = extract_text(captured_image)
    if not text:
        LOGGER.info("OCR found no text in the captured image.")
        return None

    LOGGER.info("OCR extracted %d characters, %d words.",
                len(text), len(text.split()))
    return text, captured_image


# ═════════════════════════════════════════════════════════════════════════
#  LLaMA summarization (local Ollama)
# ═════════════════════════════════════════════════════════════════════════

def summarize_with_llama(text: str, *,
                         endpoint: str = OLLAMA_URL,
                         model: str = LLAMA_MODEL) -> str:
    """Send OCR text to a local Ollama LLaMA-3 model for summarization."""
    if not text.strip():
        return "No readable text was provided for summarization."

    messages = [
        {
            "role": "system",
            "content": ("You are a helpful assistant that writes short, "
                        "clear summaries suitable for reading aloud."),
        },
        {
            "role": "user",
            "content": (
                "Read the following OCR-extracted text and provide a concise "
                "summary in 3–4 sentences:\n\n" + text
            ),
        },
    ]

    try:
        resp = requests.post(
            endpoint,
            json={"model": model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        reply = resp.json().get("message", {}).get("content", "").strip()
        return reply or "The model returned an empty summary."
    except Exception as exc:
        LOGGER.error("LLaMA summarization failed: %s", exc)
        return f"Could not generate summary: {exc}"


# ═════════════════════════════════════════════════════════════════════════
#  OCR from a static image file
# ═════════════════════════════════════════════════════════════════════════

def scan_from_file(image_path: str) -> str | None:
    """Run OCR on an image file and return extracted text."""
    img = cv2.imread(image_path)
    if img is None:
        LOGGER.warning("Could not read image: %s", image_path)
        return None
    return extract_text(img)


# ═════════════════════════════════════════════════════════════════════════
#  Standalone CLI
# ═════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live-camera OCR with auto paper detection + LLaMA summary."
    )
    p.add_argument("--image", "-i", default=None,
                   help="Path to an image file (skips live camera).")
    p.add_argument("--camera", "-c", type=int, default=0,
                   help="Camera index (default: 0).")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip the LLaMA summarization step.")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args()

    # ── Static image mode ────────────────────────────────────────────
    if args.image:
        text = scan_from_file(args.image)
        if not text:
            print("No text detected in the image.")
            return
        print("\n=== Extracted OCR Text ===")
        print(text)
        if not args.no_summary:
            print("\n=== LLaMA Summary ===")
            summary = summarize_with_llama(text)
            print(summary)
        return

    # ── Live camera mode ─────────────────────────────────────────────
    print("Starting live camera OCR...")
    print("Hold a paper/document in front of the camera.")
    print("Press [Space] to force-capture, [q/Esc] to cancel.\n")

    result = scan_paper_live(camera_index=args.camera)
    if result is None:
        print("No text was captured.")
        return

    text, warped = result

    print("\n=== Extracted OCR Text ===")
    print(text)

    if not args.no_summary:
        print("\n=== LLaMA Summary ===")
        summary = summarize_with_llama(text)
        print(summary)

    # Show the warped paper image
    cv2.imshow("Captured Paper", warped)
    print("\nPress any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()