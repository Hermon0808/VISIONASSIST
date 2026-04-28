#ocr model 
from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import pytesseract

LOGGER = logging.getLogger(__name__)

DEFAULT_TESSERACT_CONFIG = "--oem 3 --psm 6"


def preprocess_image(image: Any) -> Any:
    """Prepare an image for OCR by enlarging and thresholding it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, thresholded = cv2.threshold(
        resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def run_ocr_on_image(
    image: Any,
    *,
    lang: str = "eng",
    config: str = DEFAULT_TESSERACT_CONFIG,
) -> str | None:
    """Extract text from an in-memory OpenCV image."""
    try:
        processed = preprocess_image(image)
        text = pytesseract.image_to_string(processed, lang=lang, config=config).strip()
        return text or None
    except Exception as exc:
        LOGGER.exception("OCR failed for image input: %s", exc)
        return None


def scan_paper(image_path: str | os.PathLike[str]) -> str | None:
    """Run OCR on an image path and return the extracted text."""
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.warning("Image could not be read for OCR: %s", image_path)
        return None
    return run_ocr_on_image(image)


def scan_paper_from_camera(cap: cv2.VideoCapture) -> str | None:
    """Capture one frame from a camera and run OCR on it."""
    ok, frame = cap.read()
    if not ok:
        LOGGER.warning("Camera frame capture failed during OCR scan.")
        return None

    cv2.imshow("OCR Capture", frame)
    cv2.waitKey(500)
    cv2.destroyWindow("OCR Capture")
    return run_ocr_on_image(frame)


def save_camera_frame(
    cap: cv2.VideoCapture, output_path: str | os.PathLike[str]
) -> Path | None:
    """Capture and save a camera frame for debugging or later OCR use."""
    ok, frame = cap.read()
    if not ok:
        LOGGER.warning("Camera frame capture failed while saving OCR frame.")
        return None

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        LOGGER.warning("Failed to write OCR frame to %s", destination)
        return None
    return destination


def scan_paper_to_tempfile(cap: cv2.VideoCapture) -> tuple[Path, str] | None:
    """Capture a frame, store it temporarily, and return both path and OCR text."""
    temp_path = Path(tempfile.gettempdir()) / "_voice_assistant_ocr_frame.png"
    saved_path = save_camera_frame(cap, temp_path)
    if saved_path is None:
        return None

    text = scan_paper(saved_path)
    if not text:
        return None
    return saved_path, text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR on a single image.")
    parser.add_argument("image", help="Path to an image file")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args()
    text = scan_paper(args.image)
    if text:
        print(text)
    else:
        raise SystemExit("No text detected.")


if __name__ == "__main__":
    main()
