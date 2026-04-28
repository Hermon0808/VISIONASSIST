"""
metrics.py — Production-quality evaluation script for a multimodal AI pipeline.

Evaluates:
  1. Object Detection  (YOLOv8 / ultralytics)
  2. OCR               (pytesseract)
  3. LLM Summarization (LLaMA-3 via Ollama REST API)

All core metrics are implemented from scratch — no external metric libraries.

Usage examples:
  python metrics.py --task detection --images ./test_images --labels ./labels
  python metrics.py --task ocr --images ./ocr_images --texts ./ocr_labels
  python metrics.py --task summarization --inputs ./docs.txt --references ./refs.txt
  python metrics.py --task all --images ./test_images --labels ./labels --texts ./ocr_labels --inputs ./docs --references ./refs --graph
"""

import os
import re
import sys
import glob
import time
import argparse
import string
from collections import defaultdict

import cv2
import numpy as np
import requests
import matplotlib
matplotlib.use("Qt5Agg")       # interactive backend — shows windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Utility: Levenshtein distance (edit distance) — used by OCR evaluator
# ---------------------------------------------------------------------------

def levenshtein_distance(s: str, t: str) -> int:
    """Compute minimum edit distance between two strings (DP, O(mn) time/space)."""
    n, m = len(s), len(t)
    # dp[i][j] = edit distance between s[:i] and t[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]


# ---------------------------------------------------------------------------
# Utility: Longest Common Subsequence length — used by ROUGE-L
# ---------------------------------------------------------------------------

def lcs_length(x: list, y: list) -> int:
    """Return the length of the longest common subsequence of two lists."""
    n, m = len(x), len(y)
    # Space-optimised: only keep two rows
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)
    return prev[m]


# ---------------------------------------------------------------------------
# Utility: IoU between two bounding boxes (YOLO normalised format)
# ---------------------------------------------------------------------------

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Union for two boxes given in
    YOLO format: [x_center, y_center, width, height] (normalised 0-1).
    """
    # Convert to corner format [x1, y1, x2, y2]
    ax1 = box_a[0] - box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay2 = box_a[1] + box_a[3] / 2

    bx1 = box_b[0] - box_b[2] / 2
    by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2
    by2 = box_b[1] + box_b[3] / 2

    # Intersection rectangle
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ═══════════════════════════════════════════════════════════════════════════
# 1. Object Detection Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class ObjectDetectionEvaluator:
    """
    Evaluates YOLOv8 object-detection predictions against YOLO-format
    ground truth labels.

    Metrics produced:
      • Per-class and overall Precision, Recall
      • mAP@0.5 (mean Average Precision at IoU ≥ 0.5)
      • Average inference latency per image
    """

    IOU_THRESHOLD = 0.5

    def __init__(self, images_dir: str, labels_dir: str, model_path: str = "yolov8n.pt"):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.model_path = model_path

        # Populated by load_data / evaluate
        self.image_paths: list = []
        self.results: dict = {}
        self.timings: list = []
        # Confusion matrix data: list of (gt_class, pred_class) pairs
        self.confusion_pairs: list = []
        self.class_names: dict = {}   # id -> name from YOLO model

    # ------------------------------------------------------------------ IO
    def load_data(self):
        """Discover test images and verify that matching label files exist."""
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
        for ext in extensions:
            self.image_paths.extend(
                glob.glob(os.path.join(self.images_dir, ext))
            )
        self.image_paths.sort()

        if not self.image_paths:
            print(f"[WARN] No images found in {self.images_dir}")
            return

        # Sanity-check labels
        missing = []
        for img_path in self.image_paths:
            lbl = self._label_path_for(img_path)
            if not os.path.isfile(lbl):
                missing.append(lbl)
        if missing:
            print(f"[WARN] {len(missing)} label file(s) missing — they will be treated as 'no ground truth'.")

    def _label_path_for(self, image_path: str) -> str:
        """Return the expected label .txt path for a given image."""
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.labels_dir, f"{stem}.txt")

    @staticmethod
    def _parse_yolo_label(label_path: str) -> list:
        """
        Parse a YOLO-format label file.
        Each line: class_id x_center y_center width height
        Returns list of (class_id: int, box: np.ndarray[4]).
        """
        entries = []
        if not os.path.isfile(label_path):
            return entries
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                box = np.array([float(x) for x in parts[1:5]])
                entries.append((cls_id, box))
        return entries

    # ------------------------------------------------------------- Evaluate
    def evaluate(self):
        """
        Run YOLOv8 inference on every test image, match predictions to
        ground truth via IoU, and compute detection metrics.
        """
        from ultralytics import YOLO  # imported here to keep top-level light

        model = YOLO(self.model_path)

        # Accumulators keyed by class_id
        tp_per_class = defaultdict(int)
        fp_per_class = defaultdict(int)
        fn_per_class = defaultdict(int)
        # For mAP: collect (confidence, is_tp) per class
        det_records = defaultdict(list)  # class -> [(conf, tp_flag)]
        n_gt_per_class = defaultdict(int)

        for img_path in self.image_paths:
            gt_entries = self._parse_yolo_label(self._label_path_for(img_path))

            # --- Inference with timing ---
            t0 = time.perf_counter()
            preds = model(img_path, verbose=False)[0]
            t1 = time.perf_counter()
            self.timings.append(t1 - t0)

            # Build GT box list; track which GT boxes are matched
            gt_boxes = {}  # idx -> (cls, box)
            for idx, (cls, box) in enumerate(gt_entries):
                gt_boxes[idx] = (cls, box)
                n_gt_per_class[cls] += 1

            matched_gt = set()

            # ---- Collect predictions ----
            boxes_xyxy = preds.boxes.xyxy.cpu().numpy()   # [N, 4]
            confs      = preds.boxes.conf.cpu().numpy()    # [N]
            cls_ids    = preds.boxes.cls.cpu().numpy().astype(int)  # [N]

            # Grab class names from the model (only once)
            if not self.class_names and hasattr(model, 'names'):
                self.class_names = dict(model.names)

            # Image dimensions for normalisation
            img_h, img_w = preds.orig_shape[:2]

            # Sort predictions by confidence (descending) for correct AP calc
            order = np.argsort(-confs)

            for pi in order:
                pred_cls = int(cls_ids[pi])
                conf     = float(confs[pi])

                # Convert xyxy → YOLO normalised for IoU function
                px1, py1, px2, py2 = boxes_xyxy[pi]
                pred_box = np.array([
                    ((px1 + px2) / 2) / img_w,
                    ((py1 + py2) / 2) / img_h,
                    (px2 - px1) / img_w,
                    (py2 - py1) / img_h,
                ])

                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, (gt_cls, gt_box) in gt_boxes.items():
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.IOU_THRESHOLD and best_gt_idx >= 0:
                    gt_matched_cls = gt_boxes[best_gt_idx][0]
                    tp_per_class[pred_cls] += 1
                    matched_gt.add(best_gt_idx)
                    det_records[pred_cls].append((conf, True))
                    # Confusion matrix: record (ground_truth, prediction)
                    self.confusion_pairs.append((gt_matched_cls, pred_cls))
                else:
                    fp_per_class[pred_cls] += 1
                    det_records[pred_cls].append((conf, False))
                    # False positive: predicted class with no GT match
                    self.confusion_pairs.append((-1, pred_cls))  # -1 = background

            # Any unmatched GT boxes are false negatives
            for gt_idx, (gt_cls, _) in gt_boxes.items():
                if gt_idx not in matched_gt:
                    fn_per_class[gt_cls] += 1
                    # Missed GT: should have been detected
                    self.confusion_pairs.append((gt_cls, -1))  # -1 = background

        # ---- Aggregate metrics ----
        all_classes = sorted(
            set(tp_per_class) | set(fp_per_class) | set(fn_per_class)
        )

        per_class_metrics = {}
        ap_values = []

        for cls in all_classes:
            tp = tp_per_class[cls]
            fp = fp_per_class[cls]
            fn = fn_per_class[cls]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class_metrics[cls] = {"precision": prec, "recall": rec}

            # AP for this class (all-point interpolation)
            records = sorted(det_records[cls], key=lambda x: -x[0])
            cum_tp = 0
            cum_fp = 0
            precs = []
            recs  = []
            n_gt  = n_gt_per_class[cls]
            for _, is_tp in records:
                if is_tp:
                    cum_tp += 1
                else:
                    cum_fp += 1
                precs.append(cum_tp / (cum_tp + cum_fp))
                recs.append(cum_tp / n_gt if n_gt > 0 else 0.0)

            # Interpolate precision at 101 recall levels (COCO-style)
            ap = 0.0
            if recs:
                for t in np.linspace(0, 1, 101):
                    p_interp = 0.0
                    for p, r in zip(precs, recs):
                        if r >= t:
                            p_interp = max(p_interp, p)
                    ap += p_interp
                ap /= 101
            ap_values.append(ap)
            per_class_metrics[cls]["ap"] = ap

        total_tp = sum(tp_per_class.values())
        total_fp = sum(fp_per_class.values())
        total_fn = sum(fn_per_class.values())

        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        mean_ap      = float(np.mean(ap_values)) if ap_values else 0.0

        self.results = {
            "per_class": per_class_metrics,
            "precision": overall_prec,
            "recall": overall_rec,
            "mAP@0.5": mean_ap,
            "avg_latency_s": float(np.mean(self.timings)) if self.timings else 0.0,
            "num_images": len(self.image_paths),
        }

    # -------------------------------------------------------------- Report
    def print_results(self):
        """Print a clean summary of detection evaluation results."""
        r = self.results
        if not r:
            print("[INFO] No detection results to display. Run evaluate() first.")
            return

        print("\n=== OBJECT DETECTION ===")
        print(f"Images evaluated : {r['num_images']}")
        print(f"Precision        : {r['precision']:.4f}")
        print(f"Recall           : {r['recall']:.4f}")
        print(f"mAP@0.5          : {r['mAP@0.5']:.4f}")
        print(f"Avg latency/img  : {r['avg_latency_s']*1000:.1f} ms")

        if r["per_class"]:
            print("\n  Per-class breakdown:")
            for cls_id in sorted(r["per_class"]):
                m = r["per_class"][cls_id]
                print(f"    Class {cls_id:>3d}  —  P={m['precision']:.3f}  R={m['recall']:.3f}  AP={m['ap']:.3f}")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. OCR Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class OCREvaluator:
    """
    Evaluates pytesseract OCR output against ground truth text.

    Metrics produced:
      • Character Error Rate (CER)
      • Word Error Rate (WER)
      • Average OCR latency per image
    """

    def __init__(self, images_dir: str, texts_dir: str):
        self.images_dir = images_dir
        self.texts_dir = texts_dir

        self.pairs: list = []  # [(image_path, gt_text_path)]
        self.results: dict = {}
        self.timings: list = []

    # ------------------------------------------------------------------ IO
    def load_data(self):
        """Match test images to their ground truth .txt files."""
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(self.images_dir, ext)))
        image_paths.sort()

        if not image_paths:
            print(f"[WARN] No images found in {self.images_dir}")
            return

        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.texts_dir, f"{stem}.txt")
            if os.path.isfile(txt_path):
                self.pairs.append((img_path, txt_path))
            else:
                print(f"[WARN] Missing ground truth for {os.path.basename(img_path)}, skipping.")

        if not self.pairs:
            print("[WARN] No valid image–text pairs found.")

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Lowercase, strip, collapse whitespace, remove punctuation."""
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    # ------------------------------------------------------------- Evaluate
    def evaluate(self):
        """Run pytesseract on each image and compute CER / WER."""
        import pytesseract  # imported here to keep top-level lightweight

        total_cer_num = 0
        total_cer_den = 0
        total_wer_num = 0
        total_wer_den = 0

        for img_path, txt_path in self.pairs:
            with open(txt_path, "r", encoding="utf-8") as f:
                gt_raw = f.read()

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read image {img_path}, skipping.")
                continue

            # Convert BGR → RGB for pytesseract
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            pred_raw = pytesseract.image_to_string(img_rgb)
            t1 = time.perf_counter()
            self.timings.append(t1 - t0)

            gt   = self._normalise_text(gt_raw)
            pred = self._normalise_text(pred_raw)

            # ---- CER (character level) ----
            char_dist = levenshtein_distance(pred, gt)
            total_cer_num += char_dist
            total_cer_den += max(len(gt), 1)  # avoid division by zero

            # ---- WER (word level) ----
            gt_words   = gt.split()
            pred_words = pred.split()
            word_dist  = levenshtein_distance(pred_words, gt_words)
            total_wer_num += word_dist
            total_wer_den += max(len(gt_words), 1)

        cer = total_cer_num / total_cer_den if total_cer_den > 0 else 0.0
        wer = total_wer_num / total_wer_den if total_wer_den > 0 else 0.0

        self.results = {
            "CER": cer,
            "WER": wer,
            "num_samples": len(self.pairs),
            "avg_latency_s": float(np.mean(self.timings)) if self.timings else 0.0,
        }

    # -------------------------------------------------------------- Report
    def print_results(self):
        """Print a clean summary of OCR evaluation results."""
        r = self.results
        if not r:
            print("[INFO] No OCR results to display. Run evaluate() first.")
            return

        print("\n=== OCR ===")
        print(f"Samples evaluated : {r['num_samples']}")
        print(f"CER               : {r['CER']:.4f}")
        print(f"WER               : {r['WER']:.4f}")
        print(f"Avg latency/img   : {r['avg_latency_s']*1000:.1f} ms")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Summarization Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class SummarizationEvaluator:
    """
    Evaluates LLaMA-3 summaries (generated via Ollama) against
    reference summaries using manually implemented ROUGE metrics.

    Metrics produced:
      • ROUGE-1 (unigram overlap)
      • ROUGE-2 (bigram overlap)
      • ROUGE-L (longest common subsequence)
      • Average generation latency per document
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3"

    def __init__(self, inputs_path: str, references_path: str):
        self.inputs_path = inputs_path
        self.references_path = references_path

        self.documents: list = []
        self.references: list = []
        self.results: dict = {}
        self.timings: list = []

    # ------------------------------------------------------------------ IO
    def load_data(self):
        """
        Load source documents and reference summaries.

        Supported formats:
          • A single .txt file with entries separated by blank lines
          • A directory of .txt files (one per document)
        """
        self.documents  = self._load_texts(self.inputs_path)
        self.references = self._load_texts(self.references_path)

        if len(self.documents) != len(self.references):
            print(
                f"[WARN] Document count ({len(self.documents)}) ≠ "
                f"reference count ({len(self.references)}). "
                "Truncating to the shorter list."
            )
            n = min(len(self.documents), len(self.references))
            self.documents  = self.documents[:n]
            self.references = self.references[:n]

        if not self.documents:
            print("[WARN] No documents loaded.")

    @staticmethod
    def _load_texts(path: str) -> list:
        """Load texts from a file (blank-line separated) or a directory."""
        texts = []
        if os.path.isdir(path):
            for fp in sorted(glob.glob(os.path.join(path, "*.txt"))):
                with open(fp, "r", encoding="utf-8") as f:
                    texts.append(f.read().strip())
        elif os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Split on double newlines
            chunks = re.split(r"\n\s*\n", content)
            texts = [c.strip() for c in chunks if c.strip()]
        else:
            print(f"[ERROR] Path not found: {path}")
        return texts

    # ----------------------------------------------------- Ollama API call
    def _generate_summary(self, text: str) -> str:
        """Call the local Ollama API to generate a summary with LLaMA-3."""
        prompt = (
            "Summarize the following text concisely in one paragraph:\n\n"
            f"{text}"
        )
        payload = {
            "model": self.MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(self.OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.RequestException as e:
            print(f"[ERROR] Ollama API request failed: {e}")
            return ""

    # --------------------------------------------------- ROUGE helpers
    @staticmethod
    def _tokenize(text: str) -> list:
        """Simple whitespace + punctuation tokeniser; lowercased."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    @staticmethod
    def _ngrams(tokens: list, n: int) -> list:
        """Return list of n-gram tuples."""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    @classmethod
    def _rouge_n(cls, hypothesis: str, reference: str, n: int) -> dict:
        """
        Compute ROUGE-N precision, recall, and F1.
        n=1 → ROUGE-1, n=2 → ROUGE-2.
        """
        hyp_tokens = cls._tokenize(hypothesis)
        ref_tokens = cls._tokenize(reference)

        hyp_ngrams = cls._ngrams(hyp_tokens, n)
        ref_ngrams = cls._ngrams(ref_tokens, n)

        if not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Count overlapping n-grams (handle duplicates with multiset)
        ref_counts = defaultdict(int)
        for ng in ref_ngrams:
            ref_counts[ng] += 1

        hyp_counts = defaultdict(int)
        for ng in hyp_ngrams:
            hyp_counts[ng] += 1

        overlap = 0
        for ng, cnt in hyp_counts.items():
            overlap += min(cnt, ref_counts.get(ng, 0))

        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
        recall    = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    @classmethod
    def _rouge_l(cls, hypothesis: str, reference: str) -> dict:
        """Compute ROUGE-L using the longest common subsequence."""
        hyp_tokens = cls._tokenize(hypothesis)
        ref_tokens = cls._tokenize(reference)

        lcs_len = lcs_length(hyp_tokens, ref_tokens)

        precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
        recall    = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------- Evaluate
    def evaluate(self):
        """
        Generate summaries via Ollama and evaluate against references
        using ROUGE-1, ROUGE-2, ROUGE-L, and length reduction.
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        length_ratios = []   # summary_len / doc_len → lower = more compression

        for idx, (doc, ref) in enumerate(zip(self.documents, self.references)):
            print(f"  Summarising document {idx + 1}/{len(self.documents)} …", end="\r")

            t0 = time.perf_counter()
            generated = self._generate_summary(doc)
            t1 = time.perf_counter()
            self.timings.append(t1 - t0)

            if not generated:
                rouge1_scores.append({"precision": 0, "recall": 0, "f1": 0})
                rouge2_scores.append({"precision": 0, "recall": 0, "f1": 0})
                rougeL_scores.append({"precision": 0, "recall": 0, "f1": 0})
                continue

            rouge1_scores.append(self._rouge_n(generated, ref, 1))
            rouge2_scores.append(self._rouge_n(generated, ref, 2))
            rougeL_scores.append(self._rouge_l(generated, ref))

            # Length reduction: 1 - (summary_words / doc_words)
            doc_words = len(doc.split())
            sum_words = len(generated.split())
            if doc_words > 0:
                length_ratios.append(1.0 - (sum_words / doc_words))

        def _avg(scores: list, key: str) -> float:
            if not scores:
                return 0.0
            return float(np.mean([s[key] for s in scores]))

        self.results = {
            "ROUGE-1": {
                "precision": _avg(rouge1_scores, "precision"),
                "recall":    _avg(rouge1_scores, "recall"),
                "f1":        _avg(rouge1_scores, "f1"),
            },
            "ROUGE-2": {
                "precision": _avg(rouge2_scores, "precision"),
                "recall":    _avg(rouge2_scores, "recall"),
                "f1":        _avg(rouge2_scores, "f1"),
            },
            "ROUGE-L": {
                "precision": _avg(rougeL_scores, "precision"),
                "recall":    _avg(rougeL_scores, "recall"),
                "f1":        _avg(rougeL_scores, "f1"),
            },
            "length_reduction": float(np.mean(length_ratios)) if length_ratios else 0.0,
            "num_samples": len(self.documents),
            "avg_latency_s": float(np.mean(self.timings)) if self.timings else 0.0,
        }

    # -------------------------------------------------------------- Report
    def print_results(self):
        """Print a clean summary of summarisation evaluation results."""
        r = self.results
        if not r:
            print("[INFO] No summarisation results to display. Run evaluate() first.")
            return

        print("\n=== SUMMARIZATION ===")
        print(f"Samples evaluated : {r['num_samples']}")
        for metric_name in ("ROUGE-1", "ROUGE-2", "ROUGE-L"):
            m = r[metric_name]
            print(
                f"{metric_name}  —  "
                f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
            )
        print(f"Length reduction   : {r['length_reduction']:.1%}")
        print(f"Avg latency/doc   : {r['avg_latency_s']*1000:.1f} ms")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Performance Grapher  (structured by evaluation category)
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceGrapher:
    """
    Generates a publication-ready dashboard arranged by evaluation category:

      ┌───────────────────┬──────────────────────┬──────────────────────┬──────────────────┐
      │ Handheld Detection│ Scene-level Detection│   OCR Summarization  │ End-to-end System│
      │  (quality bars)   │  (quality bars)      │   (quality bars)     │  (latency comp.) │
      └───────────────────┴──────────────────────┴──────────────────────┴──────────────────┘
      ┌───────────────────┬──────────────────────┬──────────────────────┬──────────────────┐
      │ Detection Latency │    OCR Latency       │  Summary Latency     │ Combined Latency │
      │  (histogram)      │    (histogram)       │  (histogram)         │  (box plot)      │
      └───────────────────┴──────────────────────┴──────────────────────┴──────────────────┘
    """

    COLOURS = {
        "handheld":      {"bar": "#4C72B0", "hist": "#4C72B0", "edge": "#2B4D7A"},
        "scene":         {"bar": "#C44E52", "hist": "#C44E52", "edge": "#8B2E31"},
        "ocr_summ":      {"bar": "#55A868", "hist": "#55A868", "edge": "#347A48"},
        "endtoend":      {"bar": "#8172B3", "hist": "#8172B3", "edge": "#5B4E8C"},
    }
    BG_COLOUR  = "#F8F9FA"
    TEXT_COLOUR = "#212529"

    def __init__(self, output_path: str = "performance_dashboard.png"):
        self.output_path = output_path
        self.det_results: dict = {}
        self.det_timings: list = []
        self.ocr_results: dict = {}
        self.ocr_timings: list = []
        self.sum_results: dict = {}
        self.sum_timings: list = []

    # -------------------------------------------------------- Data ingestion
    def add_detection(self, evaluator: ObjectDetectionEvaluator) -> None:
        if evaluator.results:
            self.det_results = evaluator.results
            self.det_timings = [t * 1000 for t in evaluator.timings]

    def add_ocr(self, evaluator: OCREvaluator) -> None:
        if evaluator.results:
            self.ocr_results = evaluator.results
            self.ocr_timings = [t * 1000 for t in evaluator.timings]

    def add_summarization(self, evaluator: SummarizationEvaluator) -> None:
        if evaluator.results:
            self.sum_results = evaluator.results
            self.sum_timings = [t * 1000 for t in evaluator.timings]

    # --------------------------------------------------------- Plot helpers
    @staticmethod
    def _style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10,
                     color="#212529")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color="#495057")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color="#495057")
        ax.tick_params(colors="#495057", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#DEE2E6")
        ax.set_facecolor("#FFFFFF")
        for spine in ax.spines.values():
            spine.set_color("#CED4DA")
            spine.set_linewidth(0.7)

    def _plot_bars(self, ax, names, values, colour_key):
        colours = self.COLOURS[colour_key]
        x = np.arange(len(names))
        bars = ax.bar(x, values, width=0.52, color=colours["bar"],
                      edgecolor=colours["edge"], linewidth=1.2, alpha=0.9,
                      zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=colours["edge"])
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
        top = max(values) if values else 1.0
        ax.set_ylim(0, min(1.18, top * 1.3 + 0.05))

    def _plot_hist(self, ax, timings, colour_key, title):
        colours = self.COLOURS[colour_key]
        if timings:
            n_bins = max(5, min(20, len(timings)))
            ax.hist(timings, bins=n_bins, color=colours["hist"],
                    edgecolor=colours["edge"], alpha=0.85, linewidth=1.1,
                    zorder=3)
            mean_v = np.mean(timings)
            med_v  = np.median(timings)
            ax.axvline(mean_v, color="#E63946", ls="--", lw=1.4,
                       label=f"Mean: {mean_v:.1f} ms")
            ax.axvline(med_v, color="#457B9D", ls="-.", lw=1.4,
                       label=f"Median: {med_v:.1f} ms")
            if len(timings) > 1:
                p95 = np.percentile(timings, 95)
                ax.axvline(p95, color="#F4A261", ls=":", lw=1.4,
                           label=f"P95: {p95:.1f} ms")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9,
                      edgecolor="#CED4DA")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#ADB5BD", transform=ax.transAxes)
        self._style_axis(ax, title, xlabel="Latency (ms)", ylabel="Freq.")

    # --------------------------------------------------------- Main generate
    def generate(self) -> str:
        if not any([self.det_results, self.ocr_results, self.sum_results]):
            print("[INFO] No data to graph.")
            return ""

        fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
        fig.patch.set_facecolor(self.BG_COLOUR)
        fig.suptitle("VisionAssist — Model Performance Dashboard",
                     fontsize=18, fontweight="bold", y=1.02,
                     color=self.TEXT_COLOUR)

        # ═══════════ ROW 1: Quality metrics per category ═══════════

        # ── Col 0: Handheld Detection ──
        ax = axes[0, 0]
        if self.det_results:
            r = self.det_results
            names  = ["mAP@0.5", "Precision", "Recall"]
            values = [r["mAP@0.5"], r["precision"], r["recall"]]
            self._plot_bars(ax, names, values, "handheld")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#ADB5BD", transform=ax.transAxes)
        self._style_axis(ax, "Handheld Detection", ylabel="Score")

        # ── Col 1: Scene-level Detection (mAP + OCR CER/WER) ──
        ax = axes[0, 1]
        names, values = [], []
        if self.det_results:
            names.append("mAP@0.5")
            values.append(self.det_results["mAP@0.5"])
        if self.ocr_results:
            names.extend(["CER", "WER"])
            values.extend([self.ocr_results["CER"], self.ocr_results["WER"]])
        if names:
            self._plot_bars(ax, names, values, "scene")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#ADB5BD", transform=ax.transAxes)
        self._style_axis(ax, "Scene-level Detection", ylabel="Score / Rate")

        # ── Col 2: OCR Summarization ──
        ax = axes[0, 2]
        if self.sum_results:
            r = self.sum_results
            names  = ["ROUGE-1 F1", "ROUGE-L F1", "Length Red."]
            values = [r["ROUGE-1"]["f1"], r["ROUGE-L"]["f1"],
                      r["length_reduction"]]
            self._plot_bars(ax, names, values, "ocr_summ")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#ADB5BD", transform=ax.transAxes)
        self._style_axis(ax, "OCR Summarization", ylabel="Score")

        # ── Col 3: End-to-end Avg Latency (comparison bar) ──
        ax = axes[0, 3]
        lat_names, lat_vals, lat_colours = [], [], []
        if self.det_timings:
            lat_names.append("Detection")
            lat_vals.append(np.mean(self.det_timings))
            lat_colours.append(self.COLOURS["handheld"]["bar"])
        if self.ocr_timings:
            lat_names.append("OCR")
            lat_vals.append(np.mean(self.ocr_timings))
            lat_colours.append(self.COLOURS["scene"]["bar"])
        if self.sum_timings:
            lat_names.append("Summarize")
            lat_vals.append(np.mean(self.sum_timings))
            lat_colours.append(self.COLOURS["ocr_summ"]["bar"])
        if lat_vals:
            lat_names.append("Total")
            lat_vals.append(sum(lat_vals))
            lat_colours.append(self.COLOURS["endtoend"]["bar"])
            x = np.arange(len(lat_names))
            bars = ax.bar(x, lat_vals, width=0.52, color=lat_colours,
                          edgecolor="#333", linewidth=1, alpha=0.9, zorder=3)
            for bar, val in zip(bars, lat_vals):
                label = f"{val:.0f} ms" if val < 1000 else f"{val/1000:.2f} s"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(lat_vals) * 0.02,
                        label, ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color="#333")
            ax.set_xticks(x)
            ax.set_xticklabels(lat_names, fontsize=8, rotation=18, ha="right")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#ADB5BD", transform=ax.transAxes)
        self._style_axis(ax, "End-to-end System\nAvg. Latency", ylabel="ms")

        # ═══════════ ROW 2: Latency distributions ═══════════
        self._plot_hist(axes[1, 0], self.det_timings, "handheld",
                        "Latency — Detection")
        self._plot_hist(axes[1, 1], self.ocr_timings, "scene",
                        "Latency — OCR")
        self._plot_hist(axes[1, 2], self.sum_timings, "ocr_summ",
                        "Latency — Summarization")

        # Col 3 Row 2: Combined box plot
        ax = axes[1, 3]
        box_data, box_labels = [], []
        if self.det_timings:
            box_data.append(self.det_timings); box_labels.append("Detection")
        if self.ocr_timings:
            box_data.append(self.ocr_timings); box_labels.append("OCR")
        if self.sum_timings:
            box_data.append(self.sum_timings); box_labels.append("Summarize")
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                           widths=0.45, showmeans=True,
                           meanprops=dict(marker="D", markerfacecolor="#E63946",
                                          markersize=6))
            palette = [self.COLOURS[k]["bar"]
                       for k in ["handheld", "scene", "ocr_summ"]][:len(box_data)]
            for patch, clr in zip(bp["boxes"], palette):
                patch.set_facecolor(clr)
                patch.set_alpha(0.7)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#ADB5BD", transform=ax.transAxes)
        self._style_axis(ax, "Latency — All Components", ylabel="ms")

        # ── Show in window ──
        print("\n[✓] Displaying performance dashboard in window...")
        plt.show(block=False)
        return self.output_path

    # ----------------------------------------------- Confusion matrix plot
    def plot_confusion_matrix(self, det_eval: 'ObjectDetectionEvaluator') -> None:
        """Display a confusion matrix in a separate window."""
        if not det_eval or not det_eval.confusion_pairs:
            print("[INFO] No confusion matrix data available.")
            return

        # Collect all class IDs that appear in confusion pairs
        all_ids = set()
        for gt_c, pr_c in det_eval.confusion_pairs:
            all_ids.add(gt_c)
            all_ids.add(pr_c)
        # Sort: real classes first, then -1 (background) at the end
        real_ids = sorted(i for i in all_ids if i >= 0)
        ordered_ids = real_ids + ([-1] if -1 in all_ids else [])
        n = len(ordered_ids)
        idx_map = {cid: i for i, cid in enumerate(ordered_ids)}

        # Build the matrix
        matrix = np.zeros((n, n), dtype=int)
        for gt_c, pr_c in det_eval.confusion_pairs:
            r = idx_map.get(gt_c, None)
            c = idx_map.get(pr_c, None)
            if r is not None and c is not None:
                matrix[r, c] += 1

        # Labels
        names = det_eval.class_names
        labels = []
        for cid in ordered_ids:
            if cid == -1:
                labels.append("BG")
            elif cid in names:
                lbl = names[cid]
                labels.append(lbl[:12])  # truncate long names
            else:
                labels.append(str(cid))

        # Plot
        fig, ax = plt.subplots(figsize=(max(6, n * 0.6 + 2),
                                        max(5, n * 0.6 + 2)))
        fig.patch.set_facecolor(self.BG_COLOUR)

        cmap = plt.cm.Blues
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Tick labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=7)

        # Cell value annotations
        thresh = matrix.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if val > thresh else "#333")

        ax.set_xlabel("Predicted Class", fontsize=10, color="#495057")
        ax.set_ylabel("Ground Truth Class", fontsize=10, color="#495057")
        ax.set_title("Object Detection — Confusion Matrix (IoU ≥ 0.5)",
                     fontsize=13, fontweight="bold", pad=12, color="#212529")

        fig.tight_layout()
        print("[✓] Displaying confusion matrix in window...")
        plt.show(block=False)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Structured Terminal Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_structured_summary(
    det: ObjectDetectionEvaluator | None,
    ocr: OCREvaluator | None,
    summ: SummarizationEvaluator | None,
) -> None:
    """
    Print the four-category performance summary the user requested:
      1. Handheld Detection  —  mAP@0.5, Precision, Recall
      2. Scene-level Detection  —  mAP@0.5, CER, WER
      3. OCR Summarization  —  ROUGE-1, ROUGE-L, Length reduction
      4. End-to-end System  —  Avg. latency
    """
    SEP = "═" * 60
    THIN = "─" * 60

    print(f"\n{SEP}")
    print("    VisionAssist — Structured Performance Summary")
    print(SEP)

    # ── 1. Handheld Detection ──
    print(f"\n  ┌{'─'*56}┐")
    print(f"  │{'Handheld Detection (YOLOv8)':^56}│")
    print(f"  ├{'─'*56}┤")
    if det and det.results:
        r = det.results
        print(f"  │  {'mAP@0.5':<20}: {r['mAP@0.5']:>10.4f}{'':>24}│")
        print(f"  │  {'Precision':<20}: {r['precision']:>10.4f}{'':>24}│")
        print(f"  │  {'Recall':<20}: {r['recall']:>10.4f}{'':>24}│")
        print(f"  │  {'Avg latency':<20}: {r['avg_latency_s']*1000:>8.1f} ms{'':>22}│")
    else:
        print(f"  │  {'(not evaluated)':^54}│")
    print(f"  └{'─'*56}┘")

    # ── 2. Scene-level Detection ──
    print(f"\n  ┌{'─'*56}┐")
    print(f"  │{'Scene-level Detection':^56}│")
    print(f"  ├{'─'*56}┤")
    if det and det.results:
        print(f"  │  {'mAP@0.5':<20}: {det.results['mAP@0.5']:>10.4f}{'':>24}│")
    else:
        print(f"  │  {'mAP@0.5':<20}: {'N/A':>10}{'':>24}│")
    if ocr and ocr.results:
        print(f"  │  {'CER':<20}: {ocr.results['CER']:>10.4f}{'':>24}│")
        print(f"  │  {'WER':<20}: {ocr.results['WER']:>10.4f}{'':>24}│")
        print(f"  │  {'Avg OCR latency':<20}: {ocr.results['avg_latency_s']*1000:>8.1f} ms{'':>22}│")
    else:
        print(f"  │  {'CER':<20}: {'N/A':>10}{'':>24}│")
        print(f"  │  {'WER':<20}: {'N/A':>10}{'':>24}│")
    print(f"  └{'─'*56}┘")

    # ── 3. OCR Summarization ──
    print(f"\n  ┌{'─'*56}┐")
    print(f"  │{'OCR Summarization (LLaMA-3)':^56}│")
    print(f"  ├{'─'*56}┤")
    if summ and summ.results:
        r = summ.results
        print(f"  │  {'ROUGE-1 F1':<20}: {r['ROUGE-1']['f1']:>10.4f}{'':>24}│")
        print(f"  │  {'ROUGE-L F1':<20}: {r['ROUGE-L']['f1']:>10.4f}{'':>24}│")
        lr = r.get('length_reduction', 0.0)
        print(f"  │  {'Length reduction':<20}: {lr:>9.1%}{'':>24}│")
        print(f"  │  {'Avg latency':<20}: {r['avg_latency_s']*1000:>8.1f} ms{'':>22}│")
    else:
        print(f"  │  {'(not evaluated)':^54}│")
    print(f"  └{'─'*56}┘")

    # ── 4. End-to-end System ──
    print(f"\n  ┌{'─'*56}┐")
    print(f"  │{'End-to-end System':^56}│")
    print(f"  ├{'─'*56}┤")
    latencies = []
    labels    = []
    if det and det.results:
        val = det.results["avg_latency_s"] * 1000
        latencies.append(val)
        labels.append("Detection")
        print(f"  │  {'Detection avg':<20}: {val:>8.1f} ms{'':>22}│")
    if ocr and ocr.results:
        val = ocr.results["avg_latency_s"] * 1000
        latencies.append(val)
        labels.append("OCR")
        print(f"  │  {'OCR avg':<20}: {val:>8.1f} ms{'':>22}│")
    if summ and summ.results:
        val = summ.results["avg_latency_s"] * 1000
        latencies.append(val)
        labels.append("Summarization")
        fmt_val = f"{val:.1f} ms" if val < 1000 else f"{val/1000:.2f} s"
        print(f"  │  {'Summarization avg':<20}: {fmt_val:>10}{'':>22}│")
    if latencies:
        total = sum(latencies)
        fmt_total = f"{total:.1f} ms" if total < 1000 else f"{total/1000:.2f} s"
        print(f"  │{'':>2}{'─'*52}{'':>2}│")
        print(f"  │  {'TOTAL avg latency':<20}: {fmt_total:>10}{'':>22}│")
    else:
        print(f"  │  {'(no latency data)':^54}│")
    print(f"  └{'─'*56}┘")
    print(f"\n{SEP}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal AI Evaluation Script — compute real metrics "
                    "for object detection, OCR, and summarization.\n\n"
                    "Just press Run (no arguments needed) to evaluate everything with graphs!",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["detection", "ocr", "summarization", "all"],
        help="Which evaluation task to run (default: all).",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="./test_images",
        help="Path to folder of test images (default: ./test_images).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="./labels",
        help="Path to folder of YOLO-format ground truth labels (default: ./labels).",
    )
    parser.add_argument(
        "--texts",
        type=str,
        default="./ocr_labels",
        help="Path to folder of ground truth text files (default: ./ocr_labels).",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="./docs",
        help="Path to source documents file/dir (default: ./docs).",
    )
    parser.add_argument(
        "--references",
        type=str,
        default="./refs",
        help="Path to reference summaries file/dir (default: ./refs).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model weights path (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        default=True,
        help="Generate a performance dashboard graph (default: True).",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable the graph output.",
    )
    parser.add_argument(
        "--graph-output",
        type=str,
        default="performance_dashboard.png",
        help="Output path for the performance graph (default: performance_dashboard.png).",
    )
    parser.add_argument(
        "--ocr-images",
        type=str,
        default="./ocr_images",
        help="Path to folder of OCR test images (default: ./ocr_images).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    grapher = PerformanceGrapher(args.graph_output) if (args.graph and not args.no_graph) else None
    tasks_to_run = [args.task] if args.task != "all" else ["detection", "ocr", "summarization"]

    det_eval  = None
    ocr_eval  = None
    summ_eval = None

    # ---- Object Detection ----
    if "detection" in tasks_to_run:
        if not args.images or not args.labels:
            if args.task != "all":
                parser.error("--images and --labels are required for detection task.")
            else:
                print("[SKIP] Detection — missing --images or --labels.")
        else:
            det_eval = ObjectDetectionEvaluator(args.images, args.labels, args.model)
            det_eval.load_data()
            det_eval.evaluate()
            det_eval.print_results()
            if grapher:
                grapher.add_detection(det_eval)

    # ---- OCR ----
    if "ocr" in tasks_to_run:
        ocr_img_dir = args.ocr_images or args.images
        ocr_txt_dir = args.texts
        if not ocr_img_dir or not ocr_txt_dir:
            if args.task != "all":
                parser.error("--images and --texts are required for ocr task.")
            else:
                print("[SKIP] OCR — missing --images/--ocr-images or --texts.")
        else:
            ocr_eval = OCREvaluator(ocr_img_dir, ocr_txt_dir)
            ocr_eval.load_data()
            ocr_eval.evaluate()
            ocr_eval.print_results()
            if grapher:
                grapher.add_ocr(ocr_eval)

    # ---- Summarization ----
    if "summarization" in tasks_to_run:
        if not args.inputs or not args.references:
            if args.task != "all":
                parser.error("--inputs and --references are required for summarization task.")
            else:
                print("[SKIP] Summarization — missing --inputs or --references.")
        else:
            summ_eval = SummarizationEvaluator(args.inputs, args.references)
            summ_eval.load_data()
            summ_eval.evaluate()
            summ_eval.print_results()
            if grapher:
                grapher.add_summarization(summ_eval)

    # ---- Structured terminal summary ----
    print_structured_summary(det_eval, ocr_eval, summ_eval)

    # ---- Generate graph + confusion matrix ----
    if grapher:
        grapher.generate()
        if det_eval:
            grapher.plot_confusion_matrix(det_eval)
        # Keep all windows open until user closes them
        print("\n[INFO] Close all plot windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
