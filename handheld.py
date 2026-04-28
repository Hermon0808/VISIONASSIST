#handheld object detection model 
import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading
from queue import Queue

mp_hands_mod = None
_MEDIAPIPE_IMPORT_ERROR: Exception | None = None
try:
    import mediapipe as mp
    mp_hands_mod = mp.solutions.hands
except Exception as exc:
    _MEDIAPIPE_IMPORT_ERROR = exc

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0
COOLDOWN_SECONDS = 4
CONFIDENCE_THRESHOLD = 0.45
HAND_OVERLAP_THRESHOLD = 0.25
HAND_BOX_MARGIN = 20


def _intersection_ratio(obj_box, hand_box):
    ox1, oy1, ox2, oy2 = obj_box
    hx1, hy1, hx2, hy2 = hand_box
    ix1 = max(ox1, hx1)
    iy1 = max(oy1, hy1)
    ix2 = min(ox2, hx2)
    iy2 = min(oy2, hy2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))
    return inter_area / obj_area


# ================================================================
#  Wrapper class used by voice_assistant_main.py
# ================================================================
class HandheldMode:
    """Detect objects held in the user's hand using YOLOv8 + MediaPipe.

    Runs continuous detection in a background thread so the camera feed
    is displayed live while the voice assistant keeps listening.
    """

    def __init__(self, cap: cv2.VideoCapture, model_path: str = "yolov8n.pt") -> None:
        self._cap = cap
        self._model = YOLO(model_path)
        self._hands = None
        if mp_hands_mod is not None:
            self._hands = mp_hands_mod.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        self._last_spoken: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_description: str | None = None
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()

    def start(self) -> None:
        """Start the continuous background detection loop."""
        if self._running:
            return
        self._last_spoken.clear()
        self._running = True
        self._latest_description = None
        self._frame_ready.clear()
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def _detection_loop(self) -> None:
        """Continuously read frames, detect hand-held objects, and show annotated video."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            annotated = frame.copy()

            # --- hand detection ---
            hand_boxes: list[tuple[int, int, int, int]] = []
            if self._hands is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self._hands.process(rgb)
                if hand_results.multi_hand_landmarks:
                    for hl in hand_results.multi_hand_landmarks:
                        xs = [int(lm.x * w) for lm in hl.landmark]
                        ys = [int(lm.y * h) for lm in hl.landmark]
                        x1 = max(0, min(xs) - HAND_BOX_MARGIN)
                        y1 = max(0, min(ys) - HAND_BOX_MARGIN)
                        x2 = min(w - 1, max(xs) + HAND_BOX_MARGIN)
                        y2 = min(h - 1, max(ys) + HAND_BOX_MARGIN)
                        hand_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)

            # --- object detection (only when a hand is visible) ---
            results = self._model(frame, verbose=False)
            detected: list[str] = []
            if not hand_boxes:
                # No hand detected – draw a hint and skip object matching
                cv2.putText(annotated, "Show your hand...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            else:
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue
                        class_name = r.names[cls_id]
                        if class_name == "person":
                            continue
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                        obj_box = (bx1, by1, bx2, by2)
                        in_hand = any(
                            _intersection_ratio(obj_box, hb) >= HAND_OVERLAP_THRESHOLD
                            for hb in hand_boxes
                        )
                        if not in_hand:
                            continue
                        detected.append(class_name)
                        label = f"{class_name} {conf:.2f}"
                        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        cv2.putText(annotated, label, (bx1, max(20, by1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Handheld Mode", annotated)
            cv2.waitKey(1)

            with self._lock:
                if detected:
                    unique = list(dict.fromkeys(detected))  # preserve order, dedupe
                    if len(unique) == 1:
                        items = f"a {unique[0]}"
                    elif len(unique) == 2:
                        items = f"a {unique[0]} and a {unique[1]}"
                    else:
                        items = ", a ".join(unique[:-1]) + f", and a {unique[-1]}"
                        items = "a " + items
                    self._latest_description = f"I can see {items} in your hand."
                else:
                    self._latest_description = None
            self._frame_ready.set()

    def describe_object(self) -> str | None:
        """Return the latest detection result (waits up to 3 s for the first frame)."""
        self._frame_ready.wait(timeout=3.0)
        with self._lock:
            return self._latest_description

    def stop(self) -> None:
        """Stop the background loop and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
        try:
            cv2.destroyWindow("Handheld Mode")
        except Exception:
            pass


# ================================================================
#  Standalone mode (original behavior) – runs only when executed
#  directly:  python handheld.py
# ================================================================
if __name__ == "__main__":
    if mp_hands_mod is None:
        raise RuntimeError(
            "MediaPipe hand tracking is unavailable in this environment. "
            f"Original import error: {_MEDIAPIPE_IMPORT_ERROR!r}"
        )
    model = YOLO("yolov8n.pt")
    hands = mp_hands_mod.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    speech_queue: Queue[str | None] = Queue()
    last_spoken: dict[str, float] = {}

    def tts_worker():
        while True:
            text = speech_queue.get()
            if text is None:
                speech_queue.task_done()
                break
            engine.say(text)
            engine.runAndWait()
            speech_queue.task_done()

    threading.Thread(target=tts_worker, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = frame.copy()
        current_time = time.time()
        detected_objects: set[str] = set()
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        hand_boxes: list[tuple[int, int, int, int]] = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1 = max(0, min(xs) - HAND_BOX_MARGIN)
                y1 = max(0, min(ys) - HAND_BOX_MARGIN)
                x2 = min(w - 1, max(xs) + HAND_BOX_MARGIN)
                y2 = min(h - 1, max(ys) + HAND_BOX_MARGIN)
                hand_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        results = model(frame)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                class_name = r.names[cls_id]
                if class_name == "person":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                obj_box = (x1, y1, x2, y2)
                in_hand = any(
                    _intersection_ratio(obj_box, hand_box) >= HAND_OVERLAP_THRESHOLD
                    for hand_box in hand_boxes
                )
                if not in_hand:
                    continue
                detected_objects.add(class_name)
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for obj in detected_objects:
            if obj not in last_spoken or current_time - last_spoken[obj] >= COOLDOWN_SECONDS:
                speech_queue.put(f"In your hand: {obj}")
                last_spoken[obj] = current_time
        cv2.imshow("Hand Object Detection with Speech", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    speech_queue.put(None)
    speech_queue.join()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
