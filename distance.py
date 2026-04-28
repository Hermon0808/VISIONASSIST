#object detection model 
import cv2
import math
import time
import threading
from queue import Queue
from ultralytics import YOLO
import pyttsx3

# ---------------- CONFIG ----------------
CAMERA_INDEX = 1
CONFIDENCE_THRESHOLD = 0.60
SPEAK_COOLDOWN_SECONDS = 6.0
OBJECT_STALE_SECONDS = 1.5
MIN_PRESENCE_SECONDS = 2.0
DISTANCE_CHANGE_THRESHOLD_M = 0.7
MODEL_PATH = "yolov8n.pt"

CAMERA_HFOV_DEG = 62.0

KNOWN_WIDTH_M = {
    "person": 0.45,
    "bottle": 0.07,
    "cup": 0.08,
    "cell phone": 0.075,
    "laptop": 0.35,
    "keyboard": 0.44,
    "book": 0.16,
    "chair": 0.50,
    "tv": 0.90,
    "car": 1.80,
    "bus": 2.50,
    "truck": 2.50,
    "dog": 0.20,
    "cat": 0.18,
}
DEFAULT_WIDTH_M = 0.30


def _estimate_distance_m(class_name: str, box_width_px: int, focal_length_px: float) -> float:
    """Heuristic distance using pinhole model: D = (W * f) / w"""
    if box_width_px <= 1:
        return -1.0
    real_width_m = KNOWN_WIDTH_M.get(class_name, DEFAULT_WIDTH_M)
    return (real_width_m * focal_length_px) / float(box_width_px)


# ================================================================
#  Wrapper class used by voice_assistant_main.py
# ================================================================
class DistanceMode:
    """Estimate distance to objects in the camera frame using YOLOv8.

    Runs continuous detection in a background thread so the camera feed
    is displayed live while the voice assistant keeps listening.
    """

    def __init__(self, cap: cv2.VideoCapture, model_path: str = MODEL_PATH) -> None:
        self._cap = cap
        self._model = YOLO(model_path)
        self._last_spoken_time: dict[str, float] = {}
        self._last_spoken_distance: dict[str, float] = {}
        self._last_seen_time: dict[str, float] = {}
        self._first_seen_time: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_description: str | None = None
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()

    def start(self) -> None:
        """Start the continuous background detection loop."""
        if self._running:
            return
        self._last_spoken_time.clear()
        self._last_spoken_distance.clear()
        self._last_seen_time.clear()
        self._first_seen_time.clear()
        self._running = True
        self._latest_description = None
        self._frame_ready.clear()
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def _detection_loop(self) -> None:
        """Continuously read frames, detect objects, estimate distances, and show annotated video."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            focal_length_px = (w / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))
            annotated = frame.copy()

            nearest: dict[str, dict] = {}
            results = self._model(frame, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    cls_id = int(box.cls[0])
                    class_name = r.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    box_w = max(1, x2 - x1)
                    dist_m = _estimate_distance_m(class_name, box_w, focal_length_px)
                    if dist_m < 0:
                        continue
                    label = f"{class_name} {conf:.2f} | ~{dist_m:.2f} m"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    prev = nearest.get(class_name)
                    if prev is None or dist_m < prev["distance"]:
                        nearest[class_name] = {"distance": dist_m, "confidence": conf}

            cv2.imshow("Distance Mode", annotated)
            cv2.waitKey(1)

            # Build natural-language descriptions for detected objects
            descriptions: list[str] = []
            for class_name, data in nearest.items():
                dist_m = data["distance"]
                if dist_m < 1.0:
                    dist_str = f"about {dist_m * 100:.0f} centimeters"
                else:
                    dist_str = f"approximately {dist_m:.1f} meters"
                descriptions.append(
                    f"There is a {class_name} {dist_str} away."
                )

            with self._lock:
                if descriptions:
                    self._latest_description = " ".join(descriptions)
                else:
                    self._latest_description = None
            self._frame_ready.set()

    def get_distance_description(self) -> str | None:
        """Return the latest distance result (waits up to 3 s for the first frame)."""
        self._frame_ready.wait(timeout=3.0)
        with self._lock:
            return self._latest_description

    def stop(self) -> None:
        """Stop the background loop and close display window."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        try:
            cv2.destroyWindow("Distance Mode")
        except Exception:
            pass


# ================================================================
#  Standalone mode – runs only when executed directly
# ================================================================
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    speech_queue: Queue[str | None] = Queue()
    last_spoken_time: dict[str, float] = {}
    last_spoken_distance: dict[str, float] = {}
    last_seen_time: dict[str, float] = {}
    first_seen_time: dict[str, float] = {}

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
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        focal_length_px = (w / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))
        annotated = frame.copy()
        now = time.time()
        current_objects: dict = {}
        results = model(frame, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                cls_id = int(box.cls[0])
                class_name = r.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_w = max(1, x2 - x1)
                dist_m = _estimate_distance_m(class_name, box_w, focal_length_px)
                if dist_m < 0:
                    continue
                label = f"{class_name} {conf:.2f} | ~{dist_m:.2f} m"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                prev = current_objects.get(class_name)
                if prev is None or dist_m < prev["distance"]:
                    current_objects[class_name] = {"distance": dist_m, "confidence": conf}

        for class_name, data in current_objects.items():
            dist_m = data["distance"]
            previous_seen = last_seen_time.get(class_name, 0.0)
            was_recently_seen = (now - previous_seen) <= OBJECT_STALE_SECONDS
            if (class_name not in first_seen_time) or (not was_recently_seen):
                first_seen_time[class_name] = now
            visible_duration = now - first_seen_time[class_name]
            if visible_duration < MIN_PRESENCE_SECONDS:
                last_seen_time[class_name] = now
                continue
            cooldown_passed = (now - last_spoken_time.get(class_name, 0.0)) >= SPEAK_COOLDOWN_SECONDS
            previous_dist = last_spoken_distance.get(class_name)
            distance_changed = previous_dist is None or abs(dist_m - previous_dist) >= DISTANCE_CHANGE_THRESHOLD_M
            if (not was_recently_seen) or (cooldown_passed and distance_changed):
                speech_queue.put(f"{class_name}, approximately {dist_m:.1f} meters")
                last_spoken_time[class_name] = now
                last_spoken_distance[class_name] = dist_m
            last_seen_time[class_name] = now

        cv2.imshow("Object Detection + Voice + Distance", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    speech_queue.put(None)
    speech_queue.join()
    cap.release()
    cv2.destroyAllWindows()
