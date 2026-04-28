#man file
# ==============================================================================
# voice_assistant_main.py - Continuous Voice Assistant with local Llama-3
# ==============================================================================
#
# How to run:
#   python voice_assistant_main.py
#
# Python dependencies:
#   pip install SpeechRecognition pyttsx3 pyaudio requests opencv-python pytesseract
#   pip install ultralytics "mediapipe>=0.10,<0.11" "numpy<2"
#
# External requirements:
#   1. A local Llama-3 endpoint such as Ollama:
#        ollama pull llama3
#        ollama serve
#   2. Tesseract OCR installed and available on PATH.
#   3. A working microphone, speaker, and webcam.
#
# Global quit commands:
#   quit, exit, stop assistant, goodbye, shutdown
#
# Notes:
#   The assistant listens continuously.
#   It enters handheld, distance, or OCR mode when your spoken request matches
#   those intents, and global quit works from every mode.
#
# ==============================================================================

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Literal

import cv2
import pyttsx3
import requests
import speech_recognition as sr

from ocr import scan_paper_live, summarize_with_llama
from distance import DistanceMode
from handheld import HandheldMode

CAMERA_INDEX = 0
OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "llama3"
MAX_HISTORY_MESSAGES = 12
LISTEN_TIMEOUT = 4
PHRASE_TIME_LIMIT = 8
TTS_RATE = 170
CAMERA_WARMUP_SECONDS = 0.8
AUTO_ANNOUNCE_COOLDOWN = 6.0

Mode = Literal["chat", "handheld", "distance", "ocr"]
OcrAction = Literal["read", "summarize"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("voice_assistant")

SYSTEM_PROMPT = """
You are a helpful, voice-first assistant powered by a local Llama-3 model.
Keep answers concise, natural, and easy to speak aloud.
You can help with:
- General conversation and question answering.
- Handheld object mode, where the camera system describes what the user is holding.
- Distance mode, where the camera system estimates how far visible objects are.
- OCR mode, where paper text is scanned and either read back or summarized.
When tool context is provided, use it directly and do not invent sensor results.
""".strip()

GLOBAL_QUIT_PHRASES = [
    "goodbye",
    "bye",
    "shutdown",
    "shut down",
    "stop assistant",
    "close assistant",
]
MODE_RESET_PHRASES = [
    "quit",
    "exit",
    "stop",
    "cancel",
    "go back",
    "back to chat",
]
HANDHELD_QUIT_PHRASES = [
    "stop handheld",
    "quit handheld",
    "exit handheld",
]
DISTANCE_QUIT_PHRASES = [
    "stop distance",
    "quit distance",
    "exit distance",
]
OCR_QUIT_PHRASES = [
    "stop reading",
    "stop ocr",
    "quit ocr",
    "exit ocr",
]

HANDHELD_KEYWORDS = [
    "what's in my hand",
    "what is in my hand",
    "what am i holding",
    "tell me what i'm holding",
    "tell me what i am holding",
    "in my hand",
    "handheld",
    "hand detection",
    "handheld detection",
    "open handheld",
    "start handheld",
    "handheld mode",
    "detect my hand",
    "hand held",
]
DISTANCE_KEYWORDS = [
    "how far",
    "distance",
    "what's in the frame",
    "what is in the frame",
    "how far am i from",
    "measure distance",
    "check the distance",
    "what is in front of me",
    "object detection",
    "detect objects",
    "open distance",
    "start distance",
    "distance mode",
    "what do you see",
    "objects in frame",
    "what objects",
]
OCR_READ_KEYWORDS = [
    "read this paper",
    "read the paper",
    "read this page",
    "read this text",
    "read the text",
    "read this document",
    "what does this label say",
]
OCR_SUMMARY_KEYWORDS = [
    "summarize this paper",
    "summarize the paper",
    "summarize this document",
    "summarize this page",
    "summary of the paper",
    "give me a summary",
    "provide summary",
]


def keyword_match(text: str, phrases: list[str]) -> bool:
    """Return True when any phrase appears inside the normalized text."""
    normalized = text.lower().strip()
    return any(phrase in normalized for phrase in phrases)


def chunk_text_for_speech(text: str, *, max_chunk_chars: int = 280) -> list[str]:
    """Split long text into speech-friendly chunks."""
    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if buffer and len(candidate) > max_chunk_chars:
            chunks.append(buffer.strip())
            buffer = sentence
        else:
            buffer = candidate
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def detect_chat_intent(text: str) -> tuple[Mode, OcrAction | None]:
    """Map the latest user utterance to chat, handheld, distance, or OCR."""
    if keyword_match(text, OCR_SUMMARY_KEYWORDS):
        return "ocr", "summarize"
    if keyword_match(text, OCR_READ_KEYWORDS):
        return "ocr", "read"
    if keyword_match(text, HANDHELD_KEYWORDS):
        return "handheld", None
    if keyword_match(text, DISTANCE_KEYWORDS):
        return "distance", None
    return "chat", None


class AudioIO:
    """Microphone input and local text-to-speech output.

    pyttsx3 is initialised inside a dedicated worker thread to avoid
    COM / SAPI5 conflicts with the YOLO and MediaPipe background threads
    on Windows.  The main thread puts text into a queue and waits for
    the worker to finish speaking before continuing.
    """

    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self._tts_available = False
        self._tts_queue: Queue[str | None] = Queue()
        self._tts_ready = threading.Event()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()
        self._tts_ready.wait(timeout=5.0)

    def _tts_worker(self) -> None:
        """Background worker: create the pyttsx3 engine *here* and speak."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            self._tts_available = True
        except Exception as exc:
            LOGGER.warning("Text-to-speech is unavailable: %s", exc)
            self._tts_ready.set()
            return
        self._tts_ready.set()

        while True:
            item = self._tts_queue.get()
            if item is None:
                self._tts_queue.task_done()
                break
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception as exc:
                LOGGER.warning("TTS chunk failed: %s", exc)
            self._tts_queue.task_done()

    def listen(self) -> str:
        """Capture a single spoken utterance and return text."""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = self.recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT,
                    phrase_time_limit=PHRASE_TIME_LIMIT,
                )
            heard = self.recognizer.recognize_google(audio).strip()
            LOGGER.info("[heard] %s", heard)
            return heard
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            LOGGER.warning("Speech recognition backend failed: %s", exc)
            return ""
        except OSError as exc:
            LOGGER.error("Microphone is unavailable: %s", exc)
            self.speak("I cannot access the microphone right now.")
            return ""

    def speak(self, text: str) -> None:
        """Speak text aloud.  Blocks until all chunks have been spoken."""
        if not text.strip():
            return
        LOGGER.info("[speak] %s", text)
        if not self._tts_available:
            return
        for chunk in chunk_text_for_speech(text):
            self._tts_queue.put(chunk)
        self._tts_queue.join()


@dataclass
class LlamaClient:
    """Small wrapper around a local Llama-compatible chat endpoint."""

    endpoint_url: str = OLLAMA_URL
    model_name: str = LLAMA_MODEL
    history: list[dict[str, str]] = field(default_factory=list)

    def generate_response(
        self,
        user_message: str,
        *,
        extra_system: str = "",
        include_history: bool = True,
    ) -> str:
        """Send a request to the local model and return the response text."""
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if extra_system:
            messages.append({"role": "system", "content": extra_system})

        if include_history:
            messages.extend(self.history[-MAX_HISTORY_MESSAGES:])

        messages.append({"role": "user", "content": user_message})

        try:
            response = requests.post(
                self.endpoint_url,
                json={"model": self.model_name, "messages": messages, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            reply = response.json().get("message", {}).get("content", "").strip()
            if not reply:
                reply = "I did not get a usable answer from the local model."
        except Exception as exc:
            LOGGER.error("Llama endpoint call failed: %s", exc)
            reply = "Sorry, I cannot reach the local Llama model right now."

        if include_history:
            self.history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": reply},
                ]
            )
            self.history[:] = self.history[-MAX_HISTORY_MESSAGES:]
        return reply


@dataclass
class AssistantState:
    """Tracks the active mode and recent OCR data."""

    current_mode: Mode = "chat"
    ocr_text: str = ""
    ocr_action: OcrAction = "read"


class VoiceAssistant:
    """Always-listening conversational assistant with tool-driven modes."""

    def __init__(self) -> None:
        self.audio = AudioIO()
        self.llm = LlamaClient()
        self.state = AssistantState()
        self.camera = self._open_camera()
        self.handheld_mode: HandheldMode | None = None
        self.distance_mode: DistanceMode | None = None
        self._last_auto_desc: str | None = None
        self._last_auto_time: float = 0.0

    def _open_camera(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            LOGGER.warning("Camera index %s could not be opened.", CAMERA_INDEX)
            return None
        time.sleep(CAMERA_WARMUP_SECONDS)
        return cap

    def ensure_camera(self) -> bool:
        """Return True when visual modes can use a webcam."""
        if self.camera is None:
            self.audio.speak("The camera is not available, so that mode cannot start.")
            return False
        return True

    def stop_all_modes(self) -> None:
        """Stop any active tool mode and reset state to chat."""
        if self.handheld_mode is not None:
            self.handheld_mode.stop()
            self.handheld_mode = None
        if self.distance_mode is not None:
            self.distance_mode.stop()
            self.distance_mode = None
        self.state.current_mode = "chat"
        self.state.ocr_text = ""
        self._last_auto_desc = None
        self._last_auto_time = 0.0

    def should_quit_globally(self, text: str) -> bool:
        """True when the utterance should fully terminate the assistant."""
        return keyword_match(text, GLOBAL_QUIT_PHRASES)

    def should_exit_current_mode(self, text: str) -> bool:
        """True when the utterance should drop back to chat mode."""
        phrases = MODE_RESET_PHRASES.copy()
        if self.state.current_mode == "handheld":
            phrases.extend(HANDHELD_QUIT_PHRASES)
        elif self.state.current_mode == "distance":
            phrases.extend(DISTANCE_QUIT_PHRASES)
        elif self.state.current_mode == "ocr":
            phrases.extend(OCR_QUIT_PHRASES)
        return self.state.current_mode != "chat" and keyword_match(text, phrases)

    def enter_handheld_mode(self) -> None:
        """Initialize and announce handheld mode."""
        if not self.ensure_camera():
            return
        if self.distance_mode is not None:
            self.distance_mode.stop()
            self.distance_mode = None
        if self.handheld_mode is None and self.camera is not None:
            self.handheld_mode = HandheldMode(self.camera)
        if self.handheld_mode is None:
            self.audio.speak("I could not start handheld mode.")
            return
        self.handheld_mode.start()
        self.state.current_mode = "handheld"
        self.audio.speak(
            "Handheld mode is on. Show me what you are holding. Say stop handheld or quit to leave this mode."
        )

    def enter_distance_mode(self) -> None:
        """Initialize and announce distance mode."""
        if not self.ensure_camera():
            return
        if self.handheld_mode is not None:
            self.handheld_mode.stop()
            self.handheld_mode = None
        if self.distance_mode is None and self.camera is not None:
            self.distance_mode = DistanceMode(self.camera)
        if self.distance_mode is None:
            self.audio.speak("I could not start distance mode.")
            return
        self.distance_mode.start()
        self.state.current_mode = "distance"
        self.audio.speak(
            "Distance mode is on. Point the camera toward the object. Say stop distance or quit to leave this mode."
        )

    def enter_ocr_mode(self, action: OcrAction) -> None:
        """Open a live camera preview, auto-detect the paper, run OCR."""
        if not self.ensure_camera():
            return
        self.state.current_mode = "ocr"
        self.state.ocr_action = action
        self.audio.speak(
            "Opening the camera. Hold the paper in front of the camera. "
            "I will capture it automatically when it is fully in frame. "
            "You can also press Space to capture manually."
        )

        # Live camera preview with auto paper detection
        result = scan_paper_live(self.camera)

        if result is None:
            self.audio.speak(
                "I could not read any text from the paper. "
                "Returning to normal chat."
            )
            self.state.current_mode = "chat"
            self.state.ocr_text = ""
            return

        text, _warped_image = result
        self.state.ocr_text = text
        word_count = len(text.split())
        LOGGER.info("OCR extracted %s characters, %s words.",
                    len(text), word_count)
        self.audio.speak(f"I found about {word_count} words on the page.")

        if action == "read":
            self.audio.speak("Here is the text I found.")
            for chunk in chunk_text_for_speech(text):
                self.audio.speak(chunk)
        else:
            self.audio.speak("Generating summary with Llama...")
            summary = summarize_with_llama(text)
            self.audio.speak(summary)

        self.audio.speak(
            "You can ask more questions about this paper, "
            "or say stop reading to leave OCR mode."
        )

    def handle_chat(self, user_text: str) -> None:
        """Handle the default chat mode or trigger a specialized mode."""
        target_mode, ocr_action = detect_chat_intent(user_text)
        if target_mode == "handheld":
            self.enter_handheld_mode()
            if self.state.current_mode == "handheld":
                self.respond_from_handheld(prompt_hint=user_text)
            return
        if target_mode == "distance":
            self.enter_distance_mode()
            if self.state.current_mode == "distance":
                self.respond_from_distance(prompt_hint=user_text)
            return
        if target_mode == "ocr" and ocr_action is not None:
            self.enter_ocr_mode(ocr_action)
            return

        reply = self.llm.generate_response(user_text)
        self.audio.speak(reply)

    def respond_from_handheld(self, *, prompt_hint: str) -> None:
        """Describe the item in hand and answer follow-up questions about it."""
        if self.handheld_mode is None:
            self.audio.speak("Handheld mode is not running.")
            return
        description = self.handheld_mode.describe_object()
        if not description:
            self.audio.speak("I cannot clearly tell what is in your hand yet. Please hold it steady.")
            return
        reply = self.llm.generate_response(
            f"Camera handheld result: {description}\nUser request: {prompt_hint}",
            extra_system="Answer based on the handheld camera detection and keep it brief.",
        )
        self.audio.speak(reply)

    def respond_from_distance(self, *, prompt_hint: str) -> None:
        """Describe distance and object context for the current frame."""
        if self.distance_mode is None:
            self.audio.speak("Distance mode is not running.")
            return
        description = self.distance_mode.get_distance_description()
        if not description:
            self.audio.speak("I do not see a clear object to measure right now.")
            return
        reply = self.llm.generate_response(
            f"Distance vision result: {description}\nUser request: {prompt_hint}",
            extra_system=(
                "Explain the measured distance in natural speech. "
                "If useful, include a simple real-world comparison."
            ),
        )
        self.audio.speak(reply)

    def handle_active_mode(self, user_text: str) -> None:
        """Route user speech to the currently active mode.

        If the utterance matches a *different* mode's keywords, stop the
        current mode and switch seamlessly.
        """
        # --- Check for mode-switching intent ---
        target_mode, ocr_action = detect_chat_intent(user_text)
        if target_mode != "chat" and target_mode != self.state.current_mode:
            self.stop_all_modes()
            self.handle_chat(user_text)
            return

        if self.state.current_mode == "handheld":
            self.respond_from_handheld(prompt_hint=user_text)
            return
        if self.state.current_mode == "distance":
            self.respond_from_distance(prompt_hint=user_text)
            return
        if self.state.current_mode == "ocr":
            if not self.state.ocr_text:
                self.audio.speak("I do not have scanned paper text yet.")
                self.state.current_mode = "chat"
                return
            if "read" in user_text.lower():
                for chunk in chunk_text_for_speech(self.state.ocr_text):
                    self.audio.speak(chunk)
                return
            reply = self.llm.generate_response(
                (
                    "Use the following OCR text to answer the user's question.\n\n"
                    f"OCR text:\n{self.state.ocr_text}\n\n"
                    f"User question: {user_text}"
                ),
                extra_system="Answer only from the scanned document when possible.",
            )
            self.audio.speak(reply)
            return
        self.handle_chat(user_text)

    def run(self) -> None:
        """Start the continuous listening loop."""
        self.audio.speak("Voice assistant ready. I am listening.")
        try:
            while True:
                user_text = self.audio.listen()
                if not user_text:
                    # Auto-announce detections while in a camera mode
                    if self.state.current_mode in ("handheld", "distance"):
                        now = time.time()
                        desc = None
                        if self.state.current_mode == "handheld" and self.handheld_mode:
                            desc = self.handheld_mode.describe_object()
                        elif self.state.current_mode == "distance" and self.distance_mode:
                            desc = self.distance_mode.get_distance_description()
                        # Speak when detection changes OR cooldown expires
                        if desc and (
                            desc != self._last_auto_desc
                            or (now - self._last_auto_time) >= AUTO_ANNOUNCE_COOLDOWN
                        ):
                            self.audio.speak(desc)
                            self._last_auto_desc = desc
                            self._last_auto_time = now
                    continue

                normalized = user_text.lower().strip()

                # If inside an active mode, check for mode-exit first
                if self.should_exit_current_mode(normalized):
                    self.stop_all_modes()
                    self.audio.speak("Back to normal chat mode.")
                    continue

                # Global quit only fires from chat mode (or phrases
                # like "goodbye" / "shutdown" which are never mode-exit)
                if self.should_quit_globally(normalized):
                    self.audio.speak("Goodbye.")
                    break

                if self.state.current_mode == "chat":
                    self.handle_chat(user_text)
                else:
                    self.handle_active_mode(user_text)
        except KeyboardInterrupt:
            self.audio.speak("Shutting down.")
        finally:
            self.stop_all_modes()
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            LOGGER.info("Assistant stopped.")


def main() -> None:
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
