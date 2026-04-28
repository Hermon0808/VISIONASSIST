# ==============================================================================
# gemini.py – Continuous Voice Assistant powered by Google Gemini
# ==============================================================================
#
# How to run:
#   1. Install dependencies:
#        pip install SpeechRecognition pyttsx3 pyaudio google-generativeai
#
#   2. Set your Gemini API key (pick ONE method):
#        • Environment variable:  set GEMINI_API_KEY=your-key-here
#        • Or paste it directly into the GEMINI_API_KEY variable below.
#
#   3. Run:
#        python gemini.py
#
# The assistant listens continuously through your microphone. Every response
# from Gemini is spoken aloud via text-to-speech. Say "exit", "quit",
# "goodbye", or "stop" to end the session.
# ==============================================================================

from __future__ import annotations

import os
import re
import logging
import threading
from queue import Queue

import pyttsx3
import speech_recognition as sr
import google.generativeai as genai

# ──────────────────────────── Configuration ───────────────────────────────────

# Paste your key here OR set the GEMINI_API_KEY environment variable.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyCYgX-gjB3nTwrJWhbdZbdFMdAr_LOAPZo")

# Gemini model to use (gemini-2.0-flash is fast and free-tier friendly)
GEMINI_MODEL: str = "gemini-2.0-flash"

# Text-to-speech rate (words per minute)
TTS_RATE: int = 170

# Microphone listen timeout & max phrase length (seconds)
LISTEN_TIMEOUT: int = 5
PHRASE_TIME_LIMIT: int = 10

# Maximum conversation turns kept in memory
MAX_HISTORY_TURNS: int = 20

# Quit phrases – say any of these to end the session
QUIT_PHRASES: list[str] = [
    "quit", "exit", "stop", "goodbye", "bye",
    "shutdown", "shut down", "close", "terminate",
]

# ──────────────────────────── Logging ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("gemini_voice")

# ──────────────────────────── System prompt ───────────────────────────────────

SYSTEM_PROMPT = (
    "You are a friendly, helpful voice assistant powered by Google Gemini. "
    "Keep your answers concise, conversational, and easy to understand when "
    "spoken aloud. Avoid markdown, bullet lists, or code blocks unless the "
    "user explicitly asks for them. Prefer short sentences."
)

# ──────────────────────────── Helpers ─────────────────────────────────────────


def _chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Split long text into speech-friendly chunks at sentence boundaries."""
    if not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        candidate = f"{buf} {s}".strip() if buf else s
        if buf and len(candidate) > max_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = candidate
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _is_quit(text: str) -> bool:
    """Return True if the text matches any quit phrase."""
    t = text.lower().strip()
    return any(q in t for q in QUIT_PHRASES)


# ──────────────────────────── TTS Engine (thread-safe) ────────────────────────


class Speaker:
    """Thread-safe text-to-speech wrapper using pyttsx3.

    pyttsx3 must be created and used from the same thread. This class
    spawns a dedicated worker thread and feeds text into it via a queue.
    """

    def __init__(self, rate: int = TTS_RATE) -> None:
        self._rate = rate
        self._queue: Queue[str | None] = Queue()
        self._ready = threading.Event()
        self._available = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _worker(self) -> None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            self._available = True
        except Exception as exc:
            LOG.warning("TTS unavailable: %s", exc)
            self._ready.set()
            return
        self._ready.set()

        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception as exc:
                LOG.warning("TTS error: %s", exc)
            self._queue.task_done()

    def speak(self, text: str) -> None:
        """Speak *text* aloud. Blocks until all chunks have been spoken."""
        if not text.strip() or not self._available:
            return
        LOG.info("[SPEAK] %s", text)
        for chunk in _chunk_text(text):
            self._queue.put(chunk)
        self._queue.join()  # wait until all chunks are spoken

    def shutdown(self) -> None:
        self._queue.put(None)


# ──────────────────────────── Listener ────────────────────────────────────────


class Listener:
    """Microphone listener using SpeechRecognition + Google Web API."""

    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()

    def listen(self) -> str:
        """Block until a phrase is captured and return its text (or "")."""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                LOG.info("🎤  Listening …")
                audio = self.recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT,
                    phrase_time_limit=PHRASE_TIME_LIMIT,
                )
            text = self.recognizer.recognize_google(audio).strip()
            LOG.info("[HEARD] %s", text)
            return text
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            LOG.warning("Speech recognition error: %s", exc)
            return ""
        except OSError as exc:
            LOG.error("Microphone error: %s", exc)
            return ""


# ──────────────────────────── Gemini Chat ─────────────────────────────────────


class GeminiChat:
    """Manages a multi-turn conversation with Gemini."""

    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL) -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        self._chat = self._model.start_chat(history=[])
        LOG.info("Gemini model '%s' ready.", model_name)

    def ask(self, user_message: str) -> str:
        """Send a message and return Gemini's reply text."""
        try:
            response = self._chat.send_message(user_message)
            reply = response.text.strip()
            if not reply:
                reply = "Sorry, I didn't get a useful response. Could you try again?"
            # Keep history manageable
            if len(self._chat.history) > MAX_HISTORY_TURNS * 2:
                self._chat.history = self._chat.history[-(MAX_HISTORY_TURNS * 2):]
            return reply
        except Exception as exc:
            LOG.error("Gemini API error: %s", exc)
            return f"Sorry, something went wrong while talking to Gemini: {exc}"


# ──────────────────────────── Main Loop ───────────────────────────────────────


def main() -> None:
    # ── Validate API key ──
    api_key = GEMINI_API_KEY
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()
    if not api_key:
        print("ERROR: No Gemini API key provided. Exiting.")
        return

    # ── Init components ──
    speaker = Speaker()
    listener = Listener()
    chat = GeminiChat(api_key=api_key)

    speaker.speak(
        "Hello! I am your Gemini voice assistant. "
        "Ask me anything, and I will always respond with my voice. "
        "Say goodbye or exit whenever you want to stop."
    )

    try:
        while True:
            user_text = listener.listen()
            if not user_text:
                continue  # silence / timeout – keep listening

            # Check for quit
            if _is_quit(user_text):
                speaker.speak("Goodbye! Have a great day.")
                break

            # Get Gemini response and speak it
            speaker.speak("Let me think…")
            reply = chat.ask(user_text)
            speaker.speak(reply)

    except KeyboardInterrupt:
        speaker.speak("Shutting down.")
    finally:
        speaker.shutdown()
        LOG.info("Gemini voice assistant stopped.")


if __name__ == "__main__":
    main()
