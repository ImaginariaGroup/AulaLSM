from __future__ import annotations
import queue
import threading
from typing import Optional


class TTSEngine:
    """
    Non-blocking text-to-speech using pyttsx3.

    Runs pyttsx3 in a dedicated daemon thread so speak() returns immediately
    and never blocks the Qt main thread or the HandTracker thread.

    Tries to select a Spanish voice automatically; falls back to the
    system default if none is found.
    """

    def __init__(self, rate: int = 140) -> None:
        self._rate  = rate
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="TTSThread"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)  # sentinel → worker exits

    def speak(self, text: str) -> None:
        text = text.strip()
        if text:
            self._queue.put(text)

    # ── worker (runs in separate thread) ─────────────────────────────────────

    def _worker(self) -> None:
        try:
            import pyttsx3
        except ImportError:
            print("[TTS] pyttsx3 not installed — TTS disabled")
            return

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)

        # Prefer Spanish voice
        for voice in engine.getProperty("voices"):
            name = (voice.name or "").lower()
            vid  = (voice.id  or "").lower()
            if "spanish" in name or "es-" in vid or "sabina" in name or "helena" in name:
                engine.setProperty("voice", voice.id)
                print(f"[TTS] Using voice: {voice.name}")
                break

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS] {exc}")
