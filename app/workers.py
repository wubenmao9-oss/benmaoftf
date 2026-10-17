from __future__ import annotations

import queue
from typing import Optional

from PySide6.QtCore import QThread, Signal

from .llm import OpenAICompatibleClient


class ASRWorker(QThread):
    """Transcribe audio chunks sequentially on a background thread."""

    transcribed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        client: OpenAICompatibleClient,
        language: str = "zh",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.language = language
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._running = True

    def submit(self, wav_bytes: bytes) -> None:
        if self._running:
            self._queue.put(wav_bytes)

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

    def run(self) -> None:
        while self._running:
            try:
                wav_bytes = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if wav_bytes is None:
                break
            try:
                text = self.client.transcribe(wav_bytes, language=self.language)
                if text:
                    self.transcribed.emit(text)
            except Exception as exc:
                self.failed.emit(str(exc))


class AnswerWorker(QThread):
    """Generate interview answers from a transcript snapshot."""

    answer_started = Signal()
    delta = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        client: OpenAICompatibleClient,
        system_prompt: str,
        temperature: float = 0.4,
        max_context_chars: int = 4000,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._running = True

    def submit(self, transcript: str) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._running:
            self._queue.put(transcript)

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

    def run(self) -> None:
        while self._running:
            try:
                transcript = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if transcript is None:
                break
            context = transcript[-self.max_context_chars :]
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "下面是最近的面试对话转写（按时间顺序）：\n\n"
                        f"{context}\n\n"
                        "请判断面试官最近的问题，并直接给出我接下来要说的话。"
                    ),
                },
            ]
            try:
                self.answer_started.emit()
                parts = []
                for token in self.client.stream_chat(
                    messages, temperature=self.temperature
                ):
                    if not self._running:
                        break
                    parts.append(token)
                    self.delta.emit(token)
                self.completed.emit("".join(parts))
            except Exception as exc:
                self.failed.emit(str(exc))


class VisionWorker(QThread):
    """Answer questions from screenshots on a background thread."""

    vision_started = Signal()
    delta = Signal(str)
    completed = Signal()
    failed = Signal(str)

    def __init__(
        self,
        client: OpenAICompatibleClient,
        system_prompt: str,
        temperature: float = 0.2,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._queue: "queue.Queue[Optional[list]]" = queue.Queue()
        self._running = True

    def submit(self, images) -> None:
        if isinstance(images, (bytes, bytearray)):
            images = [bytes(images)]
        else:
            images = [bytes(item) for item in images]
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._running:
            self._queue.put(images)

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

    def run(self) -> None:
        while self._running:
            try:
                images = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if images is None:
                break
            try:
                self.vision_started.emit()
                for token in self.client.stream_vision(
                    images,
                    system_prompt=self.system_prompt,
                    user_text="请解答截图中的题目。如果是多张截图，它们属于同一道题，请合并理解。",
                    temperature=self.temperature,
                ):
                    if not self._running:
                        break
                    self.delta.emit(token)
                self.completed.emit()
            except Exception as exc:
                self.failed.emit(str(exc))


class ScreenshotWorker(QThread):
    """Grab the desktop without blocking the UI thread."""

    ready = Signal(bytes)
    failed = Signal(str)

    def __init__(self, mode: str = "window", region=None, parent=None) -> None:
        super().__init__(parent)
        self.mode = mode
        self.region = region

    def run(self) -> None:
        try:
            from .capture import active_window_rect, grab_screen

            rect = self.region
            if rect is None and self.mode == "window":
                rect = active_window_rect()
            self.ready.emit(grab_screen(rect))
        except Exception as exc:
            self.failed.emit(str(exc))
