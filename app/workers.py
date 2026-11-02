from __future__ import annotations

import queue
import threading
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
        max_tokens: int = 260,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        # 回答本来就要求 50~85 字，限一个上限，避免模型跑题越写越长
        self.max_tokens = max_tokens
        self._queue: "queue.Queue" = queue.Queue()
        self._running = True
        # 代次：每次取消或提交新问题就 +1，正在跑的流发现代次变了就立刻停
        self._generation = 0

    def cancel(self) -> None:
        """打断当前生成：让正在跑的流立刻停下，半截回答直接丢弃。"""
        self._generation += 1

    def submit(self, transcript: str, focus: str = "", tolerant: bool = False) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._running:
            # 提交新问题时顺带作废正在生成的那一条，避免旧回答继续往外蹦
            self._generation += 1
            self._queue.put((transcript, focus, tolerant))

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

    def run(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                break
            transcript, focus, tolerant = item
            generation = self._generation
            context = transcript[-self.max_context_chars :]
            focus = (focus or "").strip()
            if tolerant:
                # 转写常有错字、重复字和错误断句：让模型按大意回答，不要再说“等待问题”
                user_text = (
                    "下面是最近的面试对话转写（按时间顺序，可能有错字或断句错误）：\n\n"
                    f"{context}\n\n"
                    "【最后这段内容就是要我回答的问题】\n"
                    f"{focus[-300:]}\n\n"
                    "请忽略转写里的重复字、错字和断句错误，按最接近的意思直接给出我可以说出口的回答。"
                    "不要输出“等待问题”，不要解释，直接说答案。"
                )
            elif focus:
                user_text = (
                    "下面是最近的面试对话转写（按时间顺序）：\n\n"
                    f"{context}\n\n"
                    "【只需要回答最后这一个问题】\n"
                    f"{focus[-300:]}\n\n"
                    "前面的内容只当背景，不要重复回答，也不要重复已经说过的自我介绍。"
                    "直接给出我接下来要说的话。"
                )
            else:
                user_text = (
                    "下面是最近的面试对话转写（按时间顺序）：\n\n"
                    f"{context}\n\n"
                    "请判断面试官最近的问题，并直接给出我接下来要说的话。"
                )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ]
            cancelled = False
            try:
                self.answer_started.emit()
                parts = []
                for token in self.client.stream_chat(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    if not self._running:
                        break
                    if generation != self._generation:
                        # 已经被新问题顶掉，静默丢弃这条回答
                        cancelled = True
                        break
                    parts.append(token)
                    self.delta.emit(token)
                if not cancelled:
                    self.completed.emit("".join(parts))
            except Exception as exc:
                if generation == self._generation:
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
