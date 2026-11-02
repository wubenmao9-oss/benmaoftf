from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

FRAME_MS = 20
SAMPLE_RATE = 16000


class StreamingASRWorker(QThread):
    """本地流式识别：边听边出字，停顿后提交一句，供主界面立刻生成回答。

    设计要点：
    1. 音频按 20ms 帧连续喂给 sherpa-onnx，不需要等整段录完再上传；
    2. 停顿 250ms 只是「候选结束」，再等 200ms 宽限才真正提交，
       面试官中途改口会并回同一句，不会答半截；
    3. 提交后重置解码流，下一句从零开始。
    """

    partial = Signal(str)
    # 停顿到「候选结束」时先发一次，主界面可以先把生成请求发出去；
    # 对方若又开口会发 resumed，把这次抢跑作废（此时通常还没出字，用户看不到错误答案）
    candidate = Signal(str)
    resumed = Signal()
    utterance = Signal(str)
    ready = Signal()
    failed = Signal(str)

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 2,
        endpoint_silence_ms: int = 180,
        endpoint_grace_ms: int = 120,
        min_speech_ms: int = 200,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model_dir = Path(model_dir)
        self.num_threads = max(1, int(num_threads))
        self.endpoint_silence_ms = max(100, int(endpoint_silence_ms))
        self.endpoint_grace_ms = max(50, int(endpoint_grace_ms))
        self.min_speech_ms = max(100, int(min_speech_ms))
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=600)
        self._running = True
        self._dropped = 0

    # 从音频线程直接调用：只入队，不做解码
    def feed(self, pcm_bytes: bytes, is_speech: bool) -> None:
        if not self._running or not pcm_bytes:
            return
        try:
            self._queue.put_nowait((pcm_bytes, bool(is_speech)))
        except queue.Full:
            # 解码跟不上时丢掉最旧的一帧，保证永远处理最新声音
            try:
                self._queue.get_nowait()
                self._dropped += 1
                self._queue.put_nowait((pcm_bytes, bool(is_speech)))
            except queue.Empty:
                pass

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _model_files(self) -> tuple:
        """按文件名自动识别模型，兼容不同版本的 sherpa-onnx 导出命名。"""
        d = self.model_dir

        def pick(prefix: str) -> str:
            candidates = sorted(d.glob(f"{prefix}*.onnx"))
            if not candidates:
                return str(d / f"{prefix}.onnx")
            for tag in (".int8.", ".fp16."):
                for path in candidates:
                    if tag in path.name:
                        return str(path)
            return str(candidates[0])

        return (
            str(d / "tokens.txt"),
            pick("encoder"),
            pick("decoder"),
            pick("joiner"),
        )

    def run(self) -> None:
        try:
            import sherpa_onnx
        except Exception as exc:  # pragma: no cover - depends on host
            self.failed.emit(f"未安装 sherpa-onnx：{exc}")
            return

        tokens, encoder, decoder, joiner = self._model_files()
        missing = [p for p in (tokens, encoder, decoder, joiner) if not Path(p).exists()]
        if missing:
            self.failed.emit("流式语音模型文件缺失：" + ", ".join(Path(p).name for p in missing))
            return

        try:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=self.num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                max_active_paths=2,
                enable_endpoint_detection=False,
            )
        except Exception as exc:  # pragma: no cover - depends on host
            self.failed.emit(f"加载流式语音模型失败：{exc}")
            return

        stream = recognizer.create_stream()
        self.ready.emit()

        silence_needed = max(1, int(round(self.endpoint_silence_ms / FRAME_MS)))
        grace_needed = max(1, int(round(self.endpoint_grace_ms / FRAME_MS)))
        min_speech_frames = max(1, int(round(self.min_speech_ms / FRAME_MS)))

        last_text = ""
        speech_frames = 0
        silence_frames = 0
        candidate = False
        grace_frames = 0

        while self._running:
            try:
                item = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                break
            pcm_bytes, is_speech = item

            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                continue
            stream.accept_waveform(SAMPLE_RATE, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = (recognizer.get_result(stream) or "").strip()

            if text and text != last_text:
                last_text = text
                self.partial.emit(text)

            # 端点判断：停顿够久先当候选，宽限期内又开口就继续并句
            if is_speech:
                speech_frames += 1
                silence_frames = 0
                if candidate:
                    candidate = False
                    grace_frames = 0
                    self.resumed.emit()
                continue

            silence_frames += 1
            if not candidate:
                if silence_frames >= silence_needed and speech_frames >= min_speech_frames:
                    candidate = True
                    grace_frames = 0
                    if text:
                        self.candidate.emit(text)
                continue

            grace_frames += 1
            if grace_frames < grace_needed:
                continue

            # 提交这一句
            final_text = (recognizer.get_result(stream) or "").strip()
            if final_text:
                self.utterance.emit(final_text)
            recognizer.reset(stream)
            last_text = ""
            speech_frames = 0
            silence_frames = 0
            candidate = False
            grace_frames = 0

        self._queue = queue.Queue(maxsize=600)
