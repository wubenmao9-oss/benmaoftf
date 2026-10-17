from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .config import APP_DIR


class LocalWhisperASR:
    """faster-whisper 本地转写，接口与 OpenAICompatibleClient.transcribe 保持一致。"""

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str = "",
        hf_endpoint: str = "",
    ) -> None:
        self.model_size = model or "small"
        self.device = device or "cpu"
        self.compute_type = compute_type or "int8"
        self.download_root = self._resolve(download_root)
        self.hf_endpoint = hf_endpoint
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self._model = None

    @staticmethod
    def _resolve(path: str) -> str:
        if not path:
            return ""
        folder = Path(path)
        if not folder.is_absolute():
            folder = APP_DIR / folder
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder)

    def _local_model_dir(self) -> Optional[Path]:
        """优先使用手动下载到 download_root 的模型目录，避免联网拉取。"""
        if not self.download_root:
            return None
        root = Path(self.download_root)
        candidates = [root / f"faster-whisper-{self.model_size}", root]
        for folder in candidates:
            if (folder / "model.bin").is_file():
                return folder
        return None

    def _ensure_model(self):
        if self._model is None:
            if self.hf_endpoint:
                os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
            from faster_whisper import WhisperModel

            local = self._local_model_dir()
            if local is not None:
                self._model = WhisperModel(
                    str(local), device=self.device, compute_type=self.compute_type
                )
                return self._model

            kwargs = {"device": self.device, "compute_type": self.compute_type}
            if self.download_root:
                kwargs["download_root"] = self.download_root
            self._model = WhisperModel(self.model_size, **kwargs)
        return self._model

    @staticmethod
    def _decode(wav_bytes: bytes) -> np.ndarray:
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())

        if width != 2:
            raise RuntimeError(f"不支持的采样位宽：{width * 8} bit")

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if rate != 16000 and audio.size:
            target = int(audio.size * 16000 / rate)
            audio = np.interp(
                np.linspace(0, audio.size, target, endpoint=False),
                np.arange(audio.size),
                audio,
            ).astype(np.float32)
        return audio

    def transcribe(
        self,
        wav_bytes: bytes,
        language: str = "zh",
        prompt: Optional[str] = None,
    ) -> str:
        audio = self._decode(wav_bytes)
        if audio.size < 1600:
            return ""

        model = self._ensure_model()
        segments, _ = model.transcribe(
            audio,
            language=language or None,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
        )
        return "".join(segment.text for segment in segments).strip()
