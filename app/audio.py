from __future__ import annotations

import io
import warnings
import wave

import numpy as np
from PySide6.QtCore import QThread, Signal


def _encode_wav(samples: np.ndarray, samplerate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class AudioCapture(QThread):
    """Continuously capture system loopback audio (or microphone) in chunks."""

    chunk_ready = Signal(bytes)
    status = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        source: str = "loopback",
        chunk_seconds: float = 4.0,
        samplerate: int = 16000,
        silence_rms: float = 0.003,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.chunk_seconds = max(1.0, float(chunk_seconds))
        self.samplerate = int(samplerate)
        self.silence_rms = float(silence_rms)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _open_device(self, sc):
        if self.source == "microphone":
            device = sc.default_microphone()
            if device is None:
                raise RuntimeError("没有找到可用的麦克风")
            return device

        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("没有找到默认扬声器，无法采集系统声音")
        # 蓝牙耳机等设备会同时暴露同名的「麦克风」和「回环」两个端点，
        # get_microphone 按名字匹配时会拿到麦克风（录到的是房间声音，不是系统播放的声音）。
        # 这里必须显式挑 isloopback 的那个。
        for device in sc.all_microphones(include_loopback=True):
            if getattr(device, "isloopback", False) and device.name == speaker.name:
                return device
        device = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        if device is None:
            raise RuntimeError(
                f"无法采集系统声音（{speaker.name}）。请确认 Windows 音频设备可用，"
                "或在设置中改为麦克风模式。"
            )
        return device

    def run(self) -> None:
        try:
            import soundcard as sc
        except Exception as exc:  # pragma: no cover - depends on host
            self.failed.emit(f"加载音频库失败：{exc}")
            return

        try:
            device = self._open_device(sc)
            self.status.emit(f"音频输入：{device.name}")
            warnings.filterwarnings("ignore", message="data discontinuity in recording")

            block_frames = max(1, int(self.samplerate * 0.2))
            chunk_frames = max(block_frames, int(self.samplerate * self.chunk_seconds))
            buffer = []
            buffered = 0

            with device.recorder(
                samplerate=self.samplerate,
                channels=1,
                blocksize=block_frames,
            ) as recorder:
                while self._running:
                    data = recorder.record(numframes=block_frames)
                    if data is None:
                        continue
                    samples = np.asarray(data, dtype=np.float32).reshape(-1)
                    if samples.size == 0:
                        continue

                    buffer.append(samples)
                    buffered += samples.size
                    if buffered < chunk_frames:
                        continue

                    chunk = np.concatenate(buffer)[:chunk_frames]
                    buffer = []
                    buffered = 0

                    rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                    if rms < self.silence_rms:
                        continue
                    self.chunk_ready.emit(_encode_wav(chunk, self.samplerate))
        except Exception as exc:  # pragma: no cover - depends on host
            self.failed.emit(f"音频采集失败：{exc}")
