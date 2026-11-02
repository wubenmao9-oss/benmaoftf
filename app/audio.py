from __future__ import annotations

import io
import warnings
import wave

import numpy as np
from PySide6.QtCore import QThread, Signal

try:  # 可选依赖：装上就用 VAD 按停顿切块，没装则回退到固定窗口
    import webrtcvad
except Exception:  # pragma: no cover - depends on host
    webrtcvad = None


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
    # 流式模式：每 20ms 一帧连续输出（pcm, 是否人声），交给本地流式识别
    stream_frame = Signal(bytes, bool)
    status = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        source: str = "loopback",
        chunk_seconds: float = 4.0,
        samplerate: int = 16000,
        silence_rms: float = 0.003,
        vad_enabled: bool = True,
        silence_ms: int = 350,
        min_speech_seconds: float = 0.2,
        max_chunk_seconds: float = 8.0,
        stream_only: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.chunk_seconds = max(0.2, float(chunk_seconds))
        self.samplerate = int(samplerate)
        self.silence_rms = float(silence_rms)
        self.vad_enabled = bool(vad_enabled)
        self.silence_ms = max(100, int(silence_ms))
        self.min_speech_seconds = max(0.05, float(min_speech_seconds))
        self.max_chunk_seconds = max(1.0, float(max_chunk_seconds))
        # 只发 20ms 帧给本地流式识别，不再拼整段送云端
        self.stream_only = bool(stream_only)
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

    def _make_vad(self):
        """创建 VAD 检测器；不可用时返回 None，回退到固定窗口切块。"""
        if not self.vad_enabled:
            return None
        if webrtcvad is None:
            self.status.emit("未安装 webrtcvad，改用固定窗口切块")
            return None
        if self.samplerate not in (8000, 16000, 32000, 48000):
            self.status.emit(f"采样率 {self.samplerate} 不支持 VAD，改用固定窗口切块")
            return None
        try:
            # 2 是激进档位，能滤掉键盘声和空调声，又不至于吞掉正常说话
            return webrtcvad.Vad(2)
        except Exception:
            return None

    def _capture_with_vad(self, recorder, block_frames: int, frame_samples: int, vad) -> None:
        """按停顿切块：面试官一停 350ms 就送去识别，不用等到固定窗口满。"""
        frame_ms = frame_samples * 1000.0 / self.samplerate
        silence_needed = max(1, int(self.silence_ms / frame_ms))
        min_speech_frames = max(1, int(self.min_speech_seconds * 1000 / frame_ms))
        max_frames = max(min_speech_frames, int(self.max_chunk_seconds * 1000 / frame_ms))

        pending = np.zeros(0, dtype=np.float32)
        voiced = []
        pre_roll = []  # 开始说话前留一点音频，避免把字头切掉
        in_speech = False
        speech_frames = 0
        silence_run = 0

        while self._running:
            data = recorder.record(numframes=block_frames)
            if data is None:
                continue
            samples = np.asarray(data, dtype=np.float32).reshape(-1)
            if samples.size == 0:
                continue
            pending = (
                np.concatenate([pending, samples]) if pending.size else samples
            )

            usable = (pending.size // frame_samples) * frame_samples
            if usable == 0:
                continue
            frames = pending[:usable].reshape(-1, frame_samples)
            pending = pending[usable:]

            for frame in frames:
                pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                is_speech = bool(vad.is_speech(pcm, self.samplerate))
                if self.stream_only:
                    self.stream_frame.emit(pcm, is_speech)
                    continue

                if not in_speech:
                    if is_speech:
                        in_speech = True
                        speech_frames = 1
                        silence_run = 0
                        voiced = list(pre_roll)
                        voiced.append(frame)
                        pre_roll = []
                    else:
                        pre_roll.append(frame)
                        if len(pre_roll) > 5:  # 只保留最近 100ms
                            pre_roll.pop(0)
                    continue

                voiced.append(frame)
                if is_speech:
                    speech_frames += 1
                    silence_run = 0
                else:
                    silence_run += 1

                if silence_run < silence_needed and len(voiced) < max_frames:
                    continue

                if speech_frames >= min_speech_frames:
                    audio = np.concatenate(voiced)
                    self.chunk_ready.emit(_encode_wav(audio, self.samplerate))
                in_speech = False
                speech_frames = 0
                silence_run = 0
                voiced = []

    def _capture_by_window(self, recorder, block_frames: int) -> None:
        """固定窗口切块：没有 VAD 时的兜底方案。"""
        chunk_frames = max(block_frames, int(self.samplerate * self.chunk_seconds))
        buffer = []
        buffered = 0

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
            with device.recorder(
                samplerate=self.samplerate,
                channels=1,
                blocksize=block_frames,
            ) as recorder:
                vad = self._make_vad()
                if vad is not None:
                    frame_samples = int(self.samplerate * 0.02)
                    self._capture_with_vad(recorder, block_frames, frame_samples, vad)
                    return
                self._capture_by_window(recorder, block_frames)
        except Exception as exc:  # pragma: no cover - depends on host
            self.failed.emit(f"音频采集失败：{exc}")
