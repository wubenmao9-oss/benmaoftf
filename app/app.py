from __future__ import annotations

import importlib.util
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .asr_local import LocalWhisperASR
from .audio import AudioCapture
from .capture import logical_to_physical
from .config import APP_DIR, load_company, load_knowledge, save_config
from .control import ControlPanel
from .hotkeys import HotkeyManager
from .llm import OpenAICompatibleClient
from .notes import NoteBook
from .overlay import OverlayWindow
from .region import RegionSelector
from .remote import RemoteServer, RemoteState
from .remote_dialog import RemoteDialog
from .streaming_asr import StreamingASRWorker
from .workers import (
    ASRWorker,
    AnswerWorker,
    ScreenshotWorker,
    VisionWorker,
)


WAIT_MARKERS = {"等待问题", "等待问题中", "暂无问题"}

# 面试官一直连续说话时，最多把自动回答顺延这么久，避免永远等不到停顿
MAX_DEFER_SECONDS = 6.0

# 本地粗筛“这句像不像面试官在提问”，命中才调用大模型，省 token 也少误答
QUESTION_HINTS = (
    "?",
    "？",
    "你好",
    "您好",
    "欢迎参加",
    "准备好了",
    "我们开始",
    "开始面试",
    "带领你",
    "什么",
    "怎么",
    "怎样",
    "为什么",
    "为何",
    "如何",
    "是否",
    "能不能",
    "可不可以",
    "有没有",
    "请你",
    "请介绍",
    "介绍一下",
    "介绍",
    "说一下",
    "讲一下",
    "讲一讲",
    "聊一下",
    "聊一聊",
    "说说",
    "谈谈",
    "讲讲",
    "举个例子",
    "举个",
    "举例",
    "区别",
    "你负责",
    "你做过",
    "你用过",
    "你遇到",
    "你怎么",
    "你如何",
    "你觉",
    "你认为",
    "你会",
    "你平时",
    "哪",
    "多少",
    "多久",
    "吗",
    "呢",
    "what",
    "how",
    "why",
    "can you",
    "could you",
    "tell me",
    "introduce",
    "describe",
    "explain",
)


def looks_like_question(text: str) -> bool:
    tail = (text or "").strip()[-160:]
    if not tail:
        return False
    lowered = tail.lower()
    return any(hint in lowered for hint in QUESTION_HINTS)


class CopilotApp(QObject):
    def __init__(self, app: QApplication, config: Dict[str, Any]) -> None:
        super().__init__()
        self.app = app
        self.config = config

        self.overlay = OverlayWindow(config)
        self.panel = ControlPanel(config)

        self.hotkeys: Optional[HotkeyManager] = None
        self.audio: Optional[AudioCapture] = None
        self.asr_worker: Optional[ASRWorker] = None
        self.stream_worker: Optional[StreamingASRWorker] = None
        self.answer_worker: Optional[AnswerWorker] = None
        self.vision_worker: Optional[VisionWorker] = None
        self.shot_worker: Optional[ScreenshotWorker] = None
        self._stale_workers = []

        self.transcript = ""
        self._last_answered_len = 0
        self._answer_submit_len = 0
        self._answer_start_len = 0
        self._answer_defer_start: Optional[float] = None
        self._answering = False
        self._pending_answer = False
        # 模型把这句话判成“等待问题”时，是否已经用过一次“按大意重答”的宽容重试
        self._answer_retried = False
        self._listening = False
        self._panel_was_visible = False
        # 本地流式识别加载失败过一次就回退云端，避免反复重启
        self._stream_fallback = False
        # 候选停顿抢跑：180ms 就把生成请求发出去，200ms 内对方又开口则作废
        self._prestart_active = False
        self._prestart_text = ""
        # 提交后的一小段保护期：对方紧接着补后半句时，撤掉已发出的回答并按整句重答
        self._merge_guard_until = 0.0
        # 手机已接管显示：电脑端只做后台保活，不再弹回答框
        self._phone_active = False
        self._pending_shots: List[bytes] = []
        self.region_selector: Optional[RegionSelector] = None
        # 本次会话的题目（不含回答），随时可以保存成 markdown
        self.notes = NoteBook()
        self._current_focus = ""

        self.remote_state = RemoteState()
        self.remote = RemoteServer(self.remote_state, parent=self)
        self.remote_dialog: Optional[RemoteDialog] = None
        # 信号从 HTTP 工作线程发出，强制队列连接以保证在 GUI 线程执行
        self.remote.screenshot_requested.connect(self.screenshot_answer, Qt.QueuedConnection)
        self.remote.answer_requested.connect(
            lambda: self.request_answer(force=True), Qt.QueuedConnection
        )
        self.remote.listen_requested.connect(self._on_remote_listen, Qt.QueuedConnection)
        self.remote.auto_requested.connect(self._on_remote_auto, Qt.QueuedConnection)
        self.remote.shot_add_requested.connect(self.add_remote_shot, Qt.QueuedConnection)
        self.remote.shot_submit_requested.connect(self.submit_remote_shots, Qt.QueuedConnection)
        self.remote.clear_requested.connect(self.clear_remote, Qt.QueuedConnection)
        self.remote.phone_connected.connect(self._on_phone_connected, Qt.QueuedConnection)

        self._answer_timer = QTimer(self)
        self._answer_timer.setSingleShot(True)
        self._answer_timer.timeout.connect(self.request_answer)

        self._connect_panel()
        self._build_tray()

    def start(self) -> None:
        self._install_hotkeys(self.config["hotkeys"])
        self.panel.show()

    def _connect_panel(self) -> None:
        self.panel.start_listen.connect(self.start_listening)
        self.panel.stop_listen.connect(self.stop_listening)
        self.panel.screenshot.connect(self.screenshot_answer)
        self.panel.answer_now.connect(lambda: self.request_answer(force=True))
        self.panel.overlay_toggle.connect(self.overlay.toggle)
        self.panel.remote_toggle.connect(self.toggle_remote)
        self.panel.save_notes.connect(self.save_notes)
        self.panel.auto_toggled.connect(self._on_panel_auto)
        self.panel.config_saved.connect(self.apply_config)

    def _install_hotkeys(self, hotkeys: Dict[str, str]) -> None:
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.hotkeys = HotkeyManager(hotkeys, self)
        self.hotkeys.toggle_overlay.connect(self.overlay.toggle)
        self.hotkeys.screenshot.connect(self.screenshot_answer)
        self.hotkeys.toggle_listen.connect(self.toggle_listening)
        self.hotkeys.answer_now.connect(lambda: self.request_answer(force=True))
        self.hotkeys.quit.connect(self.shutdown)
        self.hotkeys.failed.connect(self.panel.set_status)
        self.hotkeys.start()

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(self._make_icon(), self)
        self.tray.setToolTip("笨猫面试助手")
        menu = QMenu()
        self._tray_menu = menu

        show_panel = menu.addAction("显示控制台")
        show_panel.triggered.connect(self.panel.show_and_focus)

        toggle_overlay = menu.addAction("显示 / 隐藏悬浮窗")
        toggle_overlay.triggered.connect(self.overlay.toggle)

        toggle_listen = menu.addAction("开始 / 停止监听")
        toggle_listen.triggered.connect(self.toggle_listening)

        screenshot = menu.addAction("截图答题")
        screenshot.triggered.connect(self.screenshot_answer)

        remote_action = menu.addAction("手机远程（扫码）")
        remote_action.triggered.connect(self.toggle_remote)

        menu.addSeparator()
        quit_action = menu.addAction("退出")
        quit_action.triggered.connect(self.shutdown)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    @staticmethod
    def _make_icon() -> QIcon:
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#3b6ef6"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(2, 2, 60, 60, 14, 14)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "AI")
        painter.end()
        return QIcon(pixmap)

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.panel.show_and_focus()

    def _text_client(self) -> Optional[OpenAICompatibleClient]:
        api = self.config["text_api"]
        if not api.get("api_key"):
            return None
        return OpenAICompatibleClient(
            api.get("base_url", ""), api["api_key"], api.get("model", "")
        )

    def _vision_client(self) -> Optional[OpenAICompatibleClient]:
        api = self.config["vision_api"]
        if not api.get("api_key"):
            return None
        return OpenAICompatibleClient(
            api.get("base_url", ""), api["api_key"], api.get("model", "")
        )

    def _asr_client(self):
        local = self.config.get("asr_local") or {}
        if local.get("enabled", True):
            return LocalWhisperASR(
                model=local.get("model", "small"),
                device=local.get("device", "cpu"),
                compute_type=local.get("compute_type", "int8"),
                download_root=local.get("download_root", ""),
                hf_endpoint=local.get("hf_endpoint", ""),
            )
        api = self.config["asr_api"]
        if not api.get("api_key"):
            return None
        return OpenAICompatibleClient(
            api.get("base_url", ""),
            api["api_key"],
            api.get("model", ""),
            timeout=(15.0, 120.0),
        )

    def _practice_mode(self) -> bool:
        """练习模式：麦克风收的是自己扮演面试官的提问，不用再猜“像不像提问”。"""
        return str((self.config.get("audio") or {}).get("source", "loopback")) == "microphone"

    def _min_new_chars(self) -> int:
        """练习模式下允许更短的追问（如“为什么呢”），正常面试保持原阈值。"""
        base = int(self.config["answer"].get("min_new_chars", 8))
        return min(base, 4) if self._practice_mode() else base

    def _system_prompt(self) -> str:
        prompt = self.config["prompts"]["interview"]
        knowledge_cfg = self.config.get("knowledge") or {}
        if not knowledge_cfg.get("enabled", True):
            return prompt

        # 两层资料：个人底盘（所有面试共用）+ 本次公司资料（随公司切换）
        parts: List[str] = []
        text = load_knowledge().strip()
        if text:
            limit = int(knowledge_cfg.get("max_chars", 12000))
            parts.append("【我的真实资料（回答时必须优先使用，不得编造）】\n" + text[:limit])

        company = str(knowledge_cfg.get("company") or "").strip()
        if company:
            company_text = load_company(company).strip()
            if company_text:
                limit_c = int(knowledge_cfg.get("company_max_chars", 2500))
                parts.append(
                    f"【本次面试的公司与岗位资料：{company}"
                    "（面试官问到公司、业务或岗位相关问题时使用；"
                    "资料不足就用保守说法，不要编造公司规模、产品和岗位要求）】\n"
                    + company_text[:limit_c]
                )

        if not parts:
            return prompt
        return prompt + "\n\n" + "\n\n".join(parts)

    def set_status(self, text: str) -> None:
        self.panel.set_status(text)
        self.remote_state.set_status(text)

    # ---- 悬浮窗输出（手机接管时不弹） ----

    def _overlay_begin(self, text: str) -> None:
        if not self._phone_active:
            self.overlay.begin(text)

    def _overlay_error(self, text: str) -> None:
        if not self._phone_active:
            self.overlay.error(text)

    def _overlay_set_status(self, text: str) -> None:
        if not self._phone_active:
            self.overlay.set_status(text)

    def toggle_remote(self) -> None:
        if self.remote.running:
            # 服务已在运行：重新出示二维码，避免误关掉手机端
            self.panel.showNormal()
            self.panel.show()
            self._show_remote_dialog()
        else:
            self.start_remote()

    def start_remote(self) -> None:
        try:
            self.remote.start()
        except OSError as exc:
            self.set_status(f"手机远程启动失败：{exc}")
            return
        self._show_remote_dialog()
        self.set_status("等待手机扫码…")

    def _show_remote_dialog(self) -> None:
        if self.remote_dialog is not None:
            self.remote_dialog.showNormal()
            self.remote_dialog.raise_()
            self.remote_dialog.activateWindow()
            return
        self.remote_dialog = RemoteDialog(self.remote.url, self.panel)
        self.remote_dialog.stopped.connect(self.stop_remote)
        self.remote_dialog.show()
        self.remote_dialog.raise_()

    def stop_remote(self) -> None:
        self.remote.stop()
        self._phone_active = False
        self._pending_shots.clear()
        self.remote_state.set_shots(0)
        if self.remote_dialog is not None:
            dialog, self.remote_dialog = self.remote_dialog, None
            dialog.close_silently()
            dialog.deleteLater()
        self.panel.showNormal()
        self.panel.show()
        self.set_status("手机远程已断开")

    def _on_phone_connected(self) -> None:
        if self.remote_dialog is not None:
            dialog, self.remote_dialog = self.remote_dialog, None
            dialog.close_silently()
            dialog.deleteLater()
        self._phone_active = True
        self.remote_state.set_auto(bool(self.config["answer"].get("auto", True)))
        self.overlay.hide()
        self.panel.showMinimized()
        self.set_status("手机已连接，电脑端已最小化")

    def _on_remote_auto(self, on: bool) -> None:
        self.config["answer"]["auto"] = bool(on)
        self.panel.set_auto(bool(on))
        self.remote_state.set_auto(bool(on))
        self.set_status("自动回答已开启" if on else "自动回答已关闭")

    def _on_panel_auto(self, on: bool) -> None:
        """电脑端勾选自动回答：同步到运行配置与手机端。"""
        self.config["answer"]["auto"] = bool(on)
        self.remote_state.set_auto(bool(on))
        self.set_status("自动回答已开启" if on else "自动回答已关闭")

    def clear_remote(self) -> None:
        self._pending_shots.clear()
        self.remote_state.clear()
        self.set_status("已清空手机端记录")

    def save_notes(self) -> None:
        """把本次会话的题目存成 markdown（只存题目，不存回答）。"""
        try:
            path = self.notes.save()
        except OSError as exc:
            self.set_status(f"保存笔记失败：{exc}")
            return
        if path is None:
            self.set_status("还没有题目，先答题再保存")
            return
        self.set_status(f"题目已保存：{path}")

    def _on_remote_listen(self, on: bool) -> None:
        if on:
            self.start_listening()
        else:
            self.stop_listening()

    def toggle_listening(self) -> None:
        if self._listening:
            self.stop_listening()
        else:
            self.start_listening()

    def _stream_model_files(self, stream_cfg: Dict[str, Any]) -> Optional[List[str]]:
        """返回流式模型必需文件；不齐全就返回 None。"""
        raw_dir = str(stream_cfg.get("model_dir") or "").strip()
        if not raw_dir:
            return None
        model_dir = Path(raw_dir)
        if not model_dir.is_absolute():
            model_dir = APP_DIR / model_dir
        if not model_dir.is_dir():
            return None
        if not (model_dir / "tokens.txt").exists():
            return None

        def pick(prefix: str) -> Optional[str]:
            candidates = sorted(model_dir.glob(f"{prefix}*.onnx"))
            if not candidates:
                return None
            for tag in (".int8.", ".fp16."):
                for path in candidates:
                    if tag in path.name:
                        return str(path)
            return str(candidates[0])

        encoder = pick("encoder")
        decoder = pick("decoder")
        joiner = pick("joiner")
        if not (encoder and decoder and joiner):
            return None
        return [str(model_dir / "tokens.txt"), encoder, decoder, joiner]

    def _stream_available(self, stream_cfg: Dict[str, Any]) -> bool:
        if not bool(stream_cfg.get("enabled", True)):
            return False
        try:
            if importlib.util.find_spec("sherpa_onnx") is None:
                return False
            if importlib.util.find_spec("webrtcvad") is None:
                return False
        except Exception:
            return False
        return self._stream_model_files(stream_cfg) is not None

    def start_listening(self, force_cloud: bool = False) -> None:
        if self._listening:
            return
        if not force_cloud:
            # 用户主动重新开始监听时，给本地流式识别一次重试机会
            self._stream_fallback = False

        audio_cfg = self.config["audio"]
        stream_cfg = self.config.get("asr_stream") or {}
        if not force_cloud and self._stream_available(stream_cfg):
            self._start_stream_listening(audio_cfg, stream_cfg)
            return

        if not force_cloud:
            self.set_status("本地流式模型未就绪，使用云端识别")
        client = self._asr_client()
        if client is None:
            self.set_status("未配置语音转写（本地 Whisper 未启用且无 ASR Key）")
            self.panel.tabs.setCurrentIndex(1)
            return

        self.asr_worker = ASRWorker(
            client, language=self.config["asr_api"].get("language", "zh"), parent=self
        )
        self.asr_worker.transcribed.connect(self.on_transcript)
        self.asr_worker.failed.connect(self.on_asr_failed)
        self.asr_worker.start()

        vad_cfg = audio_cfg.get("vad") or {}
        self.audio = AudioCapture(
            source=audio_cfg.get("source", "loopback"),
            chunk_seconds=float(audio_cfg.get("chunk_seconds", 4.0)),
            samplerate=int(audio_cfg.get("samplerate", 16000)),
            silence_rms=float(audio_cfg.get("silence_rms", 0.003)),
            vad_enabled=bool(vad_cfg.get("enabled", True)),
            silence_ms=int(vad_cfg.get("silence_ms", 350)),
            min_speech_seconds=float(vad_cfg.get("min_speech_seconds", 0.2)),
            max_chunk_seconds=float(vad_cfg.get("max_chunk_seconds", 8.0)),
            parent=self,
        )
        self.audio.chunk_ready.connect(self.on_audio_chunk)
        self.audio.status.connect(self.panel.set_status)
        self.audio.failed.connect(self.on_audio_failed)
        self.audio.start()

        self._listening = True
        self.remote_state.set_listening(True)
        self.panel.set_listening(True)
        if (self.config.get("asr_local") or {}).get("enabled", True):
            self.set_status("正在监听…（本地 Whisper 首次加载较慢）")
        else:
            self.set_status("正在监听…")
        self._overlay_set_status("聆听中")

        # 预热：后台先用同一段系统提示词发一个极短请求，
        # 让服务端把这段前缀缓存下来，正式提问时首字能快一些
        self._warmup_text_model()

    def _start_stream_listening(
        self, audio_cfg: Dict[str, Any], stream_cfg: Dict[str, Any]
    ) -> None:
        """本地流式识别：边听边出字，停顿 250ms + 宽限 200ms 就立刻生成回答。"""
        model_dir = str(stream_cfg.get("model_dir") or "")
        if not Path(model_dir).is_absolute():
            model_dir = str(APP_DIR / model_dir)
        practice = str(audio_cfg.get("source", "loopback")) == "microphone"
        ready_text = (
            "练习模式：正在聆听你的提问（已就绪）"
            if practice
            else "正在监听…（本地流式已就绪）"
        )
        # 练习时说话容易断续，端点放宽一点，避免一句话被切成几段
        silence_ms = int(stream_cfg.get("endpoint_silence_ms", 180))
        grace_ms = int(stream_cfg.get("endpoint_grace_ms", 200))
        if practice:
            silence_ms = max(250, silence_ms)
            grace_ms = max(250, grace_ms)
        worker = StreamingASRWorker(
            model_dir=model_dir,
            num_threads=int(stream_cfg.get("num_threads", 2)),
            endpoint_silence_ms=silence_ms,
            endpoint_grace_ms=grace_ms,
            min_speech_ms=int(stream_cfg.get("min_speech_ms", 200)),
            parent=self,
        )
        worker.partial.connect(self.on_stream_partial)
        worker.candidate.connect(self.on_stream_candidate)
        worker.resumed.connect(self.on_stream_resumed)
        worker.utterance.connect(self.on_stream_utterance)
        worker.ready.connect(lambda: self.set_status(ready_text))
        worker.failed.connect(self.on_stream_failed)
        worker.start()
        self.stream_worker = worker

        vad_cfg = audio_cfg.get("vad") or {}
        self.audio = AudioCapture(
            source=audio_cfg.get("source", "loopback"),
            chunk_seconds=float(audio_cfg.get("chunk_seconds", 4.0)),
            samplerate=int(audio_cfg.get("samplerate", 16000)),
            silence_rms=float(audio_cfg.get("silence_rms", 0.003)),
            vad_enabled=True,
            silence_ms=int(vad_cfg.get("silence_ms", 350)),
            min_speech_seconds=float(vad_cfg.get("min_speech_seconds", 0.2)),
            max_chunk_seconds=float(vad_cfg.get("max_chunk_seconds", 8.0)),
            stream_only=True,
            parent=self,
        )
        # 音频线程直接入队，少走一次 GUI 事件循环
        self.audio.stream_frame.connect(worker.feed, Qt.DirectConnection)
        self.audio.status.connect(self.panel.set_status)
        self.audio.failed.connect(self.on_audio_failed)
        self.audio.start()

        self._listening = True
        self.remote_state.set_listening(True)
        self.panel.set_listening(True)
        if str(audio_cfg.get("source", "loopback")) == "microphone":
            self.set_status("练习模式：正在聆听你的提问…（加载本地流式模型）")
        else:
            self.set_status("正在监听…（加载本地流式模型）")
        self._overlay_set_status("聆听中")
        self._warmup_text_model()

    def _warmup_text_model(self) -> None:
        """后台预热文本模型（只建缓存，不产出任何回答）。"""
        client = self._text_client()
        if client is None:
            return
        prompt = self._system_prompt()

        def run() -> None:
            try:
                for _ in client.stream_chat(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "预热"},
                    ],
                    max_tokens=1,
                ):
                    pass
            except Exception:
                # 预热失败不影响正常回答
                pass

        threading.Thread(target=run, daemon=True).start()

    def stop_listening(self) -> None:
        if not self._listening and self.audio is None and self.asr_worker is None and self.stream_worker is None:
            return
        self._listening = False
        self.remote_state.set_listening(False)
        self._answer_timer.stop()
        self._answer_defer_start = None

        if self.audio is not None:
            worker = self.audio
            self.audio = None
            worker.stop()
            if not worker.wait(2500):
                self._stale_workers.append(worker)
            else:
                worker.deleteLater()

        if self.asr_worker is not None:
            worker = self.asr_worker
            self.asr_worker = None
            worker.stop()
            if not worker.wait(2500):
                self._stale_workers.append(worker)
            else:
                worker.deleteLater()

        if self.stream_worker is not None:
            worker = self.stream_worker
            self.stream_worker = None
            worker.stop()
            if not worker.wait(3000):
                self._stale_workers.append(worker)
            else:
                worker.deleteLater()

        self.panel.set_listening(False)
        self.remote_state.clear_hint()
        self.set_status("已停止监听")

    def on_stream_partial(self, text: str) -> None:
        """本地流式识别的中间结果：只更新状态，不触发回答。"""
        text = (text or "").strip()
        if not text or not self._listening:
            return
        # 刚提交完一句，对方马上又开口：多半是同一句的后半截。
        # 此时回答通常还没出字，撤掉重答，避免把半句话当成完整问题。
        if (
            self._answering
            and not self._prestart_active
            and time.monotonic() < self._merge_guard_until
        ):
            self._merge_guard_until = 0.0
            self._abort_prestart()
            return
        if not self._answering:
            self.set_status("识别中：" + text[-60:])
            self.remote_state.set_hint()

    def on_stream_candidate(self, text: str) -> None:
        """对方停顿 180ms：先把生成请求发出去，但不显示任何内容。

        文本模型首字通常要 0.8s 以上，这 200ms 宽限期内对方改口的话，
        回答还没出字，直接取消即可，不会闪出错误答案。
        """
        text = (text or "").strip()
        if not text or not self._listening:
            return
        if not self.config["answer"].get("auto", True):
            return
        if self._answering or self._prestart_active:
            return
        if len(text) < self._min_new_chars():
            return
        if (
            self.config["answer"].get("question_only", True)
            and not self._practice_mode()
            and not looks_like_question(text[-160:])
        ):
            return
        self._prestart_answer(text)

    def on_stream_resumed(self) -> None:
        """对方在宽限期内又开口：作废刚才的抢跑。"""
        if not self._prestart_active:
            return
        self._prestart_active = False
        self._abort_prestart()

    def _same_utterance(self, a: str, b: str) -> bool:
        def norm(s: str) -> str:
            return "".join(ch for ch in s if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

        na, nb = norm(a), norm(b)
        if not na or not nb:
            return False
        return na == nb or na.startswith(nb) or nb.startswith(na)

    def _abort_prestart(self) -> None:
        if self.answer_worker is not None:
            self.answer_worker.cancel()
        self._answering = False
        self._pending_answer = False
        self._current_focus = ""
        self._prestart_text = ""
        self._merge_guard_until = 0.0
        # 回退到这句之前：后半句接上来时会连同前半句一起重新回答
        self._last_answered_len = self._answer_start_len
        if self._phone_active:
            self.remote_state.drop_answer()
        else:
            self.overlay.discard_current()
            self._overlay_set_status("聆听中")
        self.set_status("正在监听…")

    def _prestart_answer(self, focus: str) -> None:
        if self._answering or self._prestart_active:
            return
        worker = self._ensure_answer_worker()
        if worker is None:
            return
        context = self.transcript.strip()
        context = (context + "\n" + focus).strip() if context else focus
        self._answering = True
        self._prestart_active = True
        self._answer_retried = False
        self._prestart_text = focus
        self._pending_answer = False
        self._answer_start_len = self._last_answered_len
        # 这句还没正式入转写，基线先不动；提交时再对齐
        self._answer_submit_len = len(self.transcript)
        self._current_focus = focus
        self.set_status("正在生成回答…")
        self._overlay_begin("正在生成回答…")
        self.remote_state.begin_answer("interview", focus)
        worker.submit(context, focus)

    def on_stream_utterance(self, text: str) -> None:
        """一句已经等到停顿 + 宽限：立刻答，不再叠加防抖等待。"""
        text = (text or "").strip()
        if not text or not self._listening:
            return

        prestarted = self._prestart_text if self._prestart_active else ""
        if self._prestart_active:
            self._prestart_active = False

        self._append_transcript(text)

        if prestarted:
            if self._same_utterance(prestarted, text):
                # 抢跑时听到的问题和最终文本一致：不重发，直接把基线推到位
                self._prestart_text = ""
                self._last_answered_len = len(self.transcript)
                self._answer_submit_len = len(self.transcript)
                # 保护期：对方若紧接着补后半句，600ms 内撤掉这条回答重答整句
                self._merge_guard_until = time.monotonic() + 0.6
                return
            # 抢跑内容不完整：作废，按最终文本重答
            self._abort_prestart()

        if not self.config["answer"].get("auto", True):
            self.remote_state.clear_hint()
            return
        if not self._has_fresh_question():
            self.remote_state.clear_hint()
            return
        self.request_answer()

    def on_stream_failed(self, message: str) -> None:
        if self._stream_fallback:
            self.set_status(f"本地流式识别不可用：{message}")
            self.stop_listening()
            return
        self._stream_fallback = True
        self.stop_listening()
        self.set_status(f"本地流式识别不可用，已切回云端识别：{message}")
        QTimer.singleShot(0, lambda: self.start_listening(force_cloud=True))

    def _append_transcript(self, text: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        self.transcript += ("\n" if self.transcript else "") + line
        self.panel.append_transcript(line)
        self.remote_state.add_transcript(line)
        if self._answering:
            self.set_status("正在生成回答…")
            self._overlay_set_status("生成中…")
        else:
            self.set_status("正在监听…")
            self._overlay_set_status("聆听中")

        if len(self.transcript) > 20000:
            cut = len(self.transcript) - 12000
            self.transcript = self.transcript[-12000:]
            self._last_answered_len = max(0, self._last_answered_len - cut)

    def on_audio_chunk(self, wav_bytes: bytes) -> None:
        if self.asr_worker is not None:
            self.asr_worker.submit(wav_bytes)
            self.set_status("识别中…")
            # 手机端先给一句垫话，答案首字出来时会自动收起，避免干等
            if not self._answering:
                self.remote_state.set_hint()

    def on_audio_failed(self, message: str) -> None:
        self.set_status(message)
        self.stop_listening()

    def on_asr_failed(self, message: str) -> None:
        self.set_status(f"转写失败：{message}")

    def _has_fresh_question(self) -> bool:
        """上次回答之后是否真的有新的提问，避免同一段转写被重复调用。"""
        new_chars = len(self.transcript) - self._last_answered_len
        if new_chars < self._min_new_chars():
            return False
        if self.config["answer"].get("question_only", True) and not self._practice_mode():
            # 只看上次回答之后的新内容，避免旧问题一直触发
            fresh = self.transcript[self._last_answered_len :].strip()
            if not looks_like_question(fresh[-160:]):
                return False
        return True

    def on_transcript(self, text: str) -> None:
        text = (text or "").strip()
        if not text or not self._listening:
            return
        self._append_transcript(text)

        if not self.config["answer"].get("auto", True):
            self.remote_state.clear_hint()
            return
        delay = max(200, int(float(self.config["answer"].get("debounce_seconds", 1.5)) * 1000))
        now = time.monotonic()
        if self._answer_timer.isActive():
            started = self._answer_defer_start
            if started is None or now - started < MAX_DEFER_SECONDS:
                # 面试官还在继续说：顺延计时，等这半句说完再答
                if started is None:
                    self._answer_defer_start = now
                self._answer_timer.start(delay)
            # 超过上限不再顺延，让计时器到点回答，避免对方一直说就永远不回答
            return
        if not self._has_fresh_question():
            # 这段不是提问：把垫话收起来，免得手机端一直挂着
            self.remote_state.clear_hint()
            return
        self._answer_defer_start = now
        self._answer_timer.start(delay)

    def request_answer(self, force: bool = False, tolerant: bool = False) -> None:
        self._answer_timer.stop()
        self._answer_defer_start = None

        if not force:
            # 生成期间以提交时的长度为基线，只有这之后新说的话才算“新问题”
            baseline = (
                self._answer_submit_len if self._answering else self._last_answered_len
            )
            if len(self.transcript) - baseline < self._min_new_chars():
                if self._answering:
                    self._pending_answer = True
                return

        context = self.transcript.strip()
        if not context:
            if force:
                self._overlay_begin("没有可回答的内容")
                self._overlay_error("还没有转写内容，请先点击“开始监听”。")
                self.set_status("还没有转写内容，请先开始监听")
            return

        if self._answering:
            # 面试官又抛了新内容：不打断正在生成的回答。
            # 直接取消会把上一个问题的答案丢掉，还会让焦点回退、把旧问题重答一遍。
            # 这里排队，等当前这条说完立刻答新的。
            self._pending_answer = True
            return

        worker = self._ensure_answer_worker()
        if worker is None:
            return

        self._answering = True
        self._pending_answer = False
        if not tolerant:
            self._answer_retried = False
        self._answer_start_len = self._last_answered_len
        self._answer_submit_len = len(self.transcript)
        # 只把上次回答之后的新内容当成本次要回答的问题，避免把旧问题再答一遍
        focus = self.transcript[self._last_answered_len :].strip()
        self._current_focus = focus
        # 立刻推进基线：生成期间新说的话留到下一轮，否则会被重复当成问题再答一遍
        self._last_answered_len = self._answer_submit_len
        self.set_status("正在生成回答…")
        self._overlay_begin("正在生成回答…")
        self.remote_state.begin_answer("interview", focus)
        worker.submit(self.transcript, focus, tolerant=tolerant)

    def _ensure_answer_worker(self) -> Optional[AnswerWorker]:
        """按当前配置准备回答线程；没有文本模型时返回 None。"""
        client = self._text_client()
        if client is None:
            self._overlay_begin("未配置文本模型")
            self._overlay_error("请在“模型与音频”标签页填写文本模型的 API Key。")
            if not self._phone_active:
                self.panel.tabs.setCurrentIndex(1)
            self.set_status("未配置文本模型")
            return None

        if self.answer_worker is None:
            self.answer_worker = AnswerWorker(
                client,
                self._system_prompt(),
                temperature=float(self.config["text_api"].get("temperature", 0.4)),
                max_context_chars=int(self.config["answer"].get("max_context_chars", 4000)),
                parent=self,
            )
            self.answer_worker.answer_started.connect(
                lambda: self._overlay_set_status("生成中…")
            )
            self.answer_worker.delta.connect(self.on_answer_delta)
            self.answer_worker.completed.connect(self.on_answer_completed)
            self.answer_worker.failed.connect(self.on_answer_failed)
            self.answer_worker.start()
        else:
            self.answer_worker.client = client
            self.answer_worker.system_prompt = self._system_prompt()
            self.answer_worker.temperature = float(
                self.config["text_api"].get("temperature", 0.4)
            )
            self.answer_worker.max_context_chars = int(
                self.config["answer"].get("max_context_chars", 4000)
            )
        return self.answer_worker

    def on_answer_delta(self, text: str) -> None:
        if not self._phone_active:
            self.overlay.append(text)
        self.remote_state.append_answer(text)

    def on_answer_completed(self, text: str) -> None:
        self._answering = False
        compact = (text or "").strip().strip("。.！!？?（）()")
        if not compact or compact in WAIT_MARKERS:
            # 模型说“等待问题”，但转写里确实有新内容：先用宽容指令重答一次，
            # 还不成再把这句留在原地，等用户按“立即回答”手动重试
            fresh = self.transcript[self._answer_start_len :].strip()
            if not self._answer_retried and len(fresh) >= 4:
                self._answer_retried = True
                self._last_answered_len = self._answer_start_len
                self._pending_answer = False
                self.set_status("转写可能有误，按大意重答一次…")
                self._overlay_begin("转写可能有误，按大意重答一次…")
                QTimer.singleShot(0, lambda: self.request_answer(force=True, tolerant=True))
                return
            # 回退基线：这句没有真正答出来，不能算“已答过”，否则会卡住
            self._last_answered_len = self._answer_start_len
            self.remote_state.drop_answer()
            if not self._phone_active:
                self.overlay.discard_current()
            hint = (self._current_focus or "").strip()[-40:]
            self.set_status(
                f"这句没接住：{hint}（可按“立即回答”重试）"
                if hint
                else "这句没接住，可按“立即回答”重试"
            )
        else:
            # 只标记提交回答时已有的内容；生成期间新说的话要留到下一次触发
            self._last_answered_len = max(self._last_answered_len, self._answer_submit_len)
            self.notes.add_interview(self._current_focus)
            self.remote_state.end_answer()
            if not self._phone_active:
                self.overlay.finish()
            self.set_status("回答完成")
        if self._pending_answer:
            self._pending_answer = False
            # 生成期间来的新内容也要先确认是“新的提问”，避免同一段转写被重复调用
            if self._has_fresh_question():
                QTimer.singleShot(200, self.request_answer)

    def on_answer_failed(self, message: str) -> None:
        self._answering = False
        # 回退基线：这次没答成的问题，下次还能重新提交一次
        self._last_answered_len = self._answer_start_len
        self.remote_state.end_answer(message)
        self._overlay_error(message)
        self.set_status(f"回答失败：{message}")
        if self._pending_answer:
            self._pending_answer = False
            QTimer.singleShot(600, lambda: self.request_answer(force=True))

    def _capture_mode(self) -> str:
        mode = str((self.config.get("capture") or {}).get("mode", "window"))
        return mode if mode in ("window", "region", "full") else "window"

    def screenshot_answer(self) -> None:
        client = self._vision_client()
        if client is None:
            self._overlay_begin("未配置视觉模型")
            self._overlay_error("请在“模型与音频”标签页填写视觉模型的 API Key。")
            if not self._phone_active:
                self.panel.tabs.setCurrentIndex(1)
            self.set_status("未配置视觉模型")
            return

        if self.shot_worker is not None and self.shot_worker.isRunning():
            return
        if self.region_selector is not None:
            return

        # 单张截图会丢弃之前累计的多图。
        self._pending_shots.clear()
        self.remote_state.set_shots(0)

        # 悬浮窗始终藏起来，避免把上一次的答案截进题目里。
        self.overlay.hide()

        mode = self._capture_mode()
        if mode == "region" and self._phone_active:
            # 手机端没法拖框，自动退回活动窗口
            mode = "window"

        if mode == "region":
            self._start_region_capture()
            return

        if self._phone_active:
            # 电脑端已最小化，只做后台截图，不再恢复任何窗口。
            QTimer.singleShot(80, lambda: self._capture_and_ask(mode))
            return
        self._panel_was_visible = self.panel.isVisible()
        if self._panel_was_visible:
            self.panel.hide()
        QTimer.singleShot(120, lambda: self._capture_and_ask(mode))

    def _start_region_capture(self) -> None:
        # 先藏面板并留一点时间，让题目窗口回到前台，再显示框选层
        self._panel_was_visible = self.panel.isVisible() and not self._phone_active
        if self._panel_was_visible:
            self.panel.hide()
        self.set_status("请拖框选择题目区域")
        QTimer.singleShot(160, self._show_region_selector)

    def _show_region_selector(self) -> None:
        selector = RegionSelector()
        selector.selected.connect(self._on_region_selected)
        selector.cancelled.connect(self._on_region_cancelled)
        self.region_selector = selector
        selector.show()

    def _close_region_selector(self) -> None:
        selector, self.region_selector = self.region_selector, None
        if selector is not None:
            selector.hide()
            selector.deleteLater()

    def _on_region_selected(self, rect) -> None:
        self._close_region_selector()
        physical = logical_to_physical(rect)
        self.shot_worker = ScreenshotWorker("region", physical, self)
        self.shot_worker.ready.connect(self._on_screenshot_ready)
        self.shot_worker.failed.connect(self._on_screenshot_failed)
        self.shot_worker.start()

    def _on_region_cancelled(self) -> None:
        self._close_region_selector()
        self._restore_panel()
        self.set_status("已取消截图")

    def _restore_panel(self) -> None:
        if self._phone_active:
            return
        if self._panel_was_visible:
            self.panel.show()
            self._panel_was_visible = False

    def _capture_and_ask(self, mode: str = "window") -> None:
        self.shot_worker = ScreenshotWorker(mode, parent=self)
        self.shot_worker.ready.connect(self._on_screenshot_ready)
        self.shot_worker.failed.connect(self._on_screenshot_failed)
        self.shot_worker.start()

    def _on_screenshot_ready(self, png_bytes: bytes) -> None:
        self._restore_panel()
        self._ask_vision([png_bytes])

    def _on_screenshot_failed(self, message: str) -> None:
        self._restore_panel()
        if not self._phone_active:
            self.overlay.show_overlay()
            self.overlay.begin("截图失败")
            self.overlay.error(message)
        self.set_status(f"截图失败：{message}")

    def add_remote_shot(self) -> None:
        """手机端“追加截图”：只截图并累计，不立即识题。"""
        if self.shot_worker is not None and self.shot_worker.isRunning():
            return
        self.overlay.hide()
        self.set_status("正在截图…")
        mode = self._capture_mode()
        if mode == "region":
            # 手机端无法框选，退回活动窗口
            mode = "window"
        self.shot_worker = ScreenshotWorker(mode, parent=self)
        self.shot_worker.ready.connect(self._on_shot_added)
        self.shot_worker.failed.connect(self._on_shot_add_failed)
        self.shot_worker.start()

    def _on_shot_added(self, png_bytes: bytes) -> None:
        self._pending_shots.append(png_bytes)
        count = len(self._pending_shots)
        self.remote_state.set_shots(count)
        self.set_status(f"已截 {count} 张，可继续截或提交")

    def _on_shot_add_failed(self, message: str) -> None:
        self.set_status(f"截图失败：{message}")

    def submit_remote_shots(self) -> None:
        images, self._pending_shots = self._pending_shots, []
        self.remote_state.set_shots(0)
        if not images:
            self.set_status("还没有待提交的截图")
            return
        self._ask_vision(images)

    def _ask_vision(self, images: List[bytes]) -> None:
        client = self._vision_client()
        if client is None:
            self._overlay_begin("未配置视觉模型")
            self._overlay_error("请在“模型与音频”标签页填写视觉模型的 API Key。")
            if not self._phone_active:
                self.panel.tabs.setCurrentIndex(1)
            self.set_status("未配置视觉模型")
            return

        if self.vision_worker is None:
            self.vision_worker = VisionWorker(
                client,
                self.config["prompts"]["exam"],
                temperature=float(self.config["vision_api"].get("temperature", 0.2)),
                parent=self,
            )
            self.vision_worker.vision_started.connect(
                lambda: self._overlay_set_status("生成中…")
            )
            self.vision_worker.delta.connect(self.on_answer_delta)
            self.vision_worker.completed.connect(self.on_vision_completed)
            self.vision_worker.failed.connect(self.on_vision_failed)
            self.vision_worker.start()
        else:
            self.vision_worker.client = client
            self.vision_worker.system_prompt = self.config["prompts"]["exam"]
            self.vision_worker.temperature = float(
                self.config["vision_api"].get("temperature", 0.2)
            )

        self._overlay_begin("正在识题…")
        self.notes.add_exam(images)
        self.remote_state.begin_answer("exam")
        self.set_status("正在识题…")
        self.vision_worker.submit(images)

    def on_vision_completed(self) -> None:
        self.remote_state.end_answer()
        if not self._phone_active:
            self.overlay.finish()
        self.set_status("答题完成")

    def on_vision_failed(self, message: str) -> None:
        self.remote_state.end_answer(message)
        self._overlay_error(message)
        self.set_status(f"识题失败：{message}")

    def apply_config(self, config: Dict[str, Any]) -> None:
        old = self.config
        self.config = config
        self.overlay.apply_config(config)
        auto_on = bool(config["answer"].get("auto", True))
        self.remote_state.set_auto(auto_on)
        self.panel.set_auto(auto_on)

        if old.get("hotkeys") != config.get("hotkeys"):
            self._install_hotkeys(config["hotkeys"])

        if self.answer_worker is not None:
            self.answer_worker.system_prompt = self._system_prompt()
            self.answer_worker.temperature = float(
                config["text_api"].get("temperature", 0.4)
            )
            self.answer_worker.max_context_chars = int(
                config["answer"].get("max_context_chars", 4000)
            )
        if self.vision_worker is not None:
            self.vision_worker.system_prompt = config["prompts"]["exam"]
            self.vision_worker.temperature = float(
                config["vision_api"].get("temperature", 0.2)
            )

        audio_changed = (
            old.get("audio") != config.get("audio")
            or old.get("asr_api") != config.get("asr_api")
            or old.get("asr_local") != config.get("asr_local")
            or old.get("asr_stream") != config.get("asr_stream")
        )
        if self._listening and audio_changed:
            self.stop_listening()
            QTimer.singleShot(200, self.start_listening)

        old_company = str((old.get("knowledge") or {}).get("company") or "")
        new_company = str((config.get("knowledge") or {}).get("company") or "")
        if self._listening and old_company != new_company:
            # 换了公司资料，提示词前缀也变了：重新预热，别让新问题吃冷启动
            self._warmup_text_model()

        self.set_status("设置已生效")

    def shutdown(self) -> None:
        self.stop_listening()
        self.remote.stop()
        self._close_region_selector()
        for worker in (self.answer_worker, self.vision_worker):
            if worker is not None and worker.isRunning():
                worker.stop()
                if not worker.wait(1500):
                    worker.terminate()
                    worker.wait(500)
        if self.shot_worker is not None and self.shot_worker.isRunning():
            self.shot_worker.wait(1500)
        for worker in self._stale_workers:
            if worker.isRunning():
                worker.terminate()
                worker.wait(500)
        self._stale_workers.clear()
        if self.hotkeys is not None:
            self.hotkeys.stop()

        self.config["ui"].update(self.overlay.geometry_config())
        self.config["ui"].update(self.panel.geometry_config())
        try:
            save_config(self.config)
        except OSError:
            pass

        self.tray.hide()
        self.overlay.hide()
        self.panel.hide()
        self.app.quit()
