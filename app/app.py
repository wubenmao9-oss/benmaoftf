from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .asr_local import LocalWhisperASR
from .audio import AudioCapture
from .capture import logical_to_physical
from .config import load_knowledge, save_config
from .control import ControlPanel
from .hotkeys import HotkeyManager
from .llm import OpenAICompatibleClient
from .overlay import OverlayWindow
from .region import RegionSelector
from .remote import RemoteServer, RemoteState
from .remote_dialog import RemoteDialog
from .workers import (
    ASRWorker,
    AnswerWorker,
    ScreenshotWorker,
    VisionWorker,
)


WAIT_MARKERS = {"等待问题", "等待问题中", "暂无问题"}

# 本地粗筛“这句像不像面试官在提问”，命中才调用大模型，省 token 也少误答
QUESTION_HINTS = (
    "?",
    "？",
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
    "原理",
    "实现",
    "思路",
    "方案",
    "场景",
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
)


def looks_like_question(text: str) -> bool:
    tail = (text or "").strip()[-160:]
    if not tail:
        return False
    return any(hint in tail for hint in QUESTION_HINTS)


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
        self.answer_worker: Optional[AnswerWorker] = None
        self.vision_worker: Optional[VisionWorker] = None
        self.shot_worker: Optional[ScreenshotWorker] = None
        self._stale_workers = []

        self.transcript = ""
        self._last_answered_len = 0
        self._answering = False
        self._pending_answer = False
        self._listening = False
        self._panel_was_visible = False
        # 手机已接管显示：电脑端只做后台保活，不再弹回答框
        self._phone_active = False
        self._pending_shots: List[bytes] = []
        self.region_selector: Optional[RegionSelector] = None

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

    def _system_prompt(self) -> str:
        prompt = self.config["prompts"]["interview"]
        knowledge_cfg = self.config.get("knowledge") or {}
        if not knowledge_cfg.get("enabled", True):
            return prompt
        text = load_knowledge().strip()
        if not text:
            return prompt
        limit = int(knowledge_cfg.get("max_chars", 12000))
        return prompt + "\n\n【我的真实资料（回答时必须优先使用，不得编造）】\n" + text[:limit]

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

    def start_listening(self) -> None:
        if self._listening:
            return
        client = self._asr_client()
        if client is None:
            self.set_status("未配置语音转写（本地 Whisper 未启用且无 ASR Key）")
            self.panel.tabs.setCurrentIndex(1)
            return

        audio_cfg = self.config["audio"]
        self.asr_worker = ASRWorker(
            client, language=self.config["asr_api"].get("language", "zh"), parent=self
        )
        self.asr_worker.transcribed.connect(self.on_transcript)
        self.asr_worker.failed.connect(self.on_asr_failed)
        self.asr_worker.start()

        self.audio = AudioCapture(
            source=audio_cfg.get("source", "loopback"),
            chunk_seconds=float(audio_cfg.get("chunk_seconds", 4.0)),
            samplerate=int(audio_cfg.get("samplerate", 16000)),
            silence_rms=float(audio_cfg.get("silence_rms", 0.003)),
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
            self.set_status("正在监听…（本地语音模型首次加载较慢）")
        else:
            self.set_status("正在监听…")
        self._overlay_set_status("聆听中")

    def stop_listening(self) -> None:
        if not self._listening and self.audio is None and self.asr_worker is None:
            return
        self._listening = False
        self.remote_state.set_listening(False)
        self._answer_timer.stop()

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

        self.panel.set_listening(False)
        self.set_status("已停止监听")

    def on_audio_chunk(self, wav_bytes: bytes) -> None:
        if self.asr_worker is not None:
            self.asr_worker.submit(wav_bytes)
            self.set_status("识别中…")

    def on_audio_failed(self, message: str) -> None:
        self.set_status(message)
        self.stop_listening()

    def on_asr_failed(self, message: str) -> None:
        self.set_status(f"转写失败：{message}")

    def on_transcript(self, text: str) -> None:
        text = (text or "").strip()
        if not text or not self._listening:
            return
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        self.transcript += ("\n" if self.transcript else "") + line
        self.panel.append_transcript(line)
        self.remote_state.add_transcript(line)
        self.set_status("正在监听…")
        self._overlay_set_status("聆听中")

        if len(self.transcript) > 20000:
            cut = len(self.transcript) - 12000
            self.transcript = self.transcript[-12000:]
            self._last_answered_len = max(0, self._last_answered_len - cut)

        if not self.config["answer"].get("auto", True):
            return
        new_chars = len(self.transcript) - self._last_answered_len
        if new_chars < int(self.config["answer"].get("min_new_chars", 8)):
            return
        if self.config["answer"].get("question_only", True):
            # 只看上次回答之后的新内容，避免旧问题一直触发
            fresh = self.transcript[self._last_answered_len :].strip()
            if not looks_like_question(fresh[-160:]):
                return
        delay = int(float(self.config["answer"].get("debounce_seconds", 1.5)) * 1000)
        self._answer_timer.start(max(200, delay))

    def request_answer(self, force: bool = False) -> None:
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

        if not force:
            new_chars = len(self.transcript) - self._last_answered_len
            if new_chars < int(self.config["answer"].get("min_new_chars", 8)):
                return

        client = self._text_client()
        if client is None:
            self._overlay_begin("未配置文本模型")
            self._overlay_error("请在“模型与音频”标签页填写文本模型的 API Key。")
            if not self._phone_active:
                self.panel.tabs.setCurrentIndex(1)
            self.set_status("未配置文本模型")
            return

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

        self._answering = True
        self._pending_answer = False
        self._overlay_begin("正在生成回答…")
        self.remote_state.begin_answer("interview")
        self.answer_worker.submit(self.transcript)

    def on_answer_delta(self, text: str) -> None:
        if not self._phone_active:
            self.overlay.append(text)
        self.remote_state.append_answer(text)

    def on_answer_completed(self, text: str) -> None:
        self._answering = False
        self._last_answered_len = len(self.transcript)
        compact = (text or "").strip().strip("。.！!？?（）()")
        if compact in WAIT_MARKERS:
            self.remote_state.drop_answer()
            if not self._phone_active:
                self.overlay.clear()
                self.overlay.set_status("等待问题")
            self.set_status("暂无新问题")
        else:
            self.remote_state.end_answer()
            if not self._phone_active:
                self.overlay.finish()
            self.set_status("回答完成")
        if self._pending_answer:
            self._pending_answer = False
            QTimer.singleShot(200, lambda: self.request_answer(force=True))

    def on_answer_failed(self, message: str) -> None:
        self._answering = False
        self.remote_state.end_answer(message)
        self._overlay_error(message)
        self.set_status(f"回答失败：{message}")

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
        )
        if self._listening and audio_changed:
            self.stop_listening()
            QTimer.singleShot(200, self.start_listening)

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
