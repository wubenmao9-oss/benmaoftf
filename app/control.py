from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .config import load_knowledge, save_config, save_knowledge


class ControlPanel(QWidget):
    start_listen = Signal()
    stop_listen = Signal()
    screenshot = Signal()
    answer_now = Signal()
    overlay_toggle = Signal()
    remote_toggle = Signal()
    auto_toggled = Signal(bool)
    config_saved = Signal(dict)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._config = copy.deepcopy(config)
        self._syncing_auto = False
        self.setWindowTitle("笨猫面试助手")
        # 窗口可自由缩放；内容放不下时滚动，而不是把控件压扁
        self.setMinimumSize(620, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("面试 / 笔试 AI 助手")
        title.setObjectName("appTitle")
        self.status_label = QLabel("待命")
        self.status_label.setObjectName("statusLabel")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_label)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scrollable(self._build_control_tab()), "控制")
        self.tabs.addTab(self._scrollable(self._build_model_tab()), "模型与音频")
        self.tabs.addTab(self._scrollable(self._build_knowledge_tab()), "知识库")
        self.tabs.addTab(self._scrollable(self._build_prompt_tab()), "提示词")
        root.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        self.hotkey_hint = QLabel(self._hotkey_text())
        self.hotkey_hint.setObjectName("hint")
        # 不换行的话这行长文本会把窗口最小宽度撑到 900px 以上，缩放时就会挤坏布局
        self.hotkey_hint.setWordWrap(True)
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self._save)
        footer.addWidget(self.hotkey_hint, 1)
        footer.addWidget(save_btn)
        root.addLayout(footer)

        self.setStyleSheet(
            """
            QWidget { background: #14161c; color: #e8eaf0; }
            QLabel#appTitle { font-size: 18px; font-weight: 600; }
            QLabel#statusLabel { color: #7fd18b; }
            QLabel#hint { color: #8b93a7; font-size: 12px; }
            QGroupBox {
                border: 1px solid #2a2f3d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #aab2c5;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1b1e26;
                border: 1px solid #2f3546;
                border-radius: 6px;
                padding: 5px;
                color: #e8eaf0;
            }
            QPushButton {
                background: #2c3344;
                border: 1px solid #3a4358;
                border-radius: 6px;
                padding: 7px 14px;
            }
            QPushButton:hover { background: #364058; }
            QPushButton:disabled { color: #5d6577; background: #1d2029; }
            QTabWidget::pane { border: 1px solid #2a2f3d; border-radius: 8px; }
            QTabBar::tab {
                background: #1b1e26;
                color: #aab2c5;
                padding: 8px 16px;
                border: 1px solid #2a2f3d;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected { background: #2c3344; color: #ffffff; }
            """
        )

        self.set_listening(False)

        # 尺寸放在布局构建之后设置，避免被布局的 sizeHint 覆盖
        ui = self._config.get("ui", {})
        self.resize(int(ui.get("panel_width", 820)), int(ui.get("panel_height", 720)))

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """内容放不下时出现滚动条，而不是把控件压扁、文字叠在一起。"""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(page)
        return area

    def _build_control_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.start_btn = QPushButton("开始监听")
        self.stop_btn = QPushButton("停止监听")
        screenshot_btn = QPushButton("截图答题")
        answer_btn = QPushButton("立即回答")
        overlay_btn = QPushButton("显示 / 隐藏悬浮窗")
        remote_btn = QPushButton("手机远程")

        self.start_btn.clicked.connect(self.start_listen.emit)
        self.stop_btn.clicked.connect(self.stop_listen.emit)
        screenshot_btn.clicked.connect(self.screenshot.emit)
        answer_btn.clicked.connect(self.answer_now.emit)
        overlay_btn.clicked.connect(self.overlay_toggle.emit)
        remote_btn.clicked.connect(self.remote_toggle.emit)

        buttons_top = QHBoxLayout()
        for button in (self.start_btn, self.stop_btn, screenshot_btn):
            buttons_top.addWidget(button)
        buttons_top.addStretch(1)
        layout.addLayout(buttons_top)

        buttons_bottom = QHBoxLayout()
        for button in (answer_btn, overlay_btn, remote_btn):
            buttons_bottom.addWidget(button)
        buttons_bottom.addStretch(1)
        layout.addLayout(buttons_bottom)

        options = QVBoxLayout()
        self.auto_ctrl = QCheckBox("自动回答（识别到面试官提问后自动生成）")
        self.auto_ctrl.setChecked(bool(self._config["answer"]["auto"]))
        self.auto_ctrl.toggled.connect(self._set_auto)
        options.addWidget(self.auto_ctrl)

        self.question_only = QCheckBox("仅在像“提问”时才自动回答（省 token，少误答）")
        self.question_only.setChecked(bool(self._config["answer"].get("question_only", True)))
        options.addWidget(self.question_only)
        layout.addLayout(options)

        capture_row = QHBoxLayout()
        capture_row.addWidget(QLabel("截图范围"))
        self.capture_mode = QComboBox()
        self.capture_mode.addItem("只截当前活动窗口", "window")
        self.capture_mode.addItem("手动框选区域", "region")
        self.capture_mode.addItem("整个桌面（所有显示器）", "full")
        index = self.capture_mode.findData(
            (self._config.get("capture") or {}).get("mode", "window")
        )
        self.capture_mode.setCurrentIndex(max(0, index))
        capture_row.addWidget(self.capture_mode)
        capture_row.addStretch(1)
        layout.addLayout(capture_row)

        layout.addWidget(QLabel("实时转写（面试对话会持续追加到这里）："))
        self.transcript_view = QPlainTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setMaximumBlockCount(2000)
        self.transcript_view.setPlaceholderText(
            "点击“开始监听”后，系统声音会被分段转写并显示在这里。"
        )
        layout.addWidget(self.transcript_view, 1)

        note = QLabel(
            "说明：面试回答会自动显示在置顶悬浮窗；截图答题按上方“截图范围”抓取"
            "（手机端无法框选，会自动使用活动窗口）。\n"
            "悬浮窗是普通置顶窗口，不做录屏/截图规避。"
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        return page

    def _build_model_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)

        text_group = QGroupBox("文本模型（面试回答）")
        text_form = QFormLayout(text_group)
        self.text_base = QLineEdit(self._config["text_api"]["base_url"])
        self.text_key = self._secret_edit(self._config["text_api"]["api_key"])
        self.text_model = QLineEdit(self._config["text_api"]["model"])
        self.text_temp = self._temp_spin(self._config["text_api"].get("temperature", 0.4))
        text_form.addRow("Base URL", self.text_base)
        text_form.addRow("API Key", self.text_key)
        text_form.addRow("模型", self.text_model)
        text_form.addRow("温度", self.text_temp)
        layout.addWidget(text_group)

        vision_group = QGroupBox("视觉模型（截图识题）")
        vision_form = QFormLayout(vision_group)
        self.vision_base = QLineEdit(self._config["vision_api"]["base_url"])
        self.vision_key = self._secret_edit(self._config["vision_api"]["api_key"])
        self.vision_model = QLineEdit(self._config["vision_api"]["model"])
        self.vision_temp = self._temp_spin(
            self._config["vision_api"].get("temperature", 0.2)
        )
        vision_form.addRow("Base URL", self.vision_base)
        vision_form.addRow("API Key", self.vision_key)
        vision_form.addRow("模型", self.vision_model)
        vision_form.addRow("温度", self.vision_temp)
        layout.addWidget(vision_group)

        asr_group = QGroupBox("语音转写（本地 Whisper / 云端 ASR）")
        asr_form = QFormLayout(asr_group)
        self.asr_base = QLineEdit(self._config["asr_api"]["base_url"])
        self.asr_key = self._secret_edit(self._config["asr_api"]["api_key"])
        self.asr_model = QLineEdit(self._config["asr_api"]["model"])
        self.asr_language = QLineEdit(self._config["asr_api"]["language"])
        asr_form.addRow("Base URL", self.asr_base)
        asr_form.addRow("API Key", self.asr_key)
        asr_form.addRow("模型", self.asr_model)
        asr_form.addRow("语言", self.asr_language)

        local_cfg = self._config.get("asr_local", {})
        self.local_asr = QCheckBox("使用本地 Whisper 转写（不消耗 API）")
        self.local_asr.setChecked(bool(local_cfg.get("enabled", True)))
        self.local_asr_model = QComboBox()
        for label, value in (
            ("base（快，精度一般）", "base"),
            ("small（推荐）", "small"),
            ("medium（更准，较慢）", "medium"),
        ):
            self.local_asr_model.addItem(label, value)
        index = self.local_asr_model.findData(local_cfg.get("model", "small"))
        self.local_asr_model.setCurrentIndex(max(0, index))

        asr_form.addRow("", self.local_asr)
        asr_form.addRow("本地模型", self.local_asr_model)
        layout.addWidget(asr_group)

        audio_group = QGroupBox("音频与回答")
        audio_form = QFormLayout(audio_group)
        self.audio_source = QComboBox()
        self.audio_source.addItem("系统声音（回环）", "loopback")
        self.audio_source.addItem("麦克风", "microphone")
        index = self.audio_source.findData(self._config["audio"]["source"])
        self.audio_source.setCurrentIndex(max(0, index))

        self.chunk_seconds = QDoubleSpinBox()
        self.chunk_seconds.setRange(1.0, 15.0)
        self.chunk_seconds.setSingleStep(0.5)
        self.chunk_seconds.setValue(float(self._config["audio"]["chunk_seconds"]))

        self.silence_rms = QDoubleSpinBox()
        self.silence_rms.setRange(0.0, 0.1)
        self.silence_rms.setDecimals(4)
        self.silence_rms.setSingleStep(0.001)
        self.silence_rms.setValue(float(self._config["audio"]["silence_rms"]))

        self.auto_answer = QCheckBox("转写后自动生成回答")
        self.auto_answer.setChecked(bool(self._config["answer"]["auto"]))
        self.auto_answer.toggled.connect(self._set_auto)

        self.debounce_seconds = QDoubleSpinBox()
        self.debounce_seconds.setRange(0.3, 10.0)
        self.debounce_seconds.setSingleStep(0.1)
        self.debounce_seconds.setValue(float(self._config["answer"]["debounce_seconds"]))

        self.min_new_chars = QSpinBox()
        self.min_new_chars.setRange(2, 200)
        self.min_new_chars.setValue(int(self._config["answer"]["min_new_chars"]))

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(50, 100)
        self.opacity_slider.setValue(int(float(self._config["ui"]["opacity"]) * 100))

        self.font_size = QSpinBox()
        self.font_size.setRange(10, 28)
        self.font_size.setValue(int(self._config["ui"]["font_size"]))

        audio_form.addRow("声音来源", self.audio_source)
        audio_form.addRow("分段秒数", self.chunk_seconds)
        audio_form.addRow("静音阈值", self.silence_rms)
        audio_form.addRow("", self.auto_answer)
        audio_form.addRow("回答防抖（秒）", self.debounce_seconds)
        audio_form.addRow("最少新增字数", self.min_new_chars)
        audio_form.addRow("悬浮窗不透明度", self.opacity_slider)
        audio_form.addRow("悬浮窗字号", self.font_size)
        layout.addWidget(audio_group)
        # 窗口变窄时把标签换到字段上方，避免标签和输入框互相挤压
        for form in (text_form, vision_form, asr_form, audio_form):
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addStretch(1)
        return page

    def _build_knowledge_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)

        hint = QLabel(
            "把你的简历、项目细节、技术栈、真实经历和薄弱点写在这里。"
            "面试回答会优先使用这些资料，资料里没有的内容不会编造。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        import_btn = QPushButton("导入 txt / md")
        import_btn.clicked.connect(self._import_knowledge)
        self.knowledge_enabled = QCheckBox("面试回答时使用知识库")
        self.knowledge_enabled.setChecked(
            bool((self._config.get("knowledge") or {}).get("enabled", True))
        )
        buttons.addWidget(import_btn)
        buttons.addWidget(self.knowledge_enabled)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.knowledge_edit = QPlainTextEdit(load_knowledge())
        self.knowledge_edit.setPlaceholderText(
            """例：
- 姓名 / 岗位 / 工作年限
- 教育背景、证书
- 每段经历的公司、时间、职责、量化结果
- 技术栈与熟练度（哪些只是了解、哪些能深入）
- 项目难点、踩过的坑、真实数据
- 不擅长的方向（让 AI 用保守说法，不硬编）"""
        )
        layout.addWidget(self.knowledge_edit, 1)
        return page

    def _import_knowledge(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入资料", "", "文本文件 (*.txt *.md);;所有文件 (*.*)"
        )
        for path in paths:
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                QMessageBox.warning(self, "导入失败", f"{path}" + chr(10) + f"{exc}")
                continue
            self.knowledge_edit.appendPlainText(chr(10) * 2 + text.strip())


    def _build_prompt_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)

        layout.addWidget(QLabel("面试回答提示词："))
        self.interview_prompt = QPlainTextEdit(self._config["prompts"]["interview"])
        layout.addWidget(self.interview_prompt, 1)

        layout.addWidget(QLabel("截图识题提示词："))
        self.exam_prompt = QPlainTextEdit(self._config["prompts"]["exam"])
        layout.addWidget(self.exam_prompt, 1)
        return page

    def _secret_edit(self, value: str) -> QLineEdit:
        edit = QLineEdit(value)
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        edit.setPlaceholderText("留空则不启用对应功能")
        return edit

    def _temp_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 2.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(float(value))
        return spin

    def _hotkey_text(self) -> str:
        hotkeys = self._config["hotkeys"]
        return (
            f"热键：显示/隐藏 {hotkeys['toggle_overlay']} · "
            f"截图答题 {hotkeys['screenshot']} · "
            f"开始/停止监听 {hotkeys['toggle_listen']} · "
            f"立即回答 {hotkeys['answer_now']} · "
            f"退出 {hotkeys['quit']}（修改需重启）"
        )

    def collect_config(self) -> Dict[str, Any]:
        config = copy.deepcopy(self._config)
        config["text_api"].update(
            {
                "base_url": self.text_base.text().strip(),
                "api_key": self.text_key.text().strip(),
                "model": self.text_model.text().strip(),
                "temperature": float(self.text_temp.value()),
            }
        )
        config["vision_api"].update(
            {
                "base_url": self.vision_base.text().strip(),
                "api_key": self.vision_key.text().strip(),
                "model": self.vision_model.text().strip(),
                "temperature": float(self.vision_temp.value()),
            }
        )
        config["asr_api"].update(
            {
                "base_url": self.asr_base.text().strip(),
                "api_key": self.asr_key.text().strip(),
                "model": self.asr_model.text().strip(),
                "language": self.asr_language.text().strip(),
            }
        )
        config.setdefault("asr_local", {})
        config["asr_local"].update(
            {
                "enabled": self.local_asr.isChecked(),
                "model": self.local_asr_model.currentData(),
            }
        )
        config.setdefault("knowledge", {})
        config["knowledge"]["enabled"] = self.knowledge_enabled.isChecked()
        config["audio"].update(
            {
                "source": self.audio_source.currentData(),
                "chunk_seconds": float(self.chunk_seconds.value()),
                "silence_rms": float(self.silence_rms.value()),
            }
        )
        config["answer"].update(
            {
                "auto": bool(self._config["answer"]["auto"]),
                "question_only": self.question_only.isChecked(),
                "debounce_seconds": float(self.debounce_seconds.value()),
                "min_new_chars": int(self.min_new_chars.value()),
            }
        )
        config.setdefault("capture", {})
        config["capture"]["mode"] = self.capture_mode.currentData()
        config["ui"].update(
            {
                "opacity": self.opacity_slider.value() / 100.0,
                "font_size": int(self.font_size.value()),
            }
        )
        config["prompts"].update(
            {
                "interview": self.interview_prompt.toPlainText(),
                "exam": self.exam_prompt.toPlainText(),
            }
        )
        return config

    def _save(self) -> None:
        config = self.collect_config()
        try:
            save_config(config)
            save_knowledge(self.knowledge_edit.toPlainText())
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._config = copy.deepcopy(config)
        self.status_label.setText("设置已保存")
        self.config_saved.emit(config)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_auto(self, on: bool) -> None:
        on = bool(on)
        self._config["answer"]["auto"] = on
        for box in (
            getattr(self, "auto_ctrl", None),
            getattr(self, "auto_answer", None),
        ):
            if box is not None and box.isChecked() != on:
                box.blockSignals(True)
                box.setChecked(on)
                box.blockSignals(False)
        if not self._syncing_auto:
            self.auto_toggled.emit(on)

    def set_auto(self, on: bool) -> None:
        """供手机端切换自动回答时同步界面（不回传，避免信号回环）。"""
        self._syncing_auto = True
        try:
            self._set_auto(bool(on))
        finally:
            self._syncing_auto = False

    def geometry_config(self) -> Dict[str, Any]:
        return {"panel_width": self.width(), "panel_height": self.height()}

    def set_listening(self, listening: bool) -> None:
        self.start_btn.setEnabled(not listening)
        self.stop_btn.setEnabled(listening)

    def append_transcript(self, line: str) -> None:
        self.transcript_view.appendPlainText(line)

    def clear_transcript(self) -> None:
        self.transcript_view.clear()

    def show_and_focus(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
