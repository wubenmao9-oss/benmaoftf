from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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

from .config import (
    APP_DIR,
    COMPANY_TEMPLATE,
    company_path,
    delete_company,
    list_companies,
    load_company,
    load_knowledge,
    save_company,
    save_config,
    save_knowledge,
    sanitize_company_name,
)
from .llm import OpenAICompatibleClient
from .setup import (
    SetupDialog,
    check_environment,
    format_results,
    overall_level,
    LEVEL_TEXT,
)


class _OptimizeWorker(QThread):
    """后台把公司原始材料整理成结构化资料，避免阻塞界面。"""

    done = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        client: OpenAICompatibleClient,
        system: str,
        material: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._system = system
        self._material = material

    def run(self) -> None:
        try:
            chunks = []
            for piece in self._client.stream_chat(
                [
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": self._material},
                ],
                max_tokens=900,
            ):
                chunks.append(piece)
            text = "".join(chunks).strip()
            if not text:
                raise RuntimeError("模型没有返回内容")
            self.done.emit(text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ControlPanel(QWidget):
    start_listen = Signal()
    stop_listen = Signal()
    screenshot = Signal()
    answer_now = Signal()
    overlay_toggle = Signal()
    remote_toggle = Signal()
    save_notes = Signal()
    auto_toggled = Signal(bool)
    config_saved = Signal(dict)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._config = copy.deepcopy(config)
        self._syncing_auto = False
        # 公司下拉框初始化/程序化刷新时不要触发“切换即保存”
        self._company_loading = False
        self._company_shown = ""
        self._optimize_worker: Any = None
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
        save_note_btn = QPushButton("保存题目笔记")

        self.start_btn.clicked.connect(self.start_listen.emit)
        self.stop_btn.clicked.connect(self.stop_listen.emit)
        screenshot_btn.clicked.connect(self.screenshot.emit)
        answer_btn.clicked.connect(self.answer_now.emit)
        overlay_btn.clicked.connect(self.overlay_toggle.emit)
        remote_btn.clicked.connect(self.remote_toggle.emit)
        save_note_btn.clicked.connect(self.save_notes.emit)

        buttons_top = QHBoxLayout()
        for button in (self.start_btn, self.stop_btn, screenshot_btn):
            buttons_top.addWidget(button)
        buttons_top.addStretch(1)
        layout.addLayout(buttons_top)

        buttons_bottom = QHBoxLayout()
        for button in (answer_btn, overlay_btn, remote_btn, save_note_btn):
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

        practice_box = QGroupBox("练习模式（没有真实面试时，自己当面试官）")
        practice_layout = QVBoxLayout(practice_box)
        self.practice_ctrl = QCheckBox("我来当面试官：只听我的麦克风，不听系统声音")
        self.practice_ctrl.setChecked(self._config["audio"].get("source") == "microphone")
        self.practice_ctrl.toggled.connect(self._set_practice)
        practice_layout.addWidget(self.practice_ctrl)
        self.practice_hint = QLabel(
            "开启后，你对着麦克风提问，助手把你当成面试官来回答，效果和真实面试一样。"
            "提示：如果你把自己的答案朗读出来，麦克风也会听到，可能被当成新的提问。"
        )
        self.practice_hint.setObjectName("hint")
        self.practice_hint.setWordWrap(True)
        practice_layout.addWidget(self.practice_hint)
        layout.addWidget(practice_box)

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

        setup_group = QGroupBox("首次使用 / 换电脑（一键初始化）")
        setup_layout = QVBoxLayout(setup_group)
        setup_hint = QLabel(
            "新电脑或刚拿到这个程序：点「一键初始化」自动下载本地语音识别包（带进度条），"
            "下载完会自动检测 Key、模型和音频设备是否就绪。"
        )
        setup_hint.setObjectName("hint")
        setup_hint.setWordWrap(True)
        setup_layout.addWidget(setup_hint)
        setup_row = QHBoxLayout()
        init_btn = QPushButton("一键初始化")
        init_btn.clicked.connect(self._open_setup)
        check_btn = QPushButton("检测运行环境")
        check_btn.clicked.connect(self._check_environment)
        setup_row.addWidget(init_btn)
        setup_row.addWidget(check_btn)
        setup_row.addStretch(1)
        setup_layout.addLayout(setup_row)
        layout.addWidget(setup_group)

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
        self.audio_source.currentIndexChanged.connect(self._on_source_changed)

        self.stream_model = QComboBox()
        self._populate_stream_models()
        audio_form.addRow("识别模型", self.stream_model)

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

        # ---- 本次面试的公司（每家公司一份，面试前切换）----
        company_box = QGroupBox("本次面试的公司（每家公司一份）")
        company_layout = QVBoxLayout(company_box)

        company_hint = QLabel(
            "面试开始前在这里选好公司，切换后立刻生效。公司资料只在问"
            "“你了解我们公司吗 / 你了解这个岗位吗”这类问题时使用；"
            "个人资料才是所有面试共用的底盘。"
        )
        company_hint.setObjectName("hint")
        company_hint.setWordWrap(True)
        company_layout.addWidget(company_hint)

        company_row = QHBoxLayout()
        self.company_combo = QComboBox()
        self.company_combo.setMinimumWidth(160)
        self.company_combo.currentIndexChanged.connect(self._on_company_changed)
        company_row.addWidget(self.company_combo, 1)
        new_btn = QPushButton("新建")
        new_btn.clicked.connect(self._new_company)
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self._delete_company)
        company_row.addWidget(new_btn)
        company_row.addWidget(delete_btn)
        company_layout.addLayout(company_row)

        self.company_edit = QPlainTextEdit()
        self.company_edit.setPlaceholderText(
            "把公司简介、招聘 JD、官网介绍或你自己的笔记粘进来，"
            "点“AI 整理”会自动分成：公司一句话 / 业务与产品 / 岗位理解 / 我的匹配点 / 反问问题。"
        )
        self.company_edit.setMinimumHeight(180)
        company_layout.addWidget(self.company_edit)

        company_btns = QHBoxLayout()
        self.optimize_btn = QPushButton("AI 整理公司资料")
        self.optimize_btn.clicked.connect(self._optimize_company)
        save_company_btn = QPushButton("保存公司资料")
        save_company_btn.clicked.connect(self._save_company_clicked)
        company_btns.addWidget(self.optimize_btn)
        company_btns.addWidget(save_company_btn)
        company_btns.addStretch(1)
        company_layout.addLayout(company_btns)
        layout.addWidget(company_box)

        # ---- 个人资料（底盘，所有面试共用）----
        personal_box = QGroupBox("个人资料（底盘，所有面试共用）")
        personal_layout = QVBoxLayout(personal_box)

        hint = QLabel(
            "把你的简历、项目细节、技术栈、真实经历和薄弱点写在这里。"
            "面试回答会优先使用这些资料，资料里没有的内容不会编造。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        personal_layout.addWidget(hint)

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
        personal_layout.addLayout(buttons)

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
        personal_layout.addWidget(self.knowledge_edit, 1)
        layout.addWidget(personal_box, 1)

        # 用配置里选好的公司初始化下拉框（不会触发“切换即保存”）
        self._reload_company_combo(
            str((self._config.get("knowledge") or {}).get("company", "") or "")
        )
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

    # ---- 公司知识库 ----

    def _current_company(self) -> str:
        return str(self.company_combo.currentData() or "")

    def _reload_company_combo(self, select: str = "") -> None:
        self._company_loading = True
        self.company_combo.clear()
        self.company_combo.addItem("（未选择公司）", "")
        for name in list_companies():
            self.company_combo.addItem(name, name)
        index = self.company_combo.findData(select)
        self.company_combo.setCurrentIndex(index if index >= 0 else 0)
        self._company_loading = False
        self._company_shown = self._current_company()
        self.company_edit.setPlainText(load_company(self._company_shown))

    def _persist_company_selection(self, company: str) -> None:
        """切换公司立刻写配置并通知主窗口，下一句回答就生效。"""
        self._config.setdefault("knowledge", {})
        self._config["knowledge"]["company"] = company
        try:
            save_config(self._config)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.config_saved.emit(copy.deepcopy(self._config))
        if company:
            self.status_label.setText(f"已切换到：{company}（下一句回答生效）")
        else:
            self.status_label.setText("已取消公司选择")

    def _on_company_changed(self, _index: int) -> None:
        if self._company_loading:
            return
        previous = self._company_shown
        if previous and previous != self._current_company():
            # 切换前把上一家编辑框里的内容落盘，避免误切换丢资料
            try:
                save_company(previous, self.company_edit.toPlainText())
            except OSError:
                pass
        company = self._current_company()
        self._company_shown = company
        self.company_edit.setPlainText(load_company(company))
        self._persist_company_selection(company)

    def _new_company(self) -> None:
        name, ok = QInputDialog.getText(self, "新建公司", "公司名称（用作资料文件名）：")
        if not ok:
            return
        name = sanitize_company_name(name)
        if not name:
            QMessageBox.warning(
                self, "名称无效", "公司名称不能为空，也不能包含 / \\ 等路径字符。"
            )
            return
        if company_path(name).exists():
            QMessageBox.warning(self, "已存在", f"“{name}”已存在，直接在下拉框里选择即可。")
            return
        try:
            save_company(name, COMPANY_TEMPLATE.format(company=name))
        except OSError as exc:
            QMessageBox.warning(self, "创建失败", str(exc))
            return
        self._reload_company_combo(name)
        self._persist_company_selection(name)

    def _delete_company(self) -> None:
        name = self._current_company()
        if not name:
            return
        answer = QMessageBox.question(
            self, "删除公司", f"确定删除“{name}”的公司资料？删除后无法恢复。"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        delete_company(name)
        self._reload_company_combo("")
        self._persist_company_selection("")

    def _save_company_clicked(self) -> None:
        name = self._current_company()
        if not name:
            QMessageBox.warning(self, "未选择公司", "请先新建或选择一个公司。")
            return
        try:
            save_company(name, self.company_edit.toPlainText())
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.status_label.setText(f"已保存公司资料：{name}")

    def _optimize_company(self) -> None:
        material = self.company_edit.toPlainText().strip()
        if not material:
            QMessageBox.warning(
                self, "没有材料", "先把公司简介或招聘 JD 粘到上面的编辑框，再点 AI 整理。"
            )
            return
        if self._optimize_worker is not None and self._optimize_worker.isRunning():
            self.status_label.setText("上一份还在整理，请稍等")
            return
        api = self._config.get("text_api") or {}
        api_key = self.text_key.text().strip() or str(api.get("api_key", "")).strip()
        if not api_key:
            QMessageBox.warning(
                self, "未配置文本模型", "请先在“模型与音频”里填好文本模型的 API Key。"
            )
            return
        client = OpenAICompatibleClient(
            self.text_base.text().strip() or str(api.get("base_url", "")),
            api_key,
            self.text_model.text().strip() or str(api.get("model", "")),
            timeout=(15.0, 180.0),
        )
        system = (self._config.get("prompts") or {}).get("company_optimize", "")
        personal = self.knowledge_edit.toPlainText().strip()
        if personal:
            material += "\n\n【我的个人资料（用来写“我为什么匹配”）】\n" + personal[:2000]

        self.optimize_btn.setEnabled(False)
        self.status_label.setText("AI 正在整理公司资料…")
        worker = _OptimizeWorker(client, system, material, self)
        worker.done.connect(self._on_optimize_done)
        worker.failed.connect(self._on_optimize_failed)
        worker.finished.connect(worker.deleteLater)
        self._optimize_worker = worker
        worker.start()

    def _on_optimize_done(self, text: str) -> None:
        self.optimize_btn.setEnabled(True)
        self._optimize_worker = None
        self.company_edit.setPlainText(text)
        self.status_label.setText("AI 整理完成，确认内容后点“保存公司资料”")

    def _on_optimize_failed(self, message: str) -> None:
        self.optimize_btn.setEnabled(True)
        self._optimize_worker = None
        self.status_label.setText("AI 整理失败")
        QMessageBox.warning(self, "AI 整理失败", message)


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
        config["knowledge"]["company"] = self._current_company()
        config.setdefault("asr_stream", {})
        if self.stream_model.currentData():
            config["asr_stream"]["model_dir"] = self.stream_model.currentData()
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
            company = self._current_company()
            if company:
                save_company(company, self.company_edit.toPlainText())
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._config = copy.deepcopy(config)
        self._company_shown = self._current_company()
        self.status_label.setText("设置已保存")
        self.config_saved.emit(config)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_practice(self, on: bool) -> None:
        """练习模式：自己当面试官，改用麦克风收音，切换后立即保存并生效。"""
        source = "microphone" if on else "loopback"
        box = getattr(self, "audio_source", None)
        if box is not None:
            index = box.findData(source)
            if index >= 0 and box.currentIndex() != index:
                box.blockSignals(True)
                box.setCurrentIndex(index)
                box.blockSignals(False)
        config = self.collect_config()
        try:
            save_config(config)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "切换失败", str(exc))
            return
        self._config = copy.deepcopy(config)
        self.status_label.setText(
            "练习模式：正在聆听你的提问" if on else "已切回系统声音（正常面试）"
        )
        self.config_saved.emit(config)
        # 勾选练习模式就直接开始听，省得再点一次“开始监听”
        if on and not self.stop_btn.isEnabled():
            self.start_listen.emit()

    def _open_setup(self) -> None:
        """一键初始化：下载语音模型，下完自动检测环境。"""
        dialog = SetupDialog(self.collect_config(), self)
        dialog.model_ready.connect(self._on_setup_model_ready)
        dialog.exec()

    def _check_environment(self) -> None:
        results = check_environment(self.collect_config())
        level = overall_level(results)
        QMessageBox.information(
            self,
            "环境检测",
            LEVEL_TEXT.get(level, "") + "\n\n" + format_results(results),
        )

    def _on_setup_model_ready(self, rel_dir: str) -> None:
        """初始化下载完模型后，自动选中并立刻生效。"""
        self._refresh_stream_models(rel_dir)
        config = self.collect_config()
        try:
            save_config(config)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._config = copy.deepcopy(config)
        self.status_label.setText("语音模型已就绪，已自动切换")
        self.config_saved.emit(config)

    def _refresh_stream_models(self, select: str = "") -> None:
        if not hasattr(self, "stream_model"):
            return
        want = select or str(self.stream_model.currentData() or "")
        self.stream_model.blockSignals(True)
        self.stream_model.clear()
        self._populate_stream_models(want)
        self.stream_model.blockSignals(False)

    def _populate_stream_models(self, select: str = "") -> None:
        """列出 models/ 下可用的本地流式模型，切换后重启监听生效。"""
        current = select or str(
            (self._config.get("asr_stream") or {}).get("model_dir") or ""
        )
        found = []
        models_dir = APP_DIR / "models"
        if models_dir.is_dir():
            for item in sorted(models_dir.iterdir()):
                if item.is_dir() and (item / "tokens.txt").exists():
                    found.append(f"models/{item.name}")
        if current and current not in found:
            found.insert(0, current)
        for rel in found:
            self.stream_model.addItem(self._model_label(rel), rel)
        index = self.stream_model.findData(current)
        if index >= 0:
            self.stream_model.setCurrentIndex(index)

    @staticmethod
    def _model_label(rel_path: str) -> str:
        name = Path(rel_path).name
        if "2025-06-30" in name and "zh" in name:
            return "中文高精度（2025 版，英文弱）"
        if "bilingual" in name:
            return "中英双语（2023 版，较快）"
        return name

    def _on_source_changed(self, _index: int) -> None:
        """在“模型与音频”页手动改声音来源时，同步控制页的练习模式开关。"""
        source = self.audio_source.currentData()
        self._config["audio"]["source"] = source
        box = getattr(self, "practice_ctrl", None)
        if box is not None and box.isChecked() != (source == "microphone"):
            box.blockSignals(True)
            box.setChecked(source == "microphone")
            box.blockSignals(False)

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
