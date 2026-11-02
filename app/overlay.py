from __future__ import annotations

from typing import Any, Dict, List

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QFont, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizeGrip,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class OverlayWindow(QWidget):
    """Always-on-top read-only answer window. It does not hide from screen capture."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowTitle("AI 助手")
        self.setMinimumSize(360, 240)

        self._config = config
        self._buffer = ""
        # 保留最近几条回答，新回答接在下面，避免刷掉正在看的内容
        self._history: List[str] = []
        self._dirty = False
        self._drag_offset = None
        self._placed = False

        ui = config.get("ui", {})
        self.resize(int(ui.get("width", 580)), int(ui.get("height", 460)))

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("card")
        root.addWidget(card)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 8)
        card_layout.setSpacing(6)

        self.header = QWidget()
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title = QLabel("AI 助手")
        self.title.setObjectName("title")
        self.status = QLabel("待命")
        self.status.setObjectName("status")

        clear_btn = QToolButton()
        clear_btn.setText("清空")
        clear_btn.clicked.connect(self.clear)

        hide_btn = QToolButton()
        hide_btn.setText("—")
        hide_btn.setToolTip("隐藏悬浮窗（Ctrl+Alt+H 可再次显示）")
        hide_btn.clicked.connect(self.hide)

        header_layout.addWidget(self.title)
        header_layout.addWidget(self.status)
        header_layout.addStretch(1)
        header_layout.addWidget(clear_btn)
        header_layout.addWidget(hide_btn)

        self.view = QTextBrowser()
        self.view.setObjectName("answer")
        self.view.setOpenExternalLinks(False)
        self.view.setReadOnly(True)

        grip = QSizeGrip(card)
        grip.setFixedSize(16, 16)

        card_layout.addWidget(self.header)
        card_layout.addWidget(self.view, 1)
        card_layout.addWidget(grip, 0, Qt.AlignRight | Qt.AlignBottom)

        self.setStyleSheet(
            """
            QFrame#card {
                background: rgba(18, 20, 26, 238);
                border: 1px solid rgba(255, 255, 255, 28);
                border-radius: 14px;
            }
            QLabel#title {
                color: #f2f5ff;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#status {
                color: #8b93a7;
                font-size: 12px;
            }
            QToolButton {
                color: #aab2c5;
                background: transparent;
                border: none;
                padding: 2px 6px;
            }
            QToolButton:hover {
                color: #ffffff;
                background: rgba(255, 255, 255, 18);
                border-radius: 6px;
            }
            QTextBrowser#answer {
                background: transparent;
                border: none;
                color: #e8eaf0;
                selection-background-color: #3b6ef6;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 45);
                border-radius: 4px;
                min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

        for widget in (self.header, self.title, self.status):
            widget.installEventFilter(self)

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(140)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start()

        self.apply_config(config)

    def apply_config(self, config: Dict[str, Any]) -> None:
        self._config = config
        ui = config.get("ui", {})
        font = QFont("Microsoft YaHei UI", int(ui.get("font_size", 14)))
        self.view.setFont(font)
        self.setWindowOpacity(float(ui.get("opacity", 0.94)))

    def place_default(self) -> None:
        ui = self._config.get("ui", {})
        x = ui.get("x")
        y = ui.get("y")
        if x is not None and y is not None:
            self.move(int(x), int(y))
            self._placed = True
            return

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(
            area.right() - self.width() - 24,
            area.top() + 72,
        )
        self._placed = True

    def show_overlay(self) -> None:
        if not self._placed:
            self.place_default()
        if not self.isVisible():
            self.show()
        self.raise_()

    def toggle(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show_overlay()

    def begin(self, status: str) -> None:
        # 上一轮回答保留在上面，新一轮从下面接着写，不覆盖正在看的内容
        if self._buffer.strip():
            self._history.append(self._buffer.strip())
            self._history = self._history[-3:]
        self._buffer = ""
        self._dirty = True
        self._flush()
        self.set_status(status)
        self.show_overlay()

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def append(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        self._dirty = True

    def finish(self) -> None:
        self._flush()
        self.set_status("完成")

    def error(self, message: str) -> None:
        self._buffer += f"\n\n> **出错：** {message}\n"
        self._dirty = True
        self._flush()
        self.set_status("出错")

    def clear(self) -> None:
        self._buffer = ""
        self._history.clear()
        self._dirty = False
        self.view.clear()
        self.set_status("待命")

    def discard_current(self) -> None:
        """丢掉本轮（例如模型判定“等待问题”），但保留之前的回答。"""
        self._buffer = ""
        self._dirty = True
        self._flush()
        self.set_status("等待问题")

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        blocks = [block for block in self._history if block.strip()]
        if self._buffer.strip():
            blocks.append(self._buffer)
        text = "\n\n---\n\n".join(blocks)

        bar = self.view.verticalScrollBar()
        at_bottom = bar.maximum() == 0 or bar.value() >= bar.maximum() - 4
        old_value = bar.value()

        self.view.setMarkdown(text)
        if at_bottom:
            bar.setValue(bar.maximum())
        else:
            # 用户正往回看旧回答时不要把他拽回底部
            bar.setValue(min(old_value, bar.maximum()))

    def geometry_config(self) -> Dict[str, Any]:
        geo = self.geometry()
        data: Dict[str, Any] = {
            "width": geo.width(),
            "height": geo.height(),
        }
        if self._placed:
            data["x"] = geo.x()
            data["y"] = geo.y()
        return data

    def eventFilter(self, obj, event) -> bool:
        if obj in (self.header, self.title, self.status):
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._drag_offset = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                return True
            if (
                event.type() == QEvent.Type.MouseMove
                and self._drag_offset is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_offset = None
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()
