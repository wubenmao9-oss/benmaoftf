from __future__ import annotations

import io

import qrcode
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class RemoteDialog(QDialog):
    """展示扫码地址与二维码；关闭窗口即断开手机端。"""

    stopped = Signal()

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        self.url = url
        self._silent = False
        self.setWindowTitle("手机远程")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setFixedWidth(430)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        title = QLabel("手机扫码打开")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title, 0, Qt.AlignHCenter)

        hint = QLabel(
            "手机与电脑需连同一个局域网。扫码后电脑端自动最小化，" + chr(10)
            + "转写、面试回答、截图答题都会显示在手机上。"
        )
        hint.setAlignment(Qt.AlignHCenter)
        hint.setStyleSheet("color: #8b93a7; font-size: 12px;")
        root.addWidget(hint)

        image = qrcode.make(url)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), "PNG")
        qr = QLabel()
        qr.setPixmap(
            pixmap.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        qr.setAlignment(Qt.AlignHCenter)
        root.addWidget(qr, 0, Qt.AlignHCenter)

        self.url_label = QLabel(url)
        self.url_label.setWordWrap(True)
        self.url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.url_label.setStyleSheet(
            "color: #9aa3b8; font-size: 12px; background: #1b1e26;"
            "border-radius: 6px; padding: 6px;"
        )
        root.addWidget(self.url_label)

        buttons = QHBoxLayout()
        copy_btn = QPushButton("复制链接")
        copy_btn.clicked.connect(self._copy)
        stop_btn = QPushButton("断开手机端")
        stop_btn.clicked.connect(self.reject)
        buttons.addWidget(copy_btn)
        buttons.addWidget(stop_btn)
        root.addLayout(buttons)

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.url)

    def close_silently(self) -> None:
        self._silent = True
        self.close()

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        if not self._silent:
            self.stopped.emit()
