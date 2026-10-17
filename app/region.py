from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RegionSelector(QWidget):
    """全屏半透明遮罩：拖拽框选一块区域，用于只截题目所在部分。"""

    selected = Signal(QRect)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self._origin: Optional[QPoint] = None
        self._current: Optional[QPoint] = None

        desktop = QRect()
        for screen in QGuiApplication.screens():
            desktop = desktop.united(screen.geometry())
        self.setGeometry(desktop)

    # ---- 绘制 ----

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 12, 18, 120))

        rect = self._local_selection()
        if rect is None or rect.width() < 4 or rect.height() < 4:
            painter.setPen(QColor("#dfe5f2"))
            painter.drawText(
                self.rect().adjusted(0, 60, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                "拖动鼠标框选题目区域 · 松开完成 · Esc / 右键取消",
            )
            return

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(rect, Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor("#3b6ef6"), 2))
        painter.drawRect(rect)

        label = f"{rect.width()} × {rect.height()}"
        painter.setPen(QColor("#ffffff"))
        painter.drawText(rect.adjusted(0, -22, 0, 0), Qt.AlignmentFlag.AlignLeft, label)

    # ---- 交互 ----

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        rect = self._local_selection()
        self._origin = None
        self._current = None
        self.update()
        if rect is None or rect.width() < 20 or rect.height() < 20:
            self.cancelled.emit()
            return
        self.selected.emit(
            QRect(self.mapToGlobal(rect.topLeft()), rect.size())
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.setFocus()

    def _local_selection(self) -> Optional[QRect]:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()
