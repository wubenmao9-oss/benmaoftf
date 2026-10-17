from __future__ import annotations

from typing import Any, Dict, List

from PySide6.QtCore import QObject, Signal


class HotkeyManager(QObject):
    """Register global hotkeys and re-emit them as Qt signals."""

    toggle_overlay = Signal()
    screenshot = Signal()
    toggle_listen = Signal()
    answer_now = Signal()
    quit = Signal()
    failed = Signal(str)

    def __init__(self, hotkeys: Dict[str, str], parent=None) -> None:
        super().__init__(parent)
        self._hotkeys = dict(hotkeys or {})
        self._handles: List[Any] = []
        self._keyboard = None

    def start(self) -> None:
        try:
            import keyboard
        except Exception as exc:
            self.failed.emit(f"加载全局热键库失败：{exc}")
            return

        self._keyboard = keyboard
        mapping = [
            ("toggle_overlay", self.toggle_overlay),
            ("screenshot", self.screenshot),
            ("toggle_listen", self.toggle_listen),
            ("answer_now", self.answer_now),
            ("quit", self.quit),
        ]
        for name, signal in mapping:
            combo = (self._hotkeys.get(name) or "").strip()
            if not combo:
                continue
            try:
                handle = keyboard.add_hotkey(combo, signal.emit, suppress=False)
                self._handles.append(handle)
            except Exception as exc:
                self.failed.emit(f"注册热键 {combo} 失败：{exc}")

    def stop(self) -> None:
        if self._keyboard is None:
            return
        for handle in self._handles:
            try:
                self._keyboard.remove_hotkey(handle)
            except Exception:
                pass
        self._handles.clear()
        self._keyboard = None
