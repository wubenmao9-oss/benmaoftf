from __future__ import annotations

from typing import Optional, Tuple

Rect = Tuple[int, int, int, int]


def _encode_png(png_bytes: bytes) -> bytes:
    """统一压成 JPEG，避免大图把视觉请求撑爆。"""
    try:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        if image.width > 2560:
            ratio = 2560 / image.width
            image = image.resize(
                (2560, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
        return buffer.getvalue()
    except Exception:
        # Pillow 不可用时退回原始 PNG，避免直接截不了图。
        return png_bytes


def grab_screen(rect: Optional[Rect] = None) -> bytes:
    """截取屏幕；rect 为物理像素 (left, top, right, bottom)，缺省截整个虚拟桌面。"""
    import mss
    import mss.tools

    with mss.mss() as sct:
        if rect is None:
            monitor = sct.monitors[0]
        else:
            left, top, right, bottom = rect
            width = max(1, right - left)
            height = max(1, bottom - top)
            monitor = {"left": left, "top": top, "width": width, "height": height}
        shot = sct.grab(monitor)
        png_bytes = mss.tools.to_png(shot.rgb, shot.size)

    return _encode_png(png_bytes)


def grab_desktop_image() -> bytes:
    """Capture the whole virtual desktop (all monitors) as JPEG/PNG bytes."""
    return grab_screen(None)


def active_window_rect() -> Optional[Rect]:
    """返回前台窗口的物理像素矩形；取不到（或已最小化）时返回 None。"""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        # 最小化窗口会返回 -32000 之类的坐标
        if rect.left <= -30000 or rect.top <= -30000:
            return None
        if rect.right - rect.left < 40 or rect.bottom - rect.top < 40:
            return None
        return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return None


def logical_to_physical(rect: "object") -> Optional[Rect]:
    """把 Qt 的逻辑坐标矩形换算成截图用的物理像素矩形。

    多显示器且缩放比例不一致时会有少量偏差，常见同比例缩放下是精确的。
    """
    try:
        import mss
        from PySide6.QtGui import QGuiApplication

    except Exception:
        return None

    try:
        desktop = None
        for screen in QGuiApplication.screens():
            geo = screen.geometry()
            desktop = geo if desktop is None else desktop.united(geo)
        if desktop is None or desktop.width() <= 0 or desktop.height() <= 0:
            return None

        with mss.mss() as sct:
            monitor = sct.monitors[0]

        scale_x = monitor["width"] / desktop.width()
        scale_y = monitor["height"] / desktop.height()
        left = monitor["left"] + int(round((rect.x() - desktop.x()) * scale_x))
        top = monitor["top"] + int(round((rect.y() - desktop.y()) * scale_y))
        right = left + int(round(rect.width() * scale_x))
        bottom = top + int(round(rect.height() * scale_y))
        if right - left < 8 or bottom - top < 8:
            return None
        return (left, top, right, bottom)
    except Exception:
        return None
