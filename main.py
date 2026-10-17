from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QLockFile, QStandardPaths
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from app.app import CopilotApp
from app.config import load_config


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("笨猫面试助手")
    app.setQuitOnLastWindowClosed(False)
    app.setFont(QFont("Microsoft YaHei UI", 10))

    # 单实例锁：重复双击会抢热键和托盘，直接提示已运行
    lock_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
    lock = QLockFile(str(Path(lock_dir) / "bengmao-interview-copilot.lock"))
    if not lock.tryLock(100):
        QMessageBox.information(
            None, "笨猫面试助手", "程序已经在运行了，请在系统托盘里查看。"
        )
        return 0

    controller = CopilotApp(app, load_config())
    controller.start()
    code = app.exec()
    lock.unlock()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
