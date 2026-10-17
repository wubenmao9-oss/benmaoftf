"""打包为「笨猫面试助手」（Windows，PySide6 + faster-whisper）。

用法：py build.py
产物：dist/笨猫面试助手/笨猫面试助手.exe（配置、知识库、本地语音模型会一起拷进去）
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "笨猫面试助手"


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        NAME,
        "--add-data",
        "app/mobile.html;app",
        "--collect-all",
        "faster_whisper",
        "--collect-all",
        "ctranslate2",
        "--collect-all",
        "tokenizers",
        "--collect-all",
        "av",
        "--hidden-import",
        "soundcard",
        "--hidden-import",
        "soundcard.mediafoundation",
        "--hidden-import",
        "mss",
        "--hidden-import",
        "qrcode",
        "--hidden-import",
        "_cffi_backend",
        "main.py",
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("PyInstaller 打包失败")
        return result.returncode

    dist = ROOT / "dist" / NAME
    config = ROOT / "config.json"
    if config.is_file():
        shutil.copy2(config, dist / "config.json")

    knowledge = ROOT / "knowledge"
    if knowledge.is_dir():
        shutil.copytree(knowledge, dist / "knowledge", dirs_exist_ok=True)

    models = ROOT / "models"
    if models.is_dir():
        shutil.copytree(models, dist / "models", dirs_exist_ok=True)

    print()
    print(f"已生成：{dist / (NAME + '.exe')}")
    print("配置文件、知识库和本地语音模型已放在同一目录，可直接拷走使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
