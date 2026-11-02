"""打包为「笨猫面试助手」（Windows，PySide6 + faster-whisper）。

用法：
    py build.py            # 自己用：连配置、知识库、语音模型一起拷进去
    py build.py --share    # 分享给别人：配置清空 Key，不带知识库和模型，
                           # 对方启动后点「一键初始化」自己下载

产物：dist/笨猫面试助手/笨猫面试助手.exe
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent
NAME = "笨猫面试助手"


def main() -> int:
    share = "--share" in sys.argv[1:]
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--additional-hooks-dir",
        "hooks",
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
        "--collect-all",
        "sherpa_onnx",
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
    if share:
        # 分享版：Key 留空，知识库和模型都不带，由对方在程序里自己下载
        payload = json.loads(json.dumps(DEFAULT_CONFIG))
        payload["text_api"]["api_key"] = ""
        payload["vision_api"]["api_key"] = ""
        (dist / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        config = ROOT / "config.json"
        if config.is_file():
            shutil.copy2(config, dist / "config.json")

        knowledge = ROOT / "knowledge"
        if knowledge.is_dir():
            shutil.copytree(knowledge, dist / "knowledge", dirs_exist_ok=True)

        models = ROOT / "models"
        if models.is_dir():
            shutil.copytree(
                models,
                dist / "models",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("_dl"),
            )

    print()
    print(f"已生成：{dist / (NAME + '.exe')}")
    if share:
        print("分享版：Key 已清空，未包含知识库和语音模型；对方启动后点「一键初始化」即可。")
    else:
        print("配置文件、知识库和本地语音模型已放在同一目录，可直接拷走使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
