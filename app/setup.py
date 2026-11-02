"""新用户初始化：下载本地语音模型 + 检测运行环境。

写操作只发生在 models/ 下各自的目标目录，不会改动 config.json，
也不会删除或覆盖任何已有的模型文件。
"""

from __future__ import annotations

import copy
import importlib.util
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from .config import APP_DIR, config_path, load_config

# 优先国内镜像；失败再回落官方源
MIRRORS = ("https://hf-mirror.com", "https://huggingface.co")


@dataclass(frozen=True)
class PackageFile:
    name: str
    size: int


@dataclass(frozen=True)
class Package:
    key: str
    label: str
    repo: str
    folder: str
    files: Sequence[PackageFile]

    @property
    def local_dir(self) -> Path:
        return APP_DIR / "models" / self.folder

    @property
    def rel_dir(self) -> str:
        return f"models/{self.folder}"


PACKAGES = (
    Package(
        key="zh",
        label="中文高精度（2025 版，约 160 MB，推荐）",
        repo="csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        folder="sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        files=(
            PackageFile("tokens.txt", 20628),
            PackageFile("encoder.int8.onnx", 161141793),
            PackageFile("decoder.onnx", 5165083),
            PackageFile("joiner.int8.onnx", 1033416),
        ),
    ),
    Package(
        key="bilingual",
        label="中英双语（2023 版，约 190 MB，英文更好）",
        repo="csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        folder="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        files=(
            PackageFile("tokens.txt", 56317),
            PackageFile("encoder-epoch-99-avg-1.int8.onnx", 181895032),
            PackageFile("decoder-epoch-99-avg-1.int8.onnx", 13091040),
            PackageFile("joiner-epoch-99-avg-1.int8.onnx", 3228404),
        ),
    ),
)

PACKAGE_BY_KEY = {package.key: package for package in PACKAGES}


# ---------------------------------------------------------------- 环境检测


@dataclass
class CheckResult:
    level: str  # ok / warn / fail
    title: str
    detail: str


LEVEL_MARK = {"ok": "[正常]", "warn": "[提醒]", "fail": "[需处理]"}
LEVEL_TEXT = {
    "ok": "环境检测通过，可以直接用。",
    "warn": "可以运行，但有可优化的地方（下面标了「提醒」）。",
    "fail": "还不能完整使用，请先处理下面标「需处理」的问题。",
}


def _resolve_dir(raw: str) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else APP_DIR / path


def _model_missing(model_dir: Optional[Path]) -> List[str]:
    """和 streaming_asr 一样按文件名前缀找模型，兼容不同导出命名。"""
    if model_dir is None or not model_dir.is_dir():
        return ["模型目录不存在"]
    missing: List[str] = []
    if not (model_dir / "tokens.txt").is_file():
        missing.append("tokens.txt")
    for prefix in ("encoder", "decoder", "joiner"):
        found = False
        for path in model_dir.glob(prefix + "*.onnx"):
            try:
                if path.is_file() and path.stat().st_size > 1024:
                    found = True
                    break
            except OSError:
                continue
        if not found:
            missing.append(prefix + ".onnx")
    return missing


def check_environment(config: Optional[Dict[str, Any]] = None) -> List[CheckResult]:
    cfg = config if isinstance(config, dict) else load_config()
    results: List[CheckResult] = []

    path = config_path()
    if path.is_file():
        results.append(CheckResult("ok", "配置文件", f"已找到 {path}"))
    else:
        results.append(
            CheckResult("warn", "配置文件", "还没有 config.json，保存设置时会自动生成")
        )

    text_api = cfg.get("text_api") or {}
    base_url = str(text_api.get("base_url") or "").strip()
    api_key = str(text_api.get("api_key") or "").strip()
    if not base_url.lower().startswith("http"):
        results.append(
            CheckResult(
                "fail",
                "文本模型地址",
                "Base URL 为空或格式不对，应类似 https://api.deepseek.com/v1",
            )
        )
    elif not api_key:
        results.append(
            CheckResult(
                "fail",
                "文本模型 Key",
                "还没填 API Key，AI 不会生成回答。"
                "到「模型与音频」填好 Key，再点「保存设置」",
            )
        )
    else:
        results.append(
            CheckResult(
                "ok",
                "文本模型 Key",
                f"已配置（模型：{text_api.get('model') or '未填'}）",
            )
        )

    if str((cfg.get("vision_api") or {}).get("api_key") or "").strip():
        results.append(CheckResult("ok", "视觉模型 Key", "已配置，截图识题可用"))
    else:
        results.append(
            CheckResult(
                "warn",
                "视觉模型 Key",
                "没填不影响语音面试；截图答题需要支持图片输入的模型",
            )
        )

    asr = cfg.get("asr_stream") or {}
    raw_dir = str(asr.get("model_dir") or "").strip()
    if not raw_dir:
        results.append(
            CheckResult("fail", "本地语音模型", "配置里没有指定模型目录，点「一键初始化」自动下载")
        )
    else:
        missing = _model_missing(_resolve_dir(raw_dir))
        if missing:
            results.append(
                CheckResult(
                    "fail",
                    "本地语音模型",
                    f"{raw_dir} 缺少 {', '.join(missing)}；点上方「一键初始化」自动下载",
                )
            )
        else:
            results.append(CheckResult("ok", "本地语音模型", raw_dir))

    try:
        available = importlib.util.find_spec("sherpa_onnx") is not None
    except Exception:  # noqa: BLE001
        available = False
    if available:
        results.append(CheckResult("ok", "语音识别组件", "sherpa-onnx 可用"))
    else:
        results.append(
            CheckResult(
                "fail",
                "语音识别组件",
                "缺少 sherpa-onnx。源码运行请 pip install -r requirements.txt；"
                "正式版请用 Release 里的完整包",
            )
        )

    source = str((cfg.get("audio") or {}).get("source") or "loopback")
    try:
        import soundcard as sc

        if source == "microphone":
            mic = sc.default_microphone()
            if mic is None:
                results.append(
                    CheckResult("fail", "音频设备（麦克风）", "没有找到可用的麦克风")
                )
            else:
                results.append(CheckResult("ok", "音频设备（麦克风）", mic.name))
        else:
            speaker = sc.default_speaker()
            loopbacks = [
                device
                for device in sc.all_microphones(include_loopback=True)
                if getattr(device, "isloopback", False)
            ]
            if speaker is None:
                results.append(
                    CheckResult("fail", "音频设备（系统回环）", "没有找到默认扬声器")
                )
            elif not loopbacks:
                results.append(
                    CheckResult(
                        "fail",
                        "音频设备（系统回环）",
                        "没有找到回环设备，无法采集会议声音；可改用麦克风模式",
                    )
                )
            elif any(device.name == speaker.name for device in loopbacks):
                results.append(
                    CheckResult(
                        "ok",
                        "音频设备（系统回环）",
                        f"默认扬声器：{speaker.name}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "warn",
                        "音频设备（系统回环）",
                        f"回环设备有 {len(loopbacks)} 个，但没有和默认扬声器"
                        f"（{speaker.name}）同名的；开始监听可能失败",
                    )
                )
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("fail", "音频设备", f"读取音频设备失败：{exc}"))

    profile = APP_DIR / "knowledge" / "profile.md"
    try:
        knowledge = profile.read_text(encoding="utf-8").strip() if profile.is_file() else ""
    except OSError:
        knowledge = ""
    if len(knowledge) > 20:
        results.append(CheckResult("ok", "个人知识库", "已填写"))
    else:
        results.append(
            CheckResult(
                "warn",
                "个人知识库",
                "还没填简历资料，回答会比较通用（不影响使用）",
            )
        )

    return results


def overall_level(results: Sequence[CheckResult]) -> str:
    levels = {item.level for item in results}
    if "fail" in levels:
        return "fail"
    if "warn" in levels:
        return "warn"
    return "ok"


def format_results(results: Sequence[CheckResult]) -> str:
    return "\n".join(
        f"{LEVEL_MARK.get(item.level, '[提醒]')} {item.title}：{item.detail}"
        for item in results
    )


# ---------------------------------------------------------------- 下载模型


class _Cancelled(Exception):
    pass


class DownloadWorker(QThread):
    """后台下载语音模型；只写 models/ 下的目标文件，支持断点续传。"""

    progress = Signal(int, int)  # 已下载字节, 总字节
    message = Signal(str)
    failed = Signal(str)
    finished_ok = Signal(str)  # 已下载模型的相对目录

    def __init__(self, packages: Sequence[Package], parent=None) -> None:
        super().__init__(parent)
        self._packages = list(packages)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            total = sum(item.size for pkg in self._packages for item in pkg.files)
            done = 0
            for pkg in self._packages:
                pkg.local_dir.mkdir(parents=True, exist_ok=True)
                for item in pkg.files:
                    dest = pkg.local_dir / item.name
                    if dest.is_file() and dest.stat().st_size == item.size:
                        done += item.size
                        self.progress.emit(done, total)
                        continue
                    self.message.emit(f"正在下载 {item.name}（{item.size / 1048576:.0f} MB）")
                    self._fetch(pkg, item, dest, done, total)
                    done += item.size
                    self.progress.emit(done, total)
            self.finished_ok.emit(self._packages[0].rel_dir)
        except _Cancelled:
            self.failed.emit("已取消下载")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"下载失败：{exc}")

    def _fetch(self, pkg: Package, item: PackageFile, dest: Path, base: int, total: int) -> None:
        part = dest.with_name(dest.name + ".part")
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            if self._cancel:
                raise _Cancelled
            try:
                self._fetch_once(pkg, item, part, base, total)
                size = part.stat().st_size
                if item.size and size != item.size:
                    raise RuntimeError(f"文件大小不对（{size}/{item.size}），可能下载中断")
                part.replace(dest)
                return
            except _Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.message.emit(f"{item.name} 第 {attempt} 次失败：{exc}")
                if attempt >= 3:
                    break
                time.sleep(2)
        raise RuntimeError(str(last_error))

    def _fetch_once(
        self, pkg: Package, item: PackageFile, part: Path, base: int, total: int
    ) -> None:
        start = part.stat().st_size if part.is_file() else 0
        if item.size and start > item.size:
            part.unlink()  # 上次下多了，重新来
            start = 0
        last_error: Optional[Exception] = None
        for mirror in MIRRORS:
            if self._cancel:
                raise _Cancelled
            url = f"{mirror}/{pkg.repo}/resolve/main/{item.name}"
            headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
            if start:
                headers["Range"] = f"bytes={start}-"
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=30) as response:
                    status = int(getattr(response, "status", 200))
                    if start and status != 206:
                        # 服务器不支持续传，只能从头下
                        start = 0
                    mode = "ab" if start else "wb"
                    downloaded = start
                    with part.open(mode) as output:
                        while True:
                            if self._cancel:
                                raise _Cancelled
                            chunk = response.read(262144)
                            if not chunk:
                                break
                            output.write(chunk)
                            downloaded += len(chunk)
                            self.progress.emit(
                                base + min(downloaded, item.size or downloaded), total
                            )
                return
            except _Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.message.emit(f"{mirror} 连不上，换下一个地址…")
        raise RuntimeError(str(last_error))


# ---------------------------------------------------------------- 初始化窗口


class SetupDialog(QDialog):
    """一键初始化 / 环境检测。"""

    model_ready = Signal(str)

    def __init__(self, config: Dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._config = copy.deepcopy(config)
        self._worker: Optional[DownloadWorker] = None
        self.setWindowTitle("新手初始化向导")
        self.setMinimumWidth(640)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        intro = QLabel(
            "第一次使用（或换了台电脑）：先下载本地语音识别包，再填 API Key，"
            "最后跑一次环境检测。下载只写入 models/ 下的模型目录，"
            "不会改动你已经能用的配置和已有模型。"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("语音识别包"))
        self.package_combo = QComboBox()
        self.package_combo.addItem("中文高精度（2025 版，约 160 MB，推荐）", ["zh"])
        self.package_combo.addItem("中英双语（2023 版，约 190 MB，英文更好）", ["bilingual"])
        self.package_combo.addItem("两个都下载（约 350 MB）", ["zh", "bilingual"])
        row.addWidget(self.package_combo, 1)
        root.addLayout(row)

        buttons = QHBoxLayout()
        self.download_btn = QPushButton("开始下载")
        self.download_btn.clicked.connect(self._start_download)
        self.cancel_btn = QPushButton("取消下载")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_download)
        buttons.addWidget(self.download_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("等待开始")
        root.addWidget(self.progress)

        self.status = QLabel("下载完成后会自动检测环境")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        root.addWidget(QLabel("环境检测："))
        self.result_view = QPlainTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setMinimumHeight(180)
        root.addWidget(self.result_view, 1)

        bottom = QHBoxLayout()
        refresh_btn = QPushButton("重新检测")
        refresh_btn.clicked.connect(self._refresh_checks)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(refresh_btn)
        bottom.addStretch(1)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

        self._refresh_checks()

    def _refresh_checks(self) -> None:
        results = check_environment(self._config)
        self.result_view.setPlainText(format_results(results))
        self.status.setText(LEVEL_TEXT.get(overall_level(results), ""))

    def _start_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        keys = self.package_combo.currentData() or ["zh"]
        packages = [PACKAGE_BY_KEY[key] for key in keys if key in PACKAGE_BY_KEY]
        if not packages:
            return
        worker = DownloadWorker(packages, self)
        worker.progress.connect(self._on_progress)
        worker.message.connect(self.status.setText)
        worker.failed.connect(self._on_failed)
        worker.finished_ok.connect(self._on_download_done)
        worker.finished.connect(self._on_thread_finished)
        self._worker = worker
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.status.setText("开始下载，约 160 MB，视网络需要几分钟…")
        self.download_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.package_combo.setEnabled(False)
        worker.start()

    def _cancel_download(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status.setText("正在取消…")

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            return
        self.progress.setValue(int(done * 100 / total))
        self.progress.setFormat(
            f"{done / 1048576:.1f} MB / {total / 1048576:.1f} MB（%p%）"
        )

    def _on_failed(self, message: str) -> None:
        self.status.setText(message)
        self.progress.setFormat("下载未完成")
        QMessageBox.warning(self, "下载未完成", message)

    def _on_download_done(self, rel_dir: str) -> None:
        self._config.setdefault("asr_stream", {})
        self._config["asr_stream"]["model_dir"] = rel_dir
        self.progress.setValue(100)
        self.progress.setFormat("下载完成")
        self.status.setText("模型已就绪，正在检测环境…")
        self.model_ready.emit(rel_dir)
        self._refresh_checks()

    def _on_thread_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self.download_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.package_combo.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self, "正在下载", "下载还没结束，先点「取消下载」再关闭窗口。"
            )
            event.ignore()
            return
        event.accept()
