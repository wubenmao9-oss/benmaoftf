from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import APP_DIR

NOTES_DIR = APP_DIR / "notes"

_TIME_PREFIX = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s*")


def clean_question(text: str) -> str:
    """去掉转写时间戳和空行，把被 ASR 切成多段的一句话拼回来。"""
    parts = []
    for raw in (text or "").splitlines():
        line = _TIME_PREFIX.sub("", raw).strip()
        if line:
            parts.append(line)
    return " ".join(parts).strip()


class NoteBook:
    """收集本次面试 / 笔试的题目，保存成 markdown（只存题目，不存回答）。"""

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []
        self._started = datetime.now()

    @property
    def count(self) -> int:
        return len(self._items)

    def add_interview(self, question: str) -> bool:
        text = clean_question(question)
        if not text:
            return False
        if any(item["kind"] == "interview" and item["text"] == text for item in self._items):
            return False
        self._items.append({"kind": "interview", "text": text, "images": []})
        return True

    def add_exam(self, images: List[bytes]) -> bool:
        if not images:
            return False
        self._items.append({"kind": "exam", "text": "", "images": list(images)})
        return True

    def save(self) -> Optional[Path]:
        """把已收集的题目写成 markdown；没有题目时返回 None。"""
        if not self._items:
            return None

        now = datetime.now()
        stamp = now.strftime("%Y%m%d-%H%M%S")
        folder = NOTES_DIR / f"题目-{stamp}"
        folder.mkdir(parents=True, exist_ok=True)

        interviews = [item for item in self._items if item["kind"] == "interview"]
        exams = [item for item in self._items if item["kind"] == "exam"]

        lines = [
            "# 面试题目笔记",
            "",
            f"- 开始时间：{self._started.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 保存时间：{now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 题目数量：{len(interviews) + len(exams)}",
            "",
        ]

        if interviews:
            lines += [f"## 面试题（{len(interviews)}）", ""]
            for index, item in enumerate(interviews, start=1):
                lines.append(f"{index}. {item['text']}")
            lines.append("")

        if exams:
            lines += [f"## 笔试题（{len(exams)}）", ""]
            image_dir = folder / "题目截图"
            image_dir.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(exams, start=1):
                names = []
                for img_index, data in enumerate(item["images"], start=1):
                    name = f"第{index}题-{img_index}.png"
                    (image_dir / name).write_bytes(data)
                    names.append(name)
                links = " ".join(f"![第{index}题截图](题目截图/{name})" for name in names)
                lines.append(f"{index}. {links}")
            lines.append("")

        path = folder / "题目.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        self._items = []
        return path
