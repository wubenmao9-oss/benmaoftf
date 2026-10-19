from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _app_dir() -> Path:
    """打包成 exe 后，配置、知识库、模型都放在 exe 同级目录，避免写进临时解包目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_DIR = _app_dir()
KNOWLEDGE_FILE = APP_DIR / "knowledge" / "profile.md"


INTERVIEW_PROMPT = """你是我的面试实时助手。下面是一段会议软件捕获的实时转写，可能同时包含多位说话人（面试官、其他面试官、HR，甚至我自己的声音）。

请只针对“面试官最近向我提出的那个问题”作答；其他人的闲聊、我的发言、以及不是问题的内容都不要回答，直接只输出“（等待问题）”。

要求：
1. 用常见、自然的口语表达，句子简洁；不要生僻词、书面语和长难句，但可以适当用正常的专业词汇；
2. 默认用第一人称中文，控制在 30 秒内（约 120-180 字）；
3. 先给结论，再给 2-3 个要点，必要时补一个具体例子；
4. 不要复述问题，不要输出“你可以回答”之类的话；
5. 必须优先使用我提供的真实资料，不得编造项目、公司、学历、数据；资料里没有的，用保守表述并提示我补充；
6. 技术问题给出关键思路即可，不要写大段代码；
7. 如果面试官用英语提问，只用 1-2 句最简单的英文短句回答（总词数不超过 25 个），先给结论，不加客套、不展开，不要标注音标。"""


EXAM_PROMPT = """你是笔试/在线测评助手。用户会给你一张或多张题目截图（同一道题可能分成多张，请合并理解）。

输出规则（手机屏幕很小，必须极简）：
1. 选择题：只输出一行「B. 选项内容」；不要复述题目，不要解析，不要任何解释。
2. 多选题：只输出一行大写字母组合，例如「ACD」。
3. 编程题：只输出一个带语言标记的代码块（例如 ```python），代码必须完整可直接提交，代码块外不要写任何解释。
4. 填空题：按题目顺序逐行输出「1. 答案」。
5. 简答题：只写 2-3 行要点，不写开场白和总结。
6. 如果截图不清晰或信息不足，只输出一行「看不清：缺少 xxx」，不要编造。"""


DEFAULT_CONFIG: Dict[str, Any] = {
    "text_api": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-v4-pro",
        "temperature": 0.4,
    },
    "vision_api": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-flash",
        "temperature": 0.2,
    },
    "asr_api": {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": "",
        "model": "FunAudioLLM/SenseVoiceSmall",
        "language": "",
    },
    "asr_local": {
        "enabled": True,
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
        "download_root": "models/whisper",
        "hf_endpoint": "https://hf-mirror.com",
    },
    "audio": {
        "source": "loopback",
        "chunk_seconds": 4.0,
        "samplerate": 16000,
        "silence_rms": 0.003,
    },
    "capture": {
        # window=只截当前活动窗口；region=手动框选；full=所有显示器的整个桌面
        "mode": "window",
    },
    "knowledge": {
        "enabled": True,
        "max_chars": 12000,
    },
    "answer": {
        "auto": True,
        "question_only": True,
        "debounce_seconds": 1.5,
        "min_new_chars": 8,
        "max_context_chars": 4000,
    },
    "ui": {
        "opacity": 0.94,
        "font_size": 14,
        "width": 580,
        "height": 460,
        "x": None,
        "y": None,
    },
    "hotkeys": {
        "toggle_overlay": "ctrl+alt+h",
        "screenshot": "ctrl+alt+s",
        "toggle_listen": "ctrl+alt+l",
        "answer_now": "ctrl+alt+enter",
        "quit": "ctrl+alt+shift+q",
    },
    "prompts": {
        "interview": INTERVIEW_PROMPT,
        "exam": EXAM_PROMPT,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_path() -> Path:
    custom = os.environ.get("COPILOT_CONFIG")
    if custom:
        return Path(custom).expanduser()
    return APP_DIR / "config.json"


def load_config() -> Dict[str, Any]:
    path = config_path()
    if not path.exists():
        save_config(DEFAULT_CONFIG)
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        return copy.deepcopy(DEFAULT_CONFIG)
    return deep_merge(DEFAULT_CONFIG, raw)


def save_config(config: Dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_knowledge() -> str:
    try:
        return KNOWLEDGE_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


def save_knowledge(text: str) -> None:
    KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_FILE.write_text(text, encoding="utf-8")
