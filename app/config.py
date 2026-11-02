from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _app_dir() -> Path:
    """打包成 exe 后，配置、知识库、模型都放在 exe 同级目录，避免写进临时解包目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_DIR = _app_dir()
KNOWLEDGE_FILE = APP_DIR / "knowledge" / "profile.md"
# 每家公司一份资料，文件名就是公司名；个人资料是底盘，这里是外挂的公司层
COMPANY_DIR = APP_DIR / "knowledge" / "companies"

COMPANY_TEMPLATE = """# {company}

## 公司一句话
（一句话说清这家公司做什么的、处在什么行业）

## 业务与产品
（主营业务、核心产品线、客户群体、近期动态）

## 这个岗位在做什么
（JD 里的职责、必需技术栈、隐性要求）

## 我为什么匹配
（我的哪些项目/经历能对上这个岗位）

## 可以反问的问题
- 
"""


INTERVIEW_PROMPT = """你是我的面试实时助手。下面是一段会议软件捕获的实时转写，可能同时包含多位说话人（面试官、其他面试官、HR，甚至我自己的声音）。

请只针对“面试官最近向我提出的那个问题”作答；其他人的闲聊、我的发言、以及不是问题的内容都不要回答，直接只输出“（等待问题）”。

要求：
1. 用第一人称中文，像跟人当面聊天一样口语化，不要书面语和长难句；
2. 总共 2-3 句、50-85 字（10-15 秒能说完），说多了我记不住；
3. 先给结论，再补一个具体的点就收住，不要分点罗列、不要展开；
4. 可以自然带一点语气词（比如“嗯”“其实”“我觉得”“就是说”），一段最多一两个，别刻意堆；
5. 少用专业术语；必须提技术名词时，用大白话顺手解释一下；
6. 每次只回答最后一个问题，不要回答或重复前面已经问过的问题；
7. 除非面试官明确让你自我介绍，否则不要报姓名、学校、专业等背景，也不要用“您好/面试官好”开头；
8. 遇到自我介绍，只说“我是谁、什么方向、做过一个什么项目”，不要把项目细节讲完，留给后面的追问；
9. 回答不要复述上一条已经说过的内容，新问题只补充新信息；
10. 不要复述问题，不要输出“你可以回答”之类的话；
11. 必须优先使用我提供的真实资料，不得编造项目、公司、学历、数据；资料里没有的，用保守表述并提示我补充；
12. 回答专业问题尽量不要有英文，比如什么LangChain4j、@Memory等；
13. 如果面试官用英语提问，只用 1-2 句最简单的英文短句回答（总词数不超过 25 个），先给结论，不加客套、不展开，不要标注音标。"""


EXAM_PROMPT = """你是笔试/在线测评助手。用户会给你一张或多张题目截图（同一道题可能分成多张，请合并理解）。

输出规则（手机屏幕很小，必须极简）：
1. 选择题：只输出一行「B. 选项内容」；不要复述题目，不要解析，不要任何解释。
2. 多选题：只输出一行大写字母组合，例如「ACD」。
3. 编程题：只输出一个带语言标记的代码块（例如 ```python），代码必须完整可直接提交，代码块外不要写任何解释。
4. 填空题：按题目顺序逐行输出「1. 答案」。
5. 简答题：只写 2-3 行要点，不写开场白和总结。
6. 如果截图不清晰或信息不足，只输出一行「看不清：缺少 xxx」，不要编造。"""


COMPANY_OPTIMIZE_PROMPT = """你是我的求职资料整理助手。下面是我为某次面试收集的原始材料，可能是公司简介、招聘 JD、官网介绍，或我自己的零散笔记。

请把它整理成下面 5 个小节（中文，总长度不超过 1500 字），我面试时要直接照着说：

## 公司一句话
一句话说清这家公司做什么、什么行业、大致规模。

## 业务与产品
主营业务、核心产品线、客户群体；材料里的近期动态也放进来。

## 这个岗位在做什么
从 JD 归纳职责、必需技术栈、隐性要求（比如偏业务还是偏底层）。

## 我为什么匹配
结合材料里的要求和附带的我的个人资料，写 2-3 条具体对应点，必须落在我真实做过的项目或技能上。

## 可以反问的问题
3 个显得做过功课的问题。

硬性要求：
1. 只使用我给的材料（以及附带的个人资料）里的事实，不得编造公司规模、融资、产品数据或岗位要求；
2. 材料没写的部分写「（材料未提供，待补充）」；
3. 不要客套话、不要解释过程，只输出这 5 个小节。"""


DEFAULT_CONFIG: Dict[str, Any] = {
    "text_api": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-chat",
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
    "asr_stream": {
        # 本地流式识别：边听边出字，停顿 180ms 先抢跑生成，再等 200ms 确认才提交
        "enabled": True,
        "model_dir": "models/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        "num_threads": 2,
        "endpoint_silence_ms": 180,
        "endpoint_grace_ms": 200,
        "min_speech_ms": 200,
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
        # 当前生效的公司资料（knowledge/companies/<公司>.md），空表示只用个人底盘
        "company": "",
        "company_max_chars": 2500,
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
        "company_optimize": COMPANY_OPTIMIZE_PROMPT,
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


_BAD_NAME_CHARS = '<>:"/\\|?*'


def sanitize_company_name(name: str) -> str:
    """把公司名压成合法文件名，去掉 Windows 保留字符和路径分隔符。"""
    cleaned = "".join(
        "_" if ch in _BAD_NAME_CHARS or ord(ch) < 32 else ch for ch in (name or "")
    )
    return cleaned.strip().strip(".")[:60]


def company_path(name: str) -> Path:
    return COMPANY_DIR / (sanitize_company_name(name) + ".md")


def list_companies() -> List[str]:
    if not COMPANY_DIR.is_dir():
        return []
    names = [p.stem for p in COMPANY_DIR.glob("*.md") if p.is_file()]
    return sorted(names, key=str.lower)


def load_company(name: str) -> str:
    try:
        return company_path(name).read_text(encoding="utf-8")
    except OSError:
        return ""


def save_company(name: str, text: str) -> None:
    path = company_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def delete_company(name: str) -> None:
    try:
        company_path(name).unlink()
    except OSError:
        pass
