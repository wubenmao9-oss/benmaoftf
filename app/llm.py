from __future__ import annotations

import base64
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible client for chat, vision and speech-to-text."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: Tuple[float, float] = (15.0, 300.0),
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _check(self, response: requests.Response) -> None:
        if response.status_code >= 400:
            body = response.text[:800]
            raise LLMError(f"HTTP {response.status_code}: {body}")

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        with requests.post(
            self._url("/chat/completions"),
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as response:
            self._check(response)
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = item.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content is None:
                    content = (choice.get("message") or {}).get("content")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )
                if content:
                    yield content

    def transcribe(
        self,
        wav_bytes: bytes,
        language: str = "zh",
        prompt: Optional[str] = None,
    ) -> str:
        data: Dict[str, str] = {"model": self.model}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        def post(payload: Dict[str, str]) -> requests.Response:
            return requests.post(
                self._url("/audio/transcriptions"),
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("chunk.wav", wav_bytes, "audio/wav")},
                data=payload,
                timeout=(15.0, 120.0),
            )

        response = post(data)
        if response.status_code == 400 and "language" in data:
            # 部分 ASR 服务不识别 language 参数，去掉后重试一次。
            fallback = dict(data)
            fallback.pop("language", None)
            response = post(fallback)
        self._check(response)
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"ASR 返回非 JSON：{response.text[:300]}") from exc
        return (body.get("text") or "").strip()

    def stream_vision(
        self,
        image_bytes: Any,
        system_prompt: str,
        user_text: str,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        if isinstance(image_bytes, (bytes, bytearray)):
            images: List[bytes] = [bytes(image_bytes)]
        else:
            images = [bytes(item) for item in image_bytes]

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image in images:
            image_b64 = base64.b64encode(image).decode("ascii")
            mime = "image/jpeg" if image[:3] == b"\xff\xd8\xff" else "image/png"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                }
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        yield from self.stream_chat(messages, temperature=temperature)
