from __future__ import annotations

import json
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from PySide6.QtCore import QObject, Signal

MOBILE_PAGE = Path(__file__).resolve().parent / "mobile.html"


def lan_ip() -> str:
    """取本机在局域网中的 IPv4 地址。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


class RemoteState:
    """手机端轮询用的共享状态，所有读写都加锁。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "待命"
        self._listening = False
        self._auto = True
        self._shots = 0
        self._transcript = ""
        self._answers: List[Dict[str, Any]] = []
        self._answer_id = 0

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    def set_listening(self, value: bool) -> None:
        with self._lock:
            self._listening = bool(value)

    def set_auto(self, value: bool) -> None:
        with self._lock:
            self._auto = bool(value)

    def set_shots(self, count: int) -> None:
        with self._lock:
            self._shots = max(0, int(count))

    def add_transcript(self, line: str) -> None:
        with self._lock:
            self._transcript += (chr(10) if self._transcript else "") + line
            if len(self._transcript) > 8000:
                self._transcript = self._transcript[-8000:]

    def begin_answer(self, kind: str) -> None:
        with self._lock:
            self._answer_id += 1
            self._answers.append(
                {"id": self._answer_id, "kind": kind, "text": "", "done": False, "error": ""}
            )
            self._answers = self._answers[-20:]

    def append_answer(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self._answers:
                self._answers[-1]["text"] += text

    def end_answer(self, error: str = "") -> None:
        with self._lock:
            if self._answers:
                self._answers[-1]["done"] = True
                if error:
                    self._answers[-1]["error"] = error

    def drop_answer(self) -> None:
        """丢弃最后一条尚未结束的回答（例如模型判断为“等待问题”）。"""
        with self._lock:
            if self._answers and not self._answers[-1]["done"]:
                self._answers.pop()

    def clear(self) -> None:
        with self._lock:
            self._transcript = ""
            self._answers = []
            self._shots = 0
            self._status = "待命"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "listening": self._listening,
                "auto": self._auto,
                "shots": self._shots,
                "transcript": self._transcript,
                "answers": [dict(item) for item in self._answers],
            }


class RemoteServer(QObject):
    """局域网 HTTP 服务：手机扫码打开页面，轮询状态并远程触发操作。"""

    screenshot_requested = Signal()
    answer_requested = Signal()
    listen_requested = Signal(bool)
    auto_requested = Signal(bool)
    shot_add_requested = Signal()
    shot_submit_requested = Signal()
    clear_requested = Signal()
    phone_connected = Signal()

    def __init__(self, state: RemoteState, port: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.token = secrets.token_urlsafe(9)
        self._port = int(port or 0)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self.url = ""

    @property
    def running(self) -> bool:
        return self._httpd is not None

    def start(self) -> str:
        if self._httpd is not None:
            return self.url
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self._port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        port = self._httpd.server_address[1]
        self.url = f"http://{lan_ip()}:{port}/?t={self.token}"
        return self.url

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        self._thread = None
        self._connected = False
        self.url = ""

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # noqa: D401
                return

            def _authorized(self, query: Dict[str, List[str]]) -> bool:
                return query.get("t", [""])[0] == server.token

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if not self._authorized(query):
                    self._send(403, b"forbidden", "text/plain; charset=utf-8")
                    return
                if parsed.path in ("/", "/index.html"):
                    try:
                        html = MOBILE_PAGE.read_bytes()
                    except OSError as exc:
                        self._send(500, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                        return
                    self._send(200, html, "text/html; charset=utf-8")
                    if not server._connected:
                        server._connected = True
                        server.phone_connected.emit()
                elif parsed.path == "/api/state":
                    body = json.dumps(
                        server.state.snapshot(), ensure_ascii=False
                    ).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain; charset=utf-8")

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if not self._authorized(query):
                    self._send(403, b"forbidden", "text/plain; charset=utf-8")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if parsed.path == "/api/screenshot":
                    server.screenshot_requested.emit()
                elif parsed.path == "/api/answer":
                    server.answer_requested.emit()
                elif parsed.path == "/api/listen":
                    try:
                        data = json.loads(raw or b"{}")
                    except json.JSONDecodeError:
                        data = {}
                    server.listen_requested.emit(bool(data.get("on")))
                elif parsed.path == "/api/auto":
                    try:
                        data = json.loads(raw or b"{}")
                    except json.JSONDecodeError:
                        data = {}
                    server.auto_requested.emit(bool(data.get("on")))
                elif parsed.path == "/api/shot/add":
                    server.shot_add_requested.emit()
                elif parsed.path == "/api/shot/submit":
                    server.shot_submit_requested.emit()
                elif parsed.path == "/api/clear":
                    server.clear_requested.emit()
                else:
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                self._send(200, b'{"ok":true}', "application/json; charset=utf-8")

        return Handler
