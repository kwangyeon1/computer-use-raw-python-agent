from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .models import GeneratedCode, PromptBundle
from .qwen_runtime import default_qwen35_model_id
from .runtime import extract_python_code


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                    continue
                if "text" in item:
                    parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


class Qwen35SGLangRuntime:
    def __init__(
        self,
        *,
        model_id: str | Path | None = None,
        max_new_tokens: int = 256,
        server_host: str = "127.0.0.1",
        server_port: int = 31000,
        server_ready_timeout_s: float = 180.0,
        request_timeout_s: float = 180.0,
        server_python: str | None = None,
        server_extra_args: list[str] | None = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        load_format: str | None = None,
        quantization: str | None = None,
        tp_size: int = 1,
        mem_fraction_static: float | None = None,
        served_model_name: str | None = None,
        server_log_path: str | Path | None = None,
    ) -> None:
        self.model_id = str(Path(model_id).resolve()) if model_id else default_qwen35_model_id()
        self.max_new_tokens = int(max_new_tokens)
        self.server_host = str(server_host)
        self.server_port = int(server_port)
        self.server_ready_timeout_s = float(server_ready_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.server_python = str(server_python or sys.executable)
        self.server_extra_args = [str(arg) for arg in (server_extra_args or [])]
        self.trust_remote_code = bool(trust_remote_code)
        self.dtype = str(dtype)
        self.load_format = str(load_format) if load_format else None
        self.quantization = str(quantization) if quantization else None
        self.tp_size = int(tp_size)
        self.mem_fraction_static = None if mem_fraction_static is None else float(mem_fraction_static)
        self.served_model_name = str(served_model_name) if served_model_name else None
        self.server_log_path = str(server_log_path) if server_log_path else None
        self._server_process: subprocess.Popen[str] | None = None
        self._spawned_server = False

    @property
    def api_base(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    @property
    def server_pid(self) -> int | None:
        if self._server_process is None:
            return None
        if self._server_process.poll() is not None:
            return None
        return int(self._server_process.pid)

    def ensure_loaded(self) -> None:
        if self._server_ready():
            self.served_model_name = self._resolve_served_model_name()
            return
        self._start_server()
        self._wait_for_server_ready()
        self.served_model_name = self._resolve_served_model_name()

    def shutdown(self) -> None:
        if not self._spawned_server or self._server_process is None:
            return
        if self._server_process.poll() is not None:
            return
        self._server_process.terminate()
        try:
            self._server_process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._server_process.kill()
            self._server_process.wait(timeout=5.0)

    def generate_code(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
    ) -> GeneratedCode:
        del use_blank_image
        self.ensure_loaded()
        user_content: list[dict[str, Any]] = []
        data_url = self._build_data_url(image_path=image_path, image_bytes=image_bytes)
        if data_url:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_content.append({"type": "text", "text": prompt_bundle.user_prompt})
        payload = {
            "model": self.served_model_name or self.model_id,
            "messages": [
                {"role": "system", "content": prompt_bundle.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": int(max_new_tokens or self.max_new_tokens),
            "stream": False,
        }
        response = self._request_json(
            method="POST",
            path="/v1/chat/completions",
            payload=payload,
            timeout_s=self.request_timeout_s,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            raise RuntimeError("sglang response missing choices")
        message = dict(choices[0].get("message") or {})
        raw_text = _extract_message_text(message.get("content"))
        code = extract_python_code(raw_text)
        return GeneratedCode(
            code=code,
            raw_text=raw_text,
            rendered_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            model_id=self.model_id,
            prompt_bundle=prompt_bundle.to_dict(),
            screenshot_path=str(Path(image_path).resolve()) if image_path else None,
        )

    def _build_server_command(self) -> list[str]:
        command = [
            self.server_python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_id,
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
            "--dtype",
            self.dtype,
            "--tp-size",
            str(self.tp_size),
        ]
        if self.load_format:
            command.extend(["--load-format", self.load_format])
        if self.quantization:
            command.extend(["--quantization", self.quantization])
        if self.trust_remote_code:
            command.append("--trust-remote-code")
        if self.mem_fraction_static is not None:
            command.extend(["--mem-fraction-static", str(self.mem_fraction_static)])
        command.extend(self.server_extra_args)
        return command

    def _start_server(self) -> None:
        log_handle = None
        if self.server_log_path:
            log_path = Path(self.server_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        self._server_process = subprocess.Popen(
            self._build_server_command(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle or subprocess.DEVNULL,
            stderr=log_handle or subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            text=True,
        )
        self._spawned_server = True

    def _wait_for_server_ready(self) -> None:
        started = time.monotonic()
        while time.monotonic() - started < self.server_ready_timeout_s:
            if self._server_ready():
                return
            if self._server_process is not None and self._server_process.poll() is not None:
                raise RuntimeError(
                    "sglang server exited before becoming ready"
                    + self._format_log_tail()
                )
            time.sleep(0.5)
        raise RuntimeError(
            f"sglang server did not become ready within {self.server_ready_timeout_s}s"
            + self._format_log_tail()
        )

    def _server_ready(self) -> bool:
        try:
            response = self._request_json(method="GET", path="/v1/models", payload=None, timeout_s=2.0)
        except Exception:
            return False
        return bool(response.get("data"))

    def _resolve_served_model_name(self) -> str:
        if self.served_model_name:
            return self.served_model_name
        response = self._request_json(method="GET", path="/v1/models", payload=None, timeout_s=5.0)
        models = list(response.get("data") or [])
        if not models:
            return self.model_id
        first = dict(models[0])
        return str(first.get("id") or self.model_id)

    def _build_data_url(self, *, image_path: str | Path | None, image_bytes: bytes | None) -> str | None:
        if image_bytes:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        if image_path:
            image_bytes = Path(image_path).read_bytes()
            encoded = base64.b64encode(image_bytes).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        return None

    def _request_json(self, *, method: str, path: str, payload: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.api_base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"sglang request failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"sglang request failed: {exc}") from exc
        return json.loads(body)

    def _format_log_tail(self) -> str:
        if not self.server_log_path:
            return ""
        log_path = Path(self.server_log_path)
        if not log_path.exists():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = text[-4000:].strip()
        if not tail:
            return ""
        return f"\nSGLang log tail:\n{tail}"
