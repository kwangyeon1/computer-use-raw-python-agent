from __future__ import annotations

from typing import Any
import json
import subprocess
import urllib.error
import urllib.request


class ExecutorStdioClient:
    def __init__(self, command: list[str], cwd: str | None = None) -> None:
        self.command = command
        self.cwd = cwd
        self._proc: subprocess.Popen[str] | None = None

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        return self._proc

    def _rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = self._ensure_proc()
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            stderr_text = ""
            if proc.stderr is not None:
                stderr_text = proc.stderr.read()
            raise RuntimeError(f"stdio executor returned no response. stderr={stderr_text!r}")
        return json.loads(line)

    def observe(self) -> dict[str, Any]:
        return self._rpc({"action": "observe"})

    def execute(
        self,
        *,
        python_code: str,
        run_dir: str,
        step_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._rpc(
            {
                "action": "execute",
                "python_code": python_code,
                "run_dir": run_dir,
                "step_id": step_id,
                "metadata": metadata or {},
            }
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.write("__quit__\n")
                self._proc.stdin.flush()
        except BrokenPipeError:
            pass
        self._proc.terminate()
        self._proc = None


class ExecutorHttpClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    def _rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.endpoint}/rpc",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - networked path
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"http executor error status={exc.code} body={body!r}") from exc
        return json.loads(body)

    def observe(self) -> dict[str, Any]:
        return self._rpc({"action": "observe"})

    def execute(
        self,
        *,
        python_code: str,
        run_dir: str,
        step_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._rpc(
            {
                "action": "execute",
                "python_code": python_code,
                "run_dir": run_dir,
                "step_id": step_id,
                "metadata": metadata or {},
            }
        )

    def close(self) -> None:
        return None
