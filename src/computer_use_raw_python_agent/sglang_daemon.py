from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys
import time
import traceback
import uuid

from .config_utils import load_policy_from_path
from .service import build_executor_client, run_agent_control_loop
from .sglang_runtime import Qwen35SGLangRuntime


_STATE_DIR = Path("/tmp/computer_use_raw_python_agent_qwen35_sglang")
_STATE_PATH = _STATE_DIR / "agent.state.json"
_LOG_PATH = _STATE_DIR / "agent.log"
_SERVER_LOG_PATH = _STATE_DIR / "sglang.server.log"
_REQUESTS_DIR = _STATE_DIR / "requests"
_RESPONSES_DIR = _STATE_DIR / "responses"


def daemon_state_path() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_PATH


def daemon_log_path() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_PATH


def daemon_server_log_path() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _SERVER_LOG_PATH


def daemon_requests_dir() -> Path:
    _REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    return _REQUESTS_DIR


def daemon_responses_dir() -> Path:
    _RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    return _RESPONSES_DIR


@dataclass
class AgentDaemonState:
    model_id: str
    processor_id: str | None
    defaults: dict[str, Any]
    phase: str = "idle"
    runtime: Qwen35SGLangRuntime | None = None

    def ensure_runtime(self) -> Qwen35SGLangRuntime:
        if self.runtime is None:
            self.runtime = Qwen35SGLangRuntime(
                model_id=self.model_id,
                max_new_tokens=int(self.defaults.get("max_new_tokens", 256)),
                server_host=str(self.defaults.get("sglang_server_host", "127.0.0.1")),
                server_port=int(self.defaults.get("sglang_server_port", 31000)),
                server_ready_timeout_s=float(self.defaults.get("sglang_server_ready_timeout_s", 180.0)),
                request_timeout_s=float(self.defaults.get("sglang_request_timeout_s", 180.0)),
                server_python=self.defaults.get("sglang_server_python"),
                server_extra_args=list(self.defaults.get("sglang_server_extra_args", [])),
                trust_remote_code=bool(self.defaults.get("sglang_trust_remote_code", True)),
                dtype=str(self.defaults.get("sglang_dtype", "auto")),
                load_format=self.defaults.get("sglang_load_format"),
                quantization=self.defaults.get("sglang_quantization"),
                tp_size=int(self.defaults.get("sglang_tp_size", 1)),
                mem_fraction_static=self.defaults.get("sglang_mem_fraction_static"),
                served_model_name=self.defaults.get("sglang_served_model_name"),
                server_log_path=str(daemon_server_log_path()),
            )
            self.runtime.ensure_loaded()
        return self.runtime

    def to_public_dict(self) -> dict[str, Any]:
        runtime = self.runtime
        return {
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "defaults": self.defaults,
            "phase": self.phase,
            "sglang_api_base": runtime.api_base if runtime else None,
            "sglang_served_model_name": runtime.served_model_name if runtime else self.defaults.get("sglang_served_model_name"),
            "sglang_server_pid": runtime.server_pid if runtime else None,
        }


def _write_state_file(pid: int, payload: dict[str, Any]) -> None:
    daemon_state_path().write_text(json.dumps({"pid": pid, **payload}, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_state_file() -> dict[str, Any] | None:
    path = daemon_state_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _send_request(payload: dict[str, Any], *, timeout_s: float = 5.0) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    requests_dir = daemon_requests_dir()
    responses_dir = daemon_responses_dir()
    request_path = requests_dir / f"{request_id}.json"
    response_path = responses_dir / f"{request_id}.json"
    temp_path = requests_dir / f"{request_id}.tmp"
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(request_path)

    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if response_path.exists():
            try:
                return json.loads(response_path.read_text(encoding="utf-8"))
            finally:
                response_path.unlink(missing_ok=True)
                request_path.unlink(missing_ok=True)
        time.sleep(0.05)
    request_path.unlink(missing_ok=True)
    raise RuntimeError(f"sglang agent daemon did not respond within {timeout_s}s")


def daemon_is_responding() -> bool:
    try:
        response = _send_request({"action": "status"}, timeout_s=0.5)
    except Exception:
        return False
    return bool(response.get("ok"))


def daemon_process_alive() -> bool:
    state = _read_state_file()
    if not state:
        return False
    pid = state.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_daemon_process() -> None:
    requests_dir = daemon_requests_dir()
    responses_dir = daemon_responses_dir()
    for path in requests_dir.glob("*.json"):
        path.unlink()
    for path in responses_dir.glob("*.json"):
        path.unlink()
    log_handle = daemon_log_path().open("a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-m", "computer_use_raw_python_agent.sglang_daemon", "serve"],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        close_fds=True,
    )


def wait_for_daemon_ready(timeout_s: float = 10.0) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if daemon_is_responding():
            return
        time.sleep(0.1)
    raise RuntimeError(f"sglang agent daemon did not become ready within {timeout_s}s")


def _merge_defaults(current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    return merged


def _make_run_dir(base_run_dir: str | None, prompt: str) -> str:
    if not base_run_dir:
        raise RuntimeError("run_dir is not configured; provide it in config JSON or with --run-dir")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in prompt.strip())[:48].strip("-") or "run"
    ts = time.strftime("%Y%m%d-%H%M%S")
    return str(Path(base_run_dir) / f"{ts}-{slug}")


def _handle_run(daemon_state: AgentDaemonState, payload: dict[str, Any]) -> dict[str, Any]:
    if not daemon_state.model_id:
        return {"ok": False, "error": "no model is loaded; start once with --model-id"}
    prompt = str(payload["prompt"]).strip()
    overrides = dict(payload.get("overrides", {}))
    defaults = _merge_defaults(daemon_state.defaults, overrides)
    base_run_dir = str(overrides.get("run_dir") or defaults.get("run_dir") or "")
    run_dir = _make_run_dir(base_run_dir, prompt)
    policy_path = str(defaults.get("policy") or "")
    policy = load_policy_from_path(policy_path)
    executor_client = build_executor_client(
        endpoint=defaults.get("endpoint"),
        mcp_command=defaults.get("mcp_command"),
        mcp_cwd=defaults.get("mcp_cwd"),
    )
    try:
        daemon_state.phase = "running"
        _write_state_file(os.getpid(), daemon_state.to_public_dict())
        summary = run_agent_control_loop(
            runtime=daemon_state.ensure_runtime(),
            executor_client=executor_client,
            user_prompt=prompt,
            policy=policy,
            run_dir=run_dir,
            max_iterations=int(defaults.get("max_iterations", 5)),
            max_new_tokens=int(defaults.get("max_new_tokens", 256)),
            reasoning_enabled=bool(defaults.get("reasoning_enabled", False)),
            strong_visual_grounding=bool(defaults.get("strong_visual_grounding", False)),
            replan_enabled=bool(defaults.get("replan_enabled", False)),
            replan_max_attempts=int(defaults.get("replan_max_attempts", 1)),
            dependency_repair_enabled=bool(defaults.get("dependency_repair_enabled", False)),
            dependency_repair_max_attempts=int(defaults.get("dependency_repair_max_attempts", 2)),
            dependency_repair_allow_shell_fallback=bool(defaults.get("dependency_repair_allow_shell_fallback", False)),
        )
    finally:
        daemon_state.phase = "ready"
        _write_state_file(os.getpid(), daemon_state.to_public_dict())
        executor_client.close()
    return {"ok": True, "summary": summary}


def _handle_reload(daemon_state: AgentDaemonState, payload: dict[str, Any]) -> dict[str, Any]:
    model_id = str(payload.get("model_id") or daemon_state.model_id)
    processor_id = payload.get("processor_id")
    defaults = _merge_defaults(daemon_state.defaults, dict(payload.get("defaults", {})))
    if daemon_state.runtime is not None:
        daemon_state.runtime.shutdown()
        daemon_state.runtime = None
    daemon_state.model_id = model_id
    daemon_state.processor_id = str(processor_id) if processor_id else None
    daemon_state.defaults = defaults
    daemon_state.phase = "loading"
    _write_state_file(os.getpid(), daemon_state.to_public_dict())
    daemon_state.runtime = Qwen35SGLangRuntime(
        model_id=daemon_state.model_id,
        max_new_tokens=int(defaults.get("max_new_tokens", 256)),
        server_host=str(defaults.get("sglang_server_host", "127.0.0.1")),
        server_port=int(defaults.get("sglang_server_port", 31000)),
        server_ready_timeout_s=float(defaults.get("sglang_server_ready_timeout_s", 180.0)),
        request_timeout_s=float(defaults.get("sglang_request_timeout_s", 180.0)),
        server_python=defaults.get("sglang_server_python"),
        server_extra_args=list(defaults.get("sglang_server_extra_args", [])),
        trust_remote_code=bool(defaults.get("sglang_trust_remote_code", True)),
        dtype=str(defaults.get("sglang_dtype", "auto")),
        load_format=defaults.get("sglang_load_format"),
        quantization=defaults.get("sglang_quantization"),
        tp_size=int(defaults.get("sglang_tp_size", 1)),
        mem_fraction_static=defaults.get("sglang_mem_fraction_static"),
        served_model_name=defaults.get("sglang_served_model_name"),
        server_log_path=str(daemon_server_log_path()),
    )
    daemon_state.runtime.ensure_loaded()
    daemon_state.phase = "ready"
    _write_state_file(os.getpid(), daemon_state.to_public_dict())
    return {"ok": True, "status": daemon_state.to_public_dict()}


def _serve() -> int:
    requests_dir = daemon_requests_dir()
    responses_dir = daemon_responses_dir()
    daemon_state = AgentDaemonState(
        model_id="",
        processor_id=None,
        defaults={},
        phase="idle",
    )
    _write_state_file(os.getpid(), daemon_state.to_public_dict())
    try:
        while True:
            handled_any = False
            for request_path in sorted(requests_dir.glob("*.json")):
                handled_any = True
                response_path = responses_dir / request_path.name
                try:
                    payload = json.loads(request_path.read_text(encoding="utf-8"))
                    action = payload.get("action")
                    if action == "status":
                        response = {"ok": True, "status": daemon_state.to_public_dict()}
                    elif action == "reload":
                        response = _handle_reload(daemon_state, payload)
                    elif action == "run":
                        response = _handle_run(daemon_state, payload)
                    elif action == "shutdown":
                        if daemon_state.runtime is not None:
                            daemon_state.runtime.shutdown()
                        response = {"ok": True}
                        response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
                        request_path.unlink(missing_ok=True)
                        return 0
                    else:
                        response = {"ok": False, "error": f"unknown action: {action!r}"}
                except Exception as exc:  # pragma: no cover - daemon safety path
                    daemon_state.phase = "ready"
                    _write_state_file(os.getpid(), daemon_state.to_public_dict())
                    response = {
                        "ok": False,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    }
                response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
                request_path.unlink(missing_ok=True)
            if not handled_any:
                time.sleep(0.05)
    finally:
        if daemon_state.runtime is not None:
            daemon_state.runtime.shutdown()
        daemon_state_path().unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        raise SystemExit("usage: python -m computer_use_raw_python_agent.sglang_daemon serve")
    raise SystemExit(_serve())


if __name__ == "__main__":
    main()
