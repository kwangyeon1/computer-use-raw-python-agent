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
from .qwen_runtime import Qwen35RawPythonRuntime
from .service import build_executor_client, run_agent_control_loop


_STATE_DIR = Path("/tmp/computer_use_raw_python_agent_qwen35")
_STATE_PATH = _STATE_DIR / "agent.state.json"
_LOG_PATH = _STATE_DIR / "agent.log"
_REQUESTS_DIR = _STATE_DIR / "requests"
_RESPONSES_DIR = _STATE_DIR / "responses"


def daemon_state_path() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_PATH


def daemon_log_path() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_PATH


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
    compute_dtype: str
    device_map: str
    load_in_4bit: bool
    load_in_8bit: bool
    enable_fp32_cpu_offload: bool
    defaults: dict[str, Any]
    phase: str = "idle"
    runtime: Qwen35RawPythonRuntime | None = None

    def ensure_runtime(self) -> Qwen35RawPythonRuntime:
        if self.runtime is None:
            self.runtime = Qwen35RawPythonRuntime(
                model_id=self.model_id,
                processor_id=self.processor_id,
                max_new_tokens=int(self.defaults.get("max_new_tokens", 256)),
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                compute_dtype=self.compute_dtype,
                device_map=self.device_map,
                enable_fp32_cpu_offload=self.enable_fp32_cpu_offload,
            )
            self.runtime.ensure_loaded()
        return self.runtime

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "compute_dtype": self.compute_dtype,
            "device_map": self.device_map,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "enable_fp32_cpu_offload": self.enable_fp32_cpu_offload,
            "defaults": self.defaults,
            "phase": self.phase,
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
    raise RuntimeError(f"qwen agent daemon did not respond within {timeout_s}s")


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
        [sys.executable, "-m", "computer_use_raw_python_agent.qwen_daemon", "serve"],
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
    raise RuntimeError(f"qwen agent daemon did not become ready within {timeout_s}s")


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
            strong_visual_grounding=bool(defaults.get("strong_visual_grounding", False)),
            reasoning_enabled=bool(defaults.get("reasoning_enabled", False)),
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
    daemon_state.model_id = model_id
    daemon_state.processor_id = str(processor_id) if processor_id else None
    daemon_state.compute_dtype = str(payload.get("compute_dtype") or daemon_state.compute_dtype)
    daemon_state.device_map = str(payload.get("device_map") or daemon_state.device_map)
    daemon_state.load_in_4bit = bool(payload.get("load_in_4bit", not bool(payload.get("disable_4bit", False))))
    daemon_state.load_in_8bit = bool(payload.get("load_in_8bit", daemon_state.load_in_8bit))
    if daemon_state.load_in_4bit and daemon_state.load_in_8bit:
        raise RuntimeError("load_in_4bit and load_in_8bit cannot both be enabled")
    daemon_state.enable_fp32_cpu_offload = bool(
        payload.get("enable_fp32_cpu_offload", not bool(payload.get("disable_cpu_offload", False)))
    )
    daemon_state.defaults = defaults
    daemon_state.phase = "loading"
    _write_state_file(os.getpid(), daemon_state.to_public_dict())
    daemon_state.runtime = Qwen35RawPythonRuntime(
        model_id=daemon_state.model_id,
        processor_id=daemon_state.processor_id,
        max_new_tokens=int(defaults.get("max_new_tokens", 256)),
        load_in_4bit=daemon_state.load_in_4bit,
        load_in_8bit=daemon_state.load_in_8bit,
        compute_dtype=daemon_state.compute_dtype,
        device_map=daemon_state.device_map,
        enable_fp32_cpu_offload=daemon_state.enable_fp32_cpu_offload,
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
        compute_dtype="bfloat16",
        device_map="auto",
        load_in_4bit=True,
        load_in_8bit=False,
        enable_fp32_cpu_offload=True,
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
        daemon_state_path().unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        raise SystemExit("usage: python -m computer_use_raw_python_agent.qwen_daemon serve")
    raise SystemExit(_serve())


if __name__ == "__main__":
    main()
