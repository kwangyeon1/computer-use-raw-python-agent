from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from .config_utils import load_agent_config
from .qwen_daemon import (
    _send_request,
    daemon_is_responding,
    daemon_process_alive,
    daemon_state_path,
    start_daemon_process,
    wait_for_daemon_ready,
)
from .qwen_runtime import default_qwen35_model_id


def _default_qwen_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "config" / "agent.qwen35.default.json").resolve()


def _load_qwen_agent_config(path: str | None) -> tuple[dict, Path | None]:
    config_path = str(Path(path).resolve()) if path else str(_default_qwen_config_path())
    return load_agent_config(config_path)


def _state_or_empty() -> dict:
    path = daemon_state_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_explicit_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    for key in (
        "endpoint",
        "mcp_cwd",
        "run_dir",
        "policy",
        "max_iterations",
        "max_new_tokens",
        "strong_visual_grounding",
        "replan_enabled",
        "replan_max_attempts",
        "dependency_repair_enabled",
        "dependency_repair_max_attempts",
        "dependency_repair_allow_shell_fallback",
        "load_request_timeout_s",
        "run_request_timeout_s",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.mcp_command:
        overrides["mcp_command"] = list(args.mcp_command)
    return overrides


def _build_default_overrides(args: argparse.Namespace, config_defaults: dict) -> dict:
    return {**config_defaults, **_collect_explicit_overrides(args)}


def _build_run_overrides(args: argparse.Namespace, config_defaults: dict) -> dict:
    explicit_overrides = _collect_explicit_overrides(args)
    if args.model_id or args.config:
        return {**config_defaults, **explicit_overrides}
    return explicit_overrides


def _ensure_daemon_started() -> None:
    if daemon_is_responding():
        return
    if daemon_process_alive():
        return
    start_daemon_process()
    wait_for_daemon_ready()


def _resolve_load_timeout(args: argparse.Namespace, overrides: dict) -> float:
    if args.load_request_timeout_s is not None:
        return float(args.load_request_timeout_s)
    if "load_request_timeout_s" in overrides:
        return float(overrides["load_request_timeout_s"])
    return 1800.0


def _resolve_run_timeout(args: argparse.Namespace, overrides: dict) -> float:
    if args.run_request_timeout_s is not None:
        return float(args.run_request_timeout_s)
    if "run_request_timeout_s" in overrides:
        return float(overrides["run_request_timeout_s"])
    return 900.0


def _resolve_compute_dtype(args: argparse.Namespace, config_defaults: dict) -> str:
    if args.compute_dtype is not None:
        return str(args.compute_dtype)
    if "compute_dtype" in config_defaults:
        return str(config_defaults["compute_dtype"])
    return "bfloat16"


def _resolve_device_map(args: argparse.Namespace, config_defaults: dict) -> str:
    if args.device_map is not None:
        return str(args.device_map)
    if "device_map" in config_defaults:
        return str(config_defaults["device_map"])
    return "auto"


def _resolve_disable_4bit(args: argparse.Namespace, config_defaults: dict) -> bool:
    if args.disable_4bit is not None:
        return bool(args.disable_4bit)
    if "load_in_4bit" in config_defaults:
        return not bool(config_defaults["load_in_4bit"])
    return False


def _resolve_load_in_8bit(args: argparse.Namespace, config_defaults: dict) -> bool:
    if args.load_in_8bit is not None:
        return bool(args.load_in_8bit)
    if "load_in_8bit" in config_defaults:
        return bool(config_defaults["load_in_8bit"])
    return False


def _resolve_disable_cpu_offload(args: argparse.Namespace, config_defaults: dict) -> bool:
    if args.disable_cpu_offload is not None:
        return bool(args.disable_cpu_offload)
    if "enable_fp32_cpu_offload" in config_defaults:
        return not bool(config_defaults["enable_fp32_cpu_offload"])
    return False


def cmd_status(_: argparse.Namespace) -> int:
    if daemon_is_responding():
        response = _send_request({"action": "status"})
        print(json.dumps({"running": True, "responsive": True, **response}, ensure_ascii=False, indent=2))
        return 0
    state = _state_or_empty()
    pid = state.get("pid")
    running = isinstance(pid, int) and daemon_process_alive()
    payload = {"running": running, "responsive": False}
    if state:
        payload["state"] = state
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_stop(_: argparse.Namespace) -> int:
    if daemon_is_responding():
        response = _send_request({"action": "shutdown"})
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    state = _state_or_empty()
    pid = state.get("pid")
    if not (isinstance(pid, int) and daemon_process_alive()):
        print(json.dumps({"running": False}, ensure_ascii=False, indent=2))
        return 0
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not daemon_process_alive():
            daemon_state_path().unlink(missing_ok=True)
            print(json.dumps({"ok": True, "forced": True, "pid": pid}, ensure_ascii=False, indent=2))
            return 0
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    daemon_state_path().unlink(missing_ok=True)
    print(json.dumps({"ok": True, "forced": True, "signal": "SIGKILL", "pid": pid}, ensure_ascii=False, indent=2))
    return 0


def cmd_main(args: argparse.Namespace) -> int:
    config_defaults, config_path = _load_qwen_agent_config(args.config)
    default_overrides = _build_default_overrides(args, config_defaults)

    if args.model_id:
        _ensure_daemon_started()
        load_in_8bit = _resolve_load_in_8bit(args, config_defaults)
        disable_4bit = _resolve_disable_4bit(args, config_defaults)
        if load_in_8bit and not disable_4bit:
            raise SystemExit("8bit mode requires 4bit to be disabled; set load_in_4bit=false or pass --disable-4bit")
        response = _send_request(
            {
                "action": "reload",
                "model_id": args.model_id,
                "processor_id": args.processor_id,
                "compute_dtype": _resolve_compute_dtype(args, config_defaults),
                "device_map": _resolve_device_map(args, config_defaults),
                "disable_4bit": disable_4bit,
                "load_in_8bit": load_in_8bit,
                "disable_cpu_offload": _resolve_disable_cpu_offload(args, config_defaults),
                "defaults": default_overrides,
            },
            timeout_s=_resolve_load_timeout(args, default_overrides),
        )
        if not response.get("ok", False):
            raise SystemExit(response.get("error", "qwen agent reload failed"))
        if not args.prompt:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "daemon": response.get("status", {}),
                        "config_path": str(config_path) if config_path else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

    if args.prompt:
        if not daemon_is_responding():
            raise SystemExit("qwen agent daemon is not running; start it once with --model-id")
        run_overrides = _build_run_overrides(args, config_defaults)
        response = _send_request(
            {
                "action": "run",
                "prompt": args.prompt,
                "overrides": run_overrides,
            },
            timeout_s=_resolve_run_timeout(args, run_overrides),
        )
        if not response.get("ok", False):
            raise SystemExit(response.get("error", "qwen agent run failed"))
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    if args.model_id:
        return 0

    print(json.dumps({"running": daemon_is_responding(), "state": _state_or_empty()}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m computer_use_raw_python_agent.qwen_cli")
    parser.add_argument("--model-id")
    parser.add_argument("--processor-id")
    parser.add_argument("--prompt")
    parser.add_argument("--config")
    parser.add_argument("--policy")
    parser.add_argument("--run-dir")
    parser.add_argument("--endpoint")
    parser.add_argument("--mcp-command", nargs="+")
    parser.add_argument("--mcp-cwd")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--strong-visual-grounding", action="store_true", default=None)
    parser.add_argument("--replan-enabled", action="store_true", default=None)
    parser.add_argument("--replan-max-attempts", type=int)
    parser.add_argument("--dependency-repair-enabled", action="store_true", default=None)
    parser.add_argument("--dependency-repair-max-attempts", type=int)
    parser.add_argument("--dependency-repair-allow-shell-fallback", action="store_true", default=None)
    parser.add_argument("--compute-dtype")
    parser.add_argument("--device-map")
    parser.add_argument("--disable-4bit", action="store_true", default=None)
    parser.add_argument("--load-in-8bit", action="store_true", default=None)
    parser.add_argument("--disable-cpu-offload", action="store_true", default=None)
    parser.add_argument("--load-request-timeout-s", type=float)
    parser.add_argument("--run-request-timeout-s", type=float)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.status:
        raise SystemExit(cmd_status(args))
    if args.stop:
        raise SystemExit(cmd_stop(args))
    if args.model_id is None and not args.prompt and not args.status and not args.stop:
        args.model_id = default_qwen35_model_id()
    raise SystemExit(cmd_main(args))


if __name__ == "__main__":
    main()
