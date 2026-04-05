from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from .qwen_runtime import default_qwen35_model_id
from .sglang_config import default_sglang_config_path, load_sglang_agent_config
from .sglang_daemon import (
    _send_request,
    daemon_is_responding,
    daemon_process_alive,
    daemon_state_path,
    start_daemon_process,
    wait_for_daemon_ready,
)


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
        "reasoning_enabled",
        "strong_visual_grounding",
        "replan_enabled",
        "replan_max_attempts",
        "dependency_repair_enabled",
        "dependency_repair_max_attempts",
        "dependency_repair_allow_shell_fallback",
        "load_request_timeout_s",
        "run_request_timeout_s",
        "sglang_server_host",
        "sglang_server_port",
        "sglang_server_ready_timeout_s",
        "sglang_request_timeout_s",
        "sglang_server_python",
        "sglang_dtype",
        "sglang_tp_size",
        "sglang_mem_fraction_static",
        "sglang_served_model_name",
        "sglang_trust_remote_code",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.mcp_command:
        overrides["mcp_command"] = list(args.mcp_command)
    if args.sglang_server_extra_arg:
        overrides["sglang_server_extra_args"] = list(args.sglang_server_extra_arg)
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


def _kill_pid(pid: int | None, *, grace_s: float = 5.0) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    return True


def cmd_stop(_: argparse.Namespace) -> int:
    if daemon_is_responding():
        response = _send_request({"action": "shutdown"})
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    state = _state_or_empty()
    pid = state.get("pid")
    server_pid = state.get("sglang_server_pid")
    if not (isinstance(pid, int) and daemon_process_alive()):
        if _kill_pid(server_pid):
            daemon_state_path().unlink(missing_ok=True)
            print(json.dumps({"ok": True, "forced": True, "sglang_server_pid": server_pid}, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps({"running": False}, ensure_ascii=False, indent=2))
        return 0
    _kill_pid(pid)
    _kill_pid(server_pid)
    daemon_state_path().unlink(missing_ok=True)
    print(json.dumps({"ok": True, "forced": True, "pid": pid, "sglang_server_pid": server_pid}, ensure_ascii=False, indent=2))
    return 0


def cmd_main(args: argparse.Namespace) -> int:
    config_defaults, config_path = load_sglang_agent_config(args.config)
    default_overrides = _build_default_overrides(args, config_defaults)

    if args.model_id:
        _ensure_daemon_started()
        response = _send_request(
            {
                "action": "reload",
                "model_id": args.model_id,
                "processor_id": args.processor_id,
                "defaults": default_overrides,
            },
            timeout_s=_resolve_load_timeout(args, default_overrides),
        )
        if not response.get("ok", False):
            raise SystemExit(response.get("error", "sglang agent reload failed"))
        if not args.prompt:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "daemon": response.get("status", {}),
                        "config_path": str(config_path) if config_path else str(default_sglang_config_path()),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

    if args.prompt:
        if not daemon_is_responding():
            raise SystemExit("sglang agent daemon is not running; start it once with --model-id")
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
            raise SystemExit(response.get("error", "sglang agent run failed"))
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    if args.model_id:
        return 0

    print(json.dumps({"running": daemon_is_responding(), "state": _state_or_empty()}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-sglang")
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
    parser.add_argument("--reasoning-enabled", action="store_true", default=None)
    parser.add_argument("--strong-visual-grounding", action="store_true", default=None)
    parser.add_argument("--replan-enabled", action="store_true", default=None)
    parser.add_argument("--replan-max-attempts", type=int)
    parser.add_argument("--dependency-repair-enabled", action="store_true", default=None)
    parser.add_argument("--dependency-repair-max-attempts", type=int)
    parser.add_argument("--dependency-repair-allow-shell-fallback", action="store_true", default=None)
    parser.add_argument("--sglang-server-host")
    parser.add_argument("--sglang-server-port", type=int)
    parser.add_argument("--sglang-server-ready-timeout-s", type=float)
    parser.add_argument("--sglang-request-timeout-s", type=float)
    parser.add_argument("--sglang-server-python")
    parser.add_argument("--sglang-server-extra-arg", action="append")
    parser.add_argument("--sglang-trust-remote-code", action="store_true", default=None)
    parser.add_argument("--sglang-dtype")
    parser.add_argument("--sglang-tp-size", type=int)
    parser.add_argument("--sglang-mem-fraction-static", type=float)
    parser.add_argument("--sglang-served-model-name")
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
