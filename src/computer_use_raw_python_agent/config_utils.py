from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def default_config_path(cwd: str | Path | None = None) -> Path:
    base = Path(cwd or Path.cwd())
    return (base / "config" / "agent.default.json").resolve()


def load_agent_config(path: str | None) -> tuple[dict[str, Any], Path | None]:
    config_path = Path(path).resolve() if path else default_config_path()
    if not config_path.exists():
        return {}, None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    normalized: dict[str, Any] = {}
    if "endpoint" in data:
        normalized["endpoint"] = str(data["endpoint"])
    if "mcp_command" in data:
        normalized["mcp_command"] = [str(part) for part in data["mcp_command"]]
    if "mcp_cwd" in data:
        cwd_value = Path(str(data["mcp_cwd"]))
        normalized["mcp_cwd"] = str((base_dir / cwd_value).resolve()) if not cwd_value.is_absolute() else str(cwd_value)
    policy_value = data.get("policy", data.get("policy_path"))
    if policy_value is not None:
        policy_path = Path(str(policy_value))
        normalized["policy"] = str((base_dir / policy_path).resolve()) if not policy_path.is_absolute() else str(policy_path)
    run_dir_value = data.get("run_dir", data.get("run_root"))
    if run_dir_value is not None:
        run_dir = Path(str(run_dir_value))
        normalized["run_dir"] = str((base_dir / run_dir).resolve()) if not run_dir.is_absolute() else str(run_dir)
    if "max_iterations" in data:
        normalized["max_iterations"] = int(data["max_iterations"])
    if "max_new_tokens" in data:
        normalized["max_new_tokens"] = int(data["max_new_tokens"])
    if "compute_dtype" in data:
        normalized["compute_dtype"] = str(data["compute_dtype"])
    if "device_map" in data:
        normalized["device_map"] = str(data["device_map"])
    if "load_in_4bit" in data:
        normalized["load_in_4bit"] = bool(data["load_in_4bit"])
    if "load_in_8bit" in data:
        normalized["load_in_8bit"] = bool(data["load_in_8bit"])
    if "enable_fp32_cpu_offload" in data:
        normalized["enable_fp32_cpu_offload"] = bool(data["enable_fp32_cpu_offload"])
    if "strong_visual_grounding" in data:
        normalized["strong_visual_grounding"] = bool(data["strong_visual_grounding"])
    if "reasoning_enabled" in data:
        normalized["reasoning_enabled"] = bool(data["reasoning_enabled"])
    if "replan_enabled" in data:
        normalized["replan_enabled"] = bool(data["replan_enabled"])
    if "replan_max_attempts" in data:
        normalized["replan_max_attempts"] = int(data["replan_max_attempts"])
    if "dependency_repair_enabled" in data:
        normalized["dependency_repair_enabled"] = bool(data["dependency_repair_enabled"])
    if "dependency_repair_max_attempts" in data:
        normalized["dependency_repair_max_attempts"] = int(data["dependency_repair_max_attempts"])
    if "dependency_repair_allow_shell_fallback" in data:
        normalized["dependency_repair_allow_shell_fallback"] = bool(data["dependency_repair_allow_shell_fallback"])
    if "load_request_timeout_s" in data:
        normalized["load_request_timeout_s"] = float(data["load_request_timeout_s"])
    if "run_request_timeout_s" in data:
        normalized["run_request_timeout_s"] = float(data["run_request_timeout_s"])
    return normalized, config_path


def load_policy_from_path(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))
