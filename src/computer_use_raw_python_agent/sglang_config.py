from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .config_utils import load_agent_config


def default_sglang_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "config" / "agent.qwen35.sglang.default.json").resolve()


def load_sglang_agent_config(path: str | None) -> tuple[dict[str, Any], Path | None]:
    config_path = Path(path).resolve() if path else default_sglang_config_path()
    base_config, _ = load_agent_config(str(config_path))
    if not config_path.exists():
        return base_config, None

    data = json.loads(config_path.read_text(encoding="utf-8"))
    normalized = dict(base_config)

    if "sglang_server_host" in data:
        normalized["sglang_server_host"] = str(data["sglang_server_host"])
    if "sglang_server_port" in data:
        normalized["sglang_server_port"] = int(data["sglang_server_port"])
    if "sglang_server_ready_timeout_s" in data:
        normalized["sglang_server_ready_timeout_s"] = float(data["sglang_server_ready_timeout_s"])
    if "sglang_request_timeout_s" in data:
        normalized["sglang_request_timeout_s"] = float(data["sglang_request_timeout_s"])
    if "sglang_server_python" in data:
        normalized["sglang_server_python"] = str(data["sglang_server_python"])
    if "sglang_server_extra_args" in data:
        normalized["sglang_server_extra_args"] = [str(part) for part in data["sglang_server_extra_args"]]
    if "sglang_trust_remote_code" in data:
        normalized["sglang_trust_remote_code"] = bool(data["sglang_trust_remote_code"])
    if "sglang_dtype" in data:
        normalized["sglang_dtype"] = str(data["sglang_dtype"])
    if "sglang_load_format" in data:
        normalized["sglang_load_format"] = str(data["sglang_load_format"])
    if "sglang_quantization" in data:
        normalized["sglang_quantization"] = str(data["sglang_quantization"])
    if "sglang_tp_size" in data:
        normalized["sglang_tp_size"] = int(data["sglang_tp_size"])
    if "sglang_mem_fraction_static" in data:
        normalized["sglang_mem_fraction_static"] = float(data["sglang_mem_fraction_static"])
    if "sglang_served_model_name" in data:
        normalized["sglang_served_model_name"] = str(data["sglang_served_model_name"])

    return normalized, config_path
