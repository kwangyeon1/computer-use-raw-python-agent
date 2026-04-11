from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from computer_use_raw_python_agent.config_utils import load_agent_config
from computer_use_raw_python_agent.models import PromptBundle
from computer_use_raw_python_agent.runtime import ExternalCliRawPythonRuntime


def _write_cli_script(path: Path, *, mode: str) -> None:
    if mode == "json_code":
        body = """from __future__ import annotations
import json
import sys

payload = json.loads(sys.stdin.read())
assert payload["response_format"] == "python_code"
assert payload["prompt_bundle"]["session_prompt"] == "open chrome"
assert payload["generation_context"]["step_id"] == "step-001"
print(json.dumps({"python_code": "print('ok from cli')", "model_id": "wrapped-cli"}))
"""
    elif mode == "plain_text":
        body = """from __future__ import annotations
import sys

sys.stdin.read()
print('{"use_web_search": false, "query": "", "allowed_domains": [], "blocked_domains": [], "reason": ""}')
"""
    else:  # pragma: no cover - test definition guard
        raise ValueError(mode)
    path.write_text(body, encoding="utf-8")


def _make_prompt_bundle() -> PromptBundle:
    return PromptBundle(
        system_prompt="system prompt",
        user_prompt="user prompt",
        session_prompt="open chrome",
        policy={"policy_name": "test"},
    )


def test_external_cli_runtime_generate_code_supports_json_response(tmp_path: Path) -> None:
    script_path = tmp_path / "cli_json_code.py"
    _write_cli_script(script_path, mode="json_code")
    runtime = ExternalCliRawPythonRuntime(command=[sys.executable, str(script_path)], max_new_tokens=77)

    generated = runtime.generate_code(
        prompt_bundle=_make_prompt_bundle(),
        max_new_tokens=12,
        generation_context={"step_id": "step-001", "run_dir": str(tmp_path)},
    )

    assert generated.code == "print('ok from cli')"
    assert generated.raw_text == "print('ok from cli')"
    assert generated.model_id == "wrapped-cli"


def test_external_cli_runtime_generate_text_supports_plain_stdout(tmp_path: Path) -> None:
    script_path = tmp_path / "cli_plain_text.py"
    _write_cli_script(script_path, mode="plain_text")
    runtime = ExternalCliRawPythonRuntime(command=[sys.executable, str(script_path)], max_new_tokens=77)

    generated = runtime.generate_text(prompt_bundle=_make_prompt_bundle(), max_new_tokens=12)

    assert json.loads(generated.text)["use_web_search"] is False
    assert generated.model_id.startswith("external-cli:")


def test_load_agent_config_resolves_agent_cli_paths(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cli_path = tmp_path / "bin" / "fake-cli"
    cli_path.parent.mkdir()
    cli_path.write_text("#!/bin/sh\n", encoding="utf-8")
    config_path = tmp_path / "agent.json"
    config_path.write_text(
        json.dumps(
            {
                "agent_cli_command": ["./bin/fake-cli", "--flag"],
                "agent_cli_cwd": "./work",
            }
        ),
        encoding="utf-8",
    )

    config, resolved_path = load_agent_config(str(config_path))

    assert resolved_path == config_path.resolve()
    assert config["agent_cli_command"] == [str(cli_path.resolve()), "--flag"]
    assert config["agent_cli_cwd"] == str(work_dir.resolve())


def test_load_agent_config_uses_current_python_for_internal_module_wrapper(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.json"
    config_path.write_text(
        json.dumps(
            {
                "agent_cli_command": ["python", "-m", "computer_use_raw_python_agent.codex_backend", "--model", "gpt-5.4-mini"]
            }
        ),
        encoding="utf-8",
    )

    config, _ = load_agent_config(str(config_path))

    assert config["agent_cli_command"][0] == sys.executable
    assert config["agent_cli_command"][1:] == ["-m", "computer_use_raw_python_agent.codex_backend", "--model", "gpt-5.4-mini"]


def test_load_agent_config_preserves_symlink_command_path(tmp_path: Path) -> None:
    real_python = tmp_path / "python-real"
    real_python.write_text("#!/bin/sh\n", encoding="utf-8")
    symlink_python = tmp_path / "python-link"
    symlink_python.symlink_to(real_python)
    config_path = tmp_path / "agent.json"
    config_path.write_text(
        json.dumps(
            {
                "agent_cli_command": ["./python-link", "-m", "computer_use_raw_python_agent.codex_backend"],
            }
        ),
        encoding="utf-8",
    )

    config, _ = load_agent_config(str(config_path))

    assert config["agent_cli_command"][0] == str(symlink_python.absolute())


@pytest.mark.parametrize(
    ("module_name", "extra_kwargs"),
    [
        ("computer_use_raw_python_agent.daemon", {}),
        ("computer_use_raw_python_agent.qwen_daemon", {"load_in_8bit": False}),
    ],
)
def test_daemon_reload_accepts_external_cli_backend(
    module_name: str,
    extra_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_write_state_file", lambda pid, payload: None)
    script_path = tmp_path / "echo_cli.py"
    _write_cli_script(script_path, mode="plain_text")
    state = module.AgentDaemonState(
        backend_kind="",
        model_id="",
        processor_id=None,
        compute_dtype="bfloat16",
        device_map="auto",
        load_in_4bit=True,
        enable_fp32_cpu_offload=True,
        defaults={},
        **extra_kwargs,
    )

    response = module._handle_reload(
        state,
        {
            "backend_kind": "external_cli",
            "agent_cli_command": [sys.executable, str(script_path)],
            "agent_cli_cwd": str(tmp_path),
            "defaults": {"max_new_tokens": 42},
        },
    )

    assert response["ok"] is True
    assert state.backend_kind == "external_cli"
    assert state.agent_cli_command == [sys.executable, str(script_path)]
    assert state.agent_cli_cwd == str(tmp_path)
    assert state.has_backend() is True
