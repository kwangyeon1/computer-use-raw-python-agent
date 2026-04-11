from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_fake_codex(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _arg_value(flag: str, args: list[str]) -> str | None:
    if flag not in args:
        return None
    index = args.index(flag)
    if index + 1 >= len(args):
        return None
    return args[index + 1]


args = sys.argv[1:]
assert args and args[0] == "exec"
command_args = args[1:]
resume = len(args) > 1 and args[1] == "resume"
session_id = args[-2] if resume else "session-123"
cwd = _arg_value("-C", args) or os.getcwd()
output_path = _arg_value("-o", args)
assert output_path
if resume:
    resume_index = command_args.index("resume")
    assert "-C" not in command_args[resume_index:], command_args
    assert args[-1] == "-", args
prompt = sys.stdin.read()
log_path = Path(os.environ["FAKE_CODEX_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"args": args, "prompt": prompt}, ensure_ascii=False) + "\\n")

if not resume:
    sessions_root = Path(os.environ["FAKE_CODEX_SESSIONS_ROOT"])
    session_file = sessions_root / "2026" / "04" / "10" / "rollout-2026-04-10T00-00-00-session-123.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-10T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": session_id,
                    "source": "exec",
                    "cwd": cwd,
                },
            },
            ensure_ascii=False,
        )
        + "\\n",
        encoding="utf-8",
    )

Path(output_path).write_text("print('from resume')" if resume else "print('from exec')", encoding="utf-8")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _make_request(
    *,
    run_dir: Path,
    step_id: str,
    request_kind: str = "task_step",
    web_search_context: dict | None,
) -> dict:
    return {
        "action": "generate",
        "response_format": "python_code",
        "prompt_bundle": {
            "system_prompt": "Return executable Python only.",
            "user_prompt": json.dumps(
                {
                    "user_prompt": "Chrome을 열어줘",
                    "request_kind": request_kind,
                    "runtime_policy": {"policy_name": "unrestricted_local"},
                    "observation_text": "browser not open",
                    "last_execution": {
                        "return_code": 1,
                        "stderr_tail": "failed",
                        "payload_metadata": {
                            "agent_response": {
                                "python_code": "print('old code')",
                            }
                        },
                    },
                    "recent_history": ["step-000_return_code=1"],
                    "replan_requested": True,
                    "replan_reasons": ["execution_error"],
                    "web_search_context": web_search_context or {},
                    "output_requirement": "Return executable Python only.",
                },
                ensure_ascii=False,
            ),
            "session_prompt": "Chrome을 열어줘",
            "policy": {"policy_name": "unrestricted_local"},
        },
        "image_path": None,
        "image_base64": None,
        "use_blank_image": True,
        "generation_context": {
            "run_dir": str(run_dir),
            "step_id": step_id,
            "request_kind": request_kind,
            "step_index": 0 if step_id.endswith("000") else 1,
        },
    }


def test_codex_backend_resumes_with_per_run_state(tmp_path: Path) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    _write_fake_codex(fake_codex)
    sessions_root = tmp_path / "sessions"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = tmp_path / "codex.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["FAKE_CODEX_SESSIONS_ROOT"] = str(sessions_root)
    env["FAKE_CODEX_LOG"] = str(log_path)

    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "computer_use_raw_python_agent.codex_backend",
            "--codex-bin",
            str(fake_codex),
            "--sessions-root",
            str(sessions_root),
            "-C",
            str(tmp_path),
        ],
        input=json.dumps(_make_request(run_dir=run_dir, step_id="step-000", web_search_context={"query": "chrome"})),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert first.returncode == 0, first.stderr
    first_payload = json.loads(first.stdout)
    assert first_payload["session_id"] == "session-123"
    assert first_payload["backend_id"] == "codex-exec"

    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "computer_use_raw_python_agent.codex_backend",
            "--codex-bin",
            str(fake_codex),
            "--sessions-root",
            str(sessions_root),
            "-C",
            str(tmp_path),
        ],
        input=json.dumps(_make_request(run_dir=run_dir, step_id="step-001", web_search_context={"query": "chrome"})),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert second_payload["session_id"] == "session-123"
    assert second_payload["backend_id"] == "codex-resume"

    decision = subprocess.run(
        [
            sys.executable,
            "-m",
            "computer_use_raw_python_agent.codex_backend",
            "--codex-bin",
            str(fake_codex),
            "--sessions-root",
            str(sessions_root),
            "-C",
            str(tmp_path),
        ],
        input=json.dumps(
            _make_request(
                run_dir=run_dir,
                step_id="step-001.web-search-decision",
                request_kind="web_search_decision",
                web_search_context=None,
            )
        ),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert decision.returncode == 0, decision.stderr

    third = subprocess.run(
        [
            sys.executable,
            "-m",
            "computer_use_raw_python_agent.codex_backend",
            "--codex-bin",
            str(fake_codex),
            "--sessions-root",
            str(sessions_root),
            "-C",
            str(tmp_path),
        ],
        input=json.dumps(_make_request(run_dir=run_dir, step_id="step-002", web_search_context={"query": "chrome"})),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert third.returncode == 0, third.stderr

    state_path = run_dir / ".codex-agent-session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["codex_session_id"] == "session-123"
    assert state["request_count"] == 4

    log_records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert log_records[0]["args"][:2] == ["exec", "-"]
    assert log_records[1]["args"][:4] == ["exec", "resume", "session-123", "-"]
    assert log_records[2]["args"][:4] == ["exec", "resume", "session-123", "-"]
    assert log_records[3]["args"][:4] == ["exec", "resume", "session-123", "-"]
    assert "recent_history" not in log_records[0]["prompt"]
    assert "agent_response" not in log_records[0]["prompt"]
    assert '"web_search_context"' in log_records[0]["prompt"]
    assert '"web_search_context_status": "unchanged_from_previous_step"' in log_records[1]["prompt"]
    assert '"web_search_context_status": "unchanged_from_previous_step"' in log_records[3]["prompt"]
