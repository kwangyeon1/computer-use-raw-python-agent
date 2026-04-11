from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import time


_DEFAULT_STATE_FILENAME = ".codex-agent-session.json"


def _load_request() -> dict[str, Any]:
    raw = sys.stdin.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("expected JSON object on stdin")
    return payload


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _compact_last_execution(value: Any) -> dict[str, Any]:
    payload = dict(value or {})
    if not payload:
        return {}
    compact: dict[str, Any] = {}
    for key in ("step_id", "return_code", "timed_out", "duration_s", "stdout_tail", "stderr_tail", "error_info"):
        if payload.get(key) is not None:
            compact[key] = payload.get(key)
    return compact


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _json_fingerprint(value: Any) -> str | None:
    if not value:
        return None
    normalized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _drop_empty_values(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        compact[key] = value
    return compact


def _compact_prompt_payload(
    request_payload: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_bundle = dict(request_payload.get("prompt_bundle") or {})
    generation_context = dict(request_payload.get("generation_context") or {})
    user_payload = _json_object(prompt_bundle.get("user_prompt"))
    web_search_context = user_payload.get("web_search_context")
    current_web_search_hash = _json_fingerprint(web_search_context)
    previous_web_search_hash = str(state.get("last_web_search_context_hash") or "") or None
    include_web_search_context = bool(web_search_context) and current_web_search_hash != previous_web_search_hash

    compact: dict[str, Any] = {
        "request_kind": str(
            user_payload.get("request_kind")
            or generation_context.get("request_kind")
            or request_payload.get("response_format")
            or "task_step"
        ),
        "step_id": str(generation_context.get("step_id") or ""),
        "step_index": int(generation_context.get("step_index", 0) or 0),
        "user_prompt": str(user_payload.get("user_prompt") or prompt_bundle.get("session_prompt") or ""),
        "runtime_policy": dict(user_payload.get("runtime_policy") or prompt_bundle.get("policy") or {}),
        "repair_context": user_payload.get("repair_context") or None,
        "strong_visual_grounding": bool(user_payload.get("strong_visual_grounding", False)),
        "reasoning_enabled": bool(user_payload.get("reasoning_enabled", False)),
        "observation_text": user_payload.get("observation_text"),
        "last_execution": _compact_last_execution(user_payload.get("last_execution")),
        "replan_requested": bool(user_payload.get("replan_requested", False)),
        "replan_reasons": _normalize_string_list(user_payload.get("replan_reasons")),
        "web_search_state": dict(user_payload.get("web_search_state") or {}),
        "output_requirement": str(user_payload.get("output_requirement") or "").strip(),
    }
    if include_web_search_context:
        compact["web_search_context"] = web_search_context
    if current_web_search_hash and not include_web_search_context:
        compact["web_search_context_status"] = "unchanged_from_previous_step"
    next_state = {
        **state,
        "last_step_id": compact["step_id"],
        "last_request_kind": compact["request_kind"],
    }
    if current_web_search_hash:
        next_state["last_web_search_context_hash"] = current_web_search_hash
    return _drop_empty_values(compact), next_state


def _render_codex_prompt(request_payload: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    prompt_bundle = dict(request_payload.get("prompt_bundle") or {})
    compact_payload, next_state = _compact_prompt_payload(request_payload, state)
    system_prompt = str(prompt_bundle.get("system_prompt") or "").strip()
    lines = [
        "Follow the agent generation contract exactly.",
        "",
        "[system_prompt]",
        system_prompt,
        "",
        "[current_step_context]",
        json.dumps(compact_payload, ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines).strip() + "\n", next_state


def _resolve_run_dir(request_payload: dict[str, Any]) -> Path | None:
    generation_context = dict(request_payload.get("generation_context") or {})
    run_dir = str(generation_context.get("run_dir") or "").strip()
    if not run_dir:
        return None
    return Path(run_dir)


def _session_state_path(*, run_dir: Path | None, filename: str) -> Path | None:
    if run_dir is None:
        return None
    return run_dir / filename


def _load_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_state(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_session_meta(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return None
    if not first_line:
        return None
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("type") != "session_meta":
        return None
    payload = record.get("payload")
    return dict(payload) if isinstance(payload, dict) else None


def _find_newest_exec_session(
    *,
    sessions_root: Path,
    started_at_s: float,
    cwd: str | None,
) -> str | None:
    if not sessions_root.exists():
        return None
    candidates = sorted(sessions_root.rglob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    threshold = float(started_at_s) - 2.0
    for path in candidates[:200]:
        try:
            if path.stat().st_mtime < threshold:
                break
        except OSError:
            continue
        meta = _read_session_meta(path)
        if not meta:
            continue
        if str(meta.get("source") or "") != "exec":
            continue
        if cwd and str(meta.get("cwd") or "") != cwd:
            continue
        session_id = str(meta.get("id") or "").strip()
        if session_id:
            return session_id
    return None


def _materialize_image(
    request_payload: dict[str, Any],
    *,
    run_dir: Path | None,
) -> Path | None:
    image_path = str(request_payload.get("image_path") or "").strip()
    if image_path:
        path = Path(image_path)
        if path.exists():
            return path
    image_base64 = str(request_payload.get("image_base64") or "").strip()
    if not image_base64:
        return None
    raw = base64.b64decode(image_base64)
    if run_dir is not None:
        image_dir = run_dir / ".codex-agent-inputs"
        image_dir.mkdir(parents=True, exist_ok=True)
        step_id = str(((request_payload.get("generation_context") or {}).get("step_id")) or "step").strip() or "step"
        image_path = image_dir / f"{step_id}.png"
        image_path.write_bytes(raw)
        return image_path
    with tempfile.NamedTemporaryFile(prefix="codex-agent-", suffix=".png", delete=False) as handle:
        handle.write(raw)
        return Path(handle.name)


def _build_codex_command(
    *,
    codex_bin: str,
    session_id: str | None,
    model: str | None,
    config_overrides: list[str],
    cwd: str | None,
    add_dirs: list[str],
    skip_git_repo_check: bool,
    full_auto: bool,
    bypass: bool,
    search: bool,
    image_path: Path | None,
    output_path: Path,
) -> list[str]:
    command = [codex_bin, "exec"]
    option_args: list[str] = []
    if model:
        option_args.extend(["-m", model])
    for item in config_overrides:
        option_args.extend(["-c", item])
    if not session_id and cwd:
        option_args.extend(["-C", cwd])
    if not session_id:
        for item in add_dirs:
            option_args.extend(["--add-dir", item])
    if skip_git_repo_check:
        option_args.append("--skip-git-repo-check")
    if full_auto:
        option_args.append("--full-auto")
    if bypass:
        option_args.append("--dangerously-bypass-approvals-and-sandbox")
    if search and not session_id:
        option_args.append("--search")
    if image_path is not None:
        option_args.extend(["-i", str(image_path)])
    option_args.extend(["-o", str(output_path)])
    if session_id:
        command.extend(["resume", *option_args, session_id, "-"])
    else:
        command.extend([*option_args, "-"])
    return command


def _invoke_codex(
    command: list[str],
    *,
    prompt: str,
    cwd: str | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
    )


def _load_last_message(path: Path, completed: subprocess.CompletedProcess[str]) -> str:
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    stdout = str(completed.stdout or "").strip()
    if stdout:
        return stdout
    raise RuntimeError("codex did not produce a final message")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m computer_use_raw_python_agent.codex_backend")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model")
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("-C", "--cd", dest="cwd")
    parser.add_argument("--add-dir", action="append", default=[])
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--full-auto", action="store_true")
    parser.add_argument("--dangerously-bypass-approvals-and-sandbox", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--sessions-root", default=str(Path.home() / ".codex" / "sessions"))
    parser.add_argument("--session-state-filename", default=_DEFAULT_STATE_FILENAME)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request_payload = _load_request()
    run_dir = _resolve_run_dir(request_payload)
    state_path = _session_state_path(run_dir=run_dir, filename=str(args.session_state_filename))
    state = _load_state(state_path)
    prompt_text, next_state = _render_codex_prompt(request_payload, state)
    image_path = _materialize_image(request_payload, run_dir=run_dir)

    temp_dir = run_dir / ".codex-agent-tmp" if run_dir is not None else Path(tempfile.mkdtemp(prefix="codex-agent-"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / "last-message.txt"
    codex_cwd = str(Path(args.cwd).resolve()) if args.cwd else str(Path.cwd())
    session_id = str(state.get("codex_session_id") or "").strip() or None
    resumed_existing_session = bool(session_id)
    started_at_s = time.time()
    command = _build_codex_command(
        codex_bin=str(args.codex_bin),
        session_id=session_id,
        model=str(args.model) if args.model else None,
        config_overrides=[str(item) for item in args.config],
        cwd=codex_cwd,
        add_dirs=[str(Path(item).resolve()) for item in args.add_dir],
        skip_git_repo_check=bool(args.skip_git_repo_check),
        full_auto=bool(args.full_auto),
        bypass=bool(args.dangerously_bypass_approvals_and_sandbox),
        search=bool(args.search),
        image_path=image_path,
        output_path=output_path,
    )
    completed = _invoke_codex(command, prompt=prompt_text, cwd=codex_cwd)
    if completed.returncode != 0 and session_id:
        lower_stderr = str(completed.stderr or "").lower()
        if "session" in lower_stderr and ("not found" in lower_stderr or "unknown" in lower_stderr or "invalid" in lower_stderr):
            state.pop("codex_session_id", None)
            command = _build_codex_command(
                codex_bin=str(args.codex_bin),
                session_id=None,
                model=str(args.model) if args.model else None,
                config_overrides=[str(item) for item in args.config],
                cwd=codex_cwd,
                add_dirs=[str(Path(item).resolve()) for item in args.add_dir],
                skip_git_repo_check=bool(args.skip_git_repo_check),
                full_auto=bool(args.full_auto),
                bypass=bool(args.dangerously_bypass_approvals_and_sandbox),
                search=bool(args.search),
                image_path=image_path,
                output_path=output_path,
            )
            started_at_s = time.time()
            completed = _invoke_codex(command, prompt=prompt_text, cwd=codex_cwd)
            session_id = None
            resumed_existing_session = False
    if completed.returncode != 0:
        detail = str(completed.stderr or completed.stdout or f"return code {completed.returncode}").strip()
        raise SystemExit(detail or "codex exec failed")

    response_text = _load_last_message(output_path, completed)
    if not session_id:
        session_id = _find_newest_exec_session(
            sessions_root=Path(str(args.sessions_root)).expanduser(),
            started_at_s=started_at_s,
            cwd=codex_cwd,
        )

    next_state["codex_session_id"] = session_id
    next_state["codex_cwd"] = codex_cwd
    next_state["request_count"] = int(state.get("request_count", 0) or 0) + 1
    _write_state(state_path, next_state)

    print(
        json.dumps(
            {
                "raw_text": response_text,
                "model_id": f"codex:{args.model}" if args.model else "codex",
                "backend_id": "codex-resume" if resumed_existing_session else "codex-exec",
                "session_id": session_id,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
