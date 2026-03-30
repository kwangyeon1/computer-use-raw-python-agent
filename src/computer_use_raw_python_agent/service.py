from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import base64
import hashlib
import json

try:
    from .executor_client import ExecutorHttpClient, ExecutorStdioClient
    from .models import StepRequest, StepResponse
    from .prompting import render_prompt_bundle_from_step_request
    from .runtime import GUIOwlRawPythonRuntime
except ImportError:  # pragma: no cover - direct script execution fallback
    from executor_client import ExecutorHttpClient, ExecutorStdioClient
    from models import StepRequest, StepResponse
    from prompting import render_prompt_bundle_from_step_request
    from runtime import GUIOwlRawPythonRuntime


def _default_model_id() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root.parent / "models" / "gui-owl-1.5-8b-think-base").resolve())


def _load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_tail(path: str | Path, limit_chars: int = 2000) -> str:
    if not str(path).strip():
        return ""
    file_path = Path(path)
    if not file_path.exists() or file_path.is_dir():
        return ""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text[-limit_chars:]


def _ensure_run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_python_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code).replace("\r\n", "\n").strip().split("\n")).strip()


def _code_fingerprint(code: str) -> str:
    normalized = _normalize_python_code(code)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def build_executor_client(*, endpoint: str | None, mcp_command: list[str] | None, mcp_cwd: str | None):
    if bool(endpoint) == bool(mcp_command):
        raise RuntimeError("provide exactly one of endpoint or mcp_command")
    if endpoint:
        return ExecutorHttpClient(endpoint)
    return ExecutorStdioClient(mcp_command or [], cwd=mcp_cwd)


def generate_step_response(
    runtime: GUIOwlRawPythonRuntime,
    request: StepRequest,
    *,
    max_new_tokens: int,
) -> StepResponse:
    image_bytes = None
    if request.screenshot_base64:
        image_bytes = base64.b64decode(request.screenshot_base64)
    bundle = render_prompt_bundle_from_step_request(request)
    generated = runtime.generate_code(
        prompt_bundle=bundle,
        image_path=request.screenshot_path,
        image_bytes=image_bytes,
        use_blank_image=not bool(request.screenshot_path or image_bytes),
        max_new_tokens=max_new_tokens,
    )
    return StepResponse(
        python_code=generated.code,
        raw_text=generated.raw_text,
        model_id=generated.model_id,
        step_index=request.step_index,
        done=False,
        notes=[],
    )


def run_agent_control_loop(
    *,
    runtime: GUIOwlRawPythonRuntime,
    executor_client,
    user_prompt: str,
    policy: dict[str, Any],
    run_dir: str | Path,
    max_iterations: int,
    max_new_tokens: int,
    repeat_code_streak_limit: int = 2,
    replan_on_repeated_code: bool = False,
) -> dict[str, Any]:
    root = _ensure_run_dir(run_dir)
    (root / "payloads").mkdir(exist_ok=True)
    (root / "responses").mkdir(exist_ok=True)
    (root / "steps").mkdir(exist_ok=True)
    _write_json(
        root / "session.json",
        {
            "user_prompt": user_prompt,
            "policy": policy,
        },
    )

    state = executor_client.observe()
    _write_json(root / "observe-000.json", state)

    last_execution: dict[str, Any] = {}
    history: list[str] = []
    final_response: dict[str, Any] | None = None
    generated_steps = 0
    executed_steps = 0
    stopped_reason: str | None = None
    previous_code: str | None = None
    repeated_code_streak = 0
    replans_used = 0

    for step_index in range(max_iterations):
        request = StepRequest(
            user_prompt=user_prompt,
            policy=policy,
            screenshot_path=state.get("screenshot_path"),
            screenshot_base64=state.get("screenshot_base64"),
            screenshot_media_type=state.get("screenshot_media_type"),
            observation_text=state.get("observation_text"),
            recent_history=history,
            last_execution=last_execution,
            step_index=step_index,
        )
        request_path = root / "payloads" / f"step-{step_index:03d}.request.json"
        _write_json(request_path, request.to_dict())

        response = generate_step_response(runtime, request, max_new_tokens=max_new_tokens)
        generated_steps += 1
        normalized_code = _normalize_python_code(response.python_code)
        if normalized_code and normalized_code == previous_code:
            repeated_code_streak += 1
        else:
            repeated_code_streak = 1 if normalized_code else 0
        previous_code = normalized_code
        response.notes.append(f"code_fingerprint={_code_fingerprint(response.python_code)}")
        response.notes.append(f"code_repeat_streak={repeated_code_streak}")
        final_response = response.to_dict()
        response_path = root / "responses" / f"step-{step_index:03d}.response.json"
        _write_json(response_path, response.to_dict())

        if repeat_code_streak_limit > 0 and repeated_code_streak >= repeat_code_streak_limit:
            if replan_on_repeated_code and replans_used == 0:
                replans_used += 1
                repeated_code_streak = 0
                response.notes.append("replan_requested_due_to_repeated_code")
                final_response = response.to_dict()
                _write_json(response_path, response.to_dict())
                history.append(f"step-{step_index:03d}_replan_requested=repeated_code")
                history.append("system_hint=previous python code repeated; generate different code")
                continue
            stopped_reason = "repeated_code"
            response.notes.append("stopped_due_to_repeated_code")
            final_response = response.to_dict()
            _write_json(response_path, response.to_dict())
            history.append(f"step-{step_index:03d}_stopped=repeated_code")
            break

        step_id = f"step-{step_index:03d}"
        step_dir = root / "steps" / step_id
        exec_result = executor_client.execute(
            python_code=response.python_code,
            run_dir=str(step_dir),
            step_id=step_id,
            metadata={
                "agent_response": response.to_dict(),
                "step_index": step_index,
                "agent_session_id": root.name,
                "agent_step_dir": str(step_dir),
            },
        )
        _write_json(root / "responses" / f"{step_id}.executor.json", exec_result)

        record = dict(exec_result.get("record", {}))
        executed_steps += 1
        last_execution = {
            **record,
            "stdout_tail": exec_result.get("stdout_tail") or _read_tail(record.get("stdout_path", "")),
            "stderr_tail": exec_result.get("stderr_tail") or _read_tail(record.get("stderr_path", "")),
        }
        history.append(f"{step_id}_return_code={record.get('return_code', 'unknown')}")
        history.append(f"{step_id}_code_fingerprint={_code_fingerprint(response.python_code)}")
        state = {
            "screenshot_path": exec_result.get("screenshot_path"),
            "screenshot_base64": exec_result.get("screenshot_base64"),
            "screenshot_media_type": exec_result.get("screenshot_media_type"),
            "observation_text": exec_result.get("observation_text"),
        }

        if response.done:
            break

    summary = {
        "run_dir": str(root),
        "iterations": executed_steps,
        "generated_steps": generated_steps,
        "last_execution": last_execution,
        "final_response": final_response,
        "stopped_reason": stopped_reason,
        "repeat_code_streak_limit": repeat_code_streak_limit,
        "replan_on_repeated_code": replan_on_repeated_code,
        "replans_used": replans_used,
        "repeat_code_streak": repeated_code_streak,
    }
    _write_json(root / "loop-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="computer-use-raw-python-agent")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--mcp-command", nargs="+")
    parser.add_argument("--mcp-cwd")
    parser.add_argument("--model-id", default=_default_model_id())
    parser.add_argument("--processor-id")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repeat-code-streak-limit", type=int, default=2)
    parser.add_argument("--replan-on-repeated-code", action="store_true")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--disable-cpu-offload", action="store_true")
    parser.add_argument("--preload", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = GUIOwlRawPythonRuntime(
        model_id=args.model_id,
        processor_id=args.processor_id,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=not args.disable_4bit,
        compute_dtype=args.compute_dtype,
        device_map=args.device_map,
        enable_fp32_cpu_offload=not args.disable_cpu_offload,
    )
    if args.preload:
        runtime.ensure_loaded()
    executor_client = build_executor_client(
        endpoint=args.endpoint,
        mcp_command=args.mcp_command,
        mcp_cwd=args.mcp_cwd,
    )
    try:
        summary = run_agent_control_loop(
            runtime=runtime,
            executor_client=executor_client,
            user_prompt=args.prompt,
            policy=_load_json(args.policy),
            run_dir=args.run_dir,
            max_iterations=args.max_iterations,
            max_new_tokens=args.max_new_tokens,
            repeat_code_streak_limit=args.repeat_code_streak_limit,
            replan_on_repeated_code=args.replan_on_repeated_code,
        )
    finally:
        executor_client.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
