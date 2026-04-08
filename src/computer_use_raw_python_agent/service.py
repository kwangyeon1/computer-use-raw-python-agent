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
    from .prompting import render_prompt_bundle_from_step_request, render_web_search_decision_bundle_from_step_request
    from .runtime import GUIOwlRawPythonRuntime
    from .web_search import (
        SearXNGClient,
        WebSearchDecision,
        make_web_search_error_result,
        make_web_search_skipped_result,
        web_search_cache_key,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from executor_client import ExecutorHttpClient, ExecutorStdioClient
    from models import StepRequest, StepResponse
    from prompting import render_prompt_bundle_from_step_request, render_web_search_decision_bundle_from_step_request
    from runtime import GUIOwlRawPythonRuntime
    from web_search import (
        SearXNGClient,
        WebSearchDecision,
        make_web_search_error_result,
        make_web_search_skipped_result,
        web_search_cache_key,
    )


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


def _is_empty_python_code(code: str) -> bool:
    return not bool(_normalize_python_code(code))


def _has_task_complete_marker(code: str) -> bool:
    for line in str(code or "").splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        return stripped.startswith("# task_complete")
    return False


def _looks_like_completion_noop(code: str, raw_text: str) -> bool:
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    risky_tokens = (
        "pyautogui",
        "subprocess",
        "webbrowser",
        "selenium",
        "requests",
        "urllib",
        "winget",
        "pip install",
        "click(",
        "press(",
        "hotkey(",
        "typewrite(",
        "write(",
        "os.startfile",
    )
    if any(token in normalized for token in risky_tokens):
        return False
    completion_phrases = (
        "task is complete",
        "task has been completed",
        "already complete",
        "already completed",
        "completed successfully",
        "already open and on",
        "already on the",
    )
    raw_lower = str(raw_text or "").lower()
    if not any(phrase in raw_lower for phrase in completion_phrases):
        return False
    return len(normalized.splitlines()) <= 20


def _infer_response_done(*, python_code: str, raw_text: str) -> bool:
    return _has_task_complete_marker(python_code) or _looks_like_completion_noop(python_code, raw_text)


def _tail_history(history: list[str], *, limit: int = 2) -> list[str]:
    if limit <= 0:
        return []
    return list(history[-limit:])


def _history_for_step(history: list[str]) -> list[str]:
    return _tail_history(history, limit=2)


def _history_for_empty_retry(history: list[str], *, step_index: int) -> list[str]:
    retry_history = _history_for_step(history)
    retry_history.append(f"step-{step_index:03d}_empty_generation=1")
    retry_history.append("system_hint=previous model output was empty; return non-empty executable Python only")
    return retry_history


def _history_for_dependency_repair(history: list[str], *, failed_step_id: str) -> list[str]:
    failed_step_history = [entry for entry in history if entry.startswith(f"{failed_step_id}_")]
    if not failed_step_history:
        return _history_for_step(history)
    return failed_step_history[-2:]


def _history_for_web_search(history: list[str], *, limit: int = 4) -> list[str]:
    base = _history_for_step(history)
    filtered = [entry for entry in history if "_web_search_" in entry or entry.startswith("system_hint=")]
    if not filtered:
        return base
    combined = base + filtered[-2:]
    return list(dict.fromkeys(combined))[-limit:]


def _state_visual_hash(state: dict[str, Any]) -> str | None:
    screenshot_base64 = str(state.get("screenshot_base64") or "").strip()
    if screenshot_base64:
        return hashlib.sha1(screenshot_base64.encode("utf-8")).hexdigest()[:12]
    screenshot_path = str(state.get("screenshot_path") or "").strip()
    if screenshot_path:
        path = Path(screenshot_path)
        if path.exists() and path.is_file():
            return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    return None


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
        done=_infer_response_done(python_code=generated.code, raw_text=generated.raw_text),
        notes=[],
    )


def generate_web_search_decision(
    runtime: GUIOwlRawPythonRuntime,
    request: StepRequest,
    *,
    max_new_tokens: int,
    decision_max_new_tokens: int,
    use_image: bool,
    reasoning_enabled: bool,
    web_search_max_uses: int,
    web_search_uses: int,
    web_search_queries: list[str],
) -> tuple[WebSearchDecision, dict[str, Any]]:
    image_bytes = None
    image_path = None
    if use_image and request.screenshot_base64:
        image_bytes = base64.b64decode(request.screenshot_base64)
    if use_image and request.screenshot_path:
        image_path = request.screenshot_path
    bundle = render_web_search_decision_bundle_from_step_request(
        request,
        reasoning_enabled=reasoning_enabled,
        web_search_max_uses=web_search_max_uses,
        web_search_uses=web_search_uses,
        web_search_queries=web_search_queries,
    )
    generated = runtime.generate_text(
        prompt_bundle=bundle,
        image_path=image_path,
        image_bytes=image_bytes,
        use_blank_image=False,
        max_new_tokens=min(int(max_new_tokens), int(decision_max_new_tokens)),
    )
    decision = WebSearchDecision.from_text(generated.text)
    return decision, generated.to_dict()


def _extract_last_execution(exec_result: dict[str, Any]) -> dict[str, Any]:
    record = dict(exec_result.get("record", {}))
    return {
        **record,
        "stdout_tail": exec_result.get("stdout_tail") or _read_tail(record.get("stdout_path", "")),
        "stderr_tail": exec_result.get("stderr_tail") or _read_tail(record.get("stderr_path", "")),
        "error_info": exec_result.get("error_info"),
    }


def _extract_state(exec_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "screenshot_path": exec_result.get("screenshot_path"),
        "screenshot_base64": exec_result.get("screenshot_base64"),
        "screenshot_media_type": exec_result.get("screenshot_media_type"),
        "observation_text": exec_result.get("observation_text"),
    }


def _execute_code_step(
    *,
    executor_client,
    root: Path,
    step_id: str,
    python_code: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    step_dir = root / "steps" / step_id
    exec_result = executor_client.execute(
        python_code=python_code,
        run_dir=str(step_dir),
        step_id=step_id,
        metadata=metadata,
    )
    _write_json(root / "responses" / f"{step_id}.executor.json", exec_result)
    return exec_result, _extract_last_execution(exec_result), _extract_state(exec_result)


def _maybe_perform_web_search(
    *,
    runtime: GUIOwlRawPythonRuntime,
    request: StepRequest,
    root: Path,
    step_id: str,
    history: list[str],
    max_new_tokens: int,
    web_search_decision_use_image: bool,
    web_search_decision_reasoning_enabled: bool,
    web_search_decision_max_new_tokens: int,
    web_search_max_uses: int,
    web_search_uses: int,
    web_search_queries: list[str],
    web_search_cache: dict[str, dict[str, Any]],
    searxng_client: SearXNGClient,
    web_search_top_k: int,
    searxng_preferred_engines: list[str],
) -> tuple[dict[str, Any], int, list[str]]:
    search_request = StepRequest(
        user_prompt=request.user_prompt,
        policy=request.policy,
        request_kind="web_search_decision",
        repair_context={},
        replan_requested=request.replan_requested,
        replan_reasons=request.replan_reasons,
        strong_visual_grounding=request.strong_visual_grounding,
        reasoning_enabled=web_search_decision_reasoning_enabled,
        screenshot_path=request.screenshot_path if web_search_decision_use_image else None,
        screenshot_base64=request.screenshot_base64 if web_search_decision_use_image else None,
        screenshot_media_type=request.screenshot_media_type if web_search_decision_use_image else None,
        observation_text=request.observation_text,
        web_search_context={},
        recent_history=_history_for_web_search(history),
        last_execution=request.last_execution,
        step_index=request.step_index,
    )
    decision_request_path = root / "payloads" / f"{step_id}.web-search-decision.request.json"
    _write_json(decision_request_path, search_request.to_dict())
    decision, generated = generate_web_search_decision(
        runtime,
        search_request,
        max_new_tokens=max_new_tokens,
        decision_max_new_tokens=web_search_decision_max_new_tokens,
        use_image=web_search_decision_use_image,
        reasoning_enabled=web_search_decision_reasoning_enabled,
        web_search_max_uses=web_search_max_uses,
        web_search_uses=web_search_uses,
        web_search_queries=web_search_queries[-3:],
    )
    _write_json(
        root / "responses" / f"{step_id}.web-search-decision.response.json",
        {
            "decision": decision.to_dict(),
            "model_response": generated,
        },
    )

    if not decision.use_web_search:
        return {}, web_search_uses, web_search_queries

    if not decision.query:
        return {}, web_search_uses, web_search_queries

    cache_key = web_search_cache_key(
        decision.query,
        allowed_domains=decision.allowed_domains,
        blocked_domains=decision.blocked_domains,
        preferred_engines=searxng_preferred_engines,
    )
    if cache_key in web_search_cache:
        cached_payload = dict(web_search_cache[cache_key])
        cached_payload["cached"] = True
        cached_payload["reason"] = decision.reason or cached_payload.get("reason", "")
        _write_json(root / "responses" / f"{step_id}.web-search-result.json", cached_payload)
        history.append(f"{step_id}_web_search_cached={decision.query}")
        return cached_payload, web_search_uses, web_search_queries

    if web_search_uses >= web_search_max_uses:
        skipped = make_web_search_skipped_result(
            query=decision.query,
            allowed_domains=decision.allowed_domains,
            blocked_domains=decision.blocked_domains,
            reason=decision.reason or "search requested but per-run web search limit was reached",
            status="skipped_max_uses",
        ).to_dict()
        _write_json(root / "responses" / f"{step_id}.web-search-result.json", skipped)
        history.append(f"{step_id}_web_search_skipped=max_uses")
        return skipped, web_search_uses, web_search_queries

    try:
        result = searxng_client.search(
            query=decision.query,
            top_k=web_search_top_k,
            allowed_domains=decision.allowed_domains,
            blocked_domains=decision.blocked_domains,
            preferred_engines=searxng_preferred_engines,
        ).to_dict()
    except Exception as exc:
        result = make_web_search_error_result(
            query=decision.query,
            allowed_domains=decision.allowed_domains,
            blocked_domains=decision.blocked_domains,
            reason=decision.reason or "web search failed",
            error=str(exc),
        ).to_dict()
        _write_json(root / "responses" / f"{step_id}.web-search-result.json", result)
        history.append(f"{step_id}_web_search_error={decision.query}")
        return result, web_search_uses + 1, web_search_queries + [decision.query]

    result["reason"] = decision.reason or result.get("reason", "")
    web_search_cache[cache_key] = dict(result)
    _write_json(root / "responses" / f"{step_id}.web-search-result.json", result)
    history.append(f"{step_id}_web_search_query={decision.query}")
    history.append(f"{step_id}_web_search_results={int(result.get('result_count', 0))}")
    return result, web_search_uses + 1, web_search_queries + [decision.query]


def _attempt_dependency_repair(
    *,
    runtime: GUIOwlRawPythonRuntime,
    executor_client,
    root: Path,
    user_prompt: str,
    policy: dict[str, Any],
    strong_visual_grounding: bool,
    reasoning_enabled: bool,
    state: dict[str, Any],
    history: list[str],
    last_execution: dict[str, Any],
    original_response: StepResponse,
    original_step_id: str,
    step_index: int,
    max_new_tokens: int,
    repair_attempt_index: int,
    allow_shell_fallback: bool,
) -> dict[str, Any]:
    error_info = dict(last_execution.get("error_info") or {})
    install_name = str(error_info.get("install_name") or error_info.get("module_name") or "").strip()
    if not install_name:
        return {"handled": False}

    strategies = ["pip_install"]
    if allow_shell_fallback:
        strategies.append("shell_fallback")
    executions_added = 0

    for strategy_index, strategy in enumerate(strategies):
        repair_request = StepRequest(
            user_prompt=user_prompt,
            policy=policy,
            request_kind="dependency_repair",
            repair_context={
                "reason": "missing_python_module",
                "module_name": error_info.get("module_name"),
                "install_name": install_name,
                "repair_strategy": strategy,
                "repair_attempt_index": repair_attempt_index,
                "failed_python_code": original_response.python_code,
                "failed_step_id": original_step_id,
                "stderr_tail": last_execution.get("stderr_tail"),
            },
            strong_visual_grounding=strong_visual_grounding,
            reasoning_enabled=reasoning_enabled,
            screenshot_path=state.get("screenshot_path"),
            screenshot_base64=state.get("screenshot_base64"),
            screenshot_media_type=state.get("screenshot_media_type"),
            observation_text=state.get("observation_text"),
            recent_history=_history_for_dependency_repair(history, failed_step_id=original_step_id),
            last_execution=last_execution,
            step_index=step_index,
        )
        repair_request_name = f"{original_step_id}.repair-{repair_attempt_index:02d}-{strategy_index:02d}"
        _write_json(root / "payloads" / f"{repair_request_name}.request.json", repair_request.to_dict())
        repair_response = generate_step_response(runtime, repair_request, max_new_tokens=max_new_tokens)
        repair_response.notes.append("dependency_repair_mode=true")
        repair_response.notes.append(f"dependency_repair_strategy={strategy}")
        _write_json(root / "responses" / f"{repair_request_name}.response.json", repair_response.to_dict())

        repair_exec_result, repair_last_execution, repair_state = _execute_code_step(
            executor_client=executor_client,
            root=root,
            step_id=repair_request_name,
            python_code=repair_response.python_code,
            metadata={
                "agent_response": repair_response.to_dict(),
                "repair_context": repair_request.repair_context,
                "step_index": step_index,
                "agent_session_id": root.name,
                "agent_step_dir": str(root / "steps" / repair_request_name),
            },
        )
        executions_added += 1
        history.append(f"{repair_request_name}_return_code={repair_last_execution.get('return_code', 'unknown')}")
        history.append(f"{repair_request_name}_repair_strategy={strategy}")

        if int(repair_last_execution.get("return_code", 1)) != 0:
            last_execution = repair_last_execution
            state = repair_state
            continue

        retry_step_id = f"{original_step_id}.retry-{repair_attempt_index:02d}-{strategy_index:02d}"
        retry_exec_result, retry_last_execution, retry_state = _execute_code_step(
            executor_client=executor_client,
            root=root,
            step_id=retry_step_id,
            python_code=original_response.python_code,
            metadata={
                "agent_response": original_response.to_dict(),
                "dependency_repair_retry": True,
                "repair_context": repair_request.repair_context,
                "step_index": step_index,
                "agent_session_id": root.name,
                "agent_step_dir": str(root / "steps" / retry_step_id),
            },
        )
        executions_added += 1
        history.append(f"{retry_step_id}_return_code={retry_last_execution.get('return_code', 'unknown')}")
        history.append(f"{retry_step_id}_code_fingerprint={_code_fingerprint(original_response.python_code)}")

        retry_error_info = dict(retry_last_execution.get("error_info") or {})
        if int(retry_last_execution.get("return_code", 1)) == 0:
            return {
                "handled": True,
                "success": True,
                "executions_added": executions_added,
                "repair_response": repair_response.to_dict(),
                "last_execution": retry_last_execution,
                "state": retry_state,
            }
        if retry_error_info.get("kind") != "missing_python_module":
            return {
                "handled": True,
                "success": False,
                "executions_added": executions_added,
                "repair_response": repair_response.to_dict(),
                "last_execution": retry_last_execution,
                "state": retry_state,
            }
        last_execution = retry_last_execution
        state = retry_state

    return {
        "handled": True,
        "success": False,
        "executions_added": executions_added,
        "last_execution": last_execution,
        "state": state,
    }


def run_agent_control_loop(
    *,
    runtime: GUIOwlRawPythonRuntime,
    executor_client,
    user_prompt: str,
    policy: dict[str, Any],
    run_dir: str | Path,
    max_iterations: int,
    max_new_tokens: int,
    strong_visual_grounding: bool = False,
    reasoning_enabled: bool = False,
    replan_enabled: bool = False,
    replan_max_attempts: int = 1,
    web_search_enabled: bool = False,
    web_search_engine: str = "searxng",
    searxng_base_url: str = "http://127.0.0.1:8080",
    web_search_top_k: int = 5,
    web_search_max_uses: int = 3,
    web_search_timeout_s: float = 10.0,
    searxng_preferred_engines: list[str] | None = None,
    web_search_decision_use_image: bool = False,
    web_search_decision_reasoning_enabled: bool = False,
    web_search_decision_max_new_tokens: int = 64,
    dependency_repair_enabled: bool = False,
    dependency_repair_max_attempts: int = 2,
    dependency_repair_allow_shell_fallback: bool = False,
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
    if web_search_enabled and str(web_search_engine).strip().lower() != "searxng":
        raise RuntimeError("only the searxng web search engine is supported")

    state = executor_client.observe()
    _write_json(root / "observe-000.json", state)

    last_execution: dict[str, Any] = {}
    history: list[str] = []
    final_response: dict[str, Any] | None = None
    generated_steps = 0
    executed_steps = 0
    stopped_reason: str | None = None
    previous_executed_code: str | None = None
    previous_visual_hash = _state_visual_hash(state)
    pending_replan_reasons: list[str] = []
    replans_used = 0
    web_search_uses = 0
    web_search_queries: list[str] = []
    web_search_cache: dict[str, dict[str, Any]] = {}
    dependency_repairs_used = 0
    empty_generation_retries_used = 0
    normalized_preferred_search_engines = [str(engine).strip().lower() for engine in (searxng_preferred_engines or []) if str(engine).strip()]
    searxng_client = SearXNGClient(base_url=searxng_base_url, timeout_s=web_search_timeout_s) if web_search_enabled else None

    for step_index in range(max_iterations):
        active_replan_reasons = list(pending_replan_reasons)
        pending_replan_reasons = []
        request = StepRequest(
            user_prompt=user_prompt,
            policy=policy,
            replan_requested=bool(active_replan_reasons),
            replan_reasons=active_replan_reasons,
            strong_visual_grounding=strong_visual_grounding,
            reasoning_enabled=reasoning_enabled,
            screenshot_path=state.get("screenshot_path"),
            screenshot_base64=state.get("screenshot_base64"),
            screenshot_media_type=state.get("screenshot_media_type"),
            observation_text=state.get("observation_text"),
            web_search_context={},
            recent_history=_history_for_step(history),
            last_execution=last_execution,
            step_index=step_index,
        )
        step_id = f"step-{step_index:03d}"
        if web_search_enabled and request.request_kind == "task_step" and searxng_client is not None:
            web_search_context, web_search_uses, web_search_queries = _maybe_perform_web_search(
                runtime=runtime,
                request=request,
                root=root,
                step_id=step_id,
                history=history,
                max_new_tokens=max_new_tokens,
                web_search_decision_use_image=web_search_decision_use_image,
                web_search_decision_reasoning_enabled=web_search_decision_reasoning_enabled,
                web_search_decision_max_new_tokens=web_search_decision_max_new_tokens,
                web_search_max_uses=web_search_max_uses,
                web_search_uses=web_search_uses,
                web_search_queries=web_search_queries,
                web_search_cache=web_search_cache,
                searxng_client=searxng_client,
                web_search_top_k=web_search_top_k,
                searxng_preferred_engines=normalized_preferred_search_engines,
            )
            request.web_search_context = web_search_context
        request_path = root / "payloads" / f"step-{step_index:03d}.request.json"
        _write_json(request_path, request.to_dict())

        response = generate_step_response(runtime, request, max_new_tokens=max_new_tokens)
        generated_steps += 1
        normalized_code = _normalize_python_code(response.python_code)
        response_path = root / "responses" / f"step-{step_index:03d}.response.json"
        if _is_empty_python_code(response.python_code):
            empty_attempt_path = root / "responses" / f"step-{step_index:03d}.empty-attempt-00.response.json"
            response.notes.append("empty_generation_detected")
            _write_json(empty_attempt_path, response.to_dict())
            retry_request = StepRequest(
                user_prompt=user_prompt,
                policy=policy,
                replan_requested=bool(active_replan_reasons),
                replan_reasons=active_replan_reasons,
                strong_visual_grounding=strong_visual_grounding,
                reasoning_enabled=reasoning_enabled,
                screenshot_path=state.get("screenshot_path"),
                screenshot_base64=state.get("screenshot_base64"),
                screenshot_media_type=state.get("screenshot_media_type"),
                observation_text=state.get("observation_text"),
                web_search_context=request.web_search_context,
                recent_history=_history_for_empty_retry(history, step_index=step_index),
                last_execution=last_execution,
                step_index=step_index,
            )
            retry_request_path = root / "payloads" / f"step-{step_index:03d}.empty-retry-00.request.json"
            _write_json(retry_request_path, retry_request.to_dict())
            retry_response = generate_step_response(runtime, retry_request, max_new_tokens=max_new_tokens)
            generated_steps += 1
            empty_generation_retries_used += 1
            retry_response.notes.append("retry_due_to_empty_generation")
            retry_normalized_code = _normalize_python_code(retry_response.python_code)
            retry_response_path = root / "responses" / f"step-{step_index:03d}.empty-retry-00.response.json"
            _write_json(retry_response_path, retry_response.to_dict())
            if _is_empty_python_code(retry_response.python_code):
                retry_response.notes.append("stopped_due_to_empty_generation")
                final_response = retry_response.to_dict()
                _write_json(retry_response_path, retry_response.to_dict())
                _write_json(response_path, retry_response.to_dict())
                stopped_reason = "empty_generation"
                history.append(f"step-{step_index:03d}_stopped=empty_generation")
                break
            response = retry_response
            normalized_code = retry_normalized_code

        response.notes.append(f"code_fingerprint={_code_fingerprint(response.python_code)}")
        final_response = response.to_dict()
        _write_json(response_path, response.to_dict())

        _, last_execution, state = _execute_code_step(
            executor_client=executor_client,
            root=root,
            step_id=step_id,
            python_code=response.python_code,
            metadata={
                "agent_response": response.to_dict(),
                "step_index": step_index,
                "agent_session_id": root.name,
                "agent_step_dir": str(root / "steps" / step_id),
            },
        )
        executed_steps += 1
        history.append(f"{step_id}_return_code={last_execution.get('return_code', 'unknown')}")
        history.append(f"{step_id}_code_fingerprint={_code_fingerprint(response.python_code)}")

        error_info = dict(last_execution.get("error_info") or {})
        repairable_missing_module = (
            dependency_repair_enabled
            and bool(policy.get("allow_package_install", False))
            and dependency_repairs_used < dependency_repair_max_attempts
            and error_info.get("kind") == "missing_python_module"
            and bool(error_info.get("repairable", False))
        )
        if repairable_missing_module:
            repair_attempt_index = dependency_repairs_used
            dependency_repairs_used += 1
            repair_result = _attempt_dependency_repair(
                runtime=runtime,
                executor_client=executor_client,
                root=root,
                user_prompt=user_prompt,
                policy=policy,
                strong_visual_grounding=strong_visual_grounding,
                reasoning_enabled=reasoning_enabled,
                state=state,
                history=history,
                last_execution=last_execution,
                original_response=response,
                original_step_id=step_id,
                step_index=step_index,
                max_new_tokens=max_new_tokens,
                repair_attempt_index=repair_attempt_index,
                allow_shell_fallback=dependency_repair_allow_shell_fallback,
            )
            executed_steps += int(repair_result.get("executions_added", 0))
            if repair_result.get("handled"):
                last_execution = dict(repair_result.get("last_execution") or last_execution)
                state = dict(repair_result.get("state") or state)

        if response.done and int(last_execution.get("return_code", 0) or 0) == 0:
            history.append(f"{step_id}_completed=1")
            final_response = response.to_dict()
            stopped_reason = stopped_reason or "task_completed"
            break

        current_visual_hash = _state_visual_hash(state)
        replan_reasons: list[str] = []
        if previous_executed_code and normalized_code and normalized_code == previous_executed_code:
            replan_reasons.append("repeated_code_execution")
        dependency_error_handled = repairable_missing_module and dependency_repairs_used > repair_attempt_index if repairable_missing_module else False
        if int(last_execution.get("return_code", 0) or 0) != 0 and not dependency_error_handled:
            replan_reasons.append("execution_error")
        if previous_visual_hash and current_visual_hash and previous_visual_hash == current_visual_hash:
            replan_reasons.append("no_visual_change")
        previous_executed_code = normalized_code or previous_executed_code
        previous_visual_hash = current_visual_hash

        if (
            replan_enabled
            and replan_reasons
            and replans_used < replan_max_attempts
            and step_index + 1 < max_iterations
        ):
            unique_reasons = list(dict.fromkeys(replan_reasons))
            replans_used += 1
            pending_replan_reasons = unique_reasons
            response.notes.append(f"replan_reasons={','.join(unique_reasons)}")
            final_response = response.to_dict()
            _write_json(response_path, response.to_dict())
            history.append(f"{step_id}_replan_requested={','.join(unique_reasons)}")
            history.append("system_hint=previous attempt repeated, failed, or did not visibly change the UI; generate a materially different next Python step")

        if response.done:
            break
    else:
        if stopped_reason is None and final_response and not bool(final_response.get("done", False)):
            stopped_reason = "max_iterations_reached"

    summary = {
        "run_dir": str(root),
        "iterations": executed_steps,
        "generated_steps": generated_steps,
        "last_execution": last_execution,
        "final_response": final_response,
        "stopped_reason": stopped_reason,
        "strong_visual_grounding": strong_visual_grounding,
        "reasoning_enabled": reasoning_enabled,
        "replan_enabled": replan_enabled,
        "replan_max_attempts": replan_max_attempts,
        "replans_used": replans_used,
        "pending_replan_reasons": pending_replan_reasons,
        "web_search_enabled": web_search_enabled,
        "web_search_engine": web_search_engine if web_search_enabled else None,
        "searxng_base_url": searxng_base_url if web_search_enabled else None,
        "searxng_preferred_engines": normalized_preferred_search_engines if web_search_enabled else [],
        "web_search_decision_use_image": web_search_decision_use_image if web_search_enabled else False,
        "web_search_decision_reasoning_enabled": web_search_decision_reasoning_enabled if web_search_enabled else False,
        "web_search_decision_max_new_tokens": web_search_decision_max_new_tokens,
        "web_search_top_k": web_search_top_k,
        "web_search_max_uses": web_search_max_uses,
        "web_search_uses": web_search_uses,
        "web_search_queries": web_search_queries,
        "empty_generation_retries_used": empty_generation_retries_used,
        "dependency_repair_enabled": dependency_repair_enabled,
        "dependency_repair_max_attempts": dependency_repair_max_attempts,
        "dependency_repair_allow_shell_fallback": dependency_repair_allow_shell_fallback,
        "dependency_repairs_used": dependency_repairs_used,
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
    parser.add_argument("--strong-visual-grounding", action="store_true")
    parser.add_argument("--reasoning-enabled", action="store_true")
    parser.add_argument("--replan-enabled", action="store_true")
    parser.add_argument("--replan-max-attempts", type=int, default=1)
    parser.add_argument("--web-search-enabled", action="store_true")
    parser.add_argument("--web-search-engine", default="searxng")
    parser.add_argument("--searxng-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--searxng-preferred-engine", action="append")
    parser.add_argument("--web-search-decision-use-image", action="store_true")
    parser.add_argument("--web-search-decision-reasoning-enabled", action="store_true")
    parser.add_argument("--web-search-decision-max-new-tokens", type=int, default=64)
    parser.add_argument("--web-search-top-k", type=int, default=5)
    parser.add_argument("--web-search-max-uses", type=int, default=3)
    parser.add_argument("--web-search-timeout-s", type=float, default=10.0)
    parser.add_argument("--dependency-repair-enabled", action="store_true")
    parser.add_argument("--dependency-repair-max-attempts", type=int, default=2)
    parser.add_argument("--dependency-repair-allow-shell-fallback", action="store_true")
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
            strong_visual_grounding=args.strong_visual_grounding,
            reasoning_enabled=args.reasoning_enabled,
            replan_enabled=args.replan_enabled,
            replan_max_attempts=args.replan_max_attempts,
            web_search_enabled=args.web_search_enabled,
            web_search_engine=args.web_search_engine,
            searxng_base_url=args.searxng_base_url,
            searxng_preferred_engines=list(args.searxng_preferred_engine or []),
            web_search_decision_use_image=args.web_search_decision_use_image,
            web_search_decision_reasoning_enabled=args.web_search_decision_reasoning_enabled,
            web_search_decision_max_new_tokens=args.web_search_decision_max_new_tokens,
            web_search_top_k=args.web_search_top_k,
            web_search_max_uses=args.web_search_max_uses,
            web_search_timeout_s=args.web_search_timeout_s,
            dependency_repair_enabled=args.dependency_repair_enabled,
            dependency_repair_max_attempts=args.dependency_repair_max_attempts,
            dependency_repair_allow_shell_fallback=args.dependency_repair_allow_shell_fallback,
        )
    finally:
        executor_client.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
