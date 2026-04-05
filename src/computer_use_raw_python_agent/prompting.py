from __future__ import annotations

from typing import Iterable
import json

try:
    from .models import PromptBundle, RuntimePolicy, StepRequest
except ImportError:  # pragma: no cover - direct script execution fallback
    from models import PromptBundle, RuntimePolicy, StepRequest


RAW_PYTHON_SYSTEM_PROMPT = """You are a vision-conditioned local computer-use code generator.

Return executable Python only.
Do not return markdown.
Do not explain your reasoning.

You may use unrestricted local Python if the runtime policy allows it.
You may perform GUI actions, file operations, subprocess execution, and network calls.

Prefer concise, deterministic code.
Prefer helper functions when possible:
- focus_window
- press_key
- press_hotkey
- type_text
- switch_input_locale
- sleep
- wait_for_window
- capture_note

If helper functions are not sufficient, direct library usage is allowed.
Always generate code that can run as a standalone script.

If request_kind is dependency_repair:
- Generate Python only for repairing the reported dependency issue.
- Do not continue the main GUI task in that response.
- Prefer using sys.executable -m pip install <package> first.
- If repair_strategy is shell_fallback, you may use subprocess to run shell or batch installation commands.

If replan_requested is true:
- Treat replan_reasons as hard signals that the previous strategy did not work.
- Generate a materially different next step.
- Use the latest screenshot and last_execution as the primary basis for the new strategy.
- Do not repeat the same mechanism unless the screen state clearly changed and justifies it.
"""

REASONING_ENABLED_APPEND = """
If reasoning_enabled is true:
- You may use extra internal reasoning to choose the next action.
- Keep the final answer strictly executable Python only.
- Do not include prose outside Python.
- If the model emits <think> or similar reasoning markers, ensure the executable Python remains extractable as the final output.
"""

STRONG_VISUAL_GROUNDING_APPEND = """
If strong_visual_grounding is true:
- Treat the latest screenshot as primary evidence for the current computer state.
- Base the next Python step on what is visibly present on screen, not only on the original user prompt.
- Use recent_history and last_execution together with the screenshot before deciding the next action.
- If the screenshot suggests the previous action already changed the UI, adapt your next code to the new state.
- Avoid repeating the same generic code unless the screenshot and history clearly justify doing it again.
- Prefer code that interacts with what is already visible over restarting the whole task from scratch.
"""


def render_user_prompt(
    session_prompt: str,
    policy: RuntimePolicy,
    observation_text: str | None = None,
    recent_history: Iterable[str] | None = None,
    last_execution: dict | None = None,
    replan_requested: bool = False,
    replan_reasons: Iterable[str] | None = None,
    strong_visual_grounding: bool = False,
    reasoning_enabled: bool = False,
) -> str:
    history = list(recent_history or [])
    last_execution_payload = dict(last_execution or {})
    replan_reason_list = [str(item) for item in (replan_reasons or [])]
    payload = {
        "user_prompt": session_prompt,
        "runtime_policy": policy.to_dict(),
        "request_kind": "task_step",
        "repair_context": None,
        "replan_requested": replan_requested,
        "replan_reasons": replan_reason_list,
        "strong_visual_grounding": strong_visual_grounding,
        "reasoning_enabled": reasoning_enabled,
        "observation_text": observation_text,
        "last_execution": last_execution_payload or None,
        "stderr_tail": last_execution_payload.get("stderr_tail"),
        "recent_history": history,
        "output_requirement": "Return executable Python only.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_prompt_bundle(
    session_prompt: str,
    policy: RuntimePolicy,
    observation_text: str | None = None,
    recent_history: Iterable[str] | None = None,
    last_execution: dict | None = None,
    replan_requested: bool = False,
    replan_reasons: Iterable[str] | None = None,
    strong_visual_grounding: bool = False,
    reasoning_enabled: bool = False,
) -> PromptBundle:
    history = list(recent_history or [])
    last_execution_payload = dict(last_execution or {})
    replan_reason_list = [str(item) for item in (replan_reasons or [])]
    system_prompt = RAW_PYTHON_SYSTEM_PROMPT
    if reasoning_enabled:
        system_prompt = system_prompt + "\n" + REASONING_ENABLED_APPEND.strip() + "\n"
    if strong_visual_grounding:
        system_prompt = system_prompt + "\n" + STRONG_VISUAL_GROUNDING_APPEND.strip() + "\n"
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=render_user_prompt(
            session_prompt=session_prompt,
            policy=policy,
            observation_text=observation_text,
            recent_history=history,
            last_execution=last_execution_payload,
            replan_requested=replan_requested,
            replan_reasons=replan_reason_list,
            strong_visual_grounding=strong_visual_grounding,
            reasoning_enabled=reasoning_enabled,
        ),
        session_prompt=session_prompt,
        policy=policy.to_dict(),
        reasoning_enabled=reasoning_enabled,
        observation_text=observation_text,
        last_execution=last_execution_payload,
        stderr_tail=last_execution_payload.get("stderr_tail"),
        recent_history=history,
        replan_requested=replan_requested,
        replan_reasons=replan_reason_list,
    )


def render_prompt_bundle_from_step_request(request: StepRequest) -> PromptBundle:
    policy = RuntimePolicy.from_dict(request.policy)
    history = list(request.recent_history)
    bundle = render_prompt_bundle(
        session_prompt=request.user_prompt,
        policy=policy,
        observation_text=request.observation_text,
        recent_history=history,
        last_execution=request.last_execution,
        replan_requested=request.replan_requested,
        replan_reasons=request.replan_reasons,
        strong_visual_grounding=request.strong_visual_grounding,
        reasoning_enabled=request.reasoning_enabled,
    )
    user_payload = json.loads(bundle.user_prompt)
    user_payload["request_kind"] = request.request_kind
    user_payload["repair_context"] = request.repair_context or None
    user_payload["replan_requested"] = request.replan_requested
    user_payload["replan_reasons"] = request.replan_reasons
    user_payload["strong_visual_grounding"] = request.strong_visual_grounding
    user_payload["reasoning_enabled"] = request.reasoning_enabled
    user_payload["last_execution"] = request.last_execution or None
    user_payload["stderr_tail"] = (request.last_execution or {}).get("stderr_tail")
    bundle.user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return bundle
