from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable
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
When using keyboard automation, pay attention to the currently focused window and the active input locale / IME (for example Korean vs English) before typing.

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

If web_search_context is present:
- Treat it as read-only external information gathered from web search.
- Use it when selecting URLs, official download pages, troubleshooting steps, or public documentation.
- Prefer official/vendor domains when the results indicate them.
- Do not fabricate search results that are not present in web_search_context.
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

WEB_SEARCH_DECISION_SYSTEM_PROMPT = """You decide whether a read-only web search should be used before the next Python step.

Return strict JSON only. Do not return markdown.
Your response must start with `{` and end with `}`.
Output exactly one compact JSON object, then stop.
Do not restate the task.
Do not describe the screenshot.
Do not emit <think> or any other reasoning text.
The JSON object must have exactly these keys:
{
  "use_web_search": false,
  "query": "",
  "allowed_domains": [],
  "blocked_domains": [],
  "reason": ""
}

Use web search when the next step depends on public external information that may be newer than your knowledge cutoff or should be verified from authoritative public sources.

Typical reasons to search:
- finding the official download page, vendor site, or product homepage
- checking official installation instructions, product documentation, or public troubleshooting docs
- investigating a public error message, public compatibility issue, or current workaround
- finding a current URL, current version, current release, or other time-sensitive public web information
- confirming public facts that are not visible in the current screenshot and are not reliably known from the task alone

Do not use web search when:
- the next step is just local GUI continuation and the current screenshot already provides enough evidence
- the next step only depends on local state, local files, or the latest execution result
- no external public information is needed to decide the next Python step

When deciding, prioritize these questions:
1. Does the next step require current or public web information?
2. Would an official/vendor page or documentation source be safer than relying on memory?
3. Is the needed information outside, or potentially newer than, the model's reliable built-in knowledge?

If use_web_search is false, keep query empty and keep both domain lists empty.
If use_web_search is true, prefer queries that target official/vendor sources or public documentation.
Keep `reason` short and concrete, ideally under 80 characters.
"""


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _compact_last_execution_for_prompt(last_execution: dict | None) -> dict[str, Any]:
    payload = dict(last_execution or {})
    if not payload:
        return {}
    compact: dict[str, Any] = {}
    for key in ("step_id", "return_code", "timed_out", "duration_s"):
        if key in payload:
            compact[key] = payload.get(key)
    if payload.get("stdout_tail"):
        compact["stdout_tail"] = _compact_text(payload.get("stdout_tail"), max_chars=400)
    if payload.get("stderr_tail"):
        compact["stderr_tail"] = _compact_text(payload.get("stderr_tail"), max_chars=1200)
    if payload.get("error_info"):
        compact["error_info"] = payload.get("error_info")
    agent_response = (((payload.get("payload_metadata") or {}).get("agent_response")) or {})
    if agent_response:
        compact["last_agent_response"] = {
            "python_code": _compact_text(agent_response.get("python_code"), max_chars=1200),
            "notes": [str(item) for item in agent_response.get("notes", [])][:8],
        }
    return compact


def render_user_prompt(
    session_prompt: str,
    policy: RuntimePolicy,
    observation_text: str | None = None,
    recent_history: Iterable[str] | None = None,
    last_execution: dict | None = None,
    web_search_context: dict[str, Any] | None = None,
    replan_requested: bool = False,
    replan_reasons: Iterable[str] | None = None,
    strong_visual_grounding: bool = False,
    reasoning_enabled: bool = False,
) -> str:
    history = list(recent_history or [])
    last_execution_payload = _compact_last_execution_for_prompt(last_execution)
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
        "web_search_context": dict(web_search_context or {}) or None,
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
    web_search_context: dict[str, Any] | None = None,
    replan_requested: bool = False,
    replan_reasons: Iterable[str] | None = None,
    strong_visual_grounding: bool = False,
    reasoning_enabled: bool = False,
) -> PromptBundle:
    history = list(recent_history or [])
    last_execution_payload = _compact_last_execution_for_prompt(last_execution)
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
            web_search_context=web_search_context,
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
        web_search_context=dict(web_search_context or {}),
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
        web_search_context=request.web_search_context,
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
    user_payload["web_search_context"] = request.web_search_context or None
    user_payload["last_execution"] = request.last_execution or None
    user_payload["stderr_tail"] = (request.last_execution or {}).get("stderr_tail")
    bundle.user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return bundle


def render_web_search_decision_bundle_from_step_request(
    request: StepRequest,
    *,
    web_search_max_uses: int,
    web_search_uses: int,
    web_search_queries: Iterable[str] | None = None,
) -> PromptBundle:
    policy = RuntimePolicy.from_dict(request.policy)
    current_month_year = datetime.now().astimezone().strftime("%B %Y")
    system_prompt = WEB_SEARCH_DECISION_SYSTEM_PROMPT + f"\nCurrent local month/year: {current_month_year}\n"
    if request.reasoning_enabled:
        system_prompt += "\nKeep all reasoning internal. Never reveal chain-of-thought. Output only the JSON object.\n"
    last_execution_payload = _compact_last_execution_for_prompt(request.last_execution)
    payload = {
        "user_prompt": request.user_prompt,
        "request_kind": "web_search_decision",
        "strong_visual_grounding": request.strong_visual_grounding,
        "observation_text": request.observation_text,
        "last_execution": last_execution_payload or None,
        "stderr_tail": last_execution_payload.get("stderr_tail"),
        "recent_history": list(request.recent_history),
        "replan_requested": request.replan_requested,
        "replan_reasons": list(request.replan_reasons),
        "web_search_state": {
            "enabled": True,
            "uses_used": int(web_search_uses),
            "max_uses": int(web_search_max_uses),
            "uses_remaining": max(0, int(web_search_max_uses) - int(web_search_uses)),
            "recent_queries": list(web_search_queries or []),
        },
        "output_requirement": "Return strict JSON only.",
    }
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        session_prompt=request.user_prompt,
        policy=policy.to_dict(),
        reasoning_enabled=False,
        observation_text=request.observation_text,
        last_execution=last_execution_payload,
        stderr_tail=last_execution_payload.get("stderr_tail"),
        web_search_context={},
        recent_history=list(request.recent_history),
        replan_requested=request.replan_requested,
        replan_reasons=list(request.replan_reasons),
    )
