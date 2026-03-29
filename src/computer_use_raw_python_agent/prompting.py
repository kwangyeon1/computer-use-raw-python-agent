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
"""


def render_user_prompt(
    session_prompt: str,
    policy: RuntimePolicy,
    observation_text: str | None = None,
    recent_history: Iterable[str] | None = None,
) -> str:
    history = list(recent_history or [])
    payload = {
        "user_prompt": session_prompt,
        "runtime_policy": policy.to_dict(),
        "observation_text": observation_text,
        "recent_history": history,
        "output_requirement": "Return executable Python only.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_prompt_bundle(
    session_prompt: str,
    policy: RuntimePolicy,
    observation_text: str | None = None,
    recent_history: Iterable[str] | None = None,
) -> PromptBundle:
    history = list(recent_history or [])
    return PromptBundle(
        system_prompt=RAW_PYTHON_SYSTEM_PROMPT,
        user_prompt=render_user_prompt(
            session_prompt=session_prompt,
            policy=policy,
            observation_text=observation_text,
            recent_history=history,
        ),
        session_prompt=session_prompt,
        policy=policy.to_dict(),
        observation_text=observation_text,
        recent_history=history,
    )


def render_prompt_bundle_from_step_request(request: StepRequest) -> PromptBundle:
    policy = RuntimePolicy.from_dict(request.policy)
    history = list(request.recent_history)
    if request.last_execution:
        history.append(
            "last_execution="
            + json.dumps(
                request.last_execution,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return render_prompt_bundle(
        session_prompt=request.user_prompt,
        policy=policy,
        observation_text=request.observation_text,
        recent_history=history,
    )
