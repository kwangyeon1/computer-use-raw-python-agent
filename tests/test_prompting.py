from __future__ import annotations

import json

from computer_use_raw_python_agent.models import RuntimePolicy, StepRequest
from computer_use_raw_python_agent.prompting import render_prompt_bundle, render_prompt_bundle_from_step_request


def test_render_prompt_bundle_uses_gui_first_append() -> None:
    bundle = render_prompt_bundle(
        session_prompt="브라우저로 진행해줘",
        policy=RuntimePolicy.from_dict({}),
        execution_style="gui_first",
    )
    assert "Execution style: gui_first" in bundle.system_prompt
    assert '"execution_style": "gui_first"' in bundle.user_prompt


def test_render_prompt_bundle_uses_python_first_fallback_for_unknown_style() -> None:
    bundle = render_prompt_bundle(
        session_prompt="파이썬으로 진행해줘",
        policy=RuntimePolicy.from_dict({}),
        execution_style="something_else",
    )
    assert "Execution style: python_first" in bundle.system_prompt
    assert '"execution_style": "python_first"' in bundle.user_prompt


def test_render_prompt_bundle_from_step_request_preserves_execution_style() -> None:
    request = StepRequest(
        user_prompt="설치해줘",
        policy={},
        execution_style="gui_first",
        recent_history=["step-000=opened_browser"],
    )
    bundle = render_prompt_bundle_from_step_request(request)
    payload = json.loads(bundle.user_prompt)
    assert bundle.execution_style == "gui_first"
    assert payload["execution_style"] == "gui_first"
