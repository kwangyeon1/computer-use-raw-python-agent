from __future__ import annotations

import json
from pathlib import Path

from computer_use_raw_python_agent.models import StepRequest
from computer_use_raw_python_agent.service import _execute_code_step, _fallback_browser_search_url, _select_prompt_browser_url


class _FakeExecutorClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, *, python_code: str, run_dir: str, step_id: str, metadata: dict | None = None) -> dict:
        self.calls.append(
            {
                "python_code": python_code,
                "run_dir": run_dir,
                "step_id": step_id,
                "metadata": dict(metadata or {}),
            }
        )
        return {
            "ok": True,
            "record": {
                "step_id": step_id,
                "run_dir": run_dir,
                "return_code": 0,
                "payload_metadata": dict(metadata or {}),
            },
            "stdout_tail": "",
            "stderr_tail": "",
            "error_info": None,
            "observation_text": None,
            "screenshot_path": None,
            "screenshot_base64": None,
            "screenshot_media_type": None,
        }


def test_execute_code_step_records_executed_python_code(tmp_path: Path) -> None:
    executor = _FakeExecutorClient()
    root = tmp_path / "run"
    root.mkdir()

    _execute_code_step(
        executor_client=executor,
        root=root,
        step_id="step-000",
        python_code='print("expanded final code")',
        metadata={"agent_response": {"python_code": 'print("raw model code")'}},
    )

    sent_metadata = executor.calls[0]["metadata"]
    assert sent_metadata["executed_python_code"] == 'print("expanded final code")'

    persisted = json.loads((root / "responses" / "step-000.executor.json").read_text(encoding="utf-8"))
    payload_metadata = persisted["record"]["payload_metadata"]
    assert payload_metadata["executed_python_code"] == 'print("expanded final code")'


def test_execute_code_step_expands_runtime_helpers_before_execution(tmp_path: Path) -> None:
    executor = _FakeExecutorClient()
    root = tmp_path / "run"
    root.mkdir()

    _execute_code_step(
        executor_client=executor,
        root=root,
        step_id="step-001",
        python_code='installer = wait_for_stable_download("KakaoTalk*.exe")\nprint(installer)',
        metadata={"agent_response": {"python_code": 'installer = wait_for_stable_download("KakaoTalk*.exe")\nprint(installer)'}},
    )

    sent_code = executor.calls[0]["python_code"]
    assert "def wait_for_stable_download(" in sent_code
    assert 'installer = wait_for_stable_download("KakaoTalk*.exe")' in sent_code


def test_execute_code_step_expands_open_url_helper_before_execution(tmp_path: Path) -> None:
    executor = _FakeExecutorClient()
    root = tmp_path / "run"
    root.mkdir()

    _execute_code_step(
        executor_client=executor,
        root=root,
        step_id="step-002",
        python_code='open_url_and_wait("https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en", expected_title_tokens=["kakao"])',
        metadata={"agent_response": {"python_code": 'open_url_and_wait("https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en", expected_title_tokens=["kakao"])'}},
    )

    sent_code = executor.calls[0]["python_code"]
    assert "def open_url_and_wait(" in sent_code
    assert 'open_url_and_wait("https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en", expected_title_tokens=["kakao"])' in sent_code


def test_execute_code_step_auto_prefixes_open_url_for_gui_first_download_task(tmp_path: Path) -> None:
    executor = _FakeExecutorClient()
    root = tmp_path / "run"
    root.mkdir()

    request = StepRequest(
        user_prompt=(
            "카카오톡 pc버전 프로그램을 설치해줘\n"
            "Official URL: https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en"
        ),
        execution_style="gui_first",
        observation_text="",
    )

    _execute_code_step(
        executor_client=executor,
        root=root,
        step_id="step-003",
        python_code='print("next gui step")',
        request=request,
        metadata={"agent_response": {"python_code": 'print("next gui step")'}},
    )

    sent_code = executor.calls[0]["python_code"]
    assert "def open_url_and_wait(" in sent_code
    assert 'open_url_and_wait("https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en"' in sent_code
    assert 'print("next gui step")' in sent_code


def test_execute_code_step_auto_prefixes_open_url_even_when_prompt_mentions_visible_ui(tmp_path: Path) -> None:
    executor = _FakeExecutorClient()
    root = tmp_path / "run"
    root.mkdir()

    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk. Prefer continuing from the currently visible browser, "
            "search results, download UI, app window, or installer dialog when that UI is already on screen.\n\n"
            "Use Python on Windows to open the official KakaoTalk service page at "
            "https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en, "
            "find the Windows PC installer link, and download the official KakaoTalk_Setup_*.exe."
        ),
        execution_style="gui_first",
        observation_text="",
    )

    _execute_code_step(
        executor_client=executor,
        root=root,
        step_id="step-004",
        python_code='print("continue download flow")',
        request=request,
        metadata={"agent_response": {"python_code": 'print("continue download flow")'}},
    )

    sent_code = executor.calls[0]["python_code"]
    assert "def open_url_and_wait(" in sent_code
    assert 'open_url_and_wait("https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en"' in sent_code
    assert 'print("continue download flow")' in sent_code


def test_select_prompt_browser_url_prefers_page_over_direct_exe() -> None:
    prompt = (
        "Official page: https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en\n"
        "Direct installer: https://app-pc.kakaocdn.net/talk/win32/KakaoTalk_Setup.exe"
    )
    assert _select_prompt_browser_url(prompt) == "https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en"


def test_fallback_browser_search_url_uses_korean_app_keyword() -> None:
    url = _fallback_browser_search_url("카카오톡 pc버전 프로그램을 설치해줘")
    assert url is not None
    assert url.startswith("https://www.bing.com/search?q=")
    assert "%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%86%A1" in url


def test_execute_code_step_auto_prefixes_search_url_when_no_prompt_url_exists(tmp_path: Path) -> None:
    executor = _FakeExecutorClient()
    root = tmp_path / "run"
    root.mkdir()

    request = StepRequest(
        user_prompt="카카오톡 pc버전 프로그램을 설치해줘",
        execution_style="gui_first",
        observation_text="",
    )

    _execute_code_step(
        executor_client=executor,
        root=root,
        step_id="step-005",
        python_code='print("continue download flow")',
        request=request,
        metadata={"agent_response": {"python_code": 'print("continue download flow")'}},
    )

    sent_code = executor.calls[0]["python_code"]
    assert "def open_url_and_wait(" in sent_code
    assert 'open_url_and_wait("https://www.bing.com/search?q=' in sent_code
    assert 'print("continue download flow")' in sent_code
