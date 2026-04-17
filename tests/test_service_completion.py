from __future__ import annotations

from computer_use_raw_python_agent.service import (
    _dependency_repair_user_prompt,
    _expand_runtime_helpers,
    _fallback_browser_search_url,
    _generated_code_ignores_prompt_urls,
    _has_visible_gui_continuation_cues,
    _infer_response_done,
    _looks_like_gui_first_silent_install_shortcut,
    _looks_like_gui_first_visible_ui_bypass,
    _looks_like_duplicate_generation,
    _looks_like_missing_install_progress_generation,
    _is_compilable_python_code,
    _looks_like_direct_download_url_404,
    _looks_like_existing_installer_launch_task,
    _looks_like_incomplete_install_attempt,
    _looks_like_installer_launched_but_app_not_found,
    _looks_like_installer_timeout,
    _looks_like_installer_url_discovery_failure,
    _looks_like_non_executing_task_script,
    _looks_like_opened_page_only_step,
    _looks_like_reported_failure,
    _normalize_missing_module_install_name,
    _prepare_python_code_for_execution,
    _prompt_keyword_candidates,
    _rewrite_user_prompt_for_replan,
    _retry_token_budget,
    _synthesized_visible_ui_click_recovery_code,
    _should_use_framework_official_download_recovery,
    _should_omit_screenshot_for_generation,
    _visible_flow_extra_targets,
)
from computer_use_raw_python_agent.models import StepRequest


def test_task_complete_marker_requires_confirmation_script() -> None:
    code = """# task_complete
import webbrowser
webbrowser.open("https://example.com/download")
"""
    assert _infer_response_done(python_code=code, raw_text=code) is False


def test_task_complete_marker_accepts_confirmation_script() -> None:
    code = """# task_complete
print("KakaoTalk installation verified")
"""
    assert _infer_response_done(python_code=code, raw_text=code) is True


def test_opened_page_only_step_detected_for_download_like_code() -> None:
    code = """import webbrowser
webbrowser.open("https://pc.kakao.com/download")
"""
    assert _looks_like_opened_page_only_step(code) is True


def test_opened_page_only_step_detected_for_selenium_navigation_without_download() -> None:
    code = """from selenium import webdriver
driver = webdriver.Chrome()
driver.get("https://example.com/download")
"""
    assert _looks_like_opened_page_only_step(code) is True


def test_opened_page_only_step_not_detected_when_file_verification_present() -> None:
    code = """from pathlib import Path
import webbrowser
webbrowser.open("https://example.com/file.exe")
print(Path.home().exists())
"""
    assert _looks_like_opened_page_only_step(code) is False


def test_opened_page_only_step_not_detected_when_clicking_visible_download_control() -> None:
    code = """from selenium import webdriver
driver = webdriver.Chrome()
driver.get("https://example.com/download")
driver.find_element("xpath", "//a").click()
"""
    assert _looks_like_opened_page_only_step(code) is False


def test_visible_gui_continuation_cues_ignore_prompt_text_only() -> None:
    request = StepRequest(
        user_prompt=(
            "Prefer continuing from the currently visible browser, search results, download UI, "
            "app window, or installer dialog when that UI is already on screen."
        ),
        execution_style="gui_first",
        observation_text="",
    )
    assert _has_visible_gui_continuation_cues(request) is False


def test_visible_gui_continuation_cues_use_runtime_state() -> None:
    request = StepRequest(
        user_prompt="카카오톡 pc버전 프로그램을 설치해줘",
        execution_style="gui_first",
        observation_text="현재 화면에 브라우저와 다운로드 버튼이 보입니다.",
    )
    assert _has_visible_gui_continuation_cues(request) is True


def test_visible_gui_continuation_cues_detected_after_browser_open_when_screenshot_present() -> None:
    request = StepRequest(
        user_prompt="카카오톡 pc버전 프로그램을 설치해줘",
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://pc.kakao.com/talk", expected_title_tokens=["kakao"])',
            }
        },
    )
    assert _has_visible_gui_continuation_cues(request) is True


def test_visible_gui_continuation_cues_detected_from_ocr_observation_text() -> None:
    request = StepRequest(
        user_prompt="어떤 프로그램을 설치해줘",
        execution_style="gui_first",
        observation_text="OCR visible text: 공식 다운로드 페이지 | 다운로드 버튼 | Windows",
    )
    assert _has_visible_gui_continuation_cues(request) is True


def test_visible_gui_continuation_cues_ignore_generic_ocr_text_without_browser_markers() -> None:
    request = StepRequest(
        user_prompt="어떤 프로그램을 설치해줘",
        execution_style="gui_first",
        observation_text="OCR visible text: ./venv/bin/training-generator | --execution-style gui_first | response.json",
    )
    assert _has_visible_gui_continuation_cues(request) is False


def test_download_prompt_with_downloads_destination_is_not_treated_as_existing_installer_launch_task() -> None:
    prompt = (
        "Use Python to open the official installation page, extract the latest Windows installer `.exe` link, "
        "and download the installer to `~/Downloads/targetapp-windows-installer.exe`."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is False
    assert _should_omit_screenshot_for_generation(
        user_prompt=prompt,
        execution_style="python_first",
        last_execution={},
    ) is True


def test_gui_first_download_retry_keeps_screenshot_for_generation() -> None:
    prompt = (
        "Use Python to continue from the visible browser and download the installer to Downloads."
    )
    assert _should_omit_screenshot_for_generation(
        user_prompt=prompt,
        execution_style="gui_first",
        last_execution={"stdout_tail": "download retry", "stderr_tail": ""},
    ) is False


def test_prepare_python_code_for_execution_auto_clicks_download_control_for_gui_first_visible_ui() -> None:
    request = StepRequest(
        user_prompt="Use Python to open the official vendor page and download the Windows installer `.exe`.",
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        observation_text="OCR visible text with download/install cues: Download | Windows | Setup",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://vendor.example/download", expected_title_tokens=["download"])',
            }
        },
    )
    prepared = _prepare_python_code_for_execution(request, 'print("continue")')
    assert "advance_visible_download_flow(" in prepared
    assert "wait_for_stable_download(" in prepared
    assert "visible download automation incomplete:" in prepared


def test_prepare_python_code_for_execution_replaces_gui_first_http_bypass_with_browser_click_flow() -> None:
    request = StepRequest(
        user_prompt="카카오톡 pc버전 프로그램을 설치해줘",
        execution_style="gui_first",
    )
    prepared = _prepare_python_code_for_execution(
        request,
        """import urllib.request
html = urllib.request.urlopen("https://example.com").read().decode("utf-8")
print(html[:100])
""",
    )
    assert "open_url_and_wait(" in prepared
    assert "advance_visible_download_flow(" in prepared
    assert "wait_for_stable_download(" in prepared
    assert "urllib.request.urlopen" not in prepared


def test_prepare_python_code_for_execution_does_not_treat_file_write_as_gui_progress() -> None:
    request = StepRequest(
        user_prompt="targetapp 설치 파일을 받아줘",
        execution_style="gui_first",
    )
    prepared = _prepare_python_code_for_execution(
        request,
        """from pathlib import Path
import urllib.request
dest = Path.home() / "Downloads" / "targetapp.exe"
with urllib.request.urlopen("https://example.com/download", timeout=30) as response, open(dest, "wb") as fh:
    fh.write(response.read())
print(dest)
""",
    )
    assert "advance_visible_download_flow(" in prepared
    assert "urllib.request.urlopen" not in prepared


def test_prepare_python_code_for_execution_auto_clicks_search_result_for_visible_search_results() -> None:
    request = StepRequest(
        user_prompt="Use Python to open the official vendor page and download the Windows installer `.exe`.",
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        observation_text="search results | official download | windows",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.bing.com/search?q=targetapp+official+windows+download", expected_title_tokens=["targetapp"])',
            }
        },
    )
    prepared = _prepare_python_code_for_execution(request, 'print("continue")')
    assert "advance_visible_download_flow(" in prepared
    assert "search_first=True" in prepared


def test_fallback_browser_search_url_uses_korean_download_terms_for_hangul_tasks() -> None:
    url = _fallback_browser_search_url("카카오톡 pc버전 프로그램을 설치해줘")
    assert url is not None
    assert "%EA%B3%B5%EC%8B%9D" in url
    assert "%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C" in url


def test_expand_runtime_helpers_includes_browser_region_heuristic_click() -> None:
    expanded = _expand_runtime_helpers("click_search_result_like_target(extra_targets=['targetapp'])")
    assert "def _heuristic_browser_click(" in expanded
    assert "browser_search_result_region" in expanded
    assert "screen-browser-region-fallback" in expanded
    assert "crop_region=browser_region" in expanded


def test_prompt_url_violation_allows_gui_first_search_discovery_flow() -> None:
    user_prompt = (
        "Use Python to continue from the visible browser first and download the Windows installer. "
        "Official URL: https://pc.example.com/download"
    )
    search_url = "https://www.bing.com/search?q=targetapp%20official%20windows%20download"
    code = f"""open_url_and_wait({search_url!r}, expected_title_tokens=["targetapp"])
flow = advance_visible_download_flow(extra_targets=["targetapp"], search_first=True, timeout_s=18.0)
print(flow)
"""
    assert (
        _generated_code_ignores_prompt_urls(
            user_prompt=user_prompt,
            python_code=code,
            active_replan_reasons=[],
        )
        is False
    )


def test_prompt_url_violation_still_detects_unrelated_url() -> None:
    user_prompt = (
        "Use Python to continue from the visible browser first and download the Windows installer. "
        "Official URL: https://pc.example.com/download"
    )
    code = """open_url_and_wait("https://malicious.example/download", expected_title_tokens=["bad"])
print("continue")
"""
    assert (
        _generated_code_ignores_prompt_urls(
            user_prompt=user_prompt,
            python_code=code,
            active_replan_reasons=[],
        )
        is True
    )


def test_visible_ui_click_recovery_uses_search_result_helper_for_search_results() -> None:
    request = StepRequest(
        user_prompt="targetapp를 설치해줘",
        execution_style="gui_first",
        observation_text="search results | official download | windows",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.bing.com/search?q=targetapp+official+windows+download", expected_title_tokens=["targetapp"])',
            }
        },
    )
    recovery_code = _synthesized_visible_ui_click_recovery_code(request)
    assert "click_search_result_like_target(" in recovery_code


def test_reported_failure_detected_from_stdout_or_stderr_even_with_zero_exit_code() -> None:
    execution = {
        "return_code": 0,
        "stdout_tail": "Error: download failed",
        "stderr_tail": "Invoke-WebRequest : WebException",
        "error_info": None,
    }
    assert _looks_like_reported_failure(execution) is True


def test_compilable_python_code_detects_truncated_script() -> None:
    assert _is_compilable_python_code('print("hello")') is True
    assert _is_compilable_python_code('print("unterminated') is False


def test_non_executing_task_script_detects_function_only_skeleton() -> None:
    code = """import os

def download():
    print("prepare only")
"""
    assert _looks_like_non_executing_task_script(code) is True


def test_non_executing_task_script_allows_top_level_execution() -> None:
    code = """import subprocess

subprocess.run(["cmd", "/c", "echo", "ok"], check=False)
"""
    assert _looks_like_non_executing_task_script(code) is False


def test_non_executing_task_script_detects_import_only_script() -> None:
    code = """import os
import re
import sys
"""
    assert _looks_like_non_executing_task_script(code) is True


def test_direct_download_url_404_detected() -> None:
    execution = {
        "stdout_tail": "Download failed: 404 Client Error: Not Found for url: https://vendor.example/file.exe",
        "stderr_tail": "",
    }
    code = """import requests
requests.get("https://vendor.example/file.exe", timeout=30)
    """
    assert _looks_like_direct_download_url_404(execution, code) is True


def test_installer_url_discovery_failure_detected() -> None:
    execution = {
        "stdout_tail": "Fetching official installer URL...\nFailed to find installer URL.\n",
        "stderr_tail": "",
    }
    code = """import urllib.request, re
html = urllib.request.urlopen("https://vendor.example/download").read().decode("utf-8")
matches = re.findall(r'href="[^"]*\\.exe"', html)
print(matches)
    """
    assert _looks_like_installer_url_discovery_failure(execution, code) is True


def test_existing_installer_launch_task_detected() -> None:
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is True


def test_installer_timeout_detected() -> None:
    execution = {
        "timed_out": True,
        "error_info": {"kind": "timeout"},
    }
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    code = """import subprocess
subprocess.run([str(installer), "/VERYSILENT", "/SP-", "/NORESTART"], check=True, timeout=300)
"""
    assert _looks_like_installer_timeout(execution, code, prompt) is True


def test_installer_launched_but_app_not_found_detected() -> None:
    execution = {
        "stdout_tail": "Found installer: C:\\Users\\user\\Downloads\\Dbeaver\\dbeaver-le-latest-x86_64-setup.exe\n",
        "stderr_tail": "Installation failed or app not found\n",
    }
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    code = """import subprocess
subprocess.Popen([str(installer), "/SILENT"], creationflags=subprocess.CREATE_NO_WINDOW)
raise SystemExit("Installation failed or app not found")
"""
    assert _looks_like_installer_launched_but_app_not_found(execution, code, prompt) is True


def test_incomplete_install_attempt_detected_from_success_exit_without_artifacts() -> None:
    execution = {
        "return_code": 0,
        "stdout_tail": "Found installer: C:\\Users\\user\\Downloads\\Dbeaver\\dbeaver-le-latest-x86_64-setup.exe\n",
        "stderr_tail": "",
    }
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    code = """import subprocess
subprocess.Popen([str(installer), "/SILENT"], creationflags=subprocess.CREATE_NO_WINDOW)
"""
    assert _looks_like_incomplete_install_attempt(execution, code, prompt) is True


def test_incomplete_install_attempt_detected_when_script_only_scans_paths() -> None:
    execution = {
        "return_code": 0,
        "stdout_tail": "Found installer: C:\\Users\\user\\Downloads\\Dbeaver\\dbeaver-le-latest-x86_64-setup.exe\n",
        "stderr_tail": "",
    }
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    code = """from pathlib import Path
downloads = Path.home() / "Downloads" / "Dbeaver"
installer = max(downloads.glob("*.exe"))
print(f"Found installer: {installer}")
"""
    assert _looks_like_incomplete_install_attempt(execution, code, prompt) is True


def test_missing_install_progress_generation_detected_for_search_only_install_script() -> None:
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    code = """from pathlib import Path
downloads = Path.home() / "Downloads" / "Dbeaver"
installer = max(downloads.glob("*.exe"))
print(f"Found installer: {installer}")
"""
    assert _looks_like_missing_install_progress_generation(code, prompt) is True


def test_missing_install_progress_generation_detected_when_script_only_launches_final_app() -> None:
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    code = """from pathlib import Path
import os, subprocess
downloads = Path.home() / "Downloads" / "Dbeaver"
installer = max(downloads.glob("*.exe"))
exe = Path(os.environ["ProgramFiles"]) / "DBeaver" / "dbeaver.exe"
if exe.exists():
    subprocess.Popen([str(exe)])
"""
    assert _looks_like_missing_install_progress_generation(code, prompt) is True


def test_prompt_keyword_candidates_drop_generic_gui_first_words() -> None:
    text = (
        "Return executable Python only for this chunk. Prefer continuing from the currently visible "
        "browser search results app window and download button. Open https://pc.kakao.com/talk/notices/en?agent=win32 "
        "and download KakaoTalkSetup.exe."
    )
    keywords = _prompt_keyword_candidates(text)
    assert "kakao" in keywords or "kakaotalksetup" in keywords or "kakaotalk" in keywords
    assert "visible" not in keywords
    assert "browser" not in keywords
    assert "results" not in keywords
    assert "app" not in keywords


def test_prompt_keyword_candidates_include_url_host_tokens_for_korean_task() -> None:
    text = (
        'source_task": "카카오톡 pc버전 프로그램을 설치해줘"\n'
        "Official URL: https://pc.kakao.com/download"
    )
    keywords = _prompt_keyword_candidates(text)
    assert "카카오톡" in keywords
    assert "kakao" in keywords


def test_prompt_keyword_candidates_drop_replan_words() -> None:
    text = (
        "REPLAN OVERRIDE FOR THIS STEP: Previous attempt failed. "
        "Use the current screenshot as the primary source of truth for the next action."
    )
    keywords = _prompt_keyword_candidates(text)
    assert "replan" not in keywords
    assert "override" not in keywords
    assert "previous" not in keywords
    assert "attempt" not in keywords


def test_visible_flow_extra_targets_use_observation_text_keywords_when_replan_prompt_is_generic() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "The previous attempt already opened the relevant browser page."
        ),
        execution_style="gui_first",
        observation_text="OCR visible text with download/install cues: 카카오톡 official windows download | 다운로드",
    )
    keywords = _visible_flow_extra_targets(request, limit=4)
    assert "카카오톡" in keywords
    assert "replan" not in keywords


def test_visible_flow_extra_targets_prefer_task_and_prompt_url_over_noisy_ocr() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "Use executable Python on the Windows machine to obtain the official Windows installer `.exe` "
            "for the target app from this task: 카카오톡 pc버전 프로그램을 설치해줘.\n"
            "Use these exact official page URLs first before any search engine result or inferred domain:\n"
            "- https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=ko\n"
        ),
        execution_style="gui_first",
        observation_text="OCR visible text with download/install cues: -executi on-style | 다운로드 0",
    )
    keywords = _visible_flow_extra_targets(request, limit=4)
    assert "카카오톡" in keywords
    assert "kakaocorp" in keywords
    assert "executi" not in keywords
    assert "on-style" not in keywords


def test_visible_flow_extra_targets_use_last_execution_prompt_url_on_retry() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "The current screenshot clearly shows the browser path is impossible.\n"
        ),
        execution_style="gui_first",
        observation_text="OCR visible text with download/install cues: 다. 해당 테스트만 다시 확인합니다. | 다운로드 0",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=ko", expected_title_tokens=["카카오톡", "kakaocorp"])',
            }
        },
    )
    keywords = _visible_flow_extra_targets(request, limit=4)
    assert "카카오톡" in keywords
    assert "kakaocorp" in keywords
    assert "해당" not in keywords
    assert "테스트만" not in keywords


def test_prompt_keyword_candidates_ignore_percent_encoded_fragments() -> None:
    text = (
        "Open https://pc.example.com/talk/notices/en%3Fagent%3Dwin32 and continue the official flow."
    )
    keywords = _prompt_keyword_candidates(text)
    assert "3fagent" not in keywords
    assert "3dwin32" not in keywords


def test_visible_gui_continuation_cues_detected_for_gui_first_request() -> None:
    request = StepRequest(
        user_prompt=(
            "If the current screenshot shows a visible browser page or download button, "
            "continue from the visible UI first."
        ),
        execution_style="gui_first",
        observation_text="Visible browser page with a download button is open.",
        last_execution={"stdout_tail": "", "stderr_tail": ""},
        replan_requested=True,
        replan_reasons=["execution_error"],
    )
    assert _has_visible_gui_continuation_cues(request) is True


def test_visible_gui_continuation_cues_detected_for_korean_gui_first_request() -> None:
    request = StepRequest(
        user_prompt="현재 스크린샷에 브라우저와 다운로드 버튼이 보이면 그 보이는 UI를 먼저 이어서 사용하세요.",
        execution_style="gui_first",
        observation_text="공식 다운로드 페이지와 다운로드 진행 UI가 보이는 상태.",
        last_execution={"stdout_tail": "", "stderr_tail": ""},
    )
    assert _has_visible_gui_continuation_cues(request) is True


def test_framework_official_download_recovery_disabled_for_gui_first_visible_ui() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python to continue from the current screenshot and visible browser page. "
            "If a download button is already visible, press it first. "
            "Official URL: https://pc.kakao.com/talk/notices/en?agent=win32"
        ),
        execution_style="gui_first",
        observation_text="Visible browser page with KakaoTalk download button.",
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=2,
    )
    assert _should_use_framework_official_download_recovery(request) is False


def test_framework_official_download_recovery_disabled_for_gui_first_even_without_visible_ui_cues() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python to continue downloading the Windows installer from the official page. "
            "Official URL: https://pc.kakao.com/talk/notices/en?agent=win32"
        ),
        execution_style="gui_first",
        observation_text=None,
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=2,
    )
    assert _should_use_framework_official_download_recovery(request) is False


def test_gui_first_visible_ui_bypass_detected_for_network_scraping_code() -> None:
    request = StepRequest(
        user_prompt=(
            "Continue from the visible browser page and click the download button if it is already on screen. "
            "Official URL: https://www.kakaocorp.com/page/service/service/KakaoTalk"
        ),
        execution_style="gui_first",
        observation_text="Visible browser page with KakaoTalk download control.",
    )
    code = """import urllib.request, re
html = urllib.request.urlopen("https://www.kakaocorp.com/page/service/service/KakaoTalk").read().decode("utf-8")
matches = re.findall(r'href="[^"]+\\.exe"', html)
print(matches)
"""
    assert _looks_like_gui_first_visible_ui_bypass(request, code) is True


def test_gui_first_visible_ui_bypass_not_detected_for_pyautogui_flow() -> None:
    request = StepRequest(
        user_prompt="Continue from the visible browser download button.",
        execution_style="gui_first",
        observation_text="Visible browser page with a download button.",
    )
    code = """import pyautogui, time
pyautogui.click(1200, 340)
time.sleep(1)
pyautogui.press("enter")
"""
    assert _looks_like_gui_first_visible_ui_bypass(request, code) is False


def test_gui_first_silent_install_shortcut_detected_for_existing_installer_task() -> None:
    request = StepRequest(
        user_prompt=(
            "Locate the downloaded installer `.exe` in Downloads and complete the installer wizard. "
            "If installer UI is visible, continue from that visible UI first."
        ),
        execution_style="gui_first",
        observation_text="Installer wizard may appear on screen.",
    )
    code = """from pathlib import Path
import subprocess, time
installer = max((Path.home() / "Downloads").glob("*.exe"))
subprocess.Popen([str(installer), "/SILENT", "/NORESTART", "/SP-"])
time.sleep(5)
"""
    assert _looks_like_gui_first_silent_install_shortcut(request, code) is True


def test_gui_first_silent_install_shortcut_not_detected_when_gui_progress_exists() -> None:
    request = StepRequest(
        user_prompt=(
            "Locate the downloaded installer `.exe` in Downloads and complete the installer wizard. "
            "If installer UI is visible, continue from that visible UI first."
        ),
        execution_style="gui_first",
        observation_text="Installer wizard visible.",
    )
    code = """import pyautogui, time
    for _ in range(4):
    pyautogui.press("enter")
    time.sleep(1)
"""
    assert _looks_like_gui_first_silent_install_shortcut(request, code) is False


def test_expand_runtime_helpers_injects_wait_for_stable_download_definition() -> None:
    code = """from pathlib import Path
downloads = Path.home() / "Downloads"
installer = wait_for_stable_download("KakaoTalk*.exe", min_bytes=1_000_000)
print(installer)
"""
    expanded = _expand_runtime_helpers(code)
    assert "def wait_for_stable_download(" in expanded
    assert 'installer = wait_for_stable_download("KakaoTalk*.exe", min_bytes=1_000_000)' in expanded
    assert "candidate.stat().st_mtime" in expanded
    assert "min_quiet_time_s" in expanded


def test_expand_runtime_helpers_injects_open_url_and_wait_definition() -> None:
    code = """opened = open_url_and_wait(
    "https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en",
    expected_title_tokens=["kakao", "kakaotalk"],
)
print(opened)
"""
    expanded = _expand_runtime_helpers(code)
    assert "def open_url_and_wait(" in expanded
    assert 'os.startfile(target_url)' in expanded
    assert '["cmd", "/c", "start", "", target_url]' in expanded
    assert '"--new-tab", target_url' in expanded
    assert 'expected_title_tokens=["kakao", "kakaotalk"]' in expanded


def test_expand_runtime_helpers_injects_recursive_download_click_helpers() -> None:
    code = """result = click_download_like_target(timeout_s=8.0)
print(result)
"""
    expanded = _expand_runtime_helpers(code)
    assert "def click_download_like_target(" in expanded
    assert "def click_text_targets(" in expanded
    assert "def ocr_screen_text_regions(" in expanded
    assert '"download",' in expanded
    assert 'click_horizontal_bias="matched_token_right"' in expanded
    assert "browser_download_cta_region" in expanded


def test_expand_runtime_helpers_injects_advance_visible_download_flow_definition() -> None:
    code = """result = advance_visible_download_flow(extra_targets=["targetapp"], search_first=True, timeout_s=18.0)
print(result)
"""
    expanded = _expand_runtime_helpers(code)
    assert "def advance_visible_download_flow(" in expanded
    assert "def click_search_result_like_target(" in expanded
    assert "def click_download_like_target(" in expanded
    assert 'search_first=True' in expanded


def test_framework_official_download_recovery_reuses_only_matching_existing_installer_keywords() -> None:
    code = _synthesized_official_download_recovery_code_for_test(
        user_prompt=(
            "Return executable Python only for this chunk. "
            "Open https://pc.kakao.com/talk/notices/en?agent=win32 and download KakaoTalkSetup.exe."
        )
    )
    assert 'if not KEYWORDS:' in code
    assert 'if not any(keyword in lowered for keyword in KEYWORDS):' in code


def test_duplicate_generation_detected_for_same_script() -> None:
    code = """import subprocess
subprocess.run(["cmd", "/c", "echo", "ok"], check=False)
"""
    assert _looks_like_duplicate_generation(code, code) is True
    assert _looks_like_duplicate_generation(code, 'print("other")') is False


def _synthesized_official_download_recovery_code_for_test(*, user_prompt: str) -> str:
    from computer_use_raw_python_agent.service import _synthesized_official_download_recovery_code

    return _synthesized_official_download_recovery_code(user_prompt=user_prompt)


def test_replan_prompt_rewrite_for_installer_app_not_found() -> None:
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["repeated_code_execution", "installer_app_not_found"],
        last_execution={
            "stdout_tail": "Found installer: C:\\Users\\user\\Downloads\\Dbeaver\\dbeaver.exe",
            "stderr_tail": "Installation failed or app not found",
        },
    )
    assert rewritten.startswith("REPLAN OVERRIDE FOR THIS STEP:")
    assert "Do not retry `/VERYSILENT` or `/SILENT` first on this step." in rewritten
    assert "use Python GUI automation to advance it" in rewritten
    assert "Produce a materially different script from the previous attempt." in rewritten
    assert "do not add download, URL discovery, or HTML parsing logic" in rewritten
    assert "Launching only the final app executable is not enough." in rewritten
    assert "Keep the script compact." in rewritten
    assert "End this step only when the installed app process is running." in rewritten
    assert prompt not in rewritten


def test_replan_prompt_rewrite_for_gui_first_download_after_browser_open() -> None:
    prompt = (
        "Use Python on Windows to open Kakao's official PC page at https://pc.kakao.com/talk "
        "and download the Windows KakaoTalk installer only as a `.exe`."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["execution_error", "download_url_404"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en", expected_title_tokens=["kakao"])',
            },
            "stdout_tail": "Downloading from https://pc.kakao.com/talk...\nFailed to fetch page: HTTP Error 404",
        },
    )
    assert rewritten.startswith("REPLAN OVERRIDE FOR THIS STEP:")
    assert "Treat the current screenshot as the primary source of truth" in rewritten
    assert "Continue from the visible browser/download UI with Python GUI automation" in rewritten
    assert "Do not use urllib, requests, regex-based HTML scraping" in rewritten
    assert "click_download_like_target() or click_text_targets([...])" in rewritten
    assert "click the visible download control" in rewritten


def test_replan_prompt_rewrite_for_truncated_gui_repetition_failure_adds_loop_hint() -> None:
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["execution_error"],
        last_execution={
            "stderr_tail": "NameError: name 'py' is not defined",
        },
    )
    assert "Previous attempt appears to have been cut off mid-script" in rewritten
    assert "bounded loop like for _ in range(8)" in rewritten


def test_replan_prompt_rewrite_for_optional_gui_module_failure_keeps_install_context() -> None:
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["execution_error"],
        last_execution={
            "error_info": {
                "kind": "missing_python_module",
                "module_name": "win32gui",
            },
            "stderr_tail": "ModuleNotFoundError: No module named 'win32gui'",
        },
    )
    assert rewritten.startswith("REPLAN OVERRIDE FOR THIS STEP:")
    assert "Do not retry `/VERYSILENT` or `/SILENT` first on this step." in rewritten
    assert "Do not directly import win32gui, win32con, win32api, pythoncom, pywinauto" in rewritten
    assert "Prefer pyautogui, pygetwindow, psutil, and the standard library." in rewritten
    assert "Keep the script short and avoid deeply nested repeated retry loops." in rewritten


def test_replan_prompt_rewrite_for_install_path_scan_failure_avoids_broad_rglob() -> None:
    prompt = (
        "Use executable Python to locate the downloaded installer `.exe` in Downloads\\Dbeaver, "
        "run it, finish the installation, and launch the installed app once. "
        "End this chunk only after the installed app process is running."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["execution_error"],
        last_execution={
            "stderr_tail": (
                "FileNotFoundError: [WinError 3] path not found: "
                "C:\\Users\\user\\AppData\\Local\\Microsoft\\Windows\\CloudStore\\broken"
            ),
        },
    )
    assert "Do not retry `/VERYSILENT` or `/SILENT` first on this step." in rewritten
    assert "Do not rglob the whole of LOCALAPPDATA or Program Files." in rewritten
    assert "LOCALAPPDATA\\\\Programs\\\\DBeaver" in rewritten


def test_missing_module_install_name_override_for_pywin32() -> None:
    assert _normalize_missing_module_install_name("win32gui", "win32gui") == "pywin32"
    assert _normalize_missing_module_install_name("win32con", "win32con") == "pywin32"
    assert _normalize_missing_module_install_name("pyautogui", "pyautogui") == "pyautogui"


def test_dependency_repair_user_prompt_stays_in_repair_mode() -> None:
    prompt = _dependency_repair_user_prompt(
        module_name="win32gui",
        install_name="pywin32",
        strategy="pip_install",
    )
    assert "Repair the reported missing Python dependency only." in prompt
    assert "Preferred install target: pywin32" in prompt
    assert "Do not relaunch the installer or repeat the original task script here." in prompt


def test_dependency_repair_user_prompt_handles_pywin32_distribution_name() -> None:
    prompt = _dependency_repair_user_prompt(
        module_name="pywin32",
        install_name="pywin32",
        strategy="pip_install",
    )
    assert "`pywin32` is a distribution name" in prompt
    assert "Do not write `import pywin32`" in prompt


def test_retry_token_budget_increases_with_cap() -> None:
    assert _retry_token_budget(256) == 384
    assert _retry_token_budget(512) == 512
    assert _retry_token_budget(800) == 800
