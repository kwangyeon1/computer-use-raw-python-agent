from __future__ import annotations

from computer_use_raw_python_agent.service import (
    _dependency_repair_user_prompt,
    _expand_runtime_helpers,
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
    _prompt_keyword_candidates,
    _rewrite_user_prompt_for_replan,
    _retry_token_budget,
    _should_use_framework_official_download_recovery,
    _should_omit_screenshot_for_generation,
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
