from __future__ import annotations

from computer_use_raw_python_agent.service import (
    _dependency_repair_user_prompt,
    _infer_response_done,
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
    _rewrite_user_prompt_for_replan,
    _retry_token_budget,
    _should_omit_screenshot_for_generation,
)


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


def test_download_prompt_with_downloads_destination_is_not_treated_as_existing_installer_launch_task() -> None:
    prompt = (
        "Use Python to open the official installation page, extract the latest Windows installer `.exe` link, "
        "and download the installer to `~/Downloads/targetapp-windows-installer.exe`."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is False
    assert _should_omit_screenshot_for_generation(user_prompt=prompt, last_execution={}) is True


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


def test_duplicate_generation_detected_for_same_script() -> None:
    code = """import subprocess
subprocess.run(["cmd", "/c", "echo", "ok"], check=False)
"""
    assert _looks_like_duplicate_generation(code, code) is True
    assert _looks_like_duplicate_generation(code, 'print("other")') is False


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
