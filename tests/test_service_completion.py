from __future__ import annotations

from computer_use_raw_python_agent.service import (
    _infer_response_done,
    _looks_like_opened_page_only_step,
    _looks_like_reported_failure,
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


def test_reported_failure_detected_from_stdout_or_stderr_even_with_zero_exit_code() -> None:
    execution = {
        "return_code": 0,
        "stdout_tail": "Error: download failed",
        "stderr_tail": "Invoke-WebRequest : WebException",
        "error_info": None,
    }
    assert _looks_like_reported_failure(execution) is True
