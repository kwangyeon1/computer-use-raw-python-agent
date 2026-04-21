from __future__ import annotations

from computer_use_raw_python_agent.service import (
    _dependency_repair_user_prompt,
    _expand_runtime_helpers,
    _extract_click_points_from_python,
    _extract_prompt_download_glob,
    _extract_prompt_install_marker_path,
    _extract_prompt_launch_marker_path,
    _fallback_browser_search_url,
    _fallback_browser_search_url_for_request,
    _generated_code_ignores_prompt_urls,
    _has_visible_gui_continuation_cues,
    _history_for_invalid_python_retry_with_prompt,
    _infer_response_done,
    _looks_like_guessed_artifact_url_generation,
    _looks_like_gui_first_download_chunk_network_bypass,
    _looks_like_gui_first_download_chunk_install_mix,
    _looks_like_gui_first_silent_install_shortcut,
    _looks_like_gui_first_visible_ui_bypass,
    _looks_like_duplicate_generation,
    _looks_like_missing_install_progress_generation,
    _is_compilable_python_code,
    _looks_like_direct_download_url_404,
    _looks_like_existing_installer_launch_task,
    _looks_like_launch_app_chunk_task,
    _looks_like_incomplete_install_attempt,
    _looks_like_installer_launched_but_app_not_found,
    _looks_like_installer_timeout,
    _looks_like_installer_url_discovery_failure,
    _looks_like_visible_installer_observation,
    _looks_like_non_executing_task_script,
    _looks_like_opened_page_only_step,
    _looks_like_reported_failure,
    _looks_like_download_chunk_completed,
    _normalize_missing_module_install_name,
    _prepare_python_code_for_execution,
    _prompt_keyword_candidates,
    _rewrite_user_prompt_for_replan,
    _retry_token_budget,
    _select_prompt_browser_url,
    _sanitize_observation_text_for_model,
    _step_token_budget,
    _synthesized_framework_visible_download_recovery_code,
    _synthesized_visible_download_completion_code,
    _synthesized_visible_installer_recovery_code,
    _synthesized_visible_launch_recovery_code,
    _synthesized_visible_ui_click_recovery_code,
    _should_soft_allow_gui_first_download_bypass_for_auto_open,
    _uses_deprecated_ocr_helper,
    _should_use_framework_visible_launch_recovery,
    _should_use_framework_visible_installer_recovery,
    _should_use_framework_visible_download_flow,
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


def test_prepare_python_code_for_execution_does_not_auto_click_download_control_for_gui_first_visible_ui() -> None:
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
    assert prepared == 'print("continue")'
    assert "advance_visible_download_flow(" not in prepared
    assert "click_download_like_target(" not in prepared


def test_prepare_python_code_for_execution_replaces_blind_percentage_click_with_visible_download_flow() -> None:
    request = StepRequest(
        user_prompt="Continue from the visible official download page and download `targetapp-setup.exe` into Downloads.",
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        observation_text="Visible browser page with the official targetapp download button and installer file name.",
        replan_requested=True,
        replan_reasons=["execution_error"],
    )
    prepared = _prepare_python_code_for_execution(
        request,
        """import pyautogui, time
screen_w, screen_h = pyautogui.size()
btn_x = int(screen_w * 0.68)
btn_y = int(screen_h * 0.58)
pyautogui.click(btn_x, btn_y)
time.sleep(2)
""",
    )
    assert "advance_visible_download_flow(" in prepared
    assert "wait_for_stable_download(" in prepared
    assert "btn_x = int(screen_w * 0.68)" not in prepared


def test_prepare_python_code_for_execution_replaces_risky_pygetwindow_retry_with_visible_download_flow() -> None:
    request = StepRequest(
        user_prompt="Continue from the visible browser page and download `targetapp-setup.exe` only.",
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        observation_text="Visible browser page on the official vendor site with a download control.",
        replan_requested=True,
        replan_reasons=["execution_error"],
    )
    prepared = _prepare_python_code_for_execution(
        request,
        """import pygetwindow as gw
window = gw.getActiveWindow()
gw = gw.getActiveWindow()
for win in gw.getAllWindows():
    print(win.title)
""",
    )
    assert "advance_visible_download_flow(" in prepared
    assert "gw = gw.getActiveWindow()" not in prepared


def test_prepare_python_code_for_execution_replaces_single_coordinate_replan_click_with_visible_download_flow() -> None:
    request = StepRequest(
        user_prompt="Continue from the visible official download page and download `targetapp-setup.exe` only.",
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        observation_text="Visible official browser page with a download button.",
        replan_requested=True,
        replan_reasons=["execution_error"],
    )
    prepared = _prepare_python_code_for_execution(
        request,
        """import pyautogui
pyautogui.click(720, 480)
""",
    )
    assert "advance_visible_download_flow(" in prepared
    assert "pyautogui.click(720, 480)" not in prepared


def test_prepare_python_code_for_execution_replaces_browser_save_shortcut_flow_with_visible_download_flow() -> None:
    request = StepRequest(
        user_prompt="Open the official page at https://vendor.example/download and download `targetapp-setup.exe`.",
        execution_style="gui_first",
        replan_requested=True,
        replan_reasons=["execution_error"],
    )
    prepared = _prepare_python_code_for_execution(
        request,
        """import pyautogui, time
pyautogui.hotkey('ctrl', 't')
pyautogui.write('https://vendor.example/download', interval=0.05)
pyautogui.press('enter')
time.sleep(2)
pyautogui.click(600, 500)
pyautogui.hotkey('ctrl', 's')
""",
    )
    assert "advance_visible_download_flow(" in prepared
    assert "open_url_and_wait(" in prepared
    assert "pyautogui.hotkey('ctrl', 's')" not in prepared


def test_prepare_python_code_for_execution_opens_search_without_replacing_http_bypass_with_ocr_flow() -> None:
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
    assert "advance_visible_download_flow(" not in prepared
    assert "click_download_like_target(" not in prepared
    assert "browser_page_has_error_state(" not in prepared
    assert "urllib.request.urlopen" in prepared


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
    assert "advance_visible_download_flow(" not in prepared
    assert "urllib.request.urlopen" in prepared


def test_prepare_python_code_for_execution_auto_clicks_search_result_for_visible_search_results() -> None:
    request = StepRequest(
        user_prompt="Use Python to open the official vendor page and download the Windows installer `.exe`.",
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        observation_text="search results | official download | windows",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=targetapp+official+windows+download", expected_title_tokens=["targetapp"])',
            }
        },
    )
    prepared = _prepare_python_code_for_execution(request, 'print("continue")')
    assert prepared == 'print("continue")'
    assert "advance_visible_download_flow(" not in prepared
    assert "search_first = True" not in prepared


def test_deprecated_ocr_helper_calls_are_detected() -> None:
    assert _uses_deprecated_ocr_helper("result = click_download_like_target(timeout_s=3)") is True
    assert _uses_deprecated_ocr_helper("lines = ocr_screen_text_regions(max_lines=20)") is True
    assert _uses_deprecated_ocr_helper("import pyautogui\npyautogui.click(900, 420)") is False


def test_ocr_observation_text_is_sanitized_when_framework_ocr_helpers_are_disabled() -> None:
    assert _sanitize_observation_text_for_model("OCR visible text with download/install cues: 다운로드") is None
    assert _sanitize_observation_text_for_model("Visible installer wizard is open") == "Visible installer wizard is open"


def test_extract_prompt_install_marker_path() -> None:
    prompt = "Write `~/Downloads/computer-use-agent/targetapp/install-success.json` when finished."
    assert _extract_prompt_install_marker_path(prompt) == "~/Downloads/computer-use-agent/targetapp/install-success.json"


def test_extract_prompt_launch_marker_path() -> None:
    prompt = "Write `~/Downloads/computer-use-agent/targetapp/launch-success.json` only after launch succeeds."
    assert _extract_prompt_launch_marker_path(prompt) == "~/Downloads/computer-use-agent/targetapp/launch-success.json"


def test_visible_installer_observation_detected_from_language_dialog_text() -> None:
    request = StepRequest(
        user_prompt="어떤 프로그램을 설치해줘",
        execution_style="gui_first",
        observation_text="OCR visible text: Installer Language | Please select language | 확인",
    )
    assert _looks_like_visible_installer_observation(request) is True


def test_synthesized_visible_installer_recovery_code_prefers_existing_visible_installer() -> None:
    request = StepRequest(
        user_prompt=(
            "Use executable Python only. Find the existing installer `.exe` in "
            "`%USERPROFILE%\\\\Downloads\\\\computer-use-agent\\\\targetapp-1234\\\\`, launch it once, "
            "and end only when you have written `~/Downloads/computer-use-agent/targetapp-1234/install-success.json`."
        ),
        execution_style="gui_first",
        observation_text="OCR visible text: Installer Language | Please select language | 확인",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert "advance_visible_installer_flow(" in code
    assert "install-success.json" in code
    assert 'VISIBLE_INSTALLER = True' in code
    assert 'os.startfile(str(installer))' in code
    assert "TARGET_KEYWORDS" in code
    assert "SYSTEM_APP_NAMES" in code
    assert '"store.exe"' in code
    assert "installer = find_existing_installer()" in code
    assert "_iter_registry_candidate_paths()" in code
    assert "_is_valid_installed_executable(" in code
    assert "FILENAME_TARGET_KEYWORDS" in code
    assert "_matches_filename_target(" in code
    assert '"/appdata/local/temp/"' in code
    assert '"setup"' in code
    assert 'print(f"already installed: {existing}")' not in code


def test_framework_visible_installer_recovery_selected_for_gui_first_existing_installer_task() -> None:
    request = StepRequest(
        user_prompt=(
            "Use executable Python only. "
            "Find the existing installer `.exe` in `%USERPROFILE%\\\\Downloads\\\\computer-use-agent\\\\targetapp-1234\\\\`, "
            "launch it once, and then continue from the resulting installer UI."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_installer_recovery(request) is True


def test_framework_visible_installer_recovery_selected_for_downloaded_msi_install_chunk() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python to locate the downloaded MSI in `~/Downloads`, then install the app by "
            "launching `msiexec /i <installer.msi>` from Python. Prefer the normal interactive "
            "Windows Installer flow rather than a silent install."
        ),
        execution_style="gui_first",
    )
    assert _looks_like_existing_installer_launch_task(request.user_prompt) is True
    assert _should_use_framework_visible_installer_recovery(request) is True
    assert _should_use_framework_visible_download_flow(request) is False


def test_framework_visible_download_flow_selected_for_gui_first_download_only_chunk() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python to open the official vendor page and download only the Windows installer "
            "to `~/Downloads/targetapp-setup.exe`."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_download_flow(request) is True


def test_framework_visible_installer_recovery_selected_for_launch_downloaded_named_msi_chunk() -> None:
    request = StepRequest(
        user_prompt=(
            "Using Python automation, launch the downloaded DB Browser for SQLite MSI installer "
            "from Downloads and complete the installation with default GUI options. Prefer a "
            "normal MSI run via subprocess or msiexec if needed."
        ),
        execution_style="gui_first",
    )
    assert _looks_like_existing_installer_launch_task(request.user_prompt) is True
    assert _should_use_framework_visible_installer_recovery(request) is True
    assert _should_use_framework_visible_download_flow(request) is False


def test_existing_installer_recovery_not_selected_for_launch_chunk() -> None:
    request = StepRequest(
        user_prompt=(
            "Prefer reading `~/Downloads/computer-use-agent/targetapp/install-success.json` first, "
            "launch the installed app once, and write "
            "`~/Downloads/computer-use-agent/targetapp/launch-success.json` after the app is running."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_installer_recovery(request) is False


def test_launch_chunk_task_detected() -> None:
    prompt = (
        "Prefer reading `~/Downloads/computer-use-agent/targetapp/install-success.json` first, "
        "launch the installed app once, bring the app window to the foreground if needed, and "
        "write `~/Downloads/computer-use-agent/targetapp/launch-success.json` only after launch succeeded."
    )
    assert _looks_like_launch_app_chunk_task(prompt) is True


def test_framework_visible_launch_recovery_selected_for_gui_first_launch_chunk() -> None:
    request = StepRequest(
        user_prompt=(
            "Prefer reading `~/Downloads/computer-use-agent/targetapp/install-success.json` first, "
            "launch the installed app once, and write "
            "`~/Downloads/computer-use-agent/targetapp/launch-success.json` only after launch succeeded."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_launch_recovery(request) is True


def test_framework_visible_launch_recovery_not_selected_during_installer_replan() -> None:
    request = StepRequest(
        user_prompt=(
            "Prefer reading `~/Downloads/install-success.json` first, "
            "launch the installed app once, and write `~/Downloads/launch-success.json` after launch succeeded."
        ),
        execution_style="gui_first",
        last_execution={
            "payload_metadata": {
                "executed_python_code": "EXPECTED_INSTALLER_GLOB = \"TargetApp_Setup.exe\"\nflow = advance_visible_installer_flow(timeout_s=8.0)",
            }
        },
    )
    assert _should_use_framework_visible_launch_recovery(request) is False


def test_visible_flow_extra_targets_ignore_helper_names_and_keep_app_keyword() -> None:
    request = StepRequest(
        user_prompt=(
            "대상 앱과 일치하는 installer만 사용하세요. 파일명은 가능하면 `카카오톡`, `install` 같은 대상 앱 키워드를 포함해야 하며, "
            "보이는 download/install control 이 있으면 screenshot-grounded coordinate click을 우선 고려하세요. "
            "Do not use OCR helper names such as `click_download_like_target()` or `click_text_targets([...])`. "
            "Do not import pywin32, pywinauto, win32gui, win32con, win32api, pythoncom. "
            "Avoid recursively scanning %LOCALAPPDATA% or %ProgramFiles%."
        ),
        execution_style="gui_first",
    )
    assert _visible_flow_extra_targets(request, limit=3) == ["카카오톡"]


def test_visible_flow_extra_targets_prefer_explicit_installer_filename() -> None:
    request = StepRequest(
        user_prompt=(
            "Locate `C:\\Users\\user\\Downloads\\KakaoTalk_Setup.exe`, verify it is the downloaded Windows installer, "
            "and run it with Python automation. 이번엔 network-fetch logic로 돌아가지 마세요."
        ),
        execution_style="gui_first",
    )
    assert _visible_flow_extra_targets(request, limit=3) == ["kakaotalk"]


def test_synthesized_visible_launch_recovery_ignores_invalid_install_marker_and_writes_launch_marker() -> None:
    request = StepRequest(
        user_prompt=(
            "Prefer reading `~/Downloads/computer-use-agent/targetapp/install-success.json` first, "
            "launch the installed app once, bring the app window to the foreground if needed, and "
            "write `~/Downloads/computer-use-agent/targetapp/launch-success.json` only after launch succeeded."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_launch_recovery_code(request)
    assert 'INSTALL_MARKER_PATH = Path(os.path.expanduser("~/Downloads/computer-use-agent/targetapp/install-success.json"))' in code
    assert 'LAUNCH_MARKER_PATH = Path(os.path.expanduser("~/Downloads/computer-use-agent/targetapp/launch-success.json"))' in code
    assert 'CONTEXT_PATH = Path(os.path.expanduser("~/Downloads/computer-use-agent/targetapp/computer-use-agent-context.json"))' in code
    assert "ignoring invalid install marker candidate" in code
    assert "_read_context_candidate()" in code
    assert "_iter_registry_candidate_paths()" in code
    assert "_process_running(exe_path)" in code
    assert "write_install_marker(exe_path)" in code
    assert "write_launch_marker(exe_path)" in code
    assert "write_action_context(" in code


def test_synthesized_visible_launch_recovery_defaults_to_downloads_marker_and_prompt_targets() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python to confirm the installed TargetApp desktop app is launchable and running. "
            "If it is not already open, start TargetApp from the installed app or shortcut. "
            "source task: targetapp를 설치해줘"
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_launch_recovery_code(request)
    assert 'INSTALL_MARKER_PATH = Path.home() / "Downloads" / "install-success.json"' in code
    assert 'LAUNCH_MARKER_PATH = Path.home() / "Downloads" / "launch-success.json"' in code
    assert 'CONTEXT_PATH = Path.home() / "Downloads" / "computer-use-agent-context.json"' in code
    assert 'PROMPT_TARGETS = ["targetapp"]' in code
    assert "FILENAME_TARGET_KEYWORDS = _normalize_tokens(PROMPT_TARGETS, skip_extension_tokens=True)" in code
    assert "if FILENAME_TARGET_KEYWORDS and not any(keyword in key for keyword in FILENAME_TARGET_KEYWORDS):" in code


def test_fallback_browser_search_url_uses_korean_download_terms_for_hangul_tasks() -> None:
    url = _fallback_browser_search_url("카카오톡 pc버전 프로그램을 설치해줘")
    assert url is not None
    assert "%EA%B3%B5%EC%8B%9D" in url
    assert "%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C" in url


def test_fallback_browser_search_url_adds_vendor_domain_filters_from_prompt_urls() -> None:
    url = _fallback_browser_search_url("Use the official page https://pc.example.com/download and continue.")
    assert url is not None
    assert "site%3Aexample.com" in url
    assert "site%3Aexamplecorp.com" in url


def test_replan_fallback_browser_search_url_does_not_use_workflow_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Do not use urllib, requests, regex-based HTML scraping, or fresh direct-download discovery.\n"
            "Previous stderr summary: no official Windows installer/archive candidate found on the current page"
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://mydev.kr/", expected_title_tokens=["memoit193", "mydev"])',
            }
        },
    )
    url = _fallback_browser_search_url_for_request(
        request,
        prompt_url="https://mydev.kr/",
        extra_targets=["memoit193", "mydev"],
    )
    assert url is not None
    assert "memoit" in url
    assert "memoit193" not in url
    assert "mydev" in url
    assert "site%3Amydev.kr" in url
    assert "site%3Agoogle.com" not in url
    assert "network" not in url
    assert "parsing" not in url
    assert "logic" not in url


def test_select_prompt_browser_url_prefers_korean_locale_for_hangul_task() -> None:
    prompt = (
        "카카오톡 pc버전 프로그램을 설치해줘. "
        "Use these URLs first: "
        "https://www.example.com/page/service/app?lang=en "
        "https://www.example.com/page/service/app?lang=ko"
    )
    assert _select_prompt_browser_url(prompt) == "https://www.example.com/page/service/app?lang=ko"


def test_select_prompt_browser_url_canonicalizes_variant_download_page_to_root_download_page() -> None:
    prompt = (
        "Use the official vendor URLs first: "
        "https://downloads.vendor.example/download/lite/ "
        "and do not leave the official site."
    )
    assert _select_prompt_browser_url(prompt) == "https://downloads.vendor.example/download/"


def test_prompt_keyword_candidates_drop_install_control_stopwords() -> None:
    prompt = (
        "Return executable Python only. "
        "Do not repeat the same silent-install actions and switch logic. "
        "source task: 카카오톡 pc버전 프로그램을 설치해줘"
    )
    keywords = _prompt_keyword_candidates(prompt, limit=5)
    assert "actions" not in keywords
    assert "switch" not in keywords
    assert "silent-install" not in keywords


def test_synthesized_visible_download_completion_code_prefers_official_prompt_url_over_search_fallback() -> None:
    request = StepRequest(
        user_prompt="카카오톡 pc버전 프로그램을 설치해줘",
        execution_style="gui_first",
    )
    code = _synthesized_visible_download_completion_code(
        request,
        prompt_url="https://pc.example.com/talk",
    )
    assert "fallback_search_url = None" in code
    assert 'prompt_open_fallback_url = "https://www.google.com/search?q=' in code
    assert 'prompt_url = "https://pc.example.com/talk"' in code
    assert "prompt URL did not verify in browser" in code
    assert "fallback_search_url = prompt_open_fallback_url" in code
    assert "browser_page_has_error_state(" not in code
    assert code.index("open_url_and_wait(prompt_url") < code.index("advance_visible_download_flow(")


def test_synthesized_framework_visible_download_recovery_code_uses_prompt_url_without_visible_browser() -> None:
    request = StepRequest(
        user_prompt="카카오톡 pc버전 프로그램을 설치해줘",
        execution_style="gui_first",
        observation_text=None,
    )
    code = _synthesized_framework_visible_download_recovery_code(request)
    assert 'prompt_url = "https://www.google.com/search?q=' in code
    assert "open_url_and_wait(prompt_url" in code
    assert "advance_visible_download_flow(" in code
    assert "wait_for_recent_download_artifact(" in code
    assert "since_ts=download_started_at" in code


def test_synthesized_framework_visible_download_recovery_code_stays_on_visible_browser_when_grounded() -> None:
    request = StepRequest(
        user_prompt="Continue from the visible browser page and download the installer.",
        execution_style="gui_first",
        observation_text="Visible browser page with download button and installer name.",
    )
    code = _synthesized_framework_visible_download_recovery_code(request)
    assert "prompt_url = None" in code
    assert "open_url_and_wait(prompt_url" not in code
    assert "advance_visible_download_flow(" in code


def test_synthesized_visible_download_completion_code_uses_stable_replan_fallback_query() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Continue from the visible browser/download UI.\n"
            "Do not use urllib, requests, regex-based HTML scraping, or fresh direct-download discovery.\n"
            "Previous stderr summary: no official Windows installer/archive candidate found on the current page"
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://mydev.kr/", expected_title_tokens=["memoit193", "mydev"])',
            }
        },
    )
    code = _synthesized_visible_download_completion_code(
        request,
        prompt_url="https://mydev.kr/",
    )
    assert "network%20parsing%20logic" not in code
    assert "memoit" in code
    assert "site%3Amydev.kr" in code
    assert "site%3Agoogle.com" not in code


def test_synthesized_visible_download_completion_code_retries_visible_flow_before_failing_download_wait() -> None:
    request = StepRequest(
        user_prompt="Download into ~/Downloads/computer-use-agent/targetapp-1234/ and wait for the installer .exe to appear.",
        execution_style="gui_first",
    )
    code = _synthesized_visible_download_completion_code(
        request,
        prompt_url="https://download.example.com/app",
    )
    assert "for download_attempt in range(2)" in code
    assert "advanced visible download flow retry" in code
    assert "page_down_browser_view(steps=1)" in code
    assert "download_official_installer_from_page(" in code
    assert "CONTEXT_PATH = Path(os.path.expanduser(\"~/Downloads/computer-use-agent/targetapp-1234/computer-use-agent-context.json\"))" in code
    assert "ensure_action_context(CONTEXT_PATH, prompt_key=CONTEXT_PROMPT_KEY, prompt_excerpt=CONTEXT_PROMPT_EXCERPT)" in code
    assert "write_action_context(" in code


def test_existing_installer_launch_task_does_not_match_download_chunk_prompt() -> None:
    prompt = (
        "Use executable Python on the Windows machine to obtain the official Windows installer `.exe` "
        "and save it to Downloads. If an installer already exists in Downloads you may reuse it instead "
        "of downloading again, but this chunk is still the download step."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is False


def test_synthesized_visible_installer_recovery_uses_context_path_and_context_installer() -> None:
    request = StepRequest(
        user_prompt=(
            "Use executable Python only. Find the existing installer `.exe` in "
            "`%USERPROFILE%\\\\Downloads\\\\computer-use-agent\\\\targetapp-1234\\\\`, launch it once, "
            "and end only when you have written `~/Downloads/computer-use-agent/targetapp-1234/install-success.json`."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert 'CONTEXT_PATH = Path(os.path.expanduser("~/Downloads/computer-use-agent/targetapp-1234/computer-use-agent-context.json"))' in code
    assert "context_installer = _context_candidate(context_payload.get(\"installer_path\"))" in code
    assert "write_action_context(" in code
    assert "phase=\"installer_started\"" in code
    assert "phase=\"installed\"" in code


def test_extract_prompt_download_glob_uses_official_exe_url_basename() -> None:
    prompt = (
        "Use Python-first automation on Windows to download the official Windows installer `.exe` "
        "from `https://downloads.vendor.example/releases/TargetApp_Setup.exe` and wait for it to finish."
    )
    assert _extract_prompt_download_glob(prompt) == "TargetApp_Setup.exe"


def test_extract_prompt_download_glob_uses_official_msi_url_basename() -> None:
    prompt = (
        "Use Python-first automation on Windows to download the official Windows installer `.msi` "
        "from `https://downloads.vendor.example/releases/TargetApp_Setup.msi` and wait for it to finish."
    )
    assert _extract_prompt_download_glob(prompt) == "TargetApp_Setup.msi"


def test_extract_prompt_download_glob_uses_official_zip_url_basename() -> None:
    prompt = (
        "Use Python-first automation on Windows to download the official Windows installer archive `.zip` "
        "from `https://downloads.vendor.example/releases/TargetApp_Installer.zip` and wait for it to finish."
    )
    assert _extract_prompt_download_glob(prompt) == "TargetApp_Installer.zip"


def test_extract_prompt_download_glob_uses_official_alz_url_basename() -> None:
    prompt = (
        "Use Python-first automation on Windows to download the official Windows installer archive `.alz` "
        "from `https://downloads.vendor.example/releases/TargetApp_Installer.alz` and wait for it to finish."
    )
    assert _extract_prompt_download_glob(prompt) == "TargetApp_Installer.alz"


def test_extract_prompt_download_glob_uses_archive_subdir_pattern() -> None:
    prompt = (
        "Download into ~/Downloads/computer-use-agent/targetapp-1234/ "
        "and wait for the installer archive .zip to appear."
    )
    assert _extract_prompt_download_glob(prompt) == "computer-use-agent/targetapp-1234/*.zip"


def test_extract_prompt_download_glob_uses_explicit_downloads_path_filename() -> None:
    prompt = (
        "Save it to the user's Downloads folder as `~/Downloads/targetapp-windows-installer.exe` "
        "and do not finish until the file is fully present."
    )
    assert _extract_prompt_download_glob(prompt) == "targetapp-windows-installer.exe"


def test_extract_prompt_download_glob_ignores_generic_dot_exe_token() -> None:
    prompt = "Download the official Windows installer `.exe` and then run it from Downloads."
    assert _extract_prompt_download_glob(prompt) is None


def test_synthesized_visible_download_completion_code_waits_for_prompt_named_installer() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python-first automation on Windows to download the official Windows installer `.exe` "
            "from `https://downloads.vendor.example/releases/TargetApp_Setup.exe`. "
            "Save it to the user's Downloads folder as `TargetApp_Setup.exe`."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_download_completion_code(
        request,
        prompt_url="https://downloads.vendor.example/releases/TargetApp_Setup.exe",
    )
    assert code.startswith("import fnmatch\nimport time\nfrom pathlib import Path\n")
    assert 'wait_for_stable_download("TargetApp_Setup.exe"' in code
    assert 'print(f"download ready: {installer}")' in code


def test_synthesized_visible_download_completion_code_waits_for_prompt_named_archive() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python-first automation on Windows to download the official Windows installer archive `.zip` "
            "from `https://downloads.vendor.example/releases/TargetApp_Installer.zip`. "
            "Save it to the user's Downloads folder as `TargetApp_Installer.zip`."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_download_completion_code(
        request,
        prompt_url="https://downloads.vendor.example/releases/TargetApp_Installer.zip",
    )
    assert 'wait_for_stable_download("TargetApp_Installer.zip"' in code
    assert "download_official_installer_from_page(" in code
    expanded = _expand_runtime_helpers("download_official_installer_from_page('https://vendor.example/download')")
    assert 'installer_suffixes = (".exe", ".msi", ".zip", ".alz")' in expanded
    assert "installer/archive candidate" in expanded


def test_generated_code_ignores_prompt_urls_allows_same_official_registrable_host() -> None:
    prompt = "Download the installer from the official page https://vendor.example/download/."
    code = 'import urllib.request\nurllib.request.urlopen("https://cdn.vendor.example/releases/TargetApp_Installer.zip")'
    assert (
        _generated_code_ignores_prompt_urls(
            user_prompt=prompt,
            python_code=code,
            active_replan_reasons=["execution_error"],
        )
        is False
    )
    unrelated_code = 'import urllib.request\nurllib.request.urlopen("https://unrelated.example/releases/TargetApp_Installer.zip")'
    assert (
        _generated_code_ignores_prompt_urls(
            user_prompt=prompt,
            python_code=unrelated_code,
            active_replan_reasons=["execution_error"],
        )
        is True
    )


def test_synthesized_visible_download_completion_code_rejects_mismatched_context_installer() -> None:
    request = StepRequest(
        user_prompt=(
            "Open the official vendor download page and download the Windows installer `.exe` "
            "as `TargetApp_Setup.exe` into Downloads."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_download_completion_code(
        request,
        prompt_url="https://vendor.example/download/",
    )
    assert code.startswith("import fnmatch\nimport time\nfrom pathlib import Path\n")
    assert "context_installer_name = Path(context_installer).name.lower()" in code
    assert 'expected_download_glob = "targetapp_setup.exe"' in code
    assert 'print(f"ignoring mismatched context installer: {context_installer}")' in code


def test_synthesized_visible_download_completion_uses_prompt_scoped_global_context() -> None:
    request = StepRequest(
        user_prompt=(
            "Open the official vendor download page and download the Windows installer `.exe` "
            "as `TargetApp_Setup.exe` into Downloads."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_download_completion_code(
        request,
        prompt_url="https://vendor.example/download/",
    )
    assert 'Path.home() / "Downloads" / "computer-use-agent-context.json"' in code
    assert "CONTEXT_PROMPT_KEY =" in code
    assert "CONTEXT_PROMPT_EXCERPT =" in code
    assert "ensure_action_context(CONTEXT_PATH, prompt_key=CONTEXT_PROMPT_KEY, prompt_excerpt=CONTEXT_PROMPT_EXCERPT)" in code
    assert "prompt_key=CONTEXT_PROMPT_KEY" in code
    expanded = _expand_runtime_helpers(code)
    compile(expanded, "<visible-download-prompt-context>", "exec")


def test_action_context_resets_when_prompt_key_changes(tmp_path) -> None:
    context_path = tmp_path / "computer-use-agent-context.json"
    code = _expand_runtime_helpers(
        """
payload1 = write_action_context(path, prompt_key="prompt-a", prompt_excerpt="first", installer_path="old.exe")
payload2 = read_action_context(path, prompt_key="prompt-b")
payload_started = ensure_action_context(path, prompt_key="prompt-b", prompt_excerpt="second")
payload3 = write_action_context(path, prompt_key="prompt-b", prompt_excerpt="second", installer_path="new.exe")
payload4 = read_action_context(path, prompt_key="prompt-b")
"""
    )
    namespace = {"path": context_path}
    exec(code, namespace)
    assert namespace["payload1"]["installer_path"] == "old.exe"
    assert namespace["payload2"]["_prompt_mismatch"] is True
    assert "installer_path" not in namespace["payload2"]
    assert namespace["payload_started"]["phase"] == "context_started"
    assert "installer_path" not in namespace["payload_started"]
    assert namespace["payload3"]["installer_path"] == "new.exe"
    assert namespace["payload4"]["prompt_key"] == "prompt-b"
    assert namespace["payload4"]["installer_path"] == "new.exe"


def test_existing_installer_launch_task_detected_for_generic_downloaded_installer_prompt() -> None:
    prompt = (
        "Locate the downloaded installer `.exe` in Downloads, verify it is the Windows installer, "
        "run it with Python automation, and proceed through the installer wizard."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is True


def test_visible_installer_recovery_not_selected_for_generic_installer_prompt() -> None:
    request = StepRequest(
        user_prompt=(
            "Locate the downloaded installer `.exe` in Downloads, verify it is the Windows installer, "
            "run it with Python automation, and proceed through the installer wizard."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_installer_recovery(request) is True


def test_synthesized_visible_installer_recovery_prefers_prompt_named_installer() -> None:
    request = StepRequest(
        user_prompt=(
            "Locate `~/Downloads/TargetApp_Setup.exe`, verify it is the downloaded Windows installer, "
            "and run it with Python automation."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert 'EXPECTED_INSTALLER_GLOB = "TargetApp_Setup.exe"' in code
    assert "for path in TARGET_DIR.glob(pattern):" in code


def test_synthesized_visible_installer_recovery_reuses_running_installer() -> None:
    request = StepRequest(
        user_prompt=(
            "Locate `~/Downloads/TargetApp_Setup.exe`, verify it is the downloaded Windows installer, "
            "reuse an already-running matching installer if it is open, and continue through the installer wizard."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert "def _installer_process_running() -> bool:" in code
    assert 'print(f"reusing running installer: {installer}")' in code


def test_synthesized_visible_installer_recovery_extracts_prompt_named_archive() -> None:
    request = StepRequest(
        user_prompt=(
            "Use executable Python to extract the downloaded `~/Downloads/TargetApp_Installer.zip`, "
            "locate the Windows installer executable or MSI inside the extracted files, and run it."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert 'EXPECTED_INSTALLER_GLOB = "TargetApp_Installer.zip"' in code
    assert 'ARCHIVE_INSTALLER_SUFFIXES = {".zip", ".alz"}' in code
    assert "def extract_archive_installer(" in code
    assert "zipfile.ZipFile(archive)" in code
    assert "installer = extract_archive_installer(archive_installer)" in code
    expanded = _expand_runtime_helpers(code)
    compile(expanded, "<visible-installer-archive-recovery>", "exec")


def test_visible_flow_extra_targets_prefers_installer_filename_tokens() -> None:
    request = StepRequest(
        user_prompt=(
            "Launch `~/Downloads/DB.Browser.for.SQLite-v3.13.1-win32.msi` and complete installation. "
            "Previous stdout summary: network-fetch idb found"
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert 'EXTRA_TARGETS = ["db", "browser", "sqlite"]' in code
    assert "network-fetch" not in code


def test_expand_runtime_helpers_visible_installer_flow_handles_cancel_confirmation_dialogs() -> None:
    expanded = _expand_runtime_helpers("advance_visible_installer_flow(extra_targets=['targetapp'])")
    assert "def _dialog_regions(region):" in expanded
    assert "def _find_cancel_confirmation_region(region):" in expanded
    assert '"installer_cancel_detected"' in expanded
    assert '"installer_cancel_decline_keys"' in expanded
    assert "for key_name in ('alt+n', 'enter'):" in expanded or 'for key_name in ("alt+n", "enter"):' in expanded


def test_expand_runtime_helpers_visible_installer_flow_does_not_target_runner_window() -> None:
    expanded = _expand_runtime_helpers("advance_visible_installer_flow(extra_targets=['targetapp'])")
    assert '"computer-use"' in expanded
    assert '"training-generator"' in expanded
    assert '"gui-owl"' in expanded
    assert "if not generic_title_hit and title_keyword_hits <= 0:" in expanded
    assert '"no installer-like window"' in expanded
    assert '"installer_text_click_first"' in expanded
    assert "def _license_accept_prompt_visible(region):" in expanded
    assert "def _install_action_prompt_visible(region):" in expanded
    assert "def _set_license_checkbox_child_checked(region):" in expanded
    assert "SendMessageW(child_hwnd, 0x00F5, 0, 0)" in expanded
    assert "SendMessageW(child_hwnd, 0x00F1, 1, 0)" in expanded
    assert '"hwnd": hwnd' in expanded
    assert '"process_name": str(process_meta.get("name") or "")' in expanded
    assert '"installer_license_accept_keys"' in expanded
    assert '"installer_keyboard_primary_guided_install"' in expanded
    assert '"installer_guided_action_click"' in expanded
    assert "def _installer_candidate_allowed(" in expanded
    assert 'heuristic_sweep_threshold = 0 if installer_mode else 2' in expanded
    assert 'candidate_source="word_bbox"' in expanded
    assert 'candidate_source="line_bbox"' in expanded
    assert '(0.78, 0.90)' in expanded
    assert '(0.90, 0.90)' in expanded
    assert "def _enumerate_child_controls(region):" in expanded
    assert "def _click_primary_action_child_button(region):" in expanded
    assert '"installer_child_button_click"' in expanded


def test_synthesized_visible_installer_recovery_launches_msi_when_ui_not_confirmed() -> None:
    request = StepRequest(
        user_prompt=(
            "Launch `~/Downloads/TargetApp-v1.2.3.msi` and complete installation. "
            "End only when `~/Downloads/install-success.json` exists."
        ),
        execution_style="gui_first",
        observation_text="OCR visible text: desktop",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert 'VISIBLE_INSTALLER = False' in code
    assert 'subprocess.Popen(["msiexec.exe", "/i", str(installer), "/passive", "/norestart"])' in code
    assert '_launch_installer("no visible installer UI")' in code
    assert '_launch_installer("visible installer UI not confirmed")' in code


def test_synthesized_visible_installer_recovery_does_not_use_extension_token_as_filename_target() -> None:
    request = StepRequest(
        user_prompt=(
            "Locate `~/Downloads/TargetApp_Setup.exe`, verify it is the downloaded Windows installer, "
            "and run it with Python automation."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_visible_installer_recovery_code(request)
    assert "def _normalize_tokens(values, *, skip_extension_tokens: bool = False)" in code
    assert "extension_tokens = ('exe', 'msi', 'zip', 'alz', 'bat', 'cmd', 'lnk', 'com', 'scr')" in code
    assert "FILENAME_TARGET_KEYWORDS = _normalize_tokens(" in code
    target_section = code.split("FILENAME_TARGET_KEYWORDS =", 1)[1].split("def _is_temp_like_path", 1)[0]
    assert "REQUESTED_INSTALLER_KEYWORDS or EXTRA_TARGETS" in target_section
    assert ".name for path in INSTALLERS" not in target_section


def test_expand_runtime_helpers_search_result_click_avoids_blind_heuristics() -> None:
    expanded = _expand_runtime_helpers("click_search_result_like_target(extra_targets=['targetapp'])")
    assert "def _heuristic_browser_click(" in expanded
    assert "browser_search_result_region" in expanded
    assert "screen-browser-region-fallback" in expanded
    assert "active_region = _browser_window_region()" in expanded
    helper_section = expanded.split("def click_search_result_like_target(", 1)[1].split("def click_text_targets(", 1)[0]
    assert "context_match_scope=\"near\"" in helper_section
    assert "allow_heuristic_fallback=False" in helper_section
    assert "query_reject_texts.append" in helper_section
    assert "reject_texts=query_reject_texts" in helper_section
    assert "min_relative_top_px=210" in helper_section


def test_expand_runtime_helpers_includes_responsive_header_menu_flow() -> None:
    expanded = _expand_runtime_helpers("advance_visible_download_flow(extra_targets=['targetapp'])")
    assert "def open_responsive_header_menu(" in expanded
    assert 'heuristic_mode="menu"' in expanded
    assert "browser_header_menu_region" in expanded
    menu_section = expanded.split("def open_responsive_header_menu(", 1)[1].split("def click_text_targets(", 1)[0]
    assert "targets.extend" not in menu_section
    assert "min_primary_hits=1" in menu_section
    assert "skip_click_points=skip_click_points" in menu_section
    assert "max_relative_top_px=240" not in menu_section
    assert "exact_word_targets=True" not in menu_section
    assert "responsive_header_menu_prefetch" in expanded
    assert "responsive_header_menu_retry" in expanded
    assert "menu_clicked_points" in expanded
    assert "_try_menu_candidates_then_download" in expanded
    assert "def _budgeted_timeout" in expanded
    assert "visible download flow time budget exhausted" in expanded
    assert "candidate retry budget exhausted" in expanded
    assert "diversion_cues_present" in expanded
    assert "allow_heuristic_fallback=False" in expanded
    assert "download_control_scroll_retry" in expanded
    assert "download_keyboard_fallback" not in expanded
    assert "search_result_keyboard_fallback" not in expanded
    assert "def _tab_enter(" not in expanded
    assert "page_down_browser_view(steps=1" in expanded
    assert "context_targets=list(extra_targets or [])" in expanded
    assert "require_context=bool(extra_targets)" in expanded
    assert "download_context_scope = \"near\" if search_results_visible else \"page\"" in expanded
    assert "context_match_scope=download_context_scope" in expanded
    assert "def click_download_related_fallback(" in expanded
    assert "download_related_window_fallback_skipped" in expanded
    assert "clear_download_page_required" in expanded
    assert "download_action_text" in expanded
    assert "candidate_index" in expanded
    assert "skip_click_points=clicked_points" in expanded
    assert "download_related_window_fallback_page_open" in expanded
    assert 'after_menu": False' in expanded


def test_expand_runtime_helpers_overlay_dismiss_does_not_click_without_overlay() -> None:
    expanded = _expand_runtime_helpers("dismiss_browser_overlay()")
    assert "if not overlay_detected:" in expanded
    assert "overlay_context_detected and overlay_action_detected" in expanded
    assert "context_targets=overlay_context_terms" in expanded
    assert '"더보기"' not in expanded
    assert '"more"' not in expanded
    assert '"dismissed": False' in expanded
    no_overlay_section = expanded.split("if not overlay_detected:", 1)[1].split("try:", 1)[0]
    assert "SetCursorPos" not in no_overlay_section
    assert "mouse_event" not in no_overlay_section


def test_expand_runtime_helpers_open_url_does_not_pull_ocr_helper() -> None:
    expanded = _expand_runtime_helpers('open_url_and_wait("https://example.com", expected_title_tokens=["example"])')
    assert "def _browser_window_candidates()" in expanded
    assert "def _screen_text_matches_expected()" not in expanded
    assert "ocr_screen_text_regions(" not in expanded
    assert "expected visible page tokens" in expanded


def test_expand_runtime_helpers_includes_browser_error_state_detection() -> None:
    expanded = _expand_runtime_helpers('browser_page_has_error_state(expected_title_tokens=["targetapp"])')
    assert "def browser_page_has_error_state(" in expanded
    assert "404" in expanded
    assert "not found" in expanded
    assert "ocr_screen_text_regions(" in expanded


def test_expand_runtime_helpers_includes_browser_search_state_detection() -> None:
    expanded = _expand_runtime_helpers('browser_page_has_search_results(expected_title_tokens=["targetapp"])')
    assert "def browser_page_has_search_results(" in expanded
    assert "bing" in expanded
    assert "google" in expanded
    assert "duckduckgo" in expanded
    assert "ocr_screen_text_regions(" in expanded


def test_expand_runtime_helpers_includes_page_down_browser_view() -> None:
    expanded = _expand_runtime_helpers("page_down_browser_view(steps=1)")
    assert "def page_down_browser_view(" in expanded
    assert "vk_next = 0x22" in expanded


def test_expand_runtime_helpers_includes_official_page_download_recovery() -> None:
    expanded = _expand_runtime_helpers(
        'download_official_installer_from_page("https://example.com/app", extra_targets=["targetapp"], download_glob="computer-use-agent/demo/*.exe")'
    )
    assert "def download_official_installer_from_page(" in expanded
    assert "destination_dir = downloads" in expanded
    assert '.replace("\\\\u002F", "/")' in expanded
    assert "all official installer candidates failed" in expanded


def test_prompt_url_violation_rejects_search_discovery_when_official_url_exists() -> None:
    user_prompt = (
        "Use Python to continue from the visible browser first and download the Windows installer. "
        "Official URL: https://pc.example.com/download"
    )
    search_url = "https://www.google.com/search?q=targetapp%20official%20windows%20download"
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
        is True
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
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=targetapp+official+windows+download", expected_title_tokens=["targetapp"])',
            }
        },
    )
    recovery_code = _synthesized_visible_ui_click_recovery_code(request)
    assert "open_url_and_wait(" in recovery_code
    assert "advance_visible_download_flow(" in recovery_code
    assert "search_url=fallback_search_url" in recovery_code


def test_reported_failure_detected_from_stdout_or_stderr_even_with_zero_exit_code() -> None:
    execution = {
        "return_code": 0,
        "stdout_tail": "Error: download failed",
        "stderr_tail": "Invoke-WebRequest : WebException",
        "error_info": None,
    }
    assert _looks_like_reported_failure(execution) is True


def test_download_chunk_completed_accepts_official_page_recovery_marker() -> None:
    assert _looks_like_download_chunk_completed(
        user_prompt="success target: installer `.exe` exists in Downloads",
        last_execution={
            "return_code": 0,
            "stdout_tail": "download recovered from official page: C:\\\\Users\\\\qkqxl\\\\Downloads\\\\computer-use-agent\\\\targetapp\\\\KakaoTalk_Setup.exe",
            "stderr_tail": "",
        },
    ) is True


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


def test_visible_flow_extra_targets_keep_installer_prefix_and_task_keywords() -> None:
    request = StepRequest(
        user_prompt=(
            "Using Python on Windows, open the official Memoit site at https://mydev.kr/ "
            "and download only `setup_memoit193.exe` for 메모잇."
        ),
        execution_style="gui_first",
    )
    keywords = _visible_flow_extra_targets(request, limit=4)
    assert "memoit193" in keywords
    assert "memoit" in keywords
    assert "메모잇" in keywords


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


def test_guessed_artifact_url_generation_detected_for_page_prompt_same_host_exe_jump() -> None:
    prompt = (
        "Open the official vendor page at https://mydev.kr/ and download the installer only after discovering the real link from that official page."
    )
    code = """from pathlib import Path
import urllib.request
target = Path.home() / "Downloads" / "setup_memoit193.exe"
with urllib.request.urlopen("https://mydev.kr/setup_memoit193.exe", timeout=60) as resp, open(target, "wb") as fh:
    fh.write(resp.read())
"""
    assert _looks_like_guessed_artifact_url_generation(user_prompt=prompt, python_code=code) is True


def test_guessed_artifact_url_generation_not_detected_when_prompt_page_html_is_fetched_first() -> None:
    prompt = (
        "Open the official vendor page at https://mydev.kr/ and discover the installer link from that page."
    )
    code = """import re, urllib.request
html = urllib.request.urlopen("https://mydev.kr/", timeout=60).read().decode("utf-8", errors="replace")
match = re.findall(r'https://[^\\s"\']+\\.exe', html)
print(match[:1])
"""
    assert _looks_like_guessed_artifact_url_generation(user_prompt=prompt, python_code=code) is False


def test_gui_first_download_chunk_install_mix_detected_for_download_only_chunk() -> None:
    request = StepRequest(
        user_prompt=(
            "Current chunk success target: The installer `setup_memoit193.exe` is present in Downloads and has a nontrivial file size. "
            "Do only this chunk. Do not skip ahead to later chunks."
        ),
        execution_style="gui_first",
    )
    code = """from pathlib import Path
import subprocess
target = Path.home() / "Downloads" / "setup_memoit193.exe"
subprocess.Popen([str(target), "/VERYSILENT", "/SP-", "/NORESTART"])
"""
    assert _looks_like_gui_first_download_chunk_install_mix(request, code) is True


def test_gui_first_download_chunk_install_mix_ignores_browser_start_url_to_exe() -> None:
    request = StepRequest(
        user_prompt=(
            "Current chunk success target: The installer `setup_target.exe` is present in Downloads and has a nontrivial file size. "
            "Do only this chunk. Do not skip ahead to later chunks."
        ),
        execution_style="gui_first",
    )
    code = """import subprocess
subprocess.run(["start", "https://vendor.example/download/setup_target.exe"], shell=True)
"""
    assert _looks_like_gui_first_download_chunk_install_mix(request, code) is False


def test_gui_first_download_chunk_network_bypass_detected_for_html_scraping_without_gui() -> None:
    request = StepRequest(
        user_prompt=(
            "Current chunk success target: A target-app installer `.exe` or `.msi` exists in Downloads and is non-empty. "
            "Do only this chunk. Do not skip ahead to later chunks."
        ),
        execution_style="gui_first",
    )
    code = """import re
import urllib.request
html = urllib.request.urlopen("https://vendor.example/download", timeout=60).read().decode("utf-8")
links = re.findall(r'https?://[^\\s"\']+\\.exe', html)
print(links[:1])
"""
    assert _looks_like_gui_first_download_chunk_network_bypass(request, code) is True


def test_gui_first_download_bypass_can_execute_when_auto_open_prelude_is_available() -> None:
    request = StepRequest(
        user_prompt=(
            "Open the official vendor page at https://vendor.example/download and download the Windows installer `.exe`. "
            "Current chunk success target: A target-app installer `.exe` exists in Downloads."
        ),
        execution_style="gui_first",
    )
    code = """import urllib.request
html = urllib.request.urlopen("https://vendor.example/download", timeout=60).read().decode("utf-8")
print(html[:80])
"""
    assert _looks_like_gui_first_download_chunk_network_bypass(request, code) is True
    assert _should_soft_allow_gui_first_download_bypass_for_auto_open(
        request,
        code,
        gui_first_download_chunk_network_bypass=True,
    ) is False
    prepared = _prepare_python_code_for_execution(request, code)
    assert "open_url_and_wait(" in prepared
    assert "urllib.request.urlopen" in prepared


def test_gui_first_download_bypass_not_soft_allowed_when_visible_ui_exists() -> None:
    request = StepRequest(
        user_prompt="Current chunk success target: A target-app installer `.exe` exists in Downloads.",
        execution_style="gui_first",
        observation_text="visible browser page with download button",
    )
    code = """import urllib.request
urllib.request.urlopen("https://vendor.example/download").read()
"""
    assert _should_soft_allow_gui_first_download_bypass_for_auto_open(
        request,
        code,
        gui_first_download_chunk_network_bypass=True,
    ) is False


def test_gui_first_download_chunk_network_bypass_allows_screenshot_grounded_gui_code() -> None:
    request = StepRequest(
        user_prompt=(
            "Current chunk success target: A target-app installer `.exe` or `.msi` exists in Downloads and is non-empty. "
            "Do only this chunk. Do not skip ahead to later chunks."
        ),
        execution_style="gui_first",
    )
    code = """import pyautogui
import time
pyautogui.click(1180, 430)
time.sleep(3)
"""
    assert _looks_like_gui_first_download_chunk_network_bypass(request, code) is False


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


def test_invalid_retry_prompt_drops_silent_switch_guidance_after_gui_first_silent_shortcut() -> None:
    history = _history_for_invalid_python_retry_with_prompt(
        [],
        user_prompt=(
            "Locate the downloaded installer `.exe` in Downloads and complete the installer wizard. "
            "If installer UI is visible, continue from that visible UI first."
        ),
        step_index=0,
        previous_code="import subprocess\nsubprocess.Popen(['setup.exe', '/SILENT'])\n",
        gui_first_silent_install_shortcut=True,
    )

    joined = "\n".join(history)
    assert "do not start with /SILENT, /VERYSILENT, /SP-, or /NORESTART" in joined
    assert "launches it normally or advances the visible installer UI" in joined
    assert "tries silent install switches" not in joined
    assert "prefer common Windows silent installer switches" not in joined


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


def test_expand_runtime_helpers_injects_recent_download_waiter_definition() -> None:
    expanded = _expand_runtime_helpers(
        'installer = wait_for_recent_download_artifact(extra_targets=["targetapp"], since_ts=123.0)'
    )
    assert "def wait_for_recent_download_artifact(" in expanded
    assert "wait_for_stable_download(" in expanded
    assert 'installer = wait_for_recent_download_artifact(extra_targets=["targetapp"], since_ts=123.0)' in expanded


def test_expand_runtime_helpers_injects_open_url_and_wait_definition() -> None:
    code = """opened = open_url_and_wait(
    "https://www.kakaocorp.com/page/service/service/KakaoTalk?lang=en",
    expected_title_tokens=["kakao", "kakaotalk"],
)
print(opened)
"""
    expanded = _expand_runtime_helpers(code)
    assert "def ensure_windows_dpi_aware(" in expanded
    assert "def open_url_and_wait(" in expanded
    assert "ensure_windows_dpi_aware()" in expanded
    assert 'os.startfile(target_url)' in expanded
    assert '["cmd", "/c", "start", "", target_url]' in expanded
    assert '"--new-tab", target_url' in expanded
    assert 'expected_title_tokens=["kakao", "kakaotalk"]' in expanded


def test_expand_runtime_helpers_injects_recursive_download_click_helpers() -> None:
    code = """result = click_download_like_target(timeout_s=8.0)
print(result)
"""
    expanded = _expand_runtime_helpers(code)
    assert "def ensure_windows_dpi_aware(" in expanded
    assert "def click_download_like_target(" in expanded
    assert "def click_text_targets(" in expanded
    assert "def ocr_screen_text_regions(" in expanded
    assert "ensure_windows_dpi_aware()" in expanded
    assert '"download",' in expanded
    assert '"msi",' in expanded
    assert '"standard",' in expanded
    assert '"no installer",' in expanded
    assert '"nightly",' in expanded
    assert '"guide",' in expanded
    assert '"support",' in expanded
    assert 'click_horizontal_bias="center"' in expanded
    assert "visible_ocr=" in expanded
    assert '"ko-KR"' in expanded
    assert '"raw_text": raw_text' in expanded
    assert '"center_x": left + int(width / 2)' in expanded
    assert "def _word_box_candidates_for_line(line_index, line_item, line_score)" in expanded
    assert "def _exact_target_word_match(text)" in expanded
    assert '"click_left": click_left' in expanded
    assert '"click_width": click_width' in expanded
    assert '"candidate_source": "word_bbox"' in expanded
    assert "exact_word_targets=False" in expanded
    assert "allow_heuristic_fallback=False" in expanded
    assert "context_targets=context" in expanded
    assert "require_context=bool(context)" in expanded
    assert "skip_click_points=skip_click_points" in expanded
    assert "context_scope = \"near\" if browser_page_has_search_results" in expanded
    assert "context_match_scope=context_scope" in expanded
    assert "browser_download_cta_region" in expanded
    assert "heuristic_sweep_threshold = 0 if installer_mode else 2" in expanded


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
    assert 'installer_suffixes = (".exe", ".msi", ".zip", ".alz")' in code


def test_duplicate_generation_detected_for_same_script() -> None:
    code = """import subprocess
subprocess.run(["cmd", "/c", "echo", "ok"], check=False)
"""
    assert _looks_like_duplicate_generation(code, code) is True
    assert _looks_like_duplicate_generation(code, 'print("other")') is False


def test_extract_click_points_from_python_collects_unique_numeric_clicks() -> None:
    code = """import pyautogui
pyautogui.click(1000, 600)
pyautogui.doubleClick(x=1000, y=600)
click(420, 315)
pyautogui.click(1000, 600)
"""
    assert _extract_click_points_from_python(code) == [(1000, 600), (420, 315)]


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
    assert "Do not use executor-side OCR/text-click helpers" in rewritten
    assert "estimate the visible download/install control coordinates" in rewritten
    assert "This is a download-only step. Do not launch, silently install, or run the installer in this step." in rewritten
    assert "Do not guess another same-host `.exe` or `.msi` path" in rewritten


def test_replan_prompt_rewrite_for_gui_first_download_same_page_retry_avoids_previous_clicks() -> None:
    prompt = (
        "Use Python on Windows to open the official vendor page in the browser and download the Windows installer as a `.exe`."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["no_visual_change", "repeated_code_execution"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": """
open_url_and_wait("https://vendor.example/download", expected_title_tokens=["vendor"])
import pyautogui
pyautogui.click(1000, 600)
pyautogui.click(1180, 602)
""",
            },
            "stdout_tail": "Opened vendor page and clicked a visible button, but no download appeared.",
        },
    )
    assert "Stay on the currently visible browser tab/page first." in rewritten
    assert "do not reuse the same coordinates first" in rewritten
    assert "try the next distinct candidate in the same script before giving up" in rewritten
    assert "Do not treat the browser toolbar, address bar, tab strip, bookmarks bar" in rewritten
    assert "Avoid reusing these previous click coordinates first: (1000, 600), (1180, 602)." in rewritten


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


def test_replan_prompt_rewrite_keeps_source_task_hint_for_installer_replan() -> None:
    prompt = (
        "Use executable Python only. Do not download anything in this chunk. "
        "Locate the downloaded installer in Downloads, finish the installation, and end only when "
        "the installed app process is running. from this task: 메모잇 프로그램을 설치해줘."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["execution_error"],
        last_execution={
            "stderr_tail": "could not locate installed app executable for launch chunk",
        },
    )
    assert "Source task: 메모잇 프로그램을 설치해줘." in rewritten
    assert "메모잇" in _prompt_keyword_candidates(rewritten, limit=8)


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
    assert "LOCALAPPDATA\\\\Programs\\\\<TargetApp>" in rewritten


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


def test_retry_token_budget_caps_large_budgets() -> None:
    assert _retry_token_budget(256) == 256
    assert _retry_token_budget(512) == 512
    assert _retry_token_budget(800) == 640


def test_step_token_budget_caps_gui_first_download_steps() -> None:
    initial_request = StepRequest(
        user_prompt="targetapp 프로그램을 설치해줘",
        execution_style="gui_first",
        request_kind="task_step",
    )
    retry_request = StepRequest(
        user_prompt="targetapp 프로그램을 설치해줘",
        execution_style="gui_first",
        request_kind="task_step",
        replan_requested=True,
        replan_reasons=["execution_error"],
    )
    assert _step_token_budget(initial_request, 1024) == 640
    assert _step_token_budget(retry_request, 1024) == 512
