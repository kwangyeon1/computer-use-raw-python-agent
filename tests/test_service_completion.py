from __future__ import annotations

import base64
import json
import re
import urllib.parse

import computer_use_raw_python_agent.service as service_module
from computer_use_raw_python_agent.service import (
    _dependency_repair_user_prompt,
    _expand_runtime_helpers,
    _extract_click_points_from_python,
    _extract_prompt_download_glob,
    _extract_prompt_install_marker_path,
    _extract_prompt_launch_marker_path,
    _extract_prompt_urls,
    _fallback_browser_search_url,
    _fallback_browser_search_url_from_parts,
    _fallback_browser_search_url_for_request,
    _fallback_official_domain_urls,
    _generated_code_ignores_prompt_urls,
    _has_visible_gui_continuation_cues,
    _history_for_invalid_python_retry_with_prompt,
    _infer_response_done,
    _installer_filename_keywords,
    _installer_recovery_target_terms,
    _installer_launcher_pid_from_last_execution,
    _installer_ui_candidates_from_observation,
    _installer_ui_candidates_observation,
    _coerce_model_bbox,
    _execution_screenshot_region_for_request,
    _context_prompt_key_for_request,
    _context_prompt_key_for_target_terms,
    _model_ui_candidates_observation,
    _model_ui_ocr_elements_from_text,
    _model_ui_candidates_from_observation,
    _teacher_visible_installer_clicks_from_prompt,
    _score_model_ui_candidate,
    _looks_like_guessed_artifact_url_generation,
    _looks_like_gui_first_download_chunk_network_bypass,
    _looks_like_gui_first_download_chunk_install_mix,
    _looks_like_gui_first_silent_install_shortcut,
    _looks_like_gui_first_bottom_strip_click_generation,
    _looks_like_gui_first_visible_ui_bypass,
    _looks_like_duplicate_generation,
    _looks_like_missing_install_progress_generation,
    _looks_like_missing_image_template_generation,
    _looks_like_gui_first_installer_wait_without_ui_action,
    _is_compilable_python_code,
    _looks_like_direct_download_url_404,
    _looks_like_exhausted_visible_download_recovery,
    _looks_like_existing_installer_launch_task,
    _looks_like_launch_app_chunk_task,
    _looks_like_incomplete_install_attempt,
    _looks_like_installer_launched_but_app_not_found,
    _looks_like_installer_timeout,
    _looks_like_installer_url_discovery_failure,
    _looks_like_partial_download_page_navigation,
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
    _registrable_host_from_url,
    _select_prompt_browser_url,
    _select_request_prompt_browser_url,
    _select_validated_replan_search_url,
    _sanitize_observation_text_for_model,
    _search_url_matches_excluded_query,
    _search_url_validation_error,
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
    _should_use_model_ui_browser_prelude,
    _should_use_model_ui_candidates,
    _should_use_installer_ui_candidates,
    _should_use_model_ui_download_recovery,
    _should_use_model_ui_installer_recovery,
    _should_use_model_ui_launch_recovery,
    _synthesized_model_ui_download_recovery_code,
    _synthesized_model_ui_installer_recovery_code,
    _synthesized_model_ui_launch_recovery_code,
    _synthesized_model_ui_browser_prelude_code,
    _synthesized_official_download_recovery_code,
    _should_use_framework_official_download_recovery,
    _should_use_framework_official_download_retry_for_invalid_generation,
    _should_omit_screenshot_for_generation,
    _url_looks_like_search_results,
    _visible_flow_extra_targets,
)
from computer_use_raw_python_agent.models import StepRequest


def test_task_complete_marker_requires_confirmation_script() -> None:
    code = """# task_complete
import webbrowser
webbrowser.open("https://example.com/download")
"""
    assert _infer_response_done(python_code=code, raw_text=code) is False


def test_model_ui_candidate_bbox_maps_qwen_grounding_grid() -> None:
    assert _coerce_model_bbox([500, 500, 600, 600], image_size=(1920, 1080)) == (960, 540, 1152, 648)


def test_model_ui_candidate_bbox_accepts_large_absolute_coordinates() -> None:
    assert _coerce_model_bbox([1200, 200, 1460, 250], image_size=(1920, 1080)) == (1200, 200, 1460, 250)


def test_model_ui_search_result_bbox_keeps_qwen_grid_for_both_axes() -> None:
    assert _coerce_model_bbox([268, 312, 526, 338], image_size=(2560, 1440)) == (686, 449, 1347, 487)
    assert _coerce_model_bbox([282, 456, 376, 482], image_size=(2560, 1440)) == (722, 657, 963, 694)


def test_context_prompt_key_uses_top_level_source_task_across_chunks() -> None:
    download_request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: filezilla 설치해줘\n\n"
            "Download the installer into Downloads."
        )
    )
    install_request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: filezilla 설치해줘\n\n"
            "Run the installer and complete setup."
        )
    )

    download_key, download_excerpt = _context_prompt_key_for_request(download_request)
    install_key, install_excerpt = _context_prompt_key_for_request(install_request)

    assert download_key
    assert download_key == install_key
    assert download_excerpt == "source_task=filezilla 설치해줘"
    assert install_excerpt == "source_task=filezilla 설치해줘"


def test_context_prompt_key_for_target_terms_prefers_top_level_source_task() -> None:
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: 메모잇 설치해줘\n\n"
            "Use these exact page URLs first."
        )
    )

    key, excerpt = _context_prompt_key_for_target_terms(request, ["memoit193", "memoit"])

    assert key
    assert excerpt == "source_task=메모잇 설치해줘"


def test_search_result_observation_clears_after_opened_result_candidate() -> None:
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'FALLBACK_SEARCH_URL = "https://www.google.com/search?q=filezilla%20kr"',
            },
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates after opening a search result candidate",
        },
    )
    assert service_module._looks_like_search_results_observation(request) is False


def test_model_ui_candidate_scoring_prefers_download_target_text() -> None:
    request = StepRequest(
        user_prompt="sampleapp 프로그램을 설치해줘",
        execution_style="gui_first",
    )
    score, tags = _score_model_ui_candidate(
        "SampleApp Download for Windows",
        "button",
        (300, 240, 650, 300),
        request=request,
    )
    assert score >= 70
    assert "download_like" in tags
    assert "target_like" in tags


def test_model_ui_candidate_scoring_prefers_body_download_button_over_header_link() -> None:
    request = StepRequest(
        user_prompt="mobaxterm 설치해줘",
        execution_style="gui_first",
    )
    body_score, body_tags = _score_model_ui_candidate(
        "Download now",
        "button",
        (1044, 1253, 1229, 1310),
        request=request,
        screen_size=(2560, 1440),
    )
    header_score, header_tags = _score_model_ui_candidate(
        "Download",
        "link",
        (1254, 230, 1408, 288),
        request=request,
        screen_size=(2560, 1440),
    )
    assert body_score > header_score
    assert "button_kind" in body_tags
    assert "generic_header_download_penalty" in header_tags


def test_model_ui_candidate_scoring_penalizes_browser_url_text_regardless_of_position() -> None:
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
    )
    body_score, _ = _score_model_ui_candidate(
        "Download FileZilla Client",
        "button",
        (920, 700, 1250, 780),
        request=request,
        screen_size=(1920, 1080),
    )
    url_score, url_tags = _score_model_ui_candidate(
        "https://filezilla-project.org/download.php?type=client",
        "input",
        (680, 460, 1540, 520),
        request=request,
        screen_size=(1920, 1080),
    )
    chrome_fragment_score, chrome_fragment_tags = _score_model_ui_candidate(
        "Download",
        "button",
        (990, 100, 1060, 130),
        request=request,
        screen_size=(2560, 1440),
    )
    assert body_score > url_score
    assert "browser_url_penalty" in url_tags
    assert chrome_fragment_score <= 0
    assert "browser_chrome_fragment_penalty" in chrome_fragment_tags
    assert service_module._looks_like_browser_url_text("https://filezilla-project.org/download.php?type=client") is True
    assert service_module._looks_like_browser_url_text("Download FileZilla Client") is False


def test_model_ui_candidate_scoring_penalizes_blog_search_results() -> None:
    request = StepRequest(
        user_prompt="mobaxterm을 설치해줘",
        execution_style="gui_first",
    )
    official_score, official_tags = _score_model_ui_candidate(
        "Download MobaXterm Home Edition (current version) - MobaXterm",
        "link",
        (300, 240, 900, 300),
        request=request,
    )
    blog_score, blog_tags = _score_model_ui_candidate(
        "[T00] MobaXterm 설치 및 MobaXterm 사용법 - 네이버 블로그",
        "link",
        (300, 360, 900, 420),
        request=request,
    )
    assert official_score > blog_score
    assert "community_article_penalty" in blog_tags


def test_model_ui_candidate_scoring_penalizes_korean_howto_download_results() -> None:
    request = StepRequest(
        user_prompt="메모잇 설치해줘",
        execution_style="gui_first",
        observation_text="search results | 공식 다운로드 windows",
    )
    direct_score, _ = _score_model_ui_candidate(
        "메모잇 PC 다운로드",
        "link",
        (300, 240, 760, 300),
        request=request,
    )
    howto_score, howto_tags = _score_model_ui_candidate(
        "메모잇 다운로드 방법, 사용자별 윈도우 설치법",
        "link",
        (300, 360, 900, 420),
        request=request,
    )
    assert direct_score > howto_score
    assert "community_article_penalty" in howto_tags


def test_model_ui_candidate_scoring_penalizes_install_buttons_during_download_only_step() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "This is a download-only step. Do not launch, silently install, or run the installer in this step.\n"
            "End this step only when the installer file exists in Downloads."
        ),
        execution_style="gui_first",
    )
    score, tags = _score_model_ui_candidate(
        "설치",
        "button",
        (850, 160, 870, 180),
        request=request,
        screen_size=(2560, 1440),
    )
    assert score <= 0
    assert "download_chunk_install_penalty" in tags


def test_model_ui_candidate_scoring_prefers_installer_dialog_controls() -> None:
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
    )
    ok_score, ok_tags = _score_model_ui_candidate("OK", "button", (1200, 720, 1320, 780), request=request)
    download_score, download_tags = _score_model_ui_candidate("Download", "button", (1600, 700, 1850, 780), request=request)
    assert ok_score > download_score
    assert "installer_dialog_control" in ok_tags
    assert "install_chunk_download_penalty" in download_tags


def test_model_ui_candidate_scoring_penalizes_taskbar_strip() -> None:
    request = StepRequest(
        user_prompt="메모잇 설치해줘",
        execution_style="gui_first",
    )
    page_score, page_tags = _score_model_ui_candidate(
        "메모잇 다운로드 (v1.93)",
        "button",
        (680, 460, 1240, 530),
        request=request,
        screen_size=(1920, 1080),
    )
    taskbar_score, taskbar_tags = _score_model_ui_candidate(
        "Microsoft Edge",
        "button",
        (820, 1010, 1100, 1060),
        request=request,
        screen_size=(1920, 1080),
    )
    assert page_score > taskbar_score
    assert "taskbar_penalty" in taskbar_tags
    assert "taskbar_penalty" not in page_tags


def test_model_ui_candidate_scoring_penalizes_store_results_for_non_store_task() -> None:
    request = StepRequest(
        user_prompt="메모잇 설치해줘",
        execution_style="gui_first",
    )
    official_score, official_tags = _score_model_ui_candidate(
        "메모잇 다운로드 (v1.93)",
        "button",
        (680, 460, 1240, 530),
        request=request,
        screen_size=(1920, 1080),
    )
    store_score, store_tags = _score_model_ui_candidate(
        "Microsoft Store",
        "link",
        (620, 180, 910, 230),
        request=request,
        screen_size=(1920, 1080),
    )
    assert official_score > store_score
    assert "store_result_penalty" in store_tags
    assert "store_result_penalty" not in official_tags


def test_model_ui_candidate_scoring_penalizes_browser_brand_panels_for_non_browser_task() -> None:
    request = StepRequest(
        user_prompt="메모잇 설치해줘",
        execution_style="gui_first",
    )
    official_score, official_tags = _score_model_ui_candidate(
        "메모잇 다운로드 (v1.93)",
        "button",
        (680, 460, 1240, 530),
        request=request,
        screen_size=(1920, 1080),
    )
    edge_score, edge_tags = _score_model_ui_candidate(
        "Microsoft Edge",
        "link",
        (1400, 180, 1840, 260),
        request=request,
        screen_size=(1920, 1080),
    )
    assert official_score > edge_score
    assert "browser_brand_penalty" in edge_tags
    assert "browser_brand_penalty" not in official_tags


def test_model_ui_candidate_scoring_penalizes_offtarget_search_results() -> None:
    request = StepRequest(
        user_prompt="메모잇 설치해줘",
        execution_style="gui_first",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=%EB%A9%94%EB%AA%A8%EC%9E%87%20%EA%B3%B5%EC%8B%9D%20%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20pc%20windows", expected_title_tokens=["메모잇"])',
            }
        },
    )
    official_score, official_tags = _score_model_ui_candidate(
        "메모잇 다운로드 (v1.93)",
        "link",
        (680, 460, 1240, 530),
        request=request,
        screen_size=(1920, 1080),
    )
    offtarget_score, offtarget_tags = _score_model_ui_candidate(
        "메모장 - Windows 에서 다운로드 및 설치",
        "link",
        (680, 560, 1240, 630),
        request=request,
        screen_size=(1920, 1080),
    )
    assert official_score > offtarget_score
    assert "offtarget_search_result_penalty" in offtarget_tags
    assert "offtarget_search_result_penalty" not in official_tags


def test_model_ui_candidate_parser_recovers_truncated_json_text() -> None:
    raw = '''```json
{
  "elements": [
    {"text": "Download", "kind": "button", "bbox": [100, 200, 220, 240]},
    {"text": "Install for Windows", "kind": "link", "bbox": [300, 260, 520, 310]},
    {"text": "unfinished'''
    elements = _model_ui_ocr_elements_from_text(raw)
    assert [item["text"] for item in elements] == ["Download", "Install for Windows"]
    assert elements[1]["bbox"] == [300.0, 260.0, 520.0, 310.0]


def test_model_ui_installer_recovery_uses_visible_candidates(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"screenshot_size":[1920,1080],"candidates":[{"text":"확인","click_point":[960,700],"reason_tags":["installer_dialog_control"]},{"text":"취소","click_point":[1100,700],"reason_tags":["installer_dialog_control"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )
    assert len(_installer_ui_candidates_from_observation(observation)) == 2
    assert _should_use_model_ui_candidates(request) is False
    assert _should_use_installer_ui_candidates(request) is False
    assert _should_use_model_ui_installer_recovery(request) is True
    code = _synthesized_model_ui_installer_recovery_code(request)
    assert "pyautogui.click(x, y)" in code
    assert "def ensure_windows_dpi_aware(" in code
    assert "ensure_windows_dpi_aware()" in code
    assert code.index("ensure_windows_dpi_aware()") < code.index("import pyautogui")
    assert code.index("ensure_windows_dpi_aware()") < code.index("pyautogui.click(x, y)")
    assert "os.walk(root" in code
    assert "_extract_archive(" in code
    assert "msiexec.exe" in code
    assert "CONTEXT_MARKER.write_text" in code
    assert "CONTEXT_PROMPT_KEY = " in code
    assert "CONTEXT_PROMPT_EXCERPT = " in code
    assert "def _process_exists(name):" in code
    assert "def _avoid_failsafe():" in code
    assert "def _launch_installed_exe(exe):" in code
    assert "def _installed_exe_score(path):" in code
    assert "launch installed executable:" in code
    assert "if _launch_installed_exe(existing):" in code
    assert "if _launch_installed_exe(current):" in code
    assert "if not _launch_installed_exe(final):" in code
    assert "sftp" in code
    assert "sorted(matches, key=lambda p: len(str(p)))" not in code
    assert "installer_launcher_pid=" in code
    assert "proc = subprocess.Popen([str(installer)], shell=False)" in code
    assert "no installer package available for installer recovery" in code
    assert "확인" in code
    assert "취소" not in code
    assert "/VERYSILENT" not in code
    assert "def _click_visible_controls_once():" in code
    assert "visible_controls_clicked = False" in code
    assert "_click_visible_controls_once()" in code


def test_model_ui_installer_recovery_keeps_sentence_form_installer_controls(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"screenshot_size":[1920,1080],"candidates":[{"text":"약관에 동의함","click_point":[860,640],"bbox":[820,620,980,660],"reason_tags":["installer_dialog_control","checkbox_kind"]},{"text":"설치 진행","click_point":[1000,700],"reason_tags":["installer_dialog_control"]},{"text":"취소","click_point":[1180,700],"reason_tags":["installer_dialog_control"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )

    code = _synthesized_model_ui_installer_recovery_code(request)

    assert '"text": "약관에 동의함"' in code
    assert '"point": [860, 640]' in code
    assert '"bbox": [820, 620, 980, 660]' in code
    assert "def _click_points_for_visible_control(item):" not in code
    assert "checkbox_gap" not in code
    assert "left - checkbox_gap" not in code
    assert "right + checkbox_gap" not in code
    assert '"text": "설치 진행"' in code
    assert '"point": [1000, 700]' in code
    assert "취소" not in code


def test_model_ui_installer_recovery_keeps_sentence_form_english_controls(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"screenshot_size":[1920,1080],"candidates":[{"text":"I agree to the license terms","click_point":[840,620],"reason_tags":["installer_dialog_control","checkbox_kind"]},{"text":"Continue setup","click_point":[1020,700],"reason_tags":["installer_dialog_control"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )

    code = _synthesized_model_ui_installer_recovery_code(request)

    assert '"text": "I agree to the license terms"' in code
    assert '"point": [840, 620]' in code
    assert '"text": "Continue setup"' in code
    assert '"point": [1020, 700]' in code


def test_model_ui_installer_recovery_prefers_teacher_selected_click(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"screenshot_size":[1920,1080],"candidates":[{"text":"다음","click_point":[1000,700],"reason_tags":["installer_dialog_control"]},{"text":"동의","click_point":[860,640],"reason_tags":["installer_dialog_control"]}]}"""
    prompt = (
        "Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.\n\n"
        'TEACHER_VISIBLE_INSTALLER_ACTIONS: {"actions":[{"action":"click","point":[860,640],"text":"동의"}]}'
    )
    request = StepRequest(
        user_prompt=prompt,
        execution_style="gui_first",
        observation_text=observation,
    )

    assert _teacher_visible_installer_clicks_from_prompt(prompt) == [
        {"text": "동의", "point": [860, 640], "tags": ["teacher_selected", "installer_dialog_control"]}
    ]
    code = _synthesized_model_ui_installer_recovery_code(request)

    assert '"teacher_selected"' in code
    assert code.index('"point": [860, 640]') < code.index('"point": [1000, 700]')


def test_installer_ui_candidates_observation_uses_cropped_installer_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = []

        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            if len(self.calls) > 2:
                return type(
                    "Result",
                    (),
                    {
                        "text": json.dumps(
                        {
                            "controls": [
                                {
                                    "control": {
                                        "kind": "checkbox",
                                        "bbox": [150, 450, 250, 550],
                                        "point": [200, 500],
                                        "confidence": 0.99,
                                    },
                                    "label": {"text": "동의함", "bbox": [260, 450, 420, 550]},
                                }
                            ]
                        },
                            ensure_ascii=False,
                        ),
                        "model_id": "fake-model",
                    },
                )()
            if len(self.calls) > 1:
                return type(
                    "Result",
                    (),
                    {
                        "text": json.dumps(
                            {
                                "controls": [
                                    {
                                        "control": {
                                            "kind": "checkbox",
                                            "bbox": [100, 300, 200, 700],
                                            "point": [150, 500],
                                            "confidence": 0.8,
                                        },
                                        "label": {"text": "다른 선택", "bbox": [200, 300, 500, 700]},
                                    },
                                    {
                                        "control": {
                                            "kind": "checkbox",
                                            "bbox": [400, 300, 600, 700],
                                            "point": [500, 500],
                                            "confidence": 0.96,
                                        },
                                        "label": {"text": "동의함", "bbox": [600, 300, 900, 700]},
                                    },
                                ]
                            },
                            ensure_ascii=False,
                        ),
                        "model_id": "fake-model",
                    },
                )()
            return type(
                "Result",
                (),
                {
                    "text": '{"elements":[{"text":"동의함","kind":"checkbox","bbox":[100,100,300,200],"point":[200,150],"confidence":0.96}]}',
                    "model_id": "fake-model",
                },
            )()

    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (1000, 800), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        screenshot_region={"left": 60, "top": 64, "right": 1060, "bottom": 864},
    )
    runtime = FakeRuntime()
    observation = _installer_ui_candidates_observation(
        runtime=runtime,
        request=request,
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
    )

    assert observation is not None
    assert "INSTALLER_VISIBLE_UI_CANDIDATES:" in observation
    assert "MODEL_VISIBLE_UI_CANDIDATES:" not in observation
    assert len(runtime.calls) >= 3
    assert runtime.calls[0]["image_bytes"] == buffer.getvalue()
    assert any(call["image_bytes"] != buffer.getvalue() for call in runtime.calls[1:])
    user_payload = json.loads(runtime.calls[0]["prompt_bundle"].user_prompt)
    assert "screen_size" not in user_payload
    assert "crop_region" not in user_payload
    assert "crop_size" not in user_payload
    assert "This image is the installer/dialog crop." in user_payload["instructions"]
    assert any("pixel coordinates relative to this cropped image" in item for item in user_payload["instructions"])
    assert (tmp_path / "responses" / "step-001.installer-ui-candidates.json").exists()
    candidates = _installer_ui_candidates_from_observation(observation)
    assert candidates[0]["text"] == "동의함"
    assert candidates[0]["click_point"] == [260, 214]
    assert candidates[0]["bbox"] == [160, 164, 360, 264]
    assert candidates[0]["coord_space"] == "screen_abs"
    assert candidates[0]["source_coord_space"] == "installer_crop_pixel"
    assert candidates[0]["source_region"] == {"left": 60, "top": 64, "right": 1060, "bottom": 864, "width": 1000, "height": 800}
    assert candidates[0]["source_image_variant"] == "full_installer:original"
    assert "installer_choice_control" in candidates[0]["reason_tags"]
    assert "final_refined_click_point" not in candidates[0]
    assert "final_refined_bbox" not in candidates[0]
    assert "installer_choice_control" in candidates[0]["reason_tags"]


def test_installer_ui_candidates_keeps_tiny_checkbox_bbox_when_point_exists(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **_kwargs):  # type: ignore[no-untyped-def]
            return type(
                "Result",
                (),
                {
                    "text": json.dumps(
                        {
                            "elements": [
                                {
                                    "text": "동의함",
                                    "kind": "checkbox",
                                    "bbox": [12, 720, 15, 735],
                                    "point": [12, 728],
                                    "confidence": 0.95,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "model_id": "fake-model",
                },
            )()

    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (878, 543), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        screenshot_region={"left": 842, "top": 413, "right": 1720, "bottom": 956},
    )

    observation = _installer_ui_candidates_observation(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        request=request,
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
    )

    assert observation is not None
    candidates = _installer_ui_candidates_from_observation(observation)
    assert len(candidates) == 1
    assert candidates[0]["text"] == "동의함"
    assert candidates[0]["kind"] == "checkbox"
    assert 850 <= candidates[0]["click_point"][0] <= 853
    assert candidates[0]["click_point"][1] >= 808
    assert 844 <= candidates[0]["bbox"][0] <= 847


def test_installer_ui_candidates_treats_in_crop_coordinates_as_pixels(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **_kwargs):  # type: ignore[no-untyped-def]
            return type(
                "Result",
                (),
                {
                    "text": json.dumps(
                        {
                            "elements": [
                                {
                                    "text": "동의함",
                                    "kind": "label",
                                    "bbox": [160, 400, 220, 460],
                                    "point": [190, 430],
                                    "confidence": 1.0,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "model_id": "fake-model",
                },
            )()

    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (878, 543), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        screenshot_region={"left": 842, "top": 413, "right": 1720, "bottom": 956},
    )

    observation = _installer_ui_candidates_observation(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        request=request,
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
    )

    assert observation is not None
    candidates = _installer_ui_candidates_from_observation(observation)
    assert len(candidates) == 1
    assert candidates[0]["text"] == "동의함"
    assert candidates[0]["click_point"] == [1032, 843]
    assert candidates[0]["bbox"] == [1002, 813, 1062, 873]
    assert candidates[0]["source_click_point"] == [190, 430]
    assert candidates[0]["source_bbox"] == [160, 400, 220, 460]


def test_installer_ui_candidates_visually_refines_checkbox_square_from_model_anchor(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    monkeypatch.setattr(
        service_module,
        "_visual_refine_installer_choice_candidate",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("visual refine should not run in installer OCR flow")),
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return type(
                "Result",
                (),
                {
                    "text": json.dumps(
                        {
                            "elements": [
                                {
                                    "text": "동의함",
                                    "kind": "checkbox",
                                    "bbox": [12, 756, 15, 774],
                                    "point": [15, 765],
                                    "confidence": 0.95,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "model_id": "fake-model",
                },
            )()

    from io import BytesIO
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (878, 543), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([44, 435, 63, 454], outline=(80, 80, 80), width=2)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        screenshot_region={"left": 842, "top": 413, "right": 1720, "bottom": 956},
    )
    runtime = FakeRuntime()

    observation = _installer_ui_candidates_observation(
        runtime=runtime,  # type: ignore[arg-type]
        request=request,
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
    )

    assert observation is not None
    candidates = _installer_ui_candidates_from_observation(observation)
    assert len(candidates) == 1
    assert runtime.calls >= 1
    assert any(
        tag in candidates[0]["reason_tags"]
        for tag in ("visual_choice_control", "installer_spatial_crop", "installer_choice_control")
    )
    assert (
        candidates[0].get("visual_refined_source_coord_space") == "installer_crop_pixel"
        or candidates[0].get("source_spatial_crop")
        or candidates[0].get("source_image_variant") == "full_installer:original"
    )
    x, y = candidates[0]["click_point"]
    assert 840 <= x <= 905
    assert 820 <= y <= 930
    assert candidates[0]["source_click_point"][0] >= 0
    assert candidates[0]["source_click_point"][1] >= 0


def test_installer_choice_auto_recrop_boxes_uses_quadrant_primary_direction() -> None:
    boxes = service_module._installer_choice_auto_recrop_boxes(point=(52, 416), crop_size=(877, 544))

    assert boxes == [
        ("auto_choice_center", (4, 368, 100, 464)),
        ("auto_choice_right_down", (52, 440, 148, 536)),
    ]


def test_installer_choice_auto_recrop_boxes_expands_both_axes_near_center() -> None:
    boxes = service_module._installer_choice_auto_recrop_boxes(point=(438, 272), crop_size=(877, 544))
    names = [name for name, _box in boxes]

    assert names == [
        "auto_choice_center",
        "auto_choice_left_up",
        "auto_choice_left_down",
        "auto_choice_right_up",
        "auto_choice_right_down",
    ]


def test_installer_spatial_ocr_crops_splits_installer_into_quadrants() -> None:
    boxes = service_module._installer_spatial_ocr_crops((878, 543))

    assert boxes == [
        ("full_installer", (0, 0, 878, 543)),
        ("top_left", (0, 0, 439, 271)),
        ("top_right", (439, 0, 878, 271)),
        ("bottom_left", (0, 271, 439, 543)),
        ("bottom_right", (439, 271, 878, 543)),
    ]


def test_installer_ui_candidates_keeps_related_blank_choice_points(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            prompt_bundle = kwargs.get("prompt_bundle")
            user_prompt = str(getattr(prompt_bundle, "user_prompt", ""))
            if '"crop_name": "bottom_left"' in user_prompt:
                text = json.dumps(
                    {
                        "controls": [
                            {
                                "control": {
                                    "kind": "checkbox",
                                    "bbox": [106, 214, 156, 264],
                                    "point": [131, 239],
                                    "confidence": 1.0,
                                },
                                "label": {"text": "동의함", "bbox": [168, 216, 218, 262]},
                            },
                            {
                                "control": {
                                    "kind": "checkbox",
                                    "bbox": [106, 598, 156, 648],
                                    "point": [131, 623],
                                    "confidence": 1.0,
                                },
                                "label": {"text": "", "bbox": [168, 600, 218, 646]},
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            elif '"spatial_crop": "bottom_left"' in user_prompt:
                text = json.dumps(
                    {
                        "elements": [
                            {
                                "text": "동의함",
                                "kind": "checkbox",
                                "bbox": [10, 260, 100, 300],
                                "point": [55, 280],
                                "confidence": 1.0,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            else:
                text = json.dumps({"elements": []}, ensure_ascii=False)
            return type(
                "Result",
                (),
                {
                    "text": text,
                    "model_id": "fake-model",
                },
            )()

    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (878, 543), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        screenshot_region={"left": 842, "top": 413, "right": 1720, "bottom": 956},
    )

    observation = _installer_ui_candidates_observation(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        request=request,
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
    )

    assert observation is not None
    candidates = _installer_ui_candidates_from_observation(observation)
    assert candidates
    assert candidates[0]["text"] == "동의함"
    assert candidates[0]["related_click_points"]
    assert any(895 <= item["point"][0] <= 902 and item["point"][1] == 853 for item in candidates[0]["related_click_points"])
    assert "installer_related_choice_controls" in candidates[0]["reason_tags"]


def test_installer_ui_candidates_adds_choice_ocr_for_full_installer_checkbox(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            prompt_bundle = kwargs.get("prompt_bundle")
            user_prompt = str(getattr(prompt_bundle, "user_prompt", ""))
            if '"crop_name": "full_installer"' in user_prompt:
                text = json.dumps(
                    {
                        "controls": [
                            {
                                "control": {
                                    "kind": "checkbox",
                                    "bbox": [100, 700, 140, 740],
                                    "point": [120, 720],
                                    "confidence": 1.0,
                                },
                                "label": {"text": "동의함", "bbox": [150, 700, 260, 740]},
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            elif '"spatial_crop": "full_installer"' in user_prompt:
                text = json.dumps(
                    {
                        "elements": [
                            {
                                "text": "동의함",
                                "kind": "checkbox",
                                "bbox": [50, 560, 180, 620],
                                "point": [110, 590],
                                "confidence": 1.0,
                            },
                            {
                                "text": "다음 >",
                                "kind": "button",
                                "bbox": [600, 840, 760, 900],
                                "point": [680, 870],
                                "confidence": 1.0,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            else:
                text = json.dumps({"elements": []}, ensure_ascii=False)
            return type("Result", (), {"text": text, "model_id": "fake-model"})()

    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (878, 543), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        screenshot_region={"left": 842, "top": 413, "right": 1720, "bottom": 956},
    )

    observation = _installer_ui_candidates_observation(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        request=request,
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
    )

    assert observation is not None
    candidates = _installer_ui_candidates_from_observation(observation)
    checkbox = next(item for item in candidates if item["text"] == "동의함")
    assert checkbox["source_spatial_crop"] == "full_installer"
    assert checkbox["choice_ocr_click_points"]
    assert checkbox["choice_ocr_click_points"][0]["point"] == [947, 804]


def test_model_ui_installer_recovery_keeps_model_checkbox_raw_click_point(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"candidates":[{"text":"동의함","kind":"checkbox","click_point":[891,834],"bbox":[853,826,930,842],"reason_tags":["installer_dialog_control","installer_choice_control"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_installer_recovery_code(request)

    assert '"text": "동의함"' in code
    assert '"point": [891, 834]' in code
    assert '"kind": "checkbox"' in code
    assert '"bbox": [853, 826, 930, 842]' in code
    assert "if kind in ('checkbox', 'radio') and isinstance(control_bbox, list)" not in code
    assert "geometry_valid = bool(item.get('refined_geometry_valid'))" not in code
    assert "refined_bbox if geometry_valid else []" not in code
    assert "refined_bbox = item.get('refined_bbox') or []" not in code
    assert "click_points = [" not in code
    assert "int(left + width * 0.7)" not in code
    assert "int(left + width * 0.3)" not in code
    assert "x = int((left + right) / 2)" not in code
    assert "y = int((top + bottom) / 2)" not in code
    assert "pyautogui.mouseDown()" not in code
    assert "pyautogui.mouseUp()" not in code
    assert "_click_point_for_visible_control" not in code


def test_model_ui_installer_recovery_prefers_refined_checkbox_click_point(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"candidates":[{"text":"동의함","kind":"checkbox","click_point":[891,834],"refined_click_point":[910,852],"refined_label_bbox":[910,840,980,864],"bbox":[853,826,930,842],"reason_tags":["installer_dialog_control","installer_choice_control"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_installer_recovery_code(request)

    assert '"point": [891, 834]' in code
    assert '"refined_click_point": [910, 852]' in code
    assert '"refined_label_bbox": [910, 840, 980, 864]' in code
    assert "fallback_point = item.get('refined_click_point') or item.get('point') or [0, 0]" in code
    assert "refined_label_bbox = item.get('refined_label_bbox') or []" not in code
    assert "if kind in ('checkbox', 'radio') and isinstance(control_bbox, list)" not in code


def test_model_ui_installer_recovery_keeps_recrop_click_points_local_to_candidate(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"candidates":[{"text":"동의함","kind":"checkbox","click_point":[894,828],"bbox":[850,820,910,850],"recrop_click_points":[{"crop_name":"auto_choice_center","point":[894,828],"match_score":110},{"crop_name":"auto_choice_right_down","point":[899,857],"match_score":110}],"reason_tags":["installer_dialog_control","installer_choice_control","ocr_recrop_choice_control"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_installer_recovery_code(request)

    assert '"recrop_click_points"' in code
    assert '"crop_name": "auto_choice_center"' in code
    assert '"crop_name": "auto_choice_right_down"' in code
    assert "for recrop_item in item.get('recrop_click_points') or []:" in code
    assert "point_attempts.append((fallback_point, 'primary'))" in code


def test_model_ui_installer_recovery_clicks_related_choice_points(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"candidates":[{"text":"동의함","kind":"checkbox","click_point":[899,766],"bbox":[850,740,930,780],"related_click_points":[{"crop_name":"bottom_left","point":[899,853],"source_click_point":[57,440]}],"reason_tags":["installer_dialog_control","installer_choice_control","installer_related_choice_controls"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_installer_recovery_code(request)

    assert '"related_click_points"' in code
    assert '"point": [899, 853]' in code
    assert "for related_item in item.get('related_click_points') or []:" in code


def test_model_ui_installer_recovery_uses_choice_ocr_points_only_for_choices(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """INSTALLER_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of a cropped installer/dialog UI region, not Windows OCR.
{"candidates":[{"text":"동의함","kind":"checkbox","click_point":[889,726],"recrop_click_points":[{"crop_name":"auto_choice_center","point":[889,726]}],"related_click_points":[{"crop_name":"bottom_left","point":[895,842]}],"choice_ocr_click_points":[{"crop_name":"bottom_left","point":[900,853]}],"reason_tags":["installer_dialog_control","installer_choice_control"]},{"text":"다음 >","kind":"button","click_point":[1435,902],"reason_tags":["installer_dialog_control"]}]}"""
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_installer_recovery_code(request)
    compile(code, "<generated>", "exec")

    assert '"choice_ocr_click_points"' in code
    assert "def _progression_button_attempts():" in code
    assert "clicked = _click_existing_progression_buttons(reason='before choice') or clicked" in code
    assert "choice_ocr_items = (item.get('choice_ocr_click_points') or []) if kind in ('checkbox', 'radio') else []" in code
    assert "point_attempts.append((choice_point, choice_item.get('crop_name') or 'choice_ocr'))" in code
    assert "if choice_ocr_items:" in code
    assert "else:" in code
    assert "_click_existing_progression_buttons(reason='after choice')" in code


def test_installer_choice_refinement_rejects_label_text_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **_kwargs):  # type: ignore[no-untyped-def]
            return type(
                "Result",
                (),
                {
                    "text": json.dumps(
                        {
                            "controls": [
                                {
                                    "control": {
                                        "kind": "checkbox",
                                        "bbox": [100, 100, 150, 150],
                                        "point": [125, 125],
                                        "confidence": 0.95,
                                    },
                                    "label": {"text": "고클린", "bbox": [160, 100, 300, 150]},
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "model_id": "fake-model",
                },
            )()

    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (200, 120), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    result = service_module._refine_installer_choice_candidate(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        request=StepRequest(user_prompt="고클린 설치해줘", execution_style="gui_first"),
        crop_bytes=buffer.getvalue(),
        crop_size=(200, 120),
        crop_left=842,
        crop_top=413,
        candidate={"text": "동의함", "kind": "checkbox", "bbox": [850, 450, 920, 470]},
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
        index=0,
    )

    assert result is None
    debug = json.loads((tmp_path / "responses" / "step-001.installer-ui-refine-00.json").read_text(encoding="utf-8"))
    assert debug["refined"]["refinement_rejected"]["reason"] == "label_text_mismatch"


def test_installer_choice_refinement_can_use_enhanced_variant(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = []

        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            payload = json.loads(kwargs["prompt_bundle"].user_prompt)
            label_text = "동의함" if payload.get("image_variant") == "enhanced" else "다른 선택"
            return type(
                "Result",
                (),
                {
                    "text": json.dumps(
                        {
                            "controls": [
                                {
                                    "control": {
                                        "kind": "checkbox",
                                        "bbox": [100, 300, 200, 500],
                                        "point": [150, 400],
                                        "confidence": 0.95,
                                    },
                                    "label": {"text": label_text, "bbox": [240, 300, 650, 500]},
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "model_id": "fake-model",
                },
            )()

    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (120, 90), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    runtime = FakeRuntime()
    result = service_module._refine_installer_choice_candidate(
        runtime=runtime,  # type: ignore[arg-type]
        request=StepRequest(user_prompt="고클린 설치해줘", execution_style="gui_first"),
        crop_bytes=buffer.getvalue(),
        crop_size=(120, 90),
        crop_left=10,
        crop_top=20,
        candidate={"text": "동의함", "kind": "checkbox", "bbox": [30, 40, 50, 55]},
        max_new_tokens=256,
        generation_context={"run_dir": tmp_path, "step_id": "step-001"},
        index=0,
    )

    assert result is not None
    assert len(runtime.calls) == 2
    assert result["refinement_image_variant"] == "enhanced"
    assert result["refined_coord_space"] == "screen_abs"
    assert result["refined_source_coord_space"] == "installer_refine_crop_pixel"


def test_installer_ui_candidates_observation_requires_executor_region(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64="iVBORw0KGgo=",
    )

    assert _installer_ui_candidates_observation(
        runtime=object(),  # type: ignore[arg-type]
        request=request,
        max_new_tokens=256,
        generation_context=None,
    ) is None


def test_installer_launcher_pid_is_extracted_from_last_execution_stdout() -> None:
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        last_execution={"stdout_tail": "launch installer target: C:\\Downloads\\app.exe\ninstaller_launcher_pid=4321\n"},
    )

    assert _installer_launcher_pid_from_last_execution(request) == 4321


def test_installer_execution_region_prefers_launcher_pid_over_stale_region(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64="iVBORw0KGgo=",
        screenshot_region={"left": 10, "top": 20, "right": 110, "bottom": 220},
        last_execution={"stdout_tail": "installer_launcher_pid: 9876"},
    )

    assert _execution_screenshot_region_for_request(request) == {
        "mode": "installer_window",
        "expected_pid": 9876,
    }


def test_visible_flow_extra_targets_ignores_continuity_instruction_words() -> None:
    request = StepRequest(
        user_prompt=(
            "Use executable Python only. "
            "First read the soft continuity file `~/Downloads/computer-use-agent-context.json`; "
            "if it exists, reuse it. "
            "Source task: 메모잇 설치해줘. "
            "Do not trust unrelated installers."
        ),
        execution_style="gui_first",
    )
    keywords = _visible_flow_extra_targets(request, limit=6)
    assert "메모잇" in keywords
    assert "reading" not in keywords
    assert "soft" not in keywords
    assert "continuity" not in keywords
    assert "file" not in keywords
    assert "they" not in keywords
    assert "exist" not in keywords


def test_registrable_host_ignores_malformed_url() -> None:
    assert _registrable_host_from_url("http://[not-a-valid-ipv6") == ""


def test_url_looks_like_search_results_ignores_malformed_url() -> None:
    assert _url_looks_like_search_results("http://[not-a-valid-ipv6") is False


def test_fallback_browser_search_url_from_parts_ignores_malformed_url() -> None:
    url = _fallback_browser_search_url_from_parts(["kakaotalk"], ["http://[not-a-valid-ipv6"])

    assert url is not None
    assert "kakaotalk" in url


def test_fallback_browser_search_url_from_parts_avoids_exact_failed_query() -> None:
    url = _fallback_browser_search_url_from_parts(
        ["filezilla"],
        [],
        excluded_queries=["filezilla windows"],
    )

    decoded = urllib.parse.unquote(url or "")
    assert decoded
    assert "filezilla" in decoded
    assert "q=filezilla+windows" not in decoded


def test_fallback_browser_search_url_for_request_avoids_exact_failed_query_sentence() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: filezilla.\n"
            "Previous stdout summary: search failed at https://www.google.com/search?q=filezilla+windows\n"
            "Previous stderr summary: no visible download candidates.\n"
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla+windows", expected_title_tokens=["filezilla"])',
            },
        },
    )

    url = _fallback_browser_search_url_for_request(request, extra_targets=["filezilla"])
    decoded = urllib.parse.unquote(url or "")

    assert decoded
    assert "filezilla" in decoded
    assert "q=filezilla+windows" not in decoded


def test_model_ui_browser_prelude_opens_prompt_url_without_ocr_helper(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Open https://vendor.example/download and download the Windows installer.",
        execution_style="gui_first",
    )
    assert _should_use_model_ui_browser_prelude(request) is True
    code = _synthesized_model_ui_browser_prelude_code(request)
    assert "open_url_and_wait(" in code
    assert "using existing installer from Downloads before opening browser" in code
    assert "using previously downloaded artifact before opening browser" in code
    assert "def _path_matches_reuse_target(path_text):" in code
    assert "import re" in code
    assert "ignoring continuity artifact that does not match target filename/path" in code
    assert "require_target_match=True" in code
    assert "if getattr(exc, 'code', exc) in (0, None):" in code
    assert code.index("wait_for_recent_download_artifact(") < code.index("open_url_and_wait(")
    assert code.index("write_action_context(") < code.index("open_url_and_wait(")
    assert "click_text_targets(" not in code
    assert "ocr_screen_text_regions(" not in code


def test_model_ui_browser_prelude_only_runs_on_first_step(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Open https://vendor.example/download and download the Windows installer.",
        execution_style="gui_first",
        step_index=1,
    )
    assert _should_use_model_ui_browser_prelude(request) is False


def test_model_ui_browser_prelude_fallback_search_uses_task_tokens(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Use executable Python only. Do not download anything in this chunk. "
            "If no installer UI is visible yet, find the existing installer `.exe` or `.msi` in `%USERPROFILE%\\\\Downloads\\\\`, "
            "launch it once, and then continue from the resulting installer UI. "
            "Use executable Python on the Windows machine to locate the already-installed app executable "
            "for the target app from this task: mobaxterm을 설치해줘, prefer reading `~/Downloads/install-success.json` first."
        ),
        execution_style="gui_first",
    )
    url = _fallback_browser_search_url_for_request(request, extra_targets=["mobaxterm"])
    assert url is not None
    assert "mobaxterm" in url
    assert "execution" not in url
    assert "download-like" not in url


def test_model_ui_browser_prelude_not_selected_for_install_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Use executable Python only. Do not download anything in this chunk. "
            "First inspect the current screenshot and desktop state for an installer wizard, UAC prompt, license dialog, "
            "destination dialog, or completion dialog, and drive that visible UI forward if present. "
            "If no installer UI is visible yet, find the existing installer `.exe` or `.msi` in `%USERPROFILE%\\\\Downloads\\\\`, "
            "launch it once, and then continue from the resulting installer UI. "
            "Do only this chunk. Do not skip ahead to later chunks."
        ),
        execution_style="gui_first",
    )
    assert _looks_like_existing_installer_launch_task(request.user_prompt) is True
    assert _should_use_model_ui_browser_prelude(request) is False


def test_model_ui_browser_prelude_not_selected_for_verified_installer_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Use Python to locate the FileZilla installer in Downloads, start it with subprocess.Popen, "
            "and complete the Windows setup using default options unless the installer requires a straightforward confirmation. "
            "After installation, launch FileZilla Client and confirm it starts successfully without errors.\n\n"
            "Current chunk success target: FileZilla is installed and the client process starts successfully.\n\n"
            "Preconditions expected before or during this chunk:\n"
            "- A valid FileZilla installer .exe already exists in ~/Downloads.\n\n"
            "Previously verified installer artifacts on the target machine. Prefer these exact installer paths before searching Downloads broadly again:\n"
            "- `C:\\Users\\user\\Downloads\\FileZilla_3.70.4_win64_sponsored2-setup.exe`"
        ),
        execution_style="gui_first",
    )
    assert _looks_like_existing_installer_launch_task(request.user_prompt) is True
    assert _should_use_model_ui_browser_prelude(request) is False


def test_model_ui_download_recovery_uses_visible_candidates(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """MODEL_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of the latest screenshot, not Windows OCR.
{"screenshot_size":[1920,1080],"candidates":[
  {"text":"Memoit 다운로드 (v1.93)","click_point":[920,540],"reason_tags":["download_like","target_like"],"score":92},
  {"text":"북마크","click_point":[120,80],"reason_tags":["toolbar_text_penalty"],"score":12}
]}"""
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=observation,
    )
    assert _should_use_model_ui_download_recovery(request) is True
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "pyautogui.click(x, y)" in code
    assert "pyautogui.FAILSAFE = False" in code
    assert "def ensure_windows_dpi_aware(" in code
    assert "ensure_windows_dpi_aware()" in code
    assert code.index("ensure_windows_dpi_aware()") < code.index("import pyautogui")
    assert code.index("ensure_windows_dpi_aware()") < code.index("pyautogui.click(x, y)")
    assert "def _candidate_points(candidate):" in code
    assert "for delta_x in (-80, 80):" in code
    assert "return deduped[:4]" in code
    assert "def _candidate_identity(candidate):" in code
    assert "def _same_candidate(candidate, remembered):" in code
    assert "if abs(int(candidate_point[0]) - int(remembered_point[0])) <= 12" in code
    assert "return bool(candidate_text and remembered_text and candidate_text == remembered_text)" in code
    assert "def _path_matches_reuse_target(path_text):" in code
    assert "def _current_visible_signature(candidates):" in code
    assert "def _recent_attempted_candidates():" in code
    assert "if installer_path and _path_matches_reuse_target(installer_path):" in code
    assert "def _candidate_attempt_count(candidate, attempted_candidates):" in code
    assert "def _record_clicked_candidate(candidate, point):" in code
    assert "visible_click_history=history[-8:]" in code
    assert "visible_candidate_signature=_current_visible_signature(VISIBLE_CANDIDATES)" in code
    assert "CLICK_CANDIDATES = [candidate for candidate in VISIBLE_CANDIDATES if not any(_same_candidate(candidate, attempted) for attempted in ATTEMPTED_CANDIDATES)]" in code
    assert "_candidate_attempt_count(candidate, ATTEMPTED_CANDIDATES) < 2" in code
    assert "for candidate_index, candidate in enumerate(CLICK_CANDIDATES[:3], start=1):" in code
    assert "for point_index, (x, y) in enumerate(points, start=1):" in code
    assert "points = _candidate_points(candidate)" in code
    assert "_record_clicked_candidate(candidate, points[0])" in code
    assert "wait_for_recent_download_artifact(" in code
    assert "def _wait_for_download_progress(since_ts, timeout_s=6.0):" in code
    assert "def _progress_download_targets(progress_path):" in code
    assert "def _is_success_exit(exc):" in code
    assert "download progress detected:" in code
    assert "extra_targets=None" not in code
    assert "require_target_match=True" in code
    assert "downloaded artifact does not match target filename/path" in code
    assert "recent download already present" in code
    assert "since_ts=(time.time() - 30.0)" in code
    assert "timeout_s=3.0" in code
    assert "timeout_s=45.0" in code
    progress_hint_index = code.index("progress_targets = _progress_download_targets(progress)")
    rescan_index = code.index("download = wait_for_recent_download_artifact(", progress_hint_index)
    assert progress_hint_index < rescan_index
    assert "extra_targets=[*progress_targets, *TARGET_TERMS]" in code
    assert "if _is_success_exit(exc):" in code
    assert "def _is_partial_progress_exit(exc):" in code
    assert "clicked all grounded points for all visible candidates without a stable download" in code
    assert "FALLBACK_ALTERNATE_SEARCH_URLS = " in code
    assert "def _isolated_recovery_urls():" in code
    assert "def _try_download_from_isolated_recovery_page():" in code
    assert "retry_source_urls = _isolated_recovery_urls()" in code
    assert "for candidate in [PROMPT_URL, FALLBACK_SEARCH_URL, *FALLBACK_ALTERNATE_SEARCH_URLS]:" in code
    assert "return urls[:2]" in code
    assert "RECOVERY_URL_ATTEMPTS = set()" in code
    assert "def _search_query_fingerprint(query):" in code
    assert "failed_recovery_search_queries=failed_queries[-16:]" in code
    assert "failed_recovery_hosts=failed_hosts[-16:]" in code
    assert "skipping previously failed recovery URL/query/host:" in code
    assert "_record_failed_recovery_url(retry_source_url, exc)" in code
    assert "if retry_source_url in RECOVERY_URL_ATTEMPTS:" in code
    assert "skipping repeated isolated recovery page:" in code
    assert "open_url_and_wait(retry_source_url" in code
    assert "opened isolated recovery page:" in code
    assert "download_official_installer_from_page(retry_source_url, extra_targets=TARGET_TERMS" in code
    assert "download recovered from isolated recovery page" in code
    assert "isolated recovery page recovery failed:" in code
    assert "continue with latest screenshot and model-visible UI candidates after exhausting isolated recovery pages" in code
    assert "isolated recovery page did not find a stable installer: {exc}')\n            _record_failed_recovery_url(retry_source_url, exc)\n            continue" in code
    assert (
        "raise SystemExit('continue with latest screenshot and model-visible UI candidates after exhausting isolated recovery pages')"
        in code
    )
    assert "\n    recovered = _try_download_from_isolated_recovery_page()" in code
    assert "\n        recovered = _try_download_from_isolated_recovery_page()" not in code
    assert "def _colored_cta_points():" in code
    assert "def _try_colored_cta_download(since_ts):" in code
    assert "click colored CTA candidate" in code
    assert "continue with latest screenshot and model-visible UI candidates after clicking a colored download CTA candidate" in code
    assert "pyautogui.hotkey('ctrl', 'l')" not in code
    assert "no visible download-related control remains on the current screen" in code
    assert "no new visible download-related control is available on the current screen" in code
    assert "Memoit 다운로드" in code
    assert "CONTEXT_PROMPT_KEY = null" not in code
    assert "CONTEXT_PROMPT_KEY = \"\"" not in code
    assert "북마크" not in code
    assert "null" not in code


def test_model_ui_download_recovery_path_match_accepts_shared_product_root(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: 카카오톡, kakaocorp, stay, tab, guessed, irrelevant, blocked, broken.\n"
            "Use the visible download button and wait for a Windows installer in Downloads."
        ),
        execution_style="gui_first",
        observation_text="""MODEL_VISIBLE_UI_CANDIDATES:
{"screenshot_size":[1920,1080],"candidates":[
  {"text":"카카오톡 다운로드","click_point":[1062,173],"reason_tags":["download_like","target_like"],"score":93}
]}""",
    )

    target_terms = _visible_flow_extra_targets(request, limit=8)
    assert "카카오톡" in target_terms
    assert "stay" not in target_terms
    assert "tab" not in target_terms
    assert "guessed" not in target_terms
    assert "irrelevant" not in target_terms
    assert "blocked" not in target_terms
    assert "broken" not in target_terms

    code = _synthesized_model_ui_download_recovery_code(request)
    assert "if shared >= 5:" in code
    helper_source = "def _path_matches_reuse_target(path_text):" + code.split(
        "def _path_matches_reuse_target(path_text):", 1
    )[1].split("\ndef _normalize_search_query_text", 1)[0]
    namespace = {"TARGET_TERMS": ["카카오톡", "kakaocorp"], "re": re}
    exec(helper_source, namespace)

    assert namespace["_path_matches_reuse_target"](r"C:\Users\user\Downloads\KakaoTalk_Setup.exe")
    assert not namespace["_path_matches_reuse_target"](r"C:\Users\user\Downloads\Git-2.53.0-64-bit.exe")


def test_search_url_exclusion_treats_platform_only_variants_as_repeated() -> None:
    assert _search_url_matches_excluded_query(
        "https://www.google.com/search?q=filezilla%20windows%20pc",
        excluded_queries=["filezilla windows"],
    )
    assert _search_url_validation_error(
        "https://www.google.com/search?q=filezilla%20pc%20desktop%20x64",
        ["filezilla"],
        excluded_queries=["filezilla windows"],
    ) == "repeated_query"
    assert _fallback_browser_search_url_from_parts(
        ["filezilla"],
        [],
        excluded_queries=["filezilla windows"],
    ) == "https://www.google.com/search?q=filezilla%20windows%20kr"


def test_model_ui_download_recovery_excludes_browser_url_candidate(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """MODEL_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of the latest screenshot, not Windows OCR.
{"screenshot_size":[1920,1080],"candidates":[
  {"text":"https://filezilla-project.org/download.php?type=client","click_point":[980,92],"reason_tags":["download_like","browser_url_penalty"],"score":88},
  {"text":"Download FileZilla Client","click_point":[1110,754],"reason_tags":["download_like","target_like","button_shape"],"score":93}
]}"""
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "Download FileZilla Client" in code
    assert "https://filezilla-project.org/download.php?type=client" not in code


def test_model_ui_download_recovery_keeps_target_only_backup_candidate_when_download_candidate_visible(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """MODEL_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of the latest screenshot, not Windows OCR.
{"screenshot_size":[2560,1440],"candidates":[
  {"text":"Download FileZilla Client for Windows (64bit x64)","click_point":[1027,666],"reason_tags":["download_like","target_like","button_shape"],"score":83},
  {"text":"FileZilla - The free FTP solution","click_point":[942,436],"reason_tags":["clickable_kind","target_like","button_shape"],"score":48},
  {"text":"How to Install FileZilla on Windows","click_point":[1198,1026],"reason_tags":["download_like","install_like","target_like","community_article_penalty","button_shape"],"score":38}
]}"""
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "'text': 'Download FileZilla Client for Windows (64bit x64)'" in code
    assert "'text': 'FileZilla - The free FTP solution'" in code


def test_model_ui_download_recovery_avoids_reusing_current_page_url_when_no_candidates(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text="MODEL_VISIBLE_UI_CANDIDATES: []\nNo target-matching model-visible UI candidates were extracted from the latest search-results screenshot.",
    )
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "download_official_installer_from_page(current_url" not in code
    assert "PROMPT_URL or FALLBACK_SEARCH_URL" in code


def test_partial_download_page_navigation_detected_from_last_execution() -> None:
    assert _looks_like_partial_download_page_navigation(
        {
            "stdout_tail": (
                "visible candidate opened page: https://filezilla-project.org/download.php?type=client\n"
                "opened browser page for screenshot-grounded UI continuation"
            ),
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates",
        }
    ) is True


def test_exhausted_visible_download_recovery_detected_from_grounded_click_exhaustion() -> None:
    assert _looks_like_exhausted_visible_download_recovery(
        {
            "stdout_tail": (
                "click visible download candidate[1/1]: Download FileZilla Client at (1110, 754)\n"
                "candidate point 4 did not finish download yet: recent installer download did not appear\n"
                "opened isolated recovery page: https://www.google.com/search?q=filezilla%20windows"
            ),
            "stderr_tail": "clicked all grounded points for all visible candidates without a stable download: recent installer download did not appear",
        }
    ) is True


def test_model_ui_download_recovery_does_not_clear_click_history_for_stale_context_path() -> None:
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download the FileZilla Windows installer.",
        execution_style="gui_first",
        observation_text='MODEL_UI_ELEMENTS_JSON: {"elements":[{"text":"Download FileZilla Client","kind":"button","bbox":[800,700,1000,760]}]}',
    )
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "installer_path = str(payload.get('installer_path') or '').strip()" in code
    assert "if installer_path and _path_matches_reuse_target(installer_path):" in code
    assert "return []" in code


def test_fallback_browser_search_url_ignores_previous_stdout_artifact_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Use the current visible browser page for FileZilla.\n"
            "Use these exact official page URLs first before any search engine result or inferred domain:\n"
            "- https://filezilla-project.org/download.php?type=client\n"
            "Previous stdout summary: ignoring continuity artifact that does not match target filename/path: "
            "C:\\Users\\user\\Downloads\\npp.8.9.1.Installer.x64.exe\n"
        ),
        execution_style="gui_first",
    )
    url = _fallback_browser_search_url_for_request(request, extra_targets=["filezilla", "project"])
    assert url is not None
    assert "filezilla" in url
    assert "npp" not in url


def test_model_ui_download_recovery_merges_split_download_candidates(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """MODEL_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of the latest screenshot, not Windows OCR.
{"screenshot_size":[2560,1440],"candidates":[
  {"text":"다운로드","click_point":[1375,750],"bbox":[1350,740,1400,760],"reason_tags":["download_like","button_shape"],"score":61},
  {"text":"다운로드","click_point":[1425,750],"bbox":[1400,740,1450,760],"reason_tags":["download_like","button_shape"],"score":61},
  {"text":"다운로드","click_point":[1475,750],"bbox":[1450,740,1500,760],"reason_tags":["download_like","button_shape"],"score":61}
]}"""
    request = StepRequest(
        user_prompt="This is a download-only step. End this step only when the installer file exists in Downloads.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "'bbox': [1350, 740, 1500, 760]" in code
    assert "'point': [1425, 750]" in code
    assert code.count("'text': '다운로드'") == 1
    assert "lower_y = min(max(5, bottom - 6), center_y + min(20, max(8, int((bottom - top) * 0.25))))" in code


def test_model_ui_download_recovery_keeps_purchase_like_candidate_when_visible(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    observation = """MODEL_VISIBLE_UI_CANDIDATES:
These candidates come from local model visual extraction of the latest screenshot, not Windows OCR.
{"screenshot_size":[2560,1440],"candidates":[
  {"text":"Buy FileZilla Pro Single Device","click_point":[979,540],"reason_tags":["clickable_kind","target_like","button_shape"],"score":48},
  {"text":"Buy FileZilla Pro Multiple Devices","click_point":[2182,540],"reason_tags":["clickable_kind","target_like","button_shape"],"score":48},
  {"text":"Windows","click_point":[1344,792],"reason_tags":["clickable_kind","download_like","button_shape"],"score":61}
]}"""
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=observation,
    )
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "'text': 'Windows'" in code


def test_model_ui_candidates_search_results_retry_recovers_page_body_results(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return type(
                    "Result",
                    (),
                    {
                        "text": '{"elements":[{"text":"Microsoft Edge","kind":"link","bbox":[270,220,350,250],"confidence":0.95}]}',
                        "model_id": "fake-model",
                    },
                )()
            return type(
                "Result",
                (),
                {
                    "text": '{"elements":[{"text":"메모잇 - 바탕화면 메모장 포스트잇","kind":"link","bbox":[220,250,620,320],"confidence":0.98}]}',
                    "model_id": "fake-model",
                },
            )()

    screenshot_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x07\x80"
        b"\x00\x00\x04\x38"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    request = StepRequest(
        user_prompt="메모잇 설치해줘",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(screenshot_bytes).decode("ascii"),
        step_index=1,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=%EB%A9%94%EB%AA%A8%EC%9E%87%20%EA%B3%B5%EC%8B%9D%20%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20pc%20windows", expected_title_tokens=["메모잇"])',
            }
        },
    )
    runtime = FakeRuntime()
    observation = _model_ui_candidates_observation(
        runtime=runtime,
        request=request,
        max_new_tokens=256,
        generation_context=None,
    )
    assert runtime.calls == 2
    assert observation is not None
    assert "메모잇 - 바탕화면 메모장 포스트잇" in observation


def test_model_ui_candidates_search_results_retry_runs_for_low_signal_candidates(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return type(
                    "Result",
                    (),
                    {
                        "text": '{"elements":[{"text":"Microsoft","kind":"link","bbox":[276,220,313,245],"confidence":0.95}]}',
                        "model_id": "fake-model",
                    },
                )()
            return type(
                "Result",
                (),
                {
                    "text": '{"elements":[{"text":"메모잇 다운로드","kind":"link","bbox":[220,250,420,320],"confidence":0.98}]}',
                    "model_id": "fake-model",
                },
            )()

    screenshot_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x07\x80"
        b"\x00\x00\x04\x38"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    request = StepRequest(
        user_prompt="메모잇 설치해줘",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(screenshot_bytes).decode("ascii"),
        step_index=1,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=%EB%A9%94%EB%AA%A8%EC%9E%87%20%EA%B3%B5%EC%8B%9D%20%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20pc%20windows", expected_title_tokens=["메모잇"])',
            }
        },
    )
    runtime = FakeRuntime()
    observation = _model_ui_candidates_observation(
        runtime=runtime,
        request=request,
        max_new_tokens=256,
        generation_context=None,
    )
    assert runtime.calls == 2
    assert observation is not None
    assert "메모잇 다운로드" in observation


def test_model_ui_candidates_search_results_retry_runs_for_offtarget_download_candidates(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return type(
                    "Result",
                    (),
                    {
                        "text": '{"elements":[{"text":"파일질마 다운로드센터","kind":"link","bbox":[276,312,396,336],"confidence":0.95}]}',
                        "model_id": "fake-model",
                    },
                )()
            return type(
                "Result",
                (),
                {
                    "text": '{"elements":[{"text":"FileZilla - FTP Client filezilla.kr","kind":"link","bbox":[276,312,526,338],"confidence":0.98}]}',
                    "model_id": "fake-model",
                },
            )()

    screenshot_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x0a\x00"
        b"\x00\x00\x05\xa0"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(screenshot_bytes).decode("ascii"),
        step_index=1,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20pc%20desktop%20kr", expected_title_tokens=["filezilla"])',
            }
        },
    )
    runtime = FakeRuntime()
    observation = _model_ui_candidates_observation(
        runtime=runtime,
        request=request,
        max_new_tokens=256,
        generation_context=None,
    )
    assert runtime.calls == 2
    assert observation is not None
    assert "FileZilla - FTP Client filezilla.kr" in observation


def test_search_results_score_prefers_target_domain_over_side_panel_download() -> None:
    request = StepRequest(
        user_prompt="filezilla kr설치해줘",
        execution_style="gui_first",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20kr")',
            }
        },
    )
    assert _visible_flow_extra_targets(request, limit=8) == ["filezilla"]

    side_panel_score, side_panel_tags = _score_model_ui_candidate(
        "Download FileZilla Client for Windows (64-bit)",
        "link",
        (1715, 475, 2048, 518),
        request=request,
        screen_size=(2560, 1440),
    )
    domain_score, domain_tags = _score_model_ui_candidate(
        "FileZilla - FTP Client filezilla.kr",
        "link",
        (707, 449, 1347, 487),
        request=request,
        screen_size=(2560, 1440),
    )

    assert "search_result_side_panel_penalty" in side_panel_tags
    assert "search_result_domain_like" in domain_tags
    assert domain_score > side_panel_score


def test_model_ui_candidates_promotes_hangul_search_alias_when_direct_ascii_target_exists(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            return type(
                "Result",
                (),
                {
                    "text": (
                        '{"elements":['
                        '{"text":"FileZilla - FTP Client filezilla.kr","kind":"link","bbox":[276,312,526,338],"confidence":0.98},'
                        '{"text":"파일질라 다운로드","kind":"link","bbox":[276,410,460,440],"confidence":0.95},'
                        '{"text":"무료 다운로드","kind":"link","bbox":[276,500,420,530],"confidence":0.95}'
                        "]}"
                    ),
                    "model_id": "fake-model",
                },
            )()

    screenshot_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x0a\x00"
        b"\x00\x00\x05\xa0"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(screenshot_bytes).decode("ascii"),
        step_index=1,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20kr")',
            }
        },
    )

    observation = _model_ui_candidates_observation(
        runtime=FakeRuntime(),
        request=request,
        max_new_tokens=256,
        generation_context=None,
    )
    candidates = _model_ui_candidates_from_observation(observation)
    alias = next(item for item in candidates if item["text"] == "파일질라 다운로드")
    generic = next(item for item in candidates if item["text"] == "무료 다운로드")

    assert "target_alias_like" in alias["reason_tags"]
    assert "target_like" in alias["reason_tags"]
    assert "offtarget_search_result_penalty" not in alias["reason_tags"]
    assert "target_alias_like" not in generic["reason_tags"]
    assert "offtarget_search_result_penalty" in generic["reason_tags"]
    assert alias["score"] > generic["score"]


def test_model_ui_candidates_promotes_latin_search_alias_when_direct_hangul_target_exists(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            return type(
                "Result",
                (),
                {
                    "text": (
                        '{"elements":['
                        '{"text":"파일질라 다운로드","kind":"link","bbox":[276,312,526,338],"confidence":0.98},'
                        '{"text":"Download FileZilla Client for Windows","kind":"link","bbox":[276,410,620,440],"confidence":0.95},'
                        '{"text":"Download for Windows","kind":"link","bbox":[276,500,520,530],"confidence":0.95}'
                        "]}"
                    ),
                    "model_id": "fake-model",
                },
            )()

    screenshot_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x0a\x00"
        b"\x00\x00\x05\xa0"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    request = StepRequest(
        user_prompt="파일질라 설치해줘",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(screenshot_bytes).decode("ascii"),
        step_index=1,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=%ED%8C%8C%EC%9D%BC%EC%A7%88%EB%9D%BC")',
            }
        },
    )

    observation = _model_ui_candidates_observation(
        runtime=FakeRuntime(),
        request=request,
        max_new_tokens=256,
        generation_context=None,
    )
    candidates = _model_ui_candidates_from_observation(observation)
    alias = next(item for item in candidates if item["text"] == "Download FileZilla Client for Windows")
    generic = next(item for item in candidates if item["text"] == "Download for Windows")

    assert "target_alias_like" in alias["reason_tags"]
    assert "target_like" in alias["reason_tags"]
    assert "offtarget_search_result_penalty" not in alias["reason_tags"]
    assert "target_alias_like" not in generic["reason_tags"]
    assert "offtarget_search_result_penalty" in generic["reason_tags"]
    assert alias["score"] > generic["score"]


def test_model_ui_candidates_drops_zero_confidence_dummy(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)

    class FakeRuntime:
        def generate_text(self, **kwargs):  # type: ignore[no-untyped-def]
            return type(
                "Result",
                (),
                {
                    "text": '{"elements":[{"text":"filezilla","kind":"text","bbox":[1000,1000,1000,1000],"confidence":0.0}]}',
                    "model_id": "fake-model",
                },
            )()

    screenshot_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x0a\x00"
        b"\x00\x00\x05\xa0"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        screenshot_base64=base64.b64encode(screenshot_bytes).decode("ascii"),
        step_index=1,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20kr")',
            }
        },
    )
    observation = _model_ui_candidates_observation(
        runtime=FakeRuntime(),
        request=request,
        max_new_tokens=256,
        generation_context=None,
    )
    assert observation is not None
    assert "MODEL_VISIBLE_UI_CANDIDATES: []" in observation


def test_model_ui_candidates_continue_after_first_step(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="sampleapp 프로그램을 설치해줘",
        execution_style="gui_first",
        screenshot_base64="iVBORw0KGgo=",
        step_index=1,
    )
    assert _should_use_model_ui_candidates(request) is True


def test_model_ui_download_recovery_not_selected_for_install_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        observation_text='MODEL_VISIBLE_UI_CANDIDATES: {"candidates":[{"text":"확인","click_point":[900,700],"reason_tags":["installer_dialog_control"]}]}',
    )
    assert _should_use_model_ui_download_recovery(request) is False


def test_model_ui_download_recovery_not_selected_immediately_after_browser_open(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=1,
        observation_text='MODEL_VISIBLE_UI_CANDIDATES: {"candidates":[{"text":"메모잇","click_point":[900,700],"reason_tags":["target_like"]}]}',
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=sampleapp", expected_title_tokens=["sampleapp"])',
            }
        },
    )
    assert _should_use_model_ui_download_recovery(request) is False


def test_model_ui_download_recovery_selected_immediately_after_search_open_when_download_candidate_visible(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=1,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Client for Windows (64bit x64)","click_point":[1011,623],"reason_tags":["download_like","target_like","button_shape"]}'
            "]} "
        ),
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20windows", expected_title_tokens=["filezilla"])',
            }
        },
    )
    assert _should_use_model_ui_download_recovery(request) is True
    code = _synthesized_model_ui_download_recovery_code(request)
    assert "SEARCH_RESULTS_FOCUS = True" in code
    assert "after opening a search result candidate" in code


def test_model_ui_download_recovery_search_results_rank_target_above_generic_download(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        step_index=1,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"파일전송 프로그램 다운로드","click_point":[860,300],"reason_tags":["download_like","offtarget_search_result_penalty"],"score":1},'
            '{"text":"FileZilla - FTP 클라이언트","click_point":[860,468],"reason_tags":["target_like"],"score":48}'
            "]} "
        ),
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20pc%20desktop%20kr", expected_title_tokens=["filezilla"])',
            }
        },
    )

    code = _synthesized_model_ui_download_recovery_code(request)
    assert code.index("'FileZilla - FTP 클라이언트'") < code.index("'파일전송 프로그램 다운로드'")


def test_model_ui_download_recovery_allows_new_candidates_after_exhausted_attempt(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=3,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Client for Windows (64bit x86)","click_point":[1031,566],"reason_tags":["download_like","target_like"],"score":83},'
            '{"text":"Download FileZilla Pro - Download and install on Windows","click_point":[1031,1113],"reason_tags":["download_like","install_like","target_like"],"score":108}'
            "]} "
        ),
        last_execution={
            "stdout_tail": (
                "click visible download candidate[1/3]: Download at (486, 378)\n"
                "candidate point 1 did not finish download yet: recent installer download did not appear\n"
                "opened isolated recovery page: https://www.google.com/search?q=filezilla%20windows\n"
                "isolated recovery page did not find a stable installer: no official Windows installer/archive candidate found on the current page\n"
            ),
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates after opening isolated recovery page\n",
            "payload_metadata": {
                "agent_response": {
                    "python_code": (
                        "VISIBLE_CANDIDATES = ["
                        "{'text': 'Download', 'point': [486, 378]}, "
                        "{'text': 'FileZilla Server', 'point': [205, 378]}"
                        "]\n"
                        "print('previous recovery')"
                    )
                }
            },
        },
    )
    assert service_module._looks_like_exhausted_visible_download_recovery(request.last_execution) is True
    assert _should_use_model_ui_download_recovery(request) is True


def test_model_ui_download_recovery_blocks_same_candidates_after_exhausted_attempt(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=3,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download","click_point":[486,378],"reason_tags":["download_like"],"score":71},'
            '{"text":"FileZilla Server","click_point":[205,378],"reason_tags":["target_like"],"score":48}'
            "]} "
        ),
        last_execution={
            "stdout_tail": (
                "click visible download candidate[1/3]: Download at (486, 378)\n"
                "candidate point 1 did not finish download yet: recent installer download did not appear\n"
                "opened isolated recovery page: https://www.google.com/search?q=filezilla%20windows\n"
                "isolated recovery page did not find a stable installer: no official Windows installer/archive candidate found on the current page\n"
            ),
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates after opening isolated recovery page\n",
            "payload_metadata": {
                "agent_response": {
                    "python_code": (
                        "VISIBLE_CANDIDATES = ["
                        "{'text': 'Download', 'point': [486, 378]}, "
                        "{'text': 'FileZilla Server', 'point': [205, 378]}"
                        "]\n"
                        "print('previous recovery')"
                    )
                }
            },
        },
    )
    assert service_module._looks_like_exhausted_visible_download_recovery(request.last_execution) is True
    assert _should_use_model_ui_download_recovery(request) is False


def test_model_ui_download_recovery_allows_single_candidate_after_scroll_changed_candidates(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=6,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Server","click_point":[1455,300],"reason_tags":["download_like","target_like"],"score":83}'
            "]} "
        ),
        last_execution={
            "stdout_tail": (
                "click visible download candidate[1/3]: Download FileZilla Server at (1561, 525)\n"
                "candidate point 4 did not finish download yet: recent installer download did not appear\n"
                "colored CTA candidate did not finish download: recent installer download did not appear\n"
            ),
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates after clicking a colored download CTA candidate\n",
            "payload_metadata": {
                "agent_response": {
                    "python_code": (
                        "VISIBLE_CANDIDATES = ["
                        "{'text': 'Download FileZilla Server', 'point': [1561, 525]}, "
                        "{'text': '파일질라 서버 다운로드', 'point': [1459, 432]}, "
                        "{'text': 'Free Download', 'point': [1484, 280]}"
                        "]\n"
                        "print('previous recovery')"
                    )
                }
            },
        },
    )
    assert _should_use_model_ui_download_recovery(request) is True


def test_model_ui_download_recovery_blocks_same_single_candidate_after_repeated_failure(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=6,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Server","click_point":[1455,300],"reason_tags":["download_like","target_like"],"score":83}'
            "]} "
        ),
        last_execution={
            "stdout_tail": (
                "click visible download candidate[1/1]: Download FileZilla Server at (1455, 300)\n"
                "candidate point 4 did not finish download yet: recent installer download did not appear\n"
            ),
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates after clicking a colored download CTA candidate\n",
            "payload_metadata": {
                "agent_response": {
                    "python_code": (
                        "VISIBLE_CANDIDATES = ["
                        "{'text': 'Download FileZilla Server', 'point': [1455, 300]}"
                        "]\n"
                        "print('previous recovery')"
                    )
                }
            },
        },
    )
    assert _should_use_model_ui_download_recovery(request) is False


def test_model_ui_download_recovery_selected_immediately_after_official_page_open(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=1,
        observation_text='MODEL_VISIBLE_UI_CANDIDATES: {"candidates":[{"text":"다운로드","click_point":[1792,821],"reason_tags":["download_like"]}]}',
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://mydev.kr/", expected_title_tokens=["memoit","메모잇"])',
            }
        },
    )
    assert _should_use_model_ui_download_recovery(request) is True


def test_model_ui_download_recovery_selected_for_target_only_candidates_after_search_recovery(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=2,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"SampleDesk - official downloads","click_point":[900,530],"reason_tags":["clickable_kind","target_like","button_shape"]}'
            "]}"
        ),
        last_execution={
            "stderr_tail": "No official Windows installer/archive candidate found from the prompt URLs.",
        },
    )

    assert _should_use_model_ui_download_recovery(request) is True


def test_model_ui_download_recovery_retries_target_only_candidates_before_late_stall(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Use Python to stay on the visible official page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        step_index=2,
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"SampleDesk","click_point":[900,330],"reason_tags":["clickable_kind","target_like","button_shape"]},'
            '{"text":"SampleDesk","click_point":[1200,330],"reason_tags":["clickable_kind","target_like","button_shape"]}'
            "]}"
        ),
        last_execution={
            "stdout_tail": (
                "click visible download candidate: SampleDesk at (900, 330)\n"
                "candidate did not finish download yet: recent installer download did not appear"
            ),
            "stderr_tail": "clicked a grounded download candidate",
        },
    )

    assert _should_use_model_ui_download_recovery(request) is True


def test_download_artifact_only_chunk_recognizes_replan_download_only_prompt() -> None:
    prompt = (
        "REPLAN OVERRIDE FOR THIS STEP:\n"
        "Return executable Python only.\n"
        "Use the screenshot to estimate the visible download/install control coordinates, then use Python GUI actions to focus the browser, click or keyboard-navigate that control, and wait for the download artifact to stabilize in Downloads.\n"
        "This is a download-only step. Do not launch, silently install, or run the installer in this step.\n"
        "End this step only when the installer file exists in Downloads."
    )
    assert service_module._looks_like_download_artifact_only_chunk(prompt) is True


def test_extract_discovery_chunk_is_not_treated_as_download_only() -> None:
    prompt = (
        "Using Python on the target Windows machine, extract the downloaded MobaXterm ZIP into a new folder under Downloads, "
        "then search the extracted contents for the main MobaXterm executable. Keep the flow self-contained: do not redownload anything, "
        "and do not assume a fixed inner filename. If multiple executables are found, select the one that is clearly the MobaXterm application "
        "and prepare it for launch."
    )
    assert service_module._looks_like_archive_extract_or_executable_discovery_chunk(prompt) is True
    assert service_module._looks_like_download_artifact_only_chunk(prompt) is False


def test_negated_launch_markers_keep_extract_discovery_out_of_launch_classifiers() -> None:
    markers = (
        "do not launch the app",
        "do not launch",
        "don't launch",
        "do not run the app",
        "do not run yet",
        "not launch yet",
        "not run yet",
        "아직 실행하지",
        "실행하지 마",
        "실행하지 말",
        "실행하지 않고",
    )
    for marker in markers:
        prompt = (
            "Using Python on the target Windows machine, extract the downloaded ZIP into a new folder under Downloads, "
            "then search the extracted contents for the main executable. "
            f"Keep the work focused on extraction and executable discovery only; {marker}. "
            "Prepare it for launch in the next chunk."
        )
        assert service_module._looks_like_archive_extract_or_executable_discovery_chunk(prompt) is True
        assert _looks_like_existing_installer_launch_task(prompt) is False
        assert _looks_like_launch_app_chunk_task(prompt) is False


def test_model_ui_browser_prelude_not_selected_for_extract_discovery_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Using Python on the target Windows machine, extract the downloaded MobaXterm ZIP into a new folder under Downloads, "
            "then search the extracted contents for the main MobaXterm executable. Keep the flow self-contained: do not redownload anything, "
            "and do not assume a fixed inner filename. If multiple executables are found, select the one that is clearly the MobaXterm application "
            "and prepare it for launch."
        ),
        execution_style="gui_first",
        step_index=0,
        request_kind="task_step",
    )
    assert _should_use_model_ui_browser_prelude(request) is False


def test_model_ui_download_recovery_not_selected_for_extract_discovery_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Using Python on the target Windows machine, extract the downloaded MobaXterm ZIP into a new folder under Downloads, "
            "then search the extracted contents for the main MobaXterm executable. Keep the flow self-contained: do not redownload anything, "
            "and do not assume a fixed inner filename. If multiple executables are found, select the one that is clearly the MobaXterm application "
            "and prepare it for launch."
        ),
        execution_style="gui_first",
        step_index=1,
        request_kind="task_step",
        observation_text='MODEL_VISIBLE_UI_CANDIDATES: {"candidates":[{"text":"Download MobaXterm Home Edition (current version)","click_point":[868,455],"reason_tags":["download_like","target_like","button_shape"]}]}',
    )
    assert _should_use_model_ui_download_recovery(request) is False


def test_download_chunk_completed_accepts_model_ui_recovery_markers() -> None:
    prompt = "Current chunk success target: The official installer `.msi` exists in Downloads and is non-empty."
    assert _looks_like_download_chunk_completed(
        user_prompt=prompt,
        last_execution={
            "return_code": 0,
            "stdout_tail": "download recovered from official page after stalled click: C:\\Users\\me\\Downloads\\app.msi",
        },
    )
    assert _looks_like_download_chunk_completed(
        user_prompt=prompt,
        last_execution={
            "return_code": 0,
            "stdout_tail": "using previously downloaded artifact: C:\\Users\\me\\Downloads\\app.msi",
        },
    )
    assert _looks_like_download_chunk_completed(
        user_prompt=prompt,
        last_execution={
            "return_code": 0,
            "stdout_tail": "download ready after visible click: C:\\Users\\me\\Downloads\\app.msi",
        },
    )


def test_model_ui_download_recovery_selected_for_replan_download_only_prompt(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    prompt = (
        "REPLAN OVERRIDE FOR THIS STEP:\n"
        "Return executable Python only.\n"
        "Use the screenshot to estimate the visible download/install control coordinates, then use Python GUI actions to focus the browser, click or keyboard-navigate that control, and wait for the download artifact to stabilize in Downloads.\n"
        "This is a download-only step. Do not launch, silently install, or run the installer in this step.\n"
        "End this step only when the installer file exists in Downloads."
    )
    request = StepRequest(
        user_prompt=prompt,
        execution_style="gui_first",
        step_index=2,
        observation_text='MODEL_VISIBLE_UI_CANDIDATES: {"candidates":[{"text":"다운로드","click_point":[1779,698],"reason_tags":["download_like","button_shape"]}]}',
        last_execution={
            "payload_metadata": {
                "executed_python_code": "raise SystemExit('model-ui download recovery did not produce a stable downloaded artifact')",
            },
            "timed_out": True,
        },
    )
    assert _should_use_model_ui_download_recovery(request) is True


def test_installer_ui_candidates_run_on_first_installer_step(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
        screenshot_base64="iVBORw0KGgo=",
        step_index=0,
    )
    assert _should_use_model_ui_candidates(request) is False
    assert _should_use_installer_ui_candidates(request) is True


def test_missing_image_template_generation_detects_guessed_locate_on_screen() -> None:
    assert _looks_like_missing_image_template_generation("import pyautogui\npyautogui.locateOnScreen('download_button.png')") is True
    assert _looks_like_missing_image_template_generation("import pyautogui\npyautogui.locateOnScreen('mobaxterm_home_edition.exe')") is True
    assert _looks_like_missing_image_template_generation("import pyautogui\npyautogui.click(100, 200)") is False


def test_gui_first_installer_wait_without_ui_action_detected() -> None:
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
    )
    code = """import subprocess
proc = subprocess.Popen(["C:/Users/me/Downloads/setup.exe"])
while proc.poll() is None:
    pass
"""
    assert _looks_like_gui_first_installer_wait_without_ui_action(request, code) is True


def test_gui_first_installer_wait_allowed_with_ui_action() -> None:
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
    )
    code = """import subprocess, pyautogui
proc = subprocess.Popen(["C:/Users/me/Downloads/setup.exe"])
pyautogui.press("enter")
while proc.poll() is None:
    break
    """
    assert _looks_like_gui_first_installer_wait_without_ui_action(request, code) is False


def test_gui_first_download_network_bypass_ignores_launch_chunk() -> None:
    request = StepRequest(
        user_prompt=(
            "Use executable Python on the Windows machine to locate the already-installed app executable, "
            "launch the app once, write `~/Downloads/launch-success.json`, and do not redownload or reinstall the app."
        ),
        execution_style="gui_first",
    )
    code = "import json\nfrom pathlib import Path\nPath('~/Downloads/launch-success.json').expanduser().write_text(json.dumps({}))"
    assert _looks_like_gui_first_download_chunk_network_bypass(request, code) is False


def test_invalid_retry_history_discards_silent_installer_prefix() -> None:
    history = _history_for_invalid_python_retry_with_prompt(
        [],
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        step_index=0,
        previous_code='subprocess.Popen(["setup.exe", "/VERYSILENT"])',
        gui_first_silent_install_shortcut=True,
    )
    joined = "\n".join(history)
    assert "previous_python_prefix=" not in joined
    assert "discard the previous installer script shape entirely" in joined
    assert "installer_dialog_control" in joined


def test_invalid_retry_history_never_includes_previous_python_prefix() -> None:
    history = _history_for_invalid_python_retry_with_prompt(
        [],
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        step_index=1,
        previous_code="import os\n\ndef helper():\n    return os.getcwd()\n",
    )
    joined = "\n".join(history)
    assert "previous_python_prefix=" not in joined
    assert "continue the same script idea from the previous partial Python" not in joined


def test_invalid_retry_history_warns_about_bottom_strip_clicks() -> None:
    history = _history_for_invalid_python_retry_with_prompt(
        [],
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        step_index=1,
        previous_code="import pyautogui\npyautogui.click(960, 1040)\n",
        bottom_strip_click_generation=True,
    )
    joined = "\n".join(history)
    assert "taskbar/dock strip" in joined
    assert "pinned app icons" in joined


def test_gui_first_bottom_strip_click_generation_detected(tmp_path) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x07\x80"
        b"\x00\x00\x04\x38"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        screenshot_path=str(screenshot),
        observation_text="Visible browser page with download results.",
    )
    assert _looks_like_gui_first_bottom_strip_click_generation(
        request,
        "import pyautogui\npyautogui.click(972, 1045)\n",
    ) is True
    assert _looks_like_gui_first_bottom_strip_click_generation(
        request,
        "import pyautogui\npyautogui.click(972, 540)\n",
    ) is False


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


def test_prepare_python_code_for_execution_auto_clicks_download_control_for_gui_first_visible_ui(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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
    assert 'print("continue")' in prepared
    assert "click_download_like_target(" in prepared


def test_prepare_python_code_for_execution_does_not_replace_blind_percentage_click_when_ocr_helpers_disabled(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", False)
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
    assert "advance_visible_download_flow(" not in prepared
    assert "btn_x = int(screen_w * 0.68)" in prepared


def test_prepare_python_code_for_execution_replaces_blind_percentage_click_with_visible_download_flow(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
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


def test_prepare_python_code_for_execution_replaces_risky_pygetwindow_retry_with_visible_download_flow(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
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


def test_prepare_python_code_for_execution_replaces_risky_pygetwindow_installer_retry_with_visible_installer_recovery(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Use Python to locate the FileZilla installer in Downloads, start it with subprocess.Popen, "
            "and complete the Windows setup using default options unless the installer requires a straightforward confirmation. "
            "After installation, launch FileZilla Client and confirm it starts successfully without errors.\n\n"
            "Current chunk success target: FileZilla is installed and the client process starts successfully.\n\n"
            "Preconditions expected before or during this chunk:\n"
            "- A valid FileZilla installer .exe already exists in ~/Downloads.\n\n"
            "Previously verified installer artifacts on the target machine. Prefer these exact installer paths before searching Downloads broadly again:\n"
            "- `C:\\Users\\user\\Downloads\\FileZilla_3.70.4_win64_sponsored2-setup.exe`"
        ),
        execution_style="gui_first",
        screenshot_base64="ZmFrZQ==",
        observation_text="Visible browser or installer UI.",
        replan_requested=True,
        replan_reasons=["execution_error"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20official%20windows%20download", expected_title_tokens=["filezilla"])',
            }
        },
    )
    prepared = _prepare_python_code_for_execution(
        request,
        """import pygetwindow as gw
gw.activateWindow()
""",
    )
    assert "advance_visible_installer_flow(" in prepared
    assert "gw.activateWindow()" not in prepared
    assert "advance_visible_download_flow(" not in prepared


def test_prepare_python_code_for_execution_replaces_single_coordinate_replan_click_with_visible_download_flow(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
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


def test_prepare_python_code_for_execution_replaces_browser_save_shortcut_flow_with_visible_download_flow(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
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


def test_prepare_python_code_for_execution_opens_search_without_replacing_http_bypass_with_ocr_flow(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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
    assert "click_download_like_target(" in prepared
    assert "browser_page_has_error_state(" not in prepared
    assert "urllib.request.urlopen" not in prepared


def test_prepare_python_code_for_execution_does_not_treat_file_write_as_gui_progress(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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


def test_prepare_python_code_for_execution_uses_visible_download_flow_for_visible_search_results(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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
    assert "advance_visible_download_flow(" in prepared
    assert 'print("continue")' in prepared
    assert "search_first = True" in prepared


def test_deprecated_ocr_helper_calls_are_detected() -> None:
    assert _uses_deprecated_ocr_helper("result = click_download_like_target(timeout_s=3)") is True
    assert _uses_deprecated_ocr_helper("lines = ocr_screen_text_regions(max_lines=20)") is True
    assert _uses_deprecated_ocr_helper("import pyautogui\npyautogui.click(900, 420)") is False


def test_ocr_observation_text_is_sanitized_when_framework_ocr_helpers_are_disabled(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", False)
    assert _sanitize_observation_text_for_model("OCR visible text with download/install cues: 다운로드") is None
    assert _sanitize_observation_text_for_model("Visible installer wizard is open") == "Visible installer wizard is open"


def test_ocr_observation_text_is_preserved_when_framework_ocr_helpers_are_enabled(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    assert _sanitize_observation_text_for_model("OCR visible text with download/install cues: 다운로드") == (
        "OCR visible text with download/install cues: 다운로드"
    )


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
    assert "score += 55" in code
    assert "score -= 85" in code
    assert "sftp" in code
    assert '"/appdata/local/temp/"' in code
    assert '"setup"' in code
    assert '"prompt_key": CONTEXT_PROMPT_KEY' in code
    assert 'print(f"already installed: {existing}")' not in code


def test_framework_visible_installer_recovery_selected_for_gui_first_existing_installer_task(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
    request = StepRequest(
        user_prompt=(
            "Use executable Python only. "
            "Find the existing installer `.exe` in `%USERPROFILE%\\\\Downloads\\\\computer-use-agent\\\\targetapp-1234\\\\`, "
            "launch it once, and then continue from the resulting installer UI."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_installer_recovery(request) is True


def test_framework_visible_installer_recovery_selected_for_downloaded_msi_install_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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


def test_framework_visible_download_flow_selected_for_gui_first_download_only_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
    request = StepRequest(
        user_prompt=(
            "Use Python to open the official vendor page and download only the Windows installer "
            "to `~/Downloads/targetapp-setup.exe`."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_download_flow(request) is True


def test_framework_visible_installer_recovery_selected_for_launch_downloaded_named_msi_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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


def test_framework_visible_launch_recovery_selected_for_gui_first_launch_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
    request = StepRequest(
        user_prompt=(
            "Prefer reading `~/Downloads/computer-use-agent/targetapp/install-success.json` first, "
            "launch the installed app once, and write "
            "`~/Downloads/computer-use-agent/targetapp/launch-success.json` only after launch succeeded."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_launch_recovery(request) is True


def test_framework_visible_launch_recovery_disabled_for_model_ui_launch_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Prefer reading `~/Downloads/computer-use-agent/targetapp/install-success.json` first, "
            "launch the installed app once, and write "
            "`~/Downloads/computer-use-agent/targetapp/launch-success.json` only after launch succeeded."
        ),
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_launch_recovery(request) is False


def test_model_ui_launch_recovery_selected_for_model_ui_launch_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Use executable Python on the Windows machine to locate the already-installed app executable "
            "for the target app from this task: targetapp 프로그램을 설치해줘, prefer reading "
            "`~/Downloads/install-success.json`, launch the app once, and write "
            "`~/Downloads/launch-success.json` only after launch succeeded."
        ),
        execution_style="gui_first",
    )
    assert _should_use_model_ui_launch_recovery(request) is True
    code = _synthesized_model_ui_launch_recovery_code(request)
    assert "INSTALL_MARKER" in code
    assert "LAUNCH_MARKER" in code
    assert "CONTEXT_MARKER" in code
    assert "CONTEXT_PROMPT_KEY = " in code
    assert "CONTEXT_PROMPT_EXCERPT = " in code
    assert "def _process_exists(name):" in code
    assert "launched process running=" in code
    assert "MARKER_HAYSTACK" in code
    assert "TERMS.append(term)" in code
    assert "Microsoft.Data" not in code


def test_model_ui_launch_recovery_selected_for_verify_and_launch_prompt(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt=(
            "Use Python to verify that Memoit is installed by checking for the installed executable or a running Memoit process, "
            "then launch Memoit if it is not already running. Confirm the app window opens successfully."
        ),
        execution_style="gui_first",
    )
    assert _looks_like_launch_app_chunk_task(request.user_prompt) is True
    assert _should_use_model_ui_launch_recovery(request) is True


def test_install_prompt_with_verify_clause_is_not_treated_as_launch_chunk() -> None:
    prompt = (
        "Locate the most recent official FileZilla Windows installer from the Downloads folder, "
        "launch it with Python using subprocess.Popen(), and complete the installation with default options. "
        "If UAC appears, allow it. When installation finishes, verify that FileZilla is installed and launch the main app so its window opens."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is True
    assert _looks_like_launch_app_chunk_task(prompt) is False


def test_framework_visible_launch_recovery_disabled_for_model_ui_install_chunk(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app.",
        execution_style="gui_first",
    )
    assert _should_use_framework_visible_launch_recovery(request) is False


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


def test_installer_filename_keywords_drop_noise_tokens() -> None:
    assert _installer_filename_keywords("FileZilla_3.70.4_win64_sponsored2-setup.exe", limit=6) == ["filezilla"]


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
    assert '"prompt_key": CONTEXT_PROMPT_KEY' in code


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


def test_fallback_browser_search_url_keeps_hangul_queries_compact_for_install_tasks() -> None:
    url = _fallback_browser_search_url("카카오톡 pc버전 프로그램을 설치해줘")
    assert url is not None
    assert "%EA%B3%B5%EC%8B%9D" not in url
    assert "-blog" not in url
    assert "-tistory" not in url
    decoded = urllib.parse.unquote(url)
    assert "다운로드" not in decoded
    assert "pc" in decoded


def test_fallback_browser_search_url_adds_only_minimal_windows_hint_for_ascii_tasks() -> None:
    url = _fallback_browser_search_url("filezilla 설치해줘")
    assert url is not None
    decoded = urllib.parse.unquote(url)
    assert "filezilla" in decoded
    assert "windows" in decoded
    assert "download" not in decoded
    assert "official" not in decoded


def test_fallback_browser_search_url_adds_vendor_domain_filters_from_prompt_urls() -> None:
    url = _fallback_browser_search_url("Use the official page https://pc.example.com/download and continue.")
    assert url is not None
    assert "site%3Aexample.com" in url
    assert "site%3Aexamplecorp.com" not in url


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


def test_replan_fallback_browser_search_url_prioritizes_explicit_retry_target_keywords() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "The current page did not produce usable download candidates.\n"
            "If you must abandon the current page and run a new browser search, keep these exact task/product keywords in the query: filezilla.\n"
            "Do not replace those task/product keywords with generic retry wording, verifier artifact names, or unrelated product names.\n"
            "Verifier evidence mentioned C:\\Users\\user\\Downloads\\DB.Browser.for.SQLite-win64.exe and other unrelated noise."
        ),
        execution_style="gui_first",
        replan_requested=True,
    )

    url = _fallback_browser_search_url_for_request(
        request,
        extra_targets=["browser", "sqlite"],
    )
    assert url is not None
    assert "filezilla" in url
    assert "sqlite" not in url
    assert "browser" not in url


def test_select_prompt_browser_url_prefers_korean_locale_for_hangul_task() -> None:
    prompt = (
        "카카오톡 pc버전 프로그램을 설치해줘. "
        "Use these URLs first: "
        "https://www.example.com/page/service/app?lang=en "
        "https://www.example.com/page/service/app?lang=ko"
    )
    assert _select_prompt_browser_url(prompt) == "https://www.example.com/page/service/app?lang=ko"


def test_select_prompt_browser_url_accepts_bare_domain_with_korean_particle() -> None:
    prompt = "filezilla.kr에서 filezilla 설치해줘"
    assert _extract_prompt_urls(prompt) == ["https://filezilla.kr"]
    assert _select_prompt_browser_url(prompt) == "https://filezilla.kr"
    request = StepRequest(user_prompt=prompt, execution_style="gui_first")
    assert _visible_flow_extra_targets(request, limit=8) == ["filezilla"]


def test_select_prompt_browser_url_accepts_http_url_with_korean_particle() -> None:
    prompt = "https://www.filezilla.kr/theme/filezilla/download/FileZilla_3.67.0_win64-setup.exe에서 filezilla 설치해줘"
    expected = "https://www.filezilla.kr/theme/filezilla/download/FileZilla_3.67.0_win64-setup.exe"
    assert _extract_prompt_urls(prompt) == [expected]
    assert _select_prompt_browser_url(prompt) == expected


def test_model_ui_candidate_scores_hangul_download_as_target_on_matching_prompt_domain() -> None:
    request = StepRequest(
        user_prompt="filezilla.kr에서 filezilla 설치해줘",
        execution_style="gui_first",
    )
    score, tags = _score_model_ui_candidate(
        "파일질라 다운로드",
        "button",
        (1120, 980, 1780, 1060),
        request=request,
        screen_size=(2560, 1440),
    )
    assert "target_page_download_like" in tags
    assert "target_like" in tags
    assert score >= 85


def test_model_ui_candidate_penalizes_server_variant_when_task_does_not_ask_for_server() -> None:
    request = StepRequest(
        user_prompt="filezilla.kr에서 filezilla 설치해줘",
        execution_style="gui_first",
    )
    client_score, client_tags = _score_model_ui_candidate(
        "Download FileZilla Client",
        "button",
        (1120, 980, 1780, 1060),
        request=request,
        screen_size=(2560, 1440),
    )
    server_score, server_tags = _score_model_ui_candidate(
        "Download FileZilla Server",
        "button",
        (1120, 980, 1780, 1060),
        request=request,
        screen_size=(2560, 1440),
    )
    assert "server_variant_penalty" in server_tags
    assert "server_variant_penalty" not in client_tags
    assert client_score > server_score


def test_model_ui_browser_prelude_uses_bare_domain_prompt_url(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", True)
    request = StepRequest(
        user_prompt="filezilla.kr에서 filezilla 설치해줘",
        execution_style="gui_first",
    )
    code = _synthesized_model_ui_browser_prelude_code(request)
    assert 'target_url = "https://filezilla.kr"' in code
    assert "google.com/search" not in code


def test_extract_prompt_urls_does_not_treat_installer_filename_as_bare_domain() -> None:
    assert _extract_prompt_urls("Downloads/FileZilla_3.69.3_win64-setup.exe 파일을 실행해줘") == []


def test_extract_prompt_urls_does_not_treat_python_dotted_names_as_bare_domains() -> None:
    code = 'import urllib.request\nurllib.request.urlopen("https://cdn.vendor.example/releases/app.zip")'
    assert _extract_prompt_urls(code) == ["https://cdn.vendor.example/releases/app.zip"]


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
    assert "timeout_s=48.0" in code
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
    assert "for download_attempt in range(3)" in code
    assert "wait_for_recent_download_artifact(" in code
    assert "since_ts=download_started_at" in code
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


def test_existing_installer_launch_task_does_not_match_download_chunk_with_downloaded_filename_language() -> None:
    prompt = (
        "Return executable Python only for this chunk. Use executable Python on the Windows machine to obtain "
        "the official Windows installer `.exe` or `.msi` for the target app and download it into "
        "`%USERPROFILE%\\\\Downloads\\\\`. Do not require the final downloaded filename to contain the app name. "
        "Current chunk success target: A target-app installer `.exe` or `.msi` exists in "
        "`%USERPROFILE%\\\\Downloads\\\\` and is non-empty."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is False


def test_existing_installer_launch_task_does_not_match_download_only_replan_prompt() -> None:
    prompt = (
        "REPLAN OVERRIDE FOR THIS STEP:\n"
        "Continue from the visible browser/download UI.\n"
        "This is a download-only step. Do not launch, silently install, or run the installer in this step.\n"
        "End this step only when the installer file exists in Downloads with a plausible non-trivial size."
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
    assert "initial_context_payload = read_action_context(CONTEXT_PATH, prompt_key=CONTEXT_PROMPT_KEY)" in code
    assert 'carried_context_installer = _context_candidate(initial_context_payload.get("_prompt_mismatch_installer_path"))' in code
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


def test_synthesized_visible_download_completion_code_prefers_recent_artifact_over_prompt_named_installer() -> None:
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
    assert "wait_for_recent_download_artifact(" in code
    assert 'wait_for_stable_download("TargetApp_Setup.exe"' in code
    assert code.index("wait_for_recent_download_artifact(") < code.index('wait_for_stable_download("TargetApp_Setup.exe"')
    assert 'print(f"recent download ready: {installer}")' in code


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
    assert "wait_for_recent_download_artifact(" in code
    assert 'wait_for_stable_download("TargetApp_Installer.zip"' in code
    assert code.index("wait_for_recent_download_artifact(") < code.index('wait_for_stable_download("TargetApp_Installer.zip"')
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
    assert "context_haystack = ' '.join([" in code
    assert "source_url" in code
    assert 'expected_download_glob = "targetapp_setup.exe"' not in code
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
    old_installer = tmp_path / "old.exe"
    new_installer = tmp_path / "new.exe"
    old_installer.write_bytes(b"old")
    new_installer.write_bytes(b"new")
    code = _expand_runtime_helpers(
        """
payload1 = write_action_context(path, prompt_key="prompt-a", prompt_excerpt="first", installer_path=str(old_installer))
payload2 = read_action_context(path, prompt_key="prompt-b")
payload_started = ensure_action_context(path, prompt_key="prompt-b", prompt_excerpt="second")
payload3 = write_action_context(path, prompt_key="prompt-b", prompt_excerpt="second", installer_path=str(new_installer))
payload4 = read_action_context(path, prompt_key="prompt-b")
"""
    )
    namespace = {"path": context_path, "old_installer": old_installer, "new_installer": new_installer}
    exec(code, namespace)
    assert namespace["payload1"]["installer_path"] == str(old_installer)
    assert namespace["payload2"]["_prompt_mismatch"] is True
    assert namespace["payload2"]["_prompt_mismatch_installer_path"] == str(old_installer)
    assert "installer_path" not in namespace["payload2"]
    assert namespace["payload_started"]["phase"] == "context_started"
    assert "installer_path" not in namespace["payload_started"]
    assert namespace["payload3"]["installer_path"] == str(new_installer)
    assert namespace["payload4"]["prompt_key"] == "prompt-b"
    assert namespace["payload4"]["installer_path"] == str(new_installer)


def test_action_context_keeps_started_state_but_prunes_missing_installer(tmp_path) -> None:
    context_path = tmp_path / "computer-use-agent-context.json"
    missing_installer = tmp_path / "missing.exe"
    code = _expand_runtime_helpers(
        """
started = ensure_action_context(path, prompt_key="prompt-a", prompt_excerpt="first")
with_installer = write_action_context(path, prompt_key="prompt-a", prompt_excerpt="first", installer_path=str(missing_installer), source_url="https://example.test/file.exe")
after_prune = read_action_context(path, prompt_key="prompt-a")
"""
    )
    namespace = {"path": context_path, "missing_installer": missing_installer}
    exec(code, namespace)

    assert namespace["started"]["phase"] == "context_started"
    assert namespace["started"]["_exists"] is True
    assert "installer_path" not in namespace["with_installer"]
    assert "source_url" not in namespace["with_installer"]
    assert namespace["after_prune"]["_exists"] is True
    assert "installer_path" not in namespace["after_prune"]


def test_existing_installer_launch_task_detected_for_generic_downloaded_installer_prompt() -> None:
    prompt = (
        "Locate the downloaded installer `.exe` in Downloads, verify it is the Windows installer, "
        "run it with Python automation, and proceed through the installer wizard."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is True


def test_visible_installer_recovery_selected_for_generic_installer_prompt_when_ocr_helpers_enabled(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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
    assert 'candidate_source="word_bbox"' in expanded
    assert 'candidate_source="line_bbox"' in expanded
    assert "installer_primary_action_region" not in expanded
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
    assert "def _heuristic_browser_click(" not in expanded
    assert "browser_search_result_region" not in expanded
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
    assert 'heuristic_mode="menu"' not in expanded
    assert "browser_header_menu_region" not in expanded
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
    assert "isolated_recovery_url_required" in expanded
    assert "max_same_page_fallback_candidates = 16" in expanded
    assert "for candidate_index in range(1, max_same_page_fallback_candidates + 1)" in expanded
    assert "download_action_text" in expanded
    assert "candidate_index" in expanded
    assert "skip_click_points=clicked_points" in expanded
    assert "download_related_window_fallback_page_open" in expanded
    assert "keybd_event(vk_l" not in expanded
    related_fallback_section = expanded.split("def click_download_related_fallback(", 1)[1].split("def click_search_result_like_target(", 1)[0]
    assert "targets.extend(str(item).strip().lower() for item in extra_targets" in related_fallback_section
    assert "context_targets=context" in related_fallback_section
    assert "context_match_scope=\"page\"" in related_fallback_section
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


def test_existing_installer_launch_task_detected_when_chunk_says_do_not_skip_ahead() -> None:
    prompt = (
        "Use executable Python only. Do not download anything in this chunk. "
        "First inspect the current screenshot and desktop state for an installer wizard, UAC prompt, license dialog, "
        "destination dialog, or completion dialog, and drive that visible UI forward if present. "
        "If no installer UI is visible yet, find the existing installer `.exe` or `.msi` in `%USERPROFILE%\\\\Downloads\\\\`, "
        "launch it once, and then continue from the resulting installer UI. "
        "Do only this chunk. Do not skip ahead to later chunks."
    )
    assert _looks_like_existing_installer_launch_task(prompt) is True


def test_existing_installer_launch_task_detected_for_verified_installer_artifact_prompt() -> None:
    prompt = (
        "Use Python to locate the FileZilla installer in Downloads, start it with subprocess.Popen, "
        "and complete the Windows setup using default options unless the installer requires a straightforward confirmation. "
        "After installation, launch FileZilla Client and confirm it starts successfully without errors.\n\n"
        "Current chunk success target: FileZilla is installed and the client process starts successfully.\n\n"
        "Preconditions expected before or during this chunk:\n"
        "- A valid FileZilla installer .exe already exists in ~/Downloads.\n\n"
        "Previously verified installer artifacts on the target machine. Prefer these exact installer paths before searching Downloads broadly again:\n"
        "- `C:\\Users\\user\\Downloads\\FileZilla_3.70.4_win64_sponsored2-setup.exe`"
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


def test_prompt_keyword_candidates_preserve_product_phrase_tokens() -> None:
    keywords = _prompt_keyword_candidates(
        'source_task": "DB Browser for SQLite 프로그램을 설치해줘"',
        limit=4,
    )

    assert keywords[:3] == ["db", "browser", "sqlite"]
    assert "for" not in keywords


def test_visible_flow_extra_targets_preserve_product_phrase_browser_token() -> None:
    request = StepRequest(
        user_prompt="Use executable Python for the target app from this task: DB Browser for SQLite 설치해줘.",
        execution_style="gui_first",
    )

    assert _visible_flow_extra_targets(request, limit=4)[:3] == ["db", "browser", "sqlite"]
    assert "db%20browser%20sqlite" in _fallback_browser_search_url_for_request(
        request,
        extra_targets=_visible_flow_extra_targets(request, limit=4),
    )
    assert "https://sqlitebrowser.org/dl/" in _fallback_official_domain_urls(
        _visible_flow_extra_targets(request, limit=4),
        limit=12,
    )


def test_fallback_browser_search_url_adds_kr_query_hint_for_korean_task() -> None:
    url = _fallback_browser_search_url("filezilla 설치해줘")
    decoded = urllib.parse.unquote(url or "")

    assert "filezilla" in decoded
    assert "kr" in decoded
    assert "site:filezilla.kr" not in decoded


def test_fallback_browser_search_url_for_request_adds_kr_query_hint_without_exact_domain_guess() -> None:
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
    )

    url = _fallback_browser_search_url_for_request(request, extra_targets=["filezilla"])
    decoded = urllib.parse.unquote(url or "")

    assert "filezilla" in decoded
    assert "kr" in decoded
    assert "site:filezilla.kr" not in decoded


def test_fallback_alternate_search_urls_include_localized_query_not_guessed_domain() -> None:
    urls = service_module._fallback_alternate_search_urls_from_parts(["filezilla"])
    decoded = [urllib.parse.unquote(url) for url in urls]

    assert decoded[0] == "https://www.google.com/search?q=filezilla kr"
    assert all("filezilla.kr" not in url for url in urls)


def test_fallback_official_domain_urls_include_generic_project_domain_variant() -> None:
    urls = _fallback_official_domain_urls(["filezilla"])

    assert "https://filezilla.org/download/" in urls
    assert "https://filezilla-project.org/download/" in urls


def test_fallback_search_url_does_not_invent_corp_sibling_domain_from_prompt_url() -> None:
    url = _fallback_browser_search_url_from_parts(
        ["filezilla"],
        ["https://filezilla.org/download.php?type=client"],
    )
    decoded = urllib.parse.unquote(url or "")

    assert "filezilla" in decoded
    assert "filezillacorp" not in decoded


def test_visible_flow_extra_targets_preserve_replan_comma_target_terms() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: db, browser, sqlite.\n"
            "End this step only when the installer file exists in Downloads."
        ),
        execution_style="gui_first",
        replan_requested=True,
    )

    assert _visible_flow_extra_targets(request, limit=4)[:3] == ["db", "browser", "sqlite"]


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


def test_visible_flow_extra_targets_prefers_replan_preserved_target_terms() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "Continue from the visible browser/download UI before trying any new network fetch.\n"
            "Original task target terms to preserve: 카카오톡.\n"
            "This is a download-only step. Do not launch, silently install, or run the installer in this step.\n"
            "End this step only when the installer file exists in Downloads with a plausible non-trivial size."
        ),
        execution_style="gui_first",
        replan_requested=True,
    )

    assert _visible_flow_extra_targets(request, limit=8) == ["카카오톡"]


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


def test_visible_flow_extra_targets_use_search_query_from_retry_url() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "The previous attempt already opened the relevant browser page.\n"
            "Current page: https://www.google.com/search?q=%EB%A9%94%EB%AA%A8%EC%9E%87%20%EA%B3%B5%EC%8B%9D%20%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20pc%20windows\n"
        ),
        execution_style="gui_first",
    )
    keywords = _visible_flow_extra_targets(request, limit=4)
    assert "메모잇" in keywords
    assert "google" not in keywords


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


def test_visible_flow_targets_prefer_product_text_over_official_url_host() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python automation on Windows to open the official MobaXterm Home Edition download page at "
            "https://mobaxterm.mobatek.net/download-home-edition.html, then download the official installer package."
        ),
        execution_style="gui_first",
    )
    assert _visible_flow_extra_targets(request, limit=4)[0] == "mobaxterm"


def test_visible_flow_extra_targets_filters_download_flow_prompt_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "On the Windows machine, use Python to open the official FileZilla download flow starting at "
            "https://filezilla.net/. If you confirm or obtain a valid installer, write "
            "`~/Downloads/computer-use-agent-context.json` with `installer_path`, `source_url`, and `target_keywords`."
        ),
        execution_style="gui_first",
    )
    keywords = _visible_flow_extra_targets(request, limit=6)
    assert "filezilla" in keywords
    assert "flow" not in keywords
    assert "starting" not in keywords
    assert "installer_path" not in keywords
    assert "source_url" not in keywords
    assert "target_keywords" not in keywords


def test_visible_flow_extra_targets_filters_model_ui_prompt_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n"
            "MODEL_VISIBLE_UI_CANDIDATES below come from the current screenshot.\n"
            "Open the FileZilla Project download page at https://filezilla-project.org/download.php?type=client.\n"
            "If you confirm a valid installer, write `target_keywords`."
        ),
        execution_style="gui_first",
    )

    keywords = _visible_flow_extra_targets(request, limit=6)

    assert keywords == ["filezilla"]


def test_visible_flow_extra_targets_ignore_rogue_backtick_url_block_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Explicit open-target page URLs for this chunk. Treat these exact URLs as the primary runtime open targets "
            "before any generic search, and open them in order:\n"
            "- https://mydev.kr/`\n"
            "- https://mydev.kr/\n\n"
            "공식 사이트 `mydev.kr`에서 Windows용 메모잇 설치 파일 `setup_memoit193.exe`를 내려받아 `Downloads` 폴더에 저장하세요. "
            "페이지에서 보이는 공식 다운로드 링크를 우선 사용하고, 반드시 `.exe` 또는 `.msi` 설치 파일만 받으세요."
        ),
        execution_style="gui_first",
    )

    keywords = _visible_flow_extra_targets(request, limit=8)

    assert set(keywords[:3]) == {"메모잇", "memoit193", "memoit"}
    assert "가" not in keywords
    assert "있으면" not in keywords
    assert "그것" not in keywords


def test_visible_flow_extra_targets_prefer_top_level_source_task_over_broken_markdown_url_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: 메모잇 프로그램을 설치해줘\n\n"
            "Explicit open-target page URLs for this chunk. Treat these exact URLs as the primary runtime open targets before any generic search, and open them in order:\n"
            "- https://mydev.kr/\n"
            "- https://mydev.kr/`](https://mydev.kr/\n\n"
            "Open the official Memoit site at a relevant product/download page in the browser, click the \"메모잇 다운로드 (v1.93)\" download button, "
            "and use Python-based browser automation to ensure the Windows installer `setup_memoit193.exe` is downloaded into `~/Downloads`. "
            "Download the `.exe` installer only and avoid any `.zip` or archive build.\n"
        ),
        execution_style="gui_first",
    )

    keywords = _visible_flow_extra_targets(request, limit=8)

    assert keywords[:3] == ["메모잇", "memoit193", "memoit"]
    assert "downloaded" not in keywords
    assert "나" not in keywords


def test_rewrite_user_prompt_for_replan_preserves_top_level_source_task_terms_over_broken_markdown_noise() -> None:
    prompt = (
        "Return executable Python only for this chunk.\n\n"
        "Top-level source task for this run: 메모잇 프로그램을 설치해줘\n\n"
        "Explicit open-target page URLs for this chunk. Treat these exact URLs as the primary runtime open targets before any generic search, and open them in order:\n"
        "- https://mydev.kr/\n"
        "- https://mydev.kr/`](https://mydev.kr/\n\n"
        "Open the official Memoit site at a relevant product/download page in the browser, click the \"메모잇 다운로드 (v1.93)\" download button, "
        "and use Python-based browser automation to ensure the Windows installer `setup_memoit193.exe` is downloaded into `~/Downloads`. "
        "Download the `.exe` installer only and avoid any `.zip` or archive build.\n"
        "Current chunk success target: `setup_memoit193.exe` is present in Downloads.\n"
    )

    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["partial_progress_opened_page_only", "same_page_click_retry_required"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://mydev.kr/", expected_title_tokens=["memoit193", "memoit"])',
            },
            "stdout_tail": "opened browser page for screenshot-grounded UI continuation",
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates",
        },
    )

    assert "Original task target terms to preserve: 메모잇, memoit193, memoit." in rewritten
    assert "Original task target terms to preserve: downloaded, 나." not in rewritten


def test_visible_flow_extra_targets_recover_from_last_execution_visible_candidates_when_preserved_terms_are_polluted() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "Original task target terms to preserve: downloaded, 나.\n"
            "Original official URLs to preserve: https://mydev.kr/.\n"
            "If the previous click did not cause visible progress, do not reuse the same coordinates first.\n"
            "Choose a different visible download/install candidate in the page content area.\n"
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": (
                    "TARGET_TERMS = ['downloaded', '나']\n"
                    "VISIBLE_CANDIDATES = ["
                    "{'text': '메모잇 다운로드 (v1.93)', 'point': [1790, 760], 'score': 11, 'tags': ['download_like']}"
                    "]\n"
                ),
            },
        },
    )

    keywords = _visible_flow_extra_targets(request, limit=6)

    assert keywords[0] == "메모잇"
    assert "downloaded" not in keywords
    assert "나" not in keywords


def test_runtime_helper_download_official_installer_allows_keyword_matched_official_cross_domain() -> None:
    helper = service_module._RUNTIME_HELPERS["download_official_installer_from_page"]
    assert "def _host_matches_keyword_domain(host):" in helper
    assert "if registrable != allowed_registrable and not _host_matches_keyword_domain(parsed.netloc):" in helper
    assert "and not _host_matches_keyword_domain(urlparse(current_page).netloc)" in helper


def test_visible_flow_extra_targets_keep_previous_target_terms_over_replan_stdout_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "Previous stdout summary: install marker written: C:\\Program Files\\Git\\usr\\bin\\stdbuf.exe\n"
            "Previous stderr summary: no installer package available for installer recovery\n"
            "Use the existing installer already present in Downloads; do not add download logic."
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'TARGET_TERMS = ["db", "browser", "sqlite"]\nprint("run installer")',
            },
            "stdout_tail": "install marker written: C:\\Program Files\\Git\\usr\\bin\\stdbuf.exe",
        },
    )
    keywords = _visible_flow_extra_targets(request, limit=4)
    assert keywords[:3] == ["db", "browser", "sqlite"]
    assert "stdbuf" not in keywords


def test_visible_flow_extra_targets_prefer_task_keywords_over_noisy_installer_filename_and_prior_terms() -> None:
    request = StepRequest(
        user_prompt=(
            "Using the downloaded FileZilla Windows 64-bit `.exe` installer in Downloads, launch the installer with Python "
            "and complete setup using default options.\n"
            "Previously verified installer artifacts on the target machine:\n"
            "- `C:\\Users\\user\\Downloads\\FileZilla_3.70.4_win64_sponsored2-setup.exe`"
        ),
        execution_style="gui_first",
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'TARGET_TERMS = ["filezilla", "sponsored2", "sponsored", "bit"]\nprint("run installer")',
            }
        },
    )
    keywords = _visible_flow_extra_targets(request, limit=6)
    assert keywords[0] == "filezilla"
    assert "sponsored2" not in keywords
    assert "sponsored" not in keywords
    assert "bit" not in keywords


def test_visible_flow_extra_targets_ignore_replan_failure_noise_words() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: filezilla.\n"
            "Previous stderr summary: clicked all grounded points for all visible candidates without a stable download"
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'TARGET_TERMS = ["filezilla"]\nprint("retry download")',
            }
        },
    )
    assert _visible_flow_extra_targets(request, limit=8) == ["filezilla"]


def test_visible_flow_extra_targets_filters_prompt_scaffold_words_from_installer_replan() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "Source task: 고클린 설치해줘.\n"
            "Explicit open-target page URLs for this chunk:\n"
            "- https://www.gobest.kr/goclean_app/index.htm\n"
            "Find the downloaded GoClean `.exe` in the Downloads folder and execute it.\n"
            "Previous stdout summary: install marker written: C:\\Program Files\\Git\\usr\\bin\\find.exe\n"
            "Previous stderr summary: installed executable detected but process did not stay running\n"
            "Use the existing installer already present in Downloads; do not add download logic."
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": (
                    'TARGET_TERMS = ["고클린", "gocleansetup153", "gocleansetup", '
                    '"gobest", "top-level", "explicit", "find", "goclean"]'
                ),
            },
            "stdout_tail": "install marker written: C:\\Program Files\\Git\\usr\\bin\\find.exe",
            "stderr_tail": "installed executable detected but process did not stay running",
        },
    )

    keywords = _visible_flow_extra_targets(request, limit=8)

    assert "고클린" in keywords
    assert "goclean" in keywords
    assert "find" not in keywords
    assert "explicit" not in keywords
    assert "top-level" not in keywords


def test_visible_flow_extra_targets_preserved_terms_ignore_replan_control_sentence_noise() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Return executable Python only.\n"
            "Original task target terms to preserve: 카카오톡.\n"
            "Original official URLs to preserve: https://pc.kakao.com/talk/notices/ko/2983?agent=win32.\n"
            "If the previous click did not cause visible progress, do not reuse the same coordinates first.\n"
            "Choose a different visible download/install candidate in the page content area.\n"
            "Previous stdout summary: isolated recovery page did not find a stable installer."
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'TARGET_TERMS = ["카카오톡"]\nprint("retry download")',
            }
        },
    )

    assert _visible_flow_extra_targets(request, limit=8) == ["카카오톡"]


def test_model_ui_installer_recovery_uses_strong_target_matching_for_short_tokens() -> None:
    request = StepRequest(
        user_prompt="Use Python to run the downloaded MSI installer `~/Downloads/DB.Browser.for.SQLite-v3.13.1-win64.msi`.",
        execution_style="gui_first",
    )
    code = _synthesized_model_ui_installer_recovery_code(request)
    assert 'strong_terms = [term for term in raw_terms' in code
    assert "def _contains_target(text, term):" in code
    assert "any(term in name or term in full for term in terms)" not in code


def test_model_ui_installer_recovery_preserves_context_installer_path() -> None:
    request = StepRequest(
        user_prompt="Find the existing installer `.exe` in Downloads, run the installer, finish installation, and launch the installed app.",
        execution_style="gui_first",
    )
    code = _synthesized_model_ui_installer_recovery_code(request)
    assert "preserved_installer = installer_path or previous_context.get('installer_path')" in code
    assert "payload['installer_path'] = str(preserved_installer)" in code
    assert "context_payload['installer_path'] = str(preserved_installer)" in code


def test_model_ui_installer_recovery_uses_context_installer_filename_for_package_terms(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    installer = downloads / "gocleansetup153.exe"
    installer.write_bytes(b"x" * 128)
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: 고클린 설치해줘\n\n"
            "Using Python automation on Windows, run the downloaded 고클린 installer `.exe` from Downloads "
            "and complete the setup wizard with default options."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_model_ui_installer_recovery_code(request)
    prefix = code.split("\npackages = _candidate_packages()", 1)[0].replace("import pyautogui\n", "")
    namespace: dict[str, object] = {}
    exec(prefix, namespace)
    context_marker = downloads / "computer-use-agent-context.json"
    context_marker.write_text(
        json.dumps(
            {
                "prompt_key": namespace["CONTEXT_PROMPT_KEY"],
                "installer_path": str(installer),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    packages = namespace["_candidate_packages"]()

    assert namespace["_target_terms"]() == ["고클린"]
    assert "gocleansetup153" in namespace["_package_terms"](str(installer))
    assert "gocleansetup" in namespace["_package_terms"](str(installer))
    assert packages == [installer]


def test_model_ui_installer_recovery_does_not_expand_package_terms_from_mismatched_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    installer = downloads / "gocleansetup153.exe"
    installer.write_bytes(b"x" * 128)
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: 고클린 설치해줘\n\n"
            "Using Python automation on Windows, run the downloaded 고클린 installer `.exe` from Downloads."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_model_ui_installer_recovery_code(request)
    prefix = code.split("\npackages = _candidate_packages()", 1)[0].replace("import pyautogui\n", "")
    namespace: dict[str, object] = {}
    exec(prefix, namespace)
    context_marker = downloads / "computer-use-agent-context.json"
    context_marker.write_text(
        json.dumps(
            {
                "prompt_key": "previous-task",
                "installer_path": str(installer),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert namespace["_candidate_packages"]() == []


def test_installer_recovery_target_terms_ignore_installer_control_and_impl_words() -> None:
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: 고클린 설치해줘\n\n"
            "Using Python on the Windows machine, locate the downloaded `gocleansetup153.exe` in Downloads "
            "and launch it with subprocess or an equivalent automation method. Complete the installer wizard "
            "with default choices, advancing through `다음`, `설치`, and `마침` as shown."
        ),
        execution_style="gui_first",
    )
    keywords = _installer_recovery_target_terms(request, limit=8)
    assert "고클린" in keywords
    assert "gocleansetup153" in keywords
    assert "다음" not in keywords
    assert "마침" not in keywords
    assert "subprocess" not in keywords


def test_model_ui_installer_recovery_prefers_installer_before_existing_exe_scan() -> None:
    request = StepRequest(
        user_prompt=(
            "Return executable Python only for this chunk.\n\n"
            "Top-level source task for this run: 고클린 설치해줘\n\n"
            "Use Python on the Windows machine to locate the downloaded `gocleansetup153.exe` in Downloads "
            "and run it with subprocess. Continue with 기본 설치."
        ),
        execution_style="gui_first",
    )
    code = _synthesized_model_ui_installer_recovery_code(request)
    assert code.index("installer = _resolve_installer_target(packages[0]) if packages else None") < code.index(
        "existing = _find_installed_exe()"
    )


def test_recent_download_helper_rejects_portable_when_not_requested() -> None:
    expanded = _expand_runtime_helpers("wait_for_recent_download_artifact(extra_targets=['mobaxterm'])")
    assert '"portable" in lowered and not any("portable" in token for token in normalized_targets)' in expanded


def test_recent_download_helper_can_require_target_match_for_reuse() -> None:
    expanded = _expand_runtime_helpers("wait_for_recent_download_artifact(extra_targets=['filezilla'], require_target_match=True)")
    assert "require_target_match=False" in expanded
    assert "if require_target_match and normalized_targets and score[0] <= 0:" in expanded


def test_download_page_helper_unwraps_search_result_links_and_prioritizes_distinct_tlds() -> None:
    expanded = _expand_runtime_helpers("download_official_installer_from_page('https://www.google.com/search?q=filezilla%20kr', extra_targets=['filezilla'])")
    assert "from urllib.parse import parse_qs, urljoin, urlparse, unquote" in expanded
    assert "def _normalize_link(base_url, raw):" in expanded
    assert 'for key in ("q", "url", "u"):' in expanded
    assert "def _prioritize_page_links(page_candidates, *, base_is_search_engine, limit=8):" in expanded
    assert 'if tld == "kr" and any(root in compact_url for root in keyword_roots):' in expanded
    assert "return _prioritize_page_links(" in expanded


def test_prompt_keyword_candidates_ignore_percent_encoded_fragments() -> None:
    text = (
        "Open https://pc.example.com/talk/notices/en%3Fagent%3Dwin32 and continue the official flow."
    )
    keywords = _prompt_keyword_candidates(text)
    assert "3fagent" not in keywords
    assert "3dwin32" not in keywords


def test_prompt_keyword_candidates_prefer_search_query_terms_over_search_host() -> None:
    text = "Open https://www.google.com/search?q=%EB%A9%94%EB%AA%A8%EC%9E%87%20%EA%B3%B5%EC%8B%9D%20%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20pc%20windows and continue."
    keywords = _prompt_keyword_candidates(text)
    assert "메모잇" in keywords
    assert "google" not in keywords


def test_prompt_keyword_candidates_filter_replan_and_korean_boilerplate_noise() -> None:
    assert _prompt_keyword_candidates("not execution path download-like official windows download") == []
    assert (
        _prompt_keyword_candidates(
            "이 chunk는 실행 가능한 Python 코드만으로 수행하세요. 현재 스크린샷에 브라우저 검색 결과가 보입니다."
        )
        == []
    )
    assert _fallback_browser_search_url("not execution path download-like official windows download") is None
    keywords = _prompt_keyword_candidates("filezilla 설치파일을 다운로드해줘")
    assert "filezilla" in keywords
    assert "다운로드해줘" not in keywords


def test_replan_search_url_validation_rejects_generic_or_long_queries() -> None:
    assert (
        _search_url_validation_error(
            "https://www.bing.com/search?q=not+execution+path+download-like+official+windows+download",
            ["filezilla"],
        )
        == "missing_target_keyword"
    )
    assert (
        _search_url_validation_error(
            "https://www.google.com/search?q=filezilla+official+windows+download+installer+client+setup+latest",
            ["filezilla"],
        )
        == "query_too_long"
    )
    assert _search_url_validation_error("https://www.google.com/search?q=filezilla+windows", ["filezilla"]) is None


def test_replan_search_url_validation_rejects_exact_failed_query_sentence() -> None:
    assert (
        _search_url_validation_error(
            "https://www.google.com/search?q=filezilla+windows",
            ["filezilla"],
            excluded_queries=["filezilla windows"],
        )
        == "repeated_query"
    )


def test_replan_search_url_prefers_alternate_query_after_download_candidate_failure(tmp_path) -> None:
    class RuntimeShouldNotBeCalled:
        def generate_text(self, **_: object) -> object:
            raise AssertionError("deterministic alternate replan search should be selected before model retry")

    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        replan_requested=True,
        replan_reasons=["no_visible_download_candidates"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20windows")',
            },
            "stderr_tail": "download related fallback found no clickable candidates",
        },
    )

    selected = service_module._generate_validated_replan_search_url(
        runtime=RuntimeShouldNotBeCalled(),
        request=request,
        target_terms=["filezilla"],
        root=tmp_path,
        step_id="step-001",
    )

    assert selected == "https://www.google.com/search?q=filezilla%20kr"
    payload = json.loads((tmp_path / "responses" / "step-001.replan-search-url.json").read_text(encoding="utf-8"))
    assert payload["attempts"][0]["attempt"] == "preferred_alternate_search"
    assert payload["selected"] == "https://www.google.com/search?q=filezilla%20kr"


def test_replan_search_url_generation_limits_model_attempts(tmp_path) -> None:
    class AlwaysInvalidRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, **_: object) -> object:
            self.calls += 1
            return type(
                "Result",
                (),
                {
                    "text": '{"search_url":"https://www.google.com/search?q=not+execution+path+download-like+official+windows+download"}',
                    "model_id": "fake-model",
                },
            )()

    runtime = AlwaysInvalidRuntime()
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        replan_requested=True,
        replan_reasons=["execution_error"],
    )

    selected = service_module._generate_validated_replan_search_url(
        runtime=runtime,
        request=request,
        target_terms=["filezilla"],
        root=tmp_path,
        step_id="step-001",
    )

    assert runtime.calls == 2
    assert selected is not None
    payload = json.loads((tmp_path / "responses" / "step-001.replan-search-url.json").read_text(encoding="utf-8"))
    model_attempts = [item for item in payload["attempts"] if isinstance(item.get("attempt"), int)]
    assert len(model_attempts) == 2


def test_replan_search_url_not_generated_for_same_page_search_result_processing(tmp_path) -> None:
    request = StepRequest(
        user_prompt="filezilla 설치해줘",
        execution_style="gui_first",
        replan_requested=True,
        replan_reasons=["partial_progress_opened_page_only", "same_page_click_retry_required"],
    )

    should_generate = (
        request.replan_requested
        and service_module._looks_like_download_or_install_task(request.user_prompt)
        and any(
            reason
            in {
                "no_visible_download_candidates",
                "download_url_404",
                "guessed_artifact_url_404",
                "installer_url_not_found",
            }
            for reason in request.replan_reasons
        )
        and "partial_progress_opened_page_only" not in request.replan_reasons
        and "same_page_click_retry_required" not in request.replan_reasons
        and not _select_validated_replan_search_url(request.user_prompt)
    )

    assert should_generate is False


def test_select_prompt_browser_url_prefers_validated_replan_search_url() -> None:
    text = (
        "Open https://example.com/download first.\n"
        "Validated replan search URL to use if a new browser search is needed: "
        "https://www.google.com/search?q=filezilla+windows"
    )
    assert _select_validated_replan_search_url(text) == "https://www.google.com/search?q=filezilla+windows"
    assert _select_prompt_browser_url(text) == "https://www.google.com/search?q=filezilla+windows"


def test_select_prompt_browser_url_prefers_validated_replan_retry_url() -> None:
    text = (
        "Open https://example.com/download first.\n"
        "Validated replan retry URL to use if a new browser search is needed: "
        "https://www.google.com/search?q=filezilla+windows+kr"
    )
    assert _select_validated_replan_search_url(text) == "https://www.google.com/search?q=filezilla+windows+kr"
    assert _select_prompt_browser_url(text) == "https://www.google.com/search?q=filezilla+windows+kr"


def test_select_prompt_browser_url_skips_excluded_failed_search_query() -> None:
    text = (
        "Open https://example.com/download first.\n"
        "Validated replan search URL to use if a new browser search is needed: "
        "https://www.google.com/search?q=filezilla+windows"
    )
    assert (
        _select_prompt_browser_url(text, excluded_queries=["filezilla windows"])
        == "https://example.com/download"
    )


def test_select_request_prompt_browser_url_rejects_repeated_failed_search_query() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Previous stdout summary: opened isolated recovery page: "
            "https://www.google.com/search?q=filezilla%20windows\n"
            "Validated replan search URL to use if a new browser search is needed: "
            "https://www.google.com/search?q=filezilla%20windows"
        ),
        execution_style="gui_first",
        replan_requested=True,
        last_execution={"payload_metadata": {"executed_python_code": ""}},
    )
    assert _select_request_prompt_browser_url(request) is None


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


def test_framework_official_download_recovery_enabled_for_gui_first_after_visible_download_stalls() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python to continue downloading the Windows installer from the official page. "
            "Official URL: https://pc.kakao.com/talk/notices/en?agent=win32"
        ),
        execution_style="gui_first",
        observation_text=None,
        last_execution={"stderr_tail": "no visible download-related control remains on the current screen"},
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=2,
    )
    assert _should_use_framework_official_download_recovery(request) is True


def test_framework_official_download_recovery_can_use_target_search_without_prompt_url() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: sampledesk, vendorcorp.\n"
            "End this step only when the installer file exists in Downloads."
        ),
        execution_style="gui_first",
        last_execution={
            "stdout_tail": "click visible download candidate: Download at (1000, 700)\n"
            "candidate did not finish download yet: recent installer download did not appear",
            "stderr_tail": "clicked a grounded download candidate",
        },
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=4,
    )

    assert _should_use_framework_official_download_recovery(request) is True


def test_framework_official_download_recovery_does_not_repeat_after_no_candidate_failure() -> None:
    request = StepRequest(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: filezilla.\n"
            "End this step only when the installer file exists in Downloads."
        ),
        execution_style="gui_first",
        last_execution={
            "stdout_tail": "Failed to fetch page https://filezilla.org/: HTTP Error 410: Gone",
            "stderr_tail": "No official Windows installer/archive candidate found from the prompt URLs.",
        },
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=4,
    )

    assert _should_use_framework_official_download_recovery(request) is False


def test_framework_official_download_recovery_defers_to_model_visible_candidates() -> None:
    request = StepRequest(
        user_prompt="Download the Windows installer for filezilla.",
        execution_style="gui_first",
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Client for Windows","point":[1216,482],"tags":["download_like","target_like"]}'
            "]}"
        ),
        last_execution={
            "stdout_tail": "opened browser page for screenshot-grounded UI continuation",
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates",
        },
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=1,
    )

    assert _should_use_framework_official_download_recovery(request) is False


def test_official_download_recovery_follows_search_results_without_prompt_url() -> None:
    code = _synthesized_official_download_recovery_code(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: sampledesk.\n"
            "End this step only when the installer file exists in Downloads."
        )
    )

    assert 'FALLBACK_SEARCH_URL = "https://www.google.com/search?' in code
    assert "FALLBACK_DOMAIN_URLS = " in code
    assert "allow_external_search_result = not base_registrables and current_is_search_engine" in code
    assert "not allow_external_search_result" in code
    assert "resolved_is_download_catalog" in code
    assert code.index("enqueue_page(page_queue, FALLBACK_SEARCH_URL)") < code.index(
        "enqueue_page(page_queue, FALLBACK_LUCKY_URL)"
    )
    assert code.index("enqueue_page(page_queue, FALLBACK_LUCKY_URL)") < code.index(
        "for fallback_domain_url in FALLBACK_DOMAIN_URLS:"
    )
    assert "def prioritize_page_links(page_links: list[str], *, base_is_search_engine: bool, limit: int = 8) -> list[str]:" in code
    assert 'if tld == "kr" and any(root in compact_url for root in keyword_roots):' in code
    assert "score += 40" in code
    assert "def record_failed_url(url: str, reason: str) -> None:" in code
    assert "def previously_failed_url(url: str) -> bool:" in code
    assert "Skipping previously failed source URL" in code
    assert "record_failed_url(page_url, str(exc))" in code
    assert "record_failed_url(exe_url, str(exc))" in code
    assert "failed_source_urls" not in code
    assert "failed_source_hosts" not in code
    assert "failed_source_scope" not in code


def test_official_download_recovery_allows_target_named_catalog_search_results() -> None:
    code = _synthesized_official_download_recovery_code(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: sampledesk.\n"
            "End this step only when the installer file exists in Downloads."
        )
    )

    assert "resolved_has_target_keyword" in code
    assert "(not resolved_is_download_catalog or resolved_has_target_keyword)" in code


def test_official_download_recovery_disables_lucky_search_for_hangul_targets() -> None:
    code = _synthesized_official_download_recovery_code(
        user_prompt=(
            "REPLAN OVERRIDE FOR THIS STEP:\n"
            "Original task target terms to preserve: 메모잇.\n"
            "End this step only when the installer file exists in Downloads."
        )
    )

    assert "FALLBACK_LUCKY_URL = None" in code
    assert "FALLBACK_SEARCH_URL = \"https://www.google.com/search?" in code


def test_model_ui_download_recovery_stops_after_repeated_visible_click_stall_with_prompt_url() -> None:
    request = StepRequest(
        user_prompt=(
            "Use Python to download the official installer from https://pc.kakao.com/ "
            "and save it to Downloads."
        ),
        execution_style="gui_first",
        observation_text='MODEL_VISIBLE_UI_CANDIDATES: {"candidates":[{"text":"Download","point":[2200,345],"tags":["download_like"]}]}',
        last_execution={
            "stdout_tail": "click visible download candidate: Download at (2201, 345)\ncandidate did not finish download yet: recent installer download did not appear",
            "stderr_tail": "clicked a grounded download candidate; inspect the updated screenshot",
        },
        step_index=4,
    )
    assert _should_use_model_ui_download_recovery(request) is False
    assert _should_use_framework_official_download_recovery(
        StepRequest(
            user_prompt=request.user_prompt,
            execution_style=request.execution_style,
            observation_text=request.observation_text,
            last_execution=request.last_execution,
            replan_requested=True,
            replan_reasons=["execution_error"],
            step_index=request.step_index,
        )
    ) is True


def test_model_ui_download_recovery_stops_after_first_exhausted_grounded_recovery_attempt() -> None:
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Client for Windows","point":[1110,754],"tags":["download_like","target_like"]}'
            "]} "
        ),
        last_execution={
            "stdout_tail": (
                "click visible download candidate[1/1]: Download FileZilla Client for Windows at (1110, 754)\n"
                "candidate point 4 did not finish download yet: recent installer download did not appear\n"
                "opened isolated recovery page: https://www.google.com/search?q=filezilla%20windows"
            ),
            "stderr_tail": "clicked all grounded points for all visible candidates without a stable download: recent installer download did not appear",
        },
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=2,
    )

    assert _should_use_model_ui_download_recovery(request) is False
    assert _should_use_framework_official_download_recovery(request) is True


def test_framework_official_download_retry_for_invalid_generation_enabled_after_exhausted_visible_download_recovery() -> None:
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Client for Windows","point":[1110,754],"tags":["download_like","target_like"]}'
            "]} "
        ),
        last_execution={
            "stdout_tail": (
                "click visible download candidate[1/1]: Download FileZilla Client for Windows at (1110, 754)\n"
                "candidate point 4 did not finish download yet: recent installer download did not appear\n"
                "skipping repeated isolated recovery page: https://www.google.com/search?q=filezilla%20windows"
            ),
            "stderr_tail": "clicked all grounded points for all visible candidates without a stable download: recent installer download did not appear",
        },
        replan_requested=True,
        replan_reasons=["execution_error", "no_visible_download_candidates"],
        step_index=2,
    )

    assert _should_use_framework_official_download_retry_for_invalid_generation(
        request,
        prompt_url_violation=False,
        gui_first_visible_ui_violation=True,
        guessed_artifact_url_generation=False,
        gui_first_download_chunk_network_bypass=True,
        gui_first_download_chunk_install_mix=True,
    ) is True


def test_framework_official_download_retry_for_invalid_generation_disabled_while_visible_ui_is_still_actionable() -> None:
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download FileZilla Client for Windows","point":[1110,754],"tags":["download_like","target_like"]}'
            "]} "
        ),
        last_execution={
            "stdout_tail": "opened browser page for screenshot-grounded UI continuation",
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates",
        },
        replan_requested=True,
        replan_reasons=["execution_error"],
        step_index=1,
    )

    assert _should_use_framework_official_download_retry_for_invalid_generation(
        request,
        prompt_url_violation=False,
        gui_first_visible_ui_violation=True,
        guessed_artifact_url_generation=False,
        gui_first_download_chunk_network_bypass=True,
        gui_first_download_chunk_install_mix=True,
    ) is False


def test_model_ui_download_recovery_continues_late_stall_when_alternate_candidate_visible() -> None:
    request = StepRequest(
        user_prompt="Use Python to stay on the visible browser page and download only the Windows installer into Downloads.",
        execution_style="gui_first",
        observation_text=(
            'MODEL_VISIBLE_UI_CANDIDATES: {"candidates":['
            '{"text":"Download for Windows","point":[920,430],"tags":["download_like"]},'
            '{"text":"Download installer","point":[1220,700],"tags":["download_like"]}'
            "]}"
        ),
        last_execution={
            "stdout_tail": "click visible download candidate: Download for Windows at (920, 430)\ncandidate did not finish download yet: recent installer download did not appear",
            "stderr_tail": "clicked a grounded download candidate; inspect the updated screenshot",
        },
        step_index=4,
    )

    assert _should_use_model_ui_download_recovery(request) is True


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
            "Download the official Windows installer `.exe` or `.msi` into Downloads, "
            "wait until the download is complete, and end this step only when the installer file exists in Downloads and is non-empty. "
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


def test_gui_first_download_bypass_can_execute_when_auto_open_prelude_is_available(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_FRAMEWORK_OCR_UI_HELPERS_ENABLED", True)
    monkeypatch.setattr(service_module, "_MODEL_UI_CANDIDATES_ENABLED", False)
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
    assert "advance_visible_download_flow(" in prepared
    assert "urllib.request.urlopen" not in prepared


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
    assert 'variant_specs.append((Path(button_text_path), scale, "button_text"))' in expanded
    assert "target_and_cta = _has_context_target(lowered, compact) and _has_download_cta_action(lowered, compact)" in expanded
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
    assert "browser_download_cta_region" not in expanded
    assert "heuristic_sweep_threshold = 0 if installer_mode else 2" not in expanded


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
    assert 'CONTEXT_PATH = Path.home() / "Downloads" / "computer-use-agent-context.json"' in code
    assert "CONTEXT_PROMPT_KEY =" in code
    assert '"prompt_key": CONTEXT_PROMPT_KEY' in code
    assert "def write_download_context(path: Path, source_url: str) -> None:" in code
    assert "write_download_context(dest, exe_url)" in code


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


def test_synthesized_official_download_recovery_uses_target_terms_not_prompt_noise() -> None:
    prompt = (
        "Return executable Python only for this chunk. "
        "Download into C:\\Users\\kss930\\Downloads and prefer the official vendor site. "
        "Open the official SampleDesk service page at https://www.vendorcorp.com/page/service/service/SampleDesk "
        "and find the Windows installer. "
        "가능하면 `sampledesk`, `vendorcorp` 같은 대상 앱 키워드가 포함된 공식 Windows installer를 찾으세요."
    )
    code = _synthesized_official_download_recovery_code_for_test(user_prompt=prompt)
    keyword_section = code.split("KEYWORDS =", 1)[1].split("USER_AGENT =", 1)[0]

    assert "sampledesk" in keyword_section
    assert "vendorcorp" in keyword_section
    assert "users" not in keyword_section
    assert "python-driven" not in keyword_section
    assert "FALLBACK_SEARCH_URL =" in code
    assert "FALLBACK_LUCKY_URL =" in code
    assert "def normalize_link(base_url: str, raw: str) -> str:" in code
    assert "def host_matches_keyword_domain(host: str) -> bool:" in code
    assert "or host_matches_keyword_domain(urlparse(resolved).netloc)" in code
    assert 'replace("\\\\u002F", "/")' in code


def test_official_download_recovery_prefers_search_before_guessed_domains_for_keyword_only_tasks() -> None:
    code = _synthesized_official_download_recovery_code_for_test(
        user_prompt="Return executable Python only for this chunk. filezilla 설치파일을 다운로드해줘"
    )

    assert 'FALLBACK_SEARCH_URL = "https://www.google.com/search?' in code
    assert 'FALLBACK_ALTERNATE_SEARCH_URLS = ["https://www.google.com/search?q=filezilla%20kr"' in code
    assert code.index("enqueue_page(page_queue, FALLBACK_SEARCH_URL)") < code.index(
        "for fallback_alternate_search_url in FALLBACK_ALTERNATE_SEARCH_URLS:"
    )
    assert code.index("for fallback_alternate_search_url in FALLBACK_ALTERNATE_SEARCH_URLS:") < code.index(
        "enqueue_page(page_queue, FALLBACK_LUCKY_URL)"
    )


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


def test_replan_prompt_rewrite_for_download_preserves_original_target_terms() -> None:
    prompt = (
        "Open the official SampleDesk service page at https://www.vendorcorp.com/page/service/service/SampleDesk "
        "and use Python-driven browser automation to find the Windows download link. "
        "가능하면 `sampledesk`, `vendorcorp` 같은 대상 앱 키워드가 포함된 공식 Windows installer를 찾으세요."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["partial_progress_opened_page_only"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": "open_url_and_wait('https://www.vendorcorp.com/page/service/service/SampleDesk')",
            },
            "stdout_tail": "opened browser page for screenshot-grounded UI continuation",
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates",
        },
    )

    assert "Original task target terms to preserve: sampledesk, vendorcorp." in rewritten
    assert "Original official URLs to preserve: https://www.vendorcorp.com/page/service/service/SampleDesk." in rewritten
    assert "opened browser page for screenshot-grounded UI continuation" in rewritten


def test_replan_prompt_rewrite_for_visible_candidate_navigation_without_browser_open_code() -> None:
    prompt = "Use Python to stay on the visible browser page and download only the Windows installer into Downloads."
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["partial_progress_opened_page_only"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": "import pyautogui\npyautogui.click(920, 430)\n",
            },
            "stdout_tail": (
                "visible candidate opened page: https://filezilla-project.org/download.php?type=client\n"
                "opened browser page for screenshot-grounded UI continuation"
            ),
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates",
        },
    )

    assert rewritten.startswith("REPLAN OVERRIDE FOR THIS STEP:")
    assert "Treat the current screenshot as the primary source of truth" in rewritten
    assert "Continue from the visible browser/download UI with Python GUI automation" in rewritten
    assert "Avoid reusing these previous click coordinates first: (920, 430)." in rewritten


def test_replan_prompt_rewrite_for_download_no_candidates_forces_search_reset() -> None:
    prompt = (
        "Use Python to stay on the visible browser page and download only the Windows installer into Downloads."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["no_visible_download_candidates"],
        last_execution={
            "stderr_tail": "download related fallback found no clickable candidates",
        },
    )

    assert "The current page did not produce usable download candidates." in rewritten
    assert "On retry, change the search terms" in rewritten
    assert "refresh the page/search query" in rewritten


def test_replan_prompt_rewrite_for_download_no_candidates_keeps_exact_target_keywords_in_search() -> None:
    prompt = "filezilla 설치해줘"
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["no_visible_download_candidates"],
        last_execution={
            "stderr_tail": "download related fallback found no clickable candidates",
        },
    )

    assert "Original task target terms to preserve: filezilla." in rewritten
    assert "keep these exact task/product keywords in the query: filezilla." in rewritten.lower()
    assert "Do not replace those task/product keywords with generic retry wording" in rewritten


def test_replan_prompt_rewrite_for_download_no_candidates_ignores_prompt_boilerplate() -> None:
    prompt = (
        "Return executable Python only for this chunk.\n"
        "이 chunk는 실행 가능한 Python 코드만으로 수행하세요. 현재 스크린샷에 브라우저 검색 결과가 보이면 이어서 사용하세요.\n\n"
        "Use Python-driven browser automation on the current Windows desktop to open the FileZilla client download page, "
        "then download the Windows installer `.exe` only.\n"
        "작업과 일치하는 vendor, product, download 페이지를 우선 사용하세요. "
        "가능하면 `filezilla` 같은 대상 앱 키워드가 포함된 Windows installer `.exe` 또는 `.msi`를 우선 찾으세요."
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["no_visible_download_candidates"],
        last_execution={
            "stderr_tail": "not execution path download-like official windows download",
        },
    )

    assert "Original task target terms to preserve: filezilla." in rewritten
    assert "keep these exact task/product keywords in the query: filezilla." in rewritten.lower()
    assert "chunk, 가능한" not in rewritten
    assert "not, execution, path" not in rewritten


def test_replan_prompt_rewrite_for_install_chunk_after_browser_open_stays_install_chunk() -> None:
    prompt = (
        "Use Python to locate the FileZilla installer in Downloads, start it with subprocess.Popen, "
        "and complete the Windows setup using default options unless the installer requires a straightforward confirmation. "
        "After installation, launch FileZilla Client and confirm it starts successfully without errors.\n\n"
        "Current chunk success target: FileZilla is installed and the client process starts successfully.\n\n"
        "Preconditions expected before or during this chunk:\n"
        "- A valid FileZilla installer .exe already exists in ~/Downloads.\n\n"
        "Previously verified installer artifacts on the target machine. Prefer these exact installer paths before searching Downloads broadly again:\n"
        "- `C:\\Users\\user\\Downloads\\FileZilla_3.70.4_win64_sponsored2-setup.exe`"
    )
    rewritten = _rewrite_user_prompt_for_replan(
        prompt,
        active_replan_reasons=["execution_error"],
        last_execution={
            "payload_metadata": {
                "executed_python_code": 'open_url_and_wait("https://www.google.com/search?q=filezilla%20official%20windows%20download", expected_title_tokens=["filezilla"])',
            },
            "stdout_tail": "opened browser page for screenshot-grounded UI continuation",
            "stderr_tail": "continue with latest screenshot and model-visible UI candidates",
        },
    )
    assert rewritten.startswith("REPLAN OVERRIDE FOR THIS STEP:")
    assert "Use the existing installer already present in Downloads" in rewritten
    assert "End this step only when the installed app process is running." in rewritten
    assert "This is a download-only step." not in rewritten


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
