from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import ast
import base64
import hashlib
import json

try:
    from .executor_client import ExecutorHttpClient, ExecutorStdioClient
    from .models import StepRequest, StepResponse
    from .prompting import render_prompt_bundle_from_step_request, render_web_search_decision_bundle_from_step_request
    from .runtime import AgentRuntime, ExternalCliRawPythonRuntime, GUIOwlRawPythonRuntime
    from .web_search import (
        SearXNGClient,
        WebSearchDecision,
        make_web_search_error_result,
        make_web_search_skipped_result,
        web_search_cache_key,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from executor_client import ExecutorHttpClient, ExecutorStdioClient
    from models import StepRequest, StepResponse
    from prompting import render_prompt_bundle_from_step_request, render_web_search_decision_bundle_from_step_request
    from runtime import AgentRuntime, ExternalCliRawPythonRuntime, GUIOwlRawPythonRuntime
    from web_search import (
        SearXNGClient,
        WebSearchDecision,
        make_web_search_error_result,
        make_web_search_skipped_result,
        web_search_cache_key,
    )


def _default_model_id() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root.parent / "models" / "gui-owl-1.5-8b-think-base").resolve())


def _load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_tail(path: str | Path, limit_chars: int = 2000) -> str:
    if not str(path).strip():
        return ""
    file_path = Path(path)
    if not file_path.exists() or file_path.is_dir():
        return ""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text[-limit_chars:]


def _ensure_run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_python_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code).replace("\r\n", "\n").strip().split("\n")).strip()


def _code_fingerprint(code: str) -> str:
    normalized = _normalize_python_code(code)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _is_empty_python_code(code: str) -> bool:
    return not bool(_normalize_python_code(code))


def _is_compilable_python_code(code: str) -> bool:
    normalized = _normalize_python_code(code)
    if not normalized:
        return False
    try:
        compile(normalized, "<agent-generated>", "exec")
        return True
    except SyntaxError:
        return False


def _has_meaningful_top_level_execution(code: str) -> bool:
    normalized = _normalize_python_code(code)
    if not normalized:
        return False
    try:
        tree = ast.parse(normalized, mode="exec")
    except SyntaxError:
        return False

    def _is_literal_value(node: ast.AST | None) -> bool:
        if node is None:
            return True
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(_is_literal_value(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(_is_literal_value(key) and _is_literal_value(value) for key, value in zip(node.keys, node.values))
        return False

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Pass)):
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and _is_literal_value(node.value):
            continue
        if isinstance(node, ast.AnnAssign) and _is_literal_value(node.value):
            continue
        return True
    return False


def _looks_like_non_executing_task_script(code: str) -> bool:
    normalized = _normalize_python_code(code)
    if not normalized:
        return False
    if _has_task_complete_marker(normalized):
        return False
    if not _has_meaningful_top_level_execution(normalized):
        return True
    try:
        tree = ast.parse(normalized, mode="exec")
    except SyntaxError:
        return False
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function_names:
        if isinstance(tree.body[-1], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True

        def _calls_defined_helper(stmt: ast.stmt) -> bool:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in function_names:
                    return True
            return False

        non_definition_stmts = [
            stmt
            for stmt in tree.body
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not any(_calls_defined_helper(stmt) for stmt in non_definition_stmts):
            return True
    return False


def _has_gui_install_progress_action(code: str) -> bool:
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    progress_tokens = (
        "pyautogui.",
        "click(",
        "doubleclick(",
        "press(",
        "hotkey(",
        "typewrite(",
        "write(",
        "getwindowswithtitle(",
        ".activate(",
        ".restore(",
        ".maximize(",
    )
    return any(token in normalized for token in progress_tokens)


def _has_installer_launch_action(code: str) -> bool:
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    launch_tokens = (
        "subprocess.popen(",
        "subprocess.run(",
        "os.startfile(",
        "startfile(",
    )
    installer_tokens = (
        "installer",
        "setup",
        ".exe",
        "/verysilent",
        "/silent",
        "/sp-",
        "/norestart",
    )
    if any(
        launch_token in normalized and installer_token in normalized
        for launch_token in launch_tokens
        for installer_token in installer_tokens
    ):
        return True
    for line in normalized.splitlines():
        stripped = line.strip()
        if any(token in stripped for token in launch_tokens) and any(token in stripped for token in installer_tokens):
            return True
    return False


def _has_install_progress_action(code: str) -> bool:
    return _has_installer_launch_action(code) or _has_gui_install_progress_action(code)


def _looks_like_missing_install_progress_generation(code: str, user_prompt: str) -> bool:
    if not _looks_like_existing_installer_launch_task(user_prompt):
        return False
    normalized = _normalize_python_code(code).lower()
    if not normalized or _has_task_complete_marker(normalized):
        return False
    installer_context_tokens = (
        "downloads",
        "installer",
        ".exe",
        "setup",
    )
    if not any(token in normalized for token in installer_context_tokens):
        return False
    return not _has_install_progress_action(normalized)


def _has_task_complete_marker(code: str) -> bool:
    for line in str(code or "").splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        return stripped.startswith("# task_complete")
    return False


def _is_task_complete_confirmation_script(code: str) -> bool:
    normalized = _normalize_python_code(code)
    if not normalized:
        return False
    if not _has_task_complete_marker(normalized):
        return False
    lines = normalized.splitlines()
    body_lines: list[str] = []
    marker_consumed = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not marker_consumed:
            marker_consumed = True
            continue
        if stripped.startswith("#"):
            continue
        body_lines.append(stripped)
    if len(body_lines) > 12:
        return False
    allowed_prefixes = (
        "pass",
        "print(",
        "capture_note(",
        "sleep(",
        "time.sleep(",
    )
    for line in body_lines:
        if line.startswith("import "):
            module_name = line.removeprefix("import ").split(" as ", 1)[0].split(",", 1)[0].strip()
            if module_name not in {"time"}:
                return False
            continue
        if line.startswith("from "):
            return False
        if any(line.startswith(prefix) for prefix in allowed_prefixes):
            continue
        return False
    return True


def _looks_like_completion_noop(code: str, raw_text: str) -> bool:
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    risky_tokens = (
        "pyautogui",
        "subprocess",
        "webbrowser",
        "selenium",
        "requests",
        "urllib",
        "winget",
        "pip install",
        "click(",
        "press(",
        "hotkey(",
        "typewrite(",
        "write(",
        "os.startfile",
    )
    if any(token in normalized for token in risky_tokens):
        return False
    completion_phrases = (
        "task is complete",
        "task has been completed",
        "already complete",
        "already completed",
        "completed successfully",
        "already open and on",
        "already on the",
    )
    raw_lower = str(raw_text or "").lower()
    if not any(phrase in raw_lower for phrase in completion_phrases):
        return False
    return len(normalized.splitlines()) <= 20


def _infer_response_done(*, python_code: str, raw_text: str) -> bool:
    return _is_task_complete_confirmation_script(python_code) or _looks_like_completion_noop(python_code, raw_text)


def _tail_history(history: list[str], *, limit: int = 2) -> list[str]:
    if limit <= 0:
        return []
    return list(history[-limit:])


def _history_for_step(history: list[str]) -> list[str]:
    return _tail_history(history, limit=2)


def _history_for_empty_retry(history: list[str], *, step_index: int) -> list[str]:
    retry_history = _history_for_step(history)
    retry_history.append(f"step-{step_index:03d}_empty_generation=1")
    retry_history.append("system_hint=previous model output was empty; return non-empty executable Python only")
    return retry_history


def _compact_previous_python_for_retry(code: str, *, max_chars: int = 1200) -> str:
    normalized = _normalize_python_code(code)
    if len(normalized) <= max_chars:
        return normalized
    head_limit = max_chars // 2
    tail_limit = max_chars - head_limit - 5
    return normalized[:head_limit].rstrip() + "\n...\n" + normalized[-tail_limit:].lstrip()


def _history_for_invalid_python_retry(history: list[str], *, step_index: int, previous_code: str | None = None) -> list[str]:
    retry_history = _history_for_step(history)
    retry_history.append(f"step-{step_index:03d}_invalid_python_generation=1")
    retry_history.append(
        "system_hint=previous model output was invalid, truncated, or helper-only; return one complete top-level standalone Python script only with no prose, no docstrings, and no function-only skeleton"
    )
    retry_history.append(
        "system_hint=if the previous script was cut off mid-block, continue and finish the same script idea instead of restarting from scratch; close all loops, conditionals, and try blocks"
    )
    retry_history.append(
        "system_hint=the final script must actually execute the task end-to-end, including main invocation, download completion, and explicit failure on errors"
    )
    retry_history.append(
        "system_hint=for download/install tasks, avoid helper-function scaffolding on retry; start real top-level network/file/process actions early in the script"
    )
    retry_history.append(
        "system_hint=on retry for download/install tasks, do not define main() or multiple helper functions; prefer straight-line top-level code only"
    )
    retry_history.append(
        "system_hint=do not emit import-only or setup-only code; import only modules you use and start doing the actual task within the first 25 lines"
    )
    compact_previous = _compact_previous_python_for_retry(previous_code or "")
    if compact_previous:
        retry_history.append(
            "system_hint=continue the same script idea from the previous partial Python below; return the full finished script from the beginning, not only the missing tail"
        )
        retry_history.append(f"previous_python_prefix=\n{compact_previous}")
    return retry_history


def _history_for_invalid_python_retry_with_prompt(
    history: list[str],
    *,
    user_prompt: str,
    step_index: int,
    previous_code: str | None = None,
    duplicate_generation: bool = False,
) -> list[str]:
    retry_history = _history_for_invalid_python_retry(
        history,
        step_index=step_index,
        previous_code=previous_code,
    )
    if duplicate_generation:
        retry_history.append(
            "system_hint=the previous generation repeated the exact same script as the last executed step; produce a materially different script structure and control flow"
        )
    if _looks_like_existing_installer_launch_task(user_prompt):
        retry_history.append(
            "system_hint=this is an install-and-launch chunk with an existing installer file; do not emit any download helper, URL fetch, HTML parsing, or release discovery logic"
        )
        retry_history.append(
            "system_hint=return one straight-line top-level script that finds the installer in Downloads, tries silent install switches, locates the installed app exe, launches it, and verifies the process"
        )
        retry_history.append(
            "system_hint=if no installed app exe already exists, the script must either launch the installer or actively drive an already-visible installer window; a search-only script is invalid"
        )
        retry_history.append(
            "system_hint=launching only the final app exe is not install progress; the script must either call subprocess/os.startfile on the installer path itself or use Python GUI automation on a visible installer window"
        )
        retry_history.append(
            "system_hint=keep the installer GUI script compact; use short bounded loops for repeated key presses instead of many duplicated pyautogui.press lines"
        )
        retry_history.append(
            "system_hint=if you need to advance the wizard repeatedly, prefer a loop such as for _ in range(8): pyautogui.press('enter'); time.sleep(1)"
        )
        retry_history.append(
            "system_hint=prefer common Windows silent installer switches such as /VERYSILENT, /SILENT, /SP-, and /NORESTART before any GUI automation"
        )
    return retry_history


def _looks_like_duplicate_generation(code: str, previous_executed_code: str | None) -> bool:
    if not previous_executed_code:
        return False
    normalized = _normalize_python_code(code)
    previous_normalized = _normalize_python_code(previous_executed_code)
    return bool(normalized) and normalized == previous_normalized


def _history_for_dependency_repair(history: list[str], *, failed_step_id: str) -> list[str]:
    failed_step_history = [entry for entry in history if entry.startswith(f"{failed_step_id}_")]
    if not failed_step_history:
        return _history_for_step(history)
    return failed_step_history[-2:]


_MISSING_MODULE_INSTALL_NAME_OVERRIDES = {
    "pywin32": "pywin32",
    "win32api": "pywin32",
    "win32com": "pywin32",
    "win32con": "pywin32",
    "win32event": "pywin32",
    "win32gui": "pywin32",
    "win32process": "pywin32",
    "win32ui": "pywin32",
    "pythoncom": "pywin32",
}


def _normalize_missing_module_install_name(module_name: str, install_name: str) -> str:
    normalized_module = str(module_name or "").strip().lower()
    normalized_install = str(install_name or "").strip()
    if normalized_module in _MISSING_MODULE_INSTALL_NAME_OVERRIDES:
        return _MISSING_MODULE_INSTALL_NAME_OVERRIDES[normalized_module]
    return normalized_install


_OPTIONAL_WINDOWS_GUI_MODULES = {
    "pywin32",
    "pywinauto",
    "pythoncom",
    "win32api",
    "win32com",
    "win32con",
    "win32event",
    "win32gui",
    "win32process",
    "win32ui",
}


def _missing_module_name_from_execution(last_execution: dict[str, Any]) -> str:
    error_info = dict(last_execution.get("error_info") or {})
    module_name = str(error_info.get("module_name") or "").strip()
    if module_name:
        return module_name
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    )
    for line in combined.splitlines():
        marker = "No module named "
        if marker in line:
            tail = line.split(marker, 1)[1].strip().strip("'\"")
            if tail:
                return tail
    return ""


def _looks_like_optional_windows_gui_module_failure(last_execution: dict[str, Any]) -> bool:
    module_name = _missing_module_name_from_execution(last_execution).lower()
    return module_name in _OPTIONAL_WINDOWS_GUI_MODULES


def _dependency_repair_user_prompt(*, module_name: str, install_name: str, strategy: str) -> str:
    lines = [
        "Return executable Python only.",
        "Repair the reported missing Python dependency only.",
        "Do not continue the main GUI task in this response.",
        f"Missing module: {module_name}",
        f"Preferred install target: {install_name}",
    ]
    if strategy == "pip_install":
        lines.extend(
            [
                "Use sys.executable -m pip install <package> first.",
                "After installation, verify the import in the same script and exit non-zero if the import still fails.",
                "Do not relaunch the installer or repeat the original task script here.",
            ]
        )
        if module_name.strip().lower() == "pywin32":
            lines.extend(
                [
                    "Important: `pywin32` is a distribution name, not a reliable top-level import target.",
                    "Do not write `import pywin32` after installation. Either verify a concrete Win32 module such as `win32gui` or `pythoncom`, or finish after successful pip install and let the next task step avoid `import pywin32`.",
                ]
            )
    else:
        lines.extend(
            [
                "Use a shell or subprocess fallback to install the dependency, then verify the import.",
                "Do not relaunch the installer or repeat the original task script here.",
            ]
        )
    return "\n".join(lines)


def _history_for_web_search(history: list[str], *, limit: int = 4) -> list[str]:
    base = _history_for_step(history)
    filtered = [entry for entry in history if "_web_search_" in entry or entry.startswith("system_hint=")]
    if not filtered:
        return base
    combined = base + filtered[-2:]
    return list(dict.fromkeys(combined))[-limit:]


def _retry_token_budget(max_new_tokens: int) -> int:
    base = int(max_new_tokens)
    return max(base, min(512, base + 128))


def _state_visual_hash(state: dict[str, Any]) -> str | None:
    screenshot_base64 = str(state.get("screenshot_base64") or "").strip()
    if screenshot_base64:
        return hashlib.sha1(screenshot_base64.encode("utf-8")).hexdigest()[:12]
    screenshot_path = str(state.get("screenshot_path") or "").strip()
    if screenshot_path:
        path = Path(screenshot_path)
        if path.exists() and path.is_file():
            return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    return None


def _looks_like_download_or_install_task(user_prompt: str) -> bool:
    text = str(user_prompt or "").lower()
    keywords = (
        "download",
        "installer",
        "install",
        "setup",
        "다운",
        "다운로드",
        "설치",
        "인스톨",
        "setup.exe",
    )
    return any(keyword in text for keyword in keywords)


def _looks_like_existing_installer_launch_task(user_prompt: str) -> bool:
    text = str(user_prompt or "").lower()
    launch_markers = (
        "downloads\\",
        "downloads/",
        "downloaded installer",
        "already exists in downloads",
        "launch the installed app",
        "run it, finish the installation",
        "process is running",
    )
    return _looks_like_download_or_install_task(text) and any(marker in text for marker in launch_markers)


def _looks_like_installer_timeout(last_execution: dict[str, Any], python_code: str, user_prompt: str) -> bool:
    if not _looks_like_existing_installer_launch_task(user_prompt):
        return False
    if not last_execution:
        return False
    if not bool(last_execution.get("timed_out")) and str((last_execution.get("error_info") or {}).get("kind") or "").lower() != "timeout":
        return False
    normalized = _normalize_python_code(python_code).lower()
    installer_tokens = (
        "/verysilent",
        "/silent",
        "subprocess.run(",
        ".exe",
    )
    return any(token in normalized for token in installer_tokens)


def _looks_like_installer_launched_but_app_not_found(last_execution: dict[str, Any], python_code: str, user_prompt: str) -> bool:
    if not _looks_like_existing_installer_launch_task(user_prompt):
        return False
    if not last_execution:
        return False
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    if "found installer:" not in combined:
        return False
    failure_markers = (
        "installation failed or app not found",
        "app exited immediately",
        "app exited unexpectedly",
    )
    if not any(marker in combined for marker in failure_markers):
        return False
    normalized = _normalize_python_code(python_code).lower()
    return "subprocess.popen(" in normalized and ".exe" in normalized


def _looks_like_incomplete_install_attempt(last_execution: dict[str, Any], python_code: str, user_prompt: str) -> bool:
    if not _looks_like_existing_installer_launch_task(user_prompt):
        return False
    if int(last_execution.get("return_code", 0) or 0) != 0:
        return False
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    if "found installer:" not in combined:
        return False
    success_markers = (
        "found installed exe:",
        "installed exe exists",
        "process running",
        "installation complete",
        "already installed",
    )
    if any(marker in combined for marker in success_markers):
        return False
    normalized = _normalize_python_code(python_code).lower()
    if not _has_install_progress_action(normalized):
        return True
    install_tokens = (
        "subprocess.popen(",
        "subprocess.run(",
        "os.startfile(",
        "/verysilent",
        "/silent",
        ".exe",
    )
    return any(token in normalized for token in install_tokens)


def _looks_like_install_path_scan_failure(last_execution: dict[str, Any]) -> bool:
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    markers = (
        "filenotfounderror",
        "winerror 3",
        "cloudstore",
        "os.scandir",
        "pathlib.py",
        "rglob(",
    )
    return any(marker in combined for marker in markers)


def _looks_like_truncated_gui_repetition_failure(last_execution: dict[str, Any]) -> bool:
    stderr_tail = str(last_execution.get("stderr_tail") or "").lower()
    return "nameerror" in stderr_tail and "name 'py' is not defined" in stderr_tail


def _rewrite_user_prompt_for_replan(
    user_prompt: str,
    *,
    active_replan_reasons: list[str],
    last_execution: dict[str, Any],
) -> str:
    prompt = str(user_prompt or "").strip()
    unique_reasons = list(dict.fromkeys(str(reason).strip() for reason in active_replan_reasons if str(reason).strip()))
    if not prompt or not unique_reasons:
        return prompt

    if not _looks_like_existing_installer_launch_task(prompt):
        return prompt

    optional_gui_dependency_failure = _looks_like_optional_windows_gui_module_failure(last_execution)
    install_path_scan_failure = _looks_like_install_path_scan_failure(last_execution)
    truncated_gui_repetition_failure = _looks_like_truncated_gui_repetition_failure(last_execution)
    install_replan = (
        "installer_timeout" in unique_reasons
        or "installer_app_not_found" in unique_reasons
        or "execution_error" in unique_reasons
        or optional_gui_dependency_failure
    )
    override_lines = [
        "REPLAN OVERRIDE FOR THIS STEP:",
        "Return executable Python only.",
    ]
    if "repeated_code_execution" in unique_reasons:
        override_lines.append("Produce a materially different script from the previous attempt.")
    if install_replan:
        override_lines.append("Do not repeat the same silent installer launch-and-scan script.")
        override_lines.append("Do not retry `/VERYSILENT` or `/SILENT` first on this step.")
        override_lines.append(
            "First inspect the current screenshot and desktop state for an installer wizard, UAC prompt, license dialog, destination dialog, or completion dialog, and use Python GUI automation to advance it."
        )
        override_lines.append(
            "Use Python-accessible GUI control such as pyautogui, pygetwindow, ctypes, or psutil if available. Do not ask a human to click."
        )
        override_lines.append(
            "Import only GUI modules you actually call. Prefer the smallest available toolset first, such as pyautogui plus the standard library."
        )
        override_lines.append(
            "Launching only the final app executable is not enough. This step must either launch the installer path itself or operate a visible installer window with Python GUI automation."
        )
        override_lines.append(
            "Keep the script compact. Use short loops for repeated installer key presses instead of many duplicated pyautogui.press lines."
        )
        if optional_gui_dependency_failure:
            override_lines.append(
                "Previous attempt failed while importing optional Windows GUI modules. Do not directly import win32gui, win32con, win32api, pythoncom, pywinauto, or similar optional packages unless the script first proves they already import successfully."
            )
            override_lines.append(
                "Prefer pyautogui, pygetwindow, psutil, and the standard library. If an optional helper import fails, catch it and fall back instead of aborting the whole step at startup."
            )
        override_lines.append("Keep the script short and avoid deeply nested repeated retry loops.")
        override_lines.append("Only after handling visible installer UI may you scan install paths, launch the installed app, and verify the process.")
        override_lines.append("If no installer window is visible, then check running processes and common install paths before relaunching the installer once.")
        if install_path_scan_failure:
            override_lines.append(
                "Previous attempt failed while recursively scanning broad Windows directories. Do not rglob the whole of LOCALAPPDATA or Program Files."
            )
            override_lines.append(
                "Check only likely install directories such as LOCALAPPDATA\\\\DBeaver, LOCALAPPDATA\\\\Programs\\\\DBeaver, Program Files\\\\DBeaver, and Program Files (x86)\\\\DBeaver, or use os.walk with onerror handling."
            )
        if truncated_gui_repetition_failure:
            override_lines.append(
                "Previous attempt appears to have been cut off mid-script while repeating GUI actions. Rewrite it as a shorter complete script from the beginning."
            )
            override_lines.append(
                "If you need many Enter presses, use a bounded loop like for _ in range(8): pyautogui.press('enter'); time.sleep(1) rather than spelling them out line by line."
            )

    stdout_tail = str(last_execution.get("stdout_tail") or "").strip()
    stderr_tail = str(last_execution.get("stderr_tail") or "").strip()
    if stdout_tail:
        override_lines.append(f"Previous stdout summary: {stdout_tail[-240:]}")
    if stderr_tail:
        override_lines.append(f"Previous stderr summary: {stderr_tail[-240:]}")

    if install_replan:
        override_lines.extend(
            [
                "Use the existing installer already present in Downloads; do not add download, URL discovery, or HTML parsing logic.",
                "If an installer or completion window is visible, operate that window directly with Python clicks, key presses, or focus changes instead of launching the installer again.",
                "If no installer window is visible, inspect common install paths and running processes for the target app before relaunching the installer once without silent-mode repetition.",
                "End this step only when the installed app process is running.",
            ]
        )
        return "\n".join(override_lines)

    return "\n".join(override_lines) + "\n\n" + prompt


def _should_omit_screenshot_for_generation(*, user_prompt: str, last_execution: dict[str, Any]) -> bool:
    if _looks_like_existing_installer_launch_task(user_prompt):
        return False
    return _looks_like_download_or_install_task(user_prompt) and not bool(last_execution)


def _generation_screenshot_fields(
    *,
    state: dict[str, Any],
    user_prompt: str,
    last_execution: dict[str, Any],
) -> tuple[Any, Any, Any]:
    if _should_omit_screenshot_for_generation(user_prompt=user_prompt, last_execution=last_execution):
        return None, None, None
    return (
        state.get("screenshot_path"),
        state.get("screenshot_base64"),
        state.get("screenshot_media_type"),
    )


def _looks_like_opened_page_only_step(python_code: str) -> bool:
    normalized = _normalize_python_code(python_code).lower()
    if not normalized:
        return False
    opened_url = (
        "webbrowser.open(" in normalized
        or "driver.get(" in normalized
        or ".goto(" in normalized
        or ('os.startfile(' in normalized and "http" in normalized)
    )
    if not opened_url:
        return False
    stronger_progress_tokens = (
        ".click(",
        "click(",
        "downloads",
        "path(",
        "requests.",
        "urllib",
        "urlretrieve",
        "urlopen(",
        "powershell",
        "invoke-webrequest",
        "start-bitstransfer",
        "winget",
        ".exe",
        "glob(",
        "exists(",
        "stat(",
        "subprocess.run(",
        "shutil.move(",
        "rename(",
        "listdir(",
        "iterdir(",
    )
    return not any(token in normalized for token in stronger_progress_tokens)


def _looks_like_reported_failure(last_execution: dict[str, Any]) -> bool:
    if not last_execution:
        return False
    error_info = last_execution.get("error_info")
    if error_info:
        return True
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    if not combined.strip():
        return False
    failure_markers = (
        "error:",
        "error ",
        "exception",
        "traceback",
        "failed",
        "failure",
        "non-zero exit status",
        "download failed",
        "not found",
        "cannot find",
        "timed out",
        "invoke-webrequest",
        "오류",
        "실패",
    )
    return any(marker in combined for marker in failure_markers)


def _looks_like_direct_download_url_404(last_execution: dict[str, Any], python_code: str) -> bool:
    if not last_execution:
        return False
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    if "404" not in combined and "not found" not in combined:
        return False
    normalized = _normalize_python_code(python_code).lower()
    if "http" not in normalized or ".exe" not in normalized:
        return False
    direct_download_tokens = (
        "requests.get(",
        "urllib",
        "downloadfile(",
        "urlretrieve(",
        "invoke-webrequest",
        "start-bitstransfer",
        "webclient",
    )
    return any(token in normalized for token in direct_download_tokens)


def _looks_like_direct_download_url_403(last_execution: dict[str, Any], python_code: str) -> bool:
    if not last_execution:
        return False
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    if "403" not in combined and "forbidden" not in combined:
        return False
    normalized = _normalize_python_code(python_code).lower()
    if "http" not in normalized or ".exe" not in normalized:
        return False
    direct_download_tokens = (
        "requests.get(",
        "urllib",
        "downloadfile(",
        "urlretrieve(",
        "invoke-webrequest",
        "start-bitstransfer",
    )
    return any(token in normalized for token in direct_download_tokens)


def _looks_like_installer_url_discovery_failure(last_execution: dict[str, Any], python_code: str) -> bool:
    if not last_execution:
        return False
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    discovery_markers = (
        "failed to find installer url",
        "could not find installer url",
        "no installer url found",
        "failed to find download url",
        "could not find download url",
        "no download url found",
    )
    if not any(marker in combined for marker in discovery_markers):
        return False
    normalized = _normalize_python_code(python_code).lower()
    if ".exe" not in normalized:
        return False
    discovery_tokens = (
        "urllib.request.urlopen(",
        "requests.get(",
        ".read().decode(",
        "re.findall(",
        "findall(",
        "href=",
        "html",
    )
    return any(token in normalized for token in discovery_tokens)


def _looks_like_download_chunk_completed(*, user_prompt: str, last_execution: dict[str, Any]) -> bool:
    if not _looks_like_download_or_install_task(user_prompt):
        return False
    if int(last_execution.get("return_code", 0) or 0) != 0:
        return False
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    success_markers = (
        "downloaded:",
        "downloaded successfully:",
        "existing installer found:",
        "using existing installer:",
    )
    if not any(marker in combined for marker in success_markers):
        return False
    prompt_lower = str(user_prompt or "").lower()
    return any(
        marker in prompt_lower
        for marker in (
            "success target",
            "installer `.exe` exists",
            "installer `.exe`가 있",
            "downloads\\",
        )
    )


def build_executor_client(*, endpoint: str | None, mcp_command: list[str] | None, mcp_cwd: str | None):
    if bool(endpoint) == bool(mcp_command):
        raise RuntimeError("provide exactly one of endpoint or mcp_command")
    if endpoint:
        return ExecutorHttpClient(endpoint)
    return ExecutorStdioClient(mcp_command or [], cwd=mcp_cwd)


def _make_generation_context(
    *,
    run_dir: str | Path,
    step_id: str,
    request_kind: str,
    step_index: int,
) -> dict[str, Any]:
    return {
        "run_dir": str(Path(run_dir)),
        "step_id": str(step_id),
        "request_kind": str(request_kind),
        "step_index": int(step_index),
    }


def generate_step_response(
    runtime: AgentRuntime,
    request: StepRequest,
    *,
    max_new_tokens: int,
    generation_context: dict[str, Any] | None = None,
) -> StepResponse:
    image_bytes = None
    if request.screenshot_base64:
        image_bytes = base64.b64decode(request.screenshot_base64)
    bundle = render_prompt_bundle_from_step_request(request)
    generated = runtime.generate_code(
        prompt_bundle=bundle,
        image_path=request.screenshot_path,
        image_bytes=image_bytes,
        use_blank_image=not bool(request.screenshot_path or image_bytes) and not _looks_like_download_or_install_task(request.user_prompt),
        max_new_tokens=max_new_tokens,
        generation_context=generation_context,
    )
    return StepResponse(
        python_code=generated.code,
        raw_text=generated.raw_text,
        model_id=generated.model_id,
        step_index=request.step_index,
        done=_infer_response_done(python_code=generated.code, raw_text=generated.raw_text),
        notes=[],
    )


def generate_web_search_decision(
    runtime: AgentRuntime,
    request: StepRequest,
    *,
    max_new_tokens: int,
    decision_max_new_tokens: int,
    use_image: bool,
    reasoning_enabled: bool,
    web_search_max_uses: int,
    web_search_uses: int,
    web_search_queries: list[str],
    generation_context: dict[str, Any] | None = None,
) -> tuple[WebSearchDecision, dict[str, Any]]:
    image_bytes = None
    image_path = None
    if use_image and request.screenshot_base64:
        image_bytes = base64.b64decode(request.screenshot_base64)
    if use_image and request.screenshot_path:
        image_path = request.screenshot_path
    bundle = render_web_search_decision_bundle_from_step_request(
        request,
        reasoning_enabled=reasoning_enabled,
        web_search_max_uses=web_search_max_uses,
        web_search_uses=web_search_uses,
        web_search_queries=web_search_queries,
    )
    generated = runtime.generate_text(
        prompt_bundle=bundle,
        image_path=image_path,
        image_bytes=image_bytes,
        use_blank_image=False,
        max_new_tokens=min(int(max_new_tokens), int(decision_max_new_tokens)),
        generation_context=generation_context,
    )
    decision = WebSearchDecision.from_text(generated.text)
    return decision, generated.to_dict()


def _extract_last_execution(exec_result: dict[str, Any]) -> dict[str, Any]:
    record = dict(exec_result.get("record", {}))
    return {
        **record,
        "stdout_tail": exec_result.get("stdout_tail") or _read_tail(record.get("stdout_path", "")),
        "stderr_tail": exec_result.get("stderr_tail") or _read_tail(record.get("stderr_path", "")),
        "error_info": exec_result.get("error_info"),
    }


def _extract_state(exec_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "screenshot_path": exec_result.get("screenshot_path"),
        "screenshot_base64": exec_result.get("screenshot_base64"),
        "screenshot_media_type": exec_result.get("screenshot_media_type"),
        "observation_text": exec_result.get("observation_text"),
    }


def _execute_code_step(
    *,
    executor_client,
    root: Path,
    step_id: str,
    python_code: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    step_dir = root / "steps" / step_id
    exec_result = executor_client.execute(
        python_code=python_code,
        run_dir=str(step_dir),
        step_id=step_id,
        metadata=metadata,
    )
    _write_json(root / "responses" / f"{step_id}.executor.json", exec_result)
    return exec_result, _extract_last_execution(exec_result), _extract_state(exec_result)


def _maybe_perform_web_search(
    *,
    runtime: AgentRuntime,
    request: StepRequest,
    root: Path,
    step_id: str,
    history: list[str],
    max_new_tokens: int,
    web_search_decision_use_image: bool,
    web_search_decision_reasoning_enabled: bool,
    web_search_decision_max_new_tokens: int,
    web_search_max_uses: int,
    web_search_uses: int,
    web_search_queries: list[str],
    web_search_cache: dict[str, dict[str, Any]],
    searxng_client: SearXNGClient,
    web_search_top_k: int,
    searxng_preferred_engines: list[str],
) -> tuple[dict[str, Any], int, list[str]]:
    search_request = StepRequest(
        user_prompt=request.user_prompt,
        policy=request.policy,
        execution_style=request.execution_style,
        request_kind="web_search_decision",
        repair_context={},
        replan_requested=request.replan_requested,
        replan_reasons=request.replan_reasons,
        strong_visual_grounding=request.strong_visual_grounding,
        reasoning_enabled=web_search_decision_reasoning_enabled,
        screenshot_path=request.screenshot_path if web_search_decision_use_image else None,
        screenshot_base64=request.screenshot_base64 if web_search_decision_use_image else None,
        screenshot_media_type=request.screenshot_media_type if web_search_decision_use_image else None,
        observation_text=request.observation_text,
        web_search_context={},
        recent_history=_history_for_web_search(history),
        last_execution=request.last_execution,
        step_index=request.step_index,
    )
    decision_request_path = root / "payloads" / f"{step_id}.web-search-decision.request.json"
    _write_json(decision_request_path, search_request.to_dict())
    decision, generated = generate_web_search_decision(
        runtime,
        search_request,
        max_new_tokens=max_new_tokens,
        decision_max_new_tokens=web_search_decision_max_new_tokens,
        use_image=web_search_decision_use_image,
        reasoning_enabled=web_search_decision_reasoning_enabled,
        web_search_max_uses=web_search_max_uses,
        web_search_uses=web_search_uses,
        web_search_queries=web_search_queries[-3:],
        generation_context=_make_generation_context(
            run_dir=root,
            step_id=f"{step_id}.web-search-decision",
            request_kind=search_request.request_kind,
            step_index=search_request.step_index,
        ),
    )
    _write_json(
        root / "responses" / f"{step_id}.web-search-decision.response.json",
        {
            "decision": decision.to_dict(),
            "model_response": generated,
        },
    )

    if not decision.use_web_search:
        return {}, web_search_uses, web_search_queries

    if not decision.query:
        return {}, web_search_uses, web_search_queries

    cache_key = web_search_cache_key(
        decision.query,
        allowed_domains=decision.allowed_domains,
        blocked_domains=decision.blocked_domains,
        preferred_engines=searxng_preferred_engines,
    )
    if cache_key in web_search_cache:
        cached_payload = dict(web_search_cache[cache_key])
        cached_payload["cached"] = True
        cached_payload["reason"] = decision.reason or cached_payload.get("reason", "")
        _write_json(root / "responses" / f"{step_id}.web-search-result.json", cached_payload)
        history.append(f"{step_id}_web_search_cached={decision.query}")
        return cached_payload, web_search_uses, web_search_queries

    if web_search_uses >= web_search_max_uses:
        skipped = make_web_search_skipped_result(
            query=decision.query,
            allowed_domains=decision.allowed_domains,
            blocked_domains=decision.blocked_domains,
            reason=decision.reason or "search requested but per-run web search limit was reached",
            status="skipped_max_uses",
        ).to_dict()
        _write_json(root / "responses" / f"{step_id}.web-search-result.json", skipped)
        history.append(f"{step_id}_web_search_skipped=max_uses")
        return skipped, web_search_uses, web_search_queries

    try:
        result = searxng_client.search(
            query=decision.query,
            top_k=web_search_top_k,
            allowed_domains=decision.allowed_domains,
            blocked_domains=decision.blocked_domains,
            preferred_engines=searxng_preferred_engines,
        ).to_dict()
    except Exception as exc:
        result = make_web_search_error_result(
            query=decision.query,
            allowed_domains=decision.allowed_domains,
            blocked_domains=decision.blocked_domains,
            reason=decision.reason or "web search failed",
            error=str(exc),
        ).to_dict()
        _write_json(root / "responses" / f"{step_id}.web-search-result.json", result)
        history.append(f"{step_id}_web_search_error={decision.query}")
        return result, web_search_uses + 1, web_search_queries + [decision.query]

    result["reason"] = decision.reason or result.get("reason", "")
    web_search_cache[cache_key] = dict(result)
    _write_json(root / "responses" / f"{step_id}.web-search-result.json", result)
    history.append(f"{step_id}_web_search_query={decision.query}")
    history.append(f"{step_id}_web_search_results={int(result.get('result_count', 0))}")
    return result, web_search_uses + 1, web_search_queries + [decision.query]


def _attempt_dependency_repair(
    *,
    runtime: AgentRuntime,
    executor_client,
    root: Path,
    user_prompt: str,
    policy: dict[str, Any],
    strong_visual_grounding: bool,
    reasoning_enabled: bool,
    state: dict[str, Any],
    history: list[str],
    last_execution: dict[str, Any],
    original_response: StepResponse,
    original_step_id: str,
    step_index: int,
    max_new_tokens: int,
    repair_attempt_index: int,
    allow_shell_fallback: bool,
) -> dict[str, Any]:
    error_info = dict(last_execution.get("error_info") or {})
    module_name = str(error_info.get("module_name") or "").strip()
    install_name = _normalize_missing_module_install_name(
        module_name,
        str(error_info.get("install_name") or module_name).strip(),
    )
    if not install_name:
        return {"handled": False}

    strategies = ["pip_install"]
    if allow_shell_fallback:
        strategies.append("shell_fallback")
    executions_added = 0

    for strategy_index, strategy in enumerate(strategies):
        repair_request = StepRequest(
            user_prompt=_dependency_repair_user_prompt(
                module_name=module_name or install_name,
                install_name=install_name,
                strategy=strategy,
            ),
            policy=policy,
            execution_style="python_first",
            request_kind="dependency_repair",
            repair_context={
                "reason": "missing_python_module",
                "module_name": module_name,
                "install_name": install_name,
                "repair_strategy": strategy,
                "repair_attempt_index": repair_attempt_index,
                "failed_python_code": original_response.python_code,
                "failed_step_id": original_step_id,
                "stderr_tail": last_execution.get("stderr_tail"),
            },
            strong_visual_grounding=strong_visual_grounding,
            reasoning_enabled=reasoning_enabled,
            screenshot_path=state.get("screenshot_path"),
            screenshot_base64=state.get("screenshot_base64"),
            screenshot_media_type=state.get("screenshot_media_type"),
            observation_text=state.get("observation_text"),
            recent_history=_history_for_dependency_repair(history, failed_step_id=original_step_id),
            last_execution=last_execution,
            step_index=step_index,
        )
        repair_request_name = f"{original_step_id}.repair-{repair_attempt_index:02d}-{strategy_index:02d}"
        _write_json(root / "payloads" / f"{repair_request_name}.request.json", repair_request.to_dict())
        repair_response = generate_step_response(
            runtime,
            repair_request,
            max_new_tokens=max_new_tokens,
            generation_context=_make_generation_context(
                run_dir=root,
                step_id=repair_request_name,
                request_kind=repair_request.request_kind,
                step_index=repair_request.step_index,
            ),
        )
        repair_response.notes.append("dependency_repair_mode=true")
        repair_response.notes.append(f"dependency_repair_strategy={strategy}")
        _write_json(root / "responses" / f"{repair_request_name}.response.json", repair_response.to_dict())

        repair_exec_result, repair_last_execution, repair_state = _execute_code_step(
            executor_client=executor_client,
            root=root,
            step_id=repair_request_name,
            python_code=repair_response.python_code,
            metadata={
                "agent_response": repair_response.to_dict(),
                "repair_context": repair_request.repair_context,
                "step_index": step_index,
                "agent_session_id": root.name,
                "agent_step_dir": str(root / "steps" / repair_request_name),
            },
        )
        executions_added += 1
        history.append(f"{repair_request_name}_return_code={repair_last_execution.get('return_code', 'unknown')}")
        history.append(f"{repair_request_name}_repair_strategy={strategy}")

        if int(repair_last_execution.get("return_code", 1)) != 0:
            last_execution = repair_last_execution
            state = repair_state
            continue

        retry_step_id = f"{original_step_id}.retry-{repair_attempt_index:02d}-{strategy_index:02d}"
        retry_exec_result, retry_last_execution, retry_state = _execute_code_step(
            executor_client=executor_client,
            root=root,
            step_id=retry_step_id,
            python_code=original_response.python_code,
            metadata={
                "agent_response": original_response.to_dict(),
                "dependency_repair_retry": True,
                "repair_context": repair_request.repair_context,
                "step_index": step_index,
                "agent_session_id": root.name,
                "agent_step_dir": str(root / "steps" / retry_step_id),
            },
        )
        executions_added += 1
        history.append(f"{retry_step_id}_return_code={retry_last_execution.get('return_code', 'unknown')}")
        history.append(f"{retry_step_id}_code_fingerprint={_code_fingerprint(original_response.python_code)}")

        retry_error_info = dict(retry_last_execution.get("error_info") or {})
        if int(retry_last_execution.get("return_code", 1)) == 0:
            return {
                "handled": True,
                "success": True,
                "executions_added": executions_added,
                "repair_response": repair_response.to_dict(),
                "last_execution": retry_last_execution,
                "state": retry_state,
            }
        if retry_error_info.get("kind") != "missing_python_module":
            return {
                "handled": True,
                "success": False,
                "executions_added": executions_added,
                "repair_response": repair_response.to_dict(),
                "last_execution": retry_last_execution,
                "state": retry_state,
            }
        last_execution = retry_last_execution
        state = retry_state

    return {
        "handled": True,
        "success": False,
        "executions_added": executions_added,
        "last_execution": last_execution,
        "state": state,
    }


def run_agent_control_loop(
    *,
    runtime: AgentRuntime,
    executor_client,
    user_prompt: str,
    policy: dict[str, Any],
    run_dir: str | Path,
    max_iterations: int,
    max_new_tokens: int,
    execution_style: str = "python_first",
    strong_visual_grounding: bool = False,
    reasoning_enabled: bool = False,
    replan_enabled: bool = False,
    replan_max_attempts: int = 1,
    web_search_enabled: bool = False,
    web_search_engine: str = "searxng",
    searxng_base_url: str = "http://127.0.0.1:8080",
    web_search_top_k: int = 5,
    web_search_max_uses: int = 3,
    web_search_timeout_s: float = 10.0,
    searxng_preferred_engines: list[str] | None = None,
    web_search_decision_use_image: bool = False,
    web_search_decision_reasoning_enabled: bool = False,
    web_search_decision_max_new_tokens: int = 64,
    dependency_repair_enabled: bool = False,
    dependency_repair_max_attempts: int = 2,
    dependency_repair_allow_shell_fallback: bool = False,
) -> dict[str, Any]:
    root = _ensure_run_dir(run_dir)
    (root / "payloads").mkdir(exist_ok=True)
    (root / "responses").mkdir(exist_ok=True)
    (root / "steps").mkdir(exist_ok=True)
    _write_json(
        root / "session.json",
        {
            "user_prompt": user_prompt,
            "policy": policy,
            "execution_style": execution_style,
        },
    )
    if web_search_enabled and str(web_search_engine).strip().lower() != "searxng":
        raise RuntimeError("only the searxng web search engine is supported")

    state = executor_client.observe()
    _write_json(root / "observe-000.json", state)

    last_execution: dict[str, Any] = {}
    history: list[str] = []
    final_response: dict[str, Any] | None = None
    generated_steps = 0
    executed_steps = 0
    stopped_reason: str | None = None
    previous_executed_code: str | None = None
    previous_visual_hash = _state_visual_hash(state)
    pending_replan_reasons: list[str] = []
    replans_used = 0
    web_search_uses = 0
    web_search_queries: list[str] = []
    web_search_cache: dict[str, dict[str, Any]] = {}
    dependency_repairs_used = 0
    empty_generation_retries_used = 0
    invalid_generation_retries_used = 0
    normalized_preferred_search_engines = [str(engine).strip().lower() for engine in (searxng_preferred_engines or []) if str(engine).strip()]
    searxng_client = SearXNGClient(base_url=searxng_base_url, timeout_s=web_search_timeout_s) if web_search_enabled else None

    for step_index in range(max_iterations):
        active_replan_reasons = list(pending_replan_reasons)
        pending_replan_reasons = []
        request_user_prompt = _rewrite_user_prompt_for_replan(
            user_prompt,
            active_replan_reasons=active_replan_reasons,
            last_execution=last_execution,
        )
        screenshot_path, screenshot_base64, screenshot_media_type = _generation_screenshot_fields(
            state=state,
            user_prompt=user_prompt,
            last_execution=last_execution,
        )
        request = StepRequest(
            user_prompt=request_user_prompt,
            policy=policy,
            execution_style=execution_style,
            replan_requested=bool(active_replan_reasons),
            replan_reasons=active_replan_reasons,
            strong_visual_grounding=strong_visual_grounding,
            reasoning_enabled=reasoning_enabled,
            screenshot_path=screenshot_path,
            screenshot_base64=screenshot_base64,
            screenshot_media_type=screenshot_media_type,
            observation_text=state.get("observation_text"),
            web_search_context={},
            recent_history=_history_for_step(history),
            last_execution=last_execution,
            step_index=step_index,
        )
        step_id = f"step-{step_index:03d}"
        if web_search_enabled and request.request_kind == "task_step" and searxng_client is not None:
            web_search_context, web_search_uses, web_search_queries = _maybe_perform_web_search(
                runtime=runtime,
                request=request,
                root=root,
                step_id=step_id,
                history=history,
                max_new_tokens=max_new_tokens,
                web_search_decision_use_image=web_search_decision_use_image,
                web_search_decision_reasoning_enabled=web_search_decision_reasoning_enabled,
                web_search_decision_max_new_tokens=web_search_decision_max_new_tokens,
                web_search_max_uses=web_search_max_uses,
                web_search_uses=web_search_uses,
                web_search_queries=web_search_queries,
                web_search_cache=web_search_cache,
                searxng_client=searxng_client,
                web_search_top_k=web_search_top_k,
                searxng_preferred_engines=normalized_preferred_search_engines,
            )
            request.web_search_context = web_search_context
        request_path = root / "payloads" / f"step-{step_index:03d}.request.json"
        _write_json(request_path, request.to_dict())

        response = generate_step_response(
            runtime,
            request,
            max_new_tokens=max_new_tokens,
            generation_context=_make_generation_context(
                run_dir=root,
                step_id=step_id,
                request_kind=request.request_kind,
                step_index=request.step_index,
            ),
        )
        generated_steps += 1
        normalized_code = _normalize_python_code(response.python_code)
        response_path = root / "responses" / f"step-{step_index:03d}.response.json"
        if _is_empty_python_code(response.python_code):
            empty_attempt_path = root / "responses" / f"step-{step_index:03d}.empty-attempt-00.response.json"
            response.notes.append("empty_generation_detected")
            _write_json(empty_attempt_path, response.to_dict())
            retry_request = StepRequest(
                user_prompt=user_prompt,
                policy=policy,
                execution_style=execution_style,
                replan_requested=bool(active_replan_reasons),
                replan_reasons=active_replan_reasons,
                strong_visual_grounding=strong_visual_grounding,
                reasoning_enabled=reasoning_enabled,
                screenshot_path=screenshot_path,
                screenshot_base64=screenshot_base64,
                screenshot_media_type=screenshot_media_type,
                observation_text=state.get("observation_text"),
                web_search_context=request.web_search_context,
                recent_history=_history_for_empty_retry(history, step_index=step_index),
                last_execution=last_execution,
                step_index=step_index,
            )
            retry_request_path = root / "payloads" / f"step-{step_index:03d}.empty-retry-00.request.json"
            _write_json(retry_request_path, retry_request.to_dict())
            retry_response = generate_step_response(
                runtime,
                retry_request,
                max_new_tokens=max_new_tokens,
                generation_context=_make_generation_context(
                    run_dir=root,
                    step_id=f"{step_id}.empty-retry-00",
                    request_kind=retry_request.request_kind,
                    step_index=retry_request.step_index,
                ),
            )
            generated_steps += 1
            empty_generation_retries_used += 1
            retry_response.notes.append("retry_due_to_empty_generation")
            retry_normalized_code = _normalize_python_code(retry_response.python_code)
            retry_response_path = root / "responses" / f"step-{step_index:03d}.empty-retry-00.response.json"
            _write_json(retry_response_path, retry_response.to_dict())
            if _is_empty_python_code(retry_response.python_code):
                retry_response.notes.append("stopped_due_to_empty_generation")
                final_response = retry_response.to_dict()
                _write_json(retry_response_path, retry_response.to_dict())
                _write_json(response_path, retry_response.to_dict())
                stopped_reason = "empty_generation"
                history.append(f"step-{step_index:03d}_stopped=empty_generation")
                break
            response = retry_response
            normalized_code = retry_normalized_code

        duplicate_generation = _looks_like_duplicate_generation(response.python_code, previous_executed_code)
        invalid_generation = (
            not _is_compilable_python_code(response.python_code)
            or _looks_like_non_executing_task_script(response.python_code)
            or _looks_like_missing_install_progress_generation(response.python_code, user_prompt)
            or duplicate_generation
        )
        if invalid_generation:
            invalid_attempt_path = root / "responses" / f"step-{step_index:03d}.invalid-attempt-00.response.json"
            if not _is_compilable_python_code(response.python_code):
                response.notes.append("invalid_python_generation_detected")
            if _looks_like_non_executing_task_script(response.python_code):
                response.notes.append("non_executing_python_generation_detected")
            if _looks_like_missing_install_progress_generation(response.python_code, user_prompt):
                response.notes.append("missing_install_progress_generation_detected")
            if duplicate_generation:
                response.notes.append("duplicate_python_generation_detected")
            _write_json(invalid_attempt_path, response.to_dict())
            retry_request = StepRequest(
                user_prompt=user_prompt,
                policy=policy,
                execution_style=execution_style,
                replan_requested=bool(active_replan_reasons),
                replan_reasons=active_replan_reasons,
                strong_visual_grounding=strong_visual_grounding,
                reasoning_enabled=reasoning_enabled,
                screenshot_path=screenshot_path,
                screenshot_base64=screenshot_base64,
                screenshot_media_type=screenshot_media_type,
                observation_text=state.get("observation_text"),
                web_search_context=request.web_search_context,
                recent_history=_history_for_invalid_python_retry_with_prompt(
                    history,
                    user_prompt=user_prompt,
                    step_index=step_index,
                    previous_code=response.raw_text,
                    duplicate_generation=duplicate_generation,
                ),
                last_execution=last_execution,
                step_index=step_index,
            )
            retry_request_path = root / "payloads" / f"step-{step_index:03d}.invalid-retry-00.request.json"
            _write_json(retry_request_path, retry_request.to_dict())
            retry_response = generate_step_response(
                runtime,
                retry_request,
                max_new_tokens=_retry_token_budget(max_new_tokens),
                generation_context=_make_generation_context(
                    run_dir=root,
                    step_id=f"{step_id}.invalid-retry-00",
                    request_kind=retry_request.request_kind,
                    step_index=retry_request.step_index,
                ),
            )
            generated_steps += 1
            invalid_generation_retries_used += 1
            retry_response.notes.append("retry_due_to_invalid_python_generation")
            retry_normalized_code = _normalize_python_code(retry_response.python_code)
            retry_response_path = root / "responses" / f"step-{step_index:03d}.invalid-retry-00.response.json"
            _write_json(retry_response_path, retry_response.to_dict())
            retry_duplicate_generation = _looks_like_duplicate_generation(retry_response.python_code, previous_executed_code)
            retry_invalid_generation = (
                not _is_compilable_python_code(retry_response.python_code)
                or _looks_like_non_executing_task_script(retry_response.python_code)
                or _looks_like_missing_install_progress_generation(retry_response.python_code, user_prompt)
                or retry_duplicate_generation
            )
            if retry_invalid_generation:
                if not _is_compilable_python_code(retry_response.python_code):
                    retry_response.notes.append("stopped_due_to_invalid_python_generation")
                if _looks_like_non_executing_task_script(retry_response.python_code):
                    retry_response.notes.append("stopped_due_to_non_executing_python_generation")
                if _looks_like_missing_install_progress_generation(retry_response.python_code, user_prompt):
                    retry_response.notes.append("stopped_due_to_missing_install_progress_generation")
                if retry_duplicate_generation:
                    retry_response.notes.append("stopped_due_to_duplicate_python_generation")
                final_response = retry_response.to_dict()
                _write_json(retry_response_path, retry_response.to_dict())
                _write_json(response_path, retry_response.to_dict())
                stopped_reason = "invalid_python_generation"
                history.append(f"step-{step_index:03d}_stopped=invalid_python_generation")
                break
            response = retry_response
            normalized_code = retry_normalized_code

        response.notes.append(f"code_fingerprint={_code_fingerprint(response.python_code)}")
        final_response = response.to_dict()
        _write_json(response_path, response.to_dict())

        _, last_execution, state = _execute_code_step(
            executor_client=executor_client,
            root=root,
            step_id=step_id,
            python_code=response.python_code,
            metadata={
                "agent_response": response.to_dict(),
                "step_index": step_index,
                "agent_session_id": root.name,
                "agent_step_dir": str(root / "steps" / step_id),
            },
        )
        executed_steps += 1
        history.append(f"{step_id}_return_code={last_execution.get('return_code', 'unknown')}")
        history.append(f"{step_id}_code_fingerprint={_code_fingerprint(response.python_code)}")

        error_info = dict(last_execution.get("error_info") or {})
        repairable_missing_module = (
            dependency_repair_enabled
            and bool(policy.get("allow_package_install", False))
            and dependency_repairs_used < dependency_repair_max_attempts
            and error_info.get("kind") == "missing_python_module"
            and bool(error_info.get("repairable", False))
        )
        if repairable_missing_module:
            repair_attempt_index = dependency_repairs_used
            dependency_repairs_used += 1
            repair_result = _attempt_dependency_repair(
                runtime=runtime,
                executor_client=executor_client,
                root=root,
                user_prompt=user_prompt,
                policy=policy,
                strong_visual_grounding=strong_visual_grounding,
                reasoning_enabled=reasoning_enabled,
                state=state,
                history=history,
                last_execution=last_execution,
                original_response=response,
                original_step_id=step_id,
                step_index=step_index,
                max_new_tokens=max_new_tokens,
                repair_attempt_index=repair_attempt_index,
                allow_shell_fallback=dependency_repair_allow_shell_fallback,
            )
            executed_steps += int(repair_result.get("executions_added", 0))
            if repair_result.get("handled"):
                last_execution = dict(repair_result.get("last_execution") or last_execution)
                state = dict(repair_result.get("state") or state)

        if _looks_like_download_chunk_completed(user_prompt=user_prompt, last_execution=last_execution):
            response.done = True
            response.notes.append("download_chunk_completed")

        if response.done and int(last_execution.get("return_code", 0) or 0) == 0:
            history.append(f"{step_id}_completed=1")
            final_response = response.to_dict()
            stopped_reason = stopped_reason or "task_completed"
            break

        current_visual_hash = _state_visual_hash(state)
        replan_reasons: list[str] = []
        if previous_executed_code and normalized_code and normalized_code == previous_executed_code:
            replan_reasons.append("repeated_code_execution")
        if _looks_like_download_or_install_task(user_prompt) and _looks_like_opened_page_only_step(response.python_code):
            replan_reasons.append("partial_progress_opened_page_only")
        dependency_error_handled = repairable_missing_module and dependency_repairs_used > repair_attempt_index if repairable_missing_module else False
        installer_timeout = _looks_like_installer_timeout(last_execution, response.python_code, user_prompt)
        installer_app_not_found = _looks_like_installer_launched_but_app_not_found(last_execution, response.python_code, user_prompt)
        incomplete_install_attempt = _looks_like_incomplete_install_attempt(last_execution, response.python_code, user_prompt)
        if (
            (
                int(last_execution.get("return_code", 0) or 0) != 0
                or _looks_like_reported_failure(last_execution)
                or incomplete_install_attempt
            )
            and not dependency_error_handled
        ):
            if installer_timeout:
                replan_reasons.append("installer_timeout")
            elif installer_app_not_found or incomplete_install_attempt:
                replan_reasons.append("installer_app_not_found")
            else:
                replan_reasons.append("execution_error")
        if _looks_like_direct_download_url_404(last_execution, response.python_code):
            replan_reasons.append("download_url_404")
        if _looks_like_direct_download_url_403(last_execution, response.python_code):
            replan_reasons.append("download_url_403")
        if _looks_like_installer_url_discovery_failure(last_execution, response.python_code):
            replan_reasons.append("installer_url_not_found")
        if previous_visual_hash and current_visual_hash and previous_visual_hash == current_visual_hash:
            replan_reasons.append("no_visual_change")
        previous_executed_code = normalized_code or previous_executed_code
        previous_visual_hash = current_visual_hash

        if (
            replan_enabled
            and replan_reasons
            and replans_used < replan_max_attempts
            and step_index + 1 < max_iterations
        ):
            unique_reasons = list(dict.fromkeys(replan_reasons))
            replans_used += 1
            pending_replan_reasons = unique_reasons
            response.notes.append(f"replan_reasons={','.join(unique_reasons)}")
            final_response = response.to_dict()
            _write_json(response_path, response.to_dict())
            history.append(f"{step_id}_replan_requested={','.join(unique_reasons)}")
            history.append("system_hint=previous attempt repeated, failed, or did not visibly change the UI; generate a materially different next Python step")
            if "download_url_404" in unique_reasons:
                history.append(
                    "system_hint=previous direct installer URL returned 404; do not guess another filename pattern, fetch the official page HTML or current vendor page and extract a fresh official .exe link before downloading"
                )
                history.append(
                    "system_hint=use pure Python for download recovery; avoid curl, wget, powershell download commands, local http.server helpers, and literal %USERPROFILE% path strings"
                )
            if "download_url_403" in unique_reasons:
                history.append(
                    "system_hint=previous direct installer URL returned 403/Forbidden; retry with urllib.request.Request plus a browser-like User-Agent header or fetch the official download page HTML with the same header and extract the official installer link there"
                )
                history.append(
                    "system_hint=for vendor download URLs, prefer urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}) over plain urlopen(url)"
                )
            if "installer_url_not_found" in unique_reasons:
                history.append(
                    "system_hint=previous script could not discover an installer URL from the current HTML; if a vendor landing page does not expose a raw .exe, inspect at least one alternate official page or official release page in the same script before giving up"
                )
                history.append(
                    "system_hint=when parsing HTML for installers, do not search only href attributes; also scan the full HTML/text for absolute https .exe URLs and verify candidates with a real HTTP request before downloading"
                )
            if "installer_timeout" in unique_reasons:
                history.append(
                    "system_hint=previous installer launch timed out; do not rerun the same silent installer command again on the next step"
                )
                history.append(
                    "system_hint=first inspect whether the installer window, UAC prompt, or completion dialog is already visible and use Python GUI automation to advance it if needed"
                )
                history.append(
                    "system_hint=before rerunning any installer, check common install paths and running processes for the target app; if the app is already installed, launch it directly and verify the process"
                )
            if "installer_app_not_found" in unique_reasons:
                history.append(
                    "system_hint=previous step launched the installer but the app was still not found afterward; do not repeat the same silent launch-and-scan script again"
                )
                history.append(
                    "system_hint=use the latest screenshot and current desktop state to detect an installer wizard, UAC prompt, license dialog, or completion window and drive that GUI forward with Python automation"
                )
                history.append(
                    "system_hint=only search install paths after handling any visible installer window; if an installed exe appears afterward, launch it directly and verify the process"
                )
            if "execution_error" in unique_reasons and _looks_like_optional_windows_gui_module_failure(last_execution):
                history.append(
                    "system_hint=previous attempt failed importing optional Windows GUI modules; do not require win32gui/win32con/pythoncom/pywinauto on the next step unless already proven importable"
                )
                history.append(
                    "system_hint=prefer pyautogui, pygetwindow, psutil, and standard library fallbacks for installer GUI handling"
                )

        if response.done:
            break
    else:
        if stopped_reason is None and final_response and not bool(final_response.get("done", False)):
            stopped_reason = "max_iterations_reached"

    summary = {
        "run_dir": str(root),
        "iterations": executed_steps,
        "generated_steps": generated_steps,
        "last_execution": last_execution,
        "final_response": final_response,
        "stopped_reason": stopped_reason,
        "strong_visual_grounding": strong_visual_grounding,
        "execution_style": execution_style,
        "reasoning_enabled": reasoning_enabled,
        "replan_enabled": replan_enabled,
        "replan_max_attempts": replan_max_attempts,
        "replans_used": replans_used,
        "pending_replan_reasons": pending_replan_reasons,
        "web_search_enabled": web_search_enabled,
        "web_search_engine": web_search_engine if web_search_enabled else None,
        "searxng_base_url": searxng_base_url if web_search_enabled else None,
        "searxng_preferred_engines": normalized_preferred_search_engines if web_search_enabled else [],
        "web_search_decision_use_image": web_search_decision_use_image if web_search_enabled else False,
        "web_search_decision_reasoning_enabled": web_search_decision_reasoning_enabled if web_search_enabled else False,
        "web_search_decision_max_new_tokens": web_search_decision_max_new_tokens,
        "web_search_top_k": web_search_top_k,
        "web_search_max_uses": web_search_max_uses,
        "web_search_uses": web_search_uses,
        "web_search_queries": web_search_queries,
        "empty_generation_retries_used": empty_generation_retries_used,
        "invalid_generation_retries_used": invalid_generation_retries_used,
        "dependency_repair_enabled": dependency_repair_enabled,
        "dependency_repair_max_attempts": dependency_repair_max_attempts,
        "dependency_repair_allow_shell_fallback": dependency_repair_allow_shell_fallback,
        "dependency_repairs_used": dependency_repairs_used,
    }
    _write_json(root / "loop-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="computer-use-raw-python-agent")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--mcp-command", nargs="+")
    parser.add_argument("--mcp-cwd")
    parser.add_argument("--model-id")
    parser.add_argument("--agent-cli-command", nargs="+")
    parser.add_argument("--agent-cli-cwd")
    parser.add_argument("--processor-id")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--execution-style", choices=("python_first", "gui_first"), default="python_first")
    parser.add_argument("--strong-visual-grounding", action="store_true")
    parser.add_argument("--reasoning-enabled", action="store_true")
    parser.add_argument("--replan-enabled", action="store_true")
    parser.add_argument("--replan-max-attempts", type=int, default=1)
    parser.add_argument("--web-search-enabled", action="store_true")
    parser.add_argument("--web-search-engine", default="searxng")
    parser.add_argument("--searxng-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--searxng-preferred-engine", action="append")
    parser.add_argument("--web-search-decision-use-image", action="store_true")
    parser.add_argument("--web-search-decision-reasoning-enabled", action="store_true")
    parser.add_argument("--web-search-decision-max-new-tokens", type=int, default=64)
    parser.add_argument("--web-search-top-k", type=int, default=5)
    parser.add_argument("--web-search-max-uses", type=int, default=3)
    parser.add_argument("--web-search-timeout-s", type=float, default=10.0)
    parser.add_argument("--dependency-repair-enabled", action="store_true")
    parser.add_argument("--dependency-repair-max-attempts", type=int, default=2)
    parser.add_argument("--dependency-repair-allow-shell-fallback", action="store_true")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--disable-cpu-offload", action="store_true")
    parser.add_argument("--preload", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model_id and args.agent_cli_command:
        raise SystemExit("--model-id and --agent-cli-command are mutually exclusive")
    if args.agent_cli_command:
        runtime: AgentRuntime = ExternalCliRawPythonRuntime(
            command=list(args.agent_cli_command),
            cwd=args.agent_cli_cwd,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        runtime = GUIOwlRawPythonRuntime(
            model_id=args.model_id or _default_model_id(),
            processor_id=args.processor_id,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=not args.disable_4bit,
            compute_dtype=args.compute_dtype,
            device_map=args.device_map,
            enable_fp32_cpu_offload=not args.disable_cpu_offload,
        )
    if args.preload:
        runtime.ensure_loaded()
    executor_client = build_executor_client(
        endpoint=args.endpoint,
        mcp_command=args.mcp_command,
        mcp_cwd=args.mcp_cwd,
    )
    try:
        summary = run_agent_control_loop(
            runtime=runtime,
            executor_client=executor_client,
            user_prompt=args.prompt,
            policy=_load_json(args.policy),
            run_dir=args.run_dir,
            max_iterations=args.max_iterations,
            max_new_tokens=args.max_new_tokens,
            execution_style=args.execution_style,
            strong_visual_grounding=args.strong_visual_grounding,
            reasoning_enabled=args.reasoning_enabled,
            replan_enabled=args.replan_enabled,
            replan_max_attempts=args.replan_max_attempts,
            web_search_enabled=args.web_search_enabled,
            web_search_engine=args.web_search_engine,
            searxng_base_url=args.searxng_base_url,
            searxng_preferred_engines=list(args.searxng_preferred_engine or []),
            web_search_decision_use_image=args.web_search_decision_use_image,
            web_search_decision_reasoning_enabled=args.web_search_decision_reasoning_enabled,
            web_search_decision_max_new_tokens=args.web_search_decision_max_new_tokens,
            web_search_top_k=args.web_search_top_k,
            web_search_max_uses=args.web_search_max_uses,
            web_search_timeout_s=args.web_search_timeout_s,
            dependency_repair_enabled=args.dependency_repair_enabled,
            dependency_repair_max_attempts=args.dependency_repair_max_attempts,
            dependency_repair_allow_shell_fallback=args.dependency_repair_allow_shell_fallback,
        )
    finally:
        executor_client.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
