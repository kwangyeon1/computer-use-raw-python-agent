from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import ast
import base64
import hashlib
import json
import re
import urllib.parse

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


_RUNTIME_HELPERS: dict[str, str] = {
    "open_url_and_wait": """
def open_url_and_wait(url, *, expected_title_tokens=None, timeout_s=20.0, poll_interval_s=1.0, settle_time_s=2.0):
    import os
    import subprocess
    import time
    import webbrowser
    from pathlib import Path

    target_url = str(url or "").strip()
    if not target_url:
        raise SystemExit("open_url_and_wait requires a non-empty url")
    expected = [str(item).strip().lower() for item in (expected_title_tokens or []) if str(item).strip()]
    deadline = time.time() + max(float(timeout_s), float(poll_interval_s))
    started_at = time.time()

    launch_errors = []

    def _launch_windows_browser():
        launched = False
        try:
            os.startfile(target_url)
            launched = True
        except Exception as exc:
            launch_errors.append(f"os.startfile: {exc}")
        if launched:
            return True

        try:
            result = subprocess.run(
                ["cmd", "/c", "start", "", target_url],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=10,
            )
            if int(result.returncode or 0) == 0:
                return True
            launch_errors.append(f'cmd-start rc={result.returncode} stderr={result.stderr.strip()}')
        except Exception as exc:
            launch_errors.append(f"cmd-start: {exc}")

        candidate_paths = []
        env_candidates = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LocalAppData"),
        ]
        browser_relpaths = [
            ("Microsoft", "Edge", "Application", "msedge.exe"),
            ("Google", "Chrome", "Application", "chrome.exe"),
            ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
            ("Mozilla Firefox", "firefox.exe"),
            ("Opera", "launcher.exe"),
        ]
        for base in env_candidates:
            if not base:
                continue
            for relpath in browser_relpaths:
                candidate = Path(base).joinpath(*relpath)
                if candidate.exists():
                    candidate_paths.append(candidate)

        seen = set()
        unique_candidates = []
        for candidate in candidate_paths:
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        for browser_path in unique_candidates:
            try:
                subprocess.Popen([str(browser_path), "--new-tab", target_url])
                return True
            except Exception as exc:
                launch_errors.append(f"{browser_path.name}: {exc}")
        return False

    if os.name == "nt":
        if not _launch_windows_browser():
            raise SystemExit(f"failed to open url: {target_url} ({'; '.join(launch_errors)})")
    else:
        if not webbrowser.open(target_url):
            raise SystemExit(f"failed to open url: {target_url}")

    def _browser_running():
        browser_names = ("chrome.exe", "msedge.exe", "iexplore.exe", "firefox.exe", "opera.exe", "brave.exe")
        if os.name == "nt":
            result = subprocess.run(["tasklist"], capture_output=True, text=True, errors="replace", check=False)
            haystack = result.stdout.lower()
            return any(name in haystack for name in browser_names)
        return False

    def _matching_title_visible():
        if not expected:
            return False
        try:
            import pygetwindow as gw
        except Exception:
            return False
        for title in gw.getAllTitles():
            lowered = str(title or "").strip().lower()
            if lowered and any(token in lowered for token in expected):
                return True
        return False

    def _browser_window_visible():
        try:
            import pygetwindow as gw
        except Exception:
            return False
        browser_title_tokens = ("chrome", "edge", "firefox", "brave", "opera", "internet explorer")
        for title in gw.getAllTitles():
            lowered = str(title or "").strip().lower()
            if lowered and any(token in lowered for token in browser_title_tokens):
                return True
        return False

    def _activate_browser_window():
        try:
            import pygetwindow as gw
        except Exception:
            return False
        candidates = []
        browser_title_tokens = ("chrome", "edge", "firefox", "brave", "opera", "internet explorer")
        for window in gw.getAllWindows():
            title = str(getattr(window, "title", "") or "").strip()
            if not title:
                continue
            lowered = title.lower()
            score = 0
            if expected and any(token in lowered for token in expected):
                score += 80
            if any(token in lowered for token in browser_title_tokens):
                score += 30
            if any(token in lowered for token in ("download", "다운로드", "windows", "pc")):
                score += 15
            if score <= 0:
                continue
            candidates.append((score, window))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, window in candidates:
            try:
                if hasattr(window, "isMinimized") and window.isMinimized:
                    window.restore()
                    time.sleep(0.2)
                window.activate()
                time.sleep(0.3)
                return True
            except Exception:
                continue
        return False

    while time.time() < deadline:
        title_ready = _matching_title_visible()
        browser_ready = _browser_running()
        browser_window_ready = _browser_window_visible()
        elapsed = time.time() - started_at
        if title_ready and elapsed >= float(settle_time_s):
            _activate_browser_window()
            return True
        if not expected:
            if browser_window_ready and elapsed >= float(settle_time_s):
                _activate_browser_window()
                return True
            if browser_ready and elapsed >= float(settle_time_s) and _activate_browser_window():
                return True
        time.sleep(float(poll_interval_s))

    detail = f" ({'; '.join(launch_errors)})" if launch_errors else ""
    raise SystemExit(f"browser or page did not become ready for: {target_url}{detail}")
""".strip(),
    "wait_for_stable_download": """
def wait_for_stable_download(path_or_pattern, *, min_bytes=1_000_000, stable_checks=3, poll_interval_s=2.0, timeout_s=120.0, min_quiet_time_s=None, download_dir=None):
    import glob
    import os
    import time
    from pathlib import Path

    raw = str(path_or_pattern or "").strip()
    if not raw:
        raise SystemExit("wait_for_stable_download requires a path or glob pattern")
    if stable_checks < 1:
        stable_checks = 1
    if poll_interval_s <= 0:
        poll_interval_s = 1.0
    if min_quiet_time_s is None:
        min_quiet_time_s = float(stable_checks) * float(poll_interval_s)
    elif min_quiet_time_s < 0:
        min_quiet_time_s = 0.0
    deadline = time.time() + max(float(timeout_s), float(poll_interval_s))
    downloads = Path(os.path.expanduser(str(download_dir))) if download_dir else (Path.home() / "Downloads")

    def _matches():
        expanded = os.path.expanduser(raw)
        if any(ch in raw for ch in "*?[]"):
            return [Path(item) for item in glob.glob(expanded, recursive=True)]
        candidate = Path(expanded)
        if candidate.is_absolute() or raw.startswith("~"):
            return [candidate]
        return list(downloads.glob(raw))

    def _partial_files(candidate):
        candidate_name = candidate.name.lower() if candidate is not None else ""
        candidate_stem = candidate.stem.lower() if candidate is not None else ""
        partials = []
        for pattern in ("*.crdownload", "*.part", "*.partial", "*.tmp"):
            for path in downloads.glob(pattern):
                lowered = path.name.lower()
                if candidate_name and candidate_name in lowered:
                    partials.append(path)
                    continue
                if candidate_stem and candidate_stem in lowered:
                    partials.append(path)
                    continue
                if not candidate_name and not candidate_stem:
                    partials.append(path)
        return partials

    last_size = None
    stable_count = 0
    last_candidate = None

    while time.time() < deadline:
        candidates = [path for path in _matches() if path.exists() and path.is_file()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        candidate = candidates[0] if candidates else None
        if candidate is not None:
            last_candidate = candidate
            size = candidate.stat().st_size
            mtime_age = time.time() - candidate.stat().st_mtime
            if size >= int(min_bytes) and not _partial_files(candidate):
                if size == last_size:
                    stable_count += 1
                else:
                    last_size = size
                    stable_count = 1
                if stable_count >= stable_checks and mtime_age >= float(min_quiet_time_s):
                    return candidate
            else:
                last_size = size
                stable_count = 0
        time.sleep(float(poll_interval_s))

    if last_candidate is not None and last_candidate.exists():
        raise SystemExit(f"download incomplete: {last_candidate} ({last_candidate.stat().st_size} bytes)")
    raise SystemExit(f"download not found for pattern: {raw}")
""".strip(),
    "ocr_screen_text_regions": """
def ocr_screen_text_regions(image_path=None, *, max_lines=40, crop_region=None):
    import base64
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    if os.name != "nt":
        return []

    temp_path = None
    crop_temp_path = None
    path = Path(image_path) if image_path else None
    crop_left = 0
    crop_top = 0
    try:
        if path is None:
            from PIL import ImageGrab

            image = ImageGrab.grab()
            handle = tempfile.NamedTemporaryFile(prefix="ocr-screen-", suffix=".png", delete=False)
            temp_path = handle.name
            handle.close()
            image.save(temp_path, format="PNG")
            path = Path(temp_path)
        elif not path.exists():
            raise SystemExit(f"OCR image not found: {path}")

        if crop_region is not None:
            from PIL import Image

            with Image.open(path) as source_image:
                image_width, image_height = source_image.size
                crop_left = max(0, min(image_width - 1, int(crop_region.get("left") or 0)))
                crop_top = max(0, min(image_height - 1, int(crop_region.get("top") or 0)))
                crop_right = max(crop_left + 1, min(image_width, int(crop_region.get("right") or image_width)))
                crop_bottom = max(crop_top + 1, min(image_height, int(crop_region.get("bottom") or image_height)))
                cropped = source_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                handle = tempfile.NamedTemporaryFile(prefix="ocr-crop-", suffix=".png", delete=False)
                crop_temp_path = handle.name
                handle.close()
                cropped.save(crop_temp_path, format="PNG")
                path = Path(crop_temp_path)

        powershell_script = r'''
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime

function Await([object] $Operation, [type] $ResultType) {
    $asTaskMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
        Select-Object -First 1
    if ($null -eq $asTaskMethod) {
        throw "Could not locate System.WindowsRuntimeSystemExtensions.AsTask"
    }
    $asTask = $asTaskMethod.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($Operation))
    $null = $netTask.Wait(-1)
    return $netTask.Result
}

function ToBase64([string] $Value) {
    return [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes(($Value ?? "")))
}

$filePath = $env:COMPUTER_USE_OCR_IMAGE_PATH
if ([string]::IsNullOrWhiteSpace($filePath)) {
    throw "COMPUTER_USE_OCR_IMAGE_PATH is empty"
}
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType = WindowsRuntime]

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($filePath)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    throw "Windows OCR engine unavailable"
}
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

$lines = @()
foreach ($line in $result.Lines) {
    $left = 2147483647
    $top = 2147483647
    $right = -1
    $bottom = -1
    foreach ($word in $line.Words) {
        $rect = $word.BoundingRect
        if ($rect.X -lt $left) { $left = [int]$rect.X }
        if ($rect.Y -lt $top) { $top = [int]$rect.Y }
        if (($rect.X + $rect.Width) -gt $right) { $right = [int]($rect.X + $rect.Width) }
        if (($rect.Y + $rect.Height) -gt $bottom) { $bottom = [int]($rect.Y + $rect.Height) }
    }
    if ($right -lt $left -or $bottom -lt $top) {
        $left = 0
        $top = 0
        $right = 0
        $bottom = 0
    }
    $lines += @{
        text_b64 = (ToBase64 ([string]$line.Text))
        left = [int]$left
        top = [int]$top
        width = [int]([Math]::Max(0, $right - $left))
        height = [int]([Math]::Max(0, $bottom - $top))
    }
}

$payload = @{
    text_b64 = (ToBase64 ([string]$result.Text))
    lines = $lines
}
$payload | ConvertTo-Json -Depth 6 -Compress
'''

        for executable in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
            try:
                env = dict(os.environ)
                env["COMPUTER_USE_OCR_IMAGE_PATH"] = str(path)
                completed = subprocess.run(
                    [
                        executable,
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        powershell_script,
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                    env=env,
                    timeout=20,
                )
            except FileNotFoundError:
                continue
            if completed.returncode != 0:
                continue
            try:
                payload = json.loads(str(completed.stdout or "").strip() or "{}")
            except json.JSONDecodeError:
                continue
            lines = payload.get("lines")
            if not isinstance(lines, list):
                continue
            normalized = []
            for item in lines:
                if not isinstance(item, dict):
                    continue
                encoded_text = str(item.get("text_b64") or "").strip()
                if not encoded_text:
                    continue
                try:
                    text = base64.b64decode(encoded_text.encode("ascii"), validate=False).decode("utf-8", errors="replace")
                except Exception:
                    continue
                text = " ".join(text.split()).strip()
                if not text:
                    continue
                normalized.append(
                    {
                        "text": text,
                        "left": int(item.get("left") or 0) + int(crop_left),
                        "top": int(item.get("top") or 0) + int(crop_top),
                        "width": int(item.get("width") or 0),
                        "height": int(item.get("height") or 0),
                    }
                )
            normalized.sort(key=lambda item: (int(item.get("top") or 0), int(item.get("left") or 0)))
            if max_lines and len(normalized) > int(max_lines):
                normalized = normalized[: int(max_lines)]
            return normalized
        return []
    finally:
        if crop_temp_path:
            Path(crop_temp_path).unlink(missing_ok=True)
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
""".strip(),
    "click_text_targets": """
def click_text_targets(
    targets,
    *,
    avoid_targets=None,
    primary_targets=None,
    min_primary_hits=0,
    window_title_tokens=None,
    restrict_to_browser_window=False,
    click_horizontal_bias="center",
    image_path=None,
    timeout_s=10.0,
    poll_interval_s=1.0,
    prefer_bottom=True,
    double_click=False,
):
    import ctypes
    import re
    import time

    target_terms = [str(item).strip().lower() for item in (targets or []) if str(item).strip()]
    avoid_terms = [str(item).strip().lower() for item in (avoid_targets or []) if str(item).strip()]
    primary_terms = [str(item).strip().lower() for item in (primary_targets or []) if str(item).strip()]
    window_terms = [str(item).strip().lower() for item in (window_title_tokens or []) if str(item).strip()]
    minimum_primary_hits = max(0, int(min_primary_hits or 0))
    if not target_terms:
        raise SystemExit("click_text_targets requires at least one target term")
    if ctypes.sizeof(ctypes.c_void_p) == 0:
        raise SystemExit("ctypes is unavailable")

    def _best_token_match(text):
        lowered = text.lower()
        best = None
        for token in target_terms:
            if not token:
                continue
            start = lowered.find(token)
            if start < 0:
                continue
            end = start + len(token)
            candidate = (token in primary_terms, end - start, start, end, token)
            if best is None or candidate > best:
                best = candidate
        return best

    def _score_text(text):
        lowered = text.lower()
        score = 0
        primary_hits = 0
        best_match = _best_token_match(text)
        matched_token = best_match[4] if best_match is not None else ""
        matched_start = best_match[2] if best_match is not None else -1
        for idx, token in enumerate(target_terms):
            if token in lowered:
                score += max(40 - idx, 10)
        if primary_terms:
            primary_hits = sum(1 for token in primary_terms if token in lowered)
            if primary_hits < minimum_primary_hits:
                return -1
            score += primary_hits * 15
        for token in avoid_terms:
            if token and token in lowered:
                score -= 80
        if any(token in lowered for token in ("download", "다운로드", "install", "installer", "setup", "설치")):
            score += 25
        if any(token in lowered for token in ("windows", "pc", "exe", "next", "확인", "동의")):
            score += 10
        terminal_markers = (
            ".venv",
            "python -m",
            "python3",
            "pytest",
            "compileall",
            "powershell",
            "cmd /c",
            "curl ",
            "request.json",
            "response.json",
            "loop-summary",
            "run-session",
            "return-executable-python-only-for-this-chunk",
            "payloads/",
            "responses/",
            "scripts/vibe.py",
            "computer-use-raw-python",
        )
        if any(marker in lowered for marker in terminal_markers):
            score -= 120
        if "--" in lowered:
            score -= 80
        if "/" in text or "\\\\" in text:
            score -= 45
        if lowered.startswith(("http://", "https://", "www.")):
            score -= 55
        if re.fullmatch(r"[a-z0-9.-]+\.(com|net|org|co|io|app|dev|kr|tv|me|gg|ai|info)", lowered):
            score -= 55
        elif "." in lowered and " " not in lowered and not lowered.endswith(".exe"):
            score -= 35
        if any(ext in lowered for ext in (".json", ".py", ".log", ".md", ".txt")):
            score -= 60
        words = [part for part in text.split() if part]
        if len(text) <= 24 and len(words) <= 4:
            score += 12
        if len(text) > 48 or len(words) > 6:
            score -= 40
        if len(words) > 4 and any(token in lowered for token in ("download", "다운로드", "install", "installer", "setup", "설치")):
            score -= 25
        if lowered.startswith(("q ", "search ", "검색 ")) and len(words) > 2:
            score -= 45
        if matched_token:
            char_count = max(1, len(text))
            match_ratio = matched_start / char_count
            if matched_token in primary_terms:
                score += 18
            if matched_token in {"download", "다운로드", "install", "installer", "setup", "설치", "받기", "exe"}:
                score += 16
            if match_ratio >= 0.55:
                score += 18
            elif match_ratio >= 0.35:
                score += 8
            if any(sep in text for sep in ("|", "·", "•", ">", "»")):
                score += 8
        return score

    def _click_point(x, y):
        user32 = ctypes.windll.user32
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.15)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        if double_click:
            time.sleep(0.15)
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)

    def _page_down():
        user32 = ctypes.windll.user32
        vk_next = 0x22
        user32.keybd_event(vk_next, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(vk_next, 0, 0x0002, 0)
        time.sleep(0.35)

    def _screen_metrics():
        user32 = ctypes.windll.user32
        return max(1, int(user32.GetSystemMetrics(0))), max(1, int(user32.GetSystemMetrics(1)))

    def _screen_browser_region_fallback():
        screen_width, screen_height = _screen_metrics()
        left = int(screen_width * 0.26)
        top = 0
        right = max(left + 400, screen_width - 10)
        bottom = max(400, screen_height - 40)
        return {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "title": "[screen-browser-region-fallback]",
        }

    def _browser_window_region():
        if not restrict_to_browser_window:
            return None
        try:
            import pygetwindow as gw
        except Exception:
            return None

        browser_title_tokens = ("chrome", "edge", "firefox", "brave", "opera", "internet explorer")
        try:
            active_window = gw.getActiveWindow()
        except Exception:
            active_window = None

        def _window_score(window):
            title = str(getattr(window, "title", "") or "").strip()
            if not title:
                return -1
            lowered = title.lower()
            width = int(getattr(window, "width", 0) or 0)
            height = int(getattr(window, "height", 0) or 0)
            left = int(getattr(window, "left", 0) or 0)
            top = int(getattr(window, "top", 0) or 0)
            if width < 320 or height < 320:
                return -1
            if left + width < 0 or top + height < 0:
                return -1
            screen_width, screen_height = _screen_metrics()
            score = 0
            if any(token in lowered for token in browser_title_tokens):
                score += 60
            if window_terms and any(token in lowered for token in window_terms):
                score += 80
            if any(token in lowered for token in ("download", "다운로드", "official", "공식", "windows", "pc")):
                score += 15
            if any(token in lowered for token in ("terminal", "powershell", "command prompt", "cmd", "bash", "python", "codex", "explorer")):
                score -= 140
            if active_window is not None and window is active_window:
                score += 45
            if width >= int(screen_width * 0.45):
                score += 35
            if left >= int(screen_width * 0.18):
                score += 20
            if width < int(screen_width * 0.35) and left <= int(screen_width * 0.12):
                score -= 35
            if height >= int(screen_height * 0.50):
                score += 10
            score += min((width * height) // 50000, 30)
            return score

        candidates = []
        for window in gw.getAllWindows():
            score = _window_score(window)
            if score <= 0:
                continue
            candidates.append((score, window))
        if not candidates:
            return _screen_browser_region_fallback()
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, window = candidates[0]
        left = int(getattr(window, "left", 0) or 0)
        top = int(getattr(window, "top", 0) or 0)
        width = int(getattr(window, "width", 0) or 0)
        height = int(getattr(window, "height", 0) or 0)
        screen_width, _ = _screen_metrics()
        if width < int(screen_width * 0.45) and left <= int(screen_width * 0.12):
            return _screen_browser_region_fallback()
        return {
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
            "title": str(getattr(window, "title", "") or ""),
        }

    def _activate_browser_window():
        if not restrict_to_browser_window:
            return False
        try:
            import pygetwindow as gw
        except Exception:
            return False
        browser_title_tokens = ("chrome", "edge", "firefox", "brave", "opera", "internet explorer")
        candidates = []
        for window in gw.getAllWindows():
            title = str(getattr(window, "title", "") or "").strip()
            if not title:
                continue
            lowered = title.lower()
            screen_width, screen_height = _screen_metrics()
            width = int(getattr(window, "width", 0) or 0)
            height = int(getattr(window, "height", 0) or 0)
            left = int(getattr(window, "left", 0) or 0)
            score = 0
            if window_terms and any(token in lowered for token in window_terms):
                score += 80
            if any(token in lowered for token in browser_title_tokens):
                score += 50
            if any(token in lowered for token in ("download", "다운로드", "official", "공식", "windows", "pc")):
                score += 15
            if any(token in lowered for token in ("terminal", "powershell", "command prompt", "cmd", "bash", "python", "codex", "explorer")):
                score -= 140
            if width >= int(screen_width * 0.45):
                score += 35
            if left >= int(screen_width * 0.18):
                score += 20
            if height >= int(screen_height * 0.50):
                score += 10
            if score <= 0:
                continue
            candidates.append((score, window))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, window in candidates:
            try:
                if hasattr(window, "isMinimized") and window.isMinimized:
                    window.restore()
                    time.sleep(0.2)
                window.activate()
                time.sleep(0.35)
                return True
            except Exception:
                continue
        return False

    def _heuristic_browser_click(region, *, attempt_index):
        if region is None:
            return None
        left = int(region["left"])
        top = int(region["top"])
        right = int(region["right"])
        bottom = int(region["bottom"])
        width = max(1, right - left)
        height = max(1, bottom - top)
        browser_chrome = min(max(int(height * 0.10), 110), 190)
        content_top = min(bottom - 80, top + browser_chrome)
        content_height = max(120, bottom - content_top - 30)
        if not prefer_bottom:
            x_fracs = (0.18, 0.22, 0.27)
            y_fracs = (0.10, 0.20, 0.32)
            label = "browser_search_result_region"
        else:
            x_fracs = (0.97, 0.94, 0.91, 0.88, 0.85, 0.82)
            label = "browser_download_cta_region"
        cycle_len = len(y_fracs) if not prefer_bottom else len(x_fracs)
        idx = max(0, int(attempt_index)) % max(1, cycle_len)
        x = left + int(width * x_fracs[min(idx, len(x_fracs) - 1)])
        if not prefer_bottom:
            y = content_top + int(content_height * y_fracs[idx])
        else:
            header_offsets = (20, 34, 48, 62, 82, 108)
            y = top + browser_chrome + header_offsets[min(idx, len(header_offsets) - 1)]
        x = max(left + 40, min(right - 40, x))
        y = max(content_top + 20, min(bottom - 40, y))
        return {
            "text": f"[heuristic:{label}]",
            "x": int(x),
            "y": int(y),
            "score": 1,
        }

    deadline = time.time() + max(float(timeout_s), float(poll_interval_s))
    best_candidate = None
    sweep_index = 0
    while time.time() < deadline:
        if restrict_to_browser_window:
            _activate_browser_window()
        browser_region = _browser_window_region()
        lines = ocr_screen_text_regions(
            image_path=image_path,
            max_lines=200,
            crop_region=browser_region if restrict_to_browser_window and browser_region is not None else None,
        )

        def _collect_candidates(region):
            collected = []
            for item in lines:
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                score = _score_text(text)
                if score <= 0:
                    continue
                left = int(item.get("left") or 0)
                top = int(item.get("top") or 0)
                width = max(1, int(item.get("width") or 0))
                height = max(1, int(item.get("height") or 0))
                center_x = int(left + width / 2)
                center_y = int(top + height / 2)
                best_match = _best_token_match(text)
                matched_token = best_match[4] if best_match is not None else ""
                matched_start = best_match[2] if best_match is not None else 0
                matched_end = best_match[3] if best_match is not None else 0
                if click_horizontal_bias == "left_text":
                    center_x = int(left + min(max(width * 0.22, 40), 140))
                elif click_horizontal_bias in {"matched_token", "matched_token_right"} and best_match is not None:
                    char_count = max(1, len(text))
                    token_center_ratio = ((matched_start + matched_end) / 2.0) / char_count
                    if click_horizontal_bias == "matched_token_right":
                        token_center_ratio = min(0.96, token_center_ratio + 0.06)
                    center_x = int(left + width * token_center_ratio)
                if region is not None:
                    if not (
                        int(region["left"]) <= center_x <= int(region["right"])
                        and int(region["top"]) <= center_y <= int(region["bottom"])
                    ):
                        continue
                    score += 25
                    browser_height = max(1, int(region["bottom"]) - int(region["top"]))
                    relative_top = top - int(region["top"])
                    if not prefer_bottom:
                        if relative_top <= min(260, browser_height // 3):
                            score += 18
                        elif relative_top >= int(browser_height * 0.55):
                            score -= 35
                    elif relative_top <= min(220, browser_height // 3) and matched_token in {"download", "다운로드", "install", "installer", "setup", "설치", "받기", "exe"}:
                        score += 24
                if width > 520:
                    score -= 45
                elif width > 360:
                    score -= 25
                if width > 280 and len(text) > 18:
                    score -= 15
                if width <= 260 and 18 <= height <= 90:
                    score += 12
                if top < 220 and width > 320:
                    score -= 20
                if matched_token in {"download", "다운로드", "install", "installer", "setup", "설치", "받기", "exe"} and width > 280:
                    score += 28
                if matched_token in {"download", "다운로드", "install", "installer", "setup", "설치", "받기", "exe"} and top < 220 and width > 320:
                    score += 20
                if score <= 0:
                    continue
                collected.append(
                    {
                        "text": text,
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                        "x": center_x,
                        "y": center_y,
                        "score": score,
                        "matched_token": matched_token,
                    }
                )
            return collected

        candidates = _collect_candidates(browser_region)
        if not candidates and browser_region is not None:
            candidates = _collect_candidates(None)
        if candidates:
            candidates.sort(
                key=lambda item: (
                    int(item["score"]),
                    int(item["top"]) if prefer_bottom else -int(item["top"]),
                    int(item["width"]) * int(item["height"]),
                ),
                reverse=True,
            )
            best_candidate = candidates[0]
            center_x = int(best_candidate["x"])
            center_y = int(best_candidate["y"])
            _click_point(center_x, center_y)
            return {
                "text": best_candidate["text"],
                "x": center_x,
                "y": center_y,
                "score": best_candidate["score"],
            }
        if restrict_to_browser_window and browser_region is not None:
            heuristic_candidate = _heuristic_browser_click(browser_region, attempt_index=sweep_index)
            if heuristic_candidate is not None:
                _click_point(int(heuristic_candidate["x"]), int(heuristic_candidate["y"]))
                best_candidate = heuristic_candidate
                sweep_index += 1
                time.sleep(max(0.8, float(poll_interval_s)))
                continue
        sweep_index += 1
        if restrict_to_browser_window and sweep_index % 2 == 0:
            _page_down()
        time.sleep(float(poll_interval_s))
    if best_candidate is not None:
        return best_candidate
    raise SystemExit(f"could not find visible text target for {target_terms!r}")
""".strip(),
    "click_download_like_target": """
def click_download_like_target(*, extra_targets=None, avoid_targets=None, image_path=None, timeout_s=10.0):
    targets = [
        "official",
        "공식",
        "download",
        "다운로드",
        "설치",
        "install",
        "installer",
        "setup",
        "받기",
        "pc",
        "windows",
        "exe",
    ]
    avoid = [
        "android",
        "iphone",
        "ios",
        "mac",
        "macos",
        "linux",
        "portable",
        "zip",
        "archive",
        "source",
        "sdk",
        "server",
        "blog",
        "블로그",
        "forum",
        "커뮤니티",
    ]
    if extra_targets:
        targets.extend(str(item).strip().lower() for item in extra_targets if str(item).strip())
    if avoid_targets:
        avoid.extend(str(item).strip().lower() for item in avoid_targets if str(item).strip())
    return click_text_targets(
        targets,
        avoid_targets=avoid,
        primary_targets=["download", "다운로드", "install", "installer", "setup", "설치", "받기", "exe"],
        min_primary_hits=1,
        window_title_tokens=[*targets],
        restrict_to_browser_window=True,
        click_horizontal_bias="matched_token_right",
        image_path=image_path,
        timeout_s=timeout_s,
        poll_interval_s=1.0,
        prefer_bottom=True,
        double_click=False,
    )
""".strip(),
    "click_search_result_like_target": """
def click_search_result_like_target(*, extra_targets=None, avoid_targets=None, image_path=None, timeout_s=10.0):
    targets = [
        "official",
        "공식",
        "windows",
        "pc",
        "download",
        "다운로드",
    ]
    avoid = [
        "blog",
        "블로그",
        "forum",
        "커뮤니티",
        "news",
        "기사",
        "youtube",
        "광고",
        "ad",
        "ads",
        "android",
        "iphone",
        "ios",
        "mac",
        "macos",
        "linux",
        "portable",
        "zip",
        "archive",
        "source",
        "sdk",
        "server",
    ]
    if extra_targets:
        targets.extend(str(item).strip().lower() for item in extra_targets if str(item).strip())
    if avoid_targets:
        avoid.extend(str(item).strip().lower() for item in avoid_targets if str(item).strip())
    return click_text_targets(
        targets,
        avoid_targets=avoid,
        primary_targets=["official", "공식", "download", "다운로드", "windows", "pc"],
        min_primary_hits=1,
        window_title_tokens=[*targets],
        restrict_to_browser_window=True,
        click_horizontal_bias="left_text",
        image_path=image_path,
        timeout_s=timeout_s,
        poll_interval_s=1.0,
        prefer_bottom=False,
        double_click=True,
    )
""".strip(),
    "advance_visible_download_flow": """
def advance_visible_download_flow(*, extra_targets=None, image_path=None, timeout_s=18.0, search_first=False):
    import ctypes
    import time

    def _tab_enter(tab_presses, *, final_enter=True):
        user32 = ctypes.windll.user32
        vk_tab = 0x09
        vk_enter = 0x0D
        for _ in range(max(1, int(tab_presses))):
            user32.keybd_event(vk_tab, 0, 0, 0)
            time.sleep(0.05)
            user32.keybd_event(vk_tab, 0, 0x0002, 0)
            time.sleep(0.25)
        if final_enter:
            user32.keybd_event(vk_enter, 0, 0, 0)
            time.sleep(0.05)
            user32.keybd_event(vk_enter, 0, 0x0002, 0)
            time.sleep(1.0)

    attempts = []
    total_timeout = max(float(timeout_s), 6.0)
    search_timeout = min(12.0, max(6.0, total_timeout * 0.55))
    download_timeout = min(12.0, max(6.0, total_timeout * 0.55))
    if search_first:
        try:
            clicked = click_search_result_like_target(
                extra_targets=extra_targets,
                image_path=image_path,
                timeout_s=search_timeout,
            )
            attempts.append({"stage": "search_result", "clicked": clicked})
            time.sleep(4.0)
        except SystemExit as exc:
            attempts.append({"stage": "search_result", "error": str(exc)})
            if extra_targets:
                try:
                    clicked = click_text_targets(
                        [*extra_targets, "official", "공식", "download", "다운로드", "windows", "pc"],
                        primary_targets=[*extra_targets],
                        min_primary_hits=1,
                        window_title_tokens=[*extra_targets, "chrome", "edge", "firefox"],
                        restrict_to_browser_window=True,
                        click_horizontal_bias="left_text",
                        image_path=image_path,
                        timeout_s=search_timeout,
                        poll_interval_s=1.0,
                        prefer_bottom=False,
                        double_click=True,
                    )
                    attempts.append({"stage": "search_result_fallback", "clicked": clicked})
                    time.sleep(4.0)
                except SystemExit as fallback_exc:
                    attempts.append({"stage": "search_result_fallback", "error": str(fallback_exc)})
            _tab_enter(4, final_enter=True)
            attempts.append({"stage": "search_result_keyboard_fallback", "action": "tab_enter"})
            time.sleep(4.0)
    try:
        clicked = click_download_like_target(
            extra_targets=extra_targets,
            image_path=image_path,
            timeout_s=download_timeout,
        )
        attempts.append({"stage": "download_control", "clicked": clicked})
    except SystemExit as exc:
        attempts.append({"stage": "download_control", "error": str(exc)})
        _tab_enter(6, final_enter=True)
        attempts.append({"stage": "download_keyboard_fallback", "action": "tab_enter"})
    return {"attempts": attempts}
""".strip(),
}


def _directly_referenced_runtime_helpers(code: str) -> list[str]:
    normalized = _normalize_python_code(code)
    if not normalized:
        return []
    try:
        tree = ast.parse(normalized, mode="exec")
    except SyntaxError:
        return []
    defined_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    helper_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            helper_name = node.func.id
            if helper_name in _RUNTIME_HELPERS and helper_name not in defined_functions:
                helper_names.add(helper_name)
    return sorted(helper_names)


def _referenced_runtime_helpers(code: str) -> list[str]:
    pending = list(_directly_referenced_runtime_helpers(code))
    resolved: set[str] = set()
    while pending:
        helper_name = pending.pop(0)
        if helper_name in resolved:
            continue
        resolved.add(helper_name)
        for dependency in _directly_referenced_runtime_helpers(_RUNTIME_HELPERS.get(helper_name, "")):
            if dependency not in resolved:
                pending.append(dependency)
    return sorted(resolved)


def _expand_runtime_helpers(code: str) -> str:
    normalized = _normalize_python_code(code)
    helper_names = _referenced_runtime_helpers(normalized)
    if not helper_names:
        return normalized
    helper_blocks = [_RUNTIME_HELPERS[name] for name in helper_names]
    return "\n\n".join([*helper_blocks, normalized]).strip()


def _should_auto_open_prompt_url(request: StepRequest | None, code: str) -> bool:
    if request is None:
        return False
    if str(request.execution_style or "python_first").lower() != "gui_first":
        return False
    if not _looks_like_download_or_install_task(request.user_prompt):
        return False
    if _has_visible_gui_continuation_cues(request):
        return False
    prompt_url = _select_prompt_browser_url(request.user_prompt) or _fallback_browser_search_url(request.user_prompt)
    if not prompt_url:
        return False
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    if "open_url_and_wait(" in normalized:
        return False
    browser_navigation_tokens = (
        "webbrowser.open(",
        "os.startfile(",
        ".get(",
        ".goto(",
        "driver.get(",
        "page.goto(",
        "open_new_tab(",
    )
    return not any(token in normalized for token in browser_navigation_tokens)


def _should_auto_click_download_control(request: StepRequest | None, code: str) -> bool:
    if request is None:
        return False
    if str(request.execution_style or "python_first").lower() != "gui_first":
        return False
    if not _looks_like_download_or_install_task(request.user_prompt):
        return False
    if not _has_visible_gui_continuation_cues(request):
        return False
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    if "click_download_like_target(" in normalized or "click_search_result_like_target(" in normalized or "click_text_targets(" in normalized:
        return False
    if _looks_like_existing_installer_launch_task(request.user_prompt):
        return False
    gui_progress_tokens = (
        "pyautogui.",
        "pygetwindow",
        "locateonscreen(",
        "click(",
        "doubleclick(",
        "press(",
        "hotkey(",
        "typewrite(",
    )
    return not any(token in normalized for token in gui_progress_tokens)


def _should_replace_with_gui_first_browser_click(request: StepRequest | None, code: str) -> bool:
    if request is None:
        return False
    if str(request.execution_style or "python_first").lower() != "gui_first":
        return False
    if not _looks_like_download_or_install_task(request.user_prompt):
        return False
    if _looks_like_existing_installer_launch_task(request.user_prompt):
        return False
    prompt_url = _select_prompt_browser_url(request.user_prompt) or _fallback_browser_search_url(request.user_prompt)
    if not prompt_url:
        return False
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    gui_progress_tokens = (
        "pyautogui.",
        "pygetwindow",
        "click_download_like_target(",
        "click_search_result_like_target(",
        "click_text_targets(",
        "locateonscreen(",
        "click(",
        "doubleclick(",
        "press(",
        "hotkey(",
        "typewrite(",
    )
    if any(token in normalized for token in gui_progress_tokens):
        return False
    bypass_tokens = (
        "urllib.request",
        "urlopen(",
        "requests.",
        "httpx.",
        "webbrowser.open(",
        "webdriver.",
        "selenium",
        "href=",
        "download_url",
        "html =",
        "html_text",
        "re.findall(",
    )
    return any(token in normalized for token in bypass_tokens)


def _extract_prompt_download_glob(user_prompt: str) -> str | None:
    text = str(user_prompt or "")
    for pattern in (
        r"[\\/]+computer-use-agent[\\/]+([A-Za-z0-9._-]+)[\\/]+",
        r"~/Downloads/computer-use-agent/([A-Za-z0-9._-]+)/",
    ):
        match = re.search(pattern, text)
        if match:
            subdir = str(match.group(1) or "").strip()
            if subdir:
                return f"computer-use-agent/{subdir}/*.exe"
    return None


def _synthesized_visible_download_completion_code(
    request: StepRequest,
    *,
    prompt_url: str | None = None,
    timeout_s: float = 18.0,
    wait_timeout_s: float = 45.0,
    exit_on_success: bool = False,
    continue_on_failure: bool = False,
) -> str:
    extra_targets = _visible_flow_extra_targets(request, limit=2)
    search_first = _looks_like_search_results_observation(request) or _url_looks_like_search_results(prompt_url)
    lines: list[str] = []
    if prompt_url:
        lines.append(
            f'open_url_and_wait({json.dumps(prompt_url, ensure_ascii=False)}, '
            f'expected_title_tokens={json.dumps(extra_targets, ensure_ascii=False)})'
        )
    lines.extend(
        [
            "try:",
            "    flow = advance_visible_download_flow("
            f"extra_targets={json.dumps(extra_targets, ensure_ascii=False)}, "
            f"search_first={repr(bool(search_first))}, "
            f"timeout_s={float(timeout_s):.1f})",
            '    print(f"advanced visible download flow: {flow}")',
        ]
    )
    download_glob = _extract_prompt_download_glob(request.user_prompt)
    if download_glob:
        lines.extend(
            [
                "    installer = wait_for_stable_download("
                f"{json.dumps(download_glob, ensure_ascii=False)}, "
                f"min_bytes=1_000_000, timeout_s={float(wait_timeout_s):.1f})",
                '    print(f"download ready: {installer}")',
            ]
        )
    if exit_on_success:
        lines.extend(
            [
                "    raise SystemExit(0)",
                "except SystemExit as auto_exc:",
                '    if str(auto_exc).strip() in {"0", ""}:',
                "        raise",
            ]
        )
    else:
        lines.append("except SystemExit as auto_exc:")
    if continue_on_failure:
        lines.append('    print(f"visible download automation incomplete: {auto_exc}")')
    else:
        lines.append('    raise SystemExit(f"visible download automation incomplete: {auto_exc}")')
    return "\n".join(lines)


def _prepare_python_code_for_execution(request: StepRequest | None, code: str) -> str:
    normalized = _normalize_python_code(code)
    if not normalized:
        return normalized
    if _should_replace_with_gui_first_browser_click(request, normalized):
        prompt_url = _select_prompt_browser_url(request.user_prompt) or _fallback_browser_search_url(request.user_prompt)
        if prompt_url:
            normalized = _synthesized_visible_download_completion_code(
                request,
                prompt_url=prompt_url,
                timeout_s=18.0,
                wait_timeout_s=45.0,
                exit_on_success=True,
                continue_on_failure=False,
            )
            return _expand_runtime_helpers(normalized)
    if _should_auto_open_prompt_url(request, normalized):
        prompt_url = _select_prompt_browser_url(request.user_prompt) or _fallback_browser_search_url(request.user_prompt)
        if not prompt_url:
            return _expand_runtime_helpers(normalized)
        keyword_tokens = _visible_flow_extra_targets(request, limit=2)
        prelude = f'open_url_and_wait({json.dumps(prompt_url, ensure_ascii=False)}, expected_title_tokens={json.dumps(keyword_tokens, ensure_ascii=False)})'
        normalized = f"{prelude}\n\n{normalized}"
    if _should_auto_click_download_control(request, normalized):
        click_prelude = _synthesized_visible_download_completion_code(
            request,
            prompt_url=None,
            timeout_s=14.0,
            wait_timeout_s=30.0,
            exit_on_success=True,
            continue_on_failure=True,
        )
        normalized = f"{click_prelude}\n\n{normalized}"
    return _expand_runtime_helpers(normalized)


def _synthesized_visible_ui_click_recovery_code(request: StepRequest | None, *, timeout_s: float = 12.0) -> str:
    click_call = _click_helper_call_for_request(request, timeout_s=timeout_s)
    return "\n".join(
        [
            f"clicked = {click_call}",
            'print(f"clicked visible control: {clicked}")',
        ]
    )


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
    prompt_url_violation: bool = False,
    gui_first_visible_ui_violation: bool = False,
    gui_first_silent_install_shortcut: bool = False,
    store_detour_generation: bool = False,
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
    if prompt_url_violation:
        prompt_urls = _extract_prompt_urls(user_prompt)
        if prompt_urls:
            retry_history.append(
                "system_hint=this retry must start from one of the exact official URLs already present in the prompt before any other download URL guess: "
                + ", ".join(prompt_urls[:4])
            )
        retry_history.append(
            "system_hint=if you emit a direct download URL on this retry, it must be discovered from those exact prompt URLs or from official HTML fetched from them; otherwise the retry is invalid"
        )
    if gui_first_visible_ui_violation:
        retry_history.append(
            "system_hint=the previous generation ignored grounded visible browser/download/installer UI and switched to fresh network fetching or browser restart logic; on this retry continue from the visible UI with Python GUI automation first"
        )
        retry_history.append(
            "system_hint=when gui_first is active and the screenshot/observation already grounds a browser page, download control, or installer window, do not use urllib/requests/html scraping, regex link extraction, or webbrowser.open in place of that visible UI progression"
        )
    if gui_first_silent_install_shortcut:
        retry_history.append(
            "system_hint=the previous generation used a silent installer shortcut for a gui_first install chunk; on this retry do not start with /SILENT, /VERYSILENT, /SP-, or /NORESTART"
        )
        retry_history.append(
            "system_hint=for gui_first install chunks, either advance the visible installer/UAC/completion UI with Python GUI automation or launch the installer normally once and then handle the resulting window state"
        )
    if store_detour_generation:
        retry_history.append(
            "system_hint=the previous generation detoured into a platform app store listing; on this retry do not use Microsoft Store, app-store URLs, or store protocol handlers"
        )
        retry_history.append(
            "system_hint=for Windows installer tasks, stay on the official vendor or official release page path and continue toward the installer .exe or visible download/install control instead of a store listing"
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
        "downloaded installer",
        "already exists in downloads",
        "existing installer",
        "existing installer file",
        "already-downloaded installer",
        "already downloaded installer",
        "already present in downloads",
        "use the installer already present in downloads",
        "find the existing installer",
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


def _last_execution_opened_browser_for_gui_flow(last_execution: dict[str, Any]) -> bool:
    payload_metadata = dict(last_execution.get("payload_metadata") or {})
    executed_python_code = str(payload_metadata.get("executed_python_code") or "").lower()
    if not executed_python_code:
        return False
    browser_open_tokens = (
        "open_url_and_wait(",
        "os.startfile(",
        '["cmd", "/c", "start"',
        "--new-tab",
        "webbrowser.open(",
    )
    return any(token in executed_python_code for token in browser_open_tokens)


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

    download_replan = (
        _looks_like_download_or_install_task(prompt)
        and not _looks_like_existing_installer_launch_task(prompt)
        and _last_execution_opened_browser_for_gui_flow(last_execution)
        and any(
            reason in {"execution_error", "download_url_404", "installer_url_not_found"}
            for reason in unique_reasons
        )
    )
    if download_replan:
        override_lines = [
            "REPLAN OVERRIDE FOR THIS STEP:",
            "Return executable Python only.",
            "The previous attempt already opened the relevant browser page. Treat the current screenshot as the primary source of truth for the next action.",
            "Continue from the visible browser/download UI with Python GUI automation before trying any new network fetch or HTML parsing logic.",
            "Do not use urllib, requests, regex-based HTML scraping, or fresh direct-download discovery in this step unless the current screenshot clearly shows that the browser path is impossible.",
            "Prefer OCR-grounded helpers such as click_download_like_target() or click_text_targets([...]) if a download-like or installer-like control is visible on screen.",
            "Use Python GUI actions to focus the browser, activate the visible official page, click the visible download control, and then wait for the download artifact to stabilize in Downloads.",
            "If the browser is already on an official vendor page or search result page, keep following that visible path instead of restarting from scratch.",
        ]
        stdout_tail = str(last_execution.get("stdout_tail") or "").strip()
        stderr_tail = str(last_execution.get("stderr_tail") or "").strip()
        if stdout_tail:
            override_lines.append(f"Previous stdout summary: {stdout_tail[-240:]}")
        if stderr_tail:
            override_lines.append(f"Previous stderr summary: {stderr_tail[-240:]}")
        return "\n".join(override_lines)

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
                "Check only likely install directories such as LOCALAPPDATA\\\\<TargetApp>, LOCALAPPDATA\\\\Programs\\\\<TargetApp>, Program Files\\\\<TargetApp>, and Program Files (x86)\\\\<TargetApp>, or use a narrow os.walk with onerror handling."
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


def _should_omit_screenshot_for_generation(
    *,
    user_prompt: str,
    execution_style: str,
    last_execution: dict[str, Any],
) -> bool:
    if str(execution_style or "python_first").lower() == "gui_first":
        return False
    if _looks_like_existing_installer_launch_task(user_prompt):
        return False
    return _looks_like_download_discovery_retry_without_useful_visual_state(
        user_prompt=user_prompt,
        last_execution=last_execution,
    )


def _generation_screenshot_fields(
    *,
    state: dict[str, Any],
    user_prompt: str,
    execution_style: str,
    last_execution: dict[str, Any],
) -> tuple[Any, Any, Any]:
    if _should_omit_screenshot_for_generation(
        user_prompt=user_prompt,
        execution_style=execution_style,
        last_execution=last_execution,
    ):
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


def _extract_prompt_urls(text: str) -> list[str]:
    matches = re.findall(r"https?://[^\s'\"`)>]+", str(text or ""))
    seen: set[str] = set()
    urls: list[str] = []
    for raw in matches:
        cleaned = raw.rstrip(".,;:])}>")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        urls.append(cleaned)
    return urls


def _select_prompt_browser_url(text: str) -> str | None:
    prompt_urls = _extract_prompt_urls(text)
    if not prompt_urls:
        return None

    suspicious_host_prefixes = ("pc-", "app-", "apps-", "download-", "downloads-", "client-", "desktop-", "win-")
    non_vendor_hosts = (
        "apps.microsoft.com",
        "play.google.com",
        "apps.apple.com",
        "youtube.",
        "youtu.be",
        "reddit.",
        "tistory.",
        "blog.",
        "medium.com",
    )

    def _score(url: str) -> tuple[int, int]:
        lowered = url.lower()
        parsed = urllib.parse.urlparse(url)
        host = str(parsed.netloc or "").lower()
        path = str(parsed.path or "").lower()
        query = str(parsed.query or "").lower()
        labels = [label for label in host.split(".") if label]
        lead_label = labels[0] if labels else ""
        score = 0
        if lowered.endswith(".exe"):
            score -= 200
        if _url_looks_like_search_results(url):
            score -= 60
        if lead_label and any(lead_label.startswith(prefix) for prefix in suspicious_host_prefixes):
            score -= 70
        if any(host_part in host for host_part in non_vendor_hosts):
            score -= 90
        if any(token in lowered for token in ("/download", "/downloads", "/service", "/product", "/products", "/page/")):
            score += 40
        if any(token in path for token in ("/notice", "/notices", "/support", "/help", "/docs")):
            score -= 40
        if "/page/service/" in path:
            score += 16
        if path == "/" or not path:
            score -= 12
        if any(token in lowered for token in ("lang=", "locale=", "notice", "notices", "release", "releases")):
            score += 10
        if host and path and path != "/":
            score += 15
        if "cdn" in host or host.startswith("download."):
            score -= 18
        if query and "q=" in query and not _url_looks_like_search_results(url):
            score -= 5
        if lowered.count("/") <= 2:
            score -= 20
        score -= host.count("-") * 8
        if len(labels) <= 3:
            score += 8
        if lead_label and len(lead_label) <= 12:
            score += 6
        return (score, -len(url))

    return sorted(prompt_urls, key=_score, reverse=True)[0]


def _url_looks_like_search_results(url: str | None) -> bool:
    cleaned = str(url or "").strip()
    if not cleaned:
        return False
    parsed = urllib.parse.urlparse(cleaned)
    host = str(parsed.netloc or "").lower()
    path = str(parsed.path or "").lower()
    query = str(parsed.query or "").lower()
    if any(
        token in host
        for token in (
            "bing.",
            "google.",
            "duckduckgo.",
            "yahoo.",
            "naver.",
            "daum.",
            "search.",
        )
    ):
        return True
    if path.endswith("/search") or path == "/search":
        return True
    return "q=" in query and any(token in host for token in ("bing", "google", "duckduckgo", "yahoo", "naver", "daum", "search"))


def _fallback_browser_search_url(text: str) -> str | None:
    keywords = _prompt_keyword_candidates(text, limit=4)
    if not keywords:
        return None
    joined = " ".join(keywords)
    if re.search(r"[\uac00-\ud7a3]", joined):
        query_terms = [*keywords, "공식", "다운로드", "pc", "windows"]
    else:
        query_terms = [*keywords, "official", "windows", "download"]
    query = urllib.parse.quote(" ".join(query_terms))
    return f"https://www.bing.com/search?q={query}"


def _last_execution_opened_search_results(last_execution: dict[str, Any]) -> bool:
    payload_metadata = dict(last_execution.get("payload_metadata") or {})
    executed_python_code = str(payload_metadata.get("executed_python_code") or "")
    if not executed_python_code:
        return False
    return any(_url_looks_like_search_results(url) for url in _extract_prompt_urls(executed_python_code))


def _looks_like_download_discovery_retry_without_useful_visual_state(
    *,
    user_prompt: str,
    last_execution: dict[str, Any],
) -> bool:
    if not _looks_like_download_or_install_task(user_prompt):
        return False
    if not last_execution:
        return True
    combined = "\n".join(
        str(last_execution.get(key) or "")
        for key in ("stdout_tail", "stderr_tail")
    ).lower()
    markers = (
        "http error 404",
        "404: not found",
        "download failed:",
        "failed to fetch page:",
        "failed to find installer url",
        "could not find installer url",
        "no installer url found",
        "failed to find download url",
        "could not find download url",
        "no download url found",
    )
    return any(marker in combined for marker in markers)


def _generated_code_ignores_prompt_urls(
    *,
    user_prompt: str,
    python_code: str,
    active_replan_reasons: list[str],
) -> bool:
    if not _looks_like_download_or_install_task(user_prompt):
        return False
    prompt_urls = _extract_prompt_urls(user_prompt)
    if not prompt_urls:
        return False
    normalized_code = _normalize_python_code(python_code).lower()
    if "http" not in normalized_code:
        return False
    code_urls = _extract_prompt_urls(normalized_code)
    if code_urls:
        visible_download_flow_tokens = (
            "advance_visible_download_flow(",
            "click_search_result_like_target(",
            "click_download_like_target(",
            "click_text_targets(",
            "open_url_and_wait(",
        )
        allowed_discovery_urls = [
            url
            for url in code_urls
            if _url_looks_like_search_results(url)
        ]
        if allowed_discovery_urls and any(token in normalized_code for token in visible_download_flow_tokens):
            disallowed_urls = [
                url
                for url in code_urls
                if not _url_looks_like_search_results(url)
                and not any(url.lower() == prompt_url.lower() for prompt_url in prompt_urls)
            ]
            if not disallowed_urls:
                return False
    return not any(url.lower() in normalized_code for url in prompt_urls)


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


def _prompt_keyword_candidates(text: str, *, limit: int = 12) -> list[str]:
    raw_text = str(text or "")
    words = re.findall(r"[a-z0-9가-힣][a-z0-9가-힣._-]{1,}", raw_text.lower())
    stop_words = {
        "return",
        "replan",
        "override",
        "previous",
        "attempt",
        "attempts",
        "any",
        "new",
        "gui",
        "automation",
        "summary",
        "stderr",
        "stdout",
        "ocr",
        "text",
        "cues",
        "search-result",
        "retry",
        "verifier",
        "evidence",
        "pattern",
        "passed",
        "kind",
        "cont",
        "또는",
        "이을",
        "primary",
        "source",
        "truth",
        "before",
        "after",
        "next",
        "action",
        "opened",
        "relevant",
        "treat",
        "trying",
        "focus",
        "activate",
        "executable",
        "python",
        "only",
        "chunk",
        "download",
        "downloads",
        "installer",
        "install",
        "official",
        "windows",
        "current",
        "target",
        "machine",
        "community",
        "https",
        "http",
        "www",
        "com",
        "net",
        "org",
        "pc",
        "page",
        "pages",
        "html",
        "link",
        "links",
        "url",
        "prompt",
        "from",
        "into",
        "task",
        "the",
        "and",
        "open",
        "agent",
        "talk",
        "notices",
        "notice",
        "win32",
        "win64",
        "x64",
        "x86_64",
        "for",
        "this",
        "that",
        "with",
        "without",
        "then",
        "same",
        "step",
        "prefer",
        "continuing",
        "continue",
        "currently",
        "visible",
        "browser",
        "search",
        "results",
        "result",
        "window",
        "app",
        "direct",
        "fetch",
        "fetching",
        "fresh",
        "scraping",
        "silent",
        "shortcuts",
        "unless",
        "latest",
        "version",
        "shows",
        "show",
        "stalled",
        "state",
        "screenshot",
        "installer",
        "wizard",
        "control",
        "controls",
        "button",
        "buttons",
        "설치",
        "설치해줘",
        "다운로드",
        "프로그램",
        "프로그램을",
        "버전",
        "pc버전",
        "윈도우",
        "공식",
        "페이지",
        "실행",
        "파일",
        "폴더",
        "브라우저",
        "검색결과",
        "dialog",
        "when",
        "already",
        "screen",
        "grounded",
        "obtain",
        "using",
        "where",
        "there",
        "use",
        "userprofile",
        "downloads",
        "computer-use-agent",
        "targetapp",
        "ask",
        "human",
        "manual",
        "perform",
        "generated",
        "outside",
        "already",
        "available",
        "proves",
        "stalled",
    }
    korean_particle_suffixes = (
        "으로는",
        "에서는",
        "에게는",
        "한테는",
        "으로",
        "에서",
        "에게",
        "한테",
        "까지",
        "부터",
        "보다",
        "처럼",
        "라고",
        "이라",
        "라도",
        "이다",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "와",
        "과",
        "도",
    )
    seen: set[str] = set()
    result: list[str] = []
    candidate_words: list[str] = []
    task_segments: list[str] = []
    for pattern in (
        r"from this task:\s*(.+?)(?:\.\s|\n|$)",
        r"source task:\s*(.+?)(?:\.\s|\n|$)",
        r"source_task\"\s*:\s*\"(.+?)\"",
    ):
        task_segments.extend(match.group(1) for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE))
    task_candidate_words: list[str] = []
    for segment in task_segments:
        task_candidate_words.extend(re.findall(r"[a-z0-9가-힣][a-z0-9가-힣._-]{1,}", segment.lower()))
    if task_candidate_words:
        candidate_words.extend(task_candidate_words)
    url_candidate_words: list[str] = []
    for url in _extract_prompt_urls(text):
        parsed = urllib.parse.urlparse(str(url))
        host = str(parsed.netloc or "").lower()
        host_parts = [
            part
            for part in re.split(r"[^a-z0-9가-힣]+", host)
            if part
        ]
        if len(host_parts) >= 2:
            vendor = host_parts[-2]
            if vendor:
                url_candidate_words.append(vendor)
            subdomain = host_parts[0]
            if subdomain in {"pc", "app", "download", "client", "desktop"}:
                url_candidate_words.append(subdomain)
        else:
            url_candidate_words.extend(host_parts)
    candidate_words.extend(url_candidate_words)
    if not candidate_words:
        candidate_words.extend(words)
    for word in candidate_words:
        cleaned = word.strip("._-")
        if not cleaned or cleaned[0].isdigit():
            continue
        if re.search(r"[가-힣]", cleaned):
            for suffix in korean_particle_suffixes:
                if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 1:
                    cleaned = cleaned[: -len(suffix)]
                    break
        min_len = 2 if re.search(r"[가-힣]", cleaned) else 3
        if len(cleaned) < min_len or cleaned in stop_words or cleaned in seen:
            continue
        if cleaned.startswith(("targetapp-", "downloads-", "userprofile-")):
            continue
        if "." in cleaned and "/" not in cleaned:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _visible_flow_extra_targets(request: StepRequest | None, *, limit: int = 4) -> list[str]:
    if request is None:
        return []
    merged: list[str] = []
    last_execution_payload = dict(request.last_execution.get("payload_metadata") or {})
    last_execution_code = str(last_execution_payload.get("executed_python_code") or "")
    last_execution_title_tokens: list[str] = []
    for title_match in re.finditer(r"expected_title_tokens\s*=\s*\[(.*?)\]", last_execution_code, flags=re.S):
        for token_match in re.findall(r'"([^"]+)"|\'([^\']+)\'', str(title_match.group(1) or "")):
            token = next((value for value in token_match if value), "")
            if token:
                last_execution_title_tokens.append(token)
    task_segments: list[str] = []
    for pattern in (
        r"from this task:\s*(.+?)(?:\.\s|\n|$)",
        r"source task:\s*(.+?)(?:\.\s|\n|$)",
        r"source_task\"\s*:\s*\"(.+?)\"",
    ):
        task_segments.extend(
            match.group(1)
            for match in re.finditer(pattern, str(request.user_prompt or ""), flags=re.IGNORECASE)
        )
    for source_text in (
        " ".join(task_segments),
        " ".join(last_execution_title_tokens),
        _select_prompt_browser_url(request.user_prompt or "") or "",
        _select_prompt_browser_url(last_execution_code) or "",
    ):
        for keyword in _prompt_keyword_candidates(str(source_text or ""), limit=limit):
            if keyword not in merged:
                merged.append(keyword)
            if len(merged) >= limit:
                return merged
    if merged:
        return merged[:limit]
    prompt_keywords = _prompt_keyword_candidates(str(request.user_prompt or ""), limit=max(limit * 4, 12))
    generic_workflow_keywords = {
        "network",
        "parsing",
        "logic",
        "browser",
        "download",
        "control",
        "screen",
        "visible",
        "current",
        "page",
        "official",
        "opened",
        "relevant",
        "not",
        "execution-style",
        "executi",
        "on-style",
        "advanced",
        "urllib",
        "requests",
        "regex-based",
        "direct-download",
        "discovery",
        "stabilize",
        "clearly",
        "path",
        "impossible",
        "ocr-grounded",
    }
    for keyword in prompt_keywords:
        if keyword in generic_workflow_keywords or keyword in merged:
            continue
        merged.append(keyword)
        if len(merged) >= limit:
            return merged
    for source_text in (
        request.observation_text,
        request.last_execution.get("stdout_tail"),
        request.last_execution.get("stderr_tail"),
    ):
        for keyword in _prompt_keyword_candidates(str(source_text or ""), limit=limit):
            if keyword not in merged:
                merged.append(keyword)
            if len(merged) >= limit:
                return merged
    return merged


def _has_visible_gui_continuation_cues(request: StepRequest) -> bool:
    if str(request.execution_style or "python_first").lower() != "gui_first":
        return False
    if (
        bool(request.screenshot_base64 or request.screenshot_path)
        and _last_execution_opened_browser_for_gui_flow(request.last_execution)
    ):
        return True
    # Only use grounded runtime state here. The chunk prompt itself often
    # contains phrases like "current screenshot" or "visible browser", which
    # should not disable the browser-open prelude by themselves.
    combined = "\n".join(
        str(value or "")
        for value in (
            request.observation_text,
            request.last_execution.get("stdout_tail"),
            request.last_execution.get("stderr_tail"),
        )
    ).lower()
    markers = (
        "visible browser",
        "visible installer",
        "visible ui",
        "browser session",
        "browser page",
        "search results",
        "download control",
        "download button",
        "download bar",
        "installer wizard",
        "uac prompt",
        "completion dialog",
        "continue from the current",
        "continue from the visible",
        "grounded visible",
        "현재 화면",
        "보이는 ui",
        "보이는 브라우저",
        "검색 결과",
        "공식 다운로드 페이지",
        "다운로드 버튼",
        "다운로드 진행",
        "다운로드 ui",
        "설치 ui",
        "설치 마법사",
        "uac",
        "완료 대화",
        "보이는 다운로드",
        "보이는 설치",
    )
    return any(marker in combined for marker in markers)


def _looks_like_search_results_observation(request: StepRequest | None) -> bool:
    if request is None:
        return False
    combined = "\n".join(
        str(value or "")
        for value in (
            request.observation_text,
            request.last_execution.get("stdout_tail"),
            request.last_execution.get("stderr_tail"),
        )
    ).lower()
    if any(
        marker in combined
        for marker in (
            "search results",
            "검색 결과",
            "bing",
            "google",
            "duckduckgo",
            "yahoo",
            "naver",
            "daum",
        )
    ):
        return True
    return _last_execution_opened_search_results(request.last_execution)


def _click_helper_call_for_request(request: StepRequest | None, *, timeout_s: float) -> str:
    helper_name = "click_download_like_target"
    if request is not None and _looks_like_search_results_observation(request):
        helper_name = "click_search_result_like_target"
    extra_targets: list[str] = []
    if request is not None:
        extra_targets = _visible_flow_extra_targets(request, limit=2)
    helper_args: list[str] = []
    if extra_targets:
        helper_args.append(f"extra_targets={json.dumps(extra_targets, ensure_ascii=False)}")
    helper_args.append(f"timeout_s={float(timeout_s):.1f}")
    return f"{helper_name}({', '.join(helper_args)})"


def _looks_like_gui_first_visible_ui_bypass(request: StepRequest, code: str) -> bool:
    if str(request.execution_style or "python_first").lower() != "gui_first":
        return False
    if not _has_visible_gui_continuation_cues(request):
        return False
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    gui_tokens = (
        "pyautogui.",
        "pygetwindow",
        "click_download_like_target(",
        "click_search_result_like_target(",
        "click_text_targets(",
        "getwindowswithtitle(",
        ".activate(",
        ".restore(",
        ".maximize(",
        "pywinauto",
        "win32gui",
        "locateonscreen(",
        "click(",
        "doubleclick(",
        "press(",
        "hotkey(",
        "typewrite(",
    )
    if any(token in normalized for token in gui_tokens):
        return False
    bypass_tokens = (
        "urllib.request",
        "urlopen(",
        "requests.",
        "httpx.",
        "webbrowser.open(",
        "webdriver.",
        "selenium",
        "href=",
        "download_url",
        "html =",
        "html_text",
        "re.findall(",
    )
    return any(token in normalized for token in bypass_tokens)


def _looks_like_gui_first_silent_install_shortcut(request: StepRequest, code: str) -> bool:
    if str(request.execution_style or "python_first").lower() != "gui_first":
        return False
    if not _looks_like_existing_installer_launch_task(request.user_prompt):
        return False
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    silent_tokens = ("/verysilent", "/silent", "/sp-", "/norestart")
    if not any(token in normalized for token in silent_tokens):
        return False
    gui_tokens = (
        "pyautogui.",
        "pygetwindow",
        "getwindowswithtitle(",
        ".activate(",
        ".restore(",
        ".maximize(",
        "locateonscreen(",
        "click(",
        "doubleclick(",
        "press(",
        "hotkey(",
        "typewrite(",
        "sendkeys",
    )
    window_state_tokens = (
        "installer wizard",
        "uac",
        "completion dialog",
        "license",
        "destination",
        "window title",
        "foreground window",
        "tasklist",
        "process_exists",
    )
    if any(token in normalized for token in gui_tokens):
        return False
    if "subprocess.popen(" not in normalized and "subprocess.run(" not in normalized and "os.startfile(" not in normalized:
        return False
    # Launching with silent flags is fine for python_first, but for gui_first install chunks
    # it is too shallow when there is no installer-window handling or post-install verification.
    return not any(token in normalized for token in window_state_tokens)


def _looks_like_store_detour_generation(request: StepRequest, code: str) -> bool:
    if not _looks_like_download_or_install_task(request.user_prompt):
        return False
    prompt_lower = str(request.user_prompt or "").lower()
    if any(
        token in prompt_lower
        for token in (
            "microsoft store",
            "ms store",
            "windows store",
            "app store",
            "스토어",
        )
    ):
        return False
    normalized = _normalize_python_code(code).lower()
    if not normalized:
        return False
    store_tokens = (
        "ms-windows-store://",
        "apps.microsoft.com",
        "microsoft store",
        "windows store",
        "start microsoft.store",
        "shell:appsfolder",
    )
    return any(token in normalized for token in store_tokens)


def _synthesized_official_download_recovery_code(*, user_prompt: str) -> str:
    prompt_urls = _extract_prompt_urls(user_prompt)
    keyword_candidates = _prompt_keyword_candidates(user_prompt)
    return f"""from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from html import unescape
import json
import os
import re
import sys
import urllib.request

PROMPT_URLS = {json.dumps(prompt_urls, ensure_ascii=False)}
KEYWORDS = {json.dumps(keyword_candidates, ensure_ascii=False)}
USER_AGENT = "Mozilla/5.0"

downloads = Path.home() / "Downloads"
downloads.mkdir(parents=True, exist_ok=True)

generic_bad = ("portable", ".zip", ".7z", ".tar", ".gz", ".msi", ".pkg")
preferred_markers = ("setup", "installer", "install", "win64", "windows", "x64")

def score_url(url: str) -> int:
    lowered = unquote(urlparse(url).path).lower()
    score = 0
    if lowered.endswith(".exe"):
        score += 100
    for keyword in KEYWORDS:
        if keyword in lowered:
            score += 20
    for marker in preferred_markers:
        if marker in lowered:
            score += 8
    for marker in generic_bad:
        if marker in lowered:
            score -= 200
    return score

def request_bytes(url: str) -> tuple[str, bytes]:
    req = urllib.request.Request(url, headers={{"User-Agent": USER_AGENT}})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.geturl(), response.read()

def extract_links(base_url: str, html_text: str) -> tuple[list[str], list[str]]:
    page_links: list[str] = []
    exe_links: list[str] = []
    seen_pages = set()
    seen_exe = set()
    attr_matches = re.findall(r'''(?:href|src)\\s*=\\s*["\\']([^"\\']+)["\\']''', html_text, flags=re.IGNORECASE)
    for raw in attr_matches:
        resolved = urljoin(base_url, unescape(raw)).split("#", 1)[0]
        lowered = resolved.lower()
        if not lowered.startswith("http"):
            continue
        if lowered.endswith(".exe") and lowered not in seen_exe:
            seen_exe.add(lowered)
            exe_links.append(resolved)
            continue
        if any(token in lowered for token in ("download", "install", "release", "community", "edition")) and lowered not in seen_pages:
            seen_pages.add(lowered)
            page_links.append(resolved)
    for raw in re.findall(r'https://[^\\s"\\'<>]+', html_text, flags=re.IGNORECASE):
        resolved = raw.split("#", 1)[0]
        lowered = resolved.lower()
        if lowered.endswith(".exe") and lowered not in seen_exe:
            seen_exe.add(lowered)
            exe_links.append(resolved)
    return page_links[:8], exe_links

def candidate_destination(url: str) -> Path:
    name = Path(unquote(urlparse(url).path)).name or "installer.exe"
    if not name.lower().endswith(".exe"):
        name = "installer.exe"
    return downloads / name

existing_candidates = []
for path in downloads.glob("*.exe"):
    lowered = path.name.lower()
    if not KEYWORDS:
        continue
    if not any(keyword in lowered for keyword in KEYWORDS):
        continue
    if path.stat().st_size > 1_000_000:
        existing_candidates.append(path)

if existing_candidates:
    existing = max(existing_candidates, key=lambda p: p.stat().st_mtime)
    print(f"Found existing installer: {{existing}}")
    sys.exit(0)

visited_pages = set()
page_queue = list(PROMPT_URLS)
exe_candidates: list[str] = []
seen_candidate_urls = set()

while page_queue and len(visited_pages) < 10:
    page_url = page_queue.pop(0)
    if page_url in visited_pages:
        continue
    visited_pages.add(page_url)
    try:
        final_page_url, html_bytes = request_bytes(page_url)
        html_text = html_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"Failed to fetch page {{page_url}}: {{exc}}")
        continue
    extra_pages, exe_links = extract_links(final_page_url, html_text)
    for extra_page in extra_pages:
        if extra_page not in visited_pages:
            page_queue.append(extra_page)
    for exe_url in exe_links:
        if exe_url.lower() in seen_candidate_urls:
            continue
        seen_candidate_urls.add(exe_url.lower())
        exe_candidates.append(exe_url)

exe_candidates.sort(key=score_url, reverse=True)

if not exe_candidates:
    raise SystemExit("No official Windows installer .exe candidate found from the prompt URLs.")

for exe_url in exe_candidates:
    dest = candidate_destination(exe_url)
    print(f"Trying candidate: {{exe_url}}")
    try:
        req = urllib.request.Request(exe_url, headers={{"User-Agent": USER_AGENT}})
        with urllib.request.urlopen(req, timeout=90) as response, open(dest, "wb") as fh:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
        size = dest.stat().st_size if dest.exists() else 0
        if size > 1_000_000:
            print(f"Downloaded: {{dest}} ({{size}} bytes)")
            sys.exit(0)
        if dest.exists():
            dest.unlink(missing_ok=True)
    except Exception as exc:
        print(f"Candidate failed {{exe_url}}: {{exc}}")
        continue

raise SystemExit("All official installer candidates failed.")
"""


def _should_use_framework_official_download_recovery(request: StepRequest) -> bool:
    if not request.replan_requested:
        return False
    if str(request.execution_style or "python_first").lower() == "gui_first":
        return False
    if not _looks_like_download_or_install_task(request.user_prompt):
        return False
    if _has_visible_gui_continuation_cues(request):
        return False
    if not _extract_prompt_urls(request.user_prompt):
        return False
    recovery_reasons = {"download_url_404", "installer_url_not_found", "execution_error"}
    if not recovery_reasons.intersection(request.replan_reasons):
        return False
    return int(request.step_index or 0) >= 1


def _should_use_framework_visible_download_flow(request: StepRequest) -> bool:
    if str(request.execution_style or "python_first").lower() != "gui_first":
        return False
    if not _looks_like_download_or_install_task(request.user_prompt):
        return False
    if _looks_like_existing_installer_launch_task(request.user_prompt):
        return False
    return True


def generate_step_response(
    runtime: AgentRuntime,
    request: StepRequest,
    *,
    max_new_tokens: int,
    generation_context: dict[str, Any] | None = None,
) -> StepResponse:
    if _should_use_framework_official_download_recovery(request):
        code = _synthesized_official_download_recovery_code(user_prompt=request.user_prompt)
        return StepResponse(
            python_code=code,
            raw_text=code,
            model_id="framework:official-download-recovery",
            step_index=request.step_index,
            done=False,
            notes=["framework_official_download_recovery_used"],
        )
    if _should_use_framework_visible_download_flow(request):
        prompt_url = None
        if not _has_visible_gui_continuation_cues(request):
            prompt_url = _select_prompt_browser_url(request.user_prompt) or _fallback_browser_search_url(request.user_prompt)
        code = _synthesized_visible_download_completion_code(
            request,
            prompt_url=prompt_url,
            timeout_s=18.0,
            wait_timeout_s=45.0,
            exit_on_success=False,
            continue_on_failure=False,
        )
        return StepResponse(
            python_code=code,
            raw_text=code,
            model_id="framework:visible-download-flow",
            step_index=request.step_index,
            done=False,
            notes=["framework_visible_download_flow_used"],
        )
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
    request: StepRequest | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    step_dir = root / "steps" / step_id
    expanded_python_code = _prepare_python_code_for_execution(request, python_code)
    executor_metadata = dict(metadata)
    executor_metadata.setdefault("executed_python_code", expanded_python_code)
    exec_result = executor_client.execute(
        python_code=expanded_python_code,
        run_dir=str(step_dir),
        step_id=step_id,
        metadata=executor_metadata,
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
            request=repair_request,
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
            request=None,
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
            execution_style=execution_style,
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
        prompt_url_violation = _generated_code_ignores_prompt_urls(
            user_prompt=user_prompt,
            python_code=response.python_code,
            active_replan_reasons=active_replan_reasons,
        )
        gui_first_visible_ui_violation = _looks_like_gui_first_visible_ui_bypass(request, response.python_code)
        gui_first_silent_install_shortcut = _looks_like_gui_first_silent_install_shortcut(request, response.python_code)
        store_detour_generation = _looks_like_store_detour_generation(request, response.python_code)
        invalid_generation = (
            not _is_compilable_python_code(response.python_code)
            or _looks_like_non_executing_task_script(response.python_code)
            or _looks_like_missing_install_progress_generation(response.python_code, user_prompt)
            or duplicate_generation
            or prompt_url_violation
            or gui_first_visible_ui_violation
            or gui_first_silent_install_shortcut
            or store_detour_generation
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
            if prompt_url_violation:
                response.notes.append("prompt_url_violation_detected")
            if gui_first_visible_ui_violation:
                response.notes.append("gui_first_visible_ui_violation_detected")
            if gui_first_silent_install_shortcut:
                response.notes.append("gui_first_silent_install_shortcut_detected")
            if store_detour_generation:
                response.notes.append("store_detour_generation_detected")
            _write_json(invalid_attempt_path, response.to_dict())
            if gui_first_visible_ui_violation and _has_visible_gui_continuation_cues(request):
                retry_response = StepResponse(
                    python_code=_synthesized_visible_ui_click_recovery_code(request),
                    raw_text=_synthesized_visible_ui_click_recovery_code(request),
                    model_id="framework:visible-ui-click-recovery",
                    step_index=step_index,
                    done=False,
                    notes=[
                        "retry_due_to_gui_first_visible_ui_violation",
                        "framework_visible_ui_click_recovery_used",
                    ],
                )
                invalid_generation_retries_used += 1
                retry_response_path = root / "responses" / f"step-{step_index:03d}.framework-visible-ui-click-00.response.json"
                _write_json(retry_response_path, retry_response.to_dict())
                response = retry_response
                normalized_code = _normalize_python_code(response.python_code)
            elif prompt_url_violation and _should_use_framework_official_download_recovery(request):
                retry_response = StepResponse(
                    python_code=_synthesized_official_download_recovery_code(user_prompt=user_prompt),
                    raw_text=_synthesized_official_download_recovery_code(user_prompt=user_prompt),
                    model_id="framework:official-download-recovery",
                    step_index=step_index,
                    done=False,
                    notes=[
                        "retry_due_to_prompt_url_violation",
                        "framework_official_download_recovery_used",
                    ],
                )
                invalid_generation_retries_used += 1
                retry_response_path = root / "responses" / f"step-{step_index:03d}.framework-recovery-00.response.json"
                _write_json(retry_response_path, retry_response.to_dict())
                response = retry_response
                normalized_code = _normalize_python_code(response.python_code)
            else:
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
                        prompt_url_violation=prompt_url_violation,
                        gui_first_visible_ui_violation=gui_first_visible_ui_violation,
                        gui_first_silent_install_shortcut=gui_first_silent_install_shortcut,
                        store_detour_generation=store_detour_generation,
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
                retry_prompt_url_violation = _generated_code_ignores_prompt_urls(
                    user_prompt=user_prompt,
                    python_code=retry_response.python_code,
                    active_replan_reasons=active_replan_reasons,
                )
                retry_gui_first_visible_ui_violation = _looks_like_gui_first_visible_ui_bypass(request, retry_response.python_code)
                retry_gui_first_silent_install_shortcut = _looks_like_gui_first_silent_install_shortcut(request, retry_response.python_code)
                retry_store_detour_generation = _looks_like_store_detour_generation(request, retry_response.python_code)
                retry_invalid_generation = (
                    not _is_compilable_python_code(retry_response.python_code)
                    or _looks_like_non_executing_task_script(retry_response.python_code)
                    or _looks_like_missing_install_progress_generation(retry_response.python_code, user_prompt)
                    or retry_duplicate_generation
                    or retry_prompt_url_violation
                    or retry_gui_first_visible_ui_violation
                    or retry_gui_first_silent_install_shortcut
                    or retry_store_detour_generation
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
                    if retry_prompt_url_violation:
                        retry_response.notes.append("stopped_due_to_prompt_url_violation")
                    if retry_gui_first_visible_ui_violation:
                        retry_response.notes.append("stopped_due_to_gui_first_visible_ui_violation")
                    if retry_gui_first_silent_install_shortcut:
                        retry_response.notes.append("stopped_due_to_gui_first_silent_install_shortcut")
                    if retry_store_detour_generation:
                        retry_response.notes.append("stopped_due_to_store_detour_generation")
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
            request=request,
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
                prompt_urls = _extract_prompt_urls(user_prompt)
                if prompt_urls:
                    history.append(
                        "system_hint=the current prompt already contains official URL candidates; fetch these exact URLs first before inventing a different host or latest-path: "
                        + ", ".join(prompt_urls[:4])
                    )
                history.append(
                    "system_hint=do not request guessed artifact directories such as /files/latest or /download/latest as HTML unless that exact URL came from fetched official vendor HTML"
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
                prompt_urls = _extract_prompt_urls(user_prompt)
                if prompt_urls:
                    history.append(
                        "system_hint=the prompt already lists official URL candidates; start from those exact pages and follow only official links discovered there: "
                        + ", ".join(prompt_urls[:4])
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
