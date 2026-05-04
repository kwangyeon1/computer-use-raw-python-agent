from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys
import time
import traceback
import uuid

from .config_utils import load_policy_from_path
from .models import PromptBundle, StepRequest
from .qwen_runtime import Qwen35RawPythonRuntime
from .runtime import AgentRuntime, ExternalCliRawPythonRuntime
from .service import (
    _choice_control_elements_from_payload,
    _coerce_model_bbox,
    _coerce_model_point,
    _crop_png_bytes,
    _extract_json_object_or_array,
    _installer_ui_candidates_from_observation,
    _installer_ui_candidates_observation,
    _model_ui_ocr_elements_from_text,
    build_executor_client,
    run_agent_control_loop,
)


_STATE_DIR = Path("/tmp/computer_use_raw_python_agent_qwen35")
_STATE_PATH = _STATE_DIR / "agent.state.json"
_LOG_PATH = _STATE_DIR / "agent.log"
_REQUESTS_DIR = _STATE_DIR / "requests"
_RESPONSES_DIR = _STATE_DIR / "responses"


def daemon_state_path() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_PATH


def daemon_log_path() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_PATH


def daemon_requests_dir() -> Path:
    _REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    return _REQUESTS_DIR


def daemon_responses_dir() -> Path:
    _RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    return _RESPONSES_DIR


@dataclass
class AgentDaemonState:
    backend_kind: str
    model_id: str
    processor_id: str | None
    compute_dtype: str
    device_map: str
    load_in_4bit: bool
    load_in_8bit: bool
    enable_fp32_cpu_offload: bool
    defaults: dict[str, Any]
    agent_cli_command: list[str] = field(default_factory=list)
    agent_cli_cwd: str | None = None
    phase: str = "idle"
    runtime: AgentRuntime | None = None

    def ensure_runtime(self) -> AgentRuntime:
        if self.runtime is None:
            if self.backend_kind == "external_cli":
                self.runtime = ExternalCliRawPythonRuntime(
                    command=self.agent_cli_command,
                    cwd=self.agent_cli_cwd,
                    max_new_tokens=int(self.defaults.get("max_new_tokens", 256)),
                )
            else:
                self.runtime = Qwen35RawPythonRuntime(
                    model_id=self.model_id,
                    processor_id=self.processor_id,
                    max_new_tokens=int(self.defaults.get("max_new_tokens", 256)),
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    compute_dtype=self.compute_dtype,
                    device_map=self.device_map,
                    enable_fp32_cpu_offload=self.enable_fp32_cpu_offload,
                )
            self.runtime.ensure_loaded()
        return self.runtime

    def has_backend(self) -> bool:
        if self.backend_kind == "external_cli":
            return bool(self.agent_cli_command)
        return bool(self.model_id)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "agent_cli_command": self.agent_cli_command,
            "agent_cli_cwd": self.agent_cli_cwd,
            "compute_dtype": self.compute_dtype,
            "device_map": self.device_map,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "enable_fp32_cpu_offload": self.enable_fp32_cpu_offload,
            "defaults": self.defaults,
            "phase": self.phase,
        }


def _write_state_file(pid: int, payload: dict[str, Any]) -> None:
    daemon_state_path().write_text(json.dumps({"pid": pid, **payload}, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_state_file() -> dict[str, Any] | None:
    path = daemon_state_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_response_file(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _send_request(payload: dict[str, Any], *, timeout_s: float = 5.0) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    requests_dir = daemon_requests_dir()
    responses_dir = daemon_responses_dir()
    request_path = requests_dir / f"{request_id}.json"
    response_path = responses_dir / f"{request_id}.json"
    temp_path = requests_dir / f"{request_id}.tmp"
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(request_path)

    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if response_path.exists():
            try:
                return json.loads(response_path.read_text(encoding="utf-8"))
            finally:
                response_path.unlink(missing_ok=True)
                request_path.unlink(missing_ok=True)
        time.sleep(0.05)
    request_path.unlink(missing_ok=True)
    raise RuntimeError(f"qwen agent daemon did not respond within {timeout_s}s")


def daemon_is_responding() -> bool:
    try:
        response = _send_request({"action": "status"}, timeout_s=0.5)
    except Exception:
        return False
    return bool(response.get("ok"))


def daemon_process_alive() -> bool:
    state = _read_state_file()
    if not state:
        return False
    pid = state.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_daemon_process() -> None:
    requests_dir = daemon_requests_dir()
    responses_dir = daemon_responses_dir()
    for path in requests_dir.glob("*.json"):
        path.unlink()
    for path in responses_dir.glob("*.json"):
        path.unlink()
    log_handle = daemon_log_path().open("a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-m", "computer_use_raw_python_agent.qwen_daemon", "serve"],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        close_fds=True,
    )


def wait_for_daemon_ready(timeout_s: float = 10.0) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if daemon_is_responding():
            return
        time.sleep(0.1)
    raise RuntimeError(f"qwen agent daemon did not become ready within {timeout_s}s")


def _merge_defaults(current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    return merged


def _make_run_dir(base_run_dir: str | None, prompt: str) -> str:
    if not base_run_dir:
        raise RuntimeError("run_dir is not configured; provide it in config JSON or with --run-dir")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in prompt.strip())[:48].strip("-") or "run"
    ts = time.strftime("%Y%m%d-%H%M%S")
    return str(Path(base_run_dir) / f"{ts}-{slug}")


def _handle_run(daemon_state: AgentDaemonState, payload: dict[str, Any]) -> dict[str, Any]:
    if not daemon_state.has_backend():
        return {"ok": False, "error": "no backend is loaded; start once with --model-id or --agent-cli-command"}
    prompt = str(payload["prompt"]).strip()
    overrides = dict(payload.get("overrides", {}))
    defaults = _merge_defaults(daemon_state.defaults, overrides)
    base_run_dir = str(overrides.get("run_dir") or defaults.get("run_dir") or "")
    run_dir = _make_run_dir(base_run_dir, prompt)
    policy_path = str(defaults.get("policy") or "")
    policy = load_policy_from_path(policy_path)
    executor_client = build_executor_client(
        endpoint=defaults.get("endpoint"),
        mcp_command=defaults.get("mcp_command"),
        mcp_cwd=defaults.get("mcp_cwd"),
    )
    try:
        daemon_state.phase = "running"
        _write_state_file(os.getpid(), daemon_state.to_public_dict())
        summary = run_agent_control_loop(
            runtime=daemon_state.ensure_runtime(),
            executor_client=executor_client,
            user_prompt=prompt,
            policy=policy,
            run_dir=run_dir,
            max_iterations=int(defaults.get("max_iterations", 5)),
            max_new_tokens=int(defaults.get("max_new_tokens", 256)),
            execution_style=str(defaults.get("execution_style", "python_first")),
            strong_visual_grounding=bool(defaults.get("strong_visual_grounding", False)),
            reasoning_enabled=bool(defaults.get("reasoning_enabled", False)),
            replan_enabled=bool(defaults.get("replan_enabled", False)),
            replan_max_attempts=int(defaults.get("replan_max_attempts", 1)),
            web_search_enabled=bool(defaults.get("web_search_enabled", False)),
            web_search_engine=str(defaults.get("web_search_engine", "searxng")),
            searxng_base_url=str(defaults.get("searxng_base_url", "http://127.0.0.1:8080")),
            searxng_preferred_engines=[str(item) for item in defaults.get("searxng_preferred_engines", ["google"])],
            web_search_decision_use_image=bool(defaults.get("web_search_decision_use_image", False)),
            web_search_decision_reasoning_enabled=bool(defaults.get("web_search_decision_reasoning_enabled", False)),
            web_search_decision_max_new_tokens=int(defaults.get("web_search_decision_max_new_tokens", 64)),
            web_search_top_k=int(defaults.get("web_search_top_k", 5)),
            web_search_max_uses=int(defaults.get("web_search_max_uses", 3)),
            web_search_timeout_s=float(defaults.get("web_search_timeout_s", 10.0)),
            dependency_repair_enabled=bool(defaults.get("dependency_repair_enabled", False)),
            dependency_repair_max_attempts=int(defaults.get("dependency_repair_max_attempts", 2)),
            dependency_repair_allow_shell_fallback=bool(defaults.get("dependency_repair_allow_shell_fallback", False)),
        )
    finally:
        daemon_state.phase = "ready"
        _write_state_file(os.getpid(), daemon_state.to_public_dict())
        executor_client.close()
    return {"ok": True, "summary": summary}


def _handle_reload(daemon_state: AgentDaemonState, payload: dict[str, Any]) -> dict[str, Any]:
    defaults = _merge_defaults(daemon_state.defaults, dict(payload.get("defaults", {})))
    backend_kind = str(payload.get("backend_kind") or daemon_state.backend_kind or "").strip().lower()
    if not backend_kind:
        backend_kind = "external_cli" if payload.get("agent_cli_command") else "model"

    daemon_state.backend_kind = backend_kind
    daemon_state.defaults = defaults
    daemon_state.phase = "loading"
    _write_state_file(os.getpid(), daemon_state.to_public_dict())

    if daemon_state.backend_kind == "external_cli":
        agent_cli_command = [str(part) for part in (payload.get("agent_cli_command") or daemon_state.agent_cli_command)]
        if not agent_cli_command:
            raise RuntimeError("agent_cli_command is required when backend_kind=external_cli")
        daemon_state.model_id = ""
        daemon_state.processor_id = None
        daemon_state.agent_cli_command = agent_cli_command
        daemon_state.agent_cli_cwd = (
            str(payload.get("agent_cli_cwd"))
            if payload.get("agent_cli_cwd") is not None
            else daemon_state.agent_cli_cwd
        )
        daemon_state.runtime = ExternalCliRawPythonRuntime(
            command=daemon_state.agent_cli_command,
            cwd=daemon_state.agent_cli_cwd,
            max_new_tokens=int(defaults.get("max_new_tokens", 256)),
        )
    elif daemon_state.backend_kind == "model":
        model_id = str(payload.get("model_id") or daemon_state.model_id)
        processor_id = payload.get("processor_id")
        if not model_id:
            raise RuntimeError("model_id is required when backend_kind=model")
        daemon_state.model_id = model_id
        daemon_state.processor_id = str(processor_id) if processor_id else None
        daemon_state.agent_cli_command = []
        daemon_state.agent_cli_cwd = None
        daemon_state.compute_dtype = str(payload.get("compute_dtype") or daemon_state.compute_dtype)
        daemon_state.device_map = str(payload.get("device_map") or daemon_state.device_map)
        daemon_state.load_in_4bit = bool(payload.get("load_in_4bit", not bool(payload.get("disable_4bit", False))))
        daemon_state.load_in_8bit = bool(payload.get("load_in_8bit", daemon_state.load_in_8bit))
        if daemon_state.load_in_4bit and daemon_state.load_in_8bit:
            raise RuntimeError("load_in_4bit and load_in_8bit cannot both be enabled")
        daemon_state.enable_fp32_cpu_offload = bool(
            payload.get("enable_fp32_cpu_offload", not bool(payload.get("disable_cpu_offload", False)))
        )
        daemon_state.runtime = Qwen35RawPythonRuntime(
            model_id=daemon_state.model_id,
            processor_id=daemon_state.processor_id,
            max_new_tokens=int(defaults.get("max_new_tokens", 256)),
            load_in_4bit=daemon_state.load_in_4bit,
            load_in_8bit=daemon_state.load_in_8bit,
            compute_dtype=daemon_state.compute_dtype,
            device_map=daemon_state.device_map,
            enable_fp32_cpu_offload=daemon_state.enable_fp32_cpu_offload,
        )
    else:
        raise RuntimeError(f"unsupported backend_kind: {daemon_state.backend_kind}")

    daemon_state.runtime.ensure_loaded()
    daemon_state.phase = "ready"
    _write_state_file(os.getpid(), daemon_state.to_public_dict())
    return {"ok": True, "status": daemon_state.to_public_dict()}


def _handle_debug_installer_ocr(daemon_state: AgentDaemonState, payload: dict[str, Any]) -> dict[str, Any]:
    if not daemon_state.has_backend():
        return {"ok": False, "error": "no backend is loaded; start once with --model-id or --agent-cli-command"}
    defaults = _merge_defaults(daemon_state.defaults, dict(payload.get("overrides", {})))
    prompt = str(
        payload.get("prompt")
        or "Find the existing installer `.exe` in Downloads, run the installer, finish the installation, and launch the installed app."
    )
    base_run_dir = str(defaults.get("run_dir") or "")
    if not base_run_dir:
        raise RuntimeError("run_dir is not configured")
    run_dir = Path(_make_run_dir(base_run_dir, "debug-installer-ocr"))
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    executor_client = build_executor_client(
        endpoint=defaults.get("endpoint"),
        mcp_command=defaults.get("mcp_command"),
        mcp_cwd=defaults.get("mcp_cwd"),
    )
    try:
        daemon_state.phase = "running"
        _write_state_file(os.getpid(), daemon_state.to_public_dict())
        state = executor_client.observe(screenshot_region=dict(payload.get("screenshot_region") or {"mode": "installer_window"}))
        request = StepRequest(
            user_prompt=prompt,
            policy=load_policy_from_path(str(defaults.get("policy") or "")),
            execution_style="gui_first",
            screenshot_base64=state.get("screenshot_base64"),
            screenshot_media_type=state.get("screenshot_media_type"),
            screenshot_region=state.get("screenshot_region") if isinstance(state.get("screenshot_region"), dict) else None,
            observation_text=state.get("observation_text"),
            strong_visual_grounding=True,
            reasoning_enabled=False,
            step_index=int(payload.get("step_index", 1)),
        )
        observation = _installer_ui_candidates_observation(
            runtime=daemon_state.ensure_runtime(),
            request=request,
            max_new_tokens=int(payload.get("max_new_tokens") or defaults.get("max_new_tokens") or 512),
            generation_context={"run_dir": run_dir, "step_id": "debug-installer-ocr"},
        )
        candidates = _installer_ui_candidates_from_observation(observation or "")
        multi_crop_results: list[dict[str, Any]] = []
        image_bytes = None
        try:
            import base64

            image_bytes = base64.b64decode(str(state.get("screenshot_base64") or ""))
        except Exception:
            image_bytes = None
        if bool(payload.get("multi_crop", False)) and image_bytes:
            image_size = None
            if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
                image_size = (int.from_bytes(image_bytes[16:20], "big"), int.from_bytes(image_bytes[20:24], "big"))
            if image_size:
                crop_width, crop_height = image_size
                crop_specs = [
                    ("full_installer", (0, 0, crop_width, crop_height)),
                    ("lower_controls", (0, int(crop_height * 0.62), crop_width, crop_height)),
                    ("agreement_band", (0, int(crop_height * 0.68), int(crop_width * 0.72), int(crop_height * 0.92))),
                    ("lower_left_controls", (0, int(crop_height * 0.72), int(crop_width * 0.46), int(crop_height * 0.96))),
                ]
                for custom in payload.get("custom_crop_specs") or []:
                    if not isinstance(custom, dict):
                        continue
                    name = str(custom.get("name") or "").strip() or f"custom_{len(crop_specs)}"
                    box = custom.get("box")
                    if not isinstance(box, (list, tuple)) or len(box) < 4:
                        continue
                    try:
                        left, top, right, bottom = [int(value) for value in box[:4]]
                    except (TypeError, ValueError):
                        continue
                    left = max(0, min(left, crop_width - 1))
                    top = max(0, min(top, crop_height - 1))
                    right = max(left + 1, min(right, crop_width))
                    bottom = max(top + 1, min(bottom, crop_height))
                    crop_specs.append((name, (left, top, right, bottom)))
                runtime = daemon_state.ensure_runtime()
                region = state.get("screenshot_region") if isinstance(state.get("screenshot_region"), dict) else {}
                region_left = int(region.get("left") or 0)
                region_top = int(region.get("top") or 0)
                for crop_name, crop_box in crop_specs:
                    cropped = _crop_png_bytes(image_bytes, crop_box)
                    if not cropped:
                        continue
                    crop_bytes, sub_size = cropped
                    generated = runtime.generate_text(
                        prompt_bundle=PromptBundle(
                            system_prompt=(
                                "Return compact strict JSON only. Do not return markdown, Python, prose, or reasoning. "
                                "Extract clickable checkbox/radio controls from this Windows installer/dialog crop."
                            ),
                            user_prompt=json.dumps(
                                {
                                    "crop_name": crop_name,
                                    "task": prompt,
                                    "instructions": [
                                        "Return only real clickable checkbox/radio inputs and their adjacent labels.",
                                        "Ignore decorative colored bullet squares inside license/body text.",
                                        "Ignore explanatory sentences unless a visible square/circle input is adjacent on the same row.",
                                        "If an agreement checkbox/radio is visible, include it even if the label is short.",
                                        "Use bbox and point relative to this crop. For Qwen3.5/Qwen3-VL, use normalized 0-1000 coordinates.",
                                        "Output exactly: {\"controls\":[{\"control\":{\"kind\":\"checkbox|radio\",\"bbox\":[l,t,r,b],\"point\":[x,y],\"confidence\":0.0},\"label\":{\"text\":\"...\",\"bbox\":[l,t,r,b]}}]}",
                                    ],
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            session_prompt=prompt,
                            policy=request.policy,
                            execution_style="gui_first",
                            reasoning_enabled=False,
                            observation_text=None,
                            last_execution={},
                            web_search_context={},
                            recent_history=[],
                            replan_requested=False,
                            replan_reasons=[],
                        ),
                        image_bytes=crop_bytes,
                        use_blank_image=False,
                        max_new_tokens=int(payload.get("max_new_tokens") or defaults.get("max_new_tokens") or 512),
                        generation_context={"run_dir": run_dir, "step_id": f"debug-installer-ocr-{crop_name}"},
                    )
                    parsed = _extract_json_object_or_array(generated.text)
                    elements = _choice_control_elements_from_payload(parsed)
                    if not elements:
                        elements = _model_ui_ocr_elements_from_text(generated.text)
                    converted: list[dict[str, Any]] = []
                    for item in elements:
                        point = _coerce_model_point(item.get("point") or item.get("click_point"), image_size=sub_size)
                        bbox = _coerce_model_bbox(item.get("bbox") or item.get("box") or item.get("rect"), image_size=sub_size)
                        if point is None and bbox is not None:
                            point = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                        if point is None:
                            continue
                        local_x = int(crop_box[0] + point[0])
                        local_y = int(crop_box[1] + point[1])
                        converted.append(
                            {
                                "text": str(item.get("text") or item.get("label_text") or item.get("label") or ""),
                                "kind": str(item.get("kind") or item.get("type") or "control"),
                                "source_crop": crop_name,
                                "crop_point": [int(point[0]), int(point[1])],
                                "installer_crop_point": [local_x, local_y],
                                "screen_point": [region_left + local_x, region_top + local_y],
                                "raw": item,
                            }
                        )
                    multi_crop_results.append(
                        {
                            "crop_name": crop_name,
                            "crop_box": list(crop_box),
                            "crop_size": list(sub_size),
                            "model_id": generated.model_id,
                            "raw_text": generated.text[:4000],
                            "controls": converted,
                        }
                    )
        click_result: dict[str, Any] | None = None
        click_candidates = candidates
        if bool(payload.get("prefer_multi_crop", False)):
            requested_source_crop = str(payload.get("click_source_crop") or "").strip()
            flattened = [
                {
                    "text": item.get("text") or "ocr-control",
                    "click_point": item.get("screen_point"),
                    "source": result.get("crop_name"),
                    "raw": item.get("raw"),
                }
                for result in multi_crop_results
                for item in result.get("controls", [])
                if isinstance(item.get("screen_point"), list)
                and (not requested_source_crop or str(result.get("crop_name") or "") == requested_source_crop)
            ]
            if flattened:
                click_candidates = flattened
        if bool(payload.get("click", False)) and click_candidates:
            first = click_candidates[0]
            point = first.get("refined_click_point") or first.get("click_point")
            if isinstance(point, list) and len(point) >= 2:
                x, y = int(point[0]), int(point[1])
                code = (
                    "import time\n"
                    "import pyautogui\n"
                    f"pyautogui.moveTo({x}, {y}, duration=0.1)\n"
                    f"pyautogui.click({x}, {y})\n"
                    "time.sleep(0.8)\n"
                    f"print('debug installer ocr clicked {first.get('text')} at ({x}, {y})')\n"
                )
                click_result = executor_client.execute(
                    python_code=code,
                    run_dir=str(run_dir / "executor"),
                    step_id="debug-click",
                    metadata={"debug_action": "installer_ocr_click", "candidate": first},
                    screenshot_region=dict(payload.get("screenshot_region") or {"mode": "installer_window"}),
                )
        return {
            "ok": True,
            "run_dir": str(run_dir),
            "screenshot_region": state.get("screenshot_region"),
            "observation": observation,
            "candidates": candidates,
            "multi_crop_results": multi_crop_results,
            "click_result": click_result,
        }
    finally:
        daemon_state.phase = "ready"
        _write_state_file(os.getpid(), daemon_state.to_public_dict())
        executor_client.close()


def _serve() -> int:
    requests_dir = daemon_requests_dir()
    responses_dir = daemon_responses_dir()
    daemon_state = AgentDaemonState(
        backend_kind="",
        model_id="",
        processor_id=None,
        compute_dtype="bfloat16",
        device_map="auto",
        load_in_4bit=True,
        load_in_8bit=False,
        enable_fp32_cpu_offload=True,
        defaults={},
        agent_cli_command=[],
        agent_cli_cwd=None,
        phase="idle",
    )
    _write_state_file(os.getpid(), daemon_state.to_public_dict())
    try:
        while True:
            handled_any = False
            for request_path in sorted(requests_dir.glob("*.json")):
                handled_any = True
                response_path = responses_dir / request_path.name
                try:
                    payload = json.loads(request_path.read_text(encoding="utf-8"))
                    action = payload.get("action")
                    if action == "status":
                        response = {"ok": True, "status": daemon_state.to_public_dict()}
                    elif action == "reload":
                        response = _handle_reload(daemon_state, payload)
                    elif action == "run":
                        response = _handle_run(daemon_state, payload)
                    elif action == "debug_installer_ocr":
                        response = _handle_debug_installer_ocr(daemon_state, payload)
                    elif action == "shutdown":
                        response = {"ok": True}
                        _write_response_file(response_path, response)
                        request_path.unlink(missing_ok=True)
                        return 0
                    else:
                        response = {"ok": False, "error": f"unknown action: {action!r}"}
                except Exception as exc:  # pragma: no cover - daemon safety path
                    daemon_state.phase = "ready"
                    _write_state_file(os.getpid(), daemon_state.to_public_dict())
                    response = {
                        "ok": False,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    }
                _write_response_file(response_path, response)
                request_path.unlink(missing_ok=True)
            if not handled_any:
                time.sleep(0.05)
    finally:
        daemon_state_path().unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        raise SystemExit("usage: python -m computer_use_raw_python_agent.qwen_daemon serve")
    raise SystemExit(_serve())


if __name__ == "__main__":
    main()
