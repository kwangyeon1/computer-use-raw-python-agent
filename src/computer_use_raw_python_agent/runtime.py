from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
import subprocess
from typing import Any, Protocol

try:
    from .models import GeneratedCode, GeneratedText, PromptBundle
except ImportError:  # pragma: no cover - direct script execution fallback
    from models import GeneratedCode, GeneratedText, PromptBundle


_DEFAULT_BLANK_IMAGE_SIZE = (64, 64)
_DEFAULT_BLANK_IMAGE_COLOR = (255, 255, 255)
_DEFAULT_MAX_IMAGE_DIMENSION = 512


def _is_compilable_python(code: str) -> bool:
    try:
        compile(code, "<agent-generated>", "exec")
        return True
    except SyntaxError:
        return False


def _trim_to_compilable_python(code: str) -> str:
    stripped = code.rstrip()
    if not stripped or _is_compilable_python(stripped):
        return stripped

    lines = stripped.splitlines()
    while lines:
        lines = lines[:-1]
        candidate = "\n".join(lines).rstrip()
        if not candidate:
            break
        if _is_compilable_python(candidate):
            return candidate
    return stripped


def _extract_code_from_candidate(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```python\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return _trim_to_compilable_python(fenced.group(1).strip())
    generic = re.search(r"```\s*(.*?)```", stripped, flags=re.DOTALL)
    if generic:
        return _trim_to_compilable_python(generic.group(1).strip())
    opening_fence = re.search(r"```(?:python)?\s*", stripped, flags=re.IGNORECASE)
    if opening_fence:
        candidate = stripped[opening_fence.end() :]
        closing_fence = re.search(r"\n```", candidate)
        if closing_fence:
            candidate = candidate[: closing_fence.start()]
        return _trim_to_compilable_python(candidate.strip())
    return _trim_to_compilable_python(stripped)


def extract_python_code(text: str) -> str:
    without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in without_think:
        tail = without_think.rsplit("</think>", 1)[-1]
        if tail.strip():
            return _extract_code_from_candidate(tail)
    return _extract_code_from_candidate(without_think)


class AgentRuntime(Protocol):
    def ensure_loaded(self) -> None: ...

    def generate_code(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> GeneratedCode: ...

    def generate_text(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> GeneratedText: ...


def _render_external_cli_prompt(prompt_bundle: PromptBundle) -> str:
    sections = [
        ("system", prompt_bundle.system_prompt),
        ("user", prompt_bundle.user_prompt),
    ]
    rendered_sections: list[str] = []
    for label, content in sections:
        text = str(content or "").strip()
        if not text:
            continue
        rendered_sections.append(f"[{label}]\n{text}")
    return "\n\n".join(rendered_sections).strip()


def _parse_external_cli_stdout(stdout: str) -> tuple[str, str | None, str | None]:
    stripped = str(stdout or "").strip()
    if not stripped:
        raise RuntimeError("external CLI returned empty stdout")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped, None, None

    if not isinstance(payload, dict):
        return stripped, None, None
    if payload.get("ok") is False:
        raise RuntimeError(str(payload.get("error") or "external CLI returned ok=false"))

    explicit_python_code = payload.get("python_code")
    if explicit_python_code is not None and not isinstance(explicit_python_code, str):
        explicit_python_code = str(explicit_python_code)

    raw_text = None
    for key in ("raw_text", "text", "content", "output_text", "message"):
        value = payload.get(key)
        if value is None:
            continue
        raw_text = str(value)
        break
    if raw_text is None and explicit_python_code is not None:
        raw_text = explicit_python_code
    if raw_text is None:
        known_control_keys = {
            "ok",
            "error",
            "python_code",
            "raw_text",
            "text",
            "content",
            "output_text",
            "message",
            "model_id",
            "backend_id",
            "agent_id",
        }
        if not (set(payload.keys()) & known_control_keys):
            return stripped, explicit_python_code, None
    if raw_text is None:
        raise RuntimeError("external CLI JSON response must include raw_text/text/content/output_text/message or python_code")

    backend_id = payload.get("model_id") or payload.get("backend_id") or payload.get("agent_id")
    return raw_text, explicit_python_code, str(backend_id) if backend_id else None


class ExternalCliRawPythonRuntime:
    def __init__(
        self,
        *,
        command: list[str],
        cwd: str | Path | None = None,
        max_new_tokens: int = 256,
        backend_id: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("external CLI command must not be empty")
        self.command = [str(part) for part in command]
        self.cwd = str(Path(cwd).resolve()) if cwd else None
        self.max_new_tokens = max_new_tokens
        self.model_id = str(backend_id or f"external-cli:{Path(self.command[0]).name}")

    def ensure_loaded(self) -> None:
        if not self.command:
            raise RuntimeError("external CLI command must not be empty")

    def generate_code(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> GeneratedCode:
        raw_text, rendered_prompt, explicit_python_code, backend_id = self._generate_raw_text(
            prompt_bundle=prompt_bundle,
            image_path=image_path,
            image_bytes=image_bytes,
            use_blank_image=use_blank_image,
            max_new_tokens=max_new_tokens,
            response_format="python_code",
            generation_context=generation_context,
        )
        code = explicit_python_code if explicit_python_code is not None else extract_python_code(raw_text)
        return GeneratedCode(
            code=code,
            raw_text=raw_text,
            rendered_prompt=rendered_prompt,
            model_id=backend_id,
            prompt_bundle=prompt_bundle.to_dict(),
            screenshot_path=str(Path(image_path).resolve()) if image_path else None,
        )

    def generate_text(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> GeneratedText:
        raw_text, rendered_prompt, _, backend_id = self._generate_raw_text(
            prompt_bundle=prompt_bundle,
            image_path=image_path,
            image_bytes=image_bytes,
            use_blank_image=use_blank_image,
            max_new_tokens=max_new_tokens,
            response_format="text",
            generation_context=generation_context,
        )
        return GeneratedText(
            text=raw_text,
            rendered_prompt=rendered_prompt,
            model_id=backend_id,
            prompt_bundle=prompt_bundle.to_dict(),
            screenshot_path=str(Path(image_path).resolve()) if image_path else None,
        )

    def _generate_raw_text(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
        response_format: str,
        generation_context: dict[str, Any] | None = None,
    ) -> tuple[str, str, str | None, str]:
        self.ensure_loaded()
        resolved_image_path = str(Path(image_path).resolve()) if image_path else None
        rendered_prompt = _render_external_cli_prompt(prompt_bundle)
        payload = {
            "action": "generate",
            "response_format": response_format,
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "prompt_bundle": prompt_bundle.to_dict(),
            "rendered_prompt": rendered_prompt,
            "image_path": resolved_image_path,
            "image_base64": base64.b64encode(image_bytes).decode("ascii") if image_bytes else None,
            "use_blank_image": bool(use_blank_image),
            "generation_context": dict(generation_context or {}),
        }
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            cwd=self.cwd,
            check=False,
        )
        if completed.returncode != 0:
            stderr = str(completed.stderr or "").strip()
            stdout = str(completed.stdout or "").strip()
            detail = stderr or stdout or f"return code {completed.returncode}"
            raise RuntimeError(f"external CLI command failed: {detail}")
        raw_text, explicit_python_code, backend_id = _parse_external_cli_stdout(completed.stdout)
        return raw_text, rendered_prompt, explicit_python_code, str(backend_id or self.model_id)


class GUIOwlRawPythonRuntime:
    def __init__(
        self,
        *,
        model_id: str | Path,
        processor_id: str | Path | None = None,
        trust_remote_code: bool = True,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        quant_type: str = "nf4",
        compute_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        enable_fp32_cpu_offload: bool = True,
    ) -> None:
        self.model_id = str(Path(model_id).resolve())
        self.processor_id = str(Path(processor_id).resolve()) if processor_id else self.model_id
        self.trust_remote_code = trust_remote_code
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.enable_fp32_cpu_offload = enable_fp32_cpu_offload
        self._processor = None
        self._model = None
        self._torch = None
        self._image_cls = None
        self._target_device = None

    def ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if self.load_in_4bit and self.load_in_8bit:
            raise SystemExit("load_in_4bit and load_in_8bit cannot both be enabled")

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = self._build_quantization_config(
                torch,
                BitsAndBytesConfig,
                mode="4bit",
                enable_cpu_offload=False,
            )
        elif self.load_in_8bit:
            quantization_config = self._build_quantization_config(
                torch,
                BitsAndBytesConfig,
                mode="8bit",
                enable_cpu_offload=False,
            )

        self._torch = torch
        self._image_cls = Image
        self._processor = AutoProcessor.from_pretrained(
            self.processor_id,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = self._load_model_with_fallback(
            AutoModelForImageTextToText=AutoModelForImageTextToText,
            quantization_config=quantization_config,
            torch_module=torch,
            BitsAndBytesConfig=BitsAndBytesConfig,
        )
        self._target_device = self._infer_target_device(self._model)

    @property
    def processor(self):
        self.ensure_loaded()
        return self._processor

    @property
    def model(self):
        self.ensure_loaded()
        return self._model

    def generate_code(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> GeneratedCode:
        raw_text, rendered_prompt = self._generate_raw_text(
            prompt_bundle=prompt_bundle,
            image_path=image_path,
            image_bytes=image_bytes,
            use_blank_image=use_blank_image,
            max_new_tokens=max_new_tokens,
        )
        code = extract_python_code(raw_text)
        return GeneratedCode(
            code=code,
            raw_text=raw_text,
            rendered_prompt=str(rendered_prompt),
            model_id=self.model_id,
            prompt_bundle=prompt_bundle.to_dict(),
            screenshot_path=str(Path(image_path).resolve()) if image_path else None,
        )

    def generate_text(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> GeneratedText:
        raw_text, rendered_prompt = self._generate_raw_text(
            prompt_bundle=prompt_bundle,
            image_path=image_path,
            image_bytes=image_bytes,
            use_blank_image=use_blank_image,
            max_new_tokens=max_new_tokens,
        )
        return GeneratedText(
            text=raw_text,
            rendered_prompt=str(rendered_prompt),
            model_id=self.model_id,
            prompt_bundle=prompt_bundle.to_dict(),
            screenshot_path=str(Path(image_path).resolve()) if image_path else None,
        )

    def _generate_raw_text(
        self,
        *,
        prompt_bundle: PromptBundle,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        use_blank_image: bool = True,
        max_new_tokens: int | None = None,
    ) -> tuple[str, str]:
        self.ensure_loaded()
        image = self._load_image(image_path, image_bytes=image_bytes, use_blank_image=use_blank_image)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt_bundle.system_prompt}]},
            {
                "role": "user",
                "content": (
                    [{"type": "image"}] if image is not None else []
                )
                + [{"type": "text", "text": prompt_bundle.user_prompt}],
            },
        ]
        rendered_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(prompt_bundle.reasoning_enabled),
        )
        processor_kwargs: dict[str, Any] = {
            "text": [rendered_prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if image is not None:
            processor_kwargs["images"] = [image]

        batch = self.processor(**processor_kwargs)
        batch = self._move_batch_to_target_device(batch)

        outputs = self.model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens or self.max_new_tokens),
            do_sample=False,
        )
        prompt_length = int(batch["input_ids"].shape[-1])
        generated = outputs[:, prompt_length:]
        raw_text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return raw_text, str(rendered_prompt)

    def _move_batch_to_target_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self._target_device is None:
            return batch
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self._target_device)
            else:
                moved[key] = value
        return moved

    def _load_image(self, image_path: str | Path | None, *, image_bytes: bytes | None = None, use_blank_image: bool):
        image = None
        if image_bytes:
            image = self._image_cls.open(BytesIO(image_bytes)).convert("RGB")
        elif image_path:
            image = self._image_cls.open(image_path).convert("RGB")
        if image is not None:
            max_dimension = max(image.size)
            if max_dimension > _DEFAULT_MAX_IMAGE_DIMENSION:
                resample = getattr(self._image_cls, "Resampling", None)
                filter_mode = resample.LANCZOS if resample is not None else self._image_cls.LANCZOS
                image.thumbnail((_DEFAULT_MAX_IMAGE_DIMENSION, _DEFAULT_MAX_IMAGE_DIMENSION), filter_mode)
            return image
        if use_blank_image:
            return self._image_cls.new("RGB", _DEFAULT_BLANK_IMAGE_SIZE, color=_DEFAULT_BLANK_IMAGE_COLOR)
        return None

    @staticmethod
    def _resolve_dtype(torch_module, value: str):
        normalized = str(value).strip().lower()
        mapping = {
            "bf16": torch_module.bfloat16,
            "bfloat16": torch_module.bfloat16,
            "fp16": torch_module.float16,
            "float16": torch_module.float16,
            "fp32": torch_module.float32,
            "float32": torch_module.float32,
        }
        if normalized not in mapping:
            raise SystemExit(f"unsupported dtype value: {value}")
        return mapping[normalized]

    @staticmethod
    def _infer_target_device(model) -> Any:
        for _, parameter in model.named_parameters():
            return parameter.device
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for device in hf_device_map.values():
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return device
        return None

    def _build_quantization_config(self, torch_module, bitsandbytes_config_cls, *, mode: str, enable_cpu_offload: bool):
        if mode == "4bit":
            return bitsandbytes_config_cls(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._resolve_dtype(torch_module, self.compute_dtype),
                llm_int8_enable_fp32_cpu_offload=bool(enable_cpu_offload),
            )
        if mode == "8bit":
            return bitsandbytes_config_cls(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=bool(enable_cpu_offload),
            )
        raise SystemExit(f"unsupported quantization mode: {mode}")

    def _load_model_with_fallback(self, *, AutoModelForImageTextToText, quantization_config, torch_module, BitsAndBytesConfig):
        common_kwargs = {
            "dtype": self._resolve_dtype(torch_module, self.compute_dtype),
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        try:
            return AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                **common_kwargs,
            )
        except ValueError as exc:
            message = str(exc)
            quantization_mode = "4bit" if self.load_in_4bit else "8bit" if self.load_in_8bit else None
            if not quantization_mode or not self.enable_fp32_cpu_offload:
                raise
            if "dispatched on the CPU or the disk" not in message:
                raise
            fallback_quantization_config = self._build_quantization_config(
                torch_module,
                BitsAndBytesConfig,
                mode=quantization_mode,
                enable_cpu_offload=True,
            )
            return AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=fallback_quantization_config,
                **common_kwargs,
            )
