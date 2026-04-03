from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from .models import GeneratedCode, PromptBundle
except ImportError:  # pragma: no cover - direct script execution fallback
    from models import GeneratedCode, PromptBundle


_DEFAULT_BLANK_IMAGE_SIZE = (64, 64)
_DEFAULT_BLANK_IMAGE_COLOR = (255, 255, 255)


def extract_python_code(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    generic = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if generic:
        return generic.group(1).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


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
    ) -> GeneratedCode:
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
        code = extract_python_code(raw_text)
        return GeneratedCode(
            code=code,
            raw_text=raw_text,
            rendered_prompt=str(rendered_prompt),
            model_id=self.model_id,
            prompt_bundle=prompt_bundle.to_dict(),
            screenshot_path=str(Path(image_path).resolve()) if image_path else None,
        )

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
        if image_bytes:
            return self._image_cls.open(BytesIO(image_bytes)).convert("RGB")
        if image_path:
            return self._image_cls.open(image_path).convert("RGB")
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
