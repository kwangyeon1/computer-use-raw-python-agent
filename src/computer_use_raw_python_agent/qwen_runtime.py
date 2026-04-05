from __future__ import annotations

from pathlib import Path
import sys

try:
    from .runtime import GUIOwlRawPythonRuntime
except ImportError:  # pragma: no cover - direct script execution fallback
    from runtime import GUIOwlRawPythonRuntime


def default_qwen35_model_id() -> str:
    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.append((cwd / "models" / "Qwen3.5-9B").resolve())
    candidates.append((cwd.parent / "models" / "Qwen3.5-9B").resolve())
    argv0 = Path(sys.argv[0]).resolve()
    for parent in argv0.parents:
        candidates.append((parent / "models" / "Qwen3.5-9B").resolve())
        candidates.append((parent.parent / "models" / "Qwen3.5-9B").resolve())
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidates.append((parent / "models" / "Qwen3.5-9B").resolve())
        candidates.append((parent.parent / "models" / "Qwen3.5-9B").resolve())
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str((cwd.parent / "models" / "Qwen3.5-9B").resolve())


class Qwen35RawPythonRuntime(GUIOwlRawPythonRuntime):
    def __init__(
        self,
        *,
        model_id: str | Path | None = None,
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
        resolved_model_id = str(Path(model_id).resolve()) if model_id else default_qwen35_model_id()
        resolved_processor_id = str(Path(processor_id).resolve()) if processor_id else resolved_model_id
        super().__init__(
            model_id=resolved_model_id,
            processor_id=resolved_processor_id,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            quant_type=quant_type,
            compute_dtype=compute_dtype,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            enable_fp32_cpu_offload=enable_fp32_cpu_offload,
        )
