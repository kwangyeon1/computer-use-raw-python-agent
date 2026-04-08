from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RuntimePolicy:
    policy_name: str
    description: str
    flags: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimePolicy":
        known = {
            "policy_name": str(data.get("policy_name", "default")),
            "description": str(data.get("description", "")),
        }
        flags = {k: v for k, v in data.items() if k not in known}
        return cls(
            policy_name=known["policy_name"],
            description=known["description"],
            flags=flags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "description": self.description,
            **self.flags,
        }


@dataclass
class PromptBundle:
    system_prompt: str
    user_prompt: str
    session_prompt: str
    policy: dict[str, Any]
    reasoning_enabled: bool = False
    observation_text: str | None = None
    last_execution: dict[str, Any] = field(default_factory=dict)
    stderr_tail: str | None = None
    web_search_context: dict[str, Any] = field(default_factory=dict)
    recent_history: list[str] = field(default_factory=list)
    replan_requested: bool = False
    replan_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedCode:
    code: str
    raw_text: str
    rendered_prompt: str
    model_id: str
    prompt_bundle: dict[str, Any]
    screenshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedText:
    text: str
    rendered_prompt: str
    model_id: str
    prompt_bundle: dict[str, Any]
    screenshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepRequest:
    user_prompt: str
    policy: dict[str, Any] = field(default_factory=dict)
    request_kind: str = "task_step"
    repair_context: dict[str, Any] = field(default_factory=dict)
    replan_requested: bool = False
    replan_reasons: list[str] = field(default_factory=list)
    strong_visual_grounding: bool = False
    reasoning_enabled: bool = False
    screenshot_path: str | None = None
    screenshot_base64: str | None = None
    screenshot_media_type: str | None = None
    observation_text: str | None = None
    web_search_context: dict[str, Any] = field(default_factory=dict)
    recent_history: list[str] = field(default_factory=list)
    last_execution: dict[str, Any] = field(default_factory=dict)
    step_index: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRequest":
        return cls(
            user_prompt=str(data["user_prompt"]),
            policy=dict(data.get("policy", {})),
            request_kind=str(data.get("request_kind", "task_step")),
            repair_context=dict(data.get("repair_context", {})),
            replan_requested=bool(data.get("replan_requested", False)),
            replan_reasons=[str(item) for item in data.get("replan_reasons", [])],
            strong_visual_grounding=bool(data.get("strong_visual_grounding", False)),
            reasoning_enabled=bool(data.get("reasoning_enabled", False)),
            screenshot_path=data.get("screenshot_path"),
            screenshot_base64=data.get("screenshot_base64"),
            screenshot_media_type=data.get("screenshot_media_type"),
            observation_text=data.get("observation_text"),
            web_search_context=dict(data.get("web_search_context", {})),
            recent_history=[str(item) for item in data.get("recent_history", [])],
            last_execution=dict(data.get("last_execution", {})),
            step_index=int(data.get("step_index", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepResponse:
    python_code: str
    raw_text: str
    model_id: str
    step_index: int
    done: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
