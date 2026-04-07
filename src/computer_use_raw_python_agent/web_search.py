from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import unescape
from typing import Any
import json
import re
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


_FENCED_JSON_PATTERN = re.compile(r"```json\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _strip_reasoning_and_fences(text: str) -> str:
    without_think = _THINK_PATTERN.sub("", str(text))
    fenced = _FENCED_JSON_PATTERN.search(without_think)
    if fenced:
        return fenced.group(1).strip()
    stripped = without_think.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = _strip_reasoning_and_fences(text)
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(candidate[start : index + 1])
                except json.JSONDecodeError:
                    return {}
                return payload if isinstance(payload, dict) else {}
    return {}


def _normalize_domain(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    hostname = (parsed.netloc or parsed.path or "").strip().lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _domain_matches(hostname: str, expected: str) -> bool:
    normalized_host = _normalize_domain(hostname)
    normalized_expected = _normalize_domain(expected)
    if not normalized_host or not normalized_expected:
        return False
    return normalized_host == normalized_expected or normalized_host.endswith(f".{normalized_expected}")


def _sanitize_text(value: Any, *, max_chars: int = 280) -> str:
    text = _WHITESPACE_PATTERN.sub(" ", unescape(str(value or "")).strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def normalize_query(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", str(value or "").strip())


@dataclass
class WebSearchDecision:
    use_web_search: bool = False
    query: str = ""
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    reason: str = ""
    raw_text: str = ""
    parse_error: str | None = None

    @classmethod
    def from_text(cls, text: str) -> "WebSearchDecision":
        payload = _extract_json_object(text)
        if not payload:
            return cls(raw_text=str(text), parse_error="invalid_json")
        query = normalize_query(str(payload.get("query", "")))
        return cls(
            use_web_search=bool(payload.get("use_web_search", False)) and bool(query),
            query=query,
            allowed_domains=[domain for domain in (_normalize_domain(item) for item in payload.get("allowed_domains", [])) if domain],
            blocked_domains=[domain for domain in (_normalize_domain(item) for item in payload.get("blocked_domains", [])) if domain],
            reason=_sanitize_text(payload.get("reason"), max_chars=240),
            raw_text=str(text),
            parse_error=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WebSearchResult:
    source: str
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    result_count: int = 0
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    status: str = "ok"
    reason: str = ""
    requested_at: str = ""
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearXNGClient:
    def __init__(self, *, base_url: str, timeout_s: float = 10.0, user_agent: str = "computer-use-raw-python-agent/0.1") -> None:
        normalized_base = str(base_url or "").strip().rstrip("/")
        if not normalized_base:
            raise ValueError("searxng base_url must not be empty")
        self.base_url = normalized_base
        self.timeout_s = float(timeout_s)
        self.user_agent = user_agent

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> WebSearchResult:
        normalized_query = normalize_query(query)
        normalized_allowed = [domain for domain in (_normalize_domain(item) for item in (allowed_domains or [])) if domain]
        normalized_blocked = [domain for domain in (_normalize_domain(item) for item in (blocked_domains or [])) if domain]
        params = urlencode({"q": normalized_query, "format": "json"})
        request = Request(
            f"{self.base_url}/search?{params}",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        raw_results = payload.get("results", [])
        parsed_results: list[dict[str, Any]] = []
        for item in raw_results:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            domain = _normalize_domain(url)
            if normalized_allowed and not any(_domain_matches(domain, expected) for expected in normalized_allowed):
                continue
            if normalized_blocked and any(_domain_matches(domain, blocked) for blocked in normalized_blocked):
                continue
            parsed_results.append(
                {
                    "title": _sanitize_text(item.get("title"), max_chars=180),
                    "url": url,
                    "content": _sanitize_text(item.get("content"), max_chars=320),
                    "domain": domain,
                    "engine": _sanitize_text(item.get("engine"), max_chars=48),
                }
            )
            if len(parsed_results) >= max(1, int(top_k)):
                break
        return WebSearchResult(
            source="searxng",
            query=normalized_query,
            results=parsed_results,
            result_count=len(parsed_results),
            allowed_domains=normalized_allowed,
            blocked_domains=normalized_blocked,
            status="ok",
            requested_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )


def make_web_search_error_result(
    *,
    query: str,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
    reason: str,
    error: str,
) -> WebSearchResult:
    return WebSearchResult(
        source="searxng",
        query=normalize_query(query),
        results=[],
        result_count=0,
        allowed_domains=[domain for domain in (_normalize_domain(item) for item in (allowed_domains or [])) if domain],
        blocked_domains=[domain for domain in (_normalize_domain(item) for item in (blocked_domains or [])) if domain],
        status="error",
        reason=_sanitize_text(reason, max_chars=240),
        requested_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        error=_sanitize_text(error, max_chars=320),
    )


def make_web_search_skipped_result(
    *,
    query: str,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
    reason: str,
    status: str,
) -> WebSearchResult:
    return WebSearchResult(
        source="searxng",
        query=normalize_query(query),
        results=[],
        result_count=0,
        allowed_domains=[domain for domain in (_normalize_domain(item) for item in (allowed_domains or [])) if domain],
        blocked_domains=[domain for domain in (_normalize_domain(item) for item in (blocked_domains or [])) if domain],
        status=status,
        reason=_sanitize_text(reason, max_chars=240),
        requested_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def web_search_cache_key(query: str, *, allowed_domains: list[str] | None = None, blocked_domains: list[str] | None = None) -> str:
    normalized_allowed_list = [domain for domain in (_normalize_domain(item) for item in (allowed_domains or [])) if domain]
    normalized_blocked_list = [domain for domain in (_normalize_domain(item) for item in (blocked_domains or [])) if domain]
    normalized_allowed = ",".join(sorted(normalized_allowed_list))
    normalized_blocked = ",".join(sorted(normalized_blocked_list))
    return f"{normalize_query(query)}|allow={normalized_allowed}|block={normalized_blocked}"
