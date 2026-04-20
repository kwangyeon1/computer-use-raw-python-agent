from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from computer_use_raw_python_agent import web_search


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_searxng_client_forces_google_engine(monkeypatch) -> None:
    seen_urls: list[str] = []

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        return _FakeResponse(
            {
                "results": [
                    {
                        "title": "Target App",
                        "url": "https://example.com/download",
                        "content": "Official download",
                        "engine": "google",
                    }
                ]
            }
        )

    monkeypatch.setattr(web_search, "urlopen", fake_urlopen)

    client = web_search.SearXNGClient(base_url="http://searxng.local")
    result = client.search(query="target app", preferred_engines=["duckduckgo", "bing"])

    assert result.result_count == 1
    assert len(seen_urls) == 1
    params = parse_qs(urlparse(seen_urls[0]).query)
    assert params["engines"] == ["google"]


def test_searxng_client_does_not_fallback_to_all_engines_when_google_empty(monkeypatch) -> None:
    seen_urls: list[str] = []

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        return _FakeResponse({"results": []})

    monkeypatch.setattr(web_search, "urlopen", fake_urlopen)

    client = web_search.SearXNGClient(base_url="http://searxng.local")
    result = client.search(query="target app", preferred_engines=["google"])

    assert result.result_count == 0
    assert len(seen_urls) == 1
    params = parse_qs(urlparse(seen_urls[0]).query)
    assert params["engines"] == ["google"]
