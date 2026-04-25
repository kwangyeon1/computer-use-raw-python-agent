from __future__ import annotations

from computer_use_raw_python_agent import cli, qwen_cli


def test_qwen_ensure_daemon_started_restarts_stale_process(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(qwen_cli, "daemon_is_responding", lambda: False)
    monkeypatch.setattr(qwen_cli, "daemon_process_alive", lambda: True)
    monkeypatch.setattr(qwen_cli, "_force_stop_stale_daemon", lambda: events.append("stop"))
    monkeypatch.setattr(qwen_cli, "start_daemon_process", lambda: events.append("start"))
    monkeypatch.setattr(qwen_cli, "wait_for_daemon_ready", lambda: events.append("wait"))

    qwen_cli._ensure_daemon_started()

    assert events == ["stop", "start", "wait"]


def test_qwen_ensure_daemon_started_skips_restart_when_responsive(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(qwen_cli, "daemon_is_responding", lambda: True)
    monkeypatch.setattr(qwen_cli, "daemon_process_alive", lambda: True)
    monkeypatch.setattr(qwen_cli, "_force_stop_stale_daemon", lambda: events.append("stop"))
    monkeypatch.setattr(qwen_cli, "start_daemon_process", lambda: events.append("start"))
    monkeypatch.setattr(qwen_cli, "wait_for_daemon_ready", lambda: events.append("wait"))

    qwen_cli._ensure_daemon_started()

    assert events == []


def test_generic_ensure_daemon_started_restarts_stale_process(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(cli, "daemon_is_responding", lambda: False)
    monkeypatch.setattr(cli, "daemon_process_alive", lambda: True)
    monkeypatch.setattr(cli, "_force_stop_stale_daemon", lambda: events.append("stop"))
    monkeypatch.setattr(cli, "start_daemon_process", lambda: events.append("start"))
    monkeypatch.setattr(cli, "wait_for_daemon_ready", lambda: events.append("wait"))

    cli._ensure_daemon_started()

    assert events == ["stop", "start", "wait"]


def test_generic_ensure_daemon_started_starts_when_process_missing(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(cli, "daemon_is_responding", lambda: False)
    monkeypatch.setattr(cli, "daemon_process_alive", lambda: False)
    monkeypatch.setattr(cli, "_force_stop_stale_daemon", lambda: events.append("stop"))
    monkeypatch.setattr(cli, "start_daemon_process", lambda: events.append("start"))
    monkeypatch.setattr(cli, "wait_for_daemon_ready", lambda: events.append("wait"))

    cli._ensure_daemon_started()

    assert events == ["start", "wait"]
