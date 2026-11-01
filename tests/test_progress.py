"""Progress is observable but cannot alter scientific results or stdout JSON."""
import json
from threading import Event, enumerate as threads

import pytest

from wm_audit import cli
from wm_audit import progress as reporting


@pytest.fixture(autouse=True)
def visible(monkeypatch):
    monkeypatch.setenv("WM_AUDIT_PROGRESS", "1")


def test_plain_progress_counts_order_eta_and_stderr(capsys):
    assert list(reporting.progress([3, 1, 2], desc="Evaluate pilot", unit="root")) == [3, 1, 2]
    out = capsys.readouterr()
    assert not out.out
    assert "0/3 root (0%)" in out.err
    assert "3/3 root (100%)" in out.err
    assert "elapsed" in out.err and "ETA" in out.err
    assert "\r" not in out.err


def test_failure_never_reports_full_progress(capsys):
    with pytest.raises(ValueError):
        with reporting.stage("validation"):
            for value in reporting.progress([1, 2], desc="roots"):
                raise ValueError("bad data")
    text = capsys.readouterr().err
    assert "FAILED/INTERRUPTED validation" in text
    assert "DONE validation" not in text
    assert "100%" not in text


def test_heartbeat_and_worker_cleanup(monkeypatch):
    seen = Event()
    messages = []

    def collect(message):
        messages.append(message)
        if message.startswith("RUNNING"):
            seen.set()

    monkeypatch.setattr(reporting, "note", collect)
    with reporting.stage("model load", heartbeat=0.001):
        assert seen.wait(2), "heartbeat was not emitted"
    assert messages[0] == "START model load"
    assert messages[-1].startswith("DONE model load")
    assert not any(thread.name == "wm-audit-progress" for thread in threads())


def test_disabled_reporting_preserves_values(monkeypatch, capsys):
    monkeypatch.setenv("WM_AUDIT_PROGRESS", "0")
    with reporting.stage("quiet"):
        reporting.note("quiet")
        assert list(reporting.progress([1, 2], desc="quiet")) == [1, 2]
    assert capsys.readouterr() == ("", "")


def test_empty_and_explicit_generator_totals(capsys):
    assert list(reporting.progress([], desc="empty")) == []
    assert list(reporting.progress(iter([7]), desc="generator", total=1)) == [7]
    text = capsys.readouterr().err
    assert "0/0 item (100%)" in text
    assert "1/1 item (100%)" in text


def test_tty_progress(monkeypatch, capsys):
    monkeypatch.setattr(reporting.sys.stderr, "isatty", lambda: True)
    assert list(reporting.progress([1, 2], desc="terminal")) == [1, 2]
    assert "100%" in capsys.readouterr().err


def test_cli_keeps_json_on_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli, "execute", lambda args: {"passed": True})
    cli.main(["--device", "cpu", "preflight"])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"passed": True}
    assert "START preflight" in captured.err
    assert "DONE preflight" in captured.err


def test_cli_failure_is_not_done(monkeypatch, capsys):
    def fail(args):
        raise RuntimeError("deliberate test failure")
    monkeypatch.setattr(cli, "execute", fail)
    with pytest.raises(SystemExit, match="deliberate test failure"):
        cli.main(["preflight"])
    text = capsys.readouterr().err
    assert "FAILED/INTERRUPTED preflight" in text
    assert "DONE preflight" not in text