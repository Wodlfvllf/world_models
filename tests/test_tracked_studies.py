"""Supervisor contracts with synthetic files/processes; never run research."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tracked_studies", ROOT / "scripts/run_tracked_studies.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def test_commands_keep_all_studies_and_fixed_order(tmp_path):
    project = tmp_path / "project"
    rows = s.commands(project, project / ".venv/bin/python", tmp_path / "original", tmp_path / "output")
    assert [name for name, _ in rows] == ["numerical-lewm", "numerical-dinowm", "replication"]
    for name, command in rows[:2]:
        assert "--backend-controls" in command
        assert command[command.index("--model") + 1] == name.split("-", 1)[1]
    assert "--execute" in rows[2][1]
    assert "--exclude-data" not in rows[2][1]
    assert all("--trust-checkpoint" in command for _, command in rows)


def test_atomic_status_is_valid_and_replaces(tmp_path):
    path = tmp_path / "status.json"
    s.atomic_json(path, {"state": "running"})
    s.atomic_json(path, {"state": "completed"})
    assert json.loads(path.read_text()) == {"state": "completed"}
    assert not path.with_name("status.json.tmp").exists()


@pytest.mark.parametrize("kind", ["extra", "missing", "hash", "size", "symlink"])
def test_inventory_rejects_changed_payload(tmp_path, kind):
    tmp_path = tmp_path.resolve()
    path = tmp_path / "artifact"
    path.write_bytes(b"synthetic")
    inventory = {"artifact": {"sha256": s.sha(path), "bytes": path.stat().st_size}}
    if kind == "extra":
        (tmp_path / "extra").write_bytes(b"unlisted")
    elif kind == "missing":
        path.unlink()
    elif kind == "hash":
        path.write_bytes(b"different")
    elif kind == "size":
        inventory["artifact"]["bytes"] += 1
    else:
        path.rename(tmp_path / "real")
        path.symlink_to(tmp_path / "real")
    with pytest.raises(ValueError):
        s.check_inventory(tmp_path, inventory)


def test_inventory_accepts_exact_and_completion_exception(tmp_path):
    tmp_path = tmp_path.resolve()
    (tmp_path / "data").write_bytes(b"test fixture")
    (tmp_path / "complete.json").write_text("{}")
    s.check_inventory(tmp_path, {"data": {"sha256": s.sha(tmp_path / "data"), "bytes": 12}},
                      excluded=("complete.json",))


@pytest.mark.parametrize("completed", [False, True])
def test_archive_preserves_success_and_failure_status(tmp_path, completed):
    tmp_path = tmp_path.resolve()
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/source.py").write_text("# synthetic source\n")
    (tmp_path / "numerical-lewm").mkdir()
    (tmp_path / "numerical-lewm/partial.txt").write_text("synthetic only")
    (tmp_path / "status.json").write_text(json.dumps({"completed": completed}))
    (tmp_path / "numerical-lewm.log").write_text("synthetic log")
    receipt = s.archive_stage(tmp_path, "numerical-lewm")
    archive = Path(receipt["path"])
    assert s.sha(archive) == receipt["sha256"]
    assert Path(str(archive) + ".sha256").read_text().split()[0] == receipt["sha256"]
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("BACKUP.json"))
        assert json.loads(z.read("status.json"))["completed"] == completed
        assert set(z.namelist()) == set(manifest["files"]) | {"BACKUP.json"}
        for name, item in manifest["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == item["sha256"]
    with pytest.raises(FileExistsError):
        s.archive_stage(tmp_path, "numerical-lewm")


def test_supervise_preserves_output_and_exit_code(tmp_path):
    changes = []
    code = s.supervise([sys.executable, "-c", "print('SYNTHETIC progress'); raise SystemExit(3)"],
                       tmp_path, tmp_path / "process.log", lambda e, **v: changes.append((e, v)))
    assert code == 3
    assert b"SYNTHETIC progress" in (tmp_path / "process.log").read_bytes()
    assert changes[0][0] == "started" and changes[0][1]["pid"] > 0
    assert any(e == "progress" and v["line"] == "SYNTHETIC progress" for e, v in changes)


def test_supervisor_callback_failure_reaps_detached_child(tmp_path):
    pids = []

    def broken_callback(event, **values):
        pids.append(values["pid"])
        raise ValueError("synthetic status failure")

    with pytest.raises(ValueError, match="status failure"):
        s.supervise([sys.executable, "-c", "while True: pass"], tmp_path,
                    tmp_path / "child.log", broken_callback)
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_failed_archive_verification_never_publishes_zip(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    (tmp_path / "status.json").write_text("{}")
    original_sha = s.sha

    def fail_pending(path):
        if str(path).endswith(".pending"):
            raise ValueError("synthetic verification failure")
        return original_sha(path)

    monkeypatch.setattr(s, "sha", fail_pending)
    with pytest.raises(ValueError, match="verification failure"):
        s.archive_stage(tmp_path, "numerical-lewm")
    assert not (tmp_path / "numerical-lewm.zip").exists()
    assert (tmp_path / "numerical-lewm.zip.pending").exists()


def test_terminal_operations_archive_has_final_state_and_all_logs(tmp_path):
    tmp_path = tmp_path.resolve()
    (tmp_path / "status.json").write_text(json.dumps({"completed": True, "state": "completed"}))
    (tmp_path / "complete.json").write_text(json.dumps({"status_sha256": s.sha(tmp_path / "status.json")}))
    for name in ("numerical-lewm", "numerical-dinowm", "replication"):
        (tmp_path / (name + ".log")).write_text("synthetic operational log")
    receipt = s.archive_stage(tmp_path, "operations")
    with zipfile.ZipFile(receipt["path"]) as z:
        assert json.loads(z.read("status.json"))["completed"] is True
        assert "complete.json" in z.namelist()
        assert sum(name.endswith(".log") for name in z.namelist()) == 3


def test_hosted_entry_refuses_local_research(monkeypatch, tmp_path):
    monkeypatch.setattr(s.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Hosted Linux"):
        s.execute(tmp_path, tmp_path, tmp_path / "run", "0" * 40)


def test_replication_requires_completion_marker(tmp_path):
    with pytest.raises(FileNotFoundError):
        s.verify_stage("replication", tmp_path)


@pytest.mark.parametrize("field", ["model_unchanged", "backend_restored", "parameter_buffer_sha256_after"])
def test_numerical_verifier_refuses_unqualified_report(tmp_path, field):
    report = {"schema": "wm_audit.execution_layout.v1", "model_unchanged": True,
              "backend_restored": True, "parameter_buffer_sha256_before": "a",
              "parameter_buffer_sha256_after": "a", "summary": [{}], "artifacts": {}}
    report[field] = False
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "selection.json").write_text(json.dumps({"roots": list(range(8))}))
    with pytest.raises(ValueError, match="unqualified"):
        s.verify_stage("numerical-lewm", tmp_path)