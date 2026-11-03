"""Serial, durable hosted execution of all pending fixed-protocol studies.

Never repeats discovery studies, overwrites outputs, resumes implicitly, or
chooses replacement roots. Each stage is verified and independently archived.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import selectors
import shutil
import signal
import subprocess
import sys
from time import monotonic
import zipfile


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ordinary(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError(f"Require ordinary file without symlink ancestors: {path}")
    return path


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    with pending.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def commands(project, python, original_data, output):
    """Numerical studies are isolated; replication's model order stays fixed."""
    shared = ["--config", str(project / "configs/pilot.json"),
              "--spec", str(project / "configs/followup_20261101.json"),
              "--data", str(original_data), "--cache", str(project / "cache"),
              "--device", "cuda", "--trust-checkpoint", "--backend-controls"]
    result = []
    for model in ("lewm", "dinowm"):
        result.append(("numerical-" + model,
                       [str(python), "-u", str(project / "scripts/audit_execution_layout.py"),
                        *shared, "--model", model, "--output", str(output / ("numerical-" + model))]))
    result.append(("replication", [str(python), "-u", str(project / "scripts/run_replication.py"),
                                  "--config", str(project / "configs/replication_20261102.json"),
                                  "--original-data", str(original_data), "--cache", str(project / "cache"),
                                  "--output", str(output / "replication"), "--device", "cuda",
                                  "--threads", "4", "--trust-checkpoint", "--execute"]))
    return result


def check_inventory(root, inventory, *, size_key="bytes", excluded=()):
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(inventory) | set(excluded):
        raise ValueError("Missing, extra or unlisted stage artifacts")
    for name, receipt in inventory.items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe inventory member")
        path = ordinary(root / name)
        if sha(path) != receipt["sha256"] or path.stat().st_size != receipt[size_key]:
            raise ValueError("Artifact checksum/size mismatch: " + name)


def verify_stage(name, directory):
    if name.startswith("numerical-"):
        report = json.loads((directory / "report.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        if (report.get("schema") != "wm_audit.execution_layout.v1"
                or report.get("model_unchanged") is not True
                or report.get("backend_restored") is not True
                or len(selection["roots"]) != 8
                or report["parameter_buffer_sha256_before"] != report["parameter_buffer_sha256_after"]
                or not report["summary"]):
            raise ValueError("Incomplete or unqualified numerical audit")
        for artifact, digest in report["artifacts"].items():
            if sha(ordinary(directory / artifact)) != digest:
                raise ValueError("Numerical artifact changed")
        return {"report_sha256": sha(directory / "report.json"), "roots": 8,
                "csv_records": report["csv_records"], "controls": report["controls"]}
    completion = json.loads((directory / "complete.json").read_text())
    if completion.get("schema") != "wm_audit.replication.complete.v1":
        raise ValueError("No qualified replication completion marker")
    check_inventory(directory, completion["files"], excluded=("complete.json",))
    result = json.loads((directory / "replication-results.json").read_text())
    family = result["primary_family"]
    if (result.get("schema") != "wm_audit.replication.analysis.v1"
            or len(result["root_ids"]) != 128 or len(set(result["root_ids"])) != 128
            or family["size"] != 5 or len(family["claims"]) != 5
            or family["individual_confidence"] != .99):
        raise ValueError("Incomplete fixed primary replication family")
    return {"report_sha256": sha(directory / "replication-results.json"),
            "roots": 128, "primary_claims": family["claims"], "decisions": result["decisions"]}


def archive_stage(run, stage):
    """Inventory new stage kinds explicitly, including failed/partial attempts."""
    archive = run / (stage + ".zip")
    if archive.exists():
        raise FileExistsError(archive)
    pending = run / (stage + ".zip.pending")
    prefixes = [run / "sources", run / stage]
    files = [p for prefix in prefixes if prefix.exists() for p in prefix.rglob("*")
             if p.is_file() or p.is_symlink()]
    files += [p for p in (run / "launch.json", run / "status.json", run / "events.jsonl",
                          run / "complete.json", run / (stage + ".log")) if p.exists()]
    if stage == "operations":
        files += list(run.glob("*.log"))
    inventory = {}
    with zipfile.ZipFile(pending, "x", compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(set(files)):
            ordinary(path)
            name = path.relative_to(run).as_posix()
            # NPZ already compresses RGB; avoid a second expensive compression.
            out.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED)
            inventory[name] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
        out.writestr("BACKUP.json", json.dumps({
            "schema": "wm_audit.backup.v1", "files": inventory, "completed_roots": {},
            "scope": "Tracked stage, source snapshots and operational records; status determines success",
            "stage": stage,
        }, indent=2))
    with zipfile.ZipFile(pending) as check:
        if set(check.namelist()) != set(inventory) | {"BACKUP.json"}:
            raise ValueError("Archive inventory mismatch")
        for name, receipt in inventory.items():
            with check.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["sha256"]:
                    raise ValueError("Archive member checksum mismatch")
    digest = sha(pending)
    pending.rename(archive)
    with Path(str(archive) + ".sha256").open("x") as stream:
        stream.write(digest + "  " + archive.name + "\n")
    return {"path": str(archive), "sha256": digest, "bytes": archive.stat().st_size,
            "inventory_entries": len(inventory)}


def supervise(command, project, log_path, changed):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               WM_AUDIT_PROGRESS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    selector = selectors.DefaultSelector()
    pending = b""
    try:
        changed("started", pid=process.pid)
        selector.register(process.stdout, selectors.EVENT_READ)
        with log_path.open("xb") as log:
            while selector.get_map():
                events = selector.select(timeout=15)
                if not events:
                    changed("heartbeat", pid=process.pid, active=process.poll() is None)
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(chunk)
                    log.flush()
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        changed("progress", line=line.decode("utf-8", errors="replace")[-2500:])
            if pending:
                changed("progress", line=pending.decode("utf-8", errors="replace")[-2500:])
            os.fsync(log.fileno())
        return process.wait()
    finally:
        # A status/logging exception must not leave a detached GPU worker writing
        # into the stage while a failure archive is sealed.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                remaining, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                remaining, _ = process.communicate()
            if remaining:
                with log_path.open("ab") as log:
                    log.write(remaining)
                    log.flush()
                    os.fsync(log.fileno())
        process.wait()
        selector.close()
        process.stdout.close()


def execute(project, original_data, output, expected_commit):
    if sys.platform != "linux":
        raise RuntimeError("Hosted Linux only; no local research execution")
    project, original_data = project.resolve(strict=True), original_data.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise FileExistsError("Preserve existing runs; no overwrite or implicit resume")
    commit = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != expected_commit or dirty.strip():
        raise ValueError("Require the exact published commit and unchanged tracked source")
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if occupied.strip():
        raise RuntimeError("GPU has active compute processes; isolate numerical audit rather than changing their settings")
    # uv virtualenv executables legitimately link to the managed interpreter.
    python = project / ".venv/bin/python"
    ordinary(project / ".venv/pyvenv.cfg")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Require the configured project virtualenv executable")
    output.mkdir(parents=True, exist_ok=False)
    snapshots = [p for folder in ("src/wm_audit", "scripts", "configs", "tests")
                 for p in (project / folder).glob("*") if p.suffix in (".py", ".json", ".sh")]
    snapshots += [project / name for name in (
        "NUMERICAL_AUDIT_PROTOCOL_20261102.md", "REPLICATION_PROTOCOL_20261102.md",
        "EXPERIMENT_TRACKER.md", "pyproject.toml")]
    source_hashes = {}
    for path in snapshots:
        ordinary(path)
        target = output / "sources" / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        source_hashes[str(path)] = sha(path)
        if sha(target) != source_hashes[str(path)]:
            raise ValueError("Source snapshot differs")
    stages = commands(project, python, original_data, output)
    launch = {"schema": "wm_audit.tracked_studies.launch.v1", "source_commit": commit,
              "utc": utc(), "original_data": str(original_data), "output": str(output),
              "source_sha256": source_hashes, "stages": [{"name": n, "command": c} for n, c in stages],
              "scheduling": "serial isolated numerical studies, then fixed joint replication",
              "completed_discovery_rerun": False, "implicit_resume": False}
    atomic_json(output / "launch.json", launch)
    state = {"schema": "wm_audit.tracked_studies.status.v1", "source_commit": commit,
             "started_utc": utc(), "updated_utc": utc(), "completed": False, "state": "running",
             "stages": {n: {"state": "not-started", "command": c} for n, c in stages}}
    start = monotonic()

    def update(name, event, **values):
        record = {"utc": utc(), "stage": name, "event": event, **values}
        item = state["stages"][name]
        item.update(values)
        if event == "started":
            item.update(state="running", started_utc=record["utc"])
        state.update(updated_utc=record["utc"], active_stage=name, seconds=monotonic() - start)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        atomic_json(output / "status.json", state)

    atomic_json(output / "status.json", state)
    for name, command in stages:
        try:
            for path, digest in source_hashes.items():
                if sha(ordinary(path)) != digest:
                    raise ValueError("Published source changed before stage")
            code = supervise(command, project, output / (name + ".log"),
                             lambda event, **values: update(name, event, **values))
            update(name, "process_exit", exit_code=code)
            if code:
                raise RuntimeError(f"Stage exited {code}; preserve attempt, do not replace roots")
            verified = verify_stage(name, output / name)
            state["stages"][name].update(state="completed", verified=verified, finished_utc=utc())
            update(name, "verified")
            archive = archive_stage(output, name)
            state["stages"][name]["archive"] = archive
            update(name, "archived", archive=archive)
            print("STAGE_VERIFIED_AND_ARCHIVED", name, archive["sha256"], flush=True)
        except Exception as error:
            execution_passed = state["stages"][name].get("verified") is not None
            state["stages"][name].update(state="preservation-failed" if execution_passed else "failed",
                                         error=str(error), finished_utc=utc())
            state.update(state="failed", completed=False)
            update(name, "failed", error=str(error))
            # Preserve incomplete packaging distinctly; never call it verified.
            pending = output / (name + ".zip.pending")
            if pending.exists():
                pending.rename(output / (name + ".zip.failed-packaging"))
            if not (output / (name + ".zip")).exists():
                try:
                    state["stages"][name]["archive"] = archive_stage(output, name)
                    update(name, "failure_archived")
                except Exception as backup_error:
                    update(name, "failure_archive_error", backup_error=str(backup_error))
            # A separate terminal operations archive captures the real final
            # failure state, not an earlier mutable status snapshot.
            archive_stage(output, "operations")
            raise
    state.update(completed=True, state="completed", finished_utc=utc(), active_stage=None)
    atomic_json(output / "status.json", state)
    atomic_json(output / "complete.json", {"schema": "wm_audit.tracked_studies.complete.v1",
                                          "status_sha256": sha(output / "status.json"),
                                          "source_commit": commit, "utc": utc(),
                                          "stages": state["stages"]})
    operations = archive_stage(output, "operations")
    atomic_json(output / "operations-receipt.json", operations)
    print("ALL_PENDING_STUDIES_COMPLETE", str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--original-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    execute(args.project, args.original_data, args.output, args.expected_commit)


if __name__ == "__main__":
    main()