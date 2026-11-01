"""Verified full ZIP snapshots through an already authenticated fsspec filesystem.

Run inside the notebook kernel; standard library only, no torch or credentials.
Pass the PROJECT root, not its runs directory. Atomically published root NPZs
are captured even before the generation manifest exists. This preserves partial
generation, but does NOT add automatic experiment resume. Each upload is a full,
independent snapshot; no remote files are deleted. Keep the kernel alive.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SPLITS = ("editor", "scorer", "validation", "pilot", "confirmation")
ROOT_NAME = re.compile(r"(editor|scorer|validation|pilot|confirmation)-\d{6,}-[0-9a-f]{32}\.npz")
MODEL_JSON = ("training.json", "validation.json", "pilot.json", "confirmation.json",
              "report-validation.json", "report-pilot.json", "report-confirmation.json",
              "inference.json", "protocol-lock.json")


def stream_digest(handle):
    result = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


def _regular(root, path):
    path.relative_to(root)
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError("Linked backup input refused")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Nonregular backup input refused")


def _stamp(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _json_bytes(root, path):
    _regular(root, path)
    before = path.stat()
    content = path.read_bytes()
    if _stamp(before) != _stamp(path.stat()):
        raise ValueError("JSON changed during capture")
    json.loads(content)  # Reject an in-flight, partially written report/manifest.
    return content


def capture(root, destination):
    """Snapshot allowlisted artifacts; stream large roots without loading arrays.

    Invalid/changing JSON and live logs are skipped and recorded. Published NPZ
    corruption, changing immutable files, and manifest checksum mismatches fail
    the whole snapshot. Model weights are copied only for completed fits marked
    by training.json. train-cache memmaps and incomplete fits are excluded.
    """
    root = Path(root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir() or not (root / "runs").is_dir():
        raise ValueError("Pass an existing project root containing runs/")
    if (root / "runs").is_symlink():
        raise ValueError("Linked runs directory refused")
    files, json_files, manifests, skipped = set(), {}, [], []
    diagnostic_references = {}
    counts = {}

    def add_json(path):
        if not path.exists():
            return False
        _regular(root, path)
        try:
            content = _json_bytes(root, path)
        except (ValueError, FileNotFoundError):
            skipped.append(str(path.relative_to(root)))
            return False
        json_files[path] = content
        return True

    for run in sorted((root / "runs").iterdir()):
        if run.is_symlink():
            raise ValueError("Linked run refused")
        if not run.is_dir() or run.name.startswith("."):
            continue
        counts[run.name] = {split: 0 for split in SPLITS}
        for name in ("manifest-development.json", "manifest-confirmation.json"):
            path = run / "data" / name
            if add_json(path):
                manifests.append((run / "data", json.loads(json_files[path])))
        for split in SPLITS:
            folder = run / "data" / split
            if folder.is_symlink() or (run / "data").is_symlink():
                raise ValueError("Linked data directory refused")
            for path in sorted(folder.glob("*.npz")):
                if ROOT_NAME.fullmatch(path.name) and path.name.startswith(split + "-"):
                    _regular(root, path)
                    files.add(path)
                    counts[run.name][split] += 1
        for name in ("preflight.json", "replay.json", "diagnostics-plan.json"):
            add_json(run / name)
        for model in ("lewm", "dinowm"):
            directory = run / model
            if directory.is_symlink():
                raise ValueError("Linked model run refused")
            complete = add_json(directory / "training.json")
            if complete:
                training = json.loads(json_files[directory / "training.json"])
                names = training["config"]["editors"]
                if not isinstance(names, list) or not set(names) <= {"inverse", "ridge", "mlp", "rollout"}:
                    raise ValueError("Invalid fitted editor roster")
                for path in [directory / "scorer.pt", *[directory / "editors" / (n + ".pt") for n in names]]:
                    _regular(root, path)
                    files.add(path)
            for name in MODEL_JSON:
                if name != "training.json":
                    add_json(directory / name)
            for kind in ("readout", "prediction-paired", "prediction-recorded"):
                diagnostic = directory / kind
                if diagnostic.is_symlink():
                    raise ValueError("Linked diagnostic directory refused")
                report_path = diagnostic / "report.json"
                if not add_json(report_path):
                    continue  # Incomplete diagnostics are deliberately not published.
                report = json.loads(json_files[report_path])
                if report.get("schema") != "wm_audit.diagnostic.v1" or not isinstance(report.get("artifacts"), dict):
                    raise ValueError("Invalid diagnostic completion report")
                for name, expected in report["artifacts"].items():
                    if (not isinstance(name, str) or Path(name).name != name or "\\" in name
                            or name.startswith(".") or name == "report.json"
                            or Path(name).suffix not in (".json", ".pt", ".npz", ".png", ".html")):
                        raise ValueError("Invalid diagnostic artifact path")
                    path = diagnostic / name
                    _regular(root, path)
                    with path.open("rb") as stream:
                        if stream_digest(stream) != expected:
                            raise ValueError("Diagnostic artifact checksum mismatch")
                    diagnostic_references[path.relative_to(root).as_posix()] = expected
                    files.add(path)
        # Logs are optional, but never silently claim a live log is consistent.
        log = run / "console.log"
        if log.exists():
            _regular(root, log)
            skipped.append(str(log.relative_to(root)))

    # Only explicit project recovery material; never crawl arbitrary files.
    for pattern in ("configs/*.json", "src/wm_audit/*.py", "scripts/*.py", "scripts/*.sh",
                    "pyproject.toml", "README.md", "RUNNING.md", "BACKUP.md", "DIAGNOSTICS.md", ".gitignore"):
        for path in root.glob(pattern):
            _regular(root, path)
            files.add(path)
    if not any(sum(value.values()) for value in counts.values()) and not json_files:
        raise ValueError("No published experiment artifacts found yet")

    inventory = {}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path, content in sorted(json_files.items()):
            name = path.relative_to(root).as_posix()
            archive.writestr(name, content)
            inventory[name] = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
        for path in sorted(files):
            _regular(root, path)
            before = path.stat()
            if path.suffix == ".npz":
                with zipfile.ZipFile(path) as npz:
                    if npz.testzip() is not None:
                        raise ValueError("Published root failed ZIP CRC validation")
            name = path.relative_to(root).as_posix()
            digest, size = hashlib.sha256(), 0
            info = zipfile.ZipInfo(name)
            # NPZ is already compressed; avoid recompressing large roots.
            info.compress_type = zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                if _stamp(before) != _stamp(os.fstat(source.fileno())):
                    raise ValueError("Artifact replaced during capture")
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if _stamp(before) != _stamp(path.stat()) or size != before.st_size:
                raise ValueError("Artifact changed during capture")
            inventory[name] = {"sha256": digest.hexdigest(), "size_bytes": size}
        for name, expected in diagnostic_references.items():
            if inventory[name]["sha256"] != expected:
                raise ValueError("Diagnostic artifact changed between verification and capture")
        for data_dir, manifest in manifests:
            for split, references in manifest["files"].items():
                if split not in SPLITS:
                    raise ValueError("Invalid manifest split")
                for relative, expected in references.items():
                    parts = Path(relative).parts
                    if len(parts) != 2 or parts[0] != split or not ROOT_NAME.fullmatch(parts[1]):
                        raise ValueError("Invalid manifest root path")
                    name = (data_dir / relative).relative_to(root).as_posix()
                    if name not in inventory or inventory[name]["sha256"] != expected:
                        raise ValueError("Manifest root missing or checksum mismatch")
        archive.writestr("BACKUP.json", json.dumps({
            "schema": "wm_audit.backup.v1", "captured_at": datetime.now(timezone.utc).isoformat(),
            "completed_roots": counts, "files": inventory, "skipped": skipped,
            "scope": "Full snapshot; includes partial-generation published roots and completed fit artifacts. "
                     "Excludes live logs, incomplete fits, train-cache, model downloads, environments and secrets.",
            "resume": "Preserves artifacts only; current experiment runner does not automatically resume.",
        }, indent=2, allow_nan=False))
    return counts


class DriveBackup:
    """Single periodic worker using the notebook's authenticated filesystem.

    Full snapshots need temporary local disk and accumulate on Drive. A receipt
    is published ONLY after full remote readback matches SHA256. No deletion.
    status exposes error types, never raw HTTP errors which might contain tokens.
    """
    def __init__(self, fs, root, remote_dir, interval=600, temp_dir=None):
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 60:
            raise ValueError("Use a finite interval of at least 60 seconds")
        if not isinstance(remote_dir, str) or not remote_dir.strip("/"):
            raise ValueError("Use a dedicated remote backup folder")
        self.fs, self.root, self.remote_dir = fs, Path(root), remote_dir.rstrip("/")
        self.interval, self.temp_dir = interval, temp_dir
        self.stop_event, self._lock = threading.Event(), threading.Lock()
        self.thread = None
        self.status = {"state": "not_started"}

    def start(self):
        if self.thread is not None or self.stop_event.is_set():
            raise RuntimeError("This worker has already been started or stopped")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        """Stop after the current snapshot/upload; does not terminate experiments."""
        self.stop_event.set()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.once()
            except Exception:
                pass  # once() records a sanitized error; retry next interval.
            if self.stop_event.wait(self.interval):
                break

    def once(self):
        """Capture immediately, upload, read back, then acknowledge via receipt."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A backup is already in progress")
        try:
            self.status = {**self.status, "state": "capturing"}
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12] + ".zip"
            remote = self.remote_dir + "/" + name
            with tempfile.TemporaryDirectory(prefix="world-model-backup-", dir=self.temp_dir) as temporary:
                local = Path(temporary) / name
                counts = capture(self.root, local)
                with local.open("rb") as handle:
                    expected = stream_digest(handle)
                size = local.stat().st_size
                self.status = {**self.status, "state": "uploading", "pending_archive": remote, "size_bytes": size}
                self.fs.put_file(str(local), remote)
                self.status = {**self.status, "state": "verifying"}
                with self.fs.open(remote, "rb") as handle:
                    if stream_digest(handle) != expected:
                        raise ValueError("Remote archive checksum mismatch")
                with self.fs.open(remote + ".sha256", "wb") as handle:
                    handle.write(f"{expected}  {name}\n".encode())
            self.status = {"state": "verified", "archive": remote, "sha256": expected,
                           "verified_at": datetime.now(timezone.utc).isoformat(),
                           "completed_roots": counts, "size_bytes": size}
            return dict(self.status)
        except Exception as error:
            self.status = {**self.status, "state": "error", "error_type": type(error).__name__}
            raise RuntimeError(f"Backup failed ({type(error).__name__}); inspect status") from None
        finally:
            self._lock.release()