"""Backup tests use local files and a fake filesystem, never Drive credentials."""
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import pytest

helper = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/backup_world_models.py"))
capture, DriveBackup = helper["capture"], helper["DriveBackup"]


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    folder = root / "runs/pilot-01/data/editor"
    folder.mkdir(parents=True)
    published = folder / ("editor-000000-" + "a" * 32 + ".npz")
    with zipfile.ZipFile(published, "w") as archive:
        archive.writestr("test.npy", b"test fixture, not real experiment data")
    (folder / ".paired-inflight.npz").write_bytes(b"incomplete")
    (root / "runs/pilot-01/console.log").write_text("live log")
    (root / ".env").write_text("NEVER COPY")
    (root / "cache").mkdir()
    (root / "cache/weights.pt").write_bytes(b"exclude")
    (root / "configs").mkdir()
    (root / "configs/pilot.json").write_text("{}")
    return root, published


def test_partial_generation_without_manifest_preserved(project, tmp_path):
    root, published = project
    output = tmp_path / "backup.zip"
    counts = capture(root, output)
    assert counts["pilot-01"]["editor"] == 1
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert published.relative_to(root).as_posix() in names
        assert not any("inflight" in p or "cache" in p or ".env" in p or "console.log" in p for p in names)
        metadata = json.loads(archive.read("BACKUP.json"))
        for name, info in metadata["files"].items():
            content = archive.read(name)
            assert info == {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


@pytest.mark.parametrize("corrupt", [False, True])
def test_manifest_checksums(project, tmp_path, corrupt):
    root, published = project
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    manifest = {"files": {"editor": {"editor/" + published.name: "0" * 64 if corrupt else digest}}}
    (published.parent.parent / "manifest-development.json").write_text(json.dumps(manifest))
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            capture(root, tmp_path / "backup.zip")
    else:
        capture(root, tmp_path / "backup.zip")


def test_partial_json_and_incomplete_fit_excluded(project, tmp_path):
    root, _ = project
    model = root / "runs/pilot-01/lewm"
    model.mkdir()
    (model / "training.json").write_text('{"config":')
    (model / "scorer.pt").write_bytes(b"partial fit")
    capture(root, tmp_path / "backup.zip")
    with zipfile.ZipFile(tmp_path / "backup.zip") as archive:
        assert not any(name.endswith(".pt") for name in archive.namelist())
        assert "runs/pilot-01/lewm/training.json" in json.loads(archive.read("BACKUP.json"))["skipped"]


def test_complete_fit_included_without_memmaps(project, tmp_path):
    root, _ = project
    model = root / "runs/pilot-01/lewm"
    (model / "editors").mkdir(parents=True)
    (model / "training.json").write_text(json.dumps({"config": {"editors": ["ridge"]}}))
    (model / "scorer.pt").write_bytes(b"scorer fixture")
    (model / "editors/ridge.pt").write_bytes(b"editor fixture")
    (model / "train-cache").mkdir()
    (model / "train-cache/factual.npy").write_bytes(b"exclude")
    capture(root, tmp_path / "backup.zip")
    with zipfile.ZipFile(tmp_path / "backup.zip") as archive:
        assert "runs/pilot-01/lewm/editors/ridge.pt" in archive.namelist()
        assert not any("train-cache" in p for p in archive.namelist())


def test_symlink_refused(project, tmp_path):
    root, published = project
    published.unlink()
    published.symlink_to(root / ".env")
    with pytest.raises(ValueError, match="Linked"):
        capture(root, tmp_path / "backup.zip")


class LocalFS:
    def __init__(self, base, corrupt=False):
        self.base, self.corrupt = base, corrupt

    def put_file(self, local, remote):
        target = self.base / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"corrupt" if self.corrupt else Path(local).read_bytes())

    def open(self, path, mode):
        return (self.base / path).open(mode)


def test_verified_upload_and_receipt(project, tmp_path):
    worker = DriveBackup(LocalFS(tmp_path / "remote"), project[0], "world-model-backups")
    result = worker.once()
    assert result["state"] == "verified"
    remote = tmp_path / "remote" / result["archive"]
    assert hashlib.sha256(remote.read_bytes()).hexdigest() == result["sha256"]
    assert Path(str(remote) + ".sha256").read_text().startswith(result["sha256"])


def test_failed_readback_has_no_receipt(project, tmp_path):
    worker = DriveBackup(LocalFS(tmp_path / "remote", corrupt=True), project[0], "world-model-backups")
    with pytest.raises(RuntimeError, match="ValueError"):
        worker.once()
    assert worker.status["state"] == "error"
    assert not list((tmp_path / "remote").rglob("*.sha256"))


def test_double_start_refused(project, tmp_path):
    worker = DriveBackup(LocalFS(tmp_path / "remote"), project[0], "world-model-backups").start()
    try:
        with pytest.raises(RuntimeError):
            worker.start()
    finally:
        worker.stop()
        worker.thread.join(timeout=10)
    assert not worker.thread.is_alive()


@pytest.mark.parametrize("interval", [True, 0, 59, float("nan"), float("inf")])
def test_invalid_interval(project, interval):
    with pytest.raises(ValueError):
        DriveBackup(None, project[0], "world-model-backups", interval=interval)