"""Hash-verified recovery; fixtures contain no real experiment data."""
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import pytest

restore = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/inspect_world_backup.py"))["restore"]


def bundle(tmp_path, name="runs/example/a.json", content=b"{}", digest=None):
    path = tmp_path / "backup.zip"
    manifest = {"schema": "wm_audit.backup.v1", "completed_roots": {}, "files": {
        name: {"sha256": digest or hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, content)
        z.writestr("BACKUP.json", json.dumps(manifest))
    Path(str(path) + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_restore_verified_and_exclusive(tmp_path):
    path = bundle(tmp_path)
    restore(path, tmp_path / "out")
    assert (tmp_path / "out/runs/example/a.json").read_bytes() == b"{}"
    assert (tmp_path / "out/RESTORE-VERIFIED.json").exists()
    with pytest.raises(FileExistsError):
        restore(path, tmp_path / "out")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "runs/../../escape", "runs\\escape"])
def test_restore_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        restore(bundle(tmp_path, name=name), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_restore_checks_internal_hash(tmp_path):
    with pytest.raises(ValueError, match="hash/size"):
        restore(bundle(tmp_path, digest="0" * 64), tmp_path / "out")
    assert not (tmp_path / "out/RESTORE-VERIFIED.json").exists()


def test_restore_checks_receipt(tmp_path):
    path = bundle(tmp_path)
    Path(str(path) + ".sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="receipt"):
        restore(path, tmp_path / "out")


def test_continuation_never_generates_or_refits_lewm(tmp_path):
    project = Path(__file__).resolve().parents[1]
    plan = runpy.run_path(str(project / "scripts/finish_recovered_pilot.py"))["plan"]
    config = json.loads((project / "configs/smoke.json").read_text())
    config["counts"] = {key: 2 if key != "confirmation" else 0 for key in config["counts"]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    source = tmp_path / "source"
    (source / "lewm").mkdir(parents=True)
    files = {}
    for split in ("editor", "scorer", "validation", "pilot"):
        folder = source / "data" / split
        folder.mkdir(parents=True)
        files[split] = {}
        for i in range(2):
            path = folder / f"{i}.npz"
            path.write_bytes(b"hash-only fixture")
            files[split][f"{split}/{i}.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "data/manifest-development.json").write_text(json.dumps({"config": config, "files": files}))
    records = [{"root_id": str(i), "appearance": "original", "split": "validation", "method": method,
                "history_fidelity": 0, "specificity_loss": 0} for i in range(2) for method in config["editors"]]
    for name, value in (("training", {"config": config}), ("validation", {"config": config, "records": records}),
                        ("pilot", {"config": config, "primary_method": "inverse"})):
        (source / "lewm" / (name + ".json")).write_text(json.dumps(value))
    commands = plan(source, config_path, tmp_path / "new", tmp_path / "cache", "cuda")
    assert len(commands) == 6
    assert all("generate" not in command for command in commands)
    fits = [command for command in commands if "fit" in command]
    assert len(fits) == 1 and fits[0][fits[0].index("--model") + 1] == "dinowm"
    (source / "data/editor/0.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="editor"):
        plan(source, config_path, tmp_path / "new", tmp_path / "cache", "cuda")