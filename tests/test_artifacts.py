"""Local-only artifact/bootstrap/H5 tests; no real release or dataset downloads.

Synthetic Git repositories and tiny compressed H5 files test transport/schema
contracts, NOT pretrained competence or the identity of the 13 GB release.
Bootstrap command tests use explicitly labeled shell stubs, not fake GPU results.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import socket
import subprocess
import sys
import tarfile
import tomllib
import urllib.request

import numpy as np
import pytest

from wm_audit import artifacts
from wm_audit.offline_data import H5Windows, compute_action_stats, dataset_info


ROOT = Path(__file__).resolve().parents[1]


def _forbidden(*args, **kwargs):
    raise AssertionError("This local-only test must not use the network or load a checkpoint.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_h5(path: Path, lengths=(16, 18), *, constant=False) -> Path:
    import h5py
    import hdf5plugin

    rows = sum(lengths)
    offsets = np.cumsum((0, *lengths[:-1]), dtype=np.int64)
    actions = np.stack((np.linspace(-0.9, 0.8, rows), np.linspace(0.7, -0.6, rows)), axis=1).astype(np.float32)
    if constant:
        actions[:] = (0, 0.25)
    actions[offsets + lengths - 1] = np.nan
    with h5py.File(path, "w") as handle:
        pixels = handle.create_dataset("pixels", (rows, 224, 224, 3), dtype=np.uint8,
                                       chunks=(1, 224, 224, 3), **hdf5plugin.Blosc(cname="lz4", clevel=1))
        for row in range(rows):
            pixels[row] = np.full((224, 224, 3), row % 256, dtype=np.uint8)
        handle["action"] = actions
        handle["ep_offset"] = offsets
        handle["ep_len"] = np.asarray(lengths, dtype=np.int64)
        handle["state"] = np.arange(rows * 7, dtype=np.float64).reshape(rows, 7)
    return path


@pytest.fixture
def h5_path(tmp_path):
    return _write_h5(tmp_path / "original-scene-fixture.h5")


@pytest.mark.parametrize("chunk_rows", [1, 5, 17, 65536])
def test_population_stats_skip_only_validated_terminal_rows(h5_path, chunk_rows):
    import h5py

    info = dataset_info(h5_path, chunk_rows=chunk_rows)
    with h5py.File(h5_path, "r") as handle:
        raw = handle["action"][:].astype(np.float64)
    finite = raw[np.isfinite(raw).all(axis=1)]
    stats = info["action_stats"]
    assert (info["rows"], info["episodes"]) == (34, 2)
    assert info["episode_offsets"] == [0, 16] and info["episode_lengths"] == [16, 18]
    assert stats["count"] == 32 and stats["terminal_rows_excluded"] == 2 and stats["ddof"] == 0
    np.testing.assert_allclose(stats["action_mean"], finite.mean(0), atol=1e-14, rtol=0)
    np.testing.assert_allclose(stats["action_std"], finite.std(0, ddof=0), atol=1e-14, rtol=0)
    assert not np.allclose(stats["action_std"], finite.std(0, ddof=1))
    json.dumps(info, allow_nan=False)


def test_constant_action_dimensions_use_native_unit_scale(tmp_path):
    path = _write_h5(tmp_path / "constant.h5", constant=True)
    stats = compute_action_stats(path, chunk_rows=3)
    assert stats["action_population_std"] == [0.0, 0.0]
    assert stats["action_std"] == [1.0, 1.0]
    with H5Windows(path, horizon=1, normalize_actions=True) as windows:
        np.testing.assert_array_equal(windows[0]["normalized_actions"], np.zeros((3, 10)))


@pytest.mark.parametrize("row,value,match", [
    (15, [0.0, 0.0], "terminal"), (33, [np.nan, 0.0], "terminal"),
    (15, [np.inf, np.nan], "terminal"), (7, [np.nan, np.nan], "Nonterminal"),
    (7, [0.0, np.nan], "Nonterminal"), (16, [np.inf, 0.0], "Nonterminal"),
    (4, [1.01, 0.0], "native"),
])
def test_bad_actions_fail_instead_of_truncating_or_filling(h5_path, row, value, match):
    import h5py

    with h5py.File(h5_path, "r+") as handle:
        handle["action"][row] = value
    with pytest.raises(ValueError, match=match):
        dataset_info(h5_path, chunk_rows=4)


@pytest.mark.parametrize("column,values", [
    ("ep_offset", [1, 16]), ("ep_offset", [0, 15]), ("ep_offset", [0, 17]),
    ("ep_len", [16, 17]), ("ep_len", [16, 19]), ("ep_len", [0, 34]),
    ("ep_offset", [0.0, 16.0]), ("ep_len", [True, True]),
])
def test_malformed_episode_boundaries_are_rejected(h5_path, column, values):
    import h5py

    with h5py.File(h5_path, "r+") as handle:
        del handle[column]
        handle[column] = np.asarray(values)
    with pytest.raises(ValueError):
        H5Windows(h5_path, horizon=1)


def test_h5_external_links_and_wrong_pixels_are_rejected(h5_path, tmp_path):
    import h5py

    with h5py.File(h5_path, "r+") as handle:
        del handle["pixels"]
        handle["pixels"] = h5py.ExternalLink(str(tmp_path / "must-not-open.h5"), "/pixels")
    with pytest.raises(ValueError, match="link"):
        dataset_info(h5_path)
    with h5py.File(h5_path, "r+") as handle:
        del handle["pixels"]
        handle.create_dataset("pixels", (34, 196, 196, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="224"):
        dataset_info(h5_path)


def test_terminal_only_file_has_no_population_statistics(tmp_path):
    path = _write_h5(tmp_path / "terminals.h5", lengths=(1, 1))
    with pytest.raises(ValueError, match="no finite"):
        dataset_info(path)


def test_window_action_alignment_and_terminal_frame_are_exact(h5_path):
    import h5py

    with h5py.File(h5_path, "r") as handle:
        actions, states = handle["action"][:], handle["state"][:]
    with H5Windows(h5_path, horizon=1) as windows:
        assert len(windows) == 4
        assert [(w["episode"], w["start"]) for w in windows] == [(0, 0), (1, 0), (1, 1), (1, 2)]
        for window in windows:
            rows = window["row_indices"]
            assert window["pixels"].shape == (4, 224, 224, 3)
            assert window["pixels"].dtype == np.uint8 and window["pixels"].flags.c_contiguous
            assert window["context"].shape[0] == 3 and window["future"].shape[0] == 1
            assert window["actions"].shape == (3, 10) and window["actions"].dtype == np.float32
            assert "normalized_actions" not in window
            np.testing.assert_array_equal(window["pixels"][:, 0, 0, 0], rows)
            np.testing.assert_array_equal(window["state"], states[rows])
            np.testing.assert_array_equal(window["actions"], actions[rows[0]:rows[-1]].reshape(3, 10))
            assert np.isfinite(window["actions"]).all()
        assert windows[0]["row_indices"][-1] == 15   # terminal observation, no outgoing action
        assert windows[-1]["row_indices"][-1] == 33
        with pytest.raises(IndexError, match="boundary"):
            windows.window(0, 1)
        with pytest.raises(IndexError):
            windows[len(windows)]


@pytest.mark.parametrize("horizon", [1, 2, 8])
def test_multiple_horizons_stride_and_episode_selection(tmp_path, horizon):
    path = _write_h5(tmp_path / "long.h5", lengths=(16, 56))
    with H5Windows(path, horizon=horizon, episodes=[1], stride=3) as windows:
        span = (2 + horizon) * 5
        assert len(windows) == (55 - span) // 3 + 1
        for window in windows:
            assert window["episode"] == 1 and window["start"] % 3 == 0
            assert window["actions"].shape == (2 + horizon, 10)
            assert window["future"].shape[0] == horizon
            assert window["row_indices"][-1] < 72
    with H5Windows(path, horizon=8, episodes=[0]) as short:
        assert len(short) == 0
        with pytest.raises(IndexError):
            short[0]


def test_normalization_is_separate_and_never_uses_selected_episode_stats(h5_path):
    full_stats = compute_action_stats(h5_path)
    with H5Windows(h5_path, horizon=1, episodes=[1], normalize_actions=True) as windows:
        assert windows.stats == full_stats
        sample = windows[0]
        expected = ((sample["actions"].reshape(-1, 2) - np.array(full_stats["action_mean"], np.float32))
                    / np.array(full_stats["action_std"], np.float32)).reshape(3, 10)
        np.testing.assert_array_equal(sample["normalized_actions"], expected)
        assert np.max(np.abs(sample["actions"])) <= 1
    released = {"action_key": "action", "action_mean": [0.1, -0.2], "action_std": [0.5, 0.25]}
    with H5Windows(h5_path, 1, stats=released, normalize_actions=True) as windows:
        sample = windows[0]
        expected = ((sample["actions"].reshape(-1, 2) - np.array([0.1, -0.2], np.float32))
                    / np.array([0.5, 0.25], np.float32)).reshape(3, 10)
        np.testing.assert_array_equal(sample["normalized_actions"], expected)


@pytest.mark.parametrize("options", [
    {"horizon": 0}, {"horizon": True}, {"stride": 0}, {"frameskip": 1}, {"context": 2},
    {"episodes": [True, 1]}, {"episodes": [0.0]}, {"episodes": [0, 0]},
    {"episodes": [-1]}, {"episodes": [2]}, {"normalize_actions": 1},
    {"stats": {"action_key": "action", "action_mean": [0, 0], "action_std": [0, 1]}},
])
def test_invalid_window_arguments(h5_path, options):
    with pytest.raises(ValueError):
        H5Windows(h5_path, **options)


def test_episode_access_empty_selection_and_worker_safe_handles(h5_path, monkeypatch):
    with H5Windows(h5_path, horizon=1, normalize_actions=True) as windows:
        episode = windows.episode(0)
        assert episode["pixels"].shape == (16, 224, 224, 3)
        assert episode["actions"].shape == (15, 2) and np.isfinite(episode["actions"]).all()
        old_handle = windows._h5
        child_pid = os.getpid() + 1000
        monkeypatch.setattr("wm_audit.offline_data.os.getpid", lambda: child_pid)
        windows[0]
        assert windows._pid == child_pid and not old_handle.id.valid
        # This is a local dataset object's pickle, not a model/checkpoint pickle.
        with pickle.loads(pickle.dumps(windows)) as restored:
            assert restored._h5 is None
            np.testing.assert_array_equal(restored[-1]["actions"], windows[-1]["actions"])
        with pytest.raises(TypeError):
            windows[True]
    assert windows._h5 is None
    with H5Windows(h5_path, 1, episodes=[]) as empty:
        assert len(empty) == 0
        with pytest.raises(IndexError):
            empty.episode(0)


def test_changed_h5_invalidates_existing_index(h5_path):
    with H5Windows(h5_path, horizon=1) as windows:
        value = h5_path.stat()
        os.utime(h5_path, ns=(value.st_atime_ns, value.st_mtime_ns + 1_000_000))
        with pytest.raises(RuntimeError, match="changed after indexing"):
            windows[0]


def test_modules_do_not_eagerly_import_h5_or_torch(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    code = ("import sys; import wm_audit.artifacts, wm_audit.offline_data; "
            "assert not ({'h5py', 'hdf5plugin', 'torch'} & sys.modules.keys())")
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, check=True, capture_output=True)


def _tar_zst(path, entries):
    import zstandard as zstd

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for name, kind, payload in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.mode = 0o6777  # extractor must not restore privileged modes
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../../outside"
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
    path.write_bytes(zstd.ZstdCompressor(write_checksum=True).compress(buffer.getvalue()))
    return path


def test_streamed_tar_extraction_hashes_and_does_not_restore_modes(tmp_path):
    archive = _tar_zst(tmp_path / "fixture.tar.zst", [
        ("./", tarfile.DIRTYPE, b""), ("nested", tarfile.DIRTYPE, b""),
        ("nested/data.h5", tarfile.REGTYPE, b"fixture"),
    ])
    target = tmp_path / "output"
    record = artifacts.safe_extract_tar_zst(archive, target, max_bytes=20)
    assert record == {"size_bytes": 7, "files": {"nested/data.h5": {"size_bytes": 7, "sha256": _sha(b"fixture")}}}
    assert (target / "nested/data.h5").read_bytes() == b"fixture"
    assert not ((target / "nested/data.h5").stat().st_mode & 0o6000)
    with pytest.raises(FileExistsError):
        artifacts.safe_extract_tar_zst(archive, target)
    assert (target / "nested/data.h5").read_bytes() == b"fixture"


@pytest.mark.parametrize("name,kind", [
    ("../escape", tarfile.REGTYPE), ("ok/../../escape", tarfile.REGTYPE),
    ("/absolute", tarfile.REGTYPE), ("C:/escape", tarfile.REGTYPE), ("C:\\escape", tarfile.REGTYPE),
    ("link", tarfile.SYMTYPE), ("link", tarfile.LNKTYPE), ("pipe", tarfile.FIFOTYPE),
    ("device", tarfile.CHRTYPE),
])
def test_unsafe_tar_members_leave_no_published_tree(tmp_path, name, kind):
    archive = _tar_zst(tmp_path / "bad.tar.zst", [("first", tarfile.REGTYPE, b"safe"), (name, kind, b"unsafe")])
    target = tmp_path / "output"
    with pytest.raises(artifacts.ArtifactError):
        artifacts.safe_extract_tar_zst(archive, target)
    assert not target.exists() and not (tmp_path / "escape").exists()
    assert not list(tmp_path.glob(".extract-*"))


@pytest.mark.parametrize("kind", ["duplicate", "bytes", "members", "checksum"])
def test_tar_failure_limits_and_cleanup(tmp_path, kind):
    entries = [("one", tarfile.REGTYPE, b"abcd"), ("one" if kind == "duplicate" else "two", tarfile.REGTYPE, b"efgh")]
    archive = _tar_zst(tmp_path / "bad.tar.zst", entries)
    options = {"max_bytes": 3} if kind == "bytes" else {"max_members": 1} if kind == "members" else {}
    if kind == "checksum":
        value = bytearray(archive.read_bytes())
        value[-1] ^= 0xFF
        archive.write_bytes(value)
    import zstandard as zstd

    with pytest.raises((artifacts.ArtifactError, zstd.ZstdError)):
        artifacts.safe_extract_tar_zst(archive, tmp_path / "output", **options)
    assert not (tmp_path / "output").exists() and not list(tmp_path.glob(".extract-*"))


def test_raw_h5_zstd_is_not_mistaken_for_tar(h5_path, tmp_path):
    import zstandard as zstd

    data = h5_path.read_bytes()
    archive = tmp_path / "original.h5.zst"
    archive.write_bytes(zstd.ZstdCompressor(write_checksum=True).compress(data))
    destination = tmp_path / "decompressed.h5"
    record = artifacts.extract_h5_zst(archive, destination)
    assert record == {"size_bytes": len(data), "sha256": _sha(data)}
    assert destination.read_bytes() == data and dataset_info(destination)["episodes"] == 2
    with pytest.raises(FileExistsError):
        artifacts.extract_h5_zst(archive, destination)
    with pytest.raises(artifacts.ArtifactError, match="byte limit"):
        artifacts.extract_h5_zst(archive, tmp_path / "too-large.h5", max_bytes=10)
    archive.write_bytes(zstd.ZstdCompressor().compress(b"not HDF5"))
    with pytest.raises(artifacts.ArtifactError, match="raw HDF5"):
        artifacts.extract_h5_zst(archive, tmp_path / "bad.h5")
    assert not list(tmp_path.glob(".h5-*")) and not (tmp_path / "bad.h5").exists()


def test_extractors_refuse_destination_symlinks(tmp_path):
    target = tmp_path / "link"
    target.symlink_to(tmp_path / "absent")
    for extract in (artifacts.extract_h5_zst, artifacts.safe_extract_tar_zst):
        with pytest.raises(FileExistsError):
            extract(tmp_path / "not-even-read.zst", target)
    assert not (tmp_path / "absent").exists()


def _git(root, *args):
    return subprocess.run(["git", "-c", "user.name=Local fixture", "-c", "user.email=fixture@example.invalid",
                           "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={os.devnull}",
                           "-C", str(root), *args], check=True, text=True, capture_output=True).stdout.strip()


def _local_repo(root, files):
    root.mkdir()
    _git(root, "init", "--quiet")
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "synthetic source fixture")
    return _git(root, "rev-parse", "HEAD"), {name: _sha(value) for name, value in files.items()}


@pytest.fixture
def synthetic_releases(tmp_path, monkeypatch):
    if not shutil.which("git"):
        pytest.skip("Local Git fixture tests require git (also required by --download).")
    from wm_audit import models as api, simulator

    lewm_rev, lewm_hashes = _local_repo(tmp_path / "lewm-origin", {"jepa.py": b"raise AssertionError('must not execute')\n"})
    point_files = {"dinowm_swm/conf/env/swm_pusht.yaml": b"original H5 provenance\n",
                   "stable-worldmodel/stable_worldmodel/native.py": b"raise AssertionError('must not execute')\n"}
    point_rev, point_hashes = _local_repo(tmp_path / "point-origin", point_files)
    native = "stable-worldmodel/stable_worldmodel/native.py"
    monkeypatch.setattr(simulator, "POINT_REVISION", point_rev)
    monkeypatch.setattr(simulator, "SOURCE_HASHES", {"native.py": point_hashes[native]})
    monkeypatch.setattr(api, "POINT_SOURCE_REVISION", point_rev)
    monkeypatch.setattr(api, "POINT_SOURCE_HASHES", {name: digest for name, digest in point_hashes.items() if name != native})
    monkeypatch.setattr(api, "LEWM_SOURCE_REVISION", lewm_rev)
    monkeypatch.setattr(api, "LEWM_SOURCE_HASHES", lewm_hashes)
    monkeypatch.setattr(artifacts, "LEWM_SOURCE_URL", (tmp_path / "lewm-origin").as_uri())
    monkeypatch.setattr(artifacts, "POINT_SOURCE_URL", (tmp_path / "point-origin").as_uri())
    payloads = {
        "models/lewm/config.json": b'{"fixture": true}', "models/lewm/weights.pt": b"not a pickle",
        "models/dinowm/dino-wm/hydra.yaml": b"fixture: true\n",
        "models/dinowm/dino-wm/checkpoints/model_latest.pth": b"must never be deserialized",
        api.STATS_PATH: json.dumps({"action_key": "action", "action_mean": [0.1, -0.2], "action_std": [0.5, 0.25]}).encode(),
    }
    monkeypatch.setattr(api, "RELEASE_HASHES", {path: _sha(value) for path, value in payloads.items()})
    responses = {}
    for path, value in payloads.items():
        if path.startswith("models/lewm/"):
            repo, rev, remote = "quentinll/lewm-pusht", api.LEWM_HF_REVISION, path[len("models/lewm/"):]
        else:
            repo, rev, remote = "fafraob/point-cloud-pusht", api.DINOWM_HF_REVISION, path[len("models/dinowm/"):]
        responses[f"https://huggingface.co/{repo}/resolve/{rev}/{remote}"] = value
    # Real authenticated DINOv2 extraction code, but a tiny synthetic source tar.
    dino_source = b"raise AssertionError('backbone must not be constructed')\n"
    dino_relative = "dinov2/__init__.py"
    tree_hash = _sha(dino_relative.encode() + b"\0" + hashlib.sha256(dino_source).digest() + b"\n")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        member = tarfile.TarInfo(f"dinov2-{api.DINOV2_SOURCE_REVISION}/{dino_relative}")
        member.size = len(dino_source)
        bundle.addfile(member, io.BytesIO(dino_source))
    responses[api.DINOV2_ARCHIVE_URL] = buffer.getvalue()
    responses[api.DINOV2_WEIGHTS_URL] = b"synthetic frozen backbone bytes"
    monkeypatch.setattr(api, "DINOV2_ARCHIVE_SHA256", _sha(buffer.getvalue()))
    monkeypatch.setattr(api, "DINOV2_TREE_SHA256", tree_hash)
    monkeypatch.setattr(api, "DINOV2_WEIGHTS_SHA256", _sha(responses[api.DINOV2_WEIGHTS_URL]))
    calls = []

    def response(url, **kwargs):
        calls.append(url)
        assert url in responses, f"Unexpected download (especially a dataset): {url}"
        return io.BytesIO(responses[url])

    monkeypatch.setattr(api, "urlopen", response)
    monkeypatch.setattr(api.torch, "load", _forbidden)
    monkeypatch.setattr(api, "load_model", _forbidden)
    monkeypatch.setattr(api, "_load_dinov2", _forbidden)
    monkeypatch.setattr(api, "_load_dino_pickle", _forbidden)
    return api, responses, calls


def test_prepare_lewm_includes_point_provenance_stats_and_is_offline_idempotent(tmp_path, synthetic_releases):
    api, _, calls = synthetic_releases
    cache = tmp_path / "cache with spaces"
    manifest = artifacts.prepare(cache, models=("lewm",))
    assert set(manifest["sources"]) == {"lewm", "point"}
    assert api.STATS_PATH in manifest["files"] and manifest["dataset"] is None
    assert not manifest["checkpoint_loaded"]
    assert (Path(manifest["point_root"]) / "stable_worldmodel/native.py").is_file()
    for source in manifest["sources"].values():
        root = Path(source["path"])
        assert _git(root, "rev-parse", "HEAD") == source["revision"]
        assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
        assert _git(root, "rev-list", "--count", "HEAD") == "1"
    before_calls, modified = list(calls), (cache / "artifacts.json").stat().st_mtime_ns
    assert artifacts.prepare(cache, models=("lewm",)) == manifest
    assert calls == before_calls and (cache / "artifacts.json").stat().st_mtime_ns == modified
    assert json.loads((cache / "artifacts.json").read_text()) == manifest


def test_prepare_dino_downloads_backbone_without_trust_import_or_unpickle(tmp_path, synthetic_releases):
    api, _, calls = synthetic_releases
    cache = tmp_path / "cache"
    manifest = artifacts.prepare(cache)
    assert set(manifest["sources"]) == {"lewm", "point", "dinov2"}
    assert "models/dinowm/dino-wm/checkpoints/model_latest.pth" in manifest["files"]
    assert "models/dinov2/dinov2_vits14_pretrain.pth" in manifest["files"]
    assert not any("/datasets/" in url or "/resolve/main/" in url for url in calls)
    assert (cache / "models/dinov2/manifest.json").is_file()
    assert json.loads((cache / "models/dinov2/manifest.json").read_text())["python_tree_sha256"] == api.DINOV2_TREE_SHA256
    before = list(calls)
    artifacts.prepare(cache)
    assert calls == before


def test_sources_only_does_not_fetch_any_release(tmp_path, synthetic_releases):
    _, _, calls = synthetic_releases
    manifest = artifacts.prepare(tmp_path / "cache", models=())
    assert set(manifest["sources"]) == {"point"} and manifest["files"] == {} and calls == []


@pytest.mark.parametrize("kind", ["asset", "dirty", "wrong_head", "hidden_source_change", "symlink"])
def test_prepare_preserves_changed_cache_and_fails_closed(tmp_path, synthetic_releases, kind):
    api, _, _ = synthetic_releases
    cache = tmp_path / "cache"
    artifacts.prepare(cache, models=("lewm",))
    source = cache / "sources/lewm"
    if kind == "asset":
        target = cache / "models/lewm/weights.pt"
        target.write_bytes(b"corrupt existing checkpoint")
    elif kind in ("dirty", "wrong_head"):
        target = source / "user-file"
        target.write_bytes(b"user work must survive")
        if kind == "wrong_head":
            _git(source, "add", ".")
            _git(source, "commit", "--quiet", "-m", "user work")
    elif kind == "hidden_source_change":
        target = source / "jepa.py"
        _git(source, "update-index", "--assume-unchanged", "jepa.py")
        target.write_bytes(b"must fail even when Git status is clean")
    else:
        target = cache / "models/lewm/weights.pt"
        target.unlink()
        target.symlink_to(tmp_path / "outside-checkpoint")
    before = target.read_bytes() if not target.is_symlink() else None
    with pytest.raises((artifacts.ArtifactError, api.ModelArtifactError)):
        artifacts.prepare(cache, models=("lewm",))
    assert target.is_symlink() if kind == "symlink" else target.read_bytes() == before


@pytest.mark.parametrize("options", [{"models": "lewm"}, {"models": ("other",)}, {"dataset": 1}])
def test_bad_prepare_arguments_do_not_touch_disk(tmp_path, options):
    cache = tmp_path / "absent"
    with pytest.raises(ValueError):
        artifacts.prepare(cache, **options)
    assert not cache.exists()


def test_optional_dataset_uses_pinned_raw_h5_and_rechecks_output(h5_path, tmp_path, synthetic_releases, monkeypatch):
    import zstandard as zstd

    api, responses, calls = synthetic_releases
    data = zstd.ZstdCompressor(write_checksum=True).compress(h5_path.read_bytes())
    monkeypatch.setattr(artifacts, "DATASET_SHA256", _sha(data))
    monkeypatch.setattr(artifacts, "DATASET_SIZE", len(data))
    responses[artifacts.DATASET_URL] = data
    cache = tmp_path / "cache"
    manifest = artifacts.prepare(cache, models=(), dataset=True)
    record = manifest["dataset"]
    output = Path(record["path"])
    assert output.name == artifacts.DATASET_FILENAME and output.read_bytes() == h5_path.read_bytes()
    assert record["content"]["rows"] == 34 and record["content"]["episodes"] == 2
    assert record["sha256"] == _sha(h5_path.read_bytes())
    assert record["revision"] == artifacts.DATASET_REVISION
    assert calls == [artifacts.DATASET_URL]
    assert artifacts.prepare(cache, models=(), dataset=True) == manifest
    assert calls == [artifacts.DATASET_URL]
    with output.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(api.ModelArtifactError, match="SHA256"):
        artifacts.prepare(cache, models=(), dataset=True)


def test_package_metadata_has_requested_pins_and_parent_cli_entrypoint():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    project = metadata["project"]
    assert project["scripts"] == {"wm-audit": "wm_audit.cli:main"}
    assert {"numpy==1.26.4", "torch==2.10.0", "torchvision==0.25.0", "gymnasium==1.4.0",
            "pygame==2.6.1", "pymunk==7.3.1", "shapely==2.2.0", "opencv-python-headless==4.11.0.86",
            "transformers==4.57.6", "einops==0.8.2", "omegaconf==2.3.0"} <= set(project["dependencies"])
    assert metadata["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def _script_project(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "src/wm_audit").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/bootstrap.sh", root / "scripts/bootstrap.sh")
    (root / "pyproject.toml").write_text("# command-contract fixture only\n")
    (root / "src/wm_audit/models.py").write_text("# command-contract fixture only\n")
    return root


def _executable(path, content):
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + content)
    path.chmod(0o755)


def test_bootstrap_help_and_syntax_do_not_install_anything(tmp_path):
    script = ROOT / "scripts/bootstrap.sh"
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
    result = subprocess.run(["bash", str(script), "--help"], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "13.14 GB" in result.stdout and "--cpu" in result.stdout and "NEVER deserialize" in result.stdout
    bad = subprocess.run(["bash", str(script), "--not-a-flag"], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode != 0 and not list(tmp_path.iterdir())


def test_bootstrap_preserves_incompatible_existing_venv(tmp_path):
    root = _script_project(tmp_path)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/pyvenv.cfg").write_text("version = 3.12.8\ninclude-system-site-packages = false\n")
    sentinel = root / ".venv/do-not-delete"
    sentinel.write_text("user environment")
    _executable(root / ".venv/bin/python", "exit 1\n")
    result = subprocess.run(["bash", str(root / "scripts/bootstrap.sh"), "--cpu"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "preserving existing .venv" in result.stderr
    assert sentinel.read_text() == "user environment" and not (root / ".bootstrap-venv").exists()


@pytest.mark.parametrize("system,cpu,index,build", [
    ("Linux", False, "https://download.pytorch.org/whl/cu128", "+cu128"),
    ("Linux", True, "https://download.pytorch.org/whl/cpu", "+cpu"),
    ("Darwin", True, "https://pypi.org/simple", ""),
])
def test_bootstrap_command_contract_with_explicit_shell_stubs(tmp_path, system, cpu, index, build):
    """No interpreter/package/GPU/dataset downloads: verify shell orchestration only."""
    root = _script_project(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "invocations"
    python_stub = tools / "python-stub"
    _executable(python_stub, 'printf "python cwd=%s args=%s\\n" "$PWD" "$*" >> "$TEST_LOG"\ncat >/dev/null\n')
    _executable(tools / "uname", f"printf '%s\\n' '{system}'\n")
    _executable(tools / "nvidia-smi", "printf '%s\\n' 'Explicit test stub, not a GPU qualification'\n")
    _executable(tools / "git", "exit 0\n")
    _executable(tools / "uv", '''
if [[ "$1" == --version ]]; then printf '%s\n' 'uv 0.9.28 (fixture-build 2026-01-29)'; exit 0; fi
printf 'uv %s\n' "$*" >> "$TEST_LOG"
if [[ "$1" == venv ]]; then
    target="${!#}"
    mkdir -p "$target/bin"
    cp "$TEST_PYTHON" "$target/bin/python"
    printf '%s\n' 'include-system-site-packages = false' > "$target/pyvenv.cfg"
fi
while [[ $# -gt 0 ]]; do
    if [[ "$1" == --constraint ]]; then cat "$2" >> "$TEST_LOG"; fi
    shift
done
''')
    environment = dict(os.environ, PATH=f"{tools}:/usr/bin:/bin", TEST_LOG=str(log), TEST_PYTHON=str(python_stub))
    arguments = ["bash", str(root / "scripts/bootstrap.sh"), "--download", "--with-dataset", "--trust-checkpoint", "--cache", "relative cache"]
    if cpu:
        arguments.append("--cpu")
    for _ in range(2):
        subprocess.run(arguments, cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    contents = log.read_text()
    assert contents.count("uv venv --python 3.11 --managed-python") == 1
    assert f"--index-url {index}" in contents
    assert f"torch==2.10.0{build}" in contents and f"torchvision==0.25.0{build}" in contents
    assert "-e .[test]" in contents and "--constraint" in contents and "uv pip check" in contents
    assert f"python cwd={root} args=- relative cache 1" in contents
    assert not list((root / ".bootstrap-cache").glob(".torch-constraints.*"))
    assert not (root / ".bootstrap-venv").exists(), "An installed pinned uv must not need a seed venv."
    assert not (root / "relative cache").exists(), "Shell stubs must never really download data."