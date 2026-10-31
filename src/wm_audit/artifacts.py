"""Prepare authenticated, offline-ready RGB releases without loading checkpoints.

``prepare(cache: Path, models=('lewm', 'dinowm'), dataset=False) -> dict``
returns a JSON-serializable manifest and writes ``cache/artifacts.json``.
It fetches detached, shallow, immutable Git commits, verifies the source hashes
used by both the models and native simulator, and uses models._download for
every release asset. Existing modified sources/files fail closed, never reset.
No submodules, upstream installers, point-cloud packages, torch.hub, model
construction, or pickle deserialization are involved. There is deliberately
no trust_checkpoint argument: downloading does not authorize loading DINO.

The optional ORIGINAL scene-camera training dataset is quentinll/lewm-pusht,
NOT fafraob's newer sensor-camera Lance table. Its actual release is a single
13,136,247,974-byte ``.h5.zst`` stream, not a tarball. Both streamed raw HDF5
decompression and a separate strict streamed tar.zst extractor are provided.
The compressed SHA256 is the Hub's LFS OID at the immutable revision below;
the extracted SHA256 is derived during authenticated extraction and retained
in a local manifest. The archive is kept (approximately 60 GB total storage).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO


LEWM_SOURCE_URL = "https://github.com/lucas-maes/le-wm.git"
POINT_SOURCE_URL = "https://github.com/fafraob/point-lewm.git"
DATASET_REPO = "quentinll/lewm-pusht"
DATASET_REVISION = "655cd446b9929369d7d406001da85c15d1457850"
DATASET_ARCHIVE = "pusht_expert_train.h5.zst"
DATASET_FILENAME = "pusht_expert_train.h5"
DATASET_SHA256 = "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318"
DATASET_SIZE = 13_136_247_974
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{DATASET_ARCHIVE}"
_MAX_EXTRACTED_BYTES = 64 << 30
_BUFFER = 1 << 20
_H5_MAGIC = b"\x89HDF\r\n\x1a\n"


class ArtifactError(RuntimeError):
    """An existing cache, source checkout, or archive is unsafe/inconsistent."""


def _safe_path(root: Path, relative: str) -> Path:
    """Reject symlinks in every cache-relative component, including dangling ones."""
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts or "\\" in relative:
        raise ArtifactError(f"Invalid cache-relative path: {relative!r}")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ArtifactError(f"Refusing a cache symlink: {path}")
    return path


def _write_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.is_symlink():
        raise ArtifactError(f"Refusing a manifest symlink: {path}")
    if path.exists():
        if path.read_text() == serialized:
            return
        if not replace:
            raise ArtifactError(f"Existing manifest differs; preserving it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(serialized)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(root: Path, *args: str) -> str:
    # Do not let caller Git environment/config redirect worktrees or execute
    # hooks, fsmonitor, smudge filters, or recursive submodule downloads.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    command = ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false",
               "-c", "submodule.recurse=false", "-C", str(root), *args]
    try:
        return subprocess.run(command, env=environment, check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ArtifactError(f"git {args[0]} failed in {root}: {exc.stderr.strip()}") from exc


def _check_checkout(root: Path, revision: str, hashes: dict[str, str], api) -> None:
    if root.is_symlink() or not (root / ".git").is_dir() or (root / ".git").is_symlink():
        raise ArtifactError(f"Expected a standalone Git checkout, not a symlink/worktree: {root}")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root.resolve():
        raise ArtifactError(f"Git checkout escapes its cache directory: {root}")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ArtifactError(f"Existing checkout is not pinned revision {revision}; preserving {root}")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ArtifactError(f"Existing checkout is not clean; preserving {root}")
    for relative, digest in hashes.items():
        api._verify(_safe_path(root, relative), digest)


def _checkout_source(root: Path, url: str, revision: str, hashes: dict[str, str], api) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A source revision must be a full immutable Git commit.")
    if root.exists() or root.is_symlink():
        _check_checkout(root, revision, hashes, api)
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
            stage = Path(temporary) / "checkout"
            stage.mkdir()
            _git(stage, "init", "--quiet")
            _git(stage, "remote", "add", "origin", url)
            _git(stage, "fetch", "--depth", "1", "--no-tags", "--no-recurse-submodules", "origin", revision)
            _git(stage, "checkout", "--quiet", "--detach", revision)
            _check_checkout(stage, revision, hashes, api)
            if root.exists() or root.is_symlink():
                raise ArtifactError(f"Checkout appeared concurrently; preserving {root}")
            stage.rename(root)
    return {"path": str(root), "url": url, "revision": revision, "sha256": dict(hashes)}


def _download_file(cache: Path, relative: str, url: str, digest: str, api,
                   *, size: int | None = None) -> dict[str, Any]:
    path = _safe_path(cache, relative)
    api._download(path, url, digest)
    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        raise ArtifactError(f"Unexpected byte count for {path}: {actual_size}, expected {size}")
    return {"path": relative, "url": url, "sha256": digest, "size_bytes": actual_size}


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _copy_stream(source: BinaryIO, target: BinaryIO, limit: int, *, expected: int | None = None) -> dict[str, Any]:
    digest, size = hashlib.sha256(), 0
    while block := source.read(_BUFFER):
        size += len(block)
        if size > limit:
            raise ArtifactError("Extracted data exceeds the configured byte limit.")
        target.write(block)
        digest.update(block)
    if expected is not None and size != expected:
        raise ArtifactError(f"Truncated/unexpected extracted data: {size} bytes, expected {expected}")
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _new_destination(destination: str | Path) -> Path:
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Extraction never overwrites an existing destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.parent.resolve() / destination.name


class _BudgetReader:
    """Bound tar headers/padding too, not just the sizes declared by members."""

    def __init__(self, source, limit: int):
        self.source, self.limit, self.count = source, limit, 0

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(min(_BUFFER, size) if size >= 0 else _BUFFER)
        self.count += len(block)
        if self.count > self.limit:
            raise ArtifactError("Decompressed archive exceeds its total byte budget.")
        return block


def safe_extract_tar_zst(archive: str | Path, destination: str | Path, *,
                         max_bytes: int = _MAX_EXTRACTED_BYTES, max_members: int = 1024) -> dict[str, Any]:
    """Stream a tar.zst into a NEW directory; publish only after full validation.

    No extractall, symlinks, hardlinks, special/sparse files, path traversal,
    duplicate entries, ownership/mode restoration, or partial publication.
    ``max_bytes`` bounds file payloads; a separate budget bounds tar overhead.
    Callers authenticating arbitrary archives must verify their digest first.
    """
    import zstandard as zstd

    max_bytes = _positive_integer(max_bytes, "max_bytes")
    max_members = _positive_integer(max_members, "max_members")
    destination = _new_destination(destination)
    files, seen, total, members = {}, set(), 0, 0
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "content"
        stage.mkdir()
        with Path(archive).open("rb") as compressed:
            with zstd.ZstdDecompressor(max_window_size=128 << 10).stream_reader(compressed, read_across_frames=True) as stream:
                reader = _BudgetReader(stream, max_bytes + (max_members + 1) * 10240)
                with tarfile.open(fileobj=reader, mode="r|") as bundle:
                    for member in bundle:
                        members += 1
                        if members > max_members:
                            raise ArtifactError("Archive has too many members.")
                        name = PurePosixPath(member.name)
                        if (not member.name or name.is_absolute() or ".." in name.parts
                                or "\\" in member.name or ":" in member.name or "\0" in member.name
                                or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE)
                                or member.issparse() or member.size < 0):
                            raise ArtifactError(f"Unsafe archive member: {member.name!r}")
                        if str(name) == "." and member.isdir():
                            continue
                        if not name.parts or name.as_posix() in seen:
                            raise ArtifactError(f"Duplicate/empty archive member: {member.name!r}")
                        seen.add(name.as_posix())
                        target = stage.joinpath(*name.parts)
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        if total + member.size > max_bytes:
                            raise ArtifactError("Archive file sizes exceed the configured byte limit.")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = bundle.extractfile(member)
                        if source is None:
                            raise ArtifactError(f"Cannot read archive file: {member.name}")
                        with source, target.open("xb") as output:
                            record = _copy_stream(source, output, max_bytes - total, expected=member.size)
                        files[name.as_posix()] = record
                        total += record["size_bytes"]
                # Consume the frame's footer/checksum, including tar padding.
                while reader.read(_BUFFER):
                    pass
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Extraction destination appeared concurrently: {destination}")
        stage.rename(destination)
    return {"files": files, "size_bytes": total}


def extract_h5_zst(archive: str | Path, destination: str | Path, *,
                    max_bytes: int = _MAX_EXTRACTED_BYTES) -> dict[str, Any]:
    """Stream the actual raw .h5.zst release into a new file, hashing as it goes.

    Validate HDF5 magic and, when present, the Zstandard frame content length.
    ``prepare`` additionally authenticates the compressed bytes and validates
    the complete native H5 action/episode schema before publishing its directory.
    """
    import zstandard as zstd

    max_bytes = _positive_integer(max_bytes, "max_bytes")
    destination = _new_destination(destination)
    fd, temporary = tempfile.mkstemp(prefix=".h5-", dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output, Path(archive).open("rb") as compressed:
            parameters = zstd.get_frame_parameters(compressed.read(18))
            expected = parameters.content_size
            if expected in (zstd.CONTENTSIZE_UNKNOWN, zstd.CONTENTSIZE_ERROR):
                expected = None
            if expected is not None and expected > max_bytes:
                raise ArtifactError("HDF5 frame exceeds the configured byte limit.")
            compressed.seek(0)
            with zstd.ZstdDecompressor(max_window_size=128 << 10).stream_reader(compressed, read_across_frames=True) as source:
                record = _copy_stream(source, output, max_bytes, expected=expected)
        with temporary.open("rb") as stream:
            if stream.read(8) != _H5_MAGIC:
                raise ArtifactError("Expected a raw HDF5 zstd stream, not a tarball or another file type.")
        # Hard-link publication is atomic and fails if the destination appeared;
        # unlike rename/replace it cannot overwrite even an empty existing file.
        os.link(temporary, destination)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_summary(path: Path) -> dict[str, Any]:
    from .offline_data import dataset_info

    info = dataset_info(path)
    return {"rows": info["rows"], "episodes": info["episodes"], "image_shape": info["image_shape"],
            "action_stats": {key: value for key, value in info["action_stats"].items() if key != "source"}}


def _prepare_dataset(cache: Path, api) -> dict[str, Any]:
    root = _safe_path(cache, "datasets/lewm-pusht")
    path = _safe_path(cache, f"datasets/lewm-pusht/{DATASET_FILENAME}")
    manifest_path = _safe_path(cache, "datasets/lewm-pusht/manifest.json")
    identity = {"schema": "wm_audit.original_pusht.v1", "repo_id": DATASET_REPO,
                "revision": DATASET_REVISION, "archive_sha256": DATASET_SHA256,
                "archive_size_bytes": DATASET_SIZE, "file": DATASET_FILENAME,
                "pixels": "original scene-camera RGB; not sensor-view Lance"}
    manifest = None
    if root.exists():
        if not manifest_path.is_file():
            raise ArtifactError(f"Existing dataset has no verified extraction manifest; preserving {root}")
        manifest = json.loads(manifest_path.read_text())
        if (not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in identity.items())
                or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256", "")))):
            raise ArtifactError(f"Dataset manifest does not describe the pinned original archive: {manifest_path}")
        api._verify(path, manifest["sha256"])
        if path.stat().st_size != manifest.get("size_bytes") or _dataset_summary(path) != manifest.get("content"):
            raise ArtifactError("Extracted H5 no longer matches its verified content manifest.")
    archive = _download_file(cache, f"downloads/{DATASET_ARCHIVE}", DATASET_URL, DATASET_SHA256,
                             api, size=DATASET_SIZE)
    if manifest is None:
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dataset-", dir=root.parent) as temporary:
            stage = Path(temporary) / "content"
            stage.mkdir()
            output = stage / DATASET_FILENAME
            record = extract_h5_zst(cache / archive["path"], output)
            manifest = dict(identity, **record, content=_dataset_summary(output))
            _write_json(stage / "manifest.json", manifest)
            if root.exists() or root.is_symlink():
                raise ArtifactError(f"Dataset appeared concurrently; preserving {root}")
            stage.rename(root)
    return dict(manifest, path=str(path), archive=archive)


def _prepare_dinov2(cache: Path, api, files: dict[str, Any]) -> dict[str, Any]:
    archive = f"downloads/dinov2-{api.DINOV2_SOURCE_REVISION}.tar.gz"
    files[archive] = _download_file(cache, archive, api.DINOV2_ARCHIVE_URL, api.DINOV2_ARCHIVE_SHA256, api)
    _safe_path(cache, "sources/dinov2")
    root = api._dinov2_source(cache)  # Only authenticated extraction; NOT _load_dinov2.
    weights = "models/dinov2/dinov2_vits14_pretrain.pth"
    files[weights] = _download_file(cache, weights, api.DINOV2_WEIGHTS_URL, api.DINOV2_WEIGHTS_SHA256, api)
    manifest = {
        "source_repo": "facebookresearch/dinov2", "source_revision": api.DINOV2_SOURCE_REVISION,
        "archive_sha256": api.DINOV2_ARCHIVE_SHA256, "python_tree_sha256": api.DINOV2_TREE_SHA256,
        "weights_url": api.DINOV2_WEIGHTS_URL, "weights_sha256": api.DINOV2_WEIGHTS_SHA256,
        "model": "dinov2_vits14", "feature_key": "x_norm_patchtokens",
        "attention_backend": "native PyTorch; optional xFormers disabled",
        "source_pin_origin": "audit pin; original DINO-WM release used an unrecorded hub main",
    }
    _write_json(_safe_path(cache, "models/dinov2/manifest.json"), manifest)
    return dict(manifest, path=str(root))


def _validate_stats(path: Path) -> None:
    stats = json.loads(path.read_text())
    if not isinstance(stats, dict) or stats.get("action_key") != "action":
        raise ArtifactError("Released normalization must describe native action XY.")
    for key in ("action_mean", "action_std"):
        values = stats.get(key)
        if (not isinstance(values, list) or len(values) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or (key == "action_std" and any(value <= 0 for value in values))):
            raise ArtifactError(f"Invalid released {key}.")


def prepare(cache: Path, models=('lewm', 'dinowm'), dataset: bool = False) -> dict[str, Any]:
    """Fetch/verify a reproducible offline cache, with no checkpoint execution.

    ``models=()`` prepares only native Point simulator sources. LeWM also needs
    Point's pinned provenance source and released population action statistics.
    ``dataset=True`` explicitly opts into the 13.14 GB original training set;
    generated causal roots do not require it. Reruns verify rather than overwrite
    existing releases/checkouts. Only the aggregate requested-assets manifest is
    refreshed when the requested subset changes; source/asset bytes are immutable.
    """
    if type(dataset) is not bool:
        raise ValueError("dataset must be an explicit boolean.")
    if isinstance(models, str):
        raise ValueError("models must be an iterable of names, not a string.")
    selected = []
    for name in models:
        if not isinstance(name, str) or name.lower() not in ("lewm", "dinowm", "dino-wm"):
            raise ValueError("models must contain only 'lewm' and 'dinowm'.")
        name = "dinowm" if name.lower() == "dino-wm" else name.lower()
        if name not in selected:
            selected.append(name)
    # Lazy import: importing artifacts alone neither imports torch nor opens H5.
    from . import models as api
    from .simulator import POINT_REVISION, SOURCE_HASHES

    cache = Path(cache).expanduser()
    if cache.is_symlink():
        raise ArtifactError(f"Cache root must not be a symlink: {cache}")
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if api.POINT_SOURCE_REVISION != POINT_REVISION:
        raise ArtifactError("Model and simulator source revisions disagree.")
    point_hashes = dict(api.POINT_SOURCE_HASHES)
    point_hashes.update({f"stable-worldmodel/stable_worldmodel/{path}": digest for path, digest in SOURCE_HASHES.items()})
    sources = {"point": _checkout_source(_safe_path(cache, "sources/point"), POINT_SOURCE_URL,
                                          api.POINT_SOURCE_REVISION, point_hashes, api)}
    if "lewm" in selected:
        sources["lewm"] = _checkout_source(_safe_path(cache, "sources/lewm"), LEWM_SOURCE_URL,
                                            api.LEWM_SOURCE_REVISION, api.LEWM_SOURCE_HASHES, api)
    files = {}
    for relative, digest in api.RELEASE_HASHES.items():
        if not (any(relative.startswith(f"models/{name}/") for name in selected)
                or (selected and relative == api.STATS_PATH)):
            continue
        if relative.startswith("models/lewm/"):
            repo, revision, remote = "quentinll/lewm-pusht", api.LEWM_HF_REVISION, relative[len("models/lewm/"):]
        else:
            repo, revision, remote = "fafraob/point-cloud-pusht", api.DINOWM_HF_REVISION, relative[len("models/dinowm/"):]
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{remote}"
        files[relative] = _download_file(cache, relative, url, digest, api)
    if selected:
        _validate_stats(cache / api.STATS_PATH)
    if "lewm" in selected and not isinstance(json.loads((cache / "models/lewm/config.json").read_text()), dict):
        raise ArtifactError("LeWM config must be a JSON object.")
    if "dinowm" in selected:
        sources["dinov2"] = _prepare_dinov2(cache, api, files)
    data = _prepare_dataset(cache, api) if dataset else None
    manifest = {
        "schema": "wm_audit.artifacts.v1", "cache": str(cache), "models": selected,
        "sources": sources, "files": files, "dataset": data,
        "point_root": str(cache / "sources/point/stable-worldmodel"),
        "checkpoint_loaded": False, "dinowm_requires_explicit_trust_to_load": True,
    }
    _write_json(_safe_path(cache, "artifacts.json"), manifest, replace=True)
    return manifest