"""Frozen, real PushT release adapters; no surrogate encoders or dynamics.

Public contract::

    model = load_model("lewm" or "dinowm", cache, device="cpu",
                       trust_checkpoint=False)
    z = model.encode(uint8_rgb)                 # [N,T,224,224,3]
    future = model.rollout(z[:, :3], actions)   # actions [N,2+H,10]
    costs = model.goal_cost(future[:, -1], goal)

``latent_shape`` is (192,) for LeWM's PROJECTED CLS token, or (196,384)
for the Point-LeWM-author RGB/no-proprio DINO-WM's DINOv2 patch tokens.
Actions are native relative controls in [-1,1], NOT z-scored, packed as five
ordered XY pairs. The first two chunks connect the three history frames;
chunk 2 produces the first future frame. H must be positive. Rollout and
cost preserve autograd through edited history; parameters stay frozen/eval.
Inference is float32, including on CUDA (not implicit training-time bf16).

The caller bootstraps the following layout under ``cache``. HF downloads
must use the revisions below and retain the dino-wm subtree, NOT flatten it::

    sources/lewm/                         LEWM_SOURCE_REVISION
    sources/point/                        POINT_SOURCE_REVISION
    models/lewm/config.json               quentinll/lewm-pusht
    models/lewm/weights.pt
    models/dinowm/dino-wm/hydra.yaml       fafraob/point-cloud-pusht
    models/dinowm/dino-wm/checkpoints/model_latest.pth
    models/dinowm/dino-wm/swm_h5_norm_stats.json

Both adapters require that stats file. The pinned Point training config
identifies pusht_expert_train.h5 as the source of LeWM's verbatim Lance rows.
Its population statistics match native *evaluation* StandardScaler. LeWM's
training utils use sample std instead; this adapter does not pretend those
two estimators are identical, nor recompute stats on an audit split.

DINO additionally auto-downloads the hash-pinned official DINOv2 archive to
downloads/dinov2-<revision>.tar.gz, extracts sources/dinov2/, and downloads
models/dinov2/dinov2_vits14_pretrain.pth. It records provenance in
models/dinov2/manifest.json. A fully populated cache works offline. Changed
artifacts/sources fail closed, even with trust_checkpoint=True. That flag
authorizes ONLY the known DINO pickle, not arbitrary local checkpoints.
The original release did not record its DINOv2 hub revision; the explicit
historical pin here fixes that ambiguity, rather than claiming it was recorded.
Optional xFormers is disabled for this private backbone, so CPU loading does
not acquire a CUDA-kernel dependency on machines where xFormers is installed.

Runtime dependencies: numpy, torch>=2.6, torchvision, transformers==4.57.6,
einops, omegaconf. Tested with torch 2.10.0 / torchvision 0.25.0. Neither
stable_pretraining, stable_worldmodel, Hydra instantiation, nor point-cloud
CUDA packages are imported. Source modules use private import namespaces;
in particular the upstream models.dino torch.hub monkeypatch is not run.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tarfile
import tempfile
import threading
from types import ModuleType
from typing import Any
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn


LEWM_SOURCE_REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
POINT_SOURCE_REVISION = "1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424"
LEWM_HF_REVISION = "22b330c28c27ead4bfd1888615af1340e3fe9052"
DINOWM_HF_REVISION = "664278a4c0b4c5f607d0993744986b2d10b4168e"
# Historical vit_hf implementation inspected alongside the actual state dict.
VIT_HF_REFERENCE_REVISION = "5f2e0240fd9d14c516acb1a25ac1948c80713f75"
DINOV2_SOURCE_REVISION = "85a24602099d397264d5b30461ad7f3bfd726ca1"
DINOV2_ARCHIVE_SHA256 = "d6e25059c83030d72c02d5ea40973d2c5be15ec5247cd0040474b2f58f34bc7a"
DINOV2_TREE_SHA256 = "d31f667e5b045c888592118021da58dc8dac9781a525bdaf7fe600146b9f8ae7"
DINOV2_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOV2_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
DINOV2_ARCHIVE_URL = f"https://codeload.github.com/facebookresearch/dinov2/tar.gz/{DINOV2_SOURCE_REVISION}"

LEWM_SOURCE_HASHES = {
    "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
    "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
    "config/train/data/pusht.yaml": "dcdd082ebdc628cdc8a88f15cb647fce6eec87343f9fe190271946ff5d52f757",
}
POINT_SOURCE_HASHES = {
    "third_party/dino_wm/models/vit.py": "60c77bdb03f3b2d565bfdb1695c9c193a802236763d33a03e79ed382b51f0670",
    "third_party/dino_wm/models/proprio.py": "b0f7c761c45a6a889dc51f6d9cf4539480d1a0bd2bd200ad44c4b9373a573c72",
    "third_party/dino_wm/models/visual_world_model.py": "931d44c0d541d95be8236f65247de9b40ac3843fe047c1a44d4a2482d1bf636f",
    "dinowm_swm/conf/env/swm_pusht.yaml": "fb0bf9d741d621908daebc5f55db4d9385762f6f4c4f0e45b0b82a1a50e9a23b",
}
RELEASE_HASHES = {
    "models/lewm/config.json": "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09",
    "models/lewm/weights.pt": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
    "models/dinowm/dino-wm/hydra.yaml": "1f2c7767f06a82ae8d86a8be168e303b5941b5105e8cff5060e9e5cb54f3f3a2",
    "models/dinowm/dino-wm/checkpoints/model_latest.pth": "14489cfdf4132a126fd1cb3cac39370a1fb2141503cd3e41aeb3c2dead271dd0",
    "models/dinowm/dino-wm/swm_h5_norm_stats.json": "90619b3b9e0534f223e19eac595330dd426e3319b832ec3717f7ec90d2fe5de5",
}
STATS_PATH = "models/dinowm/dino-wm/swm_h5_norm_stats.json"
_IMPORT_LOCK = threading.RLock()


class ModelArtifactError(RuntimeError):
    """An artifact or executed source differs from the inspected release."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify(path: Path, expected: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned model artifact/source: {path}. See models.py cache layout.")
    actual = _sha256(path)
    if actual != expected:
        raise ModelArtifactError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}.")
    return path


def _verify_sources(root: Path, hashes: dict[str, str]) -> None:
    for relative, digest in hashes.items():
        _verify(root / relative, digest)


def _namespace(label: str, root: Path) -> str:
    return f"_wm_audit_{label}_{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]}"


def _package(name: str, directory: Path) -> None:
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        sys.modules[name] = package


def _source_module(name: str, path: Path, *, package: bool = False,
                   aliases: dict[str, str] | None = None) -> ModuleType:
    """Execute already-verified code without adding source roots to sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None,
    )
    if spec is None or spec.loader is None:
        raise ModelArtifactError(f"Cannot import pinned source {path}.")
    module = importlib.util.module_from_spec(spec)
    if aliases:
        original_import = builtins.__import__

        def redirected_import(import_name, globals=None, locals=None, fromlist=(), level=0):
            mapped = aliases.get(import_name, import_name) if level == 0 else import_name
            return original_import(mapped, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=redirected_import)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _download(path: Path, url: str, digest: str) -> Path:
    """Never overwrite a corrupt cache entry; publish only verified downloads."""
    if path.exists():
        return _verify(path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as target, urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, target)
        _verify(temporary, digest)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _python_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink():
            raise ModelArtifactError(f"Unexpected source symlink: {path}")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(_sha256(path)) + b"\n")
    return digest.hexdigest()


def _dinov2_source(cache: Path) -> Path:
    root = cache / "sources/dinov2"
    if not root.exists():
        archive = _download(cache / f"downloads/dinov2-{DINOV2_SOURCE_REVISION}.tar.gz",
                            DINOV2_ARCHIVE_URL, DINOV2_ARCHIVE_SHA256)
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dinov2-", dir=root.parent) as temporary:
            with tarfile.open(archive, "r:gz") as source:
                # The hash authenticates this archive; also reject links/path traversal.
                prefix = f"dinov2-{DINOV2_SOURCE_REVISION}"
                for member in source.getmembers():
                    parts = Path(member.name).parts
                    if (not parts or parts[0] != prefix or ".." in parts
                            or not (member.isdir() or member.isfile())):
                        raise ModelArtifactError(f"Unexpected DINOv2 archive member: {member.name}")
                source.extractall(temporary, filter="data")
            extracted = Path(temporary) / prefix
            if _python_tree_digest(extracted) != DINOV2_TREE_SHA256:
                raise ModelArtifactError("Extracted DINOv2 Python source digest mismatch.")
            try:
                extracted.rename(root)
            except FileExistsError:
                pass  # Another loader published the same tree; verify it below.
    if _python_tree_digest(root) != DINOV2_TREE_SHA256:
        raise ModelArtifactError(f"DINOv2 source tree differs from {DINOV2_SOURCE_REVISION}: {root}")
    return root


class _DinoPatchEncoder(nn.Module):
    """Lightweight equivalent of models.dino, without its unpinned hub call."""

    name = "dinov2_vits14"
    feature_key = "x_norm_patchtokens"
    latent_ndim = 2
    emb_dim = 384
    patch_size = 14

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.base_model = backbone

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.base_model.forward_features(pixels)[self.feature_key]


def _load_dinov2(cache: Path) -> _DinoPatchEncoder:
    root = _dinov2_source(cache)
    weights = _download(cache / "models/dinov2/dinov2_vits14_pretrain.pth",
                        DINOV2_WEIGHTS_URL, DINOV2_WEIGHTS_SHA256)
    namespace = _namespace("dinov2", root)
    with _IMPORT_LOCK:
        _package(namespace, root / "dinov2")
        _package(namespace + ".models", root / "dinov2/models")
        _package(namespace + ".hub", root / "dinov2/hub")
        # Upstream chooses optional kernels at import time. Keep this private
        # namespace CPU-capable even on a host with CUDA-only xFormers installed.
        previous = os.environ.get("XFORMERS_DISABLED")
        os.environ["XFORMERS_DISABLED"] = "1"
        try:
            _source_module(namespace + ".layers", root / "dinov2/layers/__init__.py", package=True)
        finally:
            if previous is None:
                os.environ.pop("XFORMERS_DISABLED", None)
            else:
                os.environ["XFORMERS_DISABLED"] = previous
        _source_module(namespace + ".models.vision_transformer", root / "dinov2/models/vision_transformer.py",
                       aliases={"dinov2.layers": namespace + ".layers"})
        factory = _source_module(namespace + ".hub.backbones", root / "dinov2/hub/backbones.py")
        with torch.random.fork_rng(devices=[]):
            backbone = factory.dinov2_vits14(pretrained=False)
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    manifest = {
        "source_repo": "facebookresearch/dinov2", "source_revision": DINOV2_SOURCE_REVISION,
        "archive_sha256": DINOV2_ARCHIVE_SHA256, "python_tree_sha256": DINOV2_TREE_SHA256,
        "weights_url": DINOV2_WEIGHTS_URL, "weights_sha256": DINOV2_WEIGHTS_SHA256,
        "model": "dinov2_vits14", "feature_key": "x_norm_patchtokens",
        "attention_backend": "native PyTorch; optional xFormers disabled",
        "source_pin_origin": "audit pin; original DINO-WM release used an unrecorded hub main",
    }
    manifest_path = weights.parent / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not manifest_path.exists() or manifest_path.read_text() != serialized:
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=weights.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(serialized)
            Path(temporary).replace(manifest_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return _DinoPatchEncoder(backbone)


def _action_stats(cache: Path) -> dict[str, Any]:
    path = _verify(cache / STATS_PATH, RELEASE_HASHES[STATS_PATH])
    return json.loads(path.read_text())


class WorldModelAdapter(nn.Module):
    """Shared strict shape/normalization contract; use load_model to construct."""

    latent_shape: tuple[int, ...]
    name: str
    history_size = 3
    action_dim = 10
    frameskip = 5
    cost_reduction: str

    def __init__(self, model: nn.Module, stats: dict[str, Any], device="cpu",
                 metadata: dict[str, Any] | None = None):
        super().__init__()
        self.model = model
        mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["action_std"], dtype=torch.float32)
        if (stats.get("action_key") != "action" or mean.shape != (2,) or std.shape != (2,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all()):
            raise ValueError("Action stats must describe two finite native XY means and positive stds.")
        self.register_buffer("action_mean", mean.repeat(self.frameskip))
        self.register_buffer("action_std", std.repeat(self.frameskip))
        self.metadata = dict(metadata or {})
        self.metadata.update(name=self.name, latent_shape=list(self.latent_shape),
                             history_size=3, frameskip=5, dtype="float32",
                             cost_reduction=self.cost_reduction)
        self.to(device=device, dtype=torch.float32)
        self.requires_grad_(False)
        self.train(False)

    @property
    def device(self) -> torch.device:
        return self.action_mean.device

    def train(self, mode: bool = True):
        # nn.Module.eval() calls train(False); train(True) must not enable dropout/BN.
        super().train(False)
        return self

    def _pixels(self, pixels: torch.Tensor | np.ndarray, mean, std) -> torch.Tensor:
        from torchvision.transforms.v2 import functional as tvf

        if isinstance(pixels, np.ndarray):
            if pixels.dtype != np.uint8:
                raise TypeError("pixels must be uint8, not pre-normalized floats.")
            pixels = torch.from_numpy(np.array(pixels, copy=True, order="C"))
        if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8:
            raise TypeError("pixels must be a torch or numpy uint8 array.")
        if pixels.ndim != 5 or tuple(pixels.shape[2:]) != (224, 224, 3) or min(pixels.shape[:2]) < 1:
            raise ValueError("pixels must have shape [N,T,224,224,3] with N,T > 0.")
        n, t = pixels.shape[:2]
        image = pixels.to(self.device).permute(0, 1, 4, 2, 3).reshape(n * t, 3, 224, 224)
        image = tvf.normalize(tvf.to_dtype(image, torch.float32, scale=True), mean, std)
        return image.reshape(n, t, 3, 224, 224)

    def _latent(self, value: torch.Tensor, label: str, history: bool = False) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{label} must be a floating torch.Tensor.")
        tail = (3, *self.latent_shape) if history else self.latent_shape
        if value.ndim != 1 + len(tail) or tuple(value.shape[1:]) != tail or value.shape[0] < 1:
            raise ValueError(f"{label} must have shape [N,{','.join(map(str, tail))}], N > 0.")
        value = value.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"{label} must be finite.")
        return value

    def _rollout_inputs(self, history, actions) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._latent(history, "history", history=True)
        if isinstance(actions, np.ndarray):
            if actions.dtype.kind not in "fiu":
                raise TypeError("actions must be real numeric native controls.")
            actions = torch.from_numpy(np.array(actions, copy=True, order="C"))
        if (not isinstance(actions, torch.Tensor) or actions.dtype == torch.bool
                or actions.is_complex()):
            raise TypeError("actions must be real torch or numpy native controls.")
        if (actions.ndim != 3 or actions.shape[0] != history.shape[0]
                or actions.shape[1] < 3 or actions.shape[2] != 10):
            raise ValueError("actions must have shape [N,2+H,10], with H >= 1 and matching N.")
        actions = actions.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise ValueError("actions must be finite native relative XY controls in [-1,1], NOT z-scored.")
        return history, (actions - self.action_mean) / self.action_std

    def goal_cost(self, pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """One terminal cost per N; no broadcasting of batches or time axes."""
        pred, goal = self._latent(pred, "pred"), self._latent(goal, "goal")
        if pred.shape != goal.shape:
            raise ValueError("pred and goal must have identical shapes; batch broadcasting is forbidden.")
        if self.cost_reduction == "sum":
            goal = goal.detach()  # JEPA.criterion explicitly detaches its target.
        error = (pred - goal).square().flatten(1)
        return error.sum(1) if self.cost_reduction == "sum" else error.mean(1)


class LeWMAdapter(WorldModelAdapter):
    name = "lewm"
    latent_shape = (192,)
    cost_reduction = "sum"

    def encode(self, pixels: torch.Tensor | np.ndarray) -> torch.Tensor:
        image = self._pixels(pixels, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return self.model.encode({"pixels": image})["emb"]

    def rollout(self, history: torch.Tensor, actions: torch.Tensor | np.ndarray) -> torch.Tensor:
        history, actions = self._rollout_inputs(history, actions)
        predictions = []
        for step in range(actions.shape[1] - 2):
            # Match JEPA's growing action prefix, then truncate BOTH streams to 3.
            act_emb = self.model.action_encoder(actions[:, :step + 3])[:, -3:]
            predicted = self.model.predict(history, act_emb)[:, -1:]
            predictions.append(predicted)
            history = torch.cat((history[:, 1:], predicted), dim=1)
        return torch.cat(predictions, dim=1)


class DinoWMAdapter(WorldModelAdapter):
    name = "dinowm"
    latent_shape = (196, 384)
    cost_reduction = "mean"

    def encode(self, pixels: torch.Tensor | np.ndarray) -> torch.Tensor:
        image = self._pixels(pixels, (0.5,) * 3, (0.5,) * 3)
        zeros = image.new_zeros(*image.shape[:2], 1)
        # Native VWorldModel resizes 224 -> 196 BEFORE the 14-pixel patch encoder.
        return self.model.encode_obs({"visual": image, "proprio": zeros})["visual"]

    @staticmethod
    def _tile(value: torch.Tensor, repeats: int) -> torch.Tensor:
        return value.unsqueeze(2).expand(-1, -1, 196, -1).repeat(1, 1, 1, repeats)

    def rollout(self, history: torch.Tensor, actions: torch.Tensor | np.ndarray) -> torch.Tensor:
        history, actions = self._rollout_inputs(history, actions)
        wm = self.model
        prop = wm.encode_proprio(history.new_zeros(history.shape[0], 3, 1))
        z = torch.cat((history, self._tile(prop, wm.num_proprio_repeat),
                       self._tile(wm.encode_act(actions[:, :3]), wm.num_action_repeat)), dim=-1)
        predictions = []
        horizon = actions.shape[1] - 2
        for step in range(horizon):
            predicted = wm.predict(z)[:, -1:]
            predictions.append(predicted[..., :384])
            if step + 1 < horizon:
                next_action = wm.encode_act(actions[:, step + 3:step + 4])
                # Keep EVERY predicted proprio channel, including patch variation.
                # Replace only action channels; never reset predicted proprio to zero.
                next_z = torch.cat((predicted[..., :-wm.action_dim],
                                    self._tile(next_action, wm.num_action_repeat)), dim=-1)
                z = torch.cat((z[:, 1:], next_z), dim=1)
        return torch.cat(predictions, dim=1)


def _metadata(name: str, cache: Path) -> dict[str, Any]:
    lewm = name == "lewm"
    paths = [key for key in RELEASE_HASHES if key.startswith(f"models/{name}/") or key == STATS_PATH]
    return {
        "source_revision": LEWM_SOURCE_REVISION if lewm else POINT_SOURCE_REVISION,
        "hf_repo": "quentinll/lewm-pusht" if lewm else "fafraob/point-cloud-pusht",
        "hf_revision": LEWM_HF_REVISION if lewm else DINOWM_HF_REVISION,
        "artifact_sha256": {path: RELEASE_HASHES[path] for path in paths},
        "normalization_file": str(cache / STATS_PATH),
        "normalization": "full-H5 population statistics; native evaluation StandardScaler",
        "normalization_provenance": "Point swm_pusht.yaml: LeWM H5 and verbatim Lance rows",
        "action_encoding": "native_relative_xy_time_major_5",
        "surface": "projected_cls" if lewm else "x_norm_patchtokens",
        "upstream_training_precision": "bf16",
        "torch_version": torch.__version__,
        "torchvision_version": importlib.metadata.version("torchvision"),
    }


def _load_lewm(cache: Path, device) -> LeWMAdapter:
    from transformers import ViTConfig, ViTModel

    root = cache / "sources/lewm"
    _verify_sources(root, LEWM_SOURCE_HASHES)
    # Pin the source of the shared-data claim too; no Point package import.
    provenance = "dinowm_swm/conf/env/swm_pusht.yaml"
    _verify(cache / "sources/point" / provenance, POINT_SOURCE_HASHES[provenance])
    cfg_path = "models/lewm/config.json"
    weight_path = "models/lewm/weights.pt"
    cfg = json.loads(_verify(cache / cfg_path, RELEASE_HASHES[cfg_path]).read_text())
    weights = _verify(cache / weight_path, RELEASE_HASHES[weight_path])
    stats = _action_stats(cache)
    namespace = _namespace("lewm", root)
    with _IMPORT_LOCK:
        _package(namespace, root)
        jepa = _source_module(namespace + ".jepa", root / "jepa.py")
        module = _source_module(namespace + ".module", root / "module.py")

    def arguments(key):
        return {k: v for k, v in cfg[key].items() if k != "_target_"}

    # Exact stable_pretraining.backbone.utils.vit_hf(tiny) construction.
    config = ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
                       intermediate_size=768, image_size=224, patch_size=14,
                       num_channels=3, hidden_act="gelu", layer_norm_eps=1e-12,
                       hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                       qkv_bias=True, initializer_range=0.02)
    config.interpolate_pos_encoding = True
    with torch.random.fork_rng(devices=[]):
        encoder = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
        projections = {}
        for key in ("projector", "pred_proj"):
            kwargs = arguments(key)
            kwargs["norm_fn"] = nn.BatchNorm1d
            projections[key] = module.MLP(**kwargs)
        model = jepa.JEPA(encoder=encoder, predictor=module.ARPredictor(**arguments("predictor")),
                          action_encoder=module.Embedder(**arguments("action_encoder")), **projections)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    metadata = _metadata("lewm", cache)
    metadata.update(vit_hf_reference_revision=VIT_HF_REFERENCE_REVISION,
                    transformers_version=importlib.metadata.version("transformers"),
                    pixels="uint8/255 -> ImageNet normalization; 224px; projected CLS")
    return LeWMAdapter(model, stats, device, metadata)


def _load_dino_pickle(path: Path, vit: ModuleType, proprio: ModuleType):
    """Remap the two known upstream class modules, without global models aliases.

    This is still trusted pickle execution, NOT a sandbox. The public loader
    verifies both caller consent and the pinned checkpoint hash first.
    """
    modules = {"models.vit": vit, "models.proprio": proprio}

    class PinnedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in modules:
                return getattr(modules[module], name)
            if module == "models" or module.startswith("models."):
                raise pickle.UnpicklingError(f"Unexpected upstream checkpoint class: {module}.{name}")
            return super().find_class(module, name)

    pickle_module = ModuleType("_wm_audit_pinned_pickle")
    pickle_module.Unpickler = PinnedUnpickler
    pickle_module.load, pickle_module.loads = pickle.load, pickle.loads
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=pickle_module)


def _load_dinowm(cache: Path, device) -> DinoWMAdapter:
    from omegaconf import OmegaConf

    weight_path = "models/dinowm/dino-wm/checkpoints/model_latest.pth"
    weights = _verify(cache / weight_path, RELEASE_HASHES[weight_path])
    cfg_path = "models/dinowm/dino-wm/hydra.yaml"
    cfg = OmegaConf.load(_verify(cache / cfg_path, RELEASE_HASHES[cfg_path]))
    stats = _action_stats(cache)
    root = cache / "sources/point"
    _verify_sources(root, POINT_SOURCE_HASHES)
    namespace = _namespace("dino_wm", root)
    with _IMPORT_LOCK:
        _package(namespace, root / "third_party/dino_wm/models")
        modules = {name: _source_module(namespace + "." + name, root / f"third_party/dino_wm/models/{name}.py")
                   for name in ("vit", "proprio", "visual_world_model")}
        payload = _load_dino_pickle(weights, modules["vit"], modules["proprio"])
    predictor, action_encoder, proprio_encoder = (payload[key] for key in
                                                  ("predictor", "action_encoder", "proprio_encoder"))
    if (action_encoder.in_chans != 10 or action_encoder.emb_dim != 10
            or proprio_encoder.in_chans != 1 or proprio_encoder.emb_dim != 10
            or tuple(predictor.pos_embedding.shape) != (1, 588, 404)):
        raise ModelArtifactError("Not the released RGB/no-proprio 3-frame DINO-WM architecture.")
    # Upstream pickles CUDA causal masks as plain attributes. Make them movable
    # buffers (not trainable parameters), preserving their exact numeric values.
    for layer in predictor.modules():
        bias = getattr(layer, "bias", None)
        if torch.is_tensor(bias) and not isinstance(bias, nn.Parameter):
            delattr(layer, "bias")
            layer.register_buffer("bias", bias, persistent=False)
    encoder = _load_dinov2(cache)
    wm = modules["visual_world_model"].VWorldModel(
        image_size=cfg.img_size, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
        encoder=encoder, proprio_encoder=proprio_encoder, action_encoder=action_encoder,
        decoder=None, predictor=predictor, proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim, concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat,
        train_encoder=False, train_predictor=False, train_decoder=False,
    )
    metadata = _metadata("dinowm", cache)
    metadata.update(dinov2_source_revision=DINOV2_SOURCE_REVISION,
                    dinov2_weights_sha256=DINOV2_WEIGHTS_SHA256, checkpoint_trusted=True,
                    dinov2_attention_backend="native PyTorch; optional xFormers disabled",
                    dinov2_source_pin_origin="audit pin; original release used unrecorded hub main",
                    pixels="uint8/255 -> Normalize(0.5,0.5) -> bilinear antialiased Resize(196)")
    return DinoWMAdapter(wm, stats, device, metadata)


def load_model(name: str, cache: Path, device="cpu", trust_checkpoint: bool = False) -> WorldModelAdapter:
    """Load only the two pinned releases. See module docstring for cache layout.

    LeWM is weights-only. DINO requires literal ``trust_checkpoint=True``
    BEFORE any file access, source import, download or pickle deserialization.
    Missing primary releases/sources raise; only DINOv2 is auto-downloaded.
    """
    if not isinstance(name, str) or name.lower() not in ("lewm", "dinowm", "dino-wm"):
        raise ValueError("name must be 'lewm' or 'dinowm' (alias 'dino-wm').")
    name = name.lower()
    if name != "lewm" and trust_checkpoint is not True:
        raise PermissionError("DINO-WM contains pickled nn.Modules; explicitly pass trust_checkpoint=True "
                              "to authorize the SHA256-pinned release only.")
    cache = Path(cache).expanduser().resolve()
    return _load_lewm(cache, device) if name == "lewm" else _load_dinowm(cache, device)