"""Adapter contracts and native parity, with explicitly labeled test doubles.

No checkpoint downloads occur during collection/unit tests. Real tests run
when WM_AUDIT_MODEL_CACHE (default: workspace/cache) contains pinned releases
and sources. Missing files skip ONLY when that variable is unset; an explicit
cache, a hash mismatch, failed import or a numerical discrepancy must fail.
Real DINO tests additionally require WM_AUDIT_TRUST_DINOWM=1 because the
release contains pickle. DINOv2 is auto-fetched once by the public loader.

The synthetic tests below exercise temporal indexing/shape errors only and
are NOT evidence of pretrained model competence. The real tests compare all
future steps with the actual pinned JEPA.rollout/VWorldModel.rollout, not a
second local implementation of our recurrence. No CUDA-only source constructor
is patched or mocked in real tests; released DINO modules are unpickled.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import MethodType, ModuleType

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.transforms import v2

from wm_audit import models
from wm_audit.models import DinoWMAdapter, LeWMAdapter, ModelArtifactError, load_model


SYNTHETIC_STATS = {"action_mean": [0.1, -0.2], "action_std": [0.5, 0.25], "action_key": "action"}


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    original = torch.get_num_threads()
    torch.set_num_threads(min(2, original))
    yield
    torch.set_num_threads(original)


class _ToyLeWM(nn.Module):
    """TEST DOUBLE: encodes action index and full 3-frame history algebraically."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(0.9)
        self.action_encoder = nn.Identity()

    def encode(self, info):
        # Exposes the first normalized RGB triplet for preprocessing assertions.
        info["emb"] = info["pixels"][..., 0, 0].repeat(1, 1, 64) * self.scale
        return info

    def predict(self, history, actions):
        shift = actions[..., :1] + 10 * actions[..., 1:2]
        return self.dropout(history.cumsum(1) * self.scale + shift)


class _ToyDinoWM(nn.Module):
    """TEST DOUBLE: deliberately predicts nonconstant per-patch proprio channels."""

    num_proprio_repeat = 1
    num_action_repeat = 1
    action_dim = 10
    proprio_dim = 10

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(0.9)

    def encode_act(self, actions):
        return actions

    def encode_proprio(self, prop):
        return (prop + 0.75).expand(*prop.shape[:2], 10)

    def encode_obs(self, obs):
        rgb = obs["visual"][..., 0, 0]
        visual = rgb.repeat(1, 1, 128).unsqueeze(2).expand(-1, -1, 196, -1)
        return {"visual": visual * self.scale, "proprio": self.encode_proprio(obs["proprio"])}

    def predict(self, z):
        vis, prop, act = z[..., :384], z[..., 384:394], z[..., 394:]
        # Both patch-varying proprio and previous visual history affect later output.
        vis = (vis.cumsum(1) + prop[..., :1] + act[..., :1]) * self.scale
        offset = torch.arange(196, device=z.device, dtype=z.dtype)[None, None, :, None] / 196
        prop = 2 * prop + 1 + offset
        return self.dropout(torch.cat((vis, prop, act), dim=-1))


@pytest.fixture(params=["lewm", "dinowm"])
def synthetic_adapter(request):
    if request.param == "lewm":
        return LeWMAdapter(_ToyLeWM(), SYNTHETIC_STATS)
    return DinoWMAdapter(_ToyDinoWM(), SYNTHETIC_STATS)


def test_synthetic_parameters_are_frozen_and_stay_eval(synthetic_adapter):
    adapter = synthetic_adapter
    assert adapter.train() is adapter and adapter.eval() is adapter
    assert all(not module.training for module in adapter.modules())
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert adapter.device == torch.device("cpu")
    assert adapter.metadata["latent_shape"] == list(adapter.latent_shape)


def test_synthetic_native_actions_use_alternating_xy_stats(synthetic_adapter):
    adapter = synthetic_adapter
    history = torch.zeros(2, 3, *adapter.latent_shape, requires_grad=True)
    actions = torch.linspace(-1, 1, 80).reshape(2, 4, 10).requires_grad_()
    original = actions.detach().clone()
    _, normalized = adapter._rollout_inputs(history, actions)
    mean = torch.tensor([0.1, -0.2])
    std = torch.tensor([0.5, 0.25])
    reference = ((actions.reshape(2, 4, 5, 2) - mean) / std).reshape(2, 4, 10)
    torch.testing.assert_close(normalized, reference, rtol=0, atol=0)
    torch.testing.assert_close(actions, original, rtol=0, atol=0)
    normalized.sum().backward()
    torch.testing.assert_close(actions.grad, (1 / std).repeat(5).expand_as(actions))


def test_recorded_action_opt_in_preserves_values_and_normalizes_once(synthetic_adapter):
    adapter = synthetic_adapter
    history = torch.zeros(1, 3, *adapter.latent_shape)
    actions = torch.linspace(-1.5, 2.05, 40).reshape(1, 4, 10)
    before = actions.clone()
    with pytest.raises(ValueError, match="recorded_actions"):
        adapter.rollout(history, actions)
    _, normalized = adapter._rollout_inputs(history, actions, recorded_actions=True)
    torch.testing.assert_close(normalized, (actions - adapter.action_mean) / adapter.action_std)
    assert torch.isfinite(adapter.rollout(history, actions, recorded_actions=True)).all()
    torch.testing.assert_close(before, actions, rtol=0, atol=0)
    bounded = actions.clamp(-1, 1)
    torch.testing.assert_close(adapter.rollout(history, bounded),
                               adapter.rollout(history, bounded, recorded_actions=True), rtol=0, atol=0)
    with pytest.raises(TypeError, match="boolean"):
        adapter.rollout(history, actions, recorded_actions=1)
    actions[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        adapter.rollout(history, actions, recorded_actions=True)


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_synthetic_lewm_temporal_indexing(horizon):
    adapter = LeWMAdapter(_ToyLeWM(), SYNTHETIC_STATS)
    history = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1).expand(-1, -1, 192).clone()
    actions = torch.linspace(-0.9, 0.9, 2 * (horizon + 2) * 10).reshape(2, horizon + 2, 10)
    predicted = adapter.rollout(history, actions)
    norm = (actions - torch.tensor([0.1, -0.2]).repeat(5)) / torch.tensor([0.5, 0.25]).repeat(5)
    expected, context = [], history.clone()
    for i in range(horizon):
        next_z = context.sum(1, keepdim=True) + norm[:, i + 2:i + 3, :1] + 10 * norm[:, i + 2:i + 3, 1:2]
        expected.append(next_z)
        context = torch.cat((context[:, 1:], next_z), dim=1)
    torch.testing.assert_close(predicted, torch.cat(expected, dim=1))
    changed = actions.clone()
    changed[:, -1] = -changed[:, -1]
    other = adapter.rollout(history, changed)
    torch.testing.assert_close(predicted[:, :-1], other[:, :-1], rtol=0, atol=0)
    assert not torch.equal(predicted[:, -1], other[:, -1])


def test_synthetic_dino_keeps_predicted_proprio_across_every_patch():
    adapter = DinoWMAdapter(_ToyDinoWM(), SYNTHETIC_STATS)
    history = torch.zeros(1, 3, 196, 384)
    # Native means -> exactly zero normalized actions; only proprio drives this test.
    actions = torch.tensor([0.1, -0.2]).repeat(5).expand(1, 5, 10)
    actual = adapter.rollout(history, actions)
    offset = torch.arange(196) / 196
    torch.testing.assert_close(actual[0, 0, :, 0], torch.full((196,), 0.75))
    torch.testing.assert_close(actual[0, 1, :, 0], 3.25 + offset)
    torch.testing.assert_close(actual[0, 2, :, 0], 10 + 4 * offset)
    # Resetting to encode_proprio(0) would instead give 1.5 at the second step.
    assert not torch.allclose(actual[0, 1, :, 0], torch.full((196,), 1.5))


def test_synthetic_rollout_history_and_action_gradients(synthetic_adapter):
    adapter = synthetic_adapter
    history = torch.zeros(2, 3, *adapter.latent_shape, requires_grad=True)
    actions = torch.zeros(2, 5, 10, requires_grad=True)
    before = history.detach().clone()
    predicted = adapter.rollout(history, actions)
    assert predicted.shape == (2, 3, *adapter.latent_shape)
    objective = adapter.goal_cost(predicted[:, -1], torch.ones_like(predicted[:, -1])).sum()
    grad_history, grad_actions = torch.autograd.grad(objective, (history, actions))
    assert torch.isfinite(grad_history).all() and (grad_history.flatten(2).abs().sum(2) > 0).all()
    assert torch.isfinite(grad_actions).all() and (grad_actions[:, 2:].abs().sum((0, 2)) > 0).all()
    torch.testing.assert_close(history, before, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in adapter.parameters())


def test_synthetic_goal_cost_reduction_and_gradient(synthetic_adapter):
    adapter = synthetic_adapter
    pred = torch.stack((torch.ones(adapter.latent_shape), torch.full(adapter.latent_shape, 2.0))).requires_grad_()
    goal = torch.zeros_like(pred, requires_grad=True)
    cost = adapter.goal_cost(pred, goal)
    multiplier = 192 if adapter.name == "lewm" else 1
    torch.testing.assert_close(cost, torch.tensor([1.0, 4.0]) * multiplier)
    cost.sum().backward()
    assert pred.grad is not None
    assert (goal.grad is None) == (adapter.name == "lewm")


def test_synthetic_uint8_preprocessing_and_numpy_equivalence(synthetic_adapter):
    adapter = synthetic_adapter
    pixels = torch.zeros(2, 2, 224, 224, 3, dtype=torch.uint8)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 17, 127, 255
    result = adapter.encode(pixels)
    expected = pixels[0, 0, 0, 0].float().mul(1 / 255)
    if adapter.name == "lewm":
        expected = (expected - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(result[0, 0, :3], expected)
    else:
        torch.testing.assert_close(result[0, 0, 0, :3], (expected - 0.5) / 0.5)
    torch.testing.assert_close(adapter.encode(pixels.numpy()), result, rtol=0, atol=0)
    # Negative-stride NumPy views are permitted, without exposing shared mutable memory.
    torch.testing.assert_close(adapter.encode(pixels.numpy()[:, :, ::-1]), result, rtol=0, atol=0)


@pytest.mark.parametrize("pixels,error", [
    (torch.zeros(1, 3, 224, 224, 3), TypeError),
    (np.zeros((1, 3, 224, 224, 3), dtype=np.float32), TypeError),
    (torch.zeros(1, 3, 3, 224, 224, dtype=torch.uint8), ValueError),
    (torch.zeros(1, 3, 196, 196, 3, dtype=torch.uint8), ValueError),
    (torch.zeros(0, 3, 224, 224, 3, dtype=torch.uint8), ValueError),
    (torch.zeros(1, 0, 224, 224, 3, dtype=torch.uint8), ValueError),
    ([], TypeError),
])
def test_synthetic_encode_rejects_bad_contract(synthetic_adapter, pixels, error):
    with pytest.raises(error):
        synthetic_adapter.encode(pixels)


@pytest.mark.parametrize("kind", ["short_history", "long_history", "integer_history", "nonfinite_history",
                                  "no_future", "wrong_action_width", "wrong_batch", "bool_actions",
                                  "complex_actions", "nan_actions", "normalized_actions"])
def test_synthetic_rollout_rejects_bad_contract(synthetic_adapter, kind):
    adapter = synthetic_adapter
    history = torch.zeros(1, 3, *adapter.latent_shape)
    actions = torch.zeros(1, 4, 10)
    if kind == "short_history":
        history = history[:, :2]
    elif kind == "long_history":
        history = torch.zeros(1, 4, *adapter.latent_shape)
    elif kind == "integer_history":
        history = history.long()
    elif kind == "nonfinite_history":
        history.fill_(float("inf"))
    elif kind == "no_future":
        actions = actions[:, :2]
    elif kind == "wrong_action_width":
        actions = actions[..., :2]
    elif kind == "wrong_batch":
        actions = actions.expand(2, -1, -1)
    elif kind == "bool_actions":
        actions = actions.bool()
    elif kind == "complex_actions":
        actions = actions.to(torch.complex64)
    elif kind == "nan_actions":
        actions[0, 0, 0] = float("nan")
    elif kind == "normalized_actions":
        actions.fill_(1.01)
    with pytest.raises((TypeError, ValueError)):
        adapter.rollout(history, actions)


@pytest.mark.parametrize("kind", ["time_axis", "wrong_latent", "wrong_batch", "nonfinite", "integer"])
def test_synthetic_cost_rejects_broadcasting(synthetic_adapter, kind):
    adapter = synthetic_adapter
    pred = torch.ones(2, *adapter.latent_shape)
    goal = torch.zeros_like(pred)
    if kind == "time_axis":
        pred, goal = pred.unsqueeze(1), goal.unsqueeze(1)
    elif kind == "wrong_latent":
        goal = goal[..., :-1]
    elif kind == "wrong_batch":
        goal = goal[:1]
    elif kind == "nonfinite":
        goal.fill_(float("nan"))
    elif kind == "integer":
        goal = goal.long()
    with pytest.raises((TypeError, ValueError)):
        adapter.goal_cost(pred, goal)


@pytest.mark.parametrize("name", ["other", "", None])
def test_unknown_model_fails_before_io(name, tmp_path):
    with pytest.raises(ValueError, match="name"):
        load_model(name, tmp_path)


@pytest.mark.parametrize("trust", [False, None, 1, "yes"])
def test_dino_requires_explicit_boolean_trust_before_io(monkeypatch, tmp_path, trust):
    def forbidden(*args, **kwargs):
        raise AssertionError("Untrusted load reached disk or deserialization.")

    monkeypatch.setattr(models, "_verify", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(PermissionError, match="trust_checkpoint=True"):
        load_model("dinowm", tmp_path, trust_checkpoint=trust)
    assert not list(tmp_path.iterdir())


def test_dino_trust_does_not_allow_arbitrary_local_pickle(monkeypatch, tmp_path):
    path = tmp_path / "models/dinowm/dino-wm/checkpoints/model_latest.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the pinned release")

    def forbidden(*args, **kwargs):
        raise AssertionError("Hash mismatch must prevent torch.load.")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ModelArtifactError, match="SHA256 mismatch"):
        load_model("dinowm", tmp_path, trust_checkpoint=True)


def test_hash_verifier_does_not_accept_missing_or_different_sources(tmp_path):
    source = tmp_path / "module.py"
    with pytest.raises(FileNotFoundError):
        models._verify(source, "0" * 64)
    source.write_text("raise AssertionError('must never run')\n")
    with pytest.raises(ModelArtifactError):
        models._verify(source, "0" * 64)


def test_missing_or_changed_stats_never_fall_back_to_identity(tmp_path):
    with pytest.raises(FileNotFoundError):
        models._action_stats(tmp_path)
    path = tmp_path / models.STATS_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(SYNTHETIC_STATS))
    with pytest.raises(ModelArtifactError, match="SHA256 mismatch"):
        models._action_stats(tmp_path)


def test_changed_dinov2_cache_is_not_silently_refetched(monkeypatch, tmp_path):
    root = tmp_path / "sources/dinov2"
    root.mkdir(parents=True)
    (root / "unexpected.py").write_text("# changed source\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("A source mismatch must not silently redownload.")

    monkeypatch.setattr(models, "_download", forbidden)
    with pytest.raises(ModelArtifactError, match="source tree"):
        models._dinov2_source(tmp_path)


def test_synthetic_download_verifies_before_publication_and_reuses_offline(monkeypatch, tmp_path):
    """TEST DOUBLE: tiny HTTP response exercises atomic cache handling, not weights."""
    data = b"synthetic transport fixture"
    digest = hashlib.sha256(data).hexdigest()
    path = tmp_path / "artifacts/weights.bin"
    monkeypatch.setattr(models, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    assert models._download(path, "https://example.invalid/fixture", digest) == path
    assert path.read_bytes() == data and list(path.parent.iterdir()) == [path]

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing verified artifact must load offline.")

    monkeypatch.setattr(models, "urlopen", forbidden)
    assert models._download(path, "https://example.invalid/fixture", digest) == path
    path.write_bytes(b"corrupt")
    with pytest.raises(ModelArtifactError):
        models._download(path, "https://example.invalid/fixture", digest)


def test_synthetic_download_failure_never_publishes_partial_file(monkeypatch, tmp_path):
    """TEST DOUBLE: bad response must fail closed and clean its temporary file."""
    monkeypatch.setattr(models, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong bytes"))
    path = tmp_path / "weights.bin"
    with pytest.raises(ModelArtifactError):
        models._download(path, "https://example.invalid/fixture", "0" * 64)
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def real_cache():
    explicit = os.environ.get("WM_AUDIT_MODEL_CACHE")
    cache = Path(explicit) if explicit else Path(__file__).resolve().parents[1] / "cache"
    if not cache.is_dir():
        if explicit:
            pytest.fail(f"Explicit WM_AUDIT_MODEL_CACHE does not exist: {cache}")
        pytest.skip("Real checkpoints absent; set WM_AUDIT_MODEL_CACHE to the populated pinned cache.")
    return cache


def _require_real(cache, relative_paths):
    missing = [str(cache / path) for path in relative_paths if not (cache / path).is_file()]
    if missing:
        message = "Missing real integration artifacts: " + ", ".join(missing)
        if os.environ.get("WM_AUDIT_MODEL_CACHE"):
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(scope="module")
def real_lewm(real_cache):
    _require_real(real_cache, ["sources/lewm/jepa.py", "sources/lewm/module.py",
                              "sources/point/dinowm_swm/conf/env/swm_pusht.yaml",
                              "models/lewm/config.json", "models/lewm/weights.pt", models.STATS_PATH])
    return load_model("lewm", real_cache)


@pytest.fixture(scope="module")
def real_dinowm(real_cache):
    if os.environ.get("WM_AUDIT_TRUST_DINOWM") != "1":
        pytest.skip("Real pickle tests require WM_AUDIT_TRUST_DINOWM=1 (explicit trusted release consent).")
    _require_real(real_cache, ["sources/point/third_party/dino_wm/models/visual_world_model.py",
                              "models/dinowm/dino-wm/hydra.yaml",
                              "models/dinowm/dino-wm/checkpoints/model_latest.pth", models.STATS_PATH])
    return load_model("dinowm", real_cache, trust_checkpoint=True)


def _deterministic_pixels(n=1, t=3):
    return torch.randint(0, 256, (n, t, 224, 224, 3), generator=torch.Generator().manual_seed(971), dtype=torch.uint8)


def _native_pixels(pixels, mean, std):
    # Independent use of the actual native evaluation transform sequence.
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean, std), v2.Resize(224)])
    n, t = pixels.shape[:2]
    image = pixels.reshape(n * t, 224, 224, 3).permute(0, 3, 1, 2)
    return transform(image).reshape(n, t, 3, 224, 224)


def _native_actions(cache, actions):
    # Independent normalization in primitive XY coordinates, NOT adapter buffers.
    stats = json.loads((cache / models.STATS_PATH).read_text())
    xy = actions.reshape(*actions.shape[:2], 5, 2)
    return ((xy - torch.tensor(stats["action_mean"])) / torch.tensor(stats["action_std"])).flatten(2)


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("horizon", [1, 3])
def test_real_lewm_encoder_and_every_rollout_step_match_upstream(real_lewm, real_cache, horizon, recorded):
    adapter = real_lewm
    pixels = _deterministic_pixels(n=2)
    actions = torch.linspace(-0.65, 0.65, 2 * (horizon + 2) * 10).reshape(2, horizon + 2, 10)
    if recorded:
        actions *= 3.2  # Demonstration values outside the simulator action box.
    native_pixels = _native_pixels(pixels, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    with torch.no_grad():
        native_z = adapter.model.encode({"pixels": native_pixels})["emb"]
        encoded = adapter.encode(pixels)
        torch.testing.assert_close(encoded, native_z, rtol=0, atol=0)
        raw_cls = adapter.model.encoder(native_pixels.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[:, 0]
        assert not torch.allclose(encoded.flatten(0, 1), raw_cls)
        native = adapter.model.rollout({"pixels": native_pixels[:, None]},
                                       _native_actions(real_cache, actions)[:, None])["predicted_emb"][:, 0, 3:]
        actual = adapter.rollout(encoded, actions, recorded_actions=recorded)
    assert native.shape == actual.shape == (2, horizon, 192)
    torch.testing.assert_close(actual, native, rtol=1e-6, atol=1e-6)
    goal = native_z[:, -1]
    expected_cost = adapter.model.criterion({"predicted_emb": actual[:, None], "goal_emb": goal[:, None, None]})[:, 0]
    torch.testing.assert_close(adapter.goal_cost(actual[:, -1], goal), expected_cost, rtol=0, atol=0)
    assert adapter.model.encoder.config.num_attention_heads == 3
    assert adapter.model.encoder.config.num_hidden_layers == 12
    assert adapter.model.encoder.config.intermediate_size == 768


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("horizon", [1, 3])
def test_real_dino_encoder_and_every_rollout_step_match_upstream(real_dinowm, real_cache, horizon, recorded):
    adapter = real_dinowm
    pixels = _deterministic_pixels()
    actions = torch.linspace(-0.7, 0.7, (horizon + 2) * 10).reshape(1, horizon + 2, 10)
    if recorded:
        actions *= 3
    image = _native_pixels(pixels, (0.5,) * 3, (0.5,) * 3)
    observation = {"visual": image, "proprio": torch.zeros(1, 3, 1)}
    with torch.no_grad():
        # Native VWorldModel itself performs the 196px resize and encoder call.
        native_z = adapter.model.encode_obs(observation)["visual"]
        encoded = adapter.encode(pixels)
        torch.testing.assert_close(encoded, native_z, rtol=0, atol=0)
        native, _ = adapter.model.rollout(observation, _native_actions(real_cache, actions))
        actual = adapter.rollout(encoded, actions, recorded_actions=recorded)
    assert actual.shape == (1, horizon, 196, 384)
    assert adapter.model.encoder_image_size == 196
    torch.testing.assert_close(actual, native["visual"][:, 3:], rtol=1e-6, atol=1e-6)
    goal = native_z[:, -1]
    # Execute the actual upstream last-step objective with proprio weight zero,
    # as in the released RGB/no-proprio evaluation wrapper's visual-only metric.
    objective_path = real_cache / "sources/point/third_party/dino_wm/planning/objectives.py"
    assert hashlib.sha256(objective_path.read_bytes()).hexdigest() == "ac7273b44bad2c84d15b0d7f11ca68e08d19ecf7ee8db697813b01a18126a4fe"
    spec = importlib.util.spec_from_file_location("_test_native_dino_objective", objective_path)
    objective_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(objective_module)
    objective = objective_module.create_objective_fn(alpha=0, base=1, mode="last")
    expected_cost = objective(native, {"visual": goal[:, None], "proprio": torch.zeros(1, 1, 10)})
    torch.testing.assert_close(adapter.goal_cost(actual[:, -1], goal), expected_cost, rtol=1e-6, atol=1e-6)
    masks = [layer.bias for layer in adapter.model.predictor.modules() if hasattr(layer, "bias")
             and isinstance(layer.bias, torch.Tensor) and not isinstance(layer.bias, nn.Parameter)]
    assert len(masks) == 6 and all(mask.device.type == "cpu" for mask in masks)
    assert all(mask.shape == (1, 1, 588, 588) for mask in masks)
    for mask in masks:
        assert (mask[..., :196, 196:] == 0).all()
        assert (mask[..., 392:, :] == 1).all()


@pytest.mark.parametrize("fixture_name", ["real_lewm", "real_dinowm"])
def test_real_multistep_history_gradients_frozen_eval_and_immutability(request, fixture_name):
    adapter = request.getfixturevalue(fixture_name)
    pixels = _deterministic_pixels()
    with torch.no_grad():
        encoded = adapter.encode(pixels)
    history = encoded.clone().requires_grad_()
    actions = torch.linspace(-0.2, 0.2, 40).reshape(1, 4, 10).requires_grad_()
    original_actions = actions.detach().clone()
    buffers = {name: buffer.clone() for name, buffer in adapter.named_buffers()}
    future = adapter.rollout(history, actions)
    loss = adapter.goal_cost(future[:, -1], encoded[:, -1]).sum()
    history_grad, action_grad = torch.autograd.grad(loss, (history, actions))
    assert torch.isfinite(history_grad).all()
    assert (history_grad.flatten(2).abs().sum(2) > 0).all(), "All three edited history frames must remain connected."
    assert torch.isfinite(action_grad).all() and (action_grad.abs().sum(2) > 0).all()
    torch.testing.assert_close(history, encoded, rtol=0, atol=0)
    torch.testing.assert_close(actions, original_actions, rtol=0, atol=0)
    for name, buffer in adapter.named_buffers():
        torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)
    adapter.train()
    assert all(not layer.training for layer in adapter.modules())
    assert all(not p.requires_grad and p.grad is None for p in adapter.parameters())
    with torch.no_grad():
        again = adapter.rollout(history, actions)
    torch.testing.assert_close(again, future, rtol=0, atol=0)


@contextmanager
def _use_encoded_history(model, encoded):
    """Bypass ONLY image encoding to compare edited-latent native recurrence."""
    name = "encode" if encoded.ndim == 3 else "encode_obs"
    previous = model.__dict__.get(name)
    had_instance_method = name in model.__dict__

    def lewm_encode(self, info):
        return dict(info, emb=encoded, act_emb=self.action_encoder(info["action"]))

    def dino_encode(self, observation):
        return {"visual": encoded, "proprio": self.encode_proprio(observation["proprio"])}

    setattr(model, name, MethodType(lewm_encode if name == "encode" else dino_encode, model))
    try:
        yield
    finally:
        if had_instance_method:
            setattr(model, name, previous)
        else:
            delattr(model, name)


@pytest.mark.parametrize("fixture_name", ["real_lewm", "real_dinowm"])
def test_real_edited_latent_native_rollout_parity(request, real_cache, fixture_name):
    """Real released predictors, deterministic non-image history, no dynamics mocks."""
    adapter = request.getfixturevalue(fixture_name)
    history = torch.randn((1, 3, *adapter.latent_shape), generator=torch.Generator().manual_seed(122))
    actions = torch.linspace(-0.3, 0.3, 50).reshape(1, 5, 10)
    normalized = _native_actions(real_cache, actions)
    with torch.no_grad(), _use_encoded_history(adapter.model, history):
        if adapter.name == "lewm":
            observation = {"pixels": torch.zeros(1, 1, 3, 3, 224, 224)}
            native = adapter.model.rollout(observation, normalized[:, None])["predicted_emb"][:, 0, 3:]
        else:
            observation = {"visual": torch.zeros(1, 3, 3, 224, 224), "proprio": torch.zeros(1, 3, 1)}
            native = adapter.model.rollout(observation, normalized)[0]["visual"][:, 3:]
        actual = adapter.rollout(history, actions)
    torch.testing.assert_close(actual, native, rtol=1e-6, atol=1e-6)


def test_real_load_does_not_import_heavy_packages_or_pollute_namespace(real_cache, real_lewm, monkeypatch):
    tracked = ("stable_worldmodel", "stable_pretraining", "pc_encoders", "models", "models.dino",
               "dinov2", "jepa", "module")
    before = {name: sys.modules.get(name) for name in tracked}
    sys_path = list(sys.path)
    hub_validator = torch.hub._validate_not_a_forked_repo
    sentinel = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", sentinel)
    # Reloading also tests that already imported private classes can be reused.
    adapter = load_model("lewm", real_cache)
    assert sys.modules["models"] is sentinel
    assert all(sys.modules.get(name) is before[name] for name in tracked if name != "models")
    assert sys.path == sys_path and torch.hub._validate_not_a_forked_repo is hub_validator
    assert adapter.model.__class__ is real_lewm.model.__class__


def test_real_dino_offline_reload_has_no_hub_or_heavy_import_side_effects(real_dinowm, real_cache, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Populated pinned cache must not use torch.hub or the network.")

    tracked = ("stable_worldmodel", "stable_pretraining", "pc_encoders", "models.vit", "models.dino", "dinov2")
    before = {name: sys.modules.get(name) for name in tracked}
    sentinel = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", sentinel)
    monkeypatch.setattr(torch.hub, "load", forbidden)
    monkeypatch.setattr(models, "urlopen", forbidden)
    sys_path = list(sys.path)
    hub_validator = torch.hub._validate_not_a_forked_repo
    original_xformers = os.environ.get("XFORMERS_DISABLED")
    adapter = load_model("dino-wm", real_cache, trust_checkpoint=True)
    assert adapter.model.__class__ is real_dinowm.model.__class__
    assert all(sys.modules.get(name) is before[name] for name in tracked)
    assert sys.modules["models"] is sentinel and sys.path == sys_path
    assert torch.hub._validate_not_a_forked_repo is hub_validator
    assert os.environ.get("XFORMERS_DISABLED") == original_xformers
    namespace = adapter.model.encoder.base_model.__class__.__module__.split(".models.")[0]
    assert not sys.modules[namespace + ".layers.attention"].XFORMERS_AVAILABLE
    manifest = json.loads((real_cache / "models/dinov2/manifest.json").read_text())
    assert manifest["source_revision"] == models.DINOV2_SOURCE_REVISION
    assert manifest["weights_sha256"] == models.DINOV2_WEIGHTS_SHA256