"""Synthetic fitting contracts, not evidence of pretrained-model competence.

No release downloads, simulator, test-split data, or GPU are required. The
world-model double is explicitly algebraic and is used only to test gradients,
freezing, candidate/action alignment, and native-target construction.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from wm_audit import fitting
from wm_audit.fitting import (
    PhysicalReadout, fit_editor, load_editor, make_editor, matched_random, save_editor,
)


STATE_SCALE = torch.tensor([512.0, 512.0, 512.0, 512.0, 1.0, 1.0])
REQUEST_SCALE = torch.tensor([32.0, 32.0, 32.0, 32.0, 1.0, 1.0])
NAMES = ("identity", "inverse", "ridge", "mlp", "rollout")


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    original = torch.get_num_threads()
    torch.set_num_threads(min(2, original))
    yield
    torch.set_num_threads(original)


def _pairs(n=160, shape=(192,), rank=3, seed=13):
    generator = torch.Generator().manual_seed(seed)
    history = torch.randn(n, 3, *shape, generator=generator)
    request = torch.randn(n, 6, generator=generator)
    request[:2] = 0
    width = history[0].numel()
    basis = torch.linalg.qr(torch.randn(width, rank, generator=generator), mode="reduced").Q.T
    target = history + (request[:, :rank] @ basis).reshape_as(history)
    return history, request * REQUEST_SCALE, target, basis


def _snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _assert_same_state(module, expected):
    actual = module.state_dict()
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


class _FrozenRolloutDouble(nn.Module):
    """TEST DOUBLE: all three history frames and candidate actions affect outputs."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))
        self.dropout = nn.Dropout(0.8)
        self.calls = []
        self.history_gradients = []

    def rollout(self, history, actions):
        self.calls.append({"batch": history.shape[0], "grad": history.requires_grad,
                           "eval": not any(m.training for m in self.modules()),
                           "frozen": not any(p.requires_grad for p in self.parameters())})
        if history.requires_grad:
            history.register_hook(lambda grad: self.history_gradients.append(grad.detach().abs().sum().item()))
        base = (history[:, 0] + 0.5 * history[:, 1] + 2 * history[:, 2]) * self.scale
        base = self.dropout(base + 0.05 * base.square())
        shifts = actions[:, 2:, :1].reshape(history.shape[0], actions.shape[1] - 2,
                                           *([1] * (history.ndim - 2)))
        return base.unsqueeze(1) + shifts


@pytest.mark.parametrize("shape,dimension", [((192,), 192), ((196, 384), 1536)])
def test_readout_fixed_exact_per_state_features(shape, dimension, monkeypatch):
    generator = torch.Generator().manual_seed(4)
    z = torch.randn(2, 3, *shape, generator=generator)
    readout = PhysicalReadout(shape)

    def forbidden(*args, **kwargs):
        raise AssertionError("A physical scorer must not use a random/learned feature projection.")

    monkeypatch.setattr(torch, "pca_lowrank", forbidden)
    actual = readout.features(z)
    if len(shape) == 1:
        expected = z
    else:
        image = z.reshape(6, 14, 14, 384).permute(0, 3, 1, 2)
        expected = F.adaptive_avg_pool2d(image, (2, 2)).reshape(2, 3, 1536)
    assert actual.shape == (2, 3, dimension)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    flat = z.reshape(6, *shape)
    assert readout.fit(flat, expected.reshape(6, dimension)[:, :6] * STATE_SCALE) is readout
    assert readout.weight.shape == (dimension, 6)
    assert readout.predict(z).shape == (2, 3, 6)
    assert readout.predict(z[0, 0]).shape == (6,)
    assert readout.predict(z[:0]).shape == (0, 3, 6)
    torch.testing.assert_close(readout.predict(z), readout.predict(flat).reshape(2, 3, 6))


@pytest.mark.parametrize("n", [5, 24])
def test_readout_ridge_matches_independent_unpenalized_bias_solution(n):
    generator = torch.Generator().manual_seed(n)
    z = torch.randn(n, 8, generator=generator, dtype=torch.float64) * 3 + 4
    y = (torch.randn(n, 6, generator=generator, dtype=torch.float64) + 2) * STATE_SCALE.double()
    strength = 0.3
    scorer = PhysicalReadout((8,), ridge=strength).double().fit(z, y)
    standardized = (z - z.mean(0)) / z.std(0, correction=0)
    augmented = torch.cat((standardized, torch.ones(n, 1, dtype=torch.float64)), 1)
    penalty = torch.eye(9, dtype=torch.float64) * strength
    penalty[-1, -1] = 0
    solution = torch.linalg.solve(augmented.T @ augmented + penalty,
                                  augmented.T @ (y / STATE_SCALE.double()))
    expected = augmented @ solution * STATE_SCALE.double()
    torch.testing.assert_close(scorer(z), expected, rtol=1e-10, atol=1e-9)
    torch.testing.assert_close(scorer.input_mean, z.mean(0))
    torch.testing.assert_close(scorer.input_std, z.std(0, correction=0))
    torch.testing.assert_close(scorer.weight, solution[:-1], rtol=1e-10, atol=1e-10)


def test_readout_constant_features_and_intercept_are_not_shrunk():
    z = torch.ones(8, 192)
    y = torch.tensor([222.0, 351.0, 128.0, 96.0, 0.25, -0.75]).expand(8, -1)
    scorer = PhysicalReadout((192,), ridge=1e8).fit(z, y)
    assert torch.equal(scorer.input_std, torch.ones(192))
    assert torch.count_nonzero(scorer.weight) == 0
    torch.testing.assert_close(scorer.predict(z * 99), y, rtol=0, atol=0)
    one = PhysicalReadout((192,)).fit(z[:1], y[:1])
    torch.testing.assert_close(one.predict(z), y, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(192,), (196, 384)])
def test_readout_safe_roundtrip_gradients_and_no_test_statistic_leakage(shape):
    generator = torch.Generator().manual_seed(26)
    z = torch.randn(12, *shape, generator=generator)
    y = torch.randn(12, 6, generator=generator) * STATE_SCALE
    scorer = PhysicalReadout(shape).fit(z, y)
    before = _snapshot(scorer)
    test_z = (z[:4].clone() + 50).requires_grad_()
    prediction = scorer.predict(test_z.reshape(2, 2, *shape).transpose(0, 1))
    prediction.square().mean().backward()
    assert test_z.grad is not None and torch.isfinite(test_z.grad).all()
    assert test_z.grad.abs().sum() > 0
    _assert_same_state(scorer, before)
    stream = io.BytesIO()
    assert all(isinstance(value, torch.Tensor) for value in scorer.state_dict().values())
    torch.save(scorer.state_dict(), stream)
    stream.seek(0)
    restored = PhysicalReadout(shape)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored.predict(test_z), scorer.predict(test_z), rtol=0, atol=0)
    # Mutating caller-owned training arrays cannot modify fitted buffers.
    z.fill_(1234)
    y.zero_()
    _assert_same_state(scorer, before)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(192,), (196, 384)])
def test_editor_shapes_exact_sham_and_safe_roundtrip(name, shape, tmp_path, monkeypatch):
    history, request, target, _ = _pairs(n=12, shape=shape)
    original = history.clone()
    model = _FrozenRolloutDouble()
    editor, log = fit_editor(name, history, request, target, rank=3, seed=19,
                             steps=2, context_dim=3, hidden_dim=12, basis_samples=8,
                             batch_size=4, model=model, actions=torch.zeros(12, 1, 3, 10),
                             anchor_y=history[:, 0].reshape(12, -1)[:, :6] * STATE_SCALE)
    assert isinstance(editor, nn.Module) and not editor.training
    assert editor.name == name and editor.latent_shape == shape
    assert editor.basis.shape == (3, 3, *shape)
    assert editor(history, request).shape == history.shape
    assert log["rank"] == 3 and log["effective_rank"] <= 3
    assert log["basis_samples"] <= 8
    assert log["final_history_mse"] >= 0
    json.dumps(log, allow_nan=False)
    # Biases are deliberately corrupted AFTER training: sham must still be exact.
    with torch.no_grad():
        if editor.mlp is not None:
            for layer in (editor.mlp[0], editor.mlp[-1]):
                layer.bias.fill_(13.75)
    double_history = history.double() + 1e-10
    output = editor(double_history, request)
    assert output.dtype == torch.float64
    assert torch.equal(output[:2], double_history[:2])
    assert torch.equal(editor(double_history, torch.zeros_like(request)), double_history)
    if name == "identity":
        assert torch.equal(output, double_history)
    torch.testing.assert_close(history, original, rtol=0, atol=0)

    path = tmp_path / "editor.pt"
    save_editor(path, editor)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"config", "state_dict"}
    assert all(isinstance(v, torch.Tensor) for part in payload.values() for v in part.values())
    assert "probe.weight" in payload["state_dict"] if name == "inverse" else "probe.weight" not in payload["state_dict"]
    original_load = torch.load

    def safe_load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    restored = load_editor(path, "cpu")
    _assert_same_state(restored, _snapshot(editor))
    torch.testing.assert_close(restored(double_history, request), output, rtol=0, atol=0)


def test_paired_ridge_generalizes_and_edits_every_history_frame():
    history, request, target, _ = _pairs(n=240)
    editor, log = fit_editor("ridge", history[:160], request[:160], target[:160],
                             rank=3, context_dim=4, basis_samples=32, seed=7)
    predicted = editor(history[160:], request[160:])
    unchanged_mse = F.mse_loss(history[160:], target[160:])
    edited_mse = F.mse_loss(predicted, target[160:])
    assert edited_mse < unchanged_mse * 1e-3
    assert log["final_history_mse"] < log["initial_history_mse"] * 1e-3
    assert ((predicted - history[160:]).square().sum((0, 2)) > 0).all()
    assert editor.effective_rank.item() == 3


def test_ridge_uses_request_context_interactions():
    generator = torch.Generator().manual_seed(123)
    n = 180
    context = torch.randn(n, 1, generator=generator)
    history = context[:, None].expand(n, 3, 8).clone()
    request = torch.zeros(n, 6)
    request[:, 0] = torch.randn(n, generator=generator) * 32
    coefficient = (request[:, 0] / 32) * (1 + 2 * context[:, 0])
    target = history + coefficient[:, None, None] / (24 ** 0.5)
    editor, _ = fit_editor("ridge", history[:120], request[:120], target[:120],
                           rank=1, context_dim=1, ridge=1e-4)
    error = F.mse_loss(editor(history[120:], request[120:]), target[120:])
    assert error < 1e-6
    zero_context = editor.context_projection.clone().zero_()
    with torch.no_grad():
        saved = editor.context_projection.clone()
        editor.context_projection.copy_(zero_context)
        ablated_error = F.mse_loss(editor(history[120:], request[120:]), target[120:])
        editor.context_projection.copy_(saved)
    assert ablated_error > error * 1000


def test_mlp_learns_nonlinear_request_residual():
    generator = torch.Generator().manual_seed(81)
    history = torch.zeros(80, 3, 8)
    r = torch.randn(80, generator=generator)
    request = torch.zeros(80, 6)
    request[:, 0] = r * 32
    target = history + r.square()[:, None, None] / (24 ** 0.5)
    ridge, _ = fit_editor("ridge", history[:60], request[:60], target[:60], rank=1, context_dim=1)
    mlp, log = fit_editor("mlp", history[:60], request[:60], target[:60], rank=1, context_dim=1,
                          steps=160, lr=1e-2, batch_size=60, hidden_dim=24)
    assert F.mse_loss(mlp(history[60:], request[60:]), target[60:]) < F.mse_loss(
        ridge(history[60:], request[60:]), target[60:]) * 0.1
    assert log["final_history_mse"] < log["losses"][0]["history_mse"] * 0.1
    assert len(log["losses"]) == 161


def test_inverse_probe_is_independent_and_uses_anchor_not_endpoint():
    generator = torch.Generator().manual_seed(302)
    history = torch.randn(180, 3, 8, generator=generator)
    history[:, -1] += 100
    anchor_y = history[:, 0, :6] * STATE_SCALE
    request = torch.randn(180, 6, generator=generator) * REQUEST_SCALE
    request[:, 3:] = 0
    delta = torch.zeros_like(history)
    delta[:, :, :3] = request[:, None, :3] / 512 * torch.tensor([1.0, 2.0, 3.0])[None, :, None]
    target = history + delta
    scorer = PhysicalReadout((8,)).fit(history[:, -1], anchor_y + 15)
    scorer_before = _snapshot(scorer)
    inverse, _ = fit_editor("inverse", history, request, target, rank=3, states_anchor=anchor_y, ridge=1e-5)
    assert inverse.probe is not scorer
    assert inverse.probe.weight.data_ptr() != scorer.weight.data_ptr()
    torch.testing.assert_close(inverse.probe.input_mean, history[:, 0].mean(0))
    assert not torch.allclose(inverse.probe.input_mean, history[:, -1].mean(0))
    edited = inverse(history, request)
    achieved = inverse.probe.predict(edited[:, 0]) - inverse.probe.predict(history[:, 0])
    torch.testing.assert_close(achieved, request, rtol=1e-4, atol=3e-4)
    assert (edited[:, 1:] - history[:, 1:]).abs().sum() > 0
    _assert_same_state(scorer, scorer_before)
    before = _snapshot(inverse)
    scorer.fit(history[:, -1], -anchor_y)
    inverse(history[:4] + 500, request[:4])
    _assert_same_state(inverse, before)


def test_basis_is_seeded_cpu_subsample_and_dino_ridge_is_compact(monkeypatch):
    history, request, target, _ = _pairs(n=20, shape=(196, 384))
    original_pca, original_ridge = torch.pca_lowrank, fitting._ridge
    calls = []

    def tracked_pca(x, *args, **kwargs):
        assert x.device.type == "cpu"
        if x.shape[1] == 3 * 196 * 384:
            assert x.shape[0] == 7
            assert kwargs["q"] <= 3 + 4
            assert kwargs["center"] is False
            calls.append(tuple(x.shape))
        return original_pca(x, *args, **kwargs)

    def tracked_ridge(x, y, *args, **kwargs):
        assert x.shape[1] == 6 * (1 + 2)
        assert y.shape[1] == 3
        return original_ridge(x, y, *args, **kwargs)

    monkeypatch.setattr(torch, "pca_lowrank", tracked_pca)
    monkeypatch.setattr(fitting, "_ridge", tracked_ridge)
    rng = torch.random.get_rng_state().clone()
    editor, log = fit_editor("ridge", history, request, target, rank=3, seed=22,
                             basis_samples=7, context_dim=2, batch_size=3)
    again, other_log = fit_editor("ridge", history, request, target, rank=3, seed=22,
                                  basis_samples=7, context_dim=2, batch_size=3)
    assert calls == [(7, 3 * 196 * 384)] * 2
    assert log == other_log and log["basis_samples"] == 7
    _assert_same_state(again, _snapshot(editor))
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_default_basis_sample_cap_and_rank_deficient_training():
    history, request, target, _ = _pairs(n=270, shape=(8,), rank=1)
    editor, log = fit_editor("ridge", history, request, target, rank=8, context_dim=2)
    assert log["basis_samples"] == 256
    assert editor.effective_rank.item() == 1
    assert torch.count_nonzero(editor.basis[1:]) == 0
    torch.testing.assert_close(editor.basis.flatten(1)[0].norm(), torch.tensor(1.0))


@pytest.mark.parametrize("name", NAMES)
def test_degenerate_sham_training_is_finite(name):
    history = torch.ones(1, 3, 8)
    editor, log = fit_editor(name, history, torch.zeros(1, 6), history.clone(), rank=8,
                             steps=1, model=_FrozenRolloutDouble(), actions=torch.zeros(1, 3, 10),
                             anchor_y=torch.ones(1, 6))
    assert log["effective_rank"] == 0 and log["final_history_mse"] == 0
    assert torch.equal(editor(history, torch.zeros(1, 6)), history)
    assert torch.equal(editor(history, torch.ones(1, 6)), history)
    json.dumps(log, allow_nan=False)


def test_editor_has_no_inference_target_access_or_mutable_training_aliases():
    history, request, target, _ = _pairs(n=32, shape=(12,))
    saved = [t.clone() for t in (history, request, target)]
    editor, _ = fit_editor("ridge", history.numpy(), request.numpy(), target.numpy(), rank=3, context_dim=2)
    for original, value in zip(saved, (history, request, target)):
        assert torch.equal(original, value)
    query_h, query_r = history[:4].clone(), request[:4].clone()
    expected = editor(query_h, query_r)
    before = _snapshot(editor)
    history.fill_(900)
    request.fill_(-200)
    target.zero_()
    editor(query_h + 100, query_r * 10)
    _assert_same_state(editor, before)
    torch.testing.assert_close(editor(query_h, query_r), expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["ridge", "mlp"])
def test_non_rollout_editors_do_not_consult_future_targets(name):
    history, request, target, _ = _pairs(n=20, shape=(8,))

    class ForbiddenModel(nn.Module):
        def rollout(self, *args):
            raise AssertionError("Immediate fitting must not access a model or future targets.")

    first, _ = fit_editor(name, history, request, target, rank=3, steps=3, context_dim=2)
    second, _ = fit_editor(name, history, request, target, rank=3, steps=3, context_dim=2,
                           model=ForbiddenModel(), actions=torch.tensor(float("nan")),
                           future_targets=torch.full((20, 1, 2, 8), 1e8))
    _assert_same_state(second, _snapshot(first))


def test_fit_existing_editor_resets_in_place_and_does_not_change_rng():
    history, request, target, _ = _pairs(n=24, shape=(8,))
    rng = torch.random.get_rng_state().clone()
    editor = make_editor("mlp", (8,), rank=3, seed=44, context_dim=2)
    fitted, first_log = fit_editor(editor, history, request, target, steps=3)
    first = _snapshot(fitted)
    refitted, second_log = fit_editor(editor, history, request, target, steps=3)
    assert fitted is editor is refitted
    assert first_log == second_log
    _assert_same_state(refitted, first)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("supplied", [False, True])
def test_rollout_frozen_model_keeps_history_gradients_and_minibatches(supplied):
    history, request, target, basis = _pairs(n=18, shape=(12,), rank=2)
    target += ((request[:, :1] / 32).square() @ basis[:1]).reshape_as(target)
    history.requires_grad_()
    target.requires_grad_()
    actions = torch.zeros(18, 2, 4, 10)
    actions[:, 1, 2:, 0] = 0.5
    model = _FrozenRolloutDouble()
    model.eval()
    with torch.no_grad():
        native = torch.stack([model.rollout(target.detach(), actions[:, k]) for k in range(2)], 1)
    model.calls.clear()
    model.train()
    model.dropout.eval()  # Deliberately heterogeneous original modes.
    modes = [module.training for module in model.modules()]
    weights = _snapshot(model)
    editor, log = fit_editor("rollout", history, request, target, rank=2, context_dim=2,
                             steps=5, lr=2e-3, seed=5, batch_size=4, model=model, actions=actions,
                             future_targets=native if supplied else None)
    assert len(log["losses"]) == 6 and log["losses"][1]["rollout_mse"] > 0
    assert model.history_gradients and max(model.history_gradients) > 0
    assert all(c["batch"] <= 4 and c["eval"] and c["frozen"] for c in model.calls)
    assert sum(not call["grad"] for call in model.calls) == (0 if supplied else 10)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.scale.requires_grad  # Restore the caller's original flag, after freezing during fit.
    assert [module.training for module in model.modules()] == modes
    _assert_same_state(model, weights)
    assert history.grad is None and target.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in editor.parameters())


def test_rollout_on_demand_and_supplied_native_targets_agree():
    history, request, target, _ = _pairs(n=16, shape=(8,))
    actions = torch.zeros(16, 4, 10)
    model = _FrozenRolloutDouble().eval().requires_grad_(False)
    with torch.no_grad():
        native = model.rollout(target, actions)
    kwargs = dict(rank=3, steps=3, batch_size=4, model=model, actions=actions, context_dim=2, seed=3)
    on_demand, _ = fit_editor("rollout", history, request, target, **kwargs)
    supplied, _ = fit_editor("rollout", history, request, target, future_targets=native, **kwargs)
    for key, value in on_demand.state_dict().items():
        torch.testing.assert_close(supplied.state_dict()[key], value, rtol=1e-6, atol=1e-7)
    assert not model.training and not model.scale.requires_grad


def test_rollout_rejects_detached_predictions_and_restores_model_flags():
    class DetachedDouble(_FrozenRolloutDouble):
        def rollout(self, history, actions):
            return super().rollout(history, actions).detach()

    history, request, target, _ = _pairs(n=12, shape=(8,))
    model = DetachedDouble()
    with pytest.raises(ValueError, match="preserve gradients"):
        fit_editor("rollout", history, request, target, rank=2, steps=1, model=model,
                   actions=torch.zeros(12, 3, 10), context_dim=2)
    assert model.training and model.dropout.training and model.scale.requires_grad
    assert model.scale.grad is None


def test_matched_random_preserves_norm_basis_span_sham_and_explicit_rng():
    history, request, target, _ = _pairs(n=100)
    editor, _ = fit_editor("ridge", history, request, target, rank=3, context_dim=3)
    learned = editor(history, request) - history
    rng = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed(77)
    random = matched_random(editor, history, request, generator)
    repeated = matched_random(editor, history, request, torch.Generator().manual_seed(77))
    assert torch.equal(random, repeated) and torch.equal(random[:2], history[:2])
    assert not torch.allclose(random, editor(history, request))
    delta = (random - history).flatten(1)
    torch.testing.assert_close(delta.norm(dim=1), learned.flatten(1).norm(dim=1), rtol=1e-5, atol=2e-6)
    basis = editor.basis.flatten(1)
    torch.testing.assert_close(delta, (delta @ basis.T) @ basis, rtol=1e-4, atol=5e-7)
    assert torch.equal(torch.random.get_rng_state(), rng)
    zero = make_editor("identity", (192,))
    assert torch.equal(matched_random(zero, history, request, generator), history)


@pytest.mark.parametrize("kwargs", [{"rank": 0}, {"rank": True}, {"seed": -1}, {"seed": 2 ** 63},
                                    {"context_dim": 0}, {"grid": 0}, {"ridge": -1}, {"ridge": float("nan")}])
def test_editor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_editor("ridge", (192,), **kwargs)


def test_readout_and_editor_shape_dtype_finite_validation():
    with pytest.raises(RuntimeError, match="fitted"):
        PhysicalReadout((192,)).predict(torch.zeros(192))
    with pytest.raises(ValueError, match="square"):
        PhysicalReadout((195, 384))
    scorer = PhysicalReadout((192,))
    with pytest.raises(ValueError, match="matching"):
        scorer.fit(torch.zeros(3, 192), torch.zeros(2, 6))
    with pytest.raises(TypeError, match="floating"):
        scorer.fit(torch.ones(3, 192, dtype=torch.int64), torch.zeros(3, 6))
    with pytest.raises(ValueError, match="finite"):
        scorer.fit(torch.full((3, 192), float("nan")), torch.zeros(3, 6))
    with pytest.raises(ValueError, match="finite"):
        scorer.fit(torch.zeros(3, 192), torch.full((3, 6), float("inf")))
    scorer.fit(torch.zeros(3, 192), torch.zeros(3, 6))
    with pytest.raises(ValueError, match="finite"):
        scorer.predict(torch.full((192,), float("inf")))
    editor = make_editor("ridge", (192,))
    with pytest.raises(ValueError):
        editor(torch.zeros(3, 2, 192), torch.zeros(3, 6))
    with pytest.raises(ValueError, match="matching"):
        editor(torch.zeros(3, 3, 192), torch.zeros(1, 6))
    with pytest.raises(TypeError, match="floating"):
        editor(torch.zeros(3, 3, 192), torch.ones(3, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        editor(torch.zeros(3, 3, 192), torch.full((3, 6), float("nan")))
    with pytest.raises(ValueError, match="name"):
        make_editor("unknown", (192,))


@pytest.mark.parametrize("failure", ["targets", "requests", "anchor_missing", "anchor_shape", "anchor_alias",
                                     "model", "action_shape", "action_range", "future_shape", "future_nan"])
def test_fit_rejects_misaligned_training_inputs(failure):
    history, request, target, _ = _pairs(n=8, shape=(8,))
    name, kwargs = "ridge", {"steps": 0}
    if failure == "targets":
        target = target[:1]
    elif failure == "requests":
        request = request[:1]
    elif failure.startswith("anchor"):
        name = "inverse"
        if failure == "anchor_shape":
            kwargs["anchor_y"] = torch.zeros(1, 6)
        elif failure == "anchor_alias":
            kwargs.update(anchor_y=torch.zeros(8, 6), states_anchor=torch.zeros(8, 6))
    else:
        name = "rollout"
        kwargs.update(model=_FrozenRolloutDouble(), actions=torch.zeros(8, 1, 3, 10))
        if failure == "model":
            kwargs["model"] = None
        elif failure == "action_shape":
            kwargs["actions"] = torch.zeros(8, 1, 3, 9)
        elif failure == "action_range":
            kwargs["actions"].fill_(2)
        elif failure == "future_shape":
            kwargs["future_targets"] = torch.zeros(8, 1, 2, 8)
        elif failure == "future_nan":
            kwargs["future_targets"] = torch.full((8, 1, 1, 8), float("nan"))
    with pytest.raises(ValueError):
        fit_editor(name, history, request, target, **kwargs)


@pytest.mark.parametrize("bad", ["version", "name", "object_config", "missing_weight"])
def test_checkpoint_rejects_unknown_or_malformed_configuration(tmp_path, bad):
    editor = make_editor("ridge", (192,), rank=2)
    path = tmp_path / "editor.pt"
    save_editor(path, editor)
    payload = torch.load(path, weights_only=True)
    if bad == "version":
        payload["config"]["version"] = torch.tensor(99)
    elif bad == "name":
        payload["config"]["name"] = torch.tensor(99)
    elif bad == "object_config":
        payload["config"]["rank"] = 2
    else:
        del payload["state_dict"]["basis"]
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_editor(path)


def test_numpy_inputs_keep_physical_units():
    generator = np.random.default_rng(42)
    z = generator.standard_normal((80, 8)).astype(np.float32)
    y = (z[:, :6] + 1) * STATE_SCALE.numpy()
    scorer = PhysicalReadout((8,), ridge=1e-5).fit(z, y)
    torch.testing.assert_close(scorer.predict(z), torch.from_numpy(y), rtol=1e-4, atol=2e-4)