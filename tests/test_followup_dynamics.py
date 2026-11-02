"""F2/F3/F4 contracts, synthetic publication tests, and cached CPU checkpoint parity.

Synthetic trajectories/H5 files are NOT research observations or qualifications.
Real tests run only from the populated pinned cache, without network/GPU access;
DINO pickle loading additionally requires WM_AUDIT_TRUST_DINOWM=1.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np
import pytest
import torch
from torch import nn

from wm_audit import diagnostics as d, followup_dynamics as f, followup_probes, models
from wm_audit.data import load_root, save_root
from wm_audit.models import DinoWMAdapter, LeWMAdapter
from wm_audit.offline_data import H5Windows
from wm_audit.provenance import digest
from wm_audit.simulator import array_digest
from wm_audit.statistics import APPEARANCES
from test_models import (SYNTHETIC_STATS, _ToyDinoWM, _ToyLeWM, _deterministic_pixels,
                         _native_actions, _native_pixels, _require_real)
from test_pipeline import _config, _synthetic_root


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def small_cpu_pool():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setenv("WM_AUDIT_PROGRESS", "0")


def _forbidden(*args, **kwargs):
    raise AssertionError("This contract test must not load a model, download, fit, or launch research.")


def _spec():
    spec = json.loads((ROOT / "configs/followup_20261101.json").read_text())
    spec.update(bootstrap_resamples=37, encode_batch_size=4, primary_horizon=2)
    return spec


def _toy(name):
    return LeWMAdapter(_ToyLeWM(), SYNTHETIC_STATS) if name == "lewm" else DinoWMAdapter(_ToyDinoWM(), SYNTHETIC_STATS)


def _trajectory(model, n=2, horizon=8):
    actual = torch.randn((n, horizon + 3, *model.latent_shape), generator=torch.Generator().manual_seed(613))
    actions = torch.linspace(-.7, .7, n * (horizon + 2) * 10).reshape(n, horizon + 2, 10)
    return actual, actions


@torch.no_grad()
def _toy_reference(model, actual, raw):
    """Independent algebraic oracle for the explicitly labeled test doubles."""
    normalized = (raw - torch.tensor([.1, -.2]).repeat(5)) / torch.tensor([.5, .25]).repeat(5)
    horizon, outputs = raw.shape[1] - 2, {}
    dino = isinstance(model, DinoWMAdapter)
    for visual in ("free", "actual"):
        for aux in (("recurrent", "reset") if dino else ("none",)):
            history, predictions = actual[:, :3].clone(), []
            prop = actual.new_full((len(actual), 3, 196, 10), .75) if dino else None
            offset = torch.arange(196, dtype=actual.dtype)[None, None, :, None] / 196
            for step in range(horizon):
                if visual == "actual":
                    history = actual[:, step:step + 3]
                # The toy's CPU cumsum accumulates before rounding to float32;
                # avoid a different low-precision reduction order in this oracle.
                total = history.double().sum(1, keepdim=True).to(history.dtype)
                if dino:
                    if aux == "reset":
                        prop = torch.full_like(prop, .75)
                    next_visual = total + prop[:, -1:, :, :1]
                    next_visual = next_visual + normalized[:, step + 2:step + 3, :1, None]
                    prop = torch.cat((prop[:, 1:], 2 * prop[:, -1:] + 1 + offset), 1)
                else:
                    shift = normalized[:, step + 2:step + 3, :1] + 10 * normalized[:, step + 2:step + 3, 1:2]
                    next_visual = total + shift
                predictions.append(next_visual)
                history = torch.cat((history[:, 1:], next_visual), 1)
            outputs[visual + "_" + aux] = torch.cat(predictions, 1)
    return outputs


@pytest.mark.parametrize("name", ["lewm", "dinowm"])
@pytest.mark.parametrize("recorded", [False, True])
def test_horizon8_full_factorial_matches_independent_toy_and_native(name, recorded):
    model = _toy(name)
    actual, actions = _trajectory(model)
    if recorded:
        actions *= 3
    before_actual, before_actions, state = actual.clone(), actions.clone(), f._state_digest(model)
    actual.requires_grad_(True)
    result = f.feedback_rollouts(model, actual, actions.numpy(), recorded_actions=recorded)
    expected = _toy_reference(model, actual, actions)
    assert set(result) == set(expected)
    native_key = "free_recurrent" if name == "dinowm" else "free_none"
    with torch.no_grad():
        native = model.rollout(actual[:, :3], actions, recorded_actions=recorded)
    torch.testing.assert_close(result[native_key], native, rtol=0, atol=0)
    for key, value in result.items():
        assert value.shape == (2, 8, *model.latent_shape)
        assert not value.requires_grad and value.grad_fn is None
        torch.testing.assert_close(value, expected[key], rtol=2e-6, atol=2e-5)
        torch.testing.assert_close(value[:, 0], native[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(actual, before_actual, rtol=0, atol=0)
    torch.testing.assert_close(actions, before_actions, rtol=0, atol=0)
    assert f._state_digest(model) == state
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def _audit_dino_inputs(monkeypatch, model, actual, raw, *, recorded=False):
    """Inspect every history position/channel of all four modes AND native parity."""
    wm, horizon = model.model, raw.shape[1] - 2
    normalized = (raw - model.action_mean) / model.action_std
    with torch.no_grad():
        zero = wm.encode_proprio(torch.zeros(len(actual), 3, 1))
        zero = zero.unsqueeze(2).expand(-1, -1, 196, -1).repeat(1, 1, 1, wm.num_proprio_repeat)
        act = wm.encode_act(normalized).unsqueeze(2).expand(-1, -1, 196, -1).repeat(1, 1, 1, wm.num_action_repeat)
    assert torch.count_nonzero(zero) > 0, "Encoded dummy proprioception must not be replaced by literal zeros."
    calls, expected_history, expected_prop, varied = [], {}, {}, []
    original_predict = wm.predict

    def checked(z):
        mode, step = divmod(len(calls), horizon)
        assert 0 <= mode < 5
        reset, oracle = mode in (1, 3), mode in (2, 3)
        prop = z[..., model.latent_shape[-1]:-wm.action_dim]
        torch.testing.assert_close(prop, zero if reset or step == 0 else expected_prop[mode])
        torch.testing.assert_close(z[..., -wm.action_dim:], act[:, step:step + 3], rtol=2e-5, atol=2e-6)
        reference = actual[:, step:step + 3] if oracle else (actual[:, :3] if step == 0 else expected_history[mode])
        torch.testing.assert_close(z[..., :384], reference, rtol=0, atol=0)
        predicted = original_predict(z)
        next_prop = predicted[:, -1:, :, 384:-wm.action_dim]
        if not reset:
            varied.append(bool((next_prop - next_prop[:, :, :1]).abs().max() > 0))
        expected_prop[mode] = torch.cat((prop[:, 1:], next_prop), 1).clone()
        expected_history[mode] = torch.cat((z[:, 1:, :, :384], predicted[:, -1:, :, :384]), 1).clone()
        calls.append((mode, step))
        return predicted

    with monkeypatch.context() as patch:
        patch.setattr(wm, "predict", checked)
        result = f.feedback_rollouts(model, actual, raw, recorded_actions=recorded)
    assert calls == [(mode, step) for mode in range(5) for step in range(horizon)]
    assert any(varied), "The fixture/released predictor exposes patch-varying auxiliary output."
    return result


def test_dino_encoded_zero_every_position_and_full_patch_auxiliary_recurrence(monkeypatch):
    model = _toy("dinowm")
    actual, actions = _trajectory(model)
    result = _audit_dino_inputs(monkeypatch, model, actual, actions)
    assert not torch.equal(result["free_recurrent"][:, 1:], result["free_reset"][:, 1:])
    assert not torch.equal(result["actual_recurrent"][:, 1:], result["actual_reset"][:, 1:])


@pytest.mark.parametrize("name", ["lewm", "dinowm"])
def test_free_rollout_future_observation_invariance_and_oracle_temporal_causality(name):
    model = _toy(name)
    actual, actions = _trajectory(model)
    original = f.feedback_rollouts(model, actual, actions)
    changed = actual.clone()
    changed[:, 6:] += 17
    perturbed = f.feedback_rollouts(model, changed, actions)
    for key in original:
        if key.startswith("free_"):
            torch.testing.assert_close(original[key], perturbed[key], rtol=0, atol=0)
        else:
            # Z[6] first enters history at s=4, predicting the fifth future step.
            torch.testing.assert_close(original[key][:, :4], perturbed[key][:, :4], rtol=0, atol=0)
            assert not torch.equal(original[key][:, 4:], perturbed[key][:, 4:])
    changed = actual.clone()
    changed[:, -1] += 1000  # Last actual target is never a prediction input.
    target_only = f.feedback_rollouts(model, changed, actions)
    for key in original:
        torch.testing.assert_close(original[key], target_only[key], rtol=0, atol=0)


@pytest.mark.parametrize("guard", ["native", "step1"])
def test_runtime_native_and_step1_parity_are_mandatory(monkeypatch, guard):
    model = _toy("dinowm")
    actual, actions = _trajectory(model, n=1, horizon=2)
    if guard == "native":
        original = model.rollout
        monkeypatch.setattr(model, "rollout", lambda *a, **kw: original(*a, **kw) + 1)
    else:
        original, calls = model.model.predict, []

        def incorrect_reset(z):
            result = original(z)
            if len(calls) == 2:  # First free/reset prediction, not the native recurrence.
                result = result.clone()
                result[..., :384] += 1
            calls.append(None)
            return result

        monkeypatch.setattr(model.model, "predict", incorrect_reset)
    with pytest.raises(AssertionError):
        f.feedback_rollouts(model, actual, actions)


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_feedback_errors_local_persistence_and_factorial_contrasts(horizon):
    actual = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=torch.float64)[None, :, None]
    offsets = {"free_recurrent": 1, "free_reset": 2, "actual_recurrent": 3, "actual_reset": 4}
    predictions = {key: actual[:, 3:] + offset for key, offset in offsets.items()}
    result = f.feedback_errors(predictions, actual, horizon)
    initial = float((actual[0, 2, 0] - actual[0, horizon + 2, 0]) ** 2)
    local = float((actual[0, horizon + 1, 0] - actual[0, horizon + 2, 0]) ** 2)
    assert result["initial_persistence"] == pytest.approx([initial])
    assert result["local_persistence"] == pytest.approx([local])
    for key, offset in offsets.items():
        np.testing.assert_array_equal(result[key], [offset ** 2])
        np.testing.assert_array_equal(result[key + "_minus_baseline"], [offset ** 2 - (initial if key.startswith("free_") else local)])
    assert result["actual_minus_free_recurrent"] == pytest.approx([8])
    assert result["actual_minus_free_reset"] == pytest.approx([12])
    assert result["free_reset_minus_recurrent"] == pytest.approx([3])
    assert result["actual_reset_minus_recurrent"] == pytest.approx([7])
    assert result["visual_aux_interaction"] == pytest.approx([4])


@pytest.mark.parametrize("bad", ["model", "rank", "short", "shape", "nan", "integer", "empty", "horizon", "raw", "boolean"])
def test_feedback_input_contracts_fail_closed(bad):
    model = _toy("lewm")
    actual, actions = _trajectory(model)
    kwargs = {}
    if bad == "model":
        model = nn.Identity()
    elif bad == "rank":
        actual = actual[0]
    elif bad == "short":
        actual = actual[:, :3]
    elif bad == "shape":
        actual = actual[..., :-1]
    elif bad == "nan":
        actual[:, -1] = float("nan")
    elif bad == "integer":
        actual = actual.long()
    elif bad == "empty":
        actual, actions = actual[:0], actions[:0]
    elif bad == "horizon":
        actions = actions[:, :-1]
    elif bad == "raw":
        actions *= 3
    else:
        kwargs["recorded_actions"] = 1
    with pytest.raises((TypeError, ValueError)):
        f.feedback_rollouts(model, actual, actions, **kwargs)


@pytest.mark.parametrize("horizon", [0, -1, 9, True, 1.0, None])
def test_feedback_metric_horizon_is_strict(horizon):
    with pytest.raises(ValueError, match="horizon"):
        f.feedback_errors({"free_none": torch.zeros(1, 8, 2)}, torch.zeros(1, 11, 2), horizon)


def test_feedback_metrics_reject_broadcast_or_nonfinite_predictions():
    actual = torch.zeros(2, 11, 3)
    for prediction in (torch.zeros(1, 8, 3), torch.full((2, 8, 3), float("inf"))):
        with pytest.raises(ValueError, match="feedback prediction"):
            f.feedback_errors({"free_none": prediction}, actual, 8)


@pytest.mark.parametrize("gain", [0., .25, 1., 2., -1.])
def test_action_effect_direction_gain_zero_effects_and_all_off_diagonal_matches(gain):
    actual = torch.tensor([[10., -3.], [11., -3.], [10., -1.], [10., -3.]], dtype=torch.float64)
    predicted = (actual[:1] + gain * (actual - actual[:1]) + torch.tensor([2., -4.])).requires_grad_()
    values, matrix = f.action_effects(predicted, actual)
    energy = np.array([.5, 2., 0.])  # The zero-effect candidate is retained.
    np.testing.assert_allclose(values["true_effect_energy"], energy)
    np.testing.assert_allclose(values["predicted_effect_energy"], gain ** 2 * energy)
    np.testing.assert_allclose(values["effect_dot"], gain * energy)
    np.testing.assert_allclose(values["effect_error"], (gain - 1) ** 2 * energy)
    np.testing.assert_allclose(values["effect_error_minus_zero"], ((gain - 1) ** 2 - 1) * energy)
    independent = np.square(predicted.detach().numpy()[:, None] - actual.numpy()[None]).mean(-1)
    np.testing.assert_allclose(matrix, independent)
    wrong = independent[~np.eye(4, dtype=bool)].reshape(4, 3).mean(1)
    np.testing.assert_allclose(values["matched_minus_all_wrong"], independent.diagonal() - wrong)
    assert len(values["effect_error"]) == 3 and len(values["matched_minus_all_wrong"]) == 4


def test_correct_assignment_is_not_confused_with_action_sensitivity_or_ties():
    actual = torch.tensor([[0.], [1.], [3.]])
    correct, _ = f.action_effects(actual, actual)
    wrong, _ = f.action_effects(actual[[1, 2, 0]], actual)
    assert correct["matched_minus_all_wrong"].mean() < 0
    assert wrong["matched_minus_all_wrong"].mean() > 0
    tied, matrix = f.action_effects(torch.ones(8, 2), torch.ones(8, 2))
    assert all(np.count_nonzero(value) == 0 for value in tied.values()) and not matrix.any()
    assert len(tied["effect_error"]) == 7 and len(tied["matched_minus_all_wrong"]) == 8


@pytest.mark.parametrize("bad", ["numpy", "rank", "singleton", "empty_features", "shape", "integer", "nan", "overflow"])
def test_bad_action_effect_banks_fail_instead_of_filtering(bad):
    actual, predicted = torch.ones(3, 2), torch.ones(3, 2)
    if bad == "numpy":
        predicted = predicted.numpy()
    elif bad == "rank":
        actual, predicted = actual[:, 0], predicted[:, 0]
    elif bad == "singleton":
        actual, predicted = actual[:1], predicted[:1]
    elif bad == "empty_features":
        actual, predicted = actual[:, :0], predicted[:, :0]
    elif bad == "shape":
        predicted = predicted[:2]
    elif bad == "integer":
        predicted = predicted.long()
    elif bad == "nan":
        predicted[1, 0] = float("nan")
    else:
        predicted = torch.full((3, 2), 1e308, dtype=torch.float64)
    with pytest.raises(ValueError):
        f.action_effects(predicted, actual)


def test_aggregate_gain_is_ratio_of_equal_root_means_with_paired_bootstrap():
    rows = [{"cluster": cluster, "appearance": "original", "horizon": 8,
             "metrics": {"effect_dot": dot, "true_effect_energy": energy, "effect_error_minus_zero": contrast}}
            for cluster, count, dot, energy, contrast in (("a", 100, 0., 1., 4.), ("b", 2, 9., 9., -2.))
            for _ in range(count)]
    spec = dict(seed=19, bootstrap_resamples=500)
    result = f.summarize(rows, spec)
    assert result == f.summarize(list(reversed(rows)), spec)
    metrics = result[0]["metrics"]
    assert metrics["effect_error_minus_zero"] == d.interval([4., -2.], seed=19, resamples=500)
    gain = metrics["aggregate_effect_gain"]
    assert gain["estimate"] == pytest.approx(.9)  # Neither row-weighted nor the mean of per-root gains (0.5).
    indices = np.random.default_rng(19).integers(2, size=(500, 2))
    ratios = np.array([0., 9.])[indices].mean(1) / np.array([1., 9.])[indices].mean(1)
    np.testing.assert_array_equal(gain["ci95"], np.quantile(ratios, [.025, .975]))
    assert gain["clusters"] == 2 and gain["undefined_resamples"] == 0


def test_gain_keeps_zero_energy_roots_and_does_not_floor_tiny_effects():
    kwargs = dict(seed=7, resamples=500)
    all_zero = f._gain_interval([0., 0.], [0., 0.], **kwargs)
    assert all_zero == {"estimate": None, "ci95": None, "clusters": 2, "undefined_resamples": 500}
    mixed = f._gain_interval([0., 2.], [0., 1.], **kwargs)
    indices = np.random.default_rng(7).integers(2, size=(500, 2))
    assert mixed == {"estimate": 2., "ci95": None, "clusters": 2,
                     "undefined_resamples": int((indices == 0).all(1).sum())}
    tiny = f._gain_interval([2e-36, 4e-36], [1e-36, 2e-36], **kwargs)
    assert tiny["estimate"] == pytest.approx(2) and tiny["ci95"] == pytest.approx([2, 2])
    assert tiny["undefined_resamples"] == 0


def test_summary_rejects_missing_contrasts_instead_of_dropping_them():
    rows = [{"cluster": "a", "appearance": "original", "horizon": 8, "metrics": {"effect_dot": 1}},
            {"cluster": "b", "appearance": "original", "horizon": 8, "metrics": {"effect_dot": 1, "extra": 2}}]
    with pytest.raises(ValueError, match="same metrics"):
        f.summarize(rows, _spec())


def test_goal_ranking_true_costs_uniform_hold_ties_and_nonadditive_contrast():
    costs = np.array([1., .2, .6, .4])
    endpoint, forecast = [9, 0, 3, 4], [4, 3, 0, 1]
    metrics, chosen = f.ranking_metrics(endpoint, forecast, costs)
    assert chosen == {"endpoint": 1, "forecast": 2}
    assert metrics["endpoint_regret"] == 0 and metrics["forecast_regret"] == pytest.approx(.4)
    assert metrics["uniform_regret"] == pytest.approx(.35) and metrics["hold_regret"] == pytest.approx(.8)
    assert metrics["best_cost"] == .2 and metrics["cost_range"] == .8 and metrics["best_coverage"] == .8
    assert metrics["no_positive_coverage"] == 0
    assert metrics["forecast_minus_endpoint_regret"] == pytest.approx(.4)
    reversed_metrics, _ = f.ranking_metrics(forecast, endpoint, costs)
    assert reversed_metrics["forecast_minus_endpoint_regret"] == pytest.approx(-.4)
    tied, chosen = f.ranking_metrics([0, 0, 1, 1], [1, 0, 0, 1], costs)
    assert chosen == {"endpoint": 0, "forecast": 1}  # Not the lower true cost among latent ties.
    assert tied["endpoint_regret"] == pytest.approx(.8)


def test_no_coverage_zero_regret_is_explicitly_not_success():
    metrics, chosen = f.ranking_metrics(np.zeros(8), np.arange(8)[::-1], np.ones(8))
    assert chosen == {"endpoint": 0, "forecast": 7}
    assert metrics["no_positive_coverage"] == 1 and metrics["best_cost"] == 1
    assert metrics["best_coverage"] == metrics["cost_range"] == 0
    assert all(value == 0 for key, value in metrics.items() if "regret" in key)


@pytest.mark.parametrize("scores,costs", [([0], [1]), ([0, 1], [0]), ([[0, 1]], [[0, 1]]),
                                        ([0, float("nan")], [0, 1]), ([0, 1], [0, float("inf")]),
                                        ([0, 1], [-.1, 1]), ([0, 1], [0, 1.1])])
def test_goal_ranking_rejects_invalid_banks(scores, costs):
    with pytest.raises(ValueError):
        f.ranking_metrics(scores, scores, costs)


class _PayloadEncoder:
    """TEST ONLY: decode synthetic six-float payloads, never physical RGB inference."""
    def encode(self, pixels):
        assert not torch.is_grad_enabled() and not any(p.requires_grad for p in self.parameters())
        assert not any(module.training for module in self.modules())
        n, t = pixels.shape[:2]
        flat = pixels.reshape(n * t, -1)
        state = np.ascontiguousarray(flat[:, :24]).view(np.float32).reshape(n, t, 6)
        z = torch.from_numpy(state.copy()) / torch.tensor([512, 512, 512, 512, 1, 1])
        z = z + torch.from_numpy(flat[:, -4].copy()).reshape(n, t, 1) * .01
        z = z.repeat(1, 1, self.latent_shape[-1] // 6)
        if len(self.latent_shape) == 2:
            z = z.unsqueeze(2).expand(-1, -1, 196, -1) + torch.arange(196)[None, None, :, None] * .001
        self.encode_calls.append(n * t)
        return z


class _PayloadLeWM(_PayloadEncoder, LeWMAdapter):
    pass


class _PayloadDino(_PayloadEncoder, DinoWMAdapter):
    pass


def _payload_model(name="lewm"):
    model = (_PayloadLeWM(_ToyLeWM(), SYNTHETIC_STATS) if name == "lewm"
             else _PayloadDino(_ToyDinoWM(), SYNTHETIC_STATS))
    model.encode_calls = []
    return model


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-f234")
    config = _config()
    config["counts"].update(editor=0, scorer=1, validation=1, pilot=2)
    config["bootstrap_resamples"] = 3  # Uncertainty must instead use follow-up spec.
    for index in (1, 2):
        root = _synthetic_root("pilot", index, config)
        root.actions[0, 2:] = 0
        root.metadata["audit"]["actions_sha256"] = array_digest(root.actions)
        if index == 2:
            root.coverage_future[:] = 0
            root.costs[:] = 1
        save_root(directory / "pilot" / (root.metadata["root_id"] + ".npz"), root)
    manifest = {"config": config, "files": {"pilot": {p.relative_to(directory).as_posix(): digest(p)
                                                     for p in sorted((directory / "pilot").glob("*.npz"))}}}
    (directory / "manifest-development.json").write_text(json.dumps(manifest))
    return config, directory


@pytest.mark.parametrize("name", ["lewm", "dinowm"])
def test_generated_workflow_full_roster_unbatched_goals_and_publication(paired, tmp_path, monkeypatch, name):
    config, data = paired
    spec, output, model = _spec(), tmp_path / "out", _payload_model(name)
    inputs, state = deepcopy((config, spec)), f._state_digest(model)
    goal_cost, scored_shapes = model.goal_cost, []

    def checked_cost(endpoints, goal):
        assert endpoints.shape == goal.shape == (config["candidates"], *model.latent_shape)
        scored_shapes.append(tuple(goal.shape))
        return goal_cost(endpoints, goal)

    monkeypatch.setattr(model, "goal_cost", checked_cost)
    result = f.dynamics_followup(config, data, output, model, spec=spec)
    report = json.loads((output / "report.json").read_text())
    assert result["kind"] == "followup_dynamics" and result["output"] == str(output / "report.json")
    assert (config, spec) == inputs and f._state_digest(model) == state
    assert report["parameter_buffer_sha256"] == state and report["spec"] == spec
    assert report["confirmatory"] is False and not report["protocol"]["readout_used"]
    assert not report["protocol"]["world_model_trained"]
    assert len(report["records"]) == 2 * 3 * 2 * 3 and len(report["summary"]) == 3 * 3
    assert len(scored_shapes) == 2 * len(report["records"])
    assert sum(model.encode_calls) == 2 * 3 * (6 + 1 + 2 * 3 * 3)
    assert max(model.encode_calls) == spec["encode_batch_size"]
    assert report["artifacts"] == {"index.html": digest(output / "index.html")}
    assert report["code_sha256"]["followup_dynamics.py"] == digest(Path(f.__file__))
    assert len(report["input_sha256"]) == 3
    assert "aggregate_effect_gain" in (output / "index.html").read_text()
    roots = {p.stem: load_root(p) for p in (data / "pilot").glob("*.npz")}
    for row in report["records"]:
        k, h = config["candidates"], row["horizon"]
        root = roots[row["cluster"]]
        np.testing.assert_array_equal(row["true_costs"], 1 - root.coverage_future[row["branch"], :, h - 1])
        ranking, chosen = f.ranking_metrics(row["endpoint_scores"], row["forecast_scores"], row["true_costs"])
        assert row["chosen"] == chosen
        assert all(row["metrics"][key] == value for key, value in ranking.items())
        for key, values in row["candidates"].items():
            assert len(values) == (k - 1 if key.startswith("effect_") or key.endswith("effect_energy") else k)
            assert row["metrics"][key] == pytest.approx(np.mean(values))
        matrix = np.array(row["prediction_to_actual_mse"])
        expected = matrix.diagonal() - matrix[~np.eye(k, dtype=bool)].reshape(k, k - 1).mean(1)
        np.testing.assert_allclose(row["candidates"]["matched_minus_all_wrong"], expected)
        assert all(np.isfinite(value) for value in row["metrics"].values())
        if root.metadata["root_index"] == 2:
            assert row["metrics"]["no_positive_coverage"] == 1
            assert row["metrics"]["endpoint_regret"] == row["metrics"]["forecast_regret"] == 0
    assert report["summary"] == f.summarize(report["records"], spec)
    assert all(metric["clusters"] == 2 for row in report["summary"] for metric in row["metrics"].values())


@pytest.mark.parametrize("kind", ["parameter", "persistent", "nonpersistent"])
def test_generated_model_mutation_blocks_completion_and_restores_flags(paired, tmp_path, monkeypatch, kind):
    config, data = paired
    model, output = _payload_model(), tmp_path / "out"
    model.register_buffer("audit_buffer", torch.tensor(4.), persistent=kind != "nonpersistent")
    model.model.scale.requires_grad_(True)
    model.model.dropout.train()
    original, mutated = model.encode, []

    def mutate(pixels):
        if not mutated:
            (model.model.scale if kind == "parameter" else model.audit_buffer).add_(1)
            mutated.append(True)
        return original(pixels)

    monkeypatch.setattr(model, "encode", mutate)
    with pytest.raises(ValueError, match="parameters or buffers changed"):
        f.dynamics_followup(config, data, output, model, spec=_spec())
    assert model.model.scale.requires_grad and model.model.dropout.training
    assert model.model.scale.grad is None
    assert not (output / "report.json").exists() and not (output / "index.html").exists()


def test_state_digest_includes_nonpersistent_aliases_scalar_and_bfloat16_bytes():
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.tensor(1.)))
    model.register_buffer("mask", torch.ones(2, 3), persistent=False)
    model.register_buffer("counter", torch.tensor(3))
    model.register_buffer("bf16", torch.tensor([.5, 2.], dtype=torch.bfloat16))
    assert "mask" not in model.state_dict()
    initial = f._state_digest(model)
    with torch.no_grad():
        for tensor in (model.weight, model.mask, model.counter, model.bf16):
            before = tensor.clone()
            tensor.add_(1)
            assert f._state_digest(model) != initial
            tensor.copy_(before)
            assert f._state_digest(model) == initial
    model.register_buffer("mask_alias", model.mask, persistent=False)
    assert f._state_digest(model) != initial


def test_nonhold_candidate_zero_and_partial_output_are_not_silently_accepted(paired, tmp_path, monkeypatch):
    config, data = paired
    original = d._load

    def bad_hold(*args):
        root = original(*args)
        root.actions[0, 2:] = .1
        return root

    monkeypatch.setattr(d, "_load", bad_hold)
    with pytest.raises(ValueError, match="hold control"):
        f.dynamics_followup(config, data, tmp_path / "bad", _payload_model(), spec=_spec())
    assert not (tmp_path / "bad/report.json").exists()
    output = tmp_path / "partial"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    monkeypatch.setattr(d, "_roster", _forbidden)
    with pytest.raises(FileExistsError):
        f.dynamics_followup(config, data, output, _payload_model(), spec=_spec())
    assert (output / "keep.txt").read_text() == "preserve"


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    import h5py
    directory = tmp_path_factory.mktemp("synthetic-recorded-f2")
    config = _config()
    config["counts"].update(scorer=1, validation=1, pilot=2)
    source = directory / "original.h5"
    count, length = 128, 29
    actions = np.tile([1.25, -1.75], (count * length, 1)).astype(np.float32)
    actions[length - 1::length] = 900  # Finite terminal labels must NEVER become controls.
    with h5py.File(source, "w") as handle:
        # Sparse fill-value image dataset: a real H5 schema, not a gigabyte fixture.
        handle.create_dataset("pixels", shape=(count * length, 224, 224, 3), dtype=np.uint8, fillvalue=0)
        handle["action"] = actions
        handle["ep_offset"] = np.arange(count, dtype=np.int64) * length
        handle["ep_len"] = np.full(count, length, dtype=np.int64)
    prior = {"kind": "frozen_prediction", "source": "recorded_h5_original_only", "config": config,
             "input_sha256": {str(source): digest(source)}, "recorded_h5_info": {"path": str(source)},
             # Deliberately non-sorted: reuse exact order/starts, not a newly sampled roster.
             "windows": [{"episode": e, "start": e % 4} for e in reversed(range(128))]}
    receipt = directory / "prior.json"
    receipt.write_text(json.dumps(prior))
    relocated = directory / "relocated/byte-identical-renamed.h5"
    relocated.parent.mkdir()
    shutil.copyfile(source, relocated)
    source.unlink()  # The old absolute path really is no longer available.
    return config, relocated, receipt


def test_recorded_relocated_authenticated_h5_exact_128_roster_and_raw_actions(recorded, tmp_path, monkeypatch):
    config, h5, prior_report = recorded
    prior = json.loads(prior_report.read_text())
    model, seen, original_window, original_predict = _payload_model(), [], H5Windows.window, f.feedback_rollouts

    def window(loader, episode, start):
        seen.append({"episode": episode, "start": start})
        value = original_window(loader, episode, start)
        np.testing.assert_array_equal(value["frame_indices"], start + np.arange(6) * 5)
        np.testing.assert_array_equal(value["actions"], np.tile([1.25, -1.75], (5, 5)))
        assert loader.action_schema == "recorded"
        return value

    def predict(adapter, actual, raw, **kwargs):
        assert kwargs == {"recorded_actions": True}
        assert raw.shape == (1, 5, 10) and raw.max() == 1.25 and raw.min() == -1.75
        result = original_predict(adapter, actual, raw, **kwargs)
        oracle = _toy_reference(adapter, actual, torch.as_tensor(raw))
        for key in oracle:
            torch.testing.assert_close(result[key], oracle[key])
        return result

    monkeypatch.setattr(H5Windows, "window", window)
    monkeypatch.setattr(f, "feedback_rollouts", predict)
    output = tmp_path / "recorded"
    state, h5_hash, prior_hash = f._state_digest(model), digest(h5), digest(prior_report)
    result = f.recorded_feedback(config, h5, prior_report, output, model, spec=_spec())
    report = json.loads((output / "report.json").read_text())
    assert result["kind"] == "followup_recorded_feedback"
    assert seen == prior["windows"] == report["windows"]
    assert len(report["records"]) == 128 * 3 and len(report["summary"]) == 3
    assert {row["cluster"] for row in report["records"]} == {str(e) for e in range(128)}
    assert all(row["appearance"] == "original" for row in report["records"])
    assert all(metric["clusters"] == 128 for row in report["summary"] for metric in row["metrics"].values())
    assert all("aggregate_effect_gain" not in row["metrics"] for row in report["summary"])
    assert report["summary"] == f.summarize(report["records"], _spec())
    assert report["input_sha256"] == {str(h5): h5_hash, str(prior_report): prior_hash}
    assert report["recorded_input"] == {"prior_path": prior["recorded_h5_info"]["path"], "path": str(h5), "sha256": h5_hash}
    assert report["parameter_buffer_sha256"] == state == f._state_digest(model)
    assert digest(h5) == h5_hash and digest(prior_report) == prior_hash
    assert report["artifacts"] == {p.name: digest(p) for p in output.iterdir() if p.name != "report.json"}
    assert json.loads((output / "windows.json").read_text()) == {"windows": seen, "prior_report_sha256": prior_hash}


@pytest.mark.parametrize("bad", ["kind", "source", "config", "hashes", "ambiguous", "missing_path", "digest", "short",
                                  "duplicate", "bool_episode", "float_start", "negative_start", "extra_key", "not_list"])
def test_prior_authentication_and_roster_fail_closed(recorded, bad):
    config, _, receipt = recorded
    prior = json.loads(receipt.read_text())
    if bad in ("kind", "source"):
        prior[bad] = "other"
    elif bad == "config":
        prior["config"]["seed"] += 1
    elif bad == "hashes":
        prior["input_sha256"] = []
    elif bad == "ambiguous":
        prior.pop("recorded_h5_info")
        prior["input_sha256"]["unrelated.h5"] = "0" * 64
    elif bad == "missing_path":
        prior["recorded_h5_info"]["path"] = "missing.h5"
    elif bad == "digest":
        prior["input_sha256"][prior["recorded_h5_info"]["path"]] = "z" * 64
    elif bad == "short":
        prior["windows"].pop()
    elif bad == "duplicate":
        prior["windows"][1] = prior["windows"][0]
    elif bad == "bool_episode":
        prior["windows"][0]["episode"] = True
    elif bad == "float_start":
        prior["windows"][0]["start"] = 1.
    elif bad == "negative_start":
        prior["windows"][0]["start"] = -1
    elif bad == "extra_key":
        prior["windows"][0]["selected_for_outcome"] = True
    else:
        prior["windows"] = {}
    with pytest.raises(ValueError):
        f._recorded_reference(prior, config)


def test_legacy_single_input_receipt_can_relocate_but_does_not_guess_among_hashes(recorded):
    config, _, receipt = recorded
    prior = json.loads(receipt.read_text())
    expected = f._recorded_reference(prior, config)
    del prior["recorded_h5_info"]
    assert f._recorded_reference(prior, config) == expected
    prior["input_sha256"]["other"] = expected[1]
    with pytest.raises(ValueError, match="unambiguously"):
        f._recorded_reference(prior, config)


@pytest.mark.parametrize("race", [False, True])
def test_h5_wrong_bytes_or_mutation_during_hash_prevent_any_output(recorded, tmp_path, monkeypatch, race):
    config, source, prior = recorded
    h5 = tmp_path / "input.h5"
    shutil.copyfile(source, h5)
    if race:
        original = f.digest

        def mutate_after_hash(path):
            value = original(path)
            if path == h5:
                with h5.open("ab") as stream:
                    stream.write(b"changed while authenticating")
            return value

        monkeypatch.setattr(f, "digest", mutate_after_hash)
    else:
        with h5.open("ab") as stream:
            stream.write(b"different bytes")
    monkeypatch.setattr(H5Windows, "__init__", _forbidden)
    with pytest.raises(ValueError, match="prior authenticated H5"):
        f.recorded_feedback(config, h5, prior, tmp_path / "out", _payload_model(), spec=_spec())
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("what", ["h5", "h5_hidden_stamp", "prior", "nonpersistent_buffer"])
def test_recorded_post_read_mutations_block_completion(recorded, tmp_path, monkeypatch, what):
    config, source, receipt = recorded
    h5, prior, output = tmp_path / "input.h5", tmp_path / "prior.json", tmp_path / "out"
    shutil.copyfile(source, h5)
    shutil.copyfile(receipt, prior)
    model = _payload_model()
    model.register_buffer("ephemeral", torch.tensor(1.), persistent=False)
    original, calls = f.feedback_errors, []
    if what == "h5_hidden_stamp":
        stamp = f._file_stamp(h5)
        monkeypatch.setattr(f, "_file_stamp", lambda path: stamp)

    def after_last_read(*args):
        result = original(*args)
        calls.append(None)
        if len(calls) == 128 * config["horizon"]:
            if what.startswith("h5"):
                before = h5.stat()
                with h5.open("ab") as stream:
                    stream.write(b"changed after last read")
                os.utime(h5, ns=(before.st_atime_ns, before.st_mtime_ns))
            elif what == "prior":
                prior.write_text(prior.read_text() + " ")
            else:
                model.ephemeral.add_(1)
        return result

    monkeypatch.setattr(f, "feedback_errors", after_last_read)
    with pytest.raises(ValueError, match="changed during follow-up"):
        f.recorded_feedback(config, h5, prior, output, model, spec=_spec())
    assert len(calls) == 128 * config["horizon"]
    assert (output / "windows.json").exists() and not (output / "report.json").exists()
    assert not (output / "index.html").exists()


def test_recorded_out_of_bounds_start_is_not_replaced_or_dropped(recorded, tmp_path):
    config, h5, receipt = recorded
    prior = json.loads(receipt.read_text())
    prior["windows"][0]["start"] = 4  # Consumes a terminal transition in length-29, span-25 episode.
    report = tmp_path / "prior.json"
    report.write_text(json.dumps(prior))
    with pytest.raises(IndexError, match="terminal action"):
        f.recorded_feedback(config, h5, report, tmp_path / "out", _payload_model(), spec=_spec())
    assert not (tmp_path / "out/report.json").exists()


@pytest.mark.parametrize("stage", ["probes", "dynamics", "recorded-feedback"])
def test_cli_dispatch_and_recorded_without_data(tmp_path, monkeypatch, capsys, stage):
    config, spec = tmp_path / "config.json", tmp_path / "spec.json"
    config.write_text('{"fixture": "config"}')
    spec.write_text('{"fixture": "spec"}')
    model, calls = object(), []

    def load(name, cache, device, **kwargs):
        assert name == "lewm" and device == "cpu" and kwargs == {"trust_checkpoint": False}
        return model

    def dispatched(*args, **kwargs):
        assert args[0] == {"fixture": "config"} and args[-1] is model
        assert kwargs == {"spec": {"fixture": "spec"}}
        assert args[1:-1] == ((tmp_path / "h5", tmp_path / "prior", tmp_path / "out") if stage == "recorded-feedback"
                             else (tmp_path / "data", tmp_path / "out"))
        calls.append(stage)
        return {"fixture_stage": stage}

    monkeypatch.setattr(models, "load_model", load)
    monkeypatch.setattr(followup_probes, "probe_followup", dispatched if stage == "probes" else _forbidden)
    monkeypatch.setattr(f, "dynamics_followup", dispatched if stage == "dynamics" else _forbidden)
    monkeypatch.setattr(f, "recorded_feedback", dispatched if stage == "recorded-feedback" else _forbidden)
    args = ["followup", "--config", str(config), "--spec", str(spec), "--cache", str(tmp_path / "cache"),
            "--output", str(tmp_path / "out"), "--model", "lewm", "--device", "cpu", "--stage", stage]
    args += (["--h5", str(tmp_path / "h5"), "--prior-report", str(tmp_path / "prior")] if stage == "recorded-feedback"
             else ["--data", str(tmp_path / "data")])
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(ROOT / "scripts/run_followup.py"), run_name="__main__")
    assert calls == [stage] and json.loads(capsys.readouterr().out) == {"fixture_stage": stage}
    assert not (tmp_path / "out").exists()  # Stub dispatch; no actual run was launched.


@pytest.mark.parametrize("stage,extra,message", [("probes", [], "require --data"), ("dynamics", [], "require --data"),
                                               ("recorded-feedback", [], "--h5 and --prior-report"),
                                               ("recorded-feedback", ["--h5", "unused"], "--h5 and --prior-report")])
def test_cli_rejects_missing_stage_inputs_before_model_or_file_io(tmp_path, monkeypatch, capsys, stage, extra, message):
    monkeypatch.setattr(models, "load_model", _forbidden)
    monkeypatch.setattr(sys, "argv", ["followup", "--config", "absent", "--spec", "absent", "--cache", "absent",
                                    "--output", str(tmp_path / "out"), "--model", "lewm", "--device", "cpu",
                                    "--stage", stage, *extra])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "scripts/run_followup.py"), run_name="__main__")
    assert exc.value.code == 2 and message in capsys.readouterr().err


@pytest.fixture(scope="module", params=["lewm", "dinowm"])
def real_checkpoint(request):
    name = request.param
    cache = Path(os.environ.get("WM_AUDIT_MODEL_CACHE", ROOT / "cache"))
    if name == "dinowm" and os.environ.get("WM_AUDIT_TRUST_DINOWM") != "1":
        pytest.skip("Pinned DINO pickle requires explicit WM_AUDIT_TRUST_DINOWM=1.")
    required = [models.STATS_PATH, "sources/point/dinowm_swm/conf/env/swm_pusht.yaml"]
    if name == "lewm":
        required += ["sources/lewm/jepa.py", "sources/lewm/module.py", "models/lewm/config.json", "models/lewm/weights.pt"]
    else:
        required += ["sources/point/third_party/dino_wm/models/visual_world_model.py",
                     "models/dinowm/dino-wm/hydra.yaml", "models/dinowm/dino-wm/checkpoints/model_latest.pth",
                     "sources/dinov2/dinov2/hub/backbones.py", "models/dinov2/dinov2_vits14_pretrain.pth"]
    _require_real(cache, required)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(models, "urlopen", _forbidden)
        patch.setattr(torch.hub, "load", _forbidden)
        model = models.load_model(name, cache, device="cpu", trust_checkpoint=name == "dinowm")
    assert all(t.device.type == "cpu" for t in (*model.parameters(), *model.buffers()))
    return model, cache


@pytest.mark.parametrize("recorded", [False, True])
def test_real_checkpoint_horizon8_native_full_controls_one_step_and_future_invariance(real_checkpoint, monkeypatch, recorded):
    model, cache = real_checkpoint
    pixels = _deterministic_pixels(n=1, t=11)
    actions = torch.linspace(-.6, .6, 100).reshape(1, 10, 10) * (3 if recorded else 1)
    state, before_actions = f._state_digest(model), actions.clone()
    with torch.no_grad():
        actual = model.encode(pixels)
        if isinstance(model, DinoWMAdapter):
            result = _audit_dino_inputs(monkeypatch, model, actual, actions, recorded=recorded)
            observation = {"visual": _native_pixels(pixels[:, :3], (.5,) * 3, (.5,) * 3), "proprio": torch.zeros(1, 3, 1)}
            native = model.model.rollout(observation, _native_actions(cache, actions))[0]["visual"][:, 3:]
            native_key, oracle_key = "free_recurrent", "actual_reset"
            wm = model.model
            normalized = (actions - model.action_mean) / model.action_std
            # Native recurrence encodes the initial three controls together, then
            # one at a time. Conv1d's float32 rounding depends on this batch shape.
            encoded_actions = torch.cat([wm.encode_act(normalized[:, :3])] + [
                wm.encode_act(normalized[:, s:s + 1]) for s in range(3, 10)], 1)
            encoded_actions = encoded_actions.unsqueeze(2).expand(-1, -1, 196, -1).repeat(1, 1, 1, wm.num_action_repeat)
            zero = wm.encode_proprio(actual.new_zeros(1, 3, 1))
            zero = zero.unsqueeze(2).expand(-1, -1, 196, -1).repeat(1, 1, 1, wm.num_proprio_repeat)
        else:
            result = f.feedback_rollouts(model, actual, actions, recorded_actions=recorded)
            image = _native_pixels(pixels[:, :3], (.485, .456, .406), (.229, .224, .225))
            native = model.model.rollout({"pixels": image[:, None]}, _native_actions(cache, actions)[:, None])["predicted_emb"][:, 0, 3:]
            native_key, oracle_key = "free_none", "actual_none"
        torch.testing.assert_close(result[native_key], native, rtol=2e-5, atol=2e-6)
        for prediction in result.values():
            assert prediction.shape == (1, 8, *model.latent_shape)
            torch.testing.assert_close(prediction[:, 0], native[:, 0], rtol=2e-5, atol=2e-6)
        for step in range(8):
            # Public native one-step calls independently start from real history,
            # encode dummy proprio anew, and use only A[s:s+3].
            one = model.rollout(actual[:, step:step + 3], actions[:, step:step + 3], recorded_actions=recorded)
            # Allow the separately batched real action encoder's FP32 roundoff,
            # NOT a different recurrence. With identical encoded controls the
            # direct three-frame real predictor below must match bit-for-bit.
            atol = 1e-5 if isinstance(model, DinoWMAdapter) else 2e-6
            torch.testing.assert_close(result[oracle_key][:, step:step + 1], one, rtol=2e-5, atol=atol)
            if isinstance(model, DinoWMAdapter):
                history = torch.cat((actual[:, step:step + 3], zero, encoded_actions[:, step:step + 3]), -1)
                direct = wm.predict(history)[:, -1:, :, :384]
                torch.testing.assert_close(result[oracle_key][:, step:step + 1], direct, rtol=0, atol=0)
        changed = actual.clone()
        changed[:, 3:] += .37
        perturbed = f.feedback_rollouts(model, changed, actions, recorded_actions=recorded)
        for key, prediction in result.items():
            torch.testing.assert_close(prediction[:, 0], perturbed[key][:, 0], rtol=0, atol=0)
            if key.startswith("free_"):
                torch.testing.assert_close(prediction, perturbed[key], rtol=0, atol=0)
        scores = f._cost(model, result[native_key][:, -1], actual[0, -1])
        np.testing.assert_array_equal(scores, model.goal_cost(result[native_key][:, -1], actual[:, -1]).double().numpy())
    torch.testing.assert_close(actions, before_actions, rtol=0, atol=0)
    assert f._state_digest(model) == state
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not any(module.training for module in model.modules())


def test_real_nonpersistent_buffers_are_covered_by_digest(real_checkpoint):
    model, _ = real_checkpoint
    buffers = [(name, value) for name, value in model.named_buffers() if name not in model.state_dict()]
    if isinstance(model, DinoWMAdapter):
        assert len(buffers) == 6 and all(value.shape == (1, 1, 588, 588) for _, value in buffers)
    if not buffers:
        # LeWM has no nonpersistent buffers; its persistent buffers remain covered.
        buffers = list(model.named_buffers())[:1]
    before = f._state_digest(model)
    with torch.no_grad():
        for _, buffer in buffers:
            original = buffer.clone()
            try:
                buffer.flatten()[0] += 1
                assert f._state_digest(model) != before
            finally:
                buffer.copy_(original)
    assert f._state_digest(model) == before