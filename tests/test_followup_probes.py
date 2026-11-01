"""Synthetic F1 contracts only: no real encoder, simulator run, GPU, or downloads."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from wm_audit import diagnostics as d, followup_probes as f, pipeline
from wm_audit.data import save_root
from wm_audit.simulator import STATE_FIELDS
from wm_audit.statistics import APPEARANCES
from test_pipeline import SyntheticAdapter, _config, _pixels, _synthetic_root


def _spec():
    value = json.loads((Path(__file__).parents[1] / "configs/followup_20261101.json").read_text())
    value.update(mlp_checkpoints=[1, 2], bootstrap_resamples=20, encode_batch_size=3,
                 mlp_batch_size=7, primary_horizon=2)
    return value


@pytest.fixture(autouse=True)
def quiet_and_small(monkeypatch):
    monkeypatch.setenv("WM_AUDIT_PROGRESS", "0")
    threads, rng = torch.get_num_threads(), torch.random.get_rng_state().clone()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)
    torch.random.set_rng_state(rng)


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    config = _config()
    config.update(bootstrap_resamples=3)  # Follow-up uncertainty MUST use spec, not this.
    config["counts"].update(editor=0, scorer=4, validation=2, pilot=2)
    directory = tmp_path_factory.mktemp("f1-synthetic") / "data"
    for split, indices in (("scorer", (0, 1, 2, 3)), ("validation", (4, 5)), ("pilot", (6, 7))):
        for index in indices:
            root = _synthetic_root(split, index, config)
            save_root(directory / split / f"{root.metadata['root_id']}.npz", root)
    manifest = {"config": config,
                "files": {split: {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((directory / split).glob("*.npz"))}
                          for split in config["counts"]}}
    (directory / "manifest-development.json").write_text(json.dumps(manifest))
    return config, directory


def _forbidden(*args, **kwargs):
    raise AssertionError("No world-model rollout, goal scoring, or pilot-driven fit is permitted.")


def _read(path):
    return json.loads(path.read_text())


def _hashes(path):
    return {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in path.iterdir() if item.is_file()}


@pytest.mark.parametrize("shape", [(6,), (49, 6)])
def test_full_f1_workflow_publication_freezing_seeds_sampling_and_artifacts(paired, tmp_path, monkeypatch, shape):
    config, data = paired
    spec, output = _spec(), tmp_path / "probe"
    originals = deepcopy((config, spec))
    model = SyntheticAdapter(shape)
    model.register_buffer("test_buffer", torch.tensor([4.0]))
    model.register_parameter("already_frozen", torch.nn.Parameter(torch.tensor(2.0), requires_grad=False))
    model.add_module("mixed_mode", torch.nn.Identity().eval())
    monkeypatch.setattr(model, "rollout", _forbidden)
    monkeypatch.setattr(model, "goal_cost", _forbidden)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    flags = [(module, module.training) for module in model.modules()]
    requires = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    opened, samples, steps, before_pilot = [], [], [], []
    original_load, image_rows, adam_step = pipeline.load_root, d._image_rows, torch.optim.AdamW.step

    def tracked_load(path):
        split = Path(path).parent.name
        opened.append(split)
        if split == "pilot":
            assert (output / "selection.json").is_file()
            assert (output / "probe-weights.pt").is_file()
            assert (output / "training-loss.json").is_file()
            assert not (output / "report.json").exists()
            if not before_pilot:
                before_pilot.append(_hashes(output))
                saved = torch.load(output / "probe-weights.pt", map_location="cpu", weights_only=True)
                assert all(isinstance(v, torch.Tensor) and not v.requires_grad for v in saved.values())
        return original_load(path)

    def tracked_rows(root, appearance, cfg, all_candidates):
        value = image_rows(root, appearance, cfg, all_candidates)
        pixels, labels, tags = value
        split = root.metadata["split"]
        samples.append((split, appearance, all_candidates, len(labels)))
        assert not torch.is_grad_enabled()
        if split in ("scorer", "validation"):
            assert appearance == 0 and all_candidates is False
            markers = pixels.reshape(len(pixels), -1)[:, -4:]
            assert set(markers[tags == "future", -1]) == {root.metadata["root_index"] % cfg["candidates"]}
        else:
            assert all_candidates is True
        return value

    def tracked_step(optimizer, *args, **kwargs):
        assert torch.is_grad_enabled()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        assert not {id(p) for p in parameters} & {id(p) for p in model.parameters()}
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert not any(module.training for module in model.modules())
        assert optimizer.defaults["lr"] == spec["mlp_learning_rate"]
        assert optimizer.defaults["weight_decay"] in spec["mlp_weight_decays"]
        assert [tuple(p.shape) for p in parameters] == [(64, parameters[0].shape[1]), (64,), (6, 64), (6,)]
        before = [p.detach().clone() for p in parameters]
        result = adam_step(optimizer, *args, **kwargs)
        steps.append(any(not torch.equal(p, old) for p, old in zip(parameters, before)))
        return result

    monkeypatch.setattr(pipeline, "load_root", tracked_load)
    monkeypatch.setattr(d, "_image_rows", tracked_rows)
    monkeypatch.setattr(torch.optim.AdamW, "step", tracked_step)
    rng, numpy_rng, python_rng = torch.random.get_rng_state().clone(), np.random.get_state(), random.getstate()
    # The public no_grad decorator must not disable learning by the separate MLP.
    with torch.no_grad():
        reference = f.probe_followup(config, data, output, model, spec=spec)
    assert torch.equal(torch.random.get_rng_state(), rng)
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    assert random.getstate() == python_rng
    assert (config, spec) == originals
    assert all(module.training == training for module, training in flags)
    assert all(parameter.requires_grad == expected and parameter.grad is None for parameter, expected in requires)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert steps and all(steps)
    assert len(steps) == 3 * 2 * 2 * math.ceil(40 / spec["mlp_batch_size"])
    assert model.encode_calls and not model.rollout_calls
    assert all(c["frozen"] and not c["grad_enabled"] and c["frames"] <= spec["encode_batch_size"] for c in model.encode_calls)
    assert max(c["frames"] for c in model.encode_calls) == 3  # Generation config batch_size is only 2.
    assert sum(c["frames"] for c in model.encode_calls) == 4 * 10 + 2 * 10 + 2 * 3 * 18
    assert samples == [("scorer", 0, False, 10)] * 4 + [("validation", 0, False, 10)] * 2 + [
        ("pilot", appearance, True, 18) for _ in range(2) for appearance in range(3)]
    assert set(opened[:opened.index("pilot")]) == {"scorer", "validation"}

    report, selection = _read(output / "report.json"), _read(output / "selection.json")
    assert reference == {"output": str(output / "report.json"), "plots": str(output / "index.html"), "kind": "followup_probes"}
    assert report["schema"] == "wm_audit.diagnostic.v1" and report["followup_schema"] == "wm_audit.followup.probes.v1"
    assert report["spec"] == selection["spec"] == spec
    assert report["state_fields"] == selection["state_fields"] == list(STATE_FIELDS)
    assert report["confirmatory"] is False and report["protocol"]["automatic_qualification"] is False
    assert report["protocol"]["model_rollout_used"] is False
    assert report["protocol"]["world_model_trained"] is False
    assert report["protocol"]["visibility_filtering"] is False
    assert report["protocol"]["pilot_opened_after_selection_publication"] is True
    assert report["training_roots"] == 4 and report["validation_roots"] == report["evaluation_roots"] == 2
    assert report["training_rows"] == 40 and report["validation_rows"] == 20
    grids = [1] if len(shape) == 1 else [4, 7]
    assert report["protocol"]["effective_ridge_grids"] == grids
    assert report["protocol"]["effective_mlp_grid"] == (1 if len(shape) == 1 else 4)
    assert report["protocol"]["training_target_scale"] == [512, 512, 512, 512, 1, 1]
    expected_methods = ["ridge_group_selected", "mlp_seed11", "mlp_seed29", "mlp_seed47", "train_mean"]
    assert report["methods"] == expected_methods
    assert set(selection["selected"]) == set(expected_methods[:-1])
    assert len(selection["candidates"]) == len(grids) * 4 + 3 * 2 * 2
    for method, selected in selection["selected"].items():
        eligible = [r for r in selection["candidates"] if
                    (r["family"] == "ridge" if method == "ridge_group_selected"
                     else r["family"] == "mlp" and r["seed"] == int(method.removeprefix("mlp_seed")))]
        assert set(selected) == {"pusher_xy", "block_xy", "orientation"}
        for group, chosen in selected.items():
            best = min(eligible, key=lambda r: (r["validation_group_losses"][group], r["candidate_id"]))
            assert chosen["candidate_id"] == best["candidate_id"]
            assert chosen["validation_loss"] == best["validation_group_losses"][group]
    losses = _read(output / "training-loss.json")
    assert len(losses["runs"]) == 6 and {r["seed"] for r in losses["runs"]} == {11, 29, 47}
    for run in losses["runs"]:
        assert [row["epoch"] for row in run["losses"]] == [1, 2]
        assert all(row["online_epoch_scaled_loss"] >= 0 and row["checkpoint_training_scaled_loss"] >= 0 for row in run["losses"])

    hashes = _hashes(output)
    assert report["artifacts"] == {key: value for key, value in hashes.items() if key != "report.json"}
    assert all(hashes[name] == expected for name, expected in before_pilot[0].items())
    assert report["pre_pilot_artifact_sha256"] == before_pilot[0]
    assert selection["weights_sha256"] == hashes["probe-weights.pt"]
    assert report["code_sha256"]["followup_probes.py"] == hashlib.sha256(Path(f.__file__).read_bytes()).hexdigest()
    assert len(report["input_sha256"]) == 9  # Eight roots plus their original generation manifest.
    weights = torch.load(output / "probe-weights.pt", map_location="cpu", weights_only=True)
    assert all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in weights.values())
    assert weights["latent_shape"].tolist() == list(shape)
    assert weights["training_target_scale"].tolist() == [512, 512, 512, 512, 1, 1]
    for groups in selection["selected"].values():
        for row in groups.values():
            prefix = row["weight_key"]
            assert weights[prefix + ".grid"].item() == row["grid"]
            regularizer = "alpha" if row["family"] == "ridge" else "weight_decay"
            assert weights[prefix + "." + regularizer].dtype == torch.float64
            assert weights[prefix + "." + regularizer].item() == row[regularizer]
            if row["family"] == "mlp":
                assert weights[prefix + ".seed"].item() == row["seed"]
                torch.testing.assert_close(weights[prefix + ".target_mean"], weights["train_mean"])

    assert len(report["records"]) == 2 * 3 * 5 * (3 + len(config["horizons"]))
    assert len(report["summary"]) == 3 * 5 * (3 + len(config["horizons"]))
    assert report["collision_ambiguity"]["samples"] == 2 * 18  # Original appearance only.
    assert report["collision_ambiguity"]["roots"] == 2
    assert report["predictions"]["saved"] and (output / "predictions.npz").stat().st_size < 10_000_000
    with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
        assert arrays["predicted"].shape == (2, 3, 5, 18, 6)
        assert arrays["actual"].shape == (2, 18, 6)
        assert arrays["predicted"].dtype == np.float64
        assert arrays["methods"].tolist() == expected_methods
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)
        assert not set(arrays.files) & {"pixels", "features", "latents"}
        assert arrays["candidate"][:9].tolist() == [-1, -1, -1, 0, 0, 1, 1, 2, 2]
        assert arrays["horizon"][:9].tolist() == [0, 0, 0, 1, 2, 1, 2, 1, 2]
        for record in report["records"]:
            i = arrays["root_ids"].tolist().index(record["root_id"])
            a, m = list(APPEARANCES).index(record["appearance"]), expected_methods.index(record["method"])
            mask = f._surfaces(f._layout(config), config)[record["surface"]]
            expected = d.physical_metrics(arrays["predicted"][i, a, m, mask], arrays["actual"][i, mask], config["scales"])
            assert all(record["metrics"][key] == pytest.approx(value) for key, value in expected.items())
            assert record["samples"] == int(mask.sum()) and record["pairs"] == int(mask.sum()) // 2
            assert sum(v["count"] for v in record["pusher_visibility"].values()) == record["samples"]
    for row in report["summary"]:
        matching = [r for r in report["records"] if all(r[key] == row[key] for key in ("appearance", "method", "surface"))]
        expected = d.interval([r["metrics"]["block_distance"] for r in matching],
                              seed=spec["seed"], resamples=spec["bootstrap_resamples"])
        assert row["metrics"]["block_distance"] == expected
        assert all(metric["clusters"] == 2 for metric in row["metrics"].values())
    assert {row["contrast"] for row in report["contrasts"]} == {
        "method_minus_train_mean", "method_minus_ridge", "appearance_minus_original"}
    assert all(metric["clusters"] == 2 for row in report["contrasts"] for metric in row["metrics"].values())
    assert (output / "overview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert all(method in (output / "index.html").read_text() for method in expected_methods[:-1])
    assert "seeds not independent" in report["protocol"]["uncertainty"]


def test_pilot_and_global_rng_cannot_change_selection_or_weights(paired, tmp_path, monkeypatch):
    config, data = paired
    spec = _spec()
    monkeypatch.setattr(f, "_plots", lambda *args: None)  # Real Agg artifacts exercised above.
    torch.default_generator.manual_seed(999)
    first_rng = torch.random.get_rng_state().clone()
    f.probe_followup(config, data, tmp_path / "a", SyntheticAdapter(), spec=spec)
    assert torch.equal(first_rng, torch.random.get_rng_state())
    original_load = d._load

    def changed_pilot(path, split, config, hashes):
        root = original_load(path, split, config, hashes)
        if split == "pilot":
            # Change both labels and encoded pixels; neither is a selection input.
            root.states_context[..., :4] += 600
            root.states_future[..., :4] += 600
            for appearance in range(3):
                for branch in range(2):
                    root.context[appearance, branch] = _pixels(root.states_context[branch], appearance, 0, branch)
                    for candidate in range(config["candidates"]):
                        root.future[appearance, branch, candidate] = _pixels(root.states_future[branch, candidate], appearance, 1, branch, candidate)
        return root

    monkeypatch.setattr(d, "_load", changed_pilot)
    torch.default_generator.manual_seed(123)
    second_rng = torch.random.get_rng_state().clone()
    f.probe_followup(config, data, tmp_path / "b", SyntheticAdapter(), spec=spec)
    assert torch.equal(second_rng, torch.random.get_rng_state())
    for name in ("selection.json", "probe-weights.pt", "training-loss.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    a, b = _read(tmp_path / "a/report.json"), _read(tmp_path / "b/report.json")
    assert a["summary"] != b["summary"]
    assert set(a["selection"]["selected"]) == set(b["selection"]["selected"])
    assert all(row["pusher_visibility"]["outside"]["count"] == row["samples"] for row in b["records"])
    assert len(a["records"]) == len(b["records"])  # Outside geometry never filters main errors.


def test_group_selection_is_independent_and_exact_ties_are_lexical():
    rows = [dict(candidate_id="z", validation_group_losses=dict(pusher_xy=0, block_xy=4, orientation=4)),
            dict(candidate_id="a", validation_group_losses=dict(pusher_xy=3, block_xy=0, orientation=3)),
            dict(candidate_id="b", validation_group_losses=dict(pusher_xy=3, block_xy=4, orientation=0)),
            dict(candidate_id="aa", validation_group_losses=dict(pusher_xy=0, block_xy=4, orientation=4))]
    for order in (rows, list(reversed(rows))):
        best = {}
        for row in order:
            f._consider(best, "ridge_group_selected", row, {})
            f._consider(best, "mlp_seed11", row, {})
        for method in best:
            assert {group: value[0]["candidate_id"] for group, value in best[method].items()} == {
                "pusher_xy": "aa", "block_xy": "a", "orientation": "b"}
        f._consider(best, "mlp_seed29", dict(candidate_id="other-seed", validation_group_losses={g: -1 for g in f._GROUPS}), {})
        assert all(v[0]["candidate_id"] != "other-seed" for v in best["mlp_seed11"].values())


def test_training_normalization_full_epochs_and_inference_mode():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(9, 5)).astype(np.float32)
    x[:, -1] = 9
    y = rng.normal(size=(9, 6)) * [512, 512, 512, 512, 1, 1] + [50, 60, 70, 80, .1, -.3]
    original_x, original_y = x.copy(), y.copy()
    states = []

    def checkpoint(epoch, state, loss):
        assert not torch.is_grad_enabled() and math.isfinite(loss)
        states.append((epoch, state))

    before = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        result = f._train_mlp(x, y, spec=_spec(), seed=11, decay=.01, device=torch.device("cpu"), checkpoint=checkpoint)
    assert torch.equal(before, torch.random.get_rng_state())
    assert result["epochs"] == 2 and [e for e, _ in states] == [1, 2]
    assert not torch.equal(states[0][1]["hidden_weight"], states[1][1]["hidden_weight"])
    for _, state in states:
        np.testing.assert_allclose(state["mean"], x.astype(np.float64).mean(0), rtol=1e-12)
        np.testing.assert_allclose(state["std"][:-1], x.astype(np.float64).std(0)[:-1], rtol=1e-12)
        assert state["std"][-1].item() == 1
        np.testing.assert_allclose(state["target_mean"], y.mean(0), rtol=1e-12)
        assert state["target_scale"].tolist() == [512, 512, 512, 512, 1, 1]
        assert all(not value.requires_grad and value.grad_fn is None for value in state.values())
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("actual_angles,predicted_angles,error", [
    ((179, -179), (170, 172), 0), ((170, -10), (-170, 10), 0),
    ((-179, 179), (179, -179), 4), ((179, -179), (-179, 179), 4),
])
def test_paired_effect_wraps_changes_and_difference(actual_angles, predicted_angles, error):
    actual, predicted = np.zeros((2, 1, 6)), np.zeros((2, 1, 6))
    for values, angles in ((actual, actual_angles), (predicted, predicted_angles)):
        angles = np.deg2rad(angles)
        values[:, 0, 4], values[:, 0, 5] = np.sin(angles), np.cos(angles)
    actual[:, 0, 2:4] = [[10, 20], [13, 24]]
    predicted[:, 0, 2:4] = [[100, 200], [101, 202]]
    result = f._paired_effect_metrics(predicted, actual)
    assert result["paired_angle_effect_mae_deg"] == pytest.approx(error, abs=1e-10)
    assert result["paired_block_effect_distance"] == pytest.approx(np.sqrt(8))
    assert result["paired_block_effect_squared_distance"] == pytest.approx(8)
    predicted[0, :, 4:] = 0
    assert f._paired_effect_metrics(predicted, actual)["paired_angle_effect_mae_deg"] == 180
    assert f._paired_effect_metrics(predicted, actual)["paired_undefined_angle_rate"] == 1


def test_pusher_circle_geometry_including_corners_and_tangency():
    xy = [(15, 15), (497, 497), (14, 15), (0, 256), (-14, 256), (-15, 256),
          (527, 256), (-11, -11), (-10, -10), (523, 523), (256, 256)]
    actual = np.zeros((len(xy), 6))
    actual[:, :2], actual[:, 5] = xy, 1
    assert f._pusher_visibility(actual).tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 1, 2, 0]


def test_collision_ambiguity_is_exact_empirical_mse_not_pixel_storage():
    pixels = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    pixels[-1, 0, 0, 0] = 1
    labels = np.zeros((4, 6))
    labels[:, 0], labels[:, 5] = [0, 2, 4, 100], 1
    accumulator = f._Collisions()
    accumulator.add(pixels[:2], labels[:2])
    accumulator.add(pixels[2:], labels[2:])  # The ambiguous RGB group crosses roots.
    result = accumulator.summary([2, 2, 2, 2, 1, 1])
    assert result["samples"] == 4 and result["roots"] == 2
    assert result["unique_rgb_hashes"] == 2 and result["repeated_rgb_hashes"] == 1
    assert result["ambiguous_rgb_hashes"] == 1 and result["ambiguous_samples"] == 3
    assert result["coordinate_mse_lower_bound"]["pusher_x"] == pytest.approx(2)
    assert result["mean_six_field_mse_lower_bound"] == pytest.approx(2 / 6)
    best = labels.copy()
    best[:3, 0] = 2  # Every deterministic one-image function must output ONE value here.
    assert result["scaled_loss_lower_bound"] == pytest.approx(d.physical_metrics(best, labels, [2, 2, 2, 2, 1, 1])["scaled_loss"])
    assert all(isinstance(key, bytes) and len(key) == 32 for key in accumulator.groups)
    assert all(v.shape == (6,) for row in accumulator.groups.values() for v in row if isinstance(v, np.ndarray))
    duplicate = f._Collisions()
    duplicate.add(pixels[:2], np.full((2, 6), 1e12 + .1))
    assert duplicate.summary(np.ones(6))["ambiguous_rgb_hashes"] == 0
    assert duplicate.summary(np.ones(6))["mean_six_field_mse_lower_bound"] == 0


def test_group_selection_and_bootstrap_weight_roots_not_frames():
    labels = np.zeros((102, 6))
    predicted = labels.copy()
    predicted[-2:, 0] = 4
    ids = np.array(["a"] * 100 + ["b"] * 2)
    score = f._group_scores(predicted, labels, ids, np.ones(6))
    assert score["pusher_xy"] == 8  # Not 32/102.
    records = []
    for root, count, error in (("a", 100, 0), ("b", 2, 4)):
        actual = np.tile([80, 90, 210, 220, 0, 1], (count, 1)).astype(float)
        prediction = actual.copy()
        prediction[:, 0] += error
        records.extend(f._root_records(root, "original", {"ridge_group_selected": prediction}, actual,
                                       {"all": np.ones(count, dtype=bool)}, np.zeros(count, dtype=np.uint8), np.ones(6)))
    summary = f._summary(records, _spec())[0]
    assert summary["metrics"]["pusher_distance"]["estimate"] == 2
    assert summary["metrics"]["pusher_distance"]["clusters"] == 2
    assert summary["pusher_visibility"]["outside"] == {"samples": 0, "roots_with_samples": 0, "roots_without_samples": 2, "metrics": None}


@pytest.mark.parametrize("key,value", [
    ("schema", "other"), ("confirmatory", True), ("name", ""), ("seed", True), ("seed", -1), ("seed", 2**63),
    ("bootstrap_resamples", 0), ("encode_batch_size", 0), ("encode_batch_size", True),
    ("ridge_grids", []), ("ridge_grids", [4, 4]), ("ridge_grids", [True, 4]), ("ridge_grids", [4.0]),
    ("ridge_alphas", [0]), ("ridge_alphas", [1, True]), ("ridge_alphas", [1, 1.0]),
    ("ridge_alphas", [float("nan")]), ("ridge_alphas", [float("inf")]),
    ("mlp_grid", 0), ("mlp_hidden", 0), ("mlp_seeds", []), ("mlp_seeds", [11, 11, 47]),
    ("mlp_seeds", [True, 29, 47]), ("mlp_seeds", [-1, 29, 47]),
    ("mlp_weight_decays", [-.1, .01]), ("mlp_weight_decays", [0, False]),
    ("mlp_weight_decays", [0, float("nan")]), ("mlp_weight_decays", [0, 0.0]),
    ("mlp_checkpoints", [0, 2]), ("mlp_checkpoints", [1, 1]), ("mlp_checkpoints", [1, 2.0]),
    ("mlp_checkpoints", "1,2"), ("mlp_learning_rate", 0), ("mlp_learning_rate", float("inf")),
    ("mlp_learning_rate", "0.001"), ("mlp_batch_size", False), ("primary_horizon", 3),
    ("small_edit_xy_tolerance", float("nan")), ("small_edit_xy_tolerance", 0),
    ("small_edit_angle_tolerance_deg", 181), ("extra", {"nested": [float("nan")]}),
])
def test_bad_or_nonfinite_spec_rejected_before_outputs(tmp_path, key, value):
    spec = _spec()
    spec[key] = value
    config = _config()
    config["counts"]["pilot"] = 2
    with pytest.raises(ValueError):
        f.probe_followup(config, tmp_path / "missing-data", tmp_path / "out", SyntheticAdapter(), spec=spec)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("spec", [None, [], {}, {"schema": "wm_audit.followup.v1"}])
def test_missing_spec_rejected(tmp_path, spec):
    with pytest.raises(ValueError, match="spec"):
        f.probe_followup(_config(), tmp_path / "data", tmp_path / "out", SyntheticAdapter(), spec=spec)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("shape,grid", [((), 4), ((2, 3, 6), 4), ((15, 6), 4), ((36, 6), 4), ((49, 6), 8)])
def test_invalid_model_shape_or_grid_precedes_output_creation(tmp_path, shape, grid):
    config, spec = _config(), _spec()
    config["counts"]["pilot"] = 2
    spec["mlp_grid"] = grid
    with pytest.raises(ValueError):
        f.probe_followup(config, tmp_path / "missing-data", tmp_path / "out", SyntheticAdapter(shape), spec=spec)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", ["shape", "nonfinite"])
def test_bad_encoder_results_fail_without_pilot_or_completion(paired, tmp_path, monkeypatch, bad):
    config, data = paired
    model, opened = SyntheticAdapter(), []
    original = pipeline.load_root

    def tracked(path):
        opened.append(Path(path).parent.name)
        return original(path)

    def encode(pixels):
        assert not torch.is_grad_enabled() and not model.gain.requires_grad
        if bad == "shape":
            return torch.zeros((len(pixels), 1, 5))
        return torch.full((len(pixels), 1, 6), float("nan"))

    monkeypatch.setattr(pipeline, "load_root", tracked)
    monkeypatch.setattr(model, "encode", encode)
    with pytest.raises(ValueError, match="model.encode"):
        f.probe_followup(config, data, tmp_path / "out", model, spec=_spec())
    assert "pilot" not in opened
    assert not (tmp_path / "out/report.json").exists() and not (tmp_path / "out/selection.json").exists()
    assert model.training and model.gain.requires_grad and model.gain.grad is None


def test_nonfinite_training_and_effect_shapes_rejected():
    with pytest.raises(ValueError, match="finite"):
        f._train_mlp(np.full((3, 2), np.nan), np.ones((3, 6)), spec=_spec(), seed=11, decay=0,
                     device=torch.device("cpu"), checkpoint=_forbidden)
    with pytest.raises(ValueError, match="matching"):
        f._train_mlp(np.ones((3, 2)), np.ones((2, 6)), spec=_spec(), seed=11, decay=0,
                     device=torch.device("cpu"), checkpoint=_forbidden)
    with pytest.raises(ValueError, match="finite"):
        f._paired_effect_metrics(np.ones((2, 6)), np.ones((2, 6)))
    with pytest.raises(ValueError, match="finite"):
        f._paired_effect_metrics(np.full((2, 3, 6), np.inf), np.ones((2, 3, 6)))
    with pytest.raises(ValueError, match="finite"):
        f._pusher_visibility(np.full((3, 6), np.nan))
    with pytest.raises(ValueError, match="checkpoint"):
        f._tensor_state({"weight": torch.tensor(float("nan"))})


def test_nonfinite_optimizer_update_fails_before_checkpoint_publication(monkeypatch):
    def corrupt_step(optimizer, *args, **kwargs):
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_step)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="Nonfinite"):
        f._train_mlp(np.ones((3, 2)), np.ones((3, 6)), spec=_spec(), seed=11, decay=0,
                     device=torch.device("cpu"), checkpoint=_forbidden)
    assert torch.equal(before, torch.random.get_rng_state())


def test_outputs_are_exclusive_even_after_interruption(paired, tmp_path, monkeypatch):
    config, data = paired
    output = tmp_path / "out"
    output.mkdir()
    (output / "selection.json").write_bytes(b"preserve this partial run")
    before = _hashes(output)
    monkeypatch.setattr(d, "_roster", _forbidden)
    with pytest.raises(FileExistsError):
        f.probe_followup(config, data, output, SyntheticAdapter(), spec=_spec())
    assert _hashes(output) == before


def test_pilot_manifest_is_still_authenticated_after_publication(paired, tmp_path, monkeypatch):
    config, data = paired
    copied, output = tmp_path / "data", tmp_path / "out"
    shutil.copytree(data, copied)
    manifest_path = copied / "manifest-development.json"
    manifest = _read(manifest_path)
    manifest["files"]["pilot"][next(iter(manifest["files"]["pilot"]))] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(f, "_plots", _forbidden)
    with pytest.raises(ValueError, match="generation manifest"):
        f.probe_followup(config, copied, output, SyntheticAdapter(), spec=_spec())
    assert (output / "selection.json").exists() and (output / "probe-weights.pt").exists()
    assert not (output / "report.json").exists()


def test_prediction_budget_skips_before_allocating_large_arrays(monkeypatch):
    monkeypatch.setattr(f, "_MAX_PREDICTION_BYTES", 100)
    layout = f._layout(_config())
    monkeypatch.setattr(f.np, "empty", _forbidden)
    arrays, info = f._prediction_arrays(128, ["ridge", "mlp11", "mlp29", "mlp47", "mean"], layout)
    assert arrays is None and info["saved"] is False and "budget" in info["reason"]


def test_collision_shape_and_unequal_root_rows_fail():
    pixels = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    labels = np.zeros((3, 6))
    accumulator = f._Collisions()
    with pytest.raises(ValueError, match="RGB"):
        accumulator.add(pixels.astype(float), labels)
    with pytest.raises(ValueError, match="finite"):
        accumulator.add(pixels, labels[:, :5])
    accumulator.add(pixels[:1], labels[:1])
    accumulator.add(pixels[1:], labels[1:])
    with pytest.raises(ValueError, match="equal-row-per-root"):
        accumulator.summary(np.ones(6))