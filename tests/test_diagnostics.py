"""Synthetic scientific-contract checks; not evidence of model competence."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import numpy as np
import pytest
import torch

from wm_audit import cli, diagnostics as d
from wm_audit.data import save_root
from test_pipeline import SyntheticAdapter, _config, _pixels, _synthetic_root


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setenv("WM_AUDIT_PROGRESS", "0")
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def paired(tmp_path):
    config = _config()
    config.update(bootstrap_resamples=30, native_margin=.05, readout_margin=.05, regret_margin=.02)
    config["counts"].update(editor=0, scorer=4, validation=2, pilot=2)
    data = tmp_path / "data"
    for split, indices in (("scorer", (0, 1, 2, 3)), ("validation", (4, 5)), ("pilot", (6, 7))):
        for index in indices:
            root = _synthetic_root(split, index, config)
            save_root(data / split / f"{root.metadata['root_id']}.npz", root)
            manifest = {"config": config, "files": {split: {p.relative_to(data).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((data / split).glob("*.npz"))} for split in config["counts"]}}
            (data / "manifest-development.json").write_text(json.dumps(manifest))
    return config, data


@pytest.mark.parametrize("n,features", [(12, 3), (3, 12)])
def test_spectral_ridge_matches_direct_solution(n, features):
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(n, features)), rng.normal(size=(n, 6))
    x[:, -1] = 4  # Constant feature handling.
    family = d.RidgeFamily(x, y)
    for alpha in (.01, 1, 100):
        state = family.fit(alpha)
        z = (torch.tensor(x) - state["mean"]) / state["std"]
        weight = torch.linalg.solve(z.T @ z + alpha * torch.eye(features), z.T @ (torch.tensor(y) - state["bias"]))
        torch.testing.assert_close(state["weight"], weight, rtol=1e-8, atol=1e-8)
        assert np.isfinite(d.predict_readout(state, x)).all()


def test_physical_errors_wrap_angle_and_keep_undefined():
    actual = np.zeros((2, 6))
    actual[:, 4:] = [np.sin(np.deg2rad(179)), np.cos(np.deg2rad(179))]
    predicted = actual.copy()
    predicted[0, 4:] = [np.sin(np.deg2rad(-179)), np.cos(np.deg2rad(-179))]
    predicted[1, 4:] = 0
    metrics = d.physical_metrics(predicted, actual, np.ones(6))
    assert metrics["angle_mae_deg"] == pytest.approx(91)
    assert metrics["undefined_angle_rate"] == .5
    predicted[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        d.physical_metrics(predicted, actual, np.ones(6))


def test_root_bootstrap_not_frame_weighted():
    rows = []
    for cluster, count, value in (("a", 10, 0), ("b", 1, 4)):
        rows += [dict(cluster=cluster, appearance="original", horizon=1, native=value,
                      persistence=value+2, zero_future_actions=value+3)] * count
    summary = d.prediction_summary(rows, {"bootstrap_resamples": 100, "seed": 2})[0]
    assert summary["metrics"]["native"]["estimate"] == 2
    assert summary["native_minus_persistence"]["ci95"] == [-2, -2]
    assert summary["metrics"]["native"]["clusters"] == 2


@pytest.mark.parametrize("shape", [(6,), (16, 6)])
def test_readout_full_workflow_no_rollouts_and_no_leakage(paired, tmp_path, monkeypatch, shape):
    config, data = paired
    output = tmp_path / "readout"
    model = SyntheticAdapter(shape)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    original = d._load
    observed = []

    def tracked(path, split, config, hashes):
        observed.append(split)
        if split == "pilot":
            assert (output / "selection.json").is_file()
        return original(path, split, config, hashes)

    monkeypatch.setattr(d, "_load", tracked)
    d.readout_diagnostic(config, data, output, model, grids=(2, 4), alphas=(.1, 10.))
    report = json.loads((output / "report.json").read_text())
    assert not model.rollout_calls
    assert all(call["frozen"] and not call["grad_enabled"] for call in model.encode_calls)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name])
    assert model.gain.requires_grad and model.training  # Restore incoming model flags.
    assert observed == ["scorer"] * 4 + ["validation"] * 2 + ["pilot"] * 2
    assert report["training_rows"] == 4 * 2 * (3+2)
    assert len(report["records"]) == 2*3*2*3
    assert report["protocol"]["prediction_latents_in_readout_training"] is False
    assert report["confirmatory"] is False
    assert report["selected"] == min(report["selection_candidates"], key=lambda r: (r["validation_scaled_loss"], r["grid"], r["alpha"]))
    for name, expected in report["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "index.html").is_file() and (output / "scatter-data.npz").is_file()
    with pytest.raises(FileExistsError):
        d.readout_diagnostic(config, data, output, model)


def test_pilot_cannot_change_readout_selection(paired, tmp_path, monkeypatch):
    config, data = paired
    # Avoid plotting twice; integration test above exercises the real plots.
    import wm_audit.diagnostic_plots as plots
    monkeypatch.setattr(plots, "readout_plots", lambda *args: None)
    d.readout_diagnostic(config, data, tmp_path / "a", SyntheticAdapter(), alphas=(.1, 10.))
    original = d._load
    def changed(path, split, config, hashes):
        root = original(path, split, config, hashes)
        if split == "pilot":
            root.states_context[..., :4] += 100
            root.states_future[..., :4] += 100
        return root
    monkeypatch.setattr(d, "_load", changed)
    d.readout_diagnostic(config, data, tmp_path / "b", SyntheticAdapter(), alphas=(.1, 10.))
    assert (tmp_path / "a/selection.json").read_bytes() == (tmp_path / "b/selection.json").read_bytes()


def test_prediction_roster_raw_actions_and_prefix(paired, tmp_path):
    config, data = paired
    model = SyntheticAdapter()
    d.prediction_diagnostic(config, tmp_path / "prediction", model, data=data)
    report = json.loads((tmp_path / "prediction/report.json").read_text())
    assert len(report["records"]) == 2*3*2*3*2
    assert len(report["summary"]) == 3*2
    calls = model.rollout_calls
    assert len(calls) == 2*3*2*3*2
    for native, zero in zip(calls[::2], calls[1::2]):
        torch.testing.assert_close(native["actions"][:, :2], zero["actions"][:, :2])
        assert torch.count_nonzero(zero["actions"][:, 2:]) == 0
        assert native["actions"].abs().max() <= 1
        assert not native["grad_enabled"] and native["frozen"] and not native["training"]
        torch.testing.assert_close(native["history"], zero["history"])
    assert model.gain.requires_grad
    assert report["protocol"]["readout_used"] is False


@pytest.mark.parametrize("finite_terminal", [False, True])
def test_recorded_windows_episode_sampling_and_terminal_safety(tmp_path, finite_terminal):
    import h5py
    config = _config()
    config.update(bootstrap_resamples=20)
    path = tmp_path / "recorded.h5"
    length, count = 40, 4
    states = np.zeros((length * count, 6), dtype=np.float32)
    states[:, :4] = np.arange(len(states))[:, None]
    states[:, 5] = 1
    actions = np.full((len(states), 2), .1, dtype=np.float32)
    if finite_terminal:
        actions[:, 0] = 1.25
    actions[length-1::length] = 9 if finite_terminal else np.nan
    with h5py.File(path, "w") as f:
        f["pixels"] = _pixels(states, 0, 0, 0)
        f["action"] = actions
        f["ep_offset"] = np.arange(count) * length
        f["ep_len"] = np.full(count, length)
    model = SyntheticAdapter()
    d.prediction_diagnostic(config, tmp_path / "recorded", model, h5=path, episodes=3)
    report = json.loads((tmp_path / "recorded/report.json").read_text())
    assert len(report["windows"]) == len({r["episode"] for r in report["windows"]}) == 3
    assert len(report["records"]) == 3*len(config["horizons"])
    assert all(w["start"] + 5*(3+config["horizon"]-1) < length for w in report["windows"])
    assert all(r["appearance"] == "original" for r in report["records"])
    assert all(torch.isfinite(c["actions"]).all() for c in model.rollout_calls)
    assert all(c["recorded_actions"] for c in model.rollout_calls)
    if finite_terminal:
        assert all(c["actions"].max() == 1.25 for c in model.rollout_calls)
        assert all(r["raw_action_max"] == 1.25 for r in report["records"])
    assert report["recorded_h5_info"]["action_stats"]["action_schema"] == "recorded"
    with pytest.raises(ValueError, match="only 4"):
        d.prediction_diagnostic(config, tmp_path / "too-many", SyntheticAdapter(), h5=path, episodes=5)
    assert not (tmp_path / "too-many/report.json").exists()


@pytest.mark.parametrize("kwargs", [{}, {"data": "a", "h5": "b"}])
def test_reject_ambiguous_prediction_sources(tmp_path, kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        d.prediction_diagnostic(_config(), tmp_path / "out", SyntheticAdapter(), **kwargs)
    assert not (tmp_path / "out").exists()


def test_cli_arguments_and_runner_plan(tmp_path):
    args = cli.parser().parse_args(["diagnose-readout", "--model", "lewm", "--config", "configs/pilot.json", "--data", "data", "--output", "out"])
    assert args.grids == [2, 4]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["diagnose-prediction", "--model", "lewm", "--config", "c", "--output", "o", "--data", "d", "--h5", "h"])
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_diagnostics.py"))
    options = Namespace(stages=None, h5=Path("recorded.h5"), models=["lewm", "dinowm"],
                        trust_checkpoint=True, cache=Path("cache"), device="cuda", config=Path("c"),
                        data=Path("data"), output=tmp_path, episodes=10)
    plan = module["commands"](options)
    assert len(plan) == 6
    assert all("--device" in command and "cuda" in command for command in plan)
    assert all(not set(command) & {"generate", "fit", "evaluate"} for command in plan)
    options.trust_checkpoint = False
    with pytest.raises(ValueError, match="trust-checkpoint"):
        module["commands"](options)


def test_backup_includes_completed_diagnostics_only(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/backup_world_models.py"))
    root = tmp_path / "project"
    diagnostic = root / "runs/new/lewm/readout"
    diagnostic.mkdir(parents=True)
    (diagnostic / "index.html").write_text("finished plot")
    report = {"schema": "wm_audit.diagnostic.v1", "artifacts": {"index.html": hashlib.sha256(b"finished plot").hexdigest()}}
    (diagnostic / "report.json").write_text(json.dumps(report))
    incomplete = root / "runs/new/lewm/prediction-paired"
    incomplete.mkdir()
    (incomplete / "index.html").write_text("unfinished")
    module["capture"](root, tmp_path / "backup.zip")
    with zipfile.ZipFile(tmp_path / "backup.zip") as z:
        assert "runs/new/lewm/readout/report.json" in z.namelist()
        assert "runs/new/lewm/readout/index.html" in z.namelist()
        assert not any("prediction-paired" in name for name in z.namelist())
    (diagnostic / "index.html").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        module["capture"](root, tmp_path / "bad.zip")


def test_changed_generation_manifest_is_rejected(paired, tmp_path):
    config, data = paired
    path = data / "manifest-development.json"
    manifest = json.loads(path.read_text())
    first = next(iter(manifest["files"]["pilot"]))
    manifest["files"]["pilot"][first] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generation manifest"):
        d.prediction_diagnostic(config, tmp_path / "out", SyntheticAdapter(), data=data)
    assert not (tmp_path / "out/report.json").exists()


def test_diagnostic_backup_rechecks_archived_digest(tmp_path, monkeypatch):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/backup_world_models.py"))
    root = tmp_path / "project"
    diagnostic = root / "runs/new/lewm/readout"
    diagnostic.mkdir(parents=True)
    artifact = diagnostic / "index.html"
    artifact.write_text("original")
    (diagnostic / "report.json").write_text(json.dumps({"schema": "wm_audit.diagnostic.v1",
        "artifacts": {"index.html": hashlib.sha256(b"original").hexdigest()}}))
    original_digest = module["stream_digest"]
    def replace_after_hash(stream):
        value = original_digest(stream)
        artifact.write_text("changed after first check")
        return value
    monkeypatch.setitem(module["capture"].__globals__, "stream_digest", replace_after_hash)
    with pytest.raises(ValueError, match="between verification and capture"):
        module["capture"](root, tmp_path / "race.zip")