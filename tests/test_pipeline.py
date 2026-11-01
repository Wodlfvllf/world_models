"""Synthetic integration contracts ONLY; no pretrained or simulator competence.

The algebraic adapter reads a test-only float payload packed into uint8 frames.
Root archives satisfy the strict storage schema but contain invented trajectories
and explicitly labeled synthetic runtime provenance, not replay qualifications.
No release downloads, real simulator, GPU, CLI or production results are used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from wm_audit import data, pipeline
from wm_audit.data import PairedRoot, load_root, save_root
from wm_audit.fitting import PhysicalReadout, load_editor
from wm_audit.simulator import (
    Anchor, EpisodeRecipe, Intervention, POINT_REVISION, SOURCE_HASHES, STATE_FIELDS, array_digest,
)
from wm_audit.statistics import APPEARANCES, METHODS, METRICS, analyze_records, circular_error_deg


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    original = torch.get_num_threads()
    torch.set_num_threads(min(2, original))
    yield
    torch.set_num_threads(original)


def _config():
    return {
        "name": "synthetic-pipeline-unit-test-not-research",
        "counts": {"editor": 4, "scorer": 4, "validation": 2, "pilot": 0, "confirmation": 0},
        "horizon": 3, "candidates": 3, "horizons": [1, 2],
        "scales": [32, 32, 32, 32, 0.25, 0.25],
        "editors": ["inverse", "ridge", "mlp", "rollout"],
        "rank": 3, "steps": 2, "batch_size": 2, "seed": 3,
        "basis_samples": 4, "context_dim": 2, "hidden_dim": 8,
    }


def _pixels(states, appearance, kind, branch, candidate=0):
    """TEST-ONLY encoding: six float32 bytestrings, then four marker bytes."""
    states = np.asarray(states, dtype=np.float32)
    pixels = np.zeros((*states.shape[:-1], 224, 224, 3), dtype=np.uint8)
    flat = pixels.reshape(-1, 224 * 224 * 3)
    flat[:, :24] = states.copy().reshape(-1, 6).view(np.uint8)
    flat[:, -4:] = (appearance, kind, branch, candidate)
    return pixels


def _synthetic_root(split, index, config, *, split_seed=None):
    """Schema fixture, not a simulator or physical replay substitute."""
    seed = data.DEFAULT_SPLIT_SEEDS[split] if split_seed is None else split_seed
    root_seed = data.derive_root_seed(seed, split, index)
    streams = {name: data._derive_seed(root_seed, name) for name in data.STREAMS}
    k, h = config["candidates"], config["horizon"]
    anchor = Anchor((80 + 3 * index, 90 + index ** 2), (210 + 2 * index, 170 + 3 * index), -0.4 + 0.07 * index)
    interventions = (Intervention(), Intervention(block_translation=(8, 0)),
                     Intervention(block_translation=(0, -6)), Intervention(block_rotation=0.1))
    intervention = interventions[index % 4]
    cf = intervention.apply(anchor, (0, 45))
    request = cf.features() - anchor.features()
    context = np.empty((2, 3, 6), dtype=np.float64)
    for branch, initial in enumerate((anchor, cf)):
        for time in range(3):
            state = initial.features()
            state[:4] += time * np.array([1, 2, 1, -1])
            if branch:
                state[:4] += request[:4] * (0.25 * time)
            angle = initial.angle + time * 0.05 + branch * intervention.block_rotation * 0.2 * time
            state[4:] = (np.sin(angle), np.cos(angle))
            context[branch, time] = state
    actions = np.zeros((k, 2 + h, 10), dtype=np.float32)
    actions[:, :2] = 0.05
    for candidate in range(k):
        for step in range(h):
            actions[candidate, 2 + step] = (0.1, -0.2, 0.3)[candidate % 3] + 0.025 * step
    future = np.empty((2, k, h, 6), dtype=np.float64)
    for branch in range(2):
        for candidate in range(k):
            for step in range(h):
                state = context[branch, -1].copy()
                state[:4] += (step + 1) * np.array([2 + 3 * candidate, -1 + 2 * candidate,
                                                 1 + 2 * candidate, 0.5 * candidate])
                if branch:
                    state[:4] += request[:4] * (candidate + 1) * (step + 1) * 0.2
                angle = np.arctan2(state[4], state[5]) + (step + 1) * 0.08 * (candidate + 1)
                state[4:] = (np.sin(angle), np.cos(angle))
                future[branch, candidate, step] = state
    # The true-image oracle prefers candidate 1, while invented native forecasts
    # tend to prefer candidate 2. Executed cost is deliberately NOT latent cost.
    goal_state = future[1, min(1, k - 1), -1]
    goal = Anchor(tuple(goal_state[:2]), tuple(goal_state[2:4]), np.arctan2(goal_state[4], goal_state[5]))
    costs = np.tile(np.array([0.05, 0.75, 0.3])[:k], (2, 1))
    if index % 4 == 0:
        costs[:] = 0.4  # Tied banks and shams must remain in the full roster.
    coverage = np.broadcast_to(1 - costs[:, :, None], (2, k, h)).copy()
    rgb_context = np.empty((3, 2, 3, 224, 224, 3), dtype=np.uint8)
    rgb_future = np.empty((3, 2, k, h, 224, 224, 3), dtype=np.uint8)
    goals = np.empty((3, 224, 224, 3), dtype=np.uint8)
    for appearance in range(3):
        for branch in range(2):
            rgb_context[appearance, branch] = _pixels(context[branch], appearance, 0, branch)
            for candidate in range(k):
                rgb_future[appearance, branch, candidate] = _pixels(
                    future[branch, candidate], appearance, 1, branch, candidate)
        goals[appearance] = _pixels(goal.features(), appearance, 2, 2)
    metadata = {
        "schema": data.SCHEMA_VERSION,
        "root_id": data._root_id(split, index, root_seed),
        "split": split, "split_seed": seed, "root_index": index, "root_seed": root_seed,
        "stream_seeds": streams,
        "config": {"horizon": h, "candidates": k, "anchor_clearance": 2.0, "max_anchor_attempts": 256},
        "frame_indices": list(range(0, 5 * (3 + h), 5)),
        "primitive_control_seconds": 0.1, "physics_substep_seconds": 0.01,
        "action_encoding": "native_relative_xy_time_major_5", "action_scale": 100.0,
        "state_fields": list(STATE_FIELDS), "block_position": "native_body_origin",
        "request_encoding": "counterfactual_minus_factual_anchor_features",
        "recipes": [EpisodeRecipe(streams["environment"], pose, goal).to_dict() for pose in (anchor, cf)],
        "intervention": asdict(intervention),
        "appearances": [asdict(value) for value in data.sample_appearances(streams["appearance"])],
        "source": {"root": "synthetic-schema-fixture-not-executed", "point_revision": POINT_REVISION,
                   "sha256": SOURCE_HASHES},
        "runtime": {"synthetic_fixture": "invented arrays; not simulator replay results"},
        "geometry": {"block_local_cog": [0.0, 45.0], "pusher_radius": 15.0,
                     "effective_tee_scale": 30.0, "declared_block_scale": 40.0,
                     "arena_inner_bounds": [7.0, 504.0],
                     "block_vertices": [[[-60, 30], [60, 30], [60, 0], [-60, 0]],
                                        [[-15, 30], [-15, 120], [15, 120], [15, 30]]]},
        "sampling": {"attempts": 1, "rejected_anchors": 0, "rejection_reasons": {},
                     "eligibility": "anchor_only", "anchor_source": "supplied", "rng": "numpy.PCG64",
                     "future_filtering": False},
        "actions_source": "supplied_before_rollout",
        "audit": {"diagnostic_sha256": [["0" * 64] * k for _ in range(2)],
                  "prefix_sha256": ["0" * 64] * 2, "native_contact_points": [[0] * k for _ in range(2)],
                  "native_terminated_any": [[False] * k for _ in range(2)],
                  "actions_sha256": array_digest(actions)},
    }
    return PairedRoot(rgb_context, rgb_future, context, future, actions, goals, costs, request,
                      np.zeros((2, 3), dtype=np.float64), coverage, json.loads(json.dumps(metadata)))


class SyntheticAdapter(nn.Module):
    """ALGEBRAIC TEST DOUBLE: neither a learned visual encoder nor real dynamics."""

    def __init__(self, shape=(6,)):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.latent_shape = shape
        self.metadata = {"name": "synthetic-unit-test-only", "latent_shape": list(shape)}
        self.encode_calls, self.rollout_calls, self.history_gradients = [], [], []

    @property
    def device(self):
        return self.gain.device

    def encode(self, pixels):
        assert isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 5
        n, t = pixels.shape[:2]
        flat = pixels.reshape(n * t, -1)
        states = np.ascontiguousarray(flat[:, :24]).view(np.float32).reshape(n, t, 6)
        markers = flat[:, -4:].copy()
        self.encode_calls.append({"markers": markers, "grad_enabled": torch.is_grad_enabled(),
                                  "frozen": not self.gain.requires_grad, "frames": n * t})
        z = torch.from_numpy(states.copy()).to(self.device) / torch.tensor([32, 32, 32, 32, 1, 1], device=self.device)
        appearance = torch.from_numpy(markers[:, 0].astype(np.float32)).reshape(n, t, 1).to(self.device)
        z = z + appearance * z.new_tensor([0.15, 0.1, -0.2, 0.03, 0.02, 0.01])
        if len(self.latent_shape) == 2:
            count = self.latent_shape[0]
            z = z.unsqueeze(2).expand(n, t, count, 6)
            z = z + torch.arange(count, device=self.device).reshape(1, 1, count, 1) * 0.001
        return z

    def rollout(self, history, actions, *, recorded_actions=False):
        assert actions.shape == (len(history), actions.shape[1], 10) and actions.shape[1] > 2
        assert torch.isfinite(actions).all()
        assert recorded_actions or actions.abs().max() <= 1
        self.rollout_calls.append({"history": history.detach().cpu().clone(), "actions": actions.detach().cpu().clone(),
                                   "grad_enabled": torch.is_grad_enabled(), "requires_grad": history.requires_grad,
                                   "frozen": not self.gain.requires_grad, "training": self.training,
                                   "recorded_actions": recorded_actions})
        if history.requires_grad:
            history.register_hook(lambda grad: self.history_gradients.append(float(grad.abs().sum())))
        h = actions.shape[1] - 2
        base = 0.2 * history[:, 0] + 0.3 * history[:, 1] + 0.5 * history[:, 2]
        shifts = actions[:, 2:, :1].reshape(len(history), h, *([1] * len(self.latent_shape)))
        steps = torch.arange(1, h + 1, device=history.device).reshape(1, h, *([1] * len(self.latent_shape)))
        direction = history.new_tensor([0.3, 0.1, 0.2, -0.1, 0.0, 0.0])
        bias = history.new_tensor([0.07, -0.035, 0.028, 0.02, 0.0, 0.0])
        return base.unsqueeze(1) + self.gain * (steps * shifts * direction + bias)

    def goal_cost(self, predicted, goal):
        assert predicted.shape == goal.shape
        return (predicted - goal).square().flatten(1).mean(1)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-pipeline-data")
    config = _config()
    # Sparse indices ensure candidate selection uses root_index, not file ordinal.
    for split, indices in {"editor": (2, 5, 0, 7), "scorer": (1, 4, 6, 8), "validation": (10, 12)}.items():
        for index in indices:
            root = _synthetic_root(split, index, config)
            save_root(directory / split / f"{root.metadata['root_id']}.npz", root)
    # Fitting must not even validate these held-out directories.
    (directory / "pilot").mkdir()
    (directory / "pilot" / "must-not-open.npz").write_bytes(b"not a valid NPZ")
    return directory, config


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    directory, config = dataset
    run = tmp_path_factory.mktemp("synthetic-pipeline-run")
    metadata = pipeline.fit(config, directory, run, SyntheticAdapter(), seed=17)
    return directory, config, run, metadata


def _forbidden(*args, **kwargs):
    raise AssertionError("This operation is forbidden by the pipeline test contract.")


def _digest_tree(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("shape", [(6,), (16, 6)])
def test_fit_real_fitting_apis_memmaps_independent_scorer_and_native_gradients(dataset, tmp_path, monkeypatch, shape):
    directory, config = dataset
    model = SyntheticAdapter(shape)
    loaded, fit_calls, scorer_rows = [], [], []
    original_load, original_fit = pipeline.load_root, pipeline.fit_editor
    original_features_fit = PhysicalReadout.fit_features
    editor_roots = [load_root(p) for p in sorted((directory / "editor").glob("*.npz"))]

    def tracked_load(path):
        loaded.append(Path(path).parent.name)
        assert loaded[-1] in ("editor", "scorer")
        return original_load(path)

    def tracked_fit(name, histories, requests, targets, **kwargs):
        assert isinstance(histories, np.memmap) and isinstance(targets, np.memmap)
        assert histories.shape == targets.shape == (4, 3, *shape)
        assert torch.as_tensor(histories).device.type == "cpu"
        assert Path(histories.filename).parent == tmp_path / "train-cache"
        np.testing.assert_array_equal(requests, np.stack([r.request for r in editor_roots]))
        assert kwargs["rank"] == config["rank"] and kwargs["steps"] == config["steps"]
        assert kwargs["batch_size"] == config["batch_size"] and kwargs["seed"] == 17
        assert not model.gain.requires_grad and not model.training
        assert "future_targets" not in kwargs
        if name == "inverse":
            np.testing.assert_array_equal(kwargs["anchor_y"], np.stack([r.states_context[0, 0] for r in editor_roots]))
            assert not np.array_equal(kwargs["anchor_y"], np.stack([r.states_context[0, -1] for r in editor_roots]))
        else:
            assert "anchor_y" not in kwargs
        if name == "rollout":
            assert kwargs["model"] is model and kwargs["actions"].shape == (4, 1, 5, 10)
            for i, root in enumerate(editor_roots):
                np.testing.assert_array_equal(kwargs["actions"][i, 0], root.actions[root.metadata["root_index"] % 3])
        else:
            assert "model" not in kwargs and "actions" not in kwargs
        fit_calls.append(name)
        return original_fit(name, histories, requests, targets, **kwargs)

    def tracked_features_fit(self, x, y):
        scorer_rows.append((torch.as_tensor(x).clone(), torch.as_tensor(y).clone()))
        return original_features_fit(self, x, y)

    monkeypatch.setattr(pipeline, "load_root", tracked_load)
    monkeypatch.setattr(pipeline, "fit_editor", tracked_fit)
    monkeypatch.setattr(PhysicalReadout, "fit_features", tracked_features_fit)
    before_rng = torch.random.get_rng_state().clone()
    result = pipeline.fit(config, directory, tmp_path, model, seed=17)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert set(loaded) == {"editor", "scorer"} and fit_calls == config["editors"]
    assert model.gain.requires_grad and model.training and model.gain.grad is None
    assert model.gain.item() == 1 and model.history_gradients and max(model.history_gradients) > 0
    assert all(c["frozen"] and not c["training"] and len(c["history"]) <= 2 for c in model.rollout_calls)
    assert any(c["requires_grad"] for c in model.rollout_calls)
    assert all(not c["grad_enabled"] and c["frozen"] and c["frames"] <= 2 for c in model.encode_calls)
    markers = np.concatenate([c["markers"] for c in model.encode_calls])
    assert set(markers[:, 0]) == {0} and set(markers[:, 1]) == {0, 1}
    assert result == json.loads((tmp_path / "training.json").read_text())
    assert result["model"] == model.metadata and result["config"] == config and result["seed"] == 17
    assert set(result["fit_logs"]) == {"scorer", *config["editors"]}
    assert "absorb systematic model error" in result["fit_logs"]["scorer"]["calibration_caveat"]
    assert "not shuffled" in result["protocol"]["wrong_request"]
    assert result["editor_candidate_indices"] == [r.metadata["root_index"] % 3 for r in editor_roots]
    assert set(result["editor_candidate_indices"]) == {0, 1, 2}
    for split in ("editor", "scorer"):
        assert result["root_ids"][split] == [load_root(p).metadata["root_id"] for p in sorted((directory / split).glob("*.npz"))]

    factual = np.load(tmp_path / "train-cache" / "factual.npy", mmap_mode="r")
    counterfactual = np.load(tmp_path / "train-cache" / "counterfactual.npy", mmap_mode="r")
    assert isinstance(factual, np.memmap) and isinstance(counterfactual, np.memmap)
    assert factual.dtype == np.float32 and result["train_cache"]["array_bytes"] == factual.nbytes + counterfactual.nbytes
    assert result["train_cache"]["retained"] and not result["train_cache"]["resume_supported"]
    with torch.no_grad():
        for i, root in enumerate(editor_roots):
            z = model.encode(root.context[0]).numpy()
            np.testing.assert_array_equal(factual[i], z[0])
            np.testing.assert_array_equal(counterfactual[i], z[1])

    # Independent reference assembly: exactly C0, each branch's 3 context states,
    # emitted selected-horizon native states, then real selected-horizon states.
    reference_x, reference_y, chosen = [], [], []
    scorer = PhysicalReadout(shape)
    with torch.no_grad():
        for path in sorted((directory / "scorer").glob("*.npz")):
            root = load_root(path)
            candidate = root.metadata["root_index"] % 3
            chosen.append(candidate)
            history = model.encode(root.context[0])
            for branch in range(2):
                emitted = model.rollout(history[branch:branch + 1], torch.from_numpy(root.actions[candidate:candidate + 1]))[0, [0, 1]]
                actual = model.encode(root.future[0, branch, candidate, [0, 1]][None])[0]
                assert not torch.allclose(emitted, actual)
                for z, y in ((history[branch], root.states_context[branch]),
                             (emitted, root.states_future[branch, candidate, [0, 1]]),
                             (actual, root.states_future[branch, candidate, [0, 1]])):
                    reference_x.append(scorer.features(z))
                    reference_y.append(torch.from_numpy(y))
    x, y = scorer_rows[0]
    assert x.shape == (4 * 2 * (3 + 2 * 2), scorer.feature_dim)
    torch.testing.assert_close(x, torch.cat(reference_x), rtol=0, atol=0)
    torch.testing.assert_close(y, torch.cat(reference_y), rtol=0, atol=0)
    assert result["fit_logs"]["scorer"]["candidate_indices"] == chosen
    assert set(chosen) == {0, 1, 2}
    if len(shape) == 2:
        assert x.shape[1] == 24 < np.prod(shape)
    payload = torch.load(tmp_path / "scorer.pt", weights_only=True)
    assert set(payload) == {"shape", "state_dict"} and payload["shape"].tolist() == list(shape)
    assert all(isinstance(v, torch.Tensor) for v in payload["state_dict"].values())
    assert all((tmp_path / "editors" / f"{name}.pt").is_file() for name in config["editors"])


@pytest.mark.parametrize("artifact", ["training.json", "scorer.pt", "editors/ridge.pt", "train-cache/factual.npy"])
def test_existing_or_incomplete_artifacts_are_never_overwritten(dataset, tmp_path, monkeypatch, artifact):
    directory, config = dataset
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preserve this original artifact exactly")
    before = _digest_tree(tmp_path)
    monkeypatch.setattr(pipeline, "load_root", _forbidden)
    with pytest.raises(FileExistsError, match="new run directory"):
        pipeline.fit(config, directory, tmp_path, SyntheticAdapter())
    assert _digest_tree(tmp_path) == before


def test_exclusive_publication_preserves_a_racing_writer_and_cleans_temporary(tmp_path):
    path = tmp_path / "scorer.pt"

    def racing_writer(temporary):
        temporary.write_bytes(b"new training")
        path.write_bytes(b"a different writer won")

    with pytest.raises(FileExistsError):
        pipeline._publish(path, racing_writer)
    assert path.read_bytes() == b"a different writer won"
    assert list(tmp_path.iterdir()) == [path]


def test_interrupted_fit_is_not_resumed_and_parent_lock_is_untouched(dataset, tmp_path, monkeypatch):
    directory, config = dataset
    lock = tmp_path / "parent-run.lock"
    lock.write_text("parent-owned")
    model = SyntheticAdapter()

    def interrupted(*args, **kwargs):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(pipeline, "fit_editor", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        pipeline.fit(config, directory, tmp_path, model)
    assert lock.read_text() == "parent-owned" and model.training and model.gain.requires_grad
    assert (tmp_path / "train-cache").is_dir() and (tmp_path / "scorer.pt").is_file()
    assert not (tmp_path / "training.json").exists()
    with pytest.raises(FileExistsError, match="new run directory"):
        pipeline.fit(config, directory, tmp_path, model)
    with pytest.raises(FileNotFoundError):
        pipeline.evaluate(config, directory, tmp_path, model)


def test_training_split_seed_reuse_is_rejected_before_encoding(tmp_path, monkeypatch):
    config = _config()
    config["counts"].update(editor=1, scorer=1)
    for split in ("editor", "scorer"):
        root = _synthetic_root(split, 0, config, split_seed=123)
        save_root(tmp_path / "data" / split / "root.npz", root)
    monkeypatch.setattr(SyntheticAdapter, "encode", _forbidden)
    with pytest.raises(ValueError, match="Split seeds must be distinct"):
        pipeline.fit(config, tmp_path / "data", tmp_path / "run", SyntheticAdapter())
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("corruption", ["missing", "extra", "duplicate", "wrong_split", "nonfinite", "pickle"])
def test_fit_uses_full_roster_and_strict_load_root(dataset, tmp_path, monkeypatch, corruption):
    directory, config = dataset
    config = deepcopy(config)
    config["counts"].update(editor=1, scorer=1)
    for split in ("editor", "scorer"):
        target = tmp_path / "data" / split / "root.npz"
        target.parent.mkdir(parents=True)
        shutil.copyfile(next((directory / split).glob("*.npz")), target)
    path = tmp_path / "data" / "editor" / "root.npz"
    if corruption == "missing":
        path.unlink()
    elif corruption in ("extra", "duplicate"):
        shutil.copyfile(path, path.with_name("copy.npz"))
        if corruption == "duplicate":
            config["counts"]["editor"] = 2
    elif corruption == "wrong_split":
        shutil.copyfile(tmp_path / "data" / "scorer" / "root.npz", path)
    else:
        root = load_root(path)
        if corruption == "nonfinite":
            # Even an unselected future horizon must be strictly validated.
            root.states_future[1, -1, -1, 0] = np.nan
        metadata = np.array(root.metadata, dtype=object) if corruption == "pickle" else np.array(json.dumps(root.metadata))
        np.savez_compressed(path, **root.arrays(), metadata=metadata)
    monkeypatch.setattr(SyntheticAdapter, "encode", _forbidden)
    with pytest.raises(ValueError):
        pipeline.fit(config, tmp_path / "data", tmp_path / "run", SyntheticAdapter())
    assert not (tmp_path / "run").exists()


def test_evaluate_full_roster_no_fitting_chunking_controls_and_safe_loads(trained, monkeypatch):
    directory, config, run, metadata = trained
    before = _digest_tree(run)
    model = SyntheticAdapter()
    original_load, original_editor, original_random = torch.load, pipeline.load_editor, pipeline.matched_random
    loads, editor_calls, random_calls, live_futures = [], defaultdict(list), [], []
    original_rollout = model.rollout

    def safe_load(path, *args, **kwargs):
        assert kwargs.get("weights_only") is True
        loads.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    def tracked_editor(path, device):
        editor = original_editor(path, device=device)
        forward = editor.forward

        def tracked_forward(history, request):
            assert not torch.is_grad_enabled()
            assert history.shape == (1, 3, 6) and request.shape == (1, 6)
            editor_calls[editor.name].append((history.clone(), request.clone()))
            return forward(history, request)

        editor.forward = tracked_forward
        return editor

    def tracked_random(editor, history, request, generator):
        assert editor.name == "mlp"
        random_calls.append((generator.initial_seed(), history.clone()))
        return original_random(editor, history, request, generator)

    def streaming_rollout(history, actions):
        assert not any(reference() is not None for reference in live_futures)
        future = original_rollout(history, actions)
        live_futures.append(weakref.ref(future))
        return future

    monkeypatch.setattr(torch, "load", safe_load)
    monkeypatch.setattr(pipeline, "load_editor", tracked_editor)
    monkeypatch.setattr(pipeline, "matched_random", tracked_random)
    monkeypatch.setattr(model, "rollout", streaming_rollout)
    monkeypatch.setattr(pipeline, "fit_editor", _forbidden)
    monkeypatch.setattr(PhysicalReadout, "fit", _forbidden)
    monkeypatch.setattr(PhysicalReadout, "fit_features", _forbidden)
    rng = torch.random.get_rng_state().clone()
    records = pipeline.evaluate(config, directory, run, model, primary_method="mlp")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert _digest_tree(run) == before
    assert Counter(loads) == Counter(["scorer.pt", *[f"{name}.pt" for name in config["editors"]]])
    assert len(records) == 2 * 3 * 9
    assert all(set(("root_id", "split", "appearance", "method", *METRICS)) <= row.keys() for row in records)
    assert all(np.isfinite([row[key] for key in (*METRICS, "history_fidelity", "specificity_loss", "latent_norm")]).all() for row in records)
    assert all(row["split"] == "validation" for row in records)
    assert model.training and model.gain.requires_grad and model.gain.grad is None
    assert all(c["frames"] <= 2 and not c["grad_enabled"] and c["frozen"] for c in model.encode_calls)
    assert set(np.concatenate([c["markers"] for c in model.encode_calls])[:, 0]) == {0, 1, 2}
    assert all(not c["grad_enabled"] and not c["requires_grad"] and not c["training"] and c["frozen"] for c in model.rollout_calls)
    assert len(model.rollout_calls) == 2 * 3 * 2 * 9  # factual shared, not repeated for each method
    assert len(editor_calls["mlp"]) == 2 * 3 * 3  # primary, reversed, zero; NOT once per candidate
    assert all(len(editor_calls[name]) == 2 * 3 for name in ("inverse", "ridge", "rollout"))
    assert len(random_calls) == 2 * 3
    assert len({seed for seed, _ in random_calls[:3]}) == len({seed for seed, _ in random_calls[3:]}) == 1
    assert random_calls[0][0] != random_calls[3][0]

    roots = [load_root(p) for p in sorted((directory / "validation").glob("*.npz"))]
    cursor = 0
    with torch.no_grad():
        for ri, root in enumerate(roots):
            for appearance in range(3):
                expected_history = SyntheticAdapter().encode(root.context[appearance, 0:1])
                calls = editor_calls["mlp"][3 * (ri * 3 + appearance):3 * (ri * 3 + appearance + 1)]
                for (history, request), sign in zip(calls, (1, -1, 0)):
                    torch.testing.assert_close(history, expected_history, rtol=0, atol=0)
                    torch.testing.assert_close(request, torch.from_numpy(root.request).float()[None] * sign)
                for start in range(0, 3, 2):
                    for call in model.rollout_calls[cursor:cursor + 9]:
                        torch.testing.assert_close(call["actions"], torch.from_numpy(root.actions[start:start + 2]), rtol=0, atol=0)
                    cursor += 9

    for root in roots:
        rows = [r for r in records if r["root_id"] == root.metadata["root_id"]]
        assert len(rows) == 27 and {r["method"] for r in rows} == set(METHODS)
        for key in ("zero_effect_loss", "offset_effect_loss"):
            assert len({r[key] for r in rows}) == 1
        for appearance in APPEARANCES:
            cells = {r["method"]: r for r in rows if r["appearance"] == appearance}
            for key in ("readout_loss", "endpoint_oracle_regret"):
                assert len({r[key] for r in cells.values()}) == 1
            assert cells["native"]["history_fidelity"] == 0
            assert cells["identity"]["effect_loss"] == pytest.approx(cells["identity"]["zero_effect_loss"])
            for key in (*METRICS, "history_fidelity", "specificity_loss", "latent_norm"):
                assert cells["zero"][key] == pytest.approx(cells["identity"][key], abs=1e-12)
            assert cells["random"]["latent_norm"] == pytest.approx(cells["mlp"]["latent_norm"], rel=2e-5, abs=1e-6)
            if np.array_equal(root.request, np.zeros(6)):
                assert all(row["regret"] == 0 for row in cells.values())
                assert all(row["zero_effect_loss"] == 0 for row in cells.values())
            else:
                assert cells["identity"]["history_fidelity"] > 0
    json.dumps(records, allow_nan=False)
    with pytest.warns(UserWarning, match="synthetic-test"):
        report = analyze_records(records, primary_method="mlp", resamples=8)
    assert report["n_roots"] == 2 and report["n_records"] == 54 and not report["outcome_filtering"]


def test_evaluation_metrics_match_independent_physical_and_real_endpoint_reference(trained):
    directory, config, run, _ = trained
    records = pipeline.evaluate(config, directory, run, SyntheticAdapter())
    root = load_root(sorted((directory / "validation").glob("*.npz"))[0])
    assert root.metadata["root_index"] == 10
    model = SyntheticAdapter().eval().requires_grad_(False)
    payload = torch.load(run / "scorer.pt", weights_only=True)
    scorer = PhysicalReadout((6,))
    scorer.load_state_dict(payload["state_dict"])
    editor = load_editor(run / "editors" / "ridge.pt")
    scales, steps = np.asarray(config["scales"]), [0, 1]
    tf, tc = root.states_future
    effect = tc[:, steps] - tf[:, steps]
    expected_zero = np.square(effect / scales).sum(-1).mean()
    entry_effect = root.states_context[1, -1] - root.states_context[0, -1]
    expected_offset = np.square((entry_effect - effect) / scales).sum(-1).mean()
    anchor_offset = np.square((root.request - effect) / scales).sum(-1).mean()
    assert expected_offset != pytest.approx(anchor_offset)

    with torch.no_grad():
        for ai, appearance in enumerate(APPEARANCES):
            cells = {r["method"]: r for r in records if r["root_id"] == root.metadata["root_id"] and r["appearance"] == appearance}
            history = model.encode(root.context[ai])
            goal = model.encode(root.goal_pixels[ai][None, None])[0, 0].expand(3, -1)
            actual = torch.stack([model.encode(root.future[ai, branch]) for branch in range(2)])
            readout = scorer(actual).double().numpy()
            actual_loss = np.square((readout[:, :, steps] - root.states_future[:, :, steps]) / scales).sum(-1).mean()
            oracle_cost = model.goal_cost(actual[1, :, -1], goal).numpy()
            oracle_regret = root.costs[1, np.argmin(oracle_cost)] - root.costs[1].min()
            assert np.argmin(oracle_cost) == 1
            pf_latent = model.rollout(history[0:1].expand(3, -1, -1), torch.from_numpy(root.actions))
            pf = scorer(pf_latent).double().numpy()
            edited = editor(history[0:1], torch.from_numpy(root.request).float()[None])
            for name, latent_history in (("native", history[1:2]), ("identity", history[0:1]), ("ridge", edited)):
                pc_latent = model.rollout(latent_history.expand(3, -1, -1), torch.from_numpy(root.actions))
                pc = scorer(pc_latent).double().numpy()
                expected_effect = np.square(((pc[:, steps] - pf[:, steps]) - effect) / scales).sum(-1).mean()
                expected_absolute = np.square((np.stack((pf[:, steps], pc[:, steps])) - root.states_future[:, :, steps]) / scales).sum(-1).mean()
                angles = circular_error_deg(np.stack((pf[:, steps, 4:], pc[:, steps, 4:])), root.states_future[:, :, steps, 4:])
                ranked = model.goal_cost(pc_latent[:, -1], goal).numpy()
                expected_regret = root.costs[1, np.argmin(ranked)] - root.costs[1].min()
                row = cells[name]
                assert row["effect_loss"] == pytest.approx(expected_effect, rel=1e-6, abs=1e-9)
                assert row["absolute_loss"] == pytest.approx(expected_absolute, rel=1e-6, abs=1e-9)
                assert row["angle_error_deg"] == pytest.approx(angles.mean(), rel=1e-6, abs=1e-7)
                assert row["regret"] == expected_regret
                assert row["readout_loss"] == pytest.approx(actual_loss, rel=1e-6, abs=1e-9)
                assert row["endpoint_oracle_regret"] == oracle_regret
                assert row["zero_effect_loss"] == pytest.approx(expected_zero)
                assert row["offset_effect_loss"] == pytest.approx(expected_offset)
                true_anchor_effect = root.states_context[1, 0] - root.states_context[0, 0]
                pred_anchor_effect = (scorer(latent_history[:, 0]) - scorer(history[0:1, 0])).double().numpy()[0]
                expected_specificity = np.square(((pred_anchor_effect - true_anchor_effect) / scales)[true_anchor_effect == 0]).sum()
                assert row["specificity_loss"] == pytest.approx(expected_specificity, rel=1e-6, abs=1e-9)
                fidelity = ((latent_history.double() - history[1:2].double()).square().mean()
                            / history[1:2].double().square().mean()).item()
                assert row["history_fidelity"] == pytest.approx(fidelity)
            assert cells["native"]["regret"] != cells["native"]["endpoint_oracle_regret"]
            assert cells["native"]["absolute_loss"] != pytest.approx(cells["native"]["readout_loss"])


def test_random_seed_is_sha_derived_root_specific_and_appearance_independent(trained):
    directory, config, run, _ = trained
    first = pipeline.evaluate(config, directory, run, SyntheticAdapter())
    # Ambient global RNG changes must not affect random controls or loaded editors.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(938)
        torch.randn(31)
        second = pipeline.evaluate(config, directory, run, SyntheticAdapter())
    assert first == second
    root = first[0]["root_id"]
    payload = json.dumps(["wm_audit.matched_random.v1", 17, root], separators=(",", ":")).encode()
    assert pipeline._random_seed(17, root) == int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2 ** 63
    assert pipeline._random_seed(17, root) != pipeline._random_seed(18, root)


@pytest.mark.parametrize("problem", ["unfitted", "nonfinite", "shape", "non_tensor"])
def test_evaluation_rejects_bad_scorer_without_refitting(trained, tmp_path, monkeypatch, problem):
    directory, config, run, _ = trained
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    payload = torch.load(copy / "scorer.pt", weights_only=True)
    if problem == "unfitted":
        payload["state_dict"]["fitted"].fill_(False)
    elif problem == "nonfinite":
        payload["state_dict"]["weight"][0, 0] = float("nan")
    elif problem == "shape":
        payload["shape"] = torch.tensor([7])
    else:
        payload["state_dict"]["weight"] = [[1.0]]
    torch.save(payload, copy / "scorer.pt")
    before = _digest_tree(copy)
    monkeypatch.setattr(PhysicalReadout, "fit_features", _forbidden)
    monkeypatch.setattr(SyntheticAdapter, "encode", _forbidden)
    with pytest.raises(ValueError):
        pipeline.evaluate(config, directory, copy, SyntheticAdapter())
    assert _digest_tree(copy) == before


def test_nonfinite_rollout_aborts_instead_of_dropping_a_candidate_or_root(trained, monkeypatch):
    directory, config, run, _ = trained
    model = SyntheticAdapter()
    original = model.rollout
    before = _digest_tree(run)

    def bad_candidate(history, actions):
        result = original(history, actions)
        if (actions[:, 2, 0] > 0.25).any():
            result[0, -1, 0] = float("nan")
        return result

    monkeypatch.setattr(model, "rollout", bad_candidate)
    with pytest.raises(ValueError, match="finite"):
        pipeline.evaluate(config, directory, run, model)
    assert _digest_tree(run) == before and model.training and model.gain.requires_grad


def test_evaluation_rejects_training_split_seed_reuse(trained, tmp_path, monkeypatch):
    _, config, run, _ = trained
    for index in (10, 12):
        root = _synthetic_root("validation", index, config, split_seed=data.DEFAULT_SPLIT_SEEDS["editor"])
        save_root(tmp_path / "validation" / f"{index}.npz", root)
    monkeypatch.setattr(SyntheticAdapter, "encode", _forbidden)
    with pytest.raises(ValueError, match="overlap training split_seed"):
        pipeline.evaluate(config, tmp_path, run, SyntheticAdapter())


@pytest.mark.parametrize("kwargs", [{"split": "editor"}, {"split": "scorer"}, {"split": "../validation"},
                                    {"primary_method": "identity"}, {"primary_method": "not-fitted"}])
def test_invalid_evaluation_split_or_primary_fails_before_loading(trained, monkeypatch, kwargs):
    directory, config, run, _ = trained
    monkeypatch.setattr(pipeline, "load_root", _forbidden)
    with pytest.raises(ValueError):
        pipeline.evaluate(config, directory, run, SyntheticAdapter(), **kwargs)


@pytest.mark.parametrize("key,value", [("batch_size", True), ("rank", 0), ("steps", -1),
                                       ("editors", ["ridge", "ridge"]), ("editors", ["../escape"]),
                                       ("horizons", [1, True]), ("horizons", [1, 1]), ("horizons", [4]),
                                       ("scales", [True, 32, 32, 32, 1, 1]), ("scales", [0, 32, 32, 32, 1, 1])])
def test_invalid_configuration_fails_before_any_data_access(dataset, tmp_path, monkeypatch, key, value):
    directory, config = dataset
    config = deepcopy(config)
    config[key] = value
    monkeypatch.setattr(pipeline, "load_root", _forbidden)
    with pytest.raises(ValueError):
        pipeline.fit(config, directory, tmp_path, SyntheticAdapter())