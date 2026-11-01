"""Streaming integration of paired roots, frozen adapters, editors and metrics.

Only original-appearance editor/scorer roots are read by :func:`fit`. Editor
histories are CPU float32 NPY memmaps, retained in ``train-cache`` for inspection
(two approximately 1.8 GB files for 2,000 DINO roots). They are not a resume
mechanism: a completed OR interrupted fit requires a new run directory. The
caller owns run locking and result/provenance publication; this module publishes
its training artifacts exclusively and never overwrites a checkpoint.

The independent physical scorer sees actual context encodings, native emitted
futures, and actual future-image encodings, with simulator physical labels. Its
native-rollout calibration can absorb systematic model error. Consequently,
evaluation also reports actual-image readout loss on EVERY evaluation root,
appearance, branch and candidate, without filtering or refitting anything.

All physical losses sum scaled squared errors over features and average over
candidates and the configured future horizons. Absolute/angle/readout losses
also average the two branches. History fidelity is MSE against the native CF
history divided by its mean squared latent value (floor: float32 epsilon).
Specificity sums scaled anchor-effect errors only on truly unchanged anchor
coordinates; no non-target invariance is imposed after the anchor.

``seed`` is the explicit fitting seed, independent of data-generation seeds;
callers using a config's default seed should pass it to fit. Random controls use
that recorded fit seed plus a SHA256-derived root identity, identically across
appearances. Wrong-request controls reverse the request, NOT shuffle it.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from .data import load_root, validate_split_disjointness
from .fitting import PhysicalReadout, fit_editor, load_editor, matched_random, save_editor
from .statistics import APPEARANCES, candidate_regret, circular_error_deg, physical_effect_loss
from .progress import note, progress, stage


__all__ = ["fit", "evaluate"]

_EDITORS = ("inverse", "ridge", "mlp", "rollout")
_EVALUATION_SPLITS = ("validation", "pilot", "confirmation")
_SCHEMA = "wm_audit.training.v1"
_CALIBRATION_CAVEAT = (
    "Native rollout calibration may absorb systematic model error. Actual-image "
    "readout_loss is reported separately on the untouched full evaluation roster; "
    "neither diagnostic filters roots."
)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _configuration(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")
    required = {"counts", "horizon", "candidates", "horizons", "scales", "editors",
                "rank", "steps", "batch_size"}
    if required - config.keys():
        raise ValueError(f"Missing pipeline configuration: {sorted(required - config.keys())}.")
    for key in ("horizon", "candidates", "rank", "batch_size", "steps"):
        _integer(config[key], key, 0 if key == "steps" else 1)
    if not isinstance(config["counts"], dict):
        raise ValueError("counts must map split names to root counts.")
    for split, count in config["counts"].items():
        _integer(count, f"counts.{split}")
    horizons = config["horizons"]
    if not isinstance(horizons, (list, tuple)) or not horizons:
        raise ValueError("horizons must be a nonempty list of distinct future steps.")
    for step in horizons:
        if _integer(step, "horizon step", 1) > config["horizon"]:
            raise ValueError("horizons must not exceed the configured horizon.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be distinct.")
    scales = np.asarray(config["scales"])
    if (scales.shape != (6,) or scales.dtype.kind not in "iuf"
            or any(isinstance(x, (bool, np.bool_)) for x in np.asarray(config["scales"], dtype=object).flat)
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("scales must contain six finite, positive physical-unit divisors.")
    names = config["editors"]
    if (not isinstance(names, (list, tuple)) or not names
            or any(name not in _EDITORS for name in names) or len(set(names)) != len(names)):
        raise ValueError(f"editors must be a nonempty, unique selection from {_EDITORS}.")
    if "seed" in config and _integer(config["seed"], "config seed") >= 2 ** 63:
        raise ValueError("config seed must fit a nonnegative int64.")
    return _json_copy(config)


def _model_spec(model):
    if not isinstance(model, nn.Module) or any(
        not callable(getattr(model, name, None)) for name in ("encode", "rollout", "goal_cost")
    ):
        raise TypeError("model must be an nn.Module adapter with encode, rollout and goal_cost.")
    shape = PhysicalReadout(getattr(model, "latent_shape", ())).latent_shape
    tensors = (*model.parameters(), *model.buffers())
    device = torch.device(getattr(model, "device", tensors[0].device if tensors else "cpu"))
    if any(t.device != device for t in tensors):
        raise ValueError("All adapter parameters and buffers must already be on the same device.")
    if not isinstance(getattr(model, "metadata", None), dict):
        raise ValueError("The adapter must expose JSON-compatible model.metadata.")
    return shape, device, _json_copy(model.metadata)


@contextmanager
def _frozen(model):
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        model.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def _checked_tensor(value, shape, label):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{label} must be a floating tensor of shape {tuple(shape)}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite; no roots can be discarded.")
    return value


@torch.no_grad()
def _encode(model, pixels, shape, batch_size):
    """Bound encoder batches in FRAMES, not roots times time steps."""
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.shape[-3:] != (224, 224, 3):
        raise ValueError("Encoding requires actual uint8 RGB frames, not precomputed labels/features.")
    prefix = pixels.shape[:-3]
    frames = pixels.reshape(-1, 224, 224, 3)
    if not len(frames):
        raise ValueError("Cannot encode an empty frame roster.")
    chunks = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size, None]
        z = _checked_tensor(model.encode(batch), (len(batch), 1, *shape), "model.encode output")
        chunks.append(z[:, 0])
    return torch.cat(chunks).reshape(*prefix, *shape)


def _rollout(model, history, actions, shape, *, recorded_actions=False):
    actions = torch.as_tensor(actions, dtype=torch.float32, device=history.device)
    expected = (len(history), actions.shape[1] - 2, *shape)
    kwargs = {"recorded_actions": True} if recorded_actions else {}
    return _checked_tensor(model.rollout(history, actions, **kwargs), expected, "model.rollout output")


def _cost(model, endpoints, goal):
    goal = goal.expand(len(endpoints), *goal.shape)
    value = _checked_tensor(model.goal_cost(endpoints, goal), (len(endpoints),), "model.goal_cost output")
    return value.detach().cpu().double().numpy()


def _paths(data_dir, split, config):
    if split not in config["counts"]:
        raise ValueError(f"Missing configured count for {split}.")
    paths = sorted((Path(data_dir) / split).glob("*.npz"))
    expected = config["counts"][split]
    if not paths or len(paths) != expected:
        raise ValueError(f"{split} requires its full nonempty roster: expected {expected}, found {len(paths)}.")
    return paths


def _root(path, split, config):
    root = load_root(path)  # Strict pickle-free schema, ALL arrays checked, no outcome filtering.
    if root.metadata["split"] != split:
        raise ValueError(f"Root split does not match directory {split}: {path}.")
    if (root.actions.shape != (config["candidates"], 2 + config["horizon"], 10)
            or root.metadata["config"]["horizon"] != config["horizon"]):
        raise ValueError("Root candidate bank/horizon does not match the configuration.")
    return root


def _identity(root):
    meta = root.metadata
    return {"root_id": meta["root_id"], "root_index": meta["root_index"],
            "root_seed": meta["root_seed"], "split_seed": meta["split_seed"],
            "environment_seed": meta["stream_seeds"]["environment"]}


def _scan(roster, config):
    """Validate disjointness without retaining compressed roots' large RGB arrays."""
    identities = {split: [] for split in roster}

    def roots():
        for split, paths in roster.items():
            for path in progress(paths, desc=f"Validate {split} archives (CPU)", unit="root"):
                root = _root(path, split, config)
                identities[split].append(_identity(root))
                yield root

    validate_split_disjointness(roots())
    return identities


def _training_path_exists(path):
    return path.exists() or path.is_symlink()


def _fresh_run(run_dir):
    for name in ("training.json", "scorer.pt", "editors", "train-cache"):
        if _training_path_exists(run_dir / name):
            raise FileExistsError(f"Training artifact already exists: {run_dir / name}. "
                                  "Completed/incomplete fits require a new run directory.")


def _publish(path, writer):
    """Atomic exclusive publication, including against a concurrent writer."""
    if _training_path_exists(path):
        raise FileExistsError(f"Refusing to overwrite {path}.")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_scorer(path, scorer):
    payload = {"shape": torch.tensor(scorer.latent_shape, dtype=torch.int64),
               "state_dict": {k: v.detach().cpu().contiguous() for k, v in scorer.state_dict().items()}}
    _publish(path, lambda temporary: torch.save(payload, temporary))


def _finite_module(module):
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in module.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite tensors.")
    if not module.fitted.item():
        raise ValueError("Checkpoint has not been fitted.")
    return module


def _load_scorer(path, shape, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {"shape", "state_dict"}
            or not isinstance(payload["shape"], torch.Tensor)
            or payload["shape"].dtype != torch.int64 or payload["shape"].ndim != 1
            or tuple(payload["shape"].tolist()) != shape
            or not isinstance(payload["state_dict"], dict)
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                   for k, v in payload["state_dict"].items())):
        raise ValueError("Invalid tensor-only physical scorer checkpoint or latent shape.")
    scorer = PhysicalReadout(shape)
    scorer.load_state_dict(payload["state_dict"], strict=True)
    _finite_module(scorer)
    if (scorer.input_std <= 0).any() or (scorer.output_scale <= 0).any():
        raise ValueError("Invalid physical scorer scales.")
    return scorer.to(device).eval()


def _scaled_loss(predicted, actual, scales):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = float(np.square((np.asarray(predicted, dtype=np.float64) - actual) / scales).sum(-1).mean())
    if not math.isfinite(value):
        raise ValueError("Nonfinite physical loss; no observations were discarded.")
    return value


@torch.no_grad()
def _fit_scorer(paths, identities, config, model, shape):
    scorer = PhysicalReadout(shape)
    steps = np.sort(config["horizons"]) - 1
    per_branch = 3 + 2 * len(steps)
    features = torch.empty(len(paths) * 2 * per_branch, scorer.feature_dim)
    labels = torch.empty(len(features), 6, dtype=torch.float64)
    surfaces = np.empty(len(features), dtype=np.int8)
    cursor, candidates = 0, []

    def append(z, y, surface):
        nonlocal cursor
        x = scorer.features(z).reshape(-1, scorer.feature_dim).cpu()
        count = len(x)
        features[cursor:cursor + count] = x
        labels[cursor:cursor + count] = torch.as_tensor(np.asarray(y).reshape(-1, 6))
        surfaces[cursor:cursor + count] = surface
        cursor += count

    for path, identity in progress(zip(paths, identities), total=len(paths),
                                   desc="Extract physical readout features", unit="root"):
        root = _root(path, "scorer", config)
        if _identity(root) != identity:
            raise ValueError("Scorer root identity changed after validation.")
        # Fixed from metadata alone, BEFORE inspecting any candidate outcome.
        candidate = root.metadata["root_index"] % config["candidates"]
        candidates.append(candidate)
        histories = _encode(model, root.context[0], shape, config["batch_size"])
        for branch in range(2):
            append(histories[branch], root.states_context[branch], 0)
            emitted = _rollout(model, histories[branch:branch + 1], root.actions[candidate:candidate + 1], shape)
            y = root.states_future[branch, candidate, steps]
            append(emitted[0, steps.tolist()], y, 1)
            del emitted
            actual = _encode(model, root.future[0, branch, candidate, steps], shape, config["batch_size"])
            append(actual, y, 2)
            del actual
    if cursor != len(features):
        raise RuntimeError("Incomplete physical scorer feature roster.")
    with stage("fit physical readout ridge (CPU)"):
        scorer.fit_features(features, labels)
    predicted = (((features - scorer.input_mean) / scorer.input_std) @ scorer.weight + scorer.bias) * scorer.output_scale
    predicted, actual = predicted.double().numpy(), labels.numpy()
    scales = np.asarray(config["scales"], dtype=np.float64)
    log = {"n_roots": len(paths), "n_rows": len(features), "feature_dim": scorer.feature_dim,
           "grid": scorer.grid, "ridge": scorer.ridge, "candidate_indices": candidates,
           "training_loss": _scaled_loss(predicted, actual, scales),
           "surface_losses": {name: _scaled_loss(predicted[surfaces == i], actual[surfaces == i], scales)
                              for i, name in enumerate(("actual_context", "native_rollout", "actual_future"))},
           "calibration_caveat": _CALIBRATION_CAVEAT}
    return _finite_module(scorer), log


def fit(config: dict, data_dir: Path, run_dir: Path, model, *, seed=0) -> dict:
    """Fit independent scorer/all configured editors; return saved training metadata.

    No validation, pilot or confirmation archive is opened. Rollout-editor
    actions have K=1, selected by root_index % K, and native CF targets are
    computed on demand by fit_editor, preserving gradients to edited histories.
    The completion marker training.json is published LAST. No resume is offered.
    """
    config = _configuration(config)
    seed = _integer(seed, "seed")
    if seed >= 2 ** 63:
        raise ValueError("seed must fit a nonnegative int64.")
    run_dir = Path(run_dir)
    _fresh_run(run_dir)
    shape, device, model_metadata = _model_spec(model)
    roster = {split: _paths(data_dir, split, config) for split in ("editor", "scorer")}
    identities = _scan(roster, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = run_dir / "train-cache"
    cache.mkdir()  # Also marks an interrupted fit; never truncate old memmaps.
    (run_dir / "editors").mkdir()
    n = len(roster["editor"])
    history_shape = (n, 3, *shape)
    histories = np.lib.format.open_memmap(cache / "factual.npy", mode="w+", dtype=np.float32, shape=history_shape)
    targets = np.lib.format.open_memmap(cache / "counterfactual.npy", mode="w+", dtype=np.float32, shape=history_shape)
    requests, anchors = np.empty((n, 6)), np.empty((n, 6))
    actions = np.empty((n, 1, 2 + config["horizon"], 10), dtype=np.float32)
    chosen = []
    logs = {}
    with _frozen(model):
        with torch.no_grad():
            for index, (path, identity) in enumerate(progress(
                    zip(roster["editor"], identities["editor"]), total=n,
                    desc=f"Encode editor pairs on {device}", unit="root")):
                root = _root(path, "editor", config)
                if _identity(root) != identity:
                    raise ValueError("Editor root identity changed after validation.")
                candidate = root.metadata["root_index"] % config["candidates"]
                chosen.append(candidate)
                z = _encode(model, root.context[0], shape, config["batch_size"]).cpu().float()
                _checked_tensor(z, (2, 3, *shape), "cached context")
                histories[index], targets[index] = z[0].numpy(), z[1].numpy()
                requests[index], anchors[index] = root.request, root.states_context[0, 0]
                actions[index, 0] = root.actions[candidate]
            histories.flush()
            targets.flush()
        scorer, logs["scorer"] = _fit_scorer(roster["scorer"], identities["scorer"], config, model, shape)
        _save_scorer(run_dir / "scorer.pt", scorer)
        del scorer
        options = {key: config[key] for key in ("lr", "basis_samples", "context_dim", "hidden_dim", "ridge", "rollout_weight")
                   if key in config}
        for name in config["editors"]:
            note(f"Fit editor {name}: CPU basis/linear initialization; gradient updates on {device} if applicable")
            extra = {"model": model, "actions": actions} if name == "rollout" else {}
            if name == "inverse":
                extra["anchor_y"] = anchors
            editor, logs[name] = fit_editor(
                name, histories, requests, targets, rank=config["rank"], steps=config["steps"],
                batch_size=config["batch_size"], seed=seed, device=device, **options, **extra,
            )
            _finite_module(editor)
            _publish(run_dir / "editors" / f"{name}.pt", lambda temporary: save_editor(temporary, editor))
            del editor
    metadata = {
        "schema": _SCHEMA, "config": config, "seed": seed, "model": model_metadata,
        "latent_shape": list(shape), "root_ids": {s: [r["root_id"] for r in rows] for s, rows in identities.items()},
        "training_roots": identities, "fit_logs": logs,
        "editor_candidate_indices": chosen,
        "train_cache": {"files": ["train-cache/factual.npy", "train-cache/counterfactual.npy"],
                        "shape": list(history_shape), "dtype": "float32", "retained": True,
                        "array_bytes": int(histories.nbytes + targets.nbytes), "resume_supported": False},
        "protocol": {"training_appearance": "original", "candidate_selection": "root_index % K (pre-outcome)",
                     "editor_future_targets": "native CF rollout on demand; never actual future encodings",
                     "scorer_labels": "simulator physical states; context plus configured future horizons",
                     "scorer_calibration_caveat": _CALIBRATION_CAVEAT,
                     "wrong_request": "direction-reversed (-request), not shuffled",
                     "random_control": "same fitted basis and full-history norm; SHA256(seed, root_id), same across appearances",
                     "offset_baseline": "states_context[1,-1] - states_context[0,-1], held constant"},
    }
    metadata = _json_copy(metadata)
    _publish(run_dir / "training.json", lambda temporary: temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"))
    return metadata


def _random_seed(seed, root_id):
    encoded = json.dumps(["wm_audit.matched_random.v1", seed, root_id], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2 ** 63)


def _histories(histories, request, editors, primary_method, seed, root_id):
    factual, native_cf = histories[0:1], histories[1:2]
    edited = {"native": native_cf, "identity": factual}
    # Each editor consumes ONE entire factual history, never future/test CF targets.
    for name, editor in editors.items():
        edited[name] = editor(factual, request)
    primary = editors[primary_method]
    edited["random"] = matched_random(primary, factual, request,
                                       torch.Generator(device="cpu").manual_seed(_random_seed(seed, root_id)))
    edited["wrong"] = primary(factual, -request)
    edited["zero"] = primary(factual, torch.zeros_like(request))
    for name, history in edited.items():
        _checked_tensor(history, factual.shape, f"{name} edited history")
    return edited


def _history_metrics(edited, factual, native_cf, scorer, root, scales):
    anchor_effect = root.states_context[1, 0] - root.states_context[0, 0]
    unchanged = anchor_effect == 0
    factual_anchor = scorer(factual[:, 0]).cpu().double().numpy()[0]
    reference = native_cf.double()
    denominator = max(reference.square().mean().item(), torch.finfo(torch.float32).eps)
    result = {}
    for name, history in edited.items():
        anchor = scorer(history[:, 0]).cpu().double().numpy()[0]
        error = ((anchor - factual_anchor) - anchor_effect) / scales
        result[name] = {
            "history_fidelity": (history.double() - reference).square().mean().item() / denominator,
            "specificity_loss": float(np.square(error[unchanged]).sum()),
            "latent_norm": (history.double() - factual.double()).norm().item(),
        }
    return result


def _actual_metrics(root, appearance, model, scorer, goal, config, shape, scales):
    steps = np.sort(config["horizons"]) - 1
    encoded_steps = sorted({*steps.tolist(), config["horizon"] - 1})
    score_positions = [encoded_steps.index(step) for step in steps]
    endpoint_position = encoded_steps.index(config["horizon"] - 1)
    k, total = config["candidates"], 0.0
    costs = np.empty(k, dtype=np.float64)
    for branch in range(2):
        for start in range(0, k, config["batch_size"]):
            stop = min(start + config["batch_size"], k)
            images = root.future[appearance, branch, start:stop][:, encoded_steps]
            actual_z = _encode(model, images, shape, config["batch_size"])
            predicted = scorer(actual_z[:, score_positions]).cpu().double().numpy()
            truth = root.states_future[branch, start:stop][:, steps]
            total += _scaled_loss(predicted, truth, scales) * (stop - start) / (2 * k)
            if branch == 1:
                costs[start:stop] = _cost(model, actual_z[:, endpoint_position], goal)
            del actual_z
    return {"readout_loss": total, "endpoint_oracle_regret": candidate_regret(root.costs[1], costs)}


def _future_metrics(root, edited, goal, model, scorer, config, shape, scales):
    """Keep one candidate chunk's latent futures, plus compact metrics/cost banks."""
    k, steps = config["candidates"], np.sort(config["horizons"]) - 1
    totals = {name: {"effect_loss": 0.0, "absolute_loss": 0.0, "angle_error_deg": 0.0} for name in edited}
    costs = {name: np.empty(k, dtype=np.float64) for name in edited}
    factual = edited["identity"]
    for start in range(0, k, config["batch_size"]):
        stop = min(start + config["batch_size"], k)
        count = stop - start
        actions = root.actions[start:stop]
        expanded = factual.expand(count, *factual.shape[1:])
        native_factual = _rollout(model, expanded, actions, shape)
        pf = scorer(native_factual).cpu().double().numpy()
        factual_cost = _cost(model, native_factual[:, -1], goal)
        del native_factual
        tf, tc = root.states_future[0, start:stop], root.states_future[1, start:stop]
        factual_absolute = _scaled_loss(pf[:, steps], tf[:, steps], scales)
        factual_angle = float(np.mean(circular_error_deg(pf[:, steps, 4:], tf[:, steps, 4:])))
        for name, history in edited.items():
            if name == "identity":
                pc, predicted_cost = pf, factual_cost
            else:
                future = _rollout(model, history.expand(count, *history.shape[1:]), actions, shape)
                pc = scorer(future).cpu().double().numpy()
                predicted_cost = _cost(model, future[:, -1], goal)
                del future
            effect = physical_effect_loss(pf, pc, tf, tc, scales, config["horizons"])
            absolute = (factual_absolute + _scaled_loss(pc[:, steps], tc[:, steps], scales)) / 2
            angle = (factual_angle + float(np.mean(circular_error_deg(pc[:, steps, 4:], tc[:, steps, 4:])))) / 2
            for field, value in (("effect_loss", effect), ("absolute_loss", absolute), ("angle_error_deg", angle)):
                totals[name][field] += value * count / k
            costs[name][start:stop] = predicted_cost
    for name in edited:
        totals[name]["regret"] = candidate_regret(root.costs[1], costs[name])
    return totals


@torch.no_grad()
def evaluate(config, data_dir, run_dir, model, split="validation", *, primary_method="ridge") -> list[dict]:
    """Return the complete root x appearance x method roster; never fit or save.

    Loads training.json, scorer.pt and editors/{name}.pt, using weights_only=True
    for tensors. Real-image diagnostics are shared across method rows, physical
    baselines across all appearances. Even tied/failed native banks are retained;
    missing, malformed, nonfinite or overlapping data/checkpoints raise instead.
    """
    config = _configuration(config)
    if split not in _EVALUATION_SPLITS:
        raise ValueError(f"Evaluation split must be one of {_EVALUATION_SPLITS}.")
    if primary_method not in config["editors"]:
        raise ValueError("primary_method must be one of the configured fitted editors.")
    run_dir = Path(run_dir)
    with (run_dir / "training.json").open(encoding="utf-8") as stream:
        training = json.load(stream)
    shape, device, _ = _model_spec(model)
    if training.get("schema") != _SCHEMA or training.get("latent_shape") != list(shape):
        raise ValueError("Incompatible training metadata or adapter latent shape.")
    seed = _integer(training["seed"], "training seed")
    paths = _paths(data_dir, split, config)
    identities = _scan({split: paths}, config)[split]
    trained = [root for roots in training["training_roots"].values() for root in roots]
    for field in ("root_id", "root_seed", "environment_seed", "split_seed"):
        if {r[field] for r in identities} & {r[field] for r in trained}:
            raise ValueError(f"Evaluation roots overlap training {field} roster.")
    scorer = _load_scorer(run_dir / "scorer.pt", shape, device)
    editors = {}
    for name in config["editors"]:
        editor = _finite_module(load_editor(run_dir / "editors" / f"{name}.pt", device=device))
        if editor.name != name or editor.latent_shape != shape:
            raise ValueError(f"Incompatible {name} editor checkpoint.")
        editors[name] = editor
    scales = np.asarray(config["scales"], dtype=np.float64)
    records = []
    with _frozen(model):
        note(f"Evaluate {split} on {device}: {len(paths)} roots x {len(APPEARANCES)} appearances; no training")
        for path, identity in progress(zip(paths, identities), total=len(paths),
                                       desc=f"Evaluate {split} on {device}", unit="root"):
            root = _root(path, split, config)
            if _identity(root) != identity:
                raise ValueError("Evaluation root identity changed after validation.")
            tf, tc = root.states_future
            zeros = np.zeros_like(tf)
            offset = np.broadcast_to(root.states_context[1, -1] - root.states_context[0, -1], tf.shape)
            baselines = {
                "zero_effect_loss": physical_effect_loss(zeros, zeros, tf, tc, scales, config["horizons"]),
                "offset_effect_loss": physical_effect_loss(zeros, offset, tf, tc, scales, config["horizons"]),
            }
            for appearance_index, appearance in enumerate(APPEARANCES):
                note(f"{split} root {identity['root_index']} / {appearance}: encode, readout and candidate rollouts")
                histories = _encode(model, root.context[appearance_index], shape, config["batch_size"])
                goal = _encode(model, root.goal_pixels[appearance_index:appearance_index + 1], shape, config["batch_size"])[0]
                request = torch.as_tensor(root.request, dtype=histories.dtype, device=device).unsqueeze(0)
                edited = _histories(histories, request, editors, primary_method, seed, identity["root_id"])
                immediate = _history_metrics(edited, histories[0:1], histories[1:2], scorer, root, scales)
                actual = _actual_metrics(root, appearance_index, model, scorer, goal, config, shape, scales)
                future = _future_metrics(root, edited, goal, model, scorer, config, shape, scales)
                for method in edited:
                    record = {"root_id": identity["root_id"], "split": split, "appearance": appearance,
                              "method": method, **future[method], **baselines, **actual, **immediate[method]}
                    for key, value in record.items():
                        if key not in ("root_id", "split", "appearance", "method") and (
                            not math.isfinite(value) or value < 0
                        ):
                            raise ValueError(f"Nonfinite/negative {key}; the full evaluation roster is required.")
                    records.append(record)
    expected = len(paths) * len(APPEARANCES) * (len(editors) + 5)
    if len(records) != expected:
        raise RuntimeError("Incomplete evaluation roster.")
    return records