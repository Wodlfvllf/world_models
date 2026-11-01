"""Exploratory prerequisites, deliberately separate from the locked audit.

Actual-image readouts never see predicted latents or editor targets. Selection
uses original-appearance validation only. The existing pilot is diagnostic,
NOT fresh confirmation. Prediction checks use no learned physical readout.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch

from .fitting import PhysicalReadout
from .pipeline import _configuration, _encode, _frozen, _model_spec, _paths, _root, _rollout, _scan
from .progress import note, progress, stage
from .provenance import code_hashes, digest, runtime, write_json
from .statistics import APPEARANCES

COORDINATES = ("pusher_x", "pusher_y", "block_x", "block_y", "sin_angle", "cos_angle")
METHODS = ("native", "persistence", "zero_future_actions")


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def interval(values, *, resamples=5000, seed=0):
    """Pointwise bootstrap of equally weighted root/episode means, bounded RAM."""
    _positive_int(resamples, "resamples")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite cluster means")
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, resamples, 256):
        indices = rng.integers(len(values), size=(min(256, resamples - start), len(values)))
        estimates.extend(values[indices].mean(1).tolist())
    return {"estimate": float(values.mean()), "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "clusters": len(values)}


def physical_metrics(predicted, actual, scales):
    predicted, actual = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 6 or not len(actual):
        raise ValueError("Physical predictions and labels must be matching nonempty [N,6]")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite physical diagnostic; no samples may be discarded")
    delta = predicted - actual
    angle = np.arctan2(predicted[:, 4], predicted[:, 5]) - np.arctan2(actual[:, 4], actual[:, 5])
    degrees = np.abs(np.arctan2(np.sin(angle), np.cos(angle))) * 180 / np.pi
    undefined = np.linalg.norm(predicted[:, 4:], axis=1) < 1e-8
    degrees[undefined] = 180.0  # Worst error, never discard an undefined direction.
    result = {"scaled_loss": float(np.square(delta / scales).sum(1).mean()),
              "pusher_distance": float(np.linalg.norm(delta[:, :2], axis=1).mean()),
              "block_distance": float(np.linalg.norm(delta[:, 2:4], axis=1).mean()),
              "angle_mae_deg": float(degrees.mean()), "undefined_angle_rate": float(undefined.mean())}
    for i, name in enumerate(COORDINATES):
        result[f"{name}_mae"] = float(np.abs(delta[:, i]).mean())
        result[f"{name}_mse"] = float(np.square(delta[:, i]).mean())
    return result


class RidgeFamily:
    """One CPU float64 eigensolve per feature grid; training-only scaling.

    Objective: sum squared residuals + alpha * squared weight norm. Intercept
    is not penalized. Use the smaller of primal/dual Gram matrices. This is
    algebraic fitting, not world-model training; no test features enter the solve.
    """
    def __init__(self, features, labels):
        x, y = torch.as_tensor(features, dtype=torch.float64), torch.as_tensor(labels, dtype=torch.float64)
        if x.ndim != 2 or y.shape != (len(x), 6) or len(x) < 2:
            raise ValueError("Ridge needs [N,D] features and [N,6] labels, N >= 2")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Nonfinite ridge training data")
        self.mean = x.mean(0)
        std = x.std(0, correction=0)
        self.std = torch.where(std > 1e-6, std, torch.ones_like(std))
        self.x = (x - self.mean) / self.std
        self.bias = y.mean(0)
        self.dual = len(x) < x.shape[1]
        gram = self.x @ self.x.T if self.dual else self.x.T @ self.x
        self.eigenvalues, self.vectors = torch.linalg.eigh(gram)
        self.eigenvalues.clamp_(min=0)
        rhs = y - self.bias if self.dual else self.x.T @ (y - self.bias)
        self.rhs = self.vectors.T @ rhs

    def fit(self, alpha):
        if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and positive")
        weights = self.vectors @ (self.rhs / (self.eigenvalues[:, None] + alpha))
        if self.dual:
            weights = self.x.T @ weights
        return {"mean": self.mean.clone(), "std": self.std.clone(),
                "weight": weights, "bias": self.bias.clone()}


def predict_readout(state, features):
    x = torch.as_tensor(features, dtype=torch.float64)
    value = ((x - state["mean"]) / state["std"]) @ state["weight"] + state["bias"]
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite readout prediction")
    return value.numpy()


def _new_output(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _roster(config, data, splits):
    manifest_path = Path(data) / "manifest-development.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("config") != config:
        raise ValueError("Diagnostic config differs from original generation manifest")
    paths = {split: _paths(data, split, config) for split in splits}
    _scan(paths, config)  # Includes cross-split root/environment disjointness.
    hashes = {str(manifest_path.resolve()): hashlib.sha256(raw).hexdigest()}
    for split, files in paths.items():
        actual = {}
        for path in progress(files, desc=f"Hash diagnostic {split} inputs", unit="root"):
            hashes[str(path.resolve())] = digest(path)
            actual[path.relative_to(data).as_posix()] = hashes[str(path.resolve())]
        if actual != manifest["files"].get(split):
            raise ValueError(f"Diagnostic {split} inputs differ from original generation manifest")
    return paths, hashes


def _load(path, split, config, hashes):
    if digest(path) != hashes[str(path.resolve())]:
        raise ValueError("Diagnostic input changed after roster validation")
    return _root(path, split, config)


def _image_rows(root, appearance, config, all_candidates):
    """Same actual-image sampling rule for fitting, selection and evaluation.

    Default fixed candidate = root_index % K avoids outcome-based sampling and
    balances 3 context + selected future horizons per branch. Optional all-bank
    prediction diagnostics are separate and do not alter this readout target.
    """
    steps = np.asarray(config["horizons"]) - 1
    candidates = list(range(config["candidates"])) if all_candidates else [root.metadata["root_index"] % config["candidates"]]
    pixels, labels, tags = [], [], []
    for branch in range(2):
        pixels.extend(root.context[appearance, branch])
        labels.extend(root.states_context[branch])
        tags.extend(["context"] * 3)
        for candidate in candidates:
            pixels.extend(root.future[appearance, branch, candidate, steps])
            labels.extend(root.states_future[branch, candidate, steps])
            tags.extend(["future"] * len(steps))
    return np.stack(pixels), np.stack(labels), np.array(tags)


def _features(model, pixels, shape, batch_size, grids):
    # Pool each encoder batch immediately: never retain an entire DINO root's patches.
    pools = {grid: PhysicalReadout(shape, grid=grid).to(model.device) for grid in grids}
    parts = {grid: [] for grid in grids}
    for start in range(0, len(pixels), batch_size):
        z = _encode(model, pixels[start:start + batch_size], shape, batch_size)
        for grid, pool in pools.items():
            parts[grid].append(pool.features(z).cpu().numpy())
    return {grid: np.concatenate(rows) for grid, rows in parts.items()}


def _collect(paths, split, config, hashes, model, shape, grids):
    chunks = {grid: [] for grid in grids}
    labels, ids = [], []
    for path in progress(paths, desc=f"Readout encode {split} original on {model.device}", unit="root"):
        root = _load(path, split, config, hashes)
        pixels, y, _ = _image_rows(root, 0, config, False)
        values = _features(model, pixels, shape, config["batch_size"], grids)
        for grid in grids:
            chunks[grid].append(values[grid])
        labels.append(y)
        ids.extend([root.metadata["root_id"]] * len(y))
    return {grid: np.concatenate(parts) for grid, parts in chunks.items()}, np.concatenate(labels), np.array(ids)


def _metric_summary(records, config):
    groups = defaultdict(list)
    for row in records:
        groups[(row["appearance"], row["method"], row["surface"])].append(row)
    result = []
    for (appearance, method, surface), rows in groups.items():
        metrics = {key: interval([r["metrics"][key] for r in rows],
                                resamples=config["bootstrap_resamples"], seed=config["seed"])
                   for key in rows[0]["metrics"]}
        for key in COORDINATES:
            mse = metrics[f"{key}_mse"]
            metrics[f"{key}_rmse"] = {"estimate": math.sqrt(mse["estimate"]),
                                       "ci95": np.sqrt(mse["ci95"]).tolist(), "clusters": mse["clusters"]}
        result.append({"appearance": appearance, "method": method, "surface": surface, "metrics": metrics})
    return result


@torch.no_grad()
def readout_diagnostic(config, data, output, model, *, grids=(2, 4), alphas=(.01, 1., 100., 10000.)):
    config = _configuration(config)
    if any(type(g) is not int or g < 1 for g in grids) or not grids or len(set(grids)) != len(grids):
        raise ValueError("Feature grids must be distinct positive integers")
    if not alphas or len(set(alphas)) != len(alphas) or any(isinstance(a, bool) or not math.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Ridge strengths must be distinct finite positive numbers")
    shape, device, metadata = _model_spec(model)
    grids = (1,) if len(shape) == 1 else tuple(grids)
    for grid in grids:
        PhysicalReadout(shape, grid=grid)  # Validate spatial grids before creating outputs.
    output = _new_output(output)
    paths, hashes = _roster(config, data, ("scorer", "validation", "pilot"))
    with _frozen(model):
        train_x, train_y, train_ids = _collect(paths["scorer"], "scorer", config, hashes, model, shape, grids)
        val_x, val_y, val_ids = _collect(paths["validation"], "validation", config, hashes, model, shape, grids)
        candidates, selected_state, selected = [], None, None
        for grid in grids:
            with stage(f"readout grid {grid}: CPU float64 ridge eigensolve"):
                family = RidgeFamily(train_x[grid], train_y)
            for alpha in progress(alphas, desc=f"Select grid {grid} on original validation", unit="alpha"):
                state = family.fit(alpha)
                prediction = predict_readout(state, val_x[grid])
                by_root = [physical_metrics(prediction[val_ids == root], val_y[val_ids == root], config["scales"])["scaled_loss"]
                           for root in sorted(set(val_ids))]
                score = float(np.mean(by_root))
                row = {"grid": grid, "alpha": alpha, "validation_scaled_loss": score,
                       "training_metrics": physical_metrics(predict_readout(state, train_x[grid]), train_y, config["scales"])}
                candidates.append(row)
                if selected is None or (score, grid, alpha) < (selected["validation_scaled_loss"], selected["grid"], selected["alpha"]):
                    selected, selected_state = row, state
            del family
        mean = train_y.mean(0)
        # Selection is published before any pilot pixels are encoded.
        write_json(output / "selection.json", {"candidates": candidates, "selected": selected,
                   "selection_split": "validation", "selection_appearance": "original", "fit_split": "scorer"})
        torch.save({**selected_state, "grid": torch.tensor(selected["grid"]), "mean_baseline": torch.tensor(mean)}, output / "readout.pt")
        del train_x, val_x
        records, scatter = [], {appearance: [[], []] for appearance in APPEARANCES}
        for path in progress(paths["pilot"], desc=f"Readout evaluate pilot on {device}", unit="root"):
            root = _load(path, "pilot", config, hashes)
            for appearance_index, appearance in enumerate(APPEARANCES):
                pixels, actual, surfaces = _image_rows(root, appearance_index, config, False)
                features = _features(model, pixels, shape, config["batch_size"], (selected["grid"],))[selected["grid"]]
                predictions = {"selected_ridge": predict_readout(selected_state, features),
                               "train_mean": np.broadcast_to(mean, actual.shape)}
                scatter[appearance][0].append(actual)
                scatter[appearance][1].append(predictions["selected_ridge"])
                for method, predicted in predictions.items():
                    for surface in ("all", "context", "future"):
                        mask = np.ones(len(actual), dtype=bool) if surface == "all" else surfaces == surface
                        records.append({"root_id": root.metadata["root_id"], "appearance": appearance,
                                        "method": method, "surface": surface,
                                        "metrics": physical_metrics(predicted[mask], actual[mask], config["scales"])})
        summary = _metric_summary(records, config)
        contrasts = []
        original = {r["root_id"]: r["metrics"]["scaled_loss"] for r in records
                if r["appearance"] == "original" and r["surface"] == "all" and r["method"] == "selected_ridge"}
        for appearance in APPEARANCES:
            rows = {method: {r["root_id"]: r["metrics"]["scaled_loss"] for r in records
                             if r["appearance"] == appearance and r["surface"] == "all" and r["method"] == method}
                    for method in ("selected_ridge", "train_mean")}
            delta = [rows["selected_ridge"][key] - rows["train_mean"][key] for key in sorted(rows["train_mean"])]
            shift = [rows["selected_ridge"][key] - original[key] for key in sorted(original)]
            contrasts.append({"appearance": appearance,
                              "selected_minus_mean": interval(delta, resamples=config["bootstrap_resamples"], seed=config["seed"]),
                              "appearance_minus_original": interval(shift, resamples=config["bootstrap_resamples"], seed=config["seed"])})
        from .diagnostic_plots import readout_plots
        readout_plots(output, scatter, candidates)
    return _finish(output, "actual_image_readout", config, metadata, hashes,
                   selected=selected, selection_candidates=candidates, records=records, summary=summary,
                   contrasts=contrasts, training_roots=len(set(train_ids)), training_rows=len(train_y),
                   protocol={"fit_split": "scorer", "selection_split": "validation", "evaluation_split": "pilot",
                             "fit_appearance": "original", "selection_appearance": "original",
                             "samples": "both branches, all context frames + configured future horizons; candidate=root_index % K",
                             "prediction_latents_in_readout_training": False,
                             "features": "LeWM CLS or DINO spatial pooling (grid x grid); grid 4 retains finer location than grid 2",
                             "angle": "atan2(sin,cos); predicted norm <1e-8 receives 180 degrees error, no exclusions"})


def _finish(output, kind, config, model_metadata, input_hashes, **results):
    # The completion report is LAST. Backups only capture a completed diagnostic.
    report = {"schema": "wm_audit.diagnostic.v1", "kind": kind, "config": config,
              "model": model_metadata, "runtime": runtime(), "code_sha256": code_hashes(),
              "artifacts": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
              "input_sha256": input_hashes, "confirmatory": False,
              "caveat": "Exploratory diagnostics on development/previously inspected pilot data. No automatic hypothesis confirmation; no outcome filtering.",
              **results}
    write_json(output / "report.json", report)
    return {"output": str(output / "report.json"), "plots": str(output / "index.html"), "kind": kind}


@torch.no_grad()
def _prediction_rows(model, history, actual, actions, horizons, identity, appearance, branch, candidate):
    """Only future controls are zeroed; the two observed-history chunks stay intact."""
    shape = tuple(model.latent_shape)
    actions = np.asarray(actions, dtype=np.float32)
    zero = actions.copy()
    zero[2:] = 0
    outputs = {"native": _rollout(model, history[None], actions[None], shape)[0],
               "persistence": history[-1:].expand(len(actions) - 2, *shape),
               "zero_future_actions": _rollout(model, history[None], zero[None], shape)[0]}
    rows = []
    for index, horizon in enumerate(horizons):
        target = actual[index]
        errors = {method: float((future[horizon - 1] - target).square().mean().item()) for method, future in outputs.items()}
        if not all(math.isfinite(value) for value in errors.values()):
            raise ValueError("Nonfinite prediction error; cannot omit this sample")
        rows.append({"cluster": identity, "appearance": appearance, "branch": branch, "candidate": candidate,
                     "horizon": horizon, **errors})
    return rows


def prediction_summary(rows, config):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["appearance"], row["horizon"])][row["cluster"]].append(row)
    summary = []
    for (appearance, horizon), clusters in sorted(groups.items()):
        means = {method: [np.mean([r[method] for r in group]) for _, group in sorted(clusters.items())]
                 for method in METHODS}
        kwargs = {"resamples": config["bootstrap_resamples"], "seed": config["seed"]}
        summary.append({"appearance": appearance, "horizon": horizon,
                        "metrics": {method: interval(values, **kwargs) for method, values in means.items()},
                        "native_minus_persistence": interval(np.array(means["native"]) - means["persistence"], **kwargs),
                        "native_minus_zero_future_actions": interval(np.array(means["native"]) - means["zero_future_actions"], **kwargs)})
    return summary


@torch.no_grad()
def prediction_diagnostic(config, output, model, *, data=None, h5=None, episodes=128):
    if (data is None) == (h5 is None):
        raise ValueError("Choose exactly one source: paired data or recorded H5")
    _positive_int(episodes, "episodes")
    config = _configuration(config)
    shape, device, metadata = _model_spec(model)
    output = _new_output(output)
    rows, input_hashes, selection = [], {}, []
    steps = np.asarray(config["horizons"]) - 1
    with _frozen(model):
        if data is not None:
            paths, input_hashes = _roster(config, data, ("pilot",))
            for path in progress(paths["pilot"], desc=f"Prediction diagnostic on {device}", unit="root"):
                root = _load(path, "pilot", config, input_hashes)
                for appearance_index, appearance in enumerate(APPEARANCES):
                    note(f"Prediction {root.metadata['root_id']} / {appearance}: all branches and candidates")
                    history = _encode(model, root.context[appearance_index], shape, config["batch_size"])
                    for branch in range(2):
                        for candidate, actions in enumerate(root.actions):
                            actual = _encode(model, root.future[appearance_index, branch, candidate, steps], shape, config["batch_size"])
                            rows.extend(_prediction_rows(model, history[branch], actual, actions, config["horizons"],
                                                         root.metadata["root_id"], appearance, branch, candidate))
            source = "generated_paired_pilot"
        else:
            from .offline_data import H5Windows
            h5 = Path(h5).resolve(strict=True)
            stamp = (h5.stat().st_size, h5.stat().st_mtime_ns)
            with stage("hash recorded H5 (CPU/disk; no model training)"):
                input_hashes[str(h5)] = digest(h5)
            with stage("validate recorded H5 schema and action boundaries"):
                windows = H5Windows(h5, horizon=config["horizon"])
            with windows:
                eligible = np.flatnonzero(windows.ep_len > windows.span)
                if len(eligible) < episodes:
                    raise ValueError(f"Requested {episodes} episodes, only {len(eligible)} have a full window")
                rng = np.random.default_rng(config["seed"])
                chosen = sorted(rng.choice(eligible, size=episodes, replace=False).tolist())
                # Selection is exclusively episode IDs/lengths and a fixed seed, never outcomes.
                selection = [{"episode": e, "start": int(rng.integers(int(windows.ep_len[e]) - windows.span))} for e in chosen]
                write_json(output / "windows.json", {"windows": selection, "seed": config["seed"]})
                for item in progress(selection, desc=f"Recorded prediction on {device}", unit="episode"):
                    window = windows.window(item["episode"], item["start"])
                    history = _encode(model, window["context"], shape, config["batch_size"])
                    actual = _encode(model, window["future"][steps], shape, config["batch_size"])
                    rows.extend(_prediction_rows(model, history, actual, window["actions"], config["horizons"],
                                                 str(item["episode"]), "original", 0, 0))
            if stamp != (h5.stat().st_size, h5.stat().st_mtime_ns):
                raise ValueError("Recorded H5 changed during diagnostics")
            source = "recorded_h5_original_only"
        summary = prediction_summary(rows, config)
        from .diagnostic_plots import prediction_plots
        prediction_plots(output, summary)
    return _finish(output, "frozen_prediction", config, metadata, input_hashes,
                   source=source, records=rows, summary=summary, windows=selection,
                   protocol={"metric": "MSE over latent coordinates, within-model only",
                             "controls": "persistence and zero FUTURE native XY actions; original two prefix chunks preserved",
                             "normalization": "raw actions passed to adapter, normalized once by released stats",
                             "sampling": "all generated pilot roots/branches/candidates or one metadata-selected window per recorded episode",
                             "uncertainty": "equal-weight root/episode bootstrap, paired contrasts; negative native-minus-control is better",
                             "recorded_limit": "Original appearance only; no semantic masks for faithful rerendering. Available release is pretraining data, not held-out world-model generalization.",
                             "readout_used": False})