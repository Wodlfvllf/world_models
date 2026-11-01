"""F1: bounded, actual-image-only measurement probes on frozen world models.

The supplied generation config owns the complete scorer/validation/pilot rosters
(128/64/128 in the committed pilot). Smaller explicit configs/specs are useful
for synthetic contract tests, not interchangeable research runs. No rollout,
editor, predicted latent, pilot selection, best-seed selection, pixel archive,
or on-disk feature cache is used here.

``probe_followup`` returns the usual diagnostics completion reference. Its
report retains the six physical state fields and root-level physical/effect
metrics, including all MLP seeds separately. Selected checkpoints are a flat
tensor dictionary, loadable with ``torch.load(..., weights_only=True)``.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import html
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import diagnostics as d
from .fitting import PhysicalReadout
from .pipeline import _configuration, _frozen, _model_spec, _publish
from .progress import note, progress, stage
from .provenance import digest, write_json
from .simulator import STATE_FIELDS
from .statistics import APPEARANCES

__all__ = ["probe_followup"]

_GROUPS = {"pusher_xy": (0, 1), "block_xy": (2, 3), "orientation": (4, 5)}
_TARGET_SCALE = (512., 512., 512., 512., 1., 1.)
_VISIBILITY = ("full_in", "partial", "outside")
_VIS_METRICS = ("pusher_distance", "block_distance", "angle_mae_deg")
_CONTRAST_METRICS = ("scaled_loss", *_VIS_METRICS, "paired_block_effect_distance",
                     "paired_angle_effect_mae_deg")
_MAX_PREDICTION_BYTES = 10_000_000


def _integer(value, name, minimum=1):
    if type(value) is not int or not minimum <= value < 2 ** 63:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**63).")
    return value


def _number(value, name, *, zero=False):
    try:
        valid = (type(value) in (int, float) and math.isfinite(value)
                 and (value >= 0 if zero else value > 0))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return float(value)


def _specification(spec, config):
    required = {"schema", "name", "seed", "bootstrap_resamples", "encode_batch_size",
                "ridge_grids", "ridge_alphas", "mlp_grid", "mlp_hidden", "mlp_seeds",
                "mlp_weight_decays", "mlp_checkpoints", "mlp_learning_rate", "mlp_batch_size",
                "primary_horizon", "small_edit_xy_tolerance", "small_edit_angle_tolerance_deg",
                "confirmatory"}
    if not isinstance(spec, dict) or required - spec.keys():
        raise ValueError("spec must be a complete follow-up specification dictionary.")
    # Also reject nonfinite values in informational/extension fields. Keep the
    # entire specification, not only the fields this F1 implementation consumes.
    try:
        spec = json.loads(json.dumps(spec, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spec must contain finite JSON-compatible values.") from exc
    if spec["schema"] != "wm_audit.followup.v1" or spec["confirmatory"] is not False:
        raise ValueError("Expected exploratory wm_audit.followup.v1, confirmatory=false.")
    if not isinstance(spec["name"], str) or not spec["name"].strip():
        raise ValueError("spec.name must be nonempty text.")
    _integer(spec["seed"], "spec.seed", 0)
    for key in ("bootstrap_resamples", "encode_batch_size", "mlp_grid", "mlp_hidden",
                "mlp_batch_size", "primary_horizon"):
        _integer(spec[key], key)
    for key in ("ridge_grids", "ridge_alphas", "mlp_seeds", "mlp_weight_decays", "mlp_checkpoints"):
        values = spec[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list.")
        for value in values:
            if key in ("ridge_grids", "mlp_seeds", "mlp_checkpoints"):
                _integer(value, key, 0 if key == "mlp_seeds" else 1)
            else:
                _number(value, key, zero=key == "mlp_weight_decays")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} must contain distinct values.")
    for key in ("mlp_learning_rate", "small_edit_xy_tolerance", "small_edit_angle_tolerance_deg"):
        _number(spec[key], key)
    if spec["small_edit_angle_tolerance_deg"] > 180:
        raise ValueError("Angular tolerance cannot exceed 180 degrees.")
    if spec["primary_horizon"] not in config["horizons"]:
        raise ValueError("primary_horizon must be among the configured evaluation horizons.")
    for split in ("scorer", "validation", "pilot"):
        _integer(config["counts"].get(split), f"counts.{split}")
    return spec


def _array(value, name, *, width=None):
    value = np.asarray(value)
    if (value.ndim != 2 or not len(value) or not value.shape[1]
            or (width is not None and value.shape[1] != width)
            or value.dtype.kind not in "fiu" or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite nonempty [N,{width or 'D'}] numeric array.")
    return value


def _finite_metrics(metrics):
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite probe metric; no samples may be discarded.")
    return metrics


def _tensor_state(state):
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite or non-tensor probe checkpoint.")
    if "std" in state and (state["std"] <= 0).any():
        raise ValueError("Probe training standard deviations must be positive.")
    return state


@torch.no_grad()
def _collect(paths, split, config, hashes, model, shape, grids, batch_size):
    chunks, labels, ids = {grid: [] for grid in grids}, [], []
    for path in progress(paths, desc=f"F1 encode {split} original on {model.device}", unit="root"):
        root = d._load(path, split, config, hashes)
        pixels, actual, _ = d._image_rows(root, 0, config, all_candidates=False)
        actual = _array(actual, "training/validation labels", width=6)
        features = d._features(model, pixels, shape, batch_size, grids)
        for grid, values in features.items():
            _array(values, "pooled features", width=PhysicalReadout(shape, grid).feature_dim)
            if len(values) != len(actual):
                raise ValueError("Feature and six-field label row counts differ.")
            chunks[grid].append(values)
        labels.append(actual)
        ids.extend([root.metadata["root_id"]] * len(actual))
    return ({grid: np.concatenate(parts) for grid, parts in chunks.items()},
            np.concatenate(labels), np.asarray(ids))


def _group_scores(predicted, actual, ids, scales):
    predicted, actual = _array(predicted, "predictions", width=6), _array(actual, "labels", width=6)
    if predicted.shape != actual.shape or np.asarray(ids).shape != (len(actual),):
        raise ValueError("Selection needs matching prediction, label and root-ID rows.")
    residual = np.square((predicted.astype(np.float64) - actual) / scales)
    roots = [ids == root for root in sorted(set(ids))]
    return _finite_metrics({group: float(np.mean([residual[mask][:, columns].sum(1).mean()
                                                   for mask in roots]))
                            for group, columns in _GROUPS.items()})


def _candidate_id(family, **settings):
    # Lexical ties are over this complete, canonical identity, NOT loop order,
    # an approximately equal score, or a favorable seed.
    return family + ":" + json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _consider(best, method, row, state):
    groups = best.setdefault(method, {})
    for group in _GROUPS:
        key = (row["validation_group_losses"][group], row["candidate_id"])
        previous = groups.get(group)
        if previous is None or key < (previous[0]["validation_group_losses"][group], previous[0]["candidate_id"]):
            groups[group] = (row, state)


@torch.no_grad()
def _predict_mlp(state, features, device, batch_size):
    features = _array(features, "MLP features", width=state["mean"].numel())
    weights = {key: state[key].to(device) for key in
               ("hidden_weight", "hidden_bias", "output_weight", "output_bias")}
    parts = []
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start:start + batch_size], dtype=torch.float64, device="cpu")
        x = ((x - state["mean"]) / state["std"]).to(device=device, dtype=torch.float32)
        value = F.linear(F.silu(F.linear(x, weights["hidden_weight"], weights["hidden_bias"])),
                         weights["output_weight"], weights["output_bias"])
        value = value.cpu().double() * state["target_scale"] + state["target_mean"]
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite MLP prediction; no samples may be discarded.")
        parts.append(value.numpy())
    return np.concatenate(parts)


def _train_mlp(features, labels, *, spec, seed, decay, device, checkpoint):
    """Only detached scorer data and a fresh MLP enter the gradient scope."""
    features, labels = _array(features, "MLP training features"), _array(labels, "MLP training labels", width=6)
    if len(features) != len(labels) or len(features) < 2:
        raise ValueError("MLP training needs matching features/labels, N >= 2.")
    logs = []
    # Explicitly works under an outer no_grad (and inference_mode) without ever
    # enabling a gradient path through the frozen encoder or its parameters.
    with torch.inference_mode(False), torch.enable_grad():
        x = torch.as_tensor(features, dtype=torch.float64, device="cpu").detach()
        y = torch.as_tensor(labels, dtype=torch.float64, device="cpu").detach()
        mean, std = x.mean(0), x.std(0, correction=0)
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        target_mean, target_scale = y.mean(0), torch.tensor(_TARGET_SCALE, dtype=torch.float64)
        normalization = _tensor_state({"mean": mean, "std": std, "target_mean": target_mean,
                                       "target_scale": target_scale})
        x, y = ((x - mean) / std).float(), ((y - target_mean) / target_scale).float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Nonfinite standardized MLP training data.")
        # Seed CPU initialization only. torch.manual_seed would also reseed CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            network = nn.Sequential(nn.Linear(x.shape[1], spec["mlp_hidden"], device="cpu", dtype=torch.float32),
                                    nn.SiLU(), nn.Linear(spec["mlp_hidden"], 6, device="cpu", dtype=torch.float32))
        network.to(device).train()
        optimizer = torch.optim.AdamW(network.parameters(), lr=spec["mlp_learning_rate"], weight_decay=decay)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        epochs, batch_size = max(spec["mlp_checkpoints"]), spec["mlp_batch_size"]
        for epoch in progress(range(1, epochs + 1), desc=f"F1 MLP seed {seed}, wd {decay:g} on {device}", unit="epoch"):
            order, total = torch.randperm(len(x), generator=generator), 0.0
            for start in range(0, len(x), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                error = network(x[indices].to(device)) - y[indices].to(device)
                loss = error.square().sum(1).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite MLP training loss.")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in network.parameters()):
                    raise ValueError("Nonfinite or missing MLP training gradients.")
                optimizer.step()
                _tensor_state(network.state_dict())
                total += float(loss.detach()) * len(indices)
            row = {"epoch": epoch, "online_epoch_scaled_loss": total / len(x)}
            _finite_metrics(row)
            logs.append(row)
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                note(f"F1 MLP seed {seed}, wd {decay:g}, {device} | epoch {epoch}/{epochs} | loss {row['online_epoch_scaled_loss']:.6g}")
            if epoch in spec["mlp_checkpoints"]:
                with torch.no_grad():
                    state = {key: value.detach().cpu().clone() for key, value in normalization.items()}
                    state.update({"hidden_weight": network[0].weight.detach().cpu().clone(),
                                  "hidden_bias": network[0].bias.detach().cpu().clone(),
                                  "output_weight": network[2].weight.detach().cpu().clone(),
                                  "output_bias": network[2].bias.detach().cpu().clone()})
                    _tensor_state(state)
                    prediction = _predict_mlp(state, features, device, batch_size)
                    row["checkpoint_training_scaled_loss"] = float(np.square((prediction - labels) / _TARGET_SCALE).sum(1).mean())
                    _finite_metrics(row)
                    checkpoint(epoch, state, row["checkpoint_training_scaled_loss"])
        optimizer.zero_grad(set_to_none=True)
    return {"seed": seed, "weight_decay": decay, "epochs": epochs, "losses": logs}


@torch.no_grad()
def _fit_probes(train_x, train_y, val_x, val_y, val_ids, grids, mlp_grid, config, spec, device):
    best, candidates, logs = {}, [], []
    for grid in grids:
        with stage(f"F1 ridge grid {grid}: CPU float64 eigensolve"):
            family = d.RidgeFamily(train_x[grid], train_y)
        for alpha in progress(sorted(spec["ridge_alphas"]), desc=f"F1 validate ridge grid {grid}", unit="alpha"):
            state = _tensor_state(family.fit(alpha))
            row = {"family": "ridge", "grid": grid, "alpha": float(alpha),
                   "candidate_id": _candidate_id("ridge", grid=grid, alpha=float(alpha)),
                   "validation_group_losses": _group_scores(d.predict_readout(state, val_x[grid]), val_y,
                                                            val_ids, config["scales"])}
            candidates.append(row)
            _consider(best, "ridge_group_selected", row, state)
        del family
    for seed in sorted(spec["mlp_seeds"]):
        for decay in sorted(spec["mlp_weight_decays"]):
            def checkpoint(epoch, state, training_loss):
                settings = {"seed": seed, "grid": mlp_grid, "weight_decay": float(decay), "epoch": epoch}
                row = {"family": "mlp", **settings, "candidate_id": _candidate_id("mlp", **settings),
                       "training_scaled_loss": training_loss,
                       "validation_group_losses": _group_scores(
                           _predict_mlp(state, val_x[mlp_grid], device, spec["mlp_batch_size"]),
                           val_y, val_ids, config["scales"])}
                candidates.append(row)
                _consider(best, f"mlp_seed{seed}", row, state)

            logs.append(_train_mlp(train_x[mlp_grid], train_y, spec=spec, seed=seed, decay=decay,
                                   device=device, checkpoint=checkpoint))
    return best, sorted(candidates, key=lambda row: row["candidate_id"]), logs


def _publish_selection(output, best, candidates, logs, mean, shape, spec, train_ids, val_ids):
    states = {row["candidate_id"]: (row, state) for groups in best.values() for row, state in groups.values()}
    keys = {name: f"candidate_{index:03d}" for index, name in enumerate(sorted(states))}
    tensors = {"schema_version": torch.tensor(1), "latent_shape": torch.tensor(shape),
               "train_mean": torch.tensor(mean, dtype=torch.float64),
               "training_target_scale": torch.tensor(_TARGET_SCALE, dtype=torch.float64)}
    for name, prefix in keys.items():
        row, state = states[name]
        tensors[prefix + ".family"] = torch.tensor(0 if row["family"] == "ridge" else 1)
        for field in ("grid", "alpha", "seed", "weight_decay", "epoch"):
            if field in row:
                dtype = torch.float64 if field in ("alpha", "weight_decay") else torch.int64
                tensors[prefix + "." + field] = torch.tensor(row[field], dtype=dtype)
        tensors.update({prefix + "." + key: value.detach().cpu().contiguous() for key, value in state.items()})
    _tensor_state(tensors)

    def save(path):
        # A file handle avoids randomized temporary filenames inside the torch
        # ZIP, so identical fits also have identical checkpoint bytes.
        with path.open("wb") as stream:
            torch.save(tensors, stream)

    _publish(output / "probe-weights.pt", save)
    selection = {"schema": "wm_audit.followup.probe-selection.v1", "spec": spec,
                 "state_fields": list(STATE_FIELDS), "target_groups": _GROUPS,
                 "fit_split": "scorer", "selection_split": "validation",
                 "fit_appearance": "original", "selection_appearance": "original",
                 "fit_root_ids": sorted(set(train_ids)), "validation_root_ids": sorted(set(val_ids)),
                 "selection_rule": "minimum equal-root mean sum of squared scaled errors within each target group; orientation uses sin/cos, not angular MAE; exact ties use lexical candidate_id",
                 "seed_rule": "select each target group separately within each MLP seed; never select a best seed",
                 "weights_file": "probe-weights.pt", "weights_sha256": digest(output / "probe-weights.pt"),
                 "tensor_keys": "candidate_NNN.<state field>; MLP returns (Linear(SiLU(Linear((x-mean)/std)))) * target_scale + target_mean",
                 "candidates": candidates,
                 "selected": {method: {group: {**{k: v for k, v in row.items() if k != "validation_group_losses"},
                                                "validation_loss": row["validation_group_losses"][group],
                                                "weight_key": keys[row["candidate_id"]]}
                                        for group, (row, _) in groups.items()}
                              for method, groups in best.items()}}
    write_json(output / "training-loss.json", {
        "schema": "wm_audit.followup.probe-loss.v1", "spec": spec, "runs": logs,
        "fit_split": "scorer", "training_target_scale": _TARGET_SCALE,
        "objective": "mean over scorer rows of summed six-field squared errors after training-target centering/scaling",
        "online_epoch_scaled_loss": "row-weighted mean of pre-update minibatch losses during one complete shuffled epoch",
        "checkpoint_training_scaled_loss": "post-epoch whole-scorer loss, not validation loss"})
    write_json(output / "selection.json", selection)
    return selection


@torch.no_grad()
def _predict_selected(best, features, mean, device, batch_size):
    cache, predictions = {}, {}
    n = len(next(iter(features.values())))
    for method, groups in best.items():
        predicted = np.empty((n, 6), dtype=np.float64)
        for group, (row, state) in groups.items():
            key = row["candidate_id"]
            if key not in cache:
                x = features[row["grid"]]
                cache[key] = (d.predict_readout(state, x) if row["family"] == "ridge"
                              else _predict_mlp(state, x, device, batch_size))
            predicted[:, _GROUPS[group]] = cache[key][:, _GROUPS[group]]
        predictions[method] = _array(predicted, "selected physical predictions", width=6)
    predictions["train_mean"] = np.broadcast_to(mean, (n, 6))
    return predictions


def _layout(config):
    k, horizons = config["candidates"], config["horizons"]
    half = 3 + k * len(horizons)
    return {"branch": np.repeat(np.arange(2, dtype=np.int8), half),
            "candidate": np.tile(np.r_[[-1] * 3, np.repeat(np.arange(k), len(horizons))], 2).astype(np.int64),
            "horizon": np.tile(np.r_[[0] * 3, np.tile(horizons, k)], 2).astype(np.int64),
            "context_index": np.tile(np.r_[np.arange(3), [-1] * (half - 3)], 2).astype(np.int64)}


def _surfaces(layout, config):
    h = layout["horizon"]
    return {"all": np.ones(len(h), dtype=bool), "context": h == 0, "future": h > 0,
            **{f"horizon_{step}": h == step for step in config["horizons"]}}


def _paired_effect_metrics(predicted, actual):
    """Paired branch-1 minus branch-0 effects, aligned frame/candidate/step."""
    predicted, actual = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if (predicted.shape != actual.shape or predicted.ndim != 3 or predicted.shape[0] != 2
            or not predicted.shape[1] or predicted.shape[2] != 6
            or not np.isfinite(predicted).all() or not np.isfinite(actual).all()):
        raise ValueError("Paired physical effects need matching finite [2,N,6] arrays.")
    error = (predicted[1, :, 2:4] - predicted[0, :, 2:4]) - (actual[1, :, 2:4] - actual[0, :, 2:4])

    def change(values):
        angle = np.arctan2(values[..., 4], values[..., 5])
        delta = angle[1] - angle[0]
        return np.arctan2(np.sin(delta), np.cos(delta))

    delta = change(predicted) - change(actual)
    degrees = np.abs(np.arctan2(np.sin(delta), np.cos(delta))) * 180 / np.pi
    undefined = (np.linalg.norm(predicted[..., 4:], axis=-1) < 1e-8).any(axis=0)
    degrees[undefined] = 180.0
    return _finite_metrics({"paired_block_effect_distance": float(np.linalg.norm(error, axis=1).mean()),
                            "paired_block_effect_squared_distance": float(np.square(error).sum(1).mean()),
                            "paired_angle_effect_mae_deg": float(degrees.mean()),
                            "paired_undefined_angle_rate": float(undefined.mean())})


def _pusher_visibility(actual):
    xy = _array(actual, "visibility labels", width=6)[:, :2].astype(np.float64)
    full = ((xy >= 15) & (xy <= 512 - 15)).all(axis=1)
    delta = xy - np.clip(xy, 0, 512)
    # Euclidean circle/rectangle intersection, including the canvas corners.
    outside = np.hypot(delta[:, 0], delta[:, 1]) >= 15
    return np.where(full, 0, np.where(outside, 2, 1)).astype(np.uint8)


class _Collisions:
    """Running centered counts/sums/sumsq, six numbers per sum, never pixels.

    One six-number origin per hash avoids cancellation for large coordinates and
    detects exact differing labels without mistaking roundoff for ambiguity.
    """
    def __init__(self):
        self.groups = {}
        self.samples = 0
        self.root_rows = []

    def add(self, pixels, actual):
        actual = _array(actual, "collision labels", width=6).astype(np.float64)
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.shape != (len(actual), 224, 224, 3):
            raise ValueError("Collision hashing needs matching actual uint8 [N,224,224,3] RGB images.")
        self.root_rows.append(len(actual))
        for image, label in zip(pixels, actual):
            key = hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
            if key not in self.groups:
                self.groups[key] = [0, label.copy(), np.zeros(6), np.zeros(6), False]
            entry = self.groups[key]
            delta = label - entry[1]
            entry[0] += 1
            entry[2] += delta
            entry[3] += np.square(delta)
            entry[4] = entry[4] or bool(np.any(delta != 0))
            if not np.isfinite(entry[2]).all() or not np.isfinite(entry[3]).all():
                raise ValueError("Nonfinite running collision moments.")
            self.samples += 1

    def summary(self, scales):
        if not self.samples or len(set(self.root_rows)) != 1:
            raise ValueError("Collision bound requires the complete equal-row-per-root evaluation sample.")
        sse = np.zeros(6)
        for count, _, total, squares, _ in self.groups.values():
            sse += np.maximum(0, squares - np.square(total) / count)
        mse = sse / self.samples
        if not np.isfinite(mse).all():
            raise ValueError("Nonfinite collision ambiguity bound.")
        return {"appearance": "original", "surface": "all", "samples": self.samples,
                "roots": len(self.root_rows), "unique_rgb_hashes": len(self.groups),
                "repeated_rgb_hashes": sum(v[0] > 1 for v in self.groups.values()),
                "ambiguous_rgb_hashes": sum(v[4] for v in self.groups.values()),
                "ambiguous_samples": sum(v[0] for v in self.groups.values() if v[4]),
                "coordinate_mse_lower_bound": dict(zip(STATE_FIELDS, mse.tolist())),
                "mean_six_field_mse_lower_bound": float(mse.mean()),
                "scaled_loss_lower_bound": float((mse / np.square(scales)).sum()),
                "formula": "sum_hash(sum((y-origin)^2) - sum(y-origin)^2 / count) / N, per field",
                "interpretation": "Empirical minimum MSE of any deterministic function of one exact resized RGB image; equal frame counts make this equal-root weighted. SHA256 equality assumes no cryptographic collision. Not a population bound or an angular-MAE/distance bound; no bootstrap over hashes.",
                "storage": "only running per-hash counts, a six-number origin, centered sums/sumsq and differing-label flag; no pixels saved"}


def _root_records(root_id, appearance, predictions, actual, surfaces, visibility, scales):
    rows, half = [], len(actual) // 2
    for method, predicted in predictions.items():
        for surface, mask in surfaces.items():
            if not np.array_equal(mask[:half], mask[half:]):
                raise ValueError("Paired branch surface masks must be identical.")
            metrics = d.physical_metrics(predicted[mask], actual[mask], scales)
            metrics.update(_paired_effect_metrics(predicted.reshape(2, half, 6)[:, mask[:half]],
                                                   actual.reshape(2, half, 6)[:, mask[:half]]))
            strata = {}
            for code, name in enumerate(_VISIBILITY):
                chosen = mask & (visibility == code)
                values = d.physical_metrics(predicted[chosen], actual[chosen], scales) if chosen.any() else None
                strata[name] = {"count": int(chosen.sum()),
                                "metrics": None if values is None else _finite_metrics({k: values[k] for k in _VIS_METRICS})}
            rows.append({"root_id": root_id, "appearance": appearance, "method": method, "surface": surface,
                         "samples": int(mask.sum()), "pairs": int(mask[:half].sum()),
                         "metrics": _finite_metrics(metrics), "pusher_visibility": strata})
    return rows


def _summary(records, spec):
    groups = defaultdict(list)
    for row in records:
        groups[(row["appearance"], row["method"], row["surface"])].append(row)
    result = []
    kwargs = {"resamples": spec["bootstrap_resamples"], "seed": spec["seed"]}
    for (appearance, method, surface), rows in progress(sorted(groups.items()), desc="F1 equal-root bootstrap", unit="slice"):
        rows = sorted(rows, key=lambda row: row["root_id"])
        if len({row["root_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate root in a follow-up summary slice.")
        metrics = {key: d.interval([r["metrics"][key] for r in rows], **kwargs) for key in rows[0]["metrics"]}
        for coordinate in d.COORDINATES:
            mse = metrics[f"{coordinate}_mse"]
            metrics[f"{coordinate}_rmse"] = {"estimate": math.sqrt(mse["estimate"]),
                                              "ci95": np.sqrt(mse["ci95"]).tolist(), "clusters": mse["clusters"]}
        visibility = {}
        for name in _VISIBILITY:
            present = [r["pusher_visibility"][name] for r in rows if r["pusher_visibility"][name]["count"]]
            visibility[name] = {"samples": sum(r["count"] for r in present), "roots_with_samples": len(present),
                                "roots_without_samples": len(rows) - len(present),
                                "metrics": {key: d.interval([r["metrics"][key] for r in present], **kwargs)
                                            for key in _VIS_METRICS} if present else None}
        result.append({"appearance": appearance, "method": method, "surface": surface,
                       "roots": len(rows), "samples": sum(r["samples"] for r in rows),
                       "pairs": sum(r["pairs"] for r in rows), "metrics": metrics, "pusher_visibility": visibility})
    return result


def _contrasts(records, spec):
    table = defaultdict(dict)
    for row in records:
        table[(row["appearance"], row["method"], row["surface"])][row["root_id"]] = row["metrics"]
    result = []
    for (appearance, method, surface), values in sorted(table.items()):
        comparisons = []
        if method != "train_mean":
            comparisons.append(("method_minus_train_mean", (appearance, "train_mean", surface)))
        if method.startswith("mlp_seed"):
            comparisons.append(("method_minus_ridge", (appearance, "ridge_group_selected", surface)))
        if appearance != "original":
            comparisons.append(("appearance_minus_original", ("original", method, surface)))
        for name, reference in comparisons:
            other = table[reference]
            if set(other) != set(values):
                raise ValueError("Contrasts require identical complete root rosters.")
            result.append({"appearance": appearance, "method": method, "surface": surface, "contrast": name,
                           "metrics": {key: d.interval([values[root][key] - other[root][key] for root in sorted(values)],
                                                        resamples=spec["bootstrap_resamples"], seed=spec["seed"])
                                       for key in _CONTRAST_METRICS}})
    return result


def _prediction_arrays(n_roots, methods, layout):
    n = len(layout["branch"])
    byte_count = n_roots * n * (6 * 8 * (len(APPEARANCES) * len(methods) + 1) + 1)
    info = {"saved": False, "limit_bytes_exclusive": _MAX_PREDICTION_BYTES,
            "estimated_numeric_bytes": byte_count,
            "prediction_axes": ["root", "appearance", "method", "row", "physical_field"],
            "actual_axes": ["root", "row", "physical_field"], "dtype": "float64",
            "row_order": "branch, then three context frames once, then candidate x configured horizons",
            "context_candidate_sentinel": -1, "context_horizon_sentinel": 0,
            "visibility_codes": dict(enumerate(_VISIBILITY))}
    if byte_count + n_roots * 1024 + 65536 >= _MAX_PREDICTION_BYTES:
        return None, {**info, "reason": "Conservative uncompressed budget exceeded; no large prediction cache allocated."}
    return {"predicted": np.empty((n_roots, len(APPEARANCES), len(methods), n, 6), dtype=np.float64),
            "actual": np.empty((n_roots, n, 6), dtype=np.float64),
            "pusher_visibility": np.empty((n_roots, n), dtype=np.uint8), **layout}, info


def _save_predictions(output, arrays, info, root_ids, methods):
    if arrays is None:
        return info
    with io.BytesIO() as stream:
        np.savez_compressed(stream, **arrays, root_ids=np.asarray(root_ids), methods=np.asarray(methods),
                            appearances=np.asarray(APPEARANCES), state_fields=np.asarray(STATE_FIELDS))
        size = stream.tell()
        if size >= _MAX_PREDICTION_BYTES:
            return {**info, "reason": "Serialized archive exceeds the strict byte limit; not published."}
        _publish(output / "predictions.npz", lambda path: path.write_bytes(stream.getvalue()))
    return {**info, "saved": True, "file": "predictions.npz", "bytes": size}


def _plots(output, summary, selection, logs, spec, collisions, limitations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [r for r in summary if r["appearance"] == "original" and r["surface"] == f"horizon_{spec['primary_horizon']}"]
    methods = ["ridge_group_selected", *[f"mlp_seed{s}" for s in sorted(spec["mlp_seeds"])], "train_mean"]
    indexed = {row["method"]: row for row in primary}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    try:
        names = [("pusher_distance", "Pusher position error", spec["small_edit_xy_tolerance"]),
                 ("block_distance", "Block position error", spec["small_edit_xy_tolerance"]),
                 ("angle_mae_deg", "Orientation error (degrees)", spec["small_edit_angle_tolerance_deg"]),
                 ("paired_block_effect_distance", "Paired block-effect error", spec["small_edit_xy_tolerance"]),
                 ("paired_angle_effect_mae_deg", "Paired angle-effect error (degrees)", spec["small_edit_angle_tolerance_deg"])]
        for ax, (key, label, tolerance) in zip(axes.flat, names):
            points = [indexed[method]["metrics"][key] for method in methods]
            x = np.arange(len(methods))
            ax.vlines(x, [p["ci95"][0] for p in points], [p["ci95"][1] for p in points], color="#2878b5")
            ax.scatter(x, [p["estimate"] for p in points], color="#2878b5")
            ax.axhline(tolerance, color="#c44e52", linestyle="--", label="diagnostic tolerance only")
            ax.set_xticks(x, [m.replace("ridge_group_selected", "ridge").replace("mlp_seed", "MLP\nseed ").replace("train_mean", "train\nmean") for m in methods])
            ax.set(title=label, ylabel="Degrees" if "deg" in key else "Native units")
            ax.grid(alpha=.2)
        for run in logs:
            axes[1, 2].plot([r["epoch"] for r in run["losses"]],
                            [r["online_epoch_scaled_loss"] for r in run["losses"]],
                            label=f"seed {run['seed']}, wd {run['weight_decay']:g}")
        axes[1, 2].set(title="Scorer-only MLP training", xlabel="Epoch", ylabel="Online scaled loss")
        axes[1, 2].legend(fontsize=8)
        fig.suptitle(f"F1 original appearance / horizon {spec['primary_horizon']} — equal-root 95% intervals; seeds are not independent roots")
        _publish(output / "overview.png", lambda path: fig.savefig(path, format="png", dpi=120))
    finally:
        plt.close(fig)
    rows = "".join("<tr><td>" + "</td><td>".join(html.escape(str(value)) for value in
                   (method, group, item["candidate_id"], f"{item['validation_loss']:.6g}")) + "</td></tr>"
                   for method, groups in selection["selected"].items() for group, item in groups.items())
    caveats = "".join(f"<li>{html.escape(text)}</li>" for text in limitations)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>F1 measurement capacity</title>
<style>body{{font:16px system-ui;background:#111b2b;color:#e8eef8;max-width:1250px;margin:auto;padding:28px}}
section{{background:#1a2940;padding:20px;margin:24px 0;border-radius:14px}}img{{max-width:100%;height:auto}}
a{{color:#8edbff}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #40516a}}table{{font-size:13px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}li{{margin:10px 0}}</style>
<h1>F1: actual-image measurement capacity</h1><p>Frozen world model; separate scorer-only probes. Exploratory, not qualification.</p>
<p><a href="report.json">All roots, appearances, surfaces, horizons and paired contrasts</a> ·
<a href="selection.json">Original-validation selection</a> · <a href="training-loss.json">Every training loss/seed</a></p>
<section><img src="overview.png" alt="Primary-slice errors for ridge, every MLP seed, and training mean; training loss curves"></section>
<section><h2>Selections fixed before pilot access</h2><table><tr><th>Method</th><th>Target group</th><th>Checkpoint</th><th>Validation loss</th></tr>{rows}</table></section>
<section><h2>Single-image collision ambiguity</h2><p>{collisions['ambiguous_rgb_hashes']} RGB hashes have different labels among {collisions['samples']} original images.
Empirical scaled-MSE lower bound: {collisions['scaled_loss_lower_bound']:.6g}. This is not an angular-error bound or a full-history information-absence claim.</p></section>
<section><h2>Limitations</h2><ul>{caveats}</ul></section>
<details><summary>Full supplied specification</summary><pre>{html.escape(json.dumps(spec, indent=2, sort_keys=True))}</pre></details></html>'''
    _publish(output / "index.html", lambda path: path.write_text(content, encoding="utf-8"))


@torch.no_grad()
def probe_followup(config, data, output, model, *, spec):
    """Fit F1 probes, publish selection before pilot access, return completion ref.

    ``config`` and ``spec`` are dictionaries. ``data`` is the authenticated
    development root; ``output`` must not exist, even from an interrupted run.
    Frozen actual-image encoding uses spec.encode_batch_size. MLP fitting alone
    enables gradients, with train-only normalization and isolated fit RNGs.
    """
    config = _configuration(config)
    spec = _specification(spec, config)
    shape, device, metadata = _model_spec(model)
    grids = (1,) if len(shape) == 1 else tuple(sorted(spec["ridge_grids"]))
    mlp_grid = 1 if len(shape) == 1 else spec["mlp_grid"]
    all_grids = tuple(sorted(set((*grids, mlp_grid))))
    for grid in all_grids:
        PhysicalReadout(shape, grid)  # Fail malformed shapes/grids BEFORE making output.
    data, output = Path(data), d._new_output(output)
    # _roster opens archives for schema/disjointness validation, so do NOT ask it
    # for pilot until BOTH selection and tensor weights have been published.
    paths, training_hashes = d._roster(config, data, ("scorer", "validation"))
    with _frozen(model), np.errstate(over="raise", invalid="raise", divide="raise"):
        train_x, train_y, train_ids = _collect(paths["scorer"], "scorer", config, training_hashes, model,
                                               shape, all_grids, spec["encode_batch_size"])
        val_x, val_y, val_ids = _collect(paths["validation"], "validation", config, training_hashes, model,
                                         shape, all_grids, spec["encode_batch_size"])
        best, candidates, logs = _fit_probes(train_x, train_y, val_x, val_y, val_ids, grids, mlp_grid, config, spec, device)
        mean = train_y.mean(0, dtype=np.float64)
        selection = _publish_selection(output, best, candidates, logs, mean, shape, spec, train_ids, val_ids)
        published = {name: digest(output / name) for name in ("selection.json", "probe-weights.pt", "training-loss.json")}
        training_rows, validation_rows = len(train_y), len(val_y)
        del train_x, val_x, train_y, val_y
        # Full cross-split disjointness and original manifest checks remain
        # mandatory. Rechecking fit inputs also rejects mid-fit mutations.
        paths, hashes = d._roster(config, data, ("scorer", "validation", "pilot"))
        if any(hashes.get(path) != value for path, value in training_hashes.items()):
            raise ValueError("Diagnostic inputs changed after fitting/selection.")
        layout = _layout(config)
        surfaces = _surfaces(layout, config)
        methods = [*best, "train_mean"]
        arrays, prediction_info = _prediction_arrays(len(paths["pilot"]), methods, layout)
        evaluation_grids = tuple(sorted({row["grid"] for groups in best.values() for row, _ in groups.values()}))
        records, root_ids, collisions = [], [], _Collisions()
        for index, path in enumerate(progress(paths["pilot"], desc=f"F1 full pilot on {device}", unit="root")):
            root = d._load(path, "pilot", config, hashes)
            root_ids.append(root.metadata["root_id"])
            for appearance_index, appearance in enumerate(APPEARANCES):
                pixels, actual, tags = d._image_rows(root, appearance_index, config, all_candidates=True)
                actual = _array(actual, "pilot labels", width=6)
                expected_tags = np.where(layout["horizon"] == 0, "context", "future")
                if actual.shape != (len(layout["branch"]), 6) or not np.array_equal(tags, expected_tags):
                    raise ValueError("Pilot image rows do not match the complete paired candidate/horizon layout.")
                if appearance_index == 0:
                    collisions.add(pixels, actual)
                features = d._features(model, pixels, shape, spec["encode_batch_size"], evaluation_grids)
                predictions = _predict_selected(best, features, mean, device, spec["mlp_batch_size"])
                if any(value.shape != actual.shape for value in predictions.values()):
                    raise ValueError("Pilot predictions and six-field labels must have matching shapes.")
                visibility = _pusher_visibility(actual)
                records.extend(_root_records(root.metadata["root_id"], appearance, predictions, actual,
                                              surfaces, visibility, config["scales"]))
                if arrays is not None:
                    arrays["predicted"][index, appearance_index] = np.stack([predictions[m] for m in methods])
                    if appearance_index == 0:
                        arrays["actual"][index], arrays["pusher_visibility"][index] = actual, visibility
                del pixels, features, predictions
        ambiguity = collisions.summary(config["scales"])
        prediction_info = _save_predictions(output, arrays, prediction_info, root_ids, methods)
        del arrays, collisions
        with stage("F1 summarize paired root uncertainty (CPU)"):
            summary, contrasts = _summary(records, spec), _contrasts(records, spec)
        limitations = [
            "Prospectively specified exploratory follow-up on previously inspected pilot data, not retrospective preregistration or fresh confirmation.",
            "Each MLP seed is a robustness repeat on the SAME physical roots; every seed is displayed separately, never selected or treated as independent observations.",
            "Per-group validation selection uses squared scaled coordinate errors (sin/cos for orientation), not pilot outcomes or angular MAE. Selected groups may come from different grids/checkpoints.",
            "No world-model training, rollout, predicted/editor latent, history, action, or proprioception enters these actual-single-image probes. Their accuracy cannot qualify predicted/edited states.",
            "A bounded MLP/grid search failing does not imply absent information or absent physics; pooling and probe capacity remain limitations.",
            "The 4-unit / 2.5-degree canonical small-edit tolerances are diagnostic only, never an automatic qualification gate. Supplied tolerances are recorded in the spec.",
            "Pusher visibility is radius-15 circle geometry on [0,512]^2, not a rendered segmentation/occlusion check. External tangency (zero intersection area) counts as outside; internal tangency is fully in.",
            "Visibility-conditioned errors average roots with that geometry and explicitly count absent roots; main metrics and paired contrasts retain EVERY root/frame without filtering.",
            "RGB-collision MSE bounds are empirical single-image observation ambiguity, not population bounds, angular-MAE bounds, or limits of the full history-and-action interface.",
            "Undefined predicted sin/cos directions receive 180 degrees, including a paired effect if either branch is undefined; no exclusion or post-hoc direction normalization.",
            "Pointwise equal-root bootstrap intervals use the follow-up seed/resample count; no multiplicity-corrected confirmatory inference. Original/primary horizon is primary; all other slices are secondary.",
            "RNG-isolated CPU initialization and local shuffling prevent seed leakage; bitwise portability across devices/PyTorch backends is not promised. Only compact selected probe weights/predictions are stored.",
        ]
        _plots(output, summary, selection, logs, spec, ambiguity, limitations)
        if any(digest(output / name) != value for name, value in published.items()):
            raise ValueError("Published selection/weights/training logs changed during pilot evaluation.")
    return d._finish(output, "followup_probes", config, metadata, hashes,
                     followup_schema="wm_audit.followup.probes.v1", spec=spec, state_fields=list(STATE_FIELDS),
                     methods=methods, selection=selection, records=records, summary=summary, contrasts=contrasts,
                     training_roots=len(set(train_ids)), validation_roots=len(set(val_ids)), evaluation_roots=len(root_ids),
                     training_rows=training_rows, validation_rows=validation_rows,
                     predictions=prediction_info, collision_ambiguity=ambiguity,
                     pre_pilot_artifact_sha256=published, limitations=limitations,
                     protocol={"fit_split": "scorer", "selection_split": "validation", "evaluation_split": "pilot",
                               "fit_appearance": "original", "selection_appearance": "original",
                               "fit_sampling": "both branches; three context frames and config horizons; candidate=root_index % K",
                               "evaluation_sampling": "all pilot roots, all three appearances, both branches, context once plus ALL candidates at config horizons",
                               "canonical_roots": {"scorer": 128, "validation": 64, "pilot": 128},
                               "roster_rule": "complete config counts via authenticated _roster; no truncation/subsampling",
                               "effective_ridge_grids": list(grids), "effective_mlp_grid": mlp_grid,
                               "training_target_scale": list(_TARGET_SCALE), "validation_loss_scales": config["scales"],
                               "primary_slice": {"appearance": "original", "surface": f"horizon_{spec['primary_horizon']}"},
                               "world_model_trained": False, "model_rollout_used": False,
                               "prediction_latents_in_readout_training": False, "pilot_opened_after_selection_publication": True,
                               "fit_normalization": "scorer feature mean/population std (std<=1e-6 replaced by 1); scorer target mean; MLP target scales512/1",
                               "uncertainty": "equal-weight root bootstrap of within-root means; paired method/appearance contrasts; seeds not independent samples",
                               "paired_effect": "matched branch-1 minus branch-0; block XY vector-error norm; wrap(predicted angular change - actual angular change)",
                               "visibility_filtering": False, "automatic_qualification": False})