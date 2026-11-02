"""Exploratory F2/F3/F4: oracle feedback, action effects, and native goal ranking.

Batch all candidates of a branch together. Reuse their encoded frames for every
control. No fitting, no root filtering, no reset of the original model weights.
"""
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch

from . import diagnostics as d
from .followup_probes import _specification
from .models import DinoWMAdapter, LeWMAdapter
from .pipeline import _checked_tensor, _configuration, _cost, _encode, _frozen, _model_spec
from .progress import progress, stage
from .provenance import digest, write_json
from .statistics import APPEARANCES


def _state_digest(model):
    """Include nonpersistent buffers (notably DINO's causal attention masks)."""
    result = hashlib.sha256()
    for kind, tensors in (("parameter", model.named_parameters(remove_duplicate=False)),
                          ("buffer", model.named_buffers(remove_duplicate=False))):
        for name, value in sorted(tensors):
            result.update(json.dumps([kind, name, list(value.shape), str(value.dtype)]).encode() + b"\0")
            # Byte views also support scalar and bfloat16 tensors without casting.
            result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def feedback_rollouts(model, actual, raw_actions, *, recorded_actions=False):
    """actual=[N,H+3,*latent_shape]; never use observations after current history.

    DINO recurrent conditions keep ALL patch-varying predicted auxiliary channels.
    Reset conditions use encode_proprio(0), not literal zero latent channels.
    """
    if not isinstance(model, (LeWMAdapter, DinoWMAdapter)):
        raise TypeError("Feedback controls require a qualified LeWM or DINO adapter.")
    shape = tuple(model.latent_shape)
    if not isinstance(actual, torch.Tensor) or actual.ndim != len(shape) + 2:
        raise ValueError("Expected actual encoded [N,H+3,*latent_shape] trajectories.")
    if actual.shape[1] < 4:
        raise ValueError("Need at least one future observation.")
    _checked_tensor(actual, (len(actual), actual.shape[1], *shape), "actual trajectory")
    history, actions = model._rollout_inputs(actual[:, :3], raw_actions, recorded_actions=recorded_actions)
    horizon = actions.shape[1] - 2
    if actual.shape[1] != horizon + 3:
        raise ValueError("Actions and actual observations have different horizons.")
    actual = actual.to(device=model.device, dtype=torch.float32)
    outputs = {}
    for visual in ("free", "actual"):
        for auxiliary in (("recurrent", "reset") if isinstance(model, DinoWMAdapter) else ("none",)):
            context, predictions = history, []
            if isinstance(model, DinoWMAdapter):
                wm = model.model
                prop = model._tile(wm.encode_proprio(history.new_zeros(len(history), 3, 1)), wm.num_proprio_repeat)
                z = torch.cat((history, prop, model._tile(wm.encode_act(actions[:, :3]), wm.num_action_repeat)), -1)
            for step in range(horizon):
                if isinstance(model, LeWMAdapter):
                    if visual == "actual":
                        context = actual[:, step:step + 3]
                    act = model.model.action_encoder(actions[:, :step + 3])[:, -3:]
                    predicted = model.model.predict(context, act)[:, -1:]
                    predictions.append(predicted)
                    context = torch.cat((context[:, 1:], predicted), 1)
                else:
                    if visual == "actual":
                        z = torch.cat((actual[:, step:step + 3], z[..., shape[-1]:]), -1)
                    if auxiliary == "reset":
                        z = torch.cat((z[..., :shape[-1]], prop, z[..., -wm.action_dim:]), -1)
                    predicted = wm.predict(z)[:, -1:]
                    predictions.append(predicted[..., :shape[-1]])
                    if step + 1 < horizon:
                        act = model._tile(wm.encode_act(actions[:, step + 3:step + 4]), wm.num_action_repeat)
                        next_z = torch.cat((predicted[..., :-wm.action_dim], act), -1)
                        z = torch.cat((z[:, 1:], next_z), 1)
            outputs[f"{visual}_{auxiliary}"] = _checked_tensor(
                torch.cat(predictions, 1), (len(history), horizon, *shape), "feedback prediction")
    native_key = "free_recurrent" if isinstance(model, DinoWMAdapter) else "free_none"
    # Mandatory runtime parity on every bank, not merely a toy recurrence test.
    native = model.rollout(history, raw_actions, recorded_actions=recorded_actions)
    torch.testing.assert_close(outputs[native_key], native, rtol=2e-5, atol=2e-6)
    for prediction in outputs.values():
        torch.testing.assert_close(prediction[:, 0], native[:, 0], rtol=2e-5, atol=2e-6)
    return outputs


def feedback_errors(outputs, actual, horizon):
    """Per-candidate errors; local persistence is aligned to the oracle history."""
    if (not isinstance(actual, torch.Tensor) or actual.ndim < 3 or not len(actual)
            or type(horizon) is not int or not 1 <= horizon <= actual.shape[1] - 3
            or not isinstance(outputs, dict) or not outputs):
        raise ValueError("Need nonempty feedback trajectories and a valid one-indexed horizon.")
    _checked_tensor(actual, actual.shape, "actual trajectory")
    shape = (len(actual), actual.shape[1] - 3, *actual.shape[2:])
    for prediction in outputs.values():
        _checked_tensor(prediction, shape, "feedback prediction")
    target = actual[:, horizon + 2]
    values = {key: (prediction[:, horizon - 1] - target).square().flatten(1).mean(1)
              for key, prediction in outputs.items()}
    values["initial_persistence"] = (actual[:, 2] - target).square().flatten(1).mean(1)
    values["local_persistence"] = (actual[:, horizon + 1] - target).square().flatten(1).mean(1)
    for key in outputs:
        baseline = "initial_persistence" if key.startswith("free_") else "local_persistence"
        values[key + "_minus_baseline"] = values[key] - values[baseline]
    for auxiliary in ("none", "recurrent", "reset"):
        if "actual_" + auxiliary in values:
            values["actual_minus_free_" + auxiliary] = values["actual_" + auxiliary] - values["free_" + auxiliary]
    if "free_reset" in values:
        for visual in ("free", "actual"):
            values[visual + "_reset_minus_recurrent"] = values[visual + "_reset"] - values[visual + "_recurrent"]
        values["visual_aux_interaction"] = (values["actual_reset"] - values["free_reset"]
                                            - values["actual_recurrent"] + values["free_recurrent"])
    return {key: value.detach().cpu().double().numpy() for key, value in values.items()}


def action_effects(predicted, actual):
    """Candidate 0 is real hold; compare effects to a zero-effect baseline.

    Dot products and energies are reported separately, allowing a ratio of
    aggregate means rather than unstable per-sample normalization.
    """
    if (not isinstance(predicted, torch.Tensor) or not isinstance(actual, torch.Tensor)
            or predicted.shape != actual.shape or actual.ndim < 2 or len(actual) < 2
            or not predicted.is_floating_point() or not actual.is_floating_point()
            or not actual[0].numel()):
        raise ValueError("Need matching candidate banks with at least two candidates.")
    p, a = predicted.double().flatten(1), actual.double().flatten(1)
    if not torch.isfinite(p).all() or not torch.isfinite(a).all():
        raise ValueError("Nonfinite candidate bank; no exclusions allowed.")
    dp, da = p[1:] - p[:1], a[1:] - a[:1]
    energy, error = da.square().mean(1), (dp - da).square().mean(1)
    matrix = torch.stack([(row[None] - a).square().mean(1) for row in p])
    diagonal = matrix.diag()
    wrong = (matrix.sum(1) - diagonal) / (len(a) - 1)
    values = {"effect_error": error, "true_effect_energy": energy,
              "predicted_effect_energy": dp.square().mean(1), "effect_dot": (dp * da).mean(1),
              "effect_error_minus_zero": error - energy,
              "matched_minus_all_wrong": diagonal - wrong}
    if not all(torch.isfinite(value).all() for value in (*values.values(), matrix)):
        raise ValueError("Nonfinite action-effect metric; no exclusions allowed.")
    return {key: value.detach().cpu().numpy() for key, value in values.items()}, matrix.detach().cpu().numpy()


def ranking_metrics(endpoint_scores, forecast_scores, costs):
    scores = [np.asarray(x, dtype=np.float64) for x in (endpoint_scores, forecast_scores, costs)]
    endpoint, forecast, costs = scores
    if costs.ndim != 1 or len(costs) < 2 or any(x.shape != costs.shape or not np.isfinite(x).all() for x in scores):
        raise ValueError("Need matching finite one-dimensional candidate scores and true costs.")
    if (costs < 0).any() or (costs > 1).any():
        raise ValueError("Coverage costs must be in [0,1].")
    best = float(costs.min())
    chosen = {"endpoint": int(endpoint.argmin()), "forecast": int(forecast.argmin())}
    metrics = {name + "_regret": float(costs[index] - best) for name, index in chosen.items()}
    metrics.update(uniform_regret=float(costs.mean() - best), hold_regret=float(costs[0] - best),
                   best_cost=best, cost_range=float(costs.max() - best), best_coverage=1 - best,
                   no_positive_coverage=float(np.all(costs == 1)))
    for name in chosen:
        metrics[name + "_minus_uniform"] = metrics[name + "_regret"] - metrics["uniform_regret"]
        metrics[name + "_minus_hold"] = metrics[name + "_regret"] - metrics["hold_regret"]
    metrics["forecast_minus_endpoint_regret"] = metrics["forecast_regret"] - metrics["endpoint_regret"]
    return metrics, chosen


def _gain_interval(dot, energy, *, seed, resamples):
    """Signed least-squares effect gain: ratio of equal-root aggregate means.

    Resample numerator and denominator with the SAME root indices. Zero-energy
    roots are never removed, and no epsilon floor or per-root ratio is used.
    If any bootstrap denominator is zero, the unconditional CI is undefined;
    report that fact rather than silently condition on favorable resamples.
    """
    dot, energy = np.asarray(dot, dtype=np.float64), np.asarray(energy, dtype=np.float64)
    if (dot.ndim != 1 or not len(dot) or dot.shape != energy.shape
            or not np.isfinite(dot).all() or not np.isfinite(energy).all() or (energy < 0).any()):
        raise ValueError("Gain requires matching finite root dot products and nonnegative energies.")
    d._positive_int(resamples, "resamples")
    denominator = float(energy.mean())
    estimate = float(dot.mean() / denominator) if denominator > 0 else None
    rng, samples, undefined = np.random.default_rng(seed), [], 0
    for start in range(0, resamples, 256):
        indices = rng.integers(len(dot), size=(min(256, resamples - start), len(dot)))
        numerator, denominator = dot[indices].mean(1), energy[indices].mean(1)
        valid = denominator > 0
        undefined += int((~valid).sum())
        samples.extend((numerator[valid] / denominator[valid]).tolist())
    if (estimate is not None and not np.isfinite(estimate)) or not np.isfinite(samples).all():
        raise ValueError("Nonfinite aggregate effect gain; no exclusions allowed.")
    return {"estimate": estimate, "ci95": None if undefined else np.quantile(samples, [.025, .975]).tolist(),
            "clusters": len(dot), "undefined_resamples": undefined}


def summarize(rows, spec):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["appearance"], row["horizon"]][row["cluster"]].append(row["metrics"])
    result = []
    for (appearance, horizon), clusters in sorted(groups.items()):
        keys = next(iter(clusters.values()))[0].keys()
        if any(set(row) != set(keys) for entries in clusters.values() for row in entries):
            raise ValueError("All roots must retain the same metrics and contrasts.")
        means = {key: [np.mean([r[key] for r in entries]) for _, entries in sorted(clusters.items())]
                 for key in keys}
        kwargs = {"seed": spec["seed"], "resamples": spec["bootstrap_resamples"]}
        metrics = {key: d.interval(values, **kwargs) for key, values in means.items()}
        if "effect_dot" in means and "true_effect_energy" in means:
            metrics["aggregate_effect_gain"] = _gain_interval(means["effect_dot"], means["true_effect_energy"], **kwargs)
        result.append({"appearance": appearance, "horizon": horizon, "metrics": metrics})
    return result


def _write_view(output, summary, source):
    content = html.escape(json.dumps(summary, indent=2))
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Feedback/action/goal audit</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style>'
        '<h1>Exploratory feedback/action/goal audit</h1><p>' + html.escape(source) + '</p>'
        '<p>Oracle observations are not deployable forecasting. Equal-root pointwise intervals; '
        'no confirmatory claim. Negative contrasts favor the first named method.</p>'
        '<a href="report.json">Full report and candidate-level records</a><pre>' + content + '</pre>')


@torch.no_grad()
def dynamics_followup(config, data, output, model, *, spec):
    config = _configuration(config)
    spec = _specification(spec, config)
    shape, device, metadata = _model_spec(model)
    if not isinstance(model, (LeWMAdapter, DinoWMAdapter)) or config["candidates"] < 2:
        raise ValueError("Need a qualified model and at least two candidates.")
    output = d._new_output(output)
    paths, hashes = d._roster(config, Path(data), ("pilot",))
    rows = []
    original_state = _state_digest(model)
    with _frozen(model):
        for path in progress(paths["pilot"], desc=f"F2/F3/F4 batched candidates on {device}", unit="root"):
            root = d._load(path, "pilot", config, hashes)
            if np.any(root.actions[0, 2:] != 0):
                raise ValueError("Candidate zero must be the predeclared hold control.")
            k, h = config["candidates"], config["horizon"]
            for ai, appearance in enumerate(APPEARANCES):
                context = _encode(model, root.context[ai], shape, spec["encode_batch_size"])
                # _cost adds the candidate batch itself; the goal is unbatched.
                goal = _encode(model, root.goal_pixels[ai], shape, spec["encode_batch_size"])
                for branch in range(2):
                    future = _encode(model, root.future[ai, branch], shape, spec["encode_batch_size"])
                    actual = torch.cat((context[branch:branch + 1].expand(k, -1, *shape), future), 1)
                    outputs = feedback_rollouts(model, actual, root.actions)
                    native = outputs["free_recurrent" if isinstance(model, DinoWMAdapter) else "free_none"]
                    for horizon in range(1, h + 1):
                        errors = feedback_errors(outputs, actual, horizon)
                        effect, matrix = action_effects(native[:, horizon - 1], future[:, horizon - 1])
                        endpoint_scores = _cost(model, future[:, horizon - 1], goal)
                        forecast_scores = _cost(model, native[:, horizon - 1], goal)
                        costs = 1 - root.coverage_future[branch, :, horizon - 1]
                        ranking, chosen = ranking_metrics(endpoint_scores, forecast_scores, costs)
                        physical = root.states_future[branch, :, horizon - 1]
                        rgb = root.future[ai, branch, :, horizon - 1].astype(np.float32) / 255
                        angle = np.arctan2(physical[:, 4], physical[:, 5])
                        delta = angle[1:] - angle[0]
                        metrics = {key: float(value.mean()) for key, value in {**errors, **effect}.items()}
                        metrics.update(ranking)
                        metrics.update(block_effect_units=float(np.linalg.norm(physical[1:, 2:4] - physical[:1, 2:4], axis=1).mean()),
                                       pusher_effect_units=float(np.linalg.norm(physical[1:, :2] - physical[:1, :2], axis=1).mean()),
                                       angle_effect_degrees=float(np.abs(np.arctan2(np.sin(delta), np.cos(delta))).mean() * 180 / np.pi),
                                       rgb_effect_mse=float(np.square(rgb[1:] - rgb[:1]).mean()))
                        if not all(np.isfinite(value) for value in metrics.values()):
                            raise ValueError("Nonfinite metric; cannot drop this bank.")
                        rows.append({"cluster": root.metadata["root_id"], "appearance": appearance,
                                     "branch": branch, "horizon": horizon, "metrics": metrics,
                                     "candidates": {key: value.tolist() for key, value in {**errors, **effect}.items()},
                                     "prediction_to_actual_mse": matrix.tolist(), "chosen": chosen,
                                     "endpoint_scores": endpoint_scores.tolist(), "forecast_scores": forecast_scores.tolist(),
                                     "true_costs": costs.tolist()})
    if _state_digest(model) != original_state:
        raise ValueError("World-model parameters or buffers changed during follow-up.")
    with stage("F2/F3/F4 root-level uncertainty"):
        summary = summarize(rows, spec)
    _write_view(output, summary, "Previously inspected generated pilot; all candidates and appearances.")
    return d._finish(output, "followup_dynamics", config, metadata, hashes, spec=spec,
                     records=rows, summary=summary, parameter_buffer_sha256=original_state,
                     protocol={"confirmatory": False, "sampling": "every pilot root/branch/candidate/appearance; every future step",
                               "world_model_trained": False, "readout_used": False,
                               "visual_control": "free versus oracle actual three-frame history",
                               "auxiliary_control": "DINO predicted patch-wise channels versus encode_proprio(0) at all history positions",
                               "parity": "free recurrence agrees with public adapter on every bank; all variants agree at step1",
                               "action_effect": "candidate k minus candidate0; zero-effect baseline; all off-diagonal matches",
                               "aggregate_effect_gain": "equal-root mean effect_dot / mean true_effect_energy; paired root bootstrap; no per-sample ratios or floors; null CI if any resample has zero energy",
                               "goal": "native checkpoint cost; true coverage at the same horizon; smallest-index ties",
                               "uncertainty": "equal root weight after candidate/branch means; pointwise paired bootstrap",
                               "oracle_warning": "actual-history prediction has extra observations; not deployable improvement"})


def _recorded_reference(prior, config):
    """Identify the prior H5 by its receipt, not its now possibly stale location."""
    if (not isinstance(prior, dict) or prior.get("kind") != "frozen_prediction"
            or prior.get("source") != "recorded_h5_original_only" or prior.get("config") != config):
        raise ValueError("Prior report must be a matching recorded diagnostic.")
    hashes = prior.get("input_sha256")
    info = prior.get("recorded_h5_info")
    old_path = info.get("path") if isinstance(info, dict) else None
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Prior report must identify the recorded H5 SHA256.")
    if old_path is None and len(hashes) == 1:
        # Legacy reports have only one input: the original recorded H5.
        old_path = next(iter(hashes))
    if not isinstance(old_path, str) or not old_path or old_path not in hashes:
        raise ValueError("Prior report must unambiguously identify its recorded H5 input.")
    expected = hashes[old_path]
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("Prior recorded H5 SHA256 is malformed.")
    windows_spec = prior.get("windows", [])
    if (not isinstance(windows_spec, list) or len(windows_spec) != 128
            or any(not isinstance(item, dict) or set(item) != {"episode", "start"}
                   or any(type(item[key]) is not int or item[key] < 0 for key in ("episode", "start"))
                   for item in windows_spec)
            or len({item["episode"] for item in windows_spec}) != 128):
        raise ValueError("Require all 128 distinct prior episode/start selections.")
    return old_path, expected, windows_spec


def _file_stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@torch.no_grad()
def recorded_feedback(config, h5, prior_report, output, model, *, spec):
    """Revalidate H5 and reuse the prior exact episode/start roster without selection."""
    from .offline_data import H5Windows
    config = _configuration(config)
    spec = _specification(spec, config)
    shape, device, metadata = _model_spec(model)
    if not isinstance(model, (LeWMAdapter, DinoWMAdapter)):
        raise TypeError("Recorded feedback requires a qualified LeWM or DINO adapter.")
    h5, prior_report = Path(h5).resolve(strict=True), Path(prior_report).resolve(strict=True)
    prior_bytes = prior_report.read_bytes()
    prior_sha256 = hashlib.sha256(prior_bytes).hexdigest()
    old_path, expected, windows_spec = _recorded_reference(json.loads(prior_bytes), config)
    stamp = _file_stamp(h5)
    with stage("Authenticate recorded H5 against prior experiment"):
        if digest(h5) != expected or _file_stamp(h5) != stamp:
            raise ValueError("Recorded input differs from prior authenticated H5.")
    output = d._new_output(output)
    write_json(output / "windows.json", {"windows": windows_spec, "prior_report_sha256": prior_sha256})
    original_state, rows = _state_digest(model), []
    with _frozen(model), H5Windows(h5, horizon=config["horizon"], action_schema="recorded") as windows:
        for item in progress(windows_spec, desc=f"Recorded F2 on {device}", unit="episode"):
            window = windows.window(item["episode"], item["start"])
            pixels = np.concatenate((window["context"], window["future"]), 0)
            actual = _encode(model, pixels, shape, spec["encode_batch_size"])[None]
            outputs = feedback_rollouts(model, actual, window["actions"][None], recorded_actions=True)
            for horizon in range(1, config["horizon"] + 1):
                errors = feedback_errors(outputs, actual, horizon)
                rows.append({"cluster": str(item["episode"]), "start": item["start"], "appearance": "original",
                             "horizon": horizon, "metrics": {key: float(value.mean()) for key, value in errors.items()}})
    if _state_digest(model) != original_state:
        raise ValueError("World-model parameters or buffers changed during follow-up.")
    with stage("Reauthenticate recorded inputs before publication"):
        if (_file_stamp(h5) != stamp or digest(h5) != expected or _file_stamp(h5) != stamp
                or digest(prior_report) != prior_sha256):
            raise ValueError("Recorded input or prior report changed during follow-up.")
    with stage("Recorded F2 episode-level uncertainty"):
        summary = summarize(rows, spec)
    _write_view(output, summary, "Recorded pretraining distribution; exact prior 128 episode/start windows.")
    return d._finish(output, "followup_recorded_feedback", config, metadata,
                     {str(h5): expected, str(prior_report): prior_sha256},
                     spec=spec, records=rows, summary=summary, windows=windows_spec,
                     recorded_input={"prior_path": old_path, "path": str(h5), "sha256": expected},
                     parameter_buffer_sha256=original_state,
                     protocol={"world_model_trained": False, "readout_used": False,
                               "recorded_controls": "raw unchanged; normalize once; no clipping or terminal transitions",
                               "uncertainty": "equal episode weight; paired pointwise bootstrap",
                               "limitation": "pretraining data, not held-out generalization; oracle history is extra information"})