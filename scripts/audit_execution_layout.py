"""Bounded CUDA-only audit of the observed generated-pilot execution layouts.

No fitting, regeneration, outcome selection, checkpoint writes, or implicit resume.
See NUMERICAL_AUDIT_PROTOCOL_20261102.md. CPU use is confined to synthetic tests
of the functions below; the command line refuses CPU/MPS and CUDA fallbacks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from time import monotonic
import csv
import warnings

import numpy as np
import torch

from wm_audit import diagnostics as d, followup_dynamics as f, models, pipeline as p
from wm_audit.data import _json_object, _reject_json_constant
from wm_audit.followup_probes import _specification
from wm_audit.progress import progress, stage
from wm_audit.provenance import code_hashes, digest, write_json


ROOTS, CANDIDATES, HORIZON = 8, 8, 8
HORIZONS = (1, 4, 8)
# The ORIGINAL manifest, authenticated in both retrieved experiment reports.
MANIFEST_SHA256 = "fc759e41bd66c48954f3d41568a9e48893ece47ae0a9719229109d08a8086485"
SOURCE_PINS = {
    "models.py": "6bc28315d63ca1000bec79f4c2ee3d1cac143315a7f8e557a905706c13337b24",
    "pipeline.py": "5572870c6bd8ecde70ff61951cd4b28c55a7cf458b58c915276dacaaa546dbbd",
    "diagnostics.py": "ba9f703163f5d71b81d3b9b8f6978b680cbadb19ab6b8fafc1fd3202157b7c4b",
    "followup_dynamics.py": "016d2bbe293c0bfc1d6a3c303c76f83a3adef88ebe34061e279070e7ab5dd14f",
}
OLD_SOURCE_PINS = {
    "models.py": "de993da1a611580b02aac0f72a17937c50718fd12499a6ccb741e9c6be9cbc1f",
    "pipeline.py": "8de45cd7be3f75d786afe31b56b528fe7cc3177a44dc66c30608c24b88d6d9ad",
    "diagnostics.py": "c0d5eb84e8ce7f241887c97c1cc88c022a98291aa35de394546b5bd473474d60",
}
COMPARISONS = (
    ("observed_paths", "old_single", "new_feedback"),
    ("encoding_at_single", "old_single", "new_single"),
    ("encoding_at_bank", "old_bank", "new_bank"),
    ("rollout_at_old_encoding", "old_single", "old_bank"),
    ("rollout_at_new_encoding", "new_single", "new_bank"),
    ("feedback_vs_public", "new_bank", "new_feedback_scalar"),
    ("metric_reduction_only", "new_feedback_scalar", "new_feedback"),
)
VARIANTS = ("old_single", "old_bank", "new_single", "new_bank", "new_feedback")
TENSOR_FIELDS = ("context", "future_common", "target_h8", "rollout_all", "rollout_h8")
METRIC_FIELDS = ("native_f32", "persistence_f32", "native_f64", "persistence_f64")
LIMITATIONS = [
    "A first-eight-root numerical diagnostic, not replication of all 128 roots or new scientific evidence.",
    "Original appearance only; omitted other appearances and F3/F4 postprocessing change allocator/call history.",
    "Old zero-action calls and new feedback controls/public parity are retained; tensor capture adds synchronization.",
    "Current generated-path code is used: archived old differences add recorded-action validation, not generated arithmetic.",
    "Crossed public-adapter cells are controlled bridges, not claimed historical runs; bank layout includes its observed strides.",
    "Only common actual horizons 1/4/8 can be compared; old encoding never produced other actual-future latents.",
    "Backend flags describe permissions, not proof of kernel selection; SDPA choices, workspace, driver, GPU and caches may matter.",
    "Same-process repeats do not reset CUDA/cuDNN algorithm caches or reproduce historical concurrent workloads.",
    "The historical backend state and full tensors were not archived; this cannot establish the cause of the earlier maxima.",
    "Only selected NPZ payloads are authenticated here; the manifest covers, but this audit does not load, the other 120 roots.",
]


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw, object_pairs_hook=_json_object,
                      parse_constant=_reject_json_constant), hashlib.sha256(raw).hexdigest()


def checked_settings(config, spec):
    config = p._configuration(config)
    spec = _specification(spec, config)
    if (config["candidates"] != CANDIDATES or config["horizon"] != HORIZON
            or tuple(config["horizons"]) != HORIZONS or config["batch_size"] != 4
            or spec["encode_batch_size"] != 32 or spec["primary_horizon"] != HORIZON
            or config["counts"]["pilot"] < ROOTS):
        raise ValueError("Require the observed K=8, H=8, horizons=[1,4,8], encoder batches 4/32 and >=8 pilot roots.")
    return config, spec


def source_receipt():
    hashes = code_hashes()
    if any(hashes.get(name) != expected for name, expected in SOURCE_PINS.items()):
        raise ValueError("Executed core source differs from the inspected generated paths; do not silently substitute layouts.")
    return {str(Path(p.__file__).parent / name): value for name, value in hashes.items()} | {
        str(Path(__file__).resolve()): digest(__file__)}


def select_roster(data, config):
    """FIRST eight serialized manifest entries, not filesystem order or outcomes."""
    data = Path(data).resolve(strict=True)
    manifest_path = data / "manifest-development.json"
    manifest, receipt = read_json(manifest_path)
    if receipt != MANIFEST_SHA256 or manifest.get("config") != config:
        raise ValueError("Require the unchanged original generation manifest and matching config.")
    files = manifest.get("files", {}).get("pilot")
    if not isinstance(files, dict) or len(files) != config["counts"]["pilot"] or len(files) < ROOTS:
        raise ValueError("Incomplete pilot manifest roster.")
    # Validate every manifest name, but never open unselected payloads.
    for relative, expected in files.items():
        parts = PurePosixPath(relative).parts
        if (len(parts) != 2 or parts[0] != "pilot" or PurePosixPath(relative).suffix != ".npz"
                or PurePosixPath(relative).as_posix() != relative or "\\" in relative
                or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Malformed pilot manifest entry.")
    selected, hashes = [], {str(manifest_path): receipt}
    for ordinal, (relative, expected) in enumerate(list(files.items())[:ROOTS]):
        path = data / relative
        if path.is_symlink() or path.parent.is_symlink() or path.resolve(strict=True).parent != data / "pilot":
            raise ValueError("Selected input must be an ordinary pilot file inside --data.")
        if digest(path) != expected:
            raise ValueError(f"Selected input differs from manifest: {relative}")
        hashes[str(path)] = expected
        selected.append({"manifest_ordinal": ordinal, "relative_path": relative,
                         "path": str(path), "sha256": expected})
    identities = p._scan({"pilot": [Path(item["path"]) for item in selected]}, config)["pilot"]
    for item, identity in zip(selected, identities, strict=True):
        if Path(item["path"]).stem != identity["root_id"]:
            raise ValueError("Archive filename and root identity differ.")
        item.update(identity)
    return selected, hashes


def array_sha256(value):
    """Dtype/shape plus exact bytes; scalar, noncontiguous and bfloat16 safe."""
    header = json.dumps([str(value.dtype), list(value.shape)]).encode() + b"\0"
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    else:
        raw = np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(header + raw).hexdigest()


def input_receipt(root):
    values = {"context_original": root.context[0], "future_original": root.future[0],
              "goal_original": root.goal_pixels[0], "raw_actions": root.actions}
    return {"arrays": {key: {"sha256": array_sha256(value), "shape": list(value.shape),
                              "dtype": str(value.dtype)} for key, value in values.items()},
            "root_metadata": json.loads(json.dumps(root.metadata, allow_nan=False))}


def model_hashes(model):
    """Separate parameter/buffer receipts in addition to the compatible joint hash."""
    result = {}
    for kind, tensors in (("parameters", model.named_parameters(remove_duplicate=False)),
                          ("buffers", model.named_buffers(remove_duplicate=False))):
        hasher, count = hashlib.sha256(), 0
        for name, value in sorted(tensors):
            hasher.update(name.encode() + b"\0" + bytes.fromhex(array_sha256(value)))
            count += 1
        result[kind + "_sha256"], result[kind + "_count"] = hasher.hexdigest(), count
    result["parameter_buffer_sha256"] = f._state_digest(model)
    return result


def describe(value):
    is_tensor = isinstance(value, torch.Tensor)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "stride_elements": list(value.stride() if is_tensor else
                                    (s // value.itemsize for s in value.strides)),
            "contiguous": value.is_contiguous() if is_tensor else bool(value.flags.c_contiguous),
            "device": str(value.device) if is_tensor else "numpy/cpu"}


@contextmanager
def capture_calls(model, label, trace):
    """Delegate unchanged calls; no tensor-input clones/casts/repacking in wrappers."""
    emitted, originals = [], {}
    try:
        for name in ("encode", "rollout"):
            original = getattr(model, name)
            originals[name] = (name in model.__dict__, original)

            def wrapped(*args, _name=name, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                trace.append({"phase": label, "method": _name,
                              "inputs": [describe(value) for value in args], "output": describe(result)})
                if _name == "rollout":
                    emitted.append(result.detach())
                return result

            setattr(model, name, wrapped)
        yield emitted
    finally:
        for name, (was_local, original) in originals.items():
            if was_local:
                setattr(model, name, original)
            else:
                delattr(model, name)


def scalar_metrics(prediction, context, targets):
    """Old .square().mean() reduction, on the actual device, for identical tensors."""
    native, persistence = [], []
    for candidate in range(CANDIDATES):
        target = targets[candidate, -1]
        native.append((prediction[candidate, -1] - target).square().mean())
        persistence.append((context[-1] - target).square().mean())
    return torch.stack(native), torch.stack(persistence)


def snapshot(root_id, context, targets, prediction, native, persistence):
    shape = tuple(context.shape[2:])
    for value, expected in ((context, (2, 3, *shape)), (targets, (2, CANDIDATES, 3, *shape)),
                            (prediction, (2, CANDIDATES, HORIZON, *shape))):
        p._checked_tensor(value, expected, "captured tensor")
        if value.dtype != torch.float32:
            raise ValueError("Audit requires actual float32 model outputs, without autocast.")
    context, targets, prediction = [v.detach().cpu().clone() for v in (context, targets, prediction)]
    native, persistence = (np.asarray(v, dtype=np.float64) for v in (native, persistence))
    if any(v.shape != (2, CANDIDATES) or not np.isfinite(v).all() for v in (native, persistence)):
        raise ValueError("Nonfinite or unmatched per-candidate metrics; no exclusions allowed.")
    records = {}
    for branch in range(2):
        for candidate in range(CANDIDATES):
            history, future, pred = context[branch], targets[branch, candidate], prediction[branch, candidate]
            records[(root_id, "original", branch, candidate, HORIZON)] = {
                "context": history, "future_common": future, "target_h8": future[-1],
                "rollout_all": pred, "rollout_h8": pred[-1],
                "native_f32": float(native[branch, candidate]),
                "persistence_f32": float(persistence[branch, candidate]),
                "native_f64": float((pred[-1].double() - future[-1].double()).square().mean()),
                "persistence_f64": float((history[-1].double() - future[-1].double()).square().mean()),
            }
    hashes = {name: array_sha256(value) for name, value in
              (("context", context), ("future_common", targets), ("rollout_all", prediction))}
    return records, hashes


@torch.no_grad()
def layout_suite(model, root, config, spec):
    """Two observed paths plus crossed PUBLIC-adapter cells, with no re-encoding."""
    shape, _, _ = p._model_spec(model)
    steps = np.asarray(HORIZONS) - 1
    trace, variants, fingerprints = [], {}, {}

    def save(name, context, targets, pred, metrics):
        records, hashes = snapshot(root.metadata["root_id"], context, targets, pred, *metrics)
        variants[name], fingerprints[name] = records, hashes

    # Exact old generated native path, including intervening zero-action calls.
    old_targets, old_predictions, old_native, old_persistence = [], [], [], []
    with capture_calls(model, "old_observed", trace) as emitted:
        old_context = p._encode(model, root.context[0], shape, config["batch_size"])
        for branch in range(2):
            targets, predictions, native, persistence = [], [], [], []
            for candidate, actions in enumerate(root.actions):
                actual = p._encode(model, root.future[0, branch, candidate, steps], shape, config["batch_size"])
                rows = d._prediction_rows(model, old_context[branch], actual, actions, config["horizons"],
                                          root.metadata["root_id"], "original", branch, candidate)
                if len(emitted) != 2:
                    raise ValueError("Old path must call native then zero-future-actions rollout.")
                predictions.append(emitted[0][0])
                emitted.clear()
                targets.append(actual)
                row = next(row for row in rows if row["horizon"] == HORIZON)
                native.append(row["native"])
                persistence.append(row["persistence"])
            old_targets.append(torch.stack(targets))
            old_predictions.append(torch.stack(predictions))
            old_native.append(native)
            old_persistence.append(persistence)
    old_targets = torch.stack(old_targets)
    save("old_single", old_context, old_targets, torch.stack(old_predictions), (old_native, old_persistence))

    # Exact new encode + feedback function. Capture its EXISTING public parity
    # call rather than substituting a second adapter execution for that output.
    new_futures, new_predictions, public_predictions = [], [], []
    new_native, new_persistence, scalar_native, scalar_persistence = [], [], [], []
    native_key = "free_recurrent" if isinstance(model, models.DinoWMAdapter) else "free_none"
    with capture_calls(model, "new_observed", trace) as emitted:
        new_context = p._encode(model, root.context[0], shape, spec["encode_batch_size"])
        p._encode(model, root.goal_pixels[0], shape, spec["encode_batch_size"])  # Observed intervening call.
        for branch in range(2):
            future = p._encode(model, root.future[0, branch], shape, spec["encode_batch_size"])
            actual = torch.cat((new_context[branch:branch + 1].expand(CANDIDATES, -1, *shape), future), 1)
            outputs = f.feedback_rollouts(model, actual, root.actions)
            if len(emitted) != 1:
                raise ValueError("New feedback path must include exactly one public-adapter parity call.")
            public_predictions.append(emitted.pop())
            pred = outputs[native_key]
            errors = f.feedback_errors(outputs, actual, HORIZON)
            new_native.append(errors[native_key])
            new_persistence.append(errors["initial_persistence"])
            scalar = scalar_metrics(pred, new_context[branch], future[:, steps])
            scalar_native.append(scalar[0].cpu().numpy())
            scalar_persistence.append(scalar[1].cpu().numpy())
            new_futures.append(future)
            new_predictions.append(pred)
            del outputs
    new_targets = torch.stack([value[:, steps] for value in new_futures])
    new_predictions = torch.stack(new_predictions)
    save("new_feedback", new_context, new_targets, new_predictions, (new_native, new_persistence))
    save("new_feedback_scalar", new_context, new_targets, new_predictions, (scalar_native, scalar_persistence))

    def save_public(name, context, targets, predictions):
        # All public-adapter factorial cells use the SAME scalar metric kernel.
        metrics = [scalar_metrics(predictions[b], context[b], targets[b]) for b in range(2)]
        save(name, context, targets, torch.stack(predictions),
             ([value[0].cpu().numpy() for value in metrics], [value[1].cpu().numpy() for value in metrics]))

    save_public("new_bank", new_context, new_targets, public_predictions)
    old_bank, new_single = [], []
    with capture_calls(model, "crossed_public_bridges", trace) as emitted:
        for branch in range(2):
            # Match the OBSERVED cat(..., future)[:, :3] stride, not expand's
            # zero batch stride. Padding is not read by the native public API.
            actual = torch.cat((old_context[branch:branch + 1].expand(CANDIDATES, -1, *shape),
                                new_futures[branch]), 1)
            old_bank.append(model.rollout(actual[:, :3], root.actions, recorded_actions=False))
            new_single.append(torch.cat([
                p._rollout(model, new_context[branch][None], actions[None], shape)
                for actions in root.actions], 0))
            emitted.clear()
    save_public("old_bank", old_context, old_targets, old_bank)
    save_public("new_single", new_context, new_targets, new_single)
    return variants, {"tensor_sha256": fingerprints, "calls": trace}


def tensor_delta(left, right):
    if left.shape != right.shape or left.dtype != right.dtype or left.numel() == 0:
        raise ValueError("Tensor comparisons require identical nonempty shapes and dtypes, never broadcasting.")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Nonfinite tensor comparison; no exclusions allowed.")
    delta = right.double() - left.double()
    return float(delta.abs().max()), float(delta.abs().mean())


def compare(left, right, *, control, comparison, lhs, rhs, ordinal):
    if not left or left.keys() != right.keys():
        raise ValueError("Record identity sets differ; match root/appearance/branch/candidate/horizon, never zip.")
    rows = []
    for key in sorted(left):
        root_id, appearance, branch, candidate, horizon = key
        a, b = left[key], right[key]
        row = dict(control=control, comparison=comparison, lhs=lhs, rhs=rhs,
                   manifest_ordinal=ordinal, root_id=root_id, appearance=appearance,
                   branch=branch, candidate=candidate, horizon=horizon)
        for name in TENSOR_FIELDS:
            row[name + "_max_abs"], row[name + "_mean_abs"] = tensor_delta(a[name], b[name])
        for name in METRIC_FIELDS:
            x, y = a[name], b[name]
            if not np.isfinite([x, y, y - x]).all():
                raise ValueError("Nonfinite metric comparison; no exclusions allowed.")
            row["lhs_" + name], row["rhs_" + name], row[name + "_delta"] = x, y, y - x
        rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["control"], row["comparison"], row["lhs"], row["rhs"])].append(row)
    summary = []
    for (control, comparison, lhs, rhs), group in groups.items():
        identities = {(r["root_id"], r["appearance"], r["branch"], r["candidate"], r["horizon"]) for r in group}
        roots = {r["root_id"] for r in group}
        expected = {(root, "original", branch, candidate, HORIZON)
                    for root in roots for branch in range(2) for candidate in range(CANDIDATES)}
        if (len(identities) != len(group) or identities != expected
                or len(group) != ROOTS * 2 * CANDIDATES or len(roots) != ROOTS):
            raise ValueError("Each comparison must retain exactly eight complete root banks, without duplicates.")
        tensors = {name: {"max_abs": max(r[name + "_max_abs"] for r in group),
                          "mean_abs": float(np.mean([r[name + "_mean_abs"] for r in group]))}
                   for name in TENSOR_FIELDS}
        metrics = {}
        for name in METRIC_FIELDS:
            delta = np.array([r[name + "_delta"] for r in group])
            metrics[name] = {"lhs_mean": float(np.mean([r["lhs_" + name] for r in group])),
                             "rhs_mean": float(np.mean([r["rhs_" + name] for r in group])),
                             "mean_delta": float(delta.mean()), "max_abs_delta": float(np.abs(delta).max()),
                             "mean_abs_delta": float(np.abs(delta).mean())}
        summary.append(dict(control=control, comparison=comparison, lhs=lhs, rhs=rhs,
                            roots=len(roots), records=len(group), tensors=tensors, metrics=metrics))
    return summary


def backend_property(path):
    owner = torch.backends
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


BACKEND_PATHS = (
    "fp32_precision", "cuda.matmul.fp32_precision", "cudnn.fp32_precision", "cudnn.conv.fp32_precision",
    "cuda.matmul.allow_tf32", "cudnn.allow_tf32", "cudnn.enabled", "cudnn.benchmark",
    "cudnn.benchmark_limit", "cudnn.deterministic", "cuda.matmul.allow_fp16_reduced_precision_reduction",
    "cuda.matmul.allow_bf16_reduced_precision_reduction", "cuda.matmul.allow_fp16_accumulation",
)


def backend_state():
    def probe(getter):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # Legacy getter deprecation, not numerical warnings.
                return getter()
        except (AttributeError, RuntimeError) as exc:
            return {"unavailable": f"{type(exc).__name__}: {exc}"}

    result = {key: probe(lambda key=key: getattr(*backend_property(key))) for key in BACKEND_PATHS}
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        result["cuda." + name] = probe(lambda name=name: getattr(torch.backends.cuda, name)())
    # This legacy getter can itself raise after a NEW per-backend precision
    # setter, even though no legacy setter was used. Record, do not mix APIs.
    result.update(float32_matmul_precision=probe(torch.get_float32_matmul_precision),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                  autocast_cuda=torch.is_autocast_enabled("cuda"), autocast_cpu=torch.is_autocast_enabled("cpu"),
                  default_dtype=str(torch.get_default_dtype()), num_threads=torch.get_num_threads(),
                  num_interop_threads=torch.get_num_interop_threads())
    return result


def control_plan(state, enabled=True):
    requested = (("matmul_ieee", "cuda.matmul.fp32_precision", "ieee"),
                 ("matmul_tf32", "cuda.matmul.fp32_precision", "tf32"),
                 ("cudnn_conv_ieee", "cudnn.conv.fp32_precision", "ieee"),
                 ("cudnn_deterministic", "cudnn.deterministic", True),
                 ("cudnn_benchmark_off", "cudnn.benchmark", False))
    result = []
    for name, path, value in requested:
        before = state.get(path)
        reason = ("disabled by CLI" if not enabled else "API unavailable" if before is None or isinstance(before, dict)
                  else "already set; no distinct intervention" if before == value else None)
        result.append(dict(name=name, path=path, value=value, before=before,
                           status="skipped" if reason else "planned", reason=reason))
    return result


@contextmanager
def backend_control(control=None):
    """One modern property ONLY; baseline makes NO setters. Restore even on error."""
    before = backend_state()
    owner = attribute = previous = None
    try:
        if control is not None:
            owner, attribute = backend_property(control["path"])
            previous = getattr(owner, attribute)
            setattr(owner, attribute, control["value"])
            if getattr(owner, attribute) != control["value"]:
                raise RuntimeError("Requested backend control was not applied.")
        yield backend_state()
    finally:
        if owner is not None:
            setattr(owner, attribute, previous)
        if backend_state() != before:
            raise RuntimeError("Backend state was not restored exactly; completion is forbidden.")


def runtime_receipt(device):
    device = torch.device(device)
    gpu = None
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        props = torch.cuda.get_device_properties(index)
        gpu = {"logical_device_index": index, "name": props.name,
               "uuid_text": str(getattr(props, "uuid", "unavailable")),
               "capability": [props.major, props.minor], "total_memory": props.total_memory}
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device), "gpu": gpu, "torch_build": torch.__config__.show(),
            "environment": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG", "CUDNN_CONV_WSCAP_DBG", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


@torch.no_grad()
def execute_audit(model, config, spec, selected, output, input_hashes, sources, *, backend_controls=True):
    """Internal CPU contract tests use synthetic adapters, never loaded releases."""
    config, spec = checked_settings(config, spec)
    if not isinstance(model, (models.LeWMAdapter, models.DinoWMAdapter)):
        raise TypeError("Require a qualified LeWM or DINO adapter.")
    if len(selected) != ROOTS or [s["manifest_ordinal"] for s in selected] != list(range(ROOTS)):
        raise ValueError("Require the first eight manifest roots in order.")
    shape, device, metadata = p._model_spec(model)
    original_backend = backend_state()
    if original_backend["autocast_cpu"] or original_backend["autocast_cuda"]:
        raise ValueError("Disable caller autocast explicitly before this float32 audit; audit never changes it.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    controls = control_plan(original_backend, backend_controls)
    active = [control for control in controls if control["status"] == "planned"]
    state_receipt = model_hashes(model)
    initial_state = state_receipt["parameter_buffer_sha256"]
    plan = {"schema": "wm_audit.execution_layout.plan.v1", "roots": selected,
            "config": config, "spec": spec, "controls": controls, "baseline_backend": original_backend,
            "input_sha256": input_hashes, "code_sha256": sources,
            "parameter_buffer_sha256": initial_state, "selection": "FIRST 8 manifest entries; no outcome selection"}
    write_json(output / "selection.json", plan)  # Before any encoding.
    start, rows, executions, inputs = monotonic(), [], [], []
    with p._frozen(model):
        for item in progress(selected, desc=f"Execution-layout audit on {device}", unit="root"):
            root = d._load(Path(item["path"]), "pilot", config, input_hashes)
            if root.metadata["root_id"] != item["root_id"]:
                raise ValueError("Selected root identity changed.")
            before_input = input_receipt(root)
            inputs.append(before_input)
            baseline = None
            phases = [("baseline", None), ("repeat", None)] + [(c["name"], c) for c in active]
            for phase, control in progress(phases, desc=f"{item['root_id']} on {device}", unit="suite"):
                with stage(f"{phase}: old/new + crossed layouts on {device}"), backend_control(control) as actual_settings:
                    variants, receipt = layout_suite(model, root, config, spec)
                state = f._state_digest(model)
                if state != initial_state:
                    raise ValueError("World-model parameters or buffers changed; completion is forbidden.")
                if input_receipt(root) != before_input:
                    raise ValueError("Input arrays or metadata changed; completion is forbidden.")
                executions.append(dict(root_id=item["root_id"], phase=phase, backend=actual_settings,
                                       parameter_buffer_sha256=state, **receipt))
                kwargs = dict(control=phase, ordinal=item["manifest_ordinal"])
                if phase != "repeat":
                    for comparison, lhs, rhs in COMPARISONS:
                        rows.extend(compare(variants[lhs], variants[rhs], comparison=comparison, lhs=lhs, rhs=rhs, **kwargs))
                if phase == "baseline":
                    baseline = variants
                else:
                    for name in VARIANTS:
                        rows.extend(compare(baseline[name], variants[name],
                                            comparison="identical_layout_repeat" if phase == "repeat" else "backend_only",
                                            lhs="baseline/" + name, rhs=phase + "/" + name, **kwargs))
                del variants
    with stage("Reauthenticate inputs, source and model before publication"):
        for path, expected in {**input_hashes, **sources}.items():
            if digest(path) != expected:
                raise ValueError(f"Input/source changed during audit: {path}")
        final_state = f._state_digest(model)
        if final_state != initial_state or backend_state() != original_backend:
            raise ValueError("Model or default backend state changed; completion is forbidden.")
    summary = summarize(rows)
    with (output / "records.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"schema": "wm_audit.execution_layout.v1", "kind": "execution_layout", "confirmatory": False,
              "runtime": runtime_receipt(device), "elapsed_seconds": monotonic() - start,
              "config": config, "spec": spec, "model": metadata, "latent_shape": list(shape),
              "model_state": state_receipt,
              "parameter_buffer_sha256_before": initial_state, "parameter_buffer_sha256_after": final_state,
              "hash_scope": "all named parameters AND buffers, including aliases and nonpersistent masks; exact bytes",
              "model_unchanged": True, "backend_restored": True,
              "baseline_backend": original_backend,
              "controls": [{**control, "status": "executed" if control["status"] == "planned" else "skipped",
                            "suites_completed": ROOTS if control["status"] == "planned" else 0} for control in controls],
              "input_sha256": input_hashes, "code_sha256": sources, "inputs": inputs,
              "historical_source": {"old_archived_sha256": OLD_SOURCE_PINS, "new_executed_sha256": SOURCE_PINS,
                                    "new_commit": "0ac3c719aa82eb852513704587b391a1b91a5617"},
              "executions": executions, "summary": summary, "csv_records": len(rows),
              "metric_convention": "rhs minus lhs; historical float32 reductions plus common CPU float64 MSE on captured tensors",
              "aggregation": "128 matched records per comparison; equal 8 candidates x 2 branches per root, hence equal-root means",
              "limitations": LIMITATIONS,
              "artifacts": {name: digest(output / name) for name in ("selection.json", "records.csv")}}
    write_json(output / "report.json", report)  # Completion marker LAST, exclusive.
    return {"report": str(output / "report.json"), "records": str(output / "records.csv"),
            "roots": ROOTS, "comparisons": len(summary)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "spec", "data", "cache", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--model", choices=("lewm", "dinowm"), required=True)
    result.add_argument("--device", default="cuda", help="CUDA device (cuda or cuda:N); CPU/MPS are forbidden")
    result.add_argument("--trust-checkpoint", action="store_true")
    result.add_argument("--backend-controls", action=argparse.BooleanOptionalAction, default=True)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        cli.error("Output exists; preserve it and choose a fresh directory.")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError):
        cli.error("--device must be cuda or cuda:N.")
    if device.type != "cuda" or not torch.cuda.is_available():
        cli.error("Real execution is CUDA-hosted only; no local CPU/MPS/fallback experiment.")
    if args.model == "dinowm" and args.trust_checkpoint is not True:
        cli.error("DINO requires --trust-checkpoint for the known hash-pinned pickle.")
    config, config_hash = read_json(args.config)
    spec, spec_hash = read_json(args.spec)
    config, spec = checked_settings(config, spec)
    sources = source_receipt()
    with stage("Authenticate FIRST eight original pilot roots; no inference"):
        selected, hashes = select_roster(args.data, config)
    hashes.update({str(args.config.resolve()): config_hash, str(args.spec.resolve()): spec_hash})
    # Do not let the adapter auto-download DINOv2 as a side effect of this audit.
    if args.model == "dinowm" and not all((args.cache / path).exists() for path in
            ("sources/dinov2", "models/dinov2/dinov2_vits14_pretrain.pth")):
        cli.error("Populate the pinned DINOv2 cache on the GPU host before auditing; no implicit downloads.")
    with stage(f"Load frozen {args.model} on {device}"):
        model = models.load_model(args.model, args.cache, device, trust_checkpoint=args.trust_checkpoint)
    result = execute_audit(model, config, spec, selected, args.output, hashes, sources,
                           backend_controls=args.backend_controls)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()