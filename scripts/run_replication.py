"""Fixed fresh-root F2/F3/F4 replication; research execution is Linux/CUDA only.

No refitting, recorded-data run, adaptive sampling, implicit resume, or overwrite.
Without --execute, authenticate inputs and print the plan without generating data
or loading models. CPU contract tests use explicit synthetic fixtures, not this
research entry point. See REPLICATION_PROTOCOL_20261102.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from time import monotonic

import numpy as np
import torch

from wm_audit import data as paired, diagnostics as d, followup_dynamics as dynamics
from wm_audit import models, pipeline
from wm_audit.followup_probes import _specification
from wm_audit.progress import note, progress, stage
from wm_audit.provenance import code_hashes, digest, read_config, runtime, write_json
from wm_audit.simulator import PushTSimulator
from wm_audit.statistics import APPEARANCES


PROJECT = Path(__file__).resolve().parents[1]
ROOTS, HORIZON, CANDIDATES = 128, 8, 8
PILOT_SPLIT_SEED = 2026110201
BOOTSTRAP_SEED, RESAMPLES = 20261102, 10000
MODELS = ("lewm", "dinowm")
REUSED_SPLITS = ("scorer", "validation")
MANIFEST_SHA256 = "fc759e41bd66c48954f3d41568a9e48893ece47ae0a9719229109d08a8086485"
ORIGINAL_COUNTS = dict(editor=256, scorer=128, validation=64, pilot=128, confirmation=0)
# Also exclude the entire originally planned 3,384-root roster, including the
# never-executed confirmation split. Earlier smoke/partial runs are subsets.
RESERVED_COUNTS = dict(editor=2000, scorer=600, validation=256, pilot=128, confirmation=400)
SOURCE_PINS = {
    "__init__.py": "f35a426e74fd7ae1aa2ae9f9e93141504482451cdedec07723dfbe2654a10489",
    "artifacts.py": "9a94ba4f2c5d982b5c382059d51405854bdc99dd2fa3ddbc576d2a7f7920ec5b",
    "cli.py": "d0d0bee3c204ece2a4291229a839cd742c0778d8610e32ddeb49f6fa98a8591a",
    "data.py": "0c5c48525cde96c3c7c3edf3fe3d361667470872b673ec66b770378f0ffaa5b4",
    "diagnostic_plots.py": "88f4435dcfacf907efbd7fdfbe3604f2b7139f045ef4fbee4780a3a4f804a048",
    "diagnostics.py": "ba9f703163f5d71b81d3b9b8f6978b680cbadb19ab6b8fafc1fd3202157b7c4b",
    "fitting.py": "b30c1c80101fdfefc6c81cdb07d012230782e707f6683459fd52e3437b132a52",
    "followup_dynamics.py": "016d2bbe293c0bfc1d6a3c303c76f83a3adef88ebe34061e279070e7ab5dd14f",
    "followup_probes.py": "942580cff5f334c4c206f29de3136669498e7d1ae173ddf3cfe93bae6aa971b9",
    "models.py": "6bc28315d63ca1000bec79f4c2ee3d1cac143315a7f8e557a905706c13337b24",
    "offline_data.py": "14379b0f7a96871b7dc84c82e6b40106ae5197e7e16afb6d35fa878b7db55d1e",
    "pipeline.py": "5572870c6bd8ecde70ff61951cd4b28c55a7cf458b58c915276dacaaa546dbbd",
    "progress.py": "47bfe7db2d82e0cbae2258cfaafcda3326321dbaeae564f8244e49cd429da8ed",
    "provenance.py": "8919de38b180238e516e5c8d52715fba3e1acc20905dffb6a60d3f1da6aa49d0",
    "simulator.py": "ea0d5845d889adc6397abb63de737e5186067464fa93a83e871ef04dc73bf455",
    "statistics.py": "8d71e49bdc83c7e09075c48de105eb397a84e7b5100e685ad2748d59aab3291c",
}
CONFIG_PINS = {
    "configs/pilot.json": "27bd903071b7e60c8f23ebecf6ddff9dee85eae323bcbde70311c9244defdc88",
    "configs/followup_20261101.json": "1b0c28763a461752c04b30c67b0247de29dec20a7e9d626350532d6c48278fe2",
}
PRIMARY = (
    ("lewm", "effect_error_minus_zero"),
    ("lewm", "matched_minus_all_wrong"),
    ("dinowm", "effect_error_minus_zero"),
    ("dinowm", "matched_minus_all_wrong"),
    ("dinowm", "forecast_minus_endpoint_regret"),
)
LIMITATIONS = [
    "Prospectively fixed after discovery; a local dated commitment, not external preregistration.",
    "Dynamics only: F1 is not executed, retrained, reselected, or replicated; original F1 remains exploratory.",
    "Same generator, candidate bank, appearances, native goal metric and two frozen checkpoints; not a new task distribution.",
    "Sparse-contact, zero-effect, out-of-frame and zero-coverage banks remain; zero regret need not mean control success.",
    "Latent fidelity/specificity does not establish block-physical correctness or permit cross-model raw-MSE comparisons.",
    "Actual-history prediction is privileged oracle information; candidate ranking is not closed-loop control.",
    "Percentile bootstrap coverage and Bonferroni FWER are nominal/asymptotic, not finite-sample guarantees.",
    "Only five original-appearance/h8 contrasts belong to the primary family; every other estimate is exploratory pointwise 95%.",
    "Checkpoint/source hashes and runtime parity do not promise bitwise portability across devices, backends or batching contexts.",
    "Technical failure aborts without dropping/replacing roots. Any retry must use these same roots and disclose prior attempts.",
]


def read_json(path):
    raw = Path(path).read_bytes()
    return (json.loads(raw, object_pairs_hook=paired._json_object,
                       parse_constant=paired._reject_json_constant), hashlib.sha256(raw).hexdigest())


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _ordinary(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError(f"Require an ordinary file without symlink ancestors: {path}")
    return path


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if not _sha(expected) or digest(_ordinary(path)) != expected:
            raise ValueError(f"Authenticated input/source changed: {path}")


def copy_exclusive(source, target, expected):
    """Copy actual bytes, never relabel an old archive as newly generated."""
    source, target = _ordinary(source), Path(target)
    verify_hashes({str(source): expected})
    target.parent.mkdir(parents=True, exist_ok=True)
    # xb also refuses a dangling symlink or a concurrent creator.
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    verify_hashes({str(source): expected, str(target): expected})


def source_receipt(config_path):
    if code_hashes() != SOURCE_PINS:
        raise ValueError("Core source differs from the qualified discovery implementation; do not silently substitute it.")
    package = Path(paired.__file__).resolve().parent
    files = {f"src/wm_audit/{name}": package / name for name in SOURCE_PINS}
    files.update({name: PROJECT / name for name in CONFIG_PINS})
    verify_hashes({str(PROJECT / name): value for name, value in CONFIG_PINS.items()})
    files.update({"scripts/run_replication.py": Path(__file__).resolve(),
                  "configs/replication_20261102.json": Path(config_path).resolve(strict=True),
                  "REPLICATION_PROTOCOL_20261102.md": PROJECT / "REPLICATION_PROTOCOL_20261102.md",
                  "pyproject.toml": PROJECT / "pyproject.toml"})
    return {name: {"path": str(_ordinary(path)), "sha256": digest(path)} for name, path in files.items()}


def checked_settings(config_path):
    # Reject duplicate keys too; read_config supplies the shared strict schema.
    read_json(config_path)
    config = read_config(config_path)
    original = read_config(PROJECT / "configs/pilot.json")
    expected = dict(original, name="fresh-root-dynamics-replication-20261102",
                    counts=dict(editor=0, scorer=128, validation=64, pilot=ROOTS, confirmation=0),
                    seed=BOOTSTRAP_SEED, bootstrap_resamples=RESAMPLES)
    if config != expected:
        raise ValueError("Require the exact fixed replication config; no count/seed/horizon/bank/fit tuning.")
    spec, _ = read_json(PROJECT / "configs/followup_20261101.json")
    spec.update(name="fresh-root-dynamics-replication-20261102", seed=BOOTSTRAP_SEED,
                bootstrap_resamples=RESAMPLES, recorded_windows="not run in this replication",
                replication_scope="dynamics only; primary inference is in replication-results.json")
    return config, _specification(spec, config)


def seed_identity(split, split_seed, index):
    seed = paired.derive_root_seed(split_seed, split, index)
    return {"root_id": paired._root_id(split, index, seed), "root_index": index,
            "root_seed": seed, "split_seed": split_seed,
            "environment_seed": paired._derive_seed(seed, "environment")}


def planned_roots():
    return [seed_identity("pilot", PILOT_SPLIT_SEED, i) for i in range(ROOTS)]


def reserved_roots():
    result = [seed_identity(split, paired.DEFAULT_SPLIT_SEEDS[split], i)
              for split in paired.SPLITS for i in range(RESERVED_COUNTS[split])]
    # The CLI's deterministic software-preflight root is a different namespace.
    return result + [seed_identity("pilot", 987654321, 0)]


def require_disjoint(fresh, excluded):
    if PILOT_SPLIT_SEED in paired.DEFAULT_SPLIT_SEEDS.values():
        raise ValueError("Fresh pilot split seed overlaps an original split seed.")
    for key in ("root_id", "root_seed", "environment_seed"):
        values = [row[key] for row in fresh]
        if len(values) != len(set(values)) or set(values) & {row[key] for row in excluded}:
            raise ValueError(f"Fresh root separation failed for {key}; never search for a replacement seed.")


def manifest_paths(data, manifest, config):
    """All five splits and all archives, including unused editor/confirmation."""
    data = Path(data).resolve(strict=True)
    files = manifest.get("files")
    if manifest.get("config") != config or not isinstance(files, dict) or set(files) != set(paired.SPLITS):
        raise ValueError("Manifest config or complete five-split roster mismatch.")
    paths, hashes = {}, {}
    for split in paired.SPLITS:
        entries = files[split]
        if not isinstance(entries, dict) or len(entries) != config["counts"][split]:
            raise ValueError(f"Incomplete {split} manifest roster.")
        paths[split] = []
        for relative, expected in sorted(entries.items()):
            parts = PurePosixPath(relative).parts
            if (len(parts) != 2 or parts[0] != split or PurePosixPath(relative).suffix != ".npz"
                    or PurePosixPath(relative).as_posix() != relative or "\\" in relative or not _sha(expected)):
                raise ValueError("Malformed or escaping manifest archive path/hash.")
            path = _ordinary(data / relative)
            paths[split].append(path)
            hashes[str(path)] = expected
    actual = {str(path) for path in data.rglob("*.npz")}
    if actual != set(hashes):
        raise ValueError("Unlisted, missing or extra archives in the complete dataset roster.")
    return paths, hashes


def scan_archives(paths, hashes, config, *, split_seeds=None, fixed_distribution=True):
    """Stream full schema validation and identities; never retain all root pixels."""
    identities = {split: [] for split in paths}
    generation = asdict(paired.GenerationConfig(horizon=config["horizon"], candidates=config["candidates"]))

    def roots():
        for split, files in paths.items():
            for path in progress(files, desc=f"Replication validate {split} (CPU)", unit="root"):
                verify_hashes({str(path): hashes[str(path)]})
                root = pipeline._root(path, split, config)
                meta = root.metadata
                identity = pipeline._identity(root)
                if path.stem != identity["root_id"]:
                    raise ValueError("Archive filename differs from root identity.")
                if fixed_distribution and (meta["config"] != generation
                        or meta["actions_source"] != "fixed_bank_v1"
                        or meta["sampling"]["anchor_source"] != "sampled"
                        or np.any(root.actions[:, :2]) or np.any(root.actions[0])):
                    raise ValueError("Generation/candidate distribution changed; no supplied or improved banks allowed.")
                if split_seeds is not None and identity != seed_identity(split, split_seeds[split], meta["root_index"]):
                    raise ValueError("Unexpected split/root/environment seed derivation.")
                identities[split].append(identity)
                verify_hashes({str(path): hashes[str(path)]})
                yield root

    paired.validate_split_disjointness(roots())
    if split_seeds is not None:
        for split, rows in identities.items():
            if sorted(r["root_index"] for r in rows) != list(range(config["counts"][split])):
                raise ValueError("Require every prespecified contiguous root index, without replacement.")
    return identities


def authenticate_original(data):
    data = Path(data).resolve(strict=True)
    path = _ordinary(data / "manifest-development.json")
    manifest, receipt = read_json(path)
    config = read_config(PROJECT / "configs/pilot.json")
    if receipt != MANIFEST_SHA256 or config["counts"] != ORIGINAL_COUNTS:
        raise ValueError("Require the authenticated original 576-root discovery manifest.")
    paths, hashes = manifest_paths(data, manifest, config)
    hashes[str(path)] = receipt
    identities = scan_archives(paths, hashes, config, split_seeds=paired.DEFAULT_SPLIT_SEEDS)
    return {"path": str(data), "manifest": manifest, "input_sha256": hashes, "identities": identities}


def additional_exclusions(directories):
    """Optional nonstandard historical rosters: inventory EVERY NPZ, no subset flag.

    No claim of external authentication for these extra local inputs. Their bytes
    and complete identities are captured and rechecked. They can only abort for
    overlap, never change the prespecified fresh roster.
    """
    identities, hashes = [], {}
    for directory in directories:
        directory = Path(directory).resolve(strict=True)
        files = sorted(directory.rglob("*.npz"))
        if not files:
            raise ValueError("An additional exclusion directory must contain root archives.")
        for path in progress(files, desc="Additional historical root exclusions (CPU)", unit="root"):
            _ordinary(path)
            before = digest(path)
            root = paired.load_root(path)
            if path.stem != root.metadata["root_id"]:
                raise ValueError("Additional archive filename differs from its identity.")
            identities.append(pipeline._identity(root))
            hashes[str(path)] = before
            verify_hashes({str(path): before})
    return identities, hashes


def generate_dataset(output, original, config, simulator, plan_sha256):
    """Only pilot generation. Reused scorer/validation bytes keep their provenance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split in REUSED_SPLITS:
        for relative, expected in progress(sorted(original["manifest"]["files"][split].items()),
                                           desc=f"Reuse original {split} bytes (CPU)", unit="root"):
            copy_exclusive(Path(original["path"]) / relative, output / relative, expected)
    counts = {split: ROOTS if split == "pilot" else 0 for split in paired.SPLITS}
    seeds = dict(paired.DEFAULT_SPLIT_SEEDS, pilot=PILOT_SPLIT_SEED)
    generated = paired.generate_splits(output, counts=counts, split_seeds=seeds,
                                      config=paired.GenerationConfig(horizon=HORIZON, candidates=CANDIDATES),
                                      simulator=simulator)
    if {split: len(paths) for split, paths in generated.items()} != counts:
        raise ValueError("Generator did not return exactly the new pilot roster.")
    files = {split: {} for split in paired.SPLITS}
    for split in REUSED_SPLITS:
        files[split] = dict(original["manifest"]["files"][split])
    files["pilot"] = {p.relative_to(output).as_posix(): digest(p) for p in generated["pilot"]}
    manifest = {
        "config": config, "files": files, "runtime": runtime(),
        "replication_provenance": {
            "schema": "wm_audit.replication.data.v1", "plan_sha256": plan_sha256,
            "original_manifest_sha256": MANIFEST_SHA256,
            "reused_splits": list(REUSED_SPLITS), "reuse": "byte-identical exclusive copies; metadata/runtime unchanged",
            "generated_counts": counts, "generated_split_seeds": seeds,
            "generation_api": "wm_audit.data.generate_splits", "generator_source_sha256": SOURCE_PINS["data.py"],
            "generation_config": asdict(paired.GenerationConfig(horizon=HORIZON, candidates=CANDIDATES)),
            "simulator_source": simulator.source,
            "manifest_runtime_scope": "new composite dataset publication, NOT regeneration of reused archives",
        },
    }
    write_json(output / "manifest-development.json", manifest)
    paths, hashes = manifest_paths(output, manifest, config)
    hashes[str(output / "manifest-development.json")] = digest(output / "manifest-development.json")
    identities = scan_archives(paths, hashes, config, split_seeds=seeds)
    if identities["pilot"] != planned_roots():
        raise ValueError("Generated pilot differs from the pre-outcome fixed roster.")
    for split in REUSED_SPLITS:
        if identities[split] != original["identities"][split]:
            raise ValueError("Original scorer/validation roster changed during reuse.")
    return manifest, hashes, identities


def bootstrap_indices(root_ids, *, seed=BOOTSTRAP_SEED, resamples=RESAMPLES):
    if (not isinstance(root_ids, list) or not root_ids or any(not isinstance(x, str) or not x for x in root_ids)
            or root_ids != sorted(set(root_ids)) or type(seed) is not int or seed < 0
            or type(resamples) is not int or resamples < 1):
        raise ValueError("Bootstrap needs a sorted unique full root roster and fixed integer seed/count.")
    return np.random.default_rng(seed).integers(len(root_ids), size=(resamples, len(root_ids)), dtype=np.int64)


def candidate_keys(name):
    modes = ("free_none", "actual_none") if name == "lewm" else (
        "free_recurrent", "free_reset", "actual_recurrent", "actual_reset")
    keys = {*modes, "initial_persistence", "local_persistence", *(m + "_minus_baseline" for m in modes)}
    keys.update(("actual_minus_free_none",) if name == "lewm" else (
        "actual_minus_free_recurrent", "actual_minus_free_reset", "free_reset_minus_recurrent",
        "actual_reset_minus_recurrent", "visual_aux_interaction"))
    return keys | {"effect_error", "true_effect_energy", "predicted_effect_energy", "effect_dot",
                   "effect_error_minus_zero", "matched_minus_all_wrong"}


def _finite_array(value, shape, label):
    # Inspect before coercion: [True, 2.] must not silently become numeric data.
    objects = np.asarray(value, dtype=object)
    if objects.shape != shape or any(type(x) not in (int, float) for x in objects.flat):
        raise ValueError(f"Invalid numeric shape/types for {label}.")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {label}; no filtering allowed.")
    return result


def aggregate_records(records, name, root_ids, *, horizon=HORIZON, candidates=CANDIDATES):
    """Validate complete banks, then average candidates (in bank) and TWO branches.

    Return one vector per metric/slice, always in the SAME sorted root order.
    Missing/duplicated cells, metrics, candidates or zero-effect banks cannot be
    repaired by dropping rows. The compact root vectors make weighting auditable.
    """
    if name not in MODELS or root_ids != sorted(set(root_ids)) or not root_ids:
        raise ValueError("Invalid model or root roster.")
    if not isinstance(records, list) or len(records) != len(root_ids) * len(APPEARANCES) * 2 * horizon:
        raise ValueError("Incomplete dynamics record roster.")
    ranking_keys = set(dynamics.ranking_metrics(np.zeros(candidates), np.zeros(candidates), np.ones(candidates))[0])
    physical_keys = {"block_effect_units", "pusher_effect_units", "angle_effect_degrees", "rgb_effect_mse"}
    ckeys, cells = candidate_keys(name), {}
    mkeys = ckeys | ranking_keys | physical_keys
    wanted_roots = set(root_ids)
    for row in records:
        if not isinstance(row, dict) or set(row) != {"cluster", "appearance", "branch", "horizon", "metrics",
                "candidates", "prediction_to_actual_mse", "chosen", "endpoint_scores", "forecast_scores", "true_costs"}:
            raise ValueError("Malformed dynamics record fields.")
        if (row["cluster"] not in wanted_roots or row["appearance"] not in APPEARANCES
                or type(row["branch"]) is not int or row["branch"] not in (0, 1)
                or type(row["horizon"]) is not int or not 1 <= row["horizon"] <= horizon):
            raise ValueError("Unexpected root/appearance/branch/horizon in dynamics records.")
        key = (row["appearance"], row["horizon"], row["cluster"], row["branch"])
        if key in cells:
            raise ValueError("Duplicate dynamics cell; roots/branches must not be reweighted.")
        if not isinstance(row["metrics"], dict) or set(row["metrics"]) != mkeys:
            raise ValueError("Missing or extra dynamics metrics/contrasts.")
        if not isinstance(row["candidates"], dict) or set(row["candidates"]) != ckeys:
            raise ValueError("Incomplete candidate-level metric roster.")
        metrics = {k: float(_finite_array([v], (1,), k)[0]) for k, v in row["metrics"].items()}
        arrays = {}
        for k, values in row["candidates"].items():
            n = candidates - 1 if k.startswith("effect_") or k.endswith("effect_energy") else candidates
            arrays[k] = _finite_array(values, (n,), k)
            if not math.isclose(metrics[k], float(arrays[k].mean()), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Candidate mean differs from the reported bank metric.")
        matrix = _finite_array(row["prediction_to_actual_mse"], (candidates, candidates), "assignment matrix")
        if (matrix < 0).any():
            raise ValueError("Assignment MSE cannot be negative.")
        matched = matrix.diagonal() - (matrix.sum(1) - matrix.diagonal()) / (candidates - 1)
        if (not np.allclose(matched, arrays["matched_minus_all_wrong"], rtol=1e-10, atol=1e-12)
                or not np.allclose(arrays["effect_error"] - arrays["true_effect_energy"],
                                   arrays["effect_error_minus_zero"], rtol=1e-10, atol=1e-12)):
            raise ValueError("Action-fidelity/specificity contrast differs from its complete bank.")
        scores = [_finite_array(row[k], (candidates,), k)
                  for k in ("endpoint_scores", "forecast_scores", "true_costs")]
        ranking, chosen = dynamics.ranking_metrics(*scores)
        if (not isinstance(row["chosen"], dict) or row["chosen"] != chosen
                or any(type(v) is not int for v in row["chosen"].values())
                or any(not math.isclose(metrics[k], v, rel_tol=1e-10, abs_tol=1e-12) for k, v in ranking.items())):
            raise ValueError("Ranking/regret/tie rule differs from the saved candidate bank.")
        cells[key] = metrics
    result = {}
    for appearance in APPEARANCES:
        for step in range(1, horizon + 1):
            result[appearance, step] = {
                k: np.asarray([(cells[appearance, step, root, 0][k] + cells[appearance, step, root, 1][k]) / 2
                               for root in root_ids], dtype=np.float64) for k in sorted(mkeys)}
    return result


def bootstrap_means(values, indices):
    """Paired root bootstrap, bounded temporary memory; no row-level resampling."""
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all()
            or not isinstance(indices, np.ndarray) or indices.ndim != 2 or not len(indices)
            or indices.shape[1] != len(values) or indices.dtype.kind not in "iu"
            or (indices < 0).any() or (indices >= len(values)).any()):
        raise ValueError("Invalid complete root vectors/bootstrap indices.")
    estimates = np.empty((len(indices), values.shape[1]), dtype=np.float64)
    for start in range(0, len(indices), 256):
        estimates[start:start + 256] = values[indices[start:start + 256]].mean(1)
    if not np.isfinite(estimates).all():
        raise ValueError("Nonfinite bootstrap result; no resamples may be dropped.")
    return estimates


def decision(interval):
    if interval[1] < 0:
        return "supports_prespecified_negative_direction"
    if interval[0] > 0:
        return "opposite_direction"
    return "inconclusive"


def analyze_reports(reports, root_ids, indices):
    """All five primary contrasts together; every other metric stays exploratory."""
    if set(reports) != set(MODELS) or len(root_ids) != ROOTS:
        raise ValueError("Both frozen models and all 128 fresh roots are mandatory.")
    if (indices.dtype != np.int64 or not np.array_equal(indices, bootstrap_indices(root_ids))):
        raise ValueError("Bootstrap draws differ from the 10,000 prespecified shared root resamples.")
    root_means, secondary, primary = {}, {}, []
    for name in MODELS:
        groups = aggregate_records(reports[name]["records"], name, root_ids,
                                   horizon=HORIZON, candidates=CANDIDATES)
        root_means[name], secondary[name] = [], []
        for (appearance, horizon), columns in progress(list(groups.items()),
                                                       desc=f"Replication intervals {name} (CPU)", unit="slice"):
            keys = list(columns)
            matrix = np.column_stack([columns[k] for k in keys])
            boot = bootstrap_means(matrix, indices)
            points = matrix.mean(0)
            pointwise = np.quantile(boot, [.025, .975], axis=0, method="linear")
            metrics = {k: {"estimate": float(points[i]), "ci95": pointwise[:, i].tolist(), "clusters": ROOTS}
                       for i, k in enumerate(keys)}
            dot, energy = keys.index("effect_dot"), keys.index("true_effect_energy")
            if (matrix[:, energy] < 0).any():
                raise ValueError("Negative true effect energy.")
            undefined = int((boot[:, energy] <= 0).sum())
            gain = float(points[dot] / points[energy]) if points[energy] > 0 else None
            gain_ci = None if undefined else np.quantile(boot[:, dot] / boot[:, energy], [.025, .975], method="linear").tolist()
            if (gain is not None and not math.isfinite(gain)) or (gain_ci is not None and not np.isfinite(gain_ci).all()):
                raise ValueError("Nonfinite aggregate effect gain.")
            metrics["aggregate_effect_gain"] = {"estimate": gain, "ci95": gain_ci, "clusters": ROOTS,
                                                "undefined_resamples": undefined}
            secondary[name].append({"appearance": appearance, "horizon": horizon, "metrics": metrics})
            root_means[name].append({"appearance": appearance, "horizon": horizon,
                                     "metrics": {k: v.tolist() for k, v in columns.items()}})
            if appearance == "original" and horizon == HORIZON:
                for model_name, metric in PRIMARY:
                    if model_name == name:
                        i = keys.index(metric)
                        ci = np.quantile(boot[:, i], [.005, .995], method="linear").tolist()
                        primary.append({"model": name, "metric": metric, "estimate": float(points[i]),
                                        "ci99": ci, "clusters": ROOTS, "decision": decision(ci)})
    if [(p["model"], p["metric"]) for p in primary] != list(PRIMARY):
        raise ValueError("Primary family is not the fixed five contrasts.")
    passed = {(p["model"], p["metric"]): p["decision"] == "supports_prespecified_negative_direction" for p in primary}
    return {
        "schema": "wm_audit.replication.analysis.v1", "root_ids": root_ids,
        "status": "prespecified fresh-root replication; not external preregistration",
        "primary_family": {"size": 5, "fwer_alpha": .05, "method": "Bonferroni two-sided percentile bootstrap",
                           "individual_confidence": .99, "quantiles": [.005, .995], "resamples": RESAMPLES,
                           "seed": BOOTSTRAP_SEED, "appearance": "original", "horizon": HORIZON,
                           "unit": "equal-root mean after candidate means and equal two-branch means",
                           "shared_resamples_across_models_and_metrics": True, "claims": primary},
        "decisions": {
            "action_fidelity_and_specificity_by_model": {
                name: all(passed[name, metric] for metric in ("effect_error_minus_zero", "matched_minus_all_wrong"))
                for name in MODELS},
            "dino_forecast_beats_endpoint_ranking": passed["dinowm", "forecast_minus_endpoint_regret"],
            "all_five_supported": all(passed.values()),
            "rule": "support only when the entire 99% interval is strictly below zero; touching/crossing zero is inconclusive",
        },
        "exploratory_pointwise95": secondary,
        "exploratory_note": "All metric/slice estimates, including primary metrics' pointwise views; NOT additional primary tests.",
        "root_means": root_means, "limitations": LIMITATIONS,
    }


def validate_report(report, name, config, spec, data_hashes, model_metadata, state_sha256):
    required_inputs = {path: value for path, value in data_hashes.items()
                       if Path(path).name == "manifest-development.json" or Path(path).parent.name == "pilot"}
    if (report.get("schema") != "wm_audit.diagnostic.v1" or report.get("kind") != "followup_dynamics"
            or report.get("config") != config or report.get("spec") != spec
            or report.get("code_sha256") != SOURCE_PINS or report.get("input_sha256") != required_inputs
            or report.get("model") != model_metadata or report.get("parameter_buffer_sha256") != state_sha256
            or report.get("protocol", {}).get("world_model_trained") is not False
            or report.get("protocol", {}).get("readout_used") is not False):
        raise ValueError(f"Unqualified or mismatched {name} dynamics report provenance.")


def hosted_preflight(cache, *, trust_checkpoint, threads):
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Research execution requires hosted Linux CUDA; CPU/MPS/local fallback is forbidden.")
    if trust_checkpoint is not True:
        raise PermissionError("Both models are mandatory; --trust-checkpoint authorizes only the pinned DINO release.")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer.")
    torch.set_num_threads(threads)
    cache = Path(cache).resolve(strict=True)
    # Prevent the adapter's optional missing-DINOv2 download path during a run.
    hashes = {str(cache / name): value for name, value in models.RELEASE_HASHES.items()}
    hashes[str(cache / "models/dinov2/dinov2_vits14_pretrain.pth")] = models.DINOV2_WEIGHTS_SHA256
    verify_hashes(hashes)
    if models._python_tree_digest(cache / "sources/dinov2") != models.DINOV2_TREE_SHA256:
        raise ValueError("Pinned DINOv2 source must already be present; no implicit setup/download.")
    x = torch.ones(8, 8, device="cuda")
    if (x @ x).sum().item() != 512:
        raise RuntimeError("CUDA kernel preflight failed.")
    return hashes


def execute(args):
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output exists, including partial runs; never overwrite or implicitly resume.")
    if any(p.is_symlink() for p in output.parents):
        raise ValueError("Output ancestors must not be symlinks.")
    config, spec = checked_settings(args.config)
    sources = source_receipt(args.config)
    with stage("Authenticate ALL original splits and root separation (CPU)"):
        original = authenticate_original(args.original_data)
        extra, extra_hashes = additional_exclusions(args.exclude_data)
        excluded = [row for rows in original["identities"].values() for row in rows] + reserved_roots() + extra
        fresh = planned_roots()
        require_disjoint(fresh, excluded)
    root_ids = [r["root_id"] for r in fresh]
    plan = {"schema": "wm_audit.replication.plan.v1", "config": config, "spec": spec,
            "pilot_split_seed": PILOT_SPLIT_SEED, "pilot_roots": fresh,
            "original_manifest_sha256": MANIFEST_SHA256, "original_identities": original["identities"],
            "reserved_counts": RESERVED_COUNTS, "preflight_seed_excluded": 987654321,
            "additional_excluded_identities": extra, "additional_exclusion_sha256": extra_hashes,
            "source_receipts": sources, "original_input_sha256": original["input_sha256"],
            "primary_family": [dict(model=m, metric=k) for m, k in PRIMARY],
            "resampling": {"seed": BOOTSTRAP_SEED, "resamples": RESAMPLES, "rng": "numpy.PCG64",
                           "root_order": root_ids, "ci99_quantiles": [.005, .995], "quantile_method": "linear"},
            "f1": "not executed; original scorer/validation retained byte-identically, no refit/selection",
            "output": str(output), "limitations": LIMITATIONS}
    if not args.execute:
        return {"executed": False, "plan": plan}
    started = monotonic()
    with stage("Hosted CUDA and pinned cache preflight"):
        cache_hashes = hosted_preflight(args.cache, trust_checkpoint=args.trust_checkpoint, threads=args.threads)
        simulator = PushTSimulator(args.cache / "sources/point/stable-worldmodel")
    output.mkdir(parents=True, exist_ok=False)
    watched = original["input_sha256"] | extra_hashes | cache_hashes
    watched.update({v["path"]: v["sha256"] for v in sources.values()})
    for name, receipt in sources.items():
        target = output / "sources" / name
        copy_exclusive(receipt["path"], target, receipt["sha256"])
        watched[str(target)] = receipt["sha256"]
    original_manifest = output / "original-manifest-development.json"
    copy_exclusive(Path(original["path"]) / "manifest-development.json", original_manifest, MANIFEST_SHA256)
    watched[str(original_manifest)] = MANIFEST_SHA256
    indices = bootstrap_indices(root_ids)
    with (output / "bootstrap-indices.npy").open("xb") as stream:
        np.save(stream, indices, allow_pickle=False)
    watched[str(output / "bootstrap-indices.npy")] = digest(output / "bootstrap-indices.npy")
    plan.update(runtime=runtime(), bootstrap_indices_sha256=watched[str(output / "bootstrap-indices.npy")])
    write_json(output / "plan.json", plan)  # Published BEFORE generation or test encodings.
    watched[str(output / "plan.json")] = digest(output / "plan.json")
    with stage("Generate ONLY 128 new pilot roots; reuse original scorer/validation (CPU)"):
        manifest, data_hashes, identities = generate_dataset(output / "data", original, config, simulator,
                                                            watched[str(output / "plan.json")])
        require_disjoint(identities["pilot"], excluded)
        watched.update(data_hashes)
        verify_hashes(watched)
    write_json(output / "data-complete.json", {"input_sha256": data_hashes, "identities": identities,
                                               "original_manifest_sha256": MANIFEST_SHA256})
    watched[str(output / "data-complete.json")] = digest(output / "data-complete.json")
    generation_seconds = monotonic() - started
    reports, times = {}, {}
    for name in MODELS:
        with stage(f"Frozen {name} replication on CUDA; all roots, branches, candidates, appearances"):
            manifest_paths(output / "data", manifest, config)
            verify_hashes(watched)
            begin = monotonic()
            model = models.load_model(name, args.cache, "cuda", trust_checkpoint=args.trust_checkpoint)
            state_sha256 = dynamics._state_digest(model)
            target = output / "dynamics" / name
            result = dynamics.dynamics_followup(config, output / "data", target, model, spec=spec)
            if Path(result["output"]) != target / "report.json":
                raise ValueError("Unexpected dynamics report location.")
            report, report_hash = read_json(target / "report.json")
            validate_report(report, name, config, spec, data_hashes, model.metadata, state_sha256)
            if dynamics._state_digest(model) != state_sha256:
                raise ValueError("Model parameters/buffers changed after dynamics evaluation.")
            if set(report["artifacts"]) != {"index.html"}:
                raise ValueError("Unexpected dynamics artifact roster.")
            watched[str(target / "report.json")] = report_hash
            watched[str(target / "index.html")] = report["artifacts"]["index.html"]
            reports[name], times[name] = report, monotonic() - begin
            del model
            gc.collect()
            torch.cuda.empty_cache()
            verify_hashes(watched)
    with stage("Joint five-contrast replication decision; all secondary intervals (CPU)"):
        analysis = analyze_reports(reports, root_ids, indices)
        analysis.update(plan_sha256=watched[str(output / "plan.json")], data_manifest_sha256=digest(output / "data/manifest-development.json"),
                        bootstrap_indices_sha256=plan["bootstrap_indices_sha256"],
                        dynamics_report_sha256={name: watched[str(output / "dynamics" / name / "report.json")] for name in MODELS},
                        runtime=runtime(), timing_seconds={"preflight_generation_validation": generation_seconds, **times})
        manifest_paths(output / "data", manifest, config)
        verify_hashes(watched)
        write_json(output / "replication-results.json", analysis)
    # Explicit recursive inventory: the older generic backup helper does not
    # recognize these new stage names. Completion is LAST, never inferred from HTML.
    files = sorted(p for p in output.rglob("*") if p.is_file() or p.is_symlink())
    inventory = {p.relative_to(output).as_posix(): {"sha256": digest(_ordinary(p)), "bytes": p.stat().st_size}
                 for p in progress(files, desc="Seal complete replication inventory (CPU)", unit="file")}
    verify_hashes(watched)
    write_json(output / "complete.json", {"schema": "wm_audit.replication.complete.v1", "files": inventory,
                                          "seconds": monotonic() - started, "runtime": runtime()})
    note("Replication complete; use the joint replication-results.json, not legacy exploratory HTML, for primary decisions.")
    return {"executed": True, "output": str(output), "decisions": analysis["decisions"],
            "seconds": monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/replication_20261102.json")
    parser.add_argument("--original-data", type=Path, required=True)
    parser.add_argument("--exclude-data", type=Path, action="append", default=[],
                        help="Additional historical root directories; every NPZ is inventoried, never a selected subset")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Otherwise authenticate and print the plan only")
    args = parser.parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    args.cache = args.cache.expanduser().resolve()
    try:
        result = execute(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, PermissionError) as error:
        raise SystemExit(f"replication: {error}") from error
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()