"""Read-only CPU verification; no wm_audit, torch, models, or simulator imports.

Prints a JSON receipt to stdout, never writes artifacts. Existing report summaries
are independently recomputed with math.fsum, rather than the audit's NumPy mean.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
DATA = WORK / "runs/recovered-20261031/runs/pilot-gpu-01/data"
TF = ("context", "future_common", "target_h8", "rollout_all", "rollout_h8")
MF = ("native_f32", "persistence_f32", "native_f64", "persistence_f64")
COMPS = (
    ("observed_paths", "old_single", "new_feedback"),
    ("encoding_at_single", "old_single", "new_single"),
    ("encoding_at_bank", "old_bank", "new_bank"),
    ("rollout_at_old_encoding", "old_single", "old_bank"),
    ("rollout_at_new_encoding", "new_single", "new_bank"),
    ("feedback_vs_public", "new_bank", "new_feedback_scalar"),
    ("metric_reduction_only", "new_feedback_scalar", "new_feedback"),
)
VAR = ("old_single", "old_bank", "new_single", "new_bank", "new_feedback")
CONTROL = (
    ("matmul_ieee", "cuda.matmul.fp32_precision", "ieee"),
    ("matmul_tf32", "cuda.matmul.fp32_precision", "tf32"),
    ("cudnn_conv_ieee", "cudnn.conv.fp32_precision", "ieee"),
    ("cudnn_deterministic", "cudnn.deterministic", True),
    ("cudnn_benchmark_off", "cudnn.benchmark", False),
)
FAIL = []
WARN = []


def check(ok, label):
    if not ok:
        FAIL.append(label)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Nonfinite JSON constant " + value)

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject)


def sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def avg(values):
    return math.fsum(values) / len(values)


def group_key(row):
    return tuple(row[key] for key in ("control", "comparison", "lhs", "rhs"))


def identity(row):
    return tuple(row[key] for key in ("root_id", "appearance", "branch", "candidate", "horizon"))


def compact(summary):
    return {
        "native_f32": summary["metrics"]["native_f32"],
        "native_f64": summary["metrics"]["native_f64"],
        "persistence_f32": summary["metrics"]["persistence_f32"],
        "tensor_max_abs": {key: summary["tensors"][key]["max_abs"] for key in TF},
    }


def verify_model(model):
    root = BASE / model
    stage = "numerical-" + model
    output = root / stage
    report = read(output / "report.json")
    plan = read(output / "selection.json")
    backup = read(root / "BACKUP.json")
    marker = read(root / "RESTORE-VERIFIED.json")
    archive = BASE / (stage + ".zip")
    archive_sha = sha(archive)
    receipt = (BASE / (stage + ".zip.sha256")).read_text().split()
    check(receipt == [archive_sha, archive.name], model + " archive SHA receipt")
    check(marker == {"archive_sha256": archive_sha, "files_verified": len(backup["files"]),
                     "completed_roots": backup["completed_roots"]}, model + " RESTORE marker")
    check(len(backup["files"]) == 65 and backup["stage"] == stage
          and backup["schema"] == "wm_audit.backup.v1", model + " backup scope")
    with zipfile.ZipFile(archive) as zipped:
        check(zipped.testzip() is None, model + " ZIP CRC")
        names = zipped.namelist()
        check(len(names) == len(set(names)), model + " ZIP duplicate members")
        check(set(names) == set(backup["files"]) | {"BACKUP.json"}, model + " ZIP inventory completeness")
        check(zipped.read("BACKUP.json") == (root / "BACKUP.json").read_bytes(), model + " backup manifest equality")
        for name, record in backup["files"].items():
            path = root / name
            check(not Path(name).is_absolute() and ".." not in Path(name).parts
                  and not path.is_symlink(), model + " safe path " + name)
            check(path.is_file() and path.stat().st_size == record["size_bytes"]
                  and sha(path) == record["sha256"], model + " restored inventory " + name)
            payload = zipped.read(name)
            check(len(payload) == record["size_bytes"]
                  and hashlib.sha256(payload).hexdigest() == record["sha256"], model + " archived member " + name)
    check({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
          == set(backup["files"]) | {"BACKUP.json", "RESTORE-VERIFIED.json"}, model + " restored file set")
    artifacts = {name: sha(output / name) for name in ("report.json", "selection.json", "records.csv")}
    check(report["artifacts"] == {name: artifacts[name] for name in ("selection.json", "records.csv")},
          model + " report artifact checksums")
    status = read(root / "status.json")
    stage_status = status["stages"][stage]
    check(stage_status["state"] == "completed" and stage_status["exit_code"] == 0
          and stage_status["verified"]["report_sha256"] == artifacts["report.json"]
          and stage_status["verified"]["csv_records"] == 7680
          and stage_status["verified"]["roots"] == 8, model + " completed numerical stage")
    if not status["completed"]:
        WARN.append(model + ": archived overall tracker completed=false/state=" + status["state"]
                    + "; numerical stage completed with exit_code=0")
    for field in ("config", "spec", "baseline_backend", "input_sha256", "code_sha256"):
        check(plan[field] == report[field], model + " plan/report " + field)
    check(report["schema"] == "wm_audit.execution_layout.v1"
          and report["kind"] == "execution_layout" and report["confirmatory"] is False, model + " report kind")
    check(report["model"]["name"] == model and report["csv_records"] == 7680
          and len(report["summary"]) == 60, model + " report dimensions")
    before = report["parameter_buffer_sha256_before"]
    check(report["parameter_buffer_sha256_after"] == before == plan["parameter_buffer_sha256"]
          == report["model_state"]["parameter_buffer_sha256"] and report["model_unchanged"] is True
          and report["backend_restored"] is True, model + " model/backend receipts")
    roster = plan["roots"]
    check(len(roster) == 8 and [x["manifest_ordinal"] for x in roster] == list(range(8))
          and [x["root_index"] for x in roster] == list(range(8)), model + " first eight roots")
    manifest = read(DATA / "manifest-development.json")
    expected_manifest = "fc759e41bd66c48954f3d41568a9e48893ece47ae0a9719229109d08a8086485"
    check(sha(DATA / "manifest-development.json") == expected_manifest, model + " original manifest SHA")
    check([(x["relative_path"], x["sha256"]) for x in roster]
          == list(manifest["files"]["pilot"].items())[:8], model + " serialized manifest order")
    check(manifest["config"] == report["config"], model + " original manifest config")
    check(len(report["inputs"]) == 8 and len(report["input_sha256"]) == 11
          and len(report["code_sha256"]) == 17, model + " input/source receipt counts")
    for remote, expected in report["code_sha256"].items():
        relative = remote.split("/world_models_restored_20261103/", 1)[1]
        check(sha(root / "sources" / relative) == expected, model + " executed source " + relative)
    for remote, expected in report["input_sha256"].items():
        relative = remote.split("/world_models_restored_20261103/", 1)[1]
        local = DATA / relative.split("/data/", 1)[1] if "/data/" in relative else root / "sources" / relative
        check(sha(local) == expected, model + " original input " + local.name)
    for index, (item, input_record) in enumerate(zip(roster, report["inputs"], strict=True)):
        with np.load(DATA / item["relative_path"], allow_pickle=False) as arrays:
            check(json.loads(arrays["metadata"].item()) == input_record["root_metadata"],
                  model + " metadata " + str(index))
            for label, field in (("context_original", "context"), ("future_original", "future"),
                                 ("goal_original", "goal_pixels"), ("raw_actions", "actions")):
                value = arrays[field] if field == "actions" else arrays[field][0]
                header = json.dumps([str(value.dtype), list(value.shape)]).encode() + b"\0"
                actual = {"sha256": hashlib.sha256(header + np.ascontiguousarray(value).tobytes()).hexdigest(),
                          "shape": list(value.shape), "dtype": str(value.dtype)}
                check(actual == input_record["arrays"][label], model + " input array " + str(index) + " " + label)
        print(model, "authenticated input", index + 1, "/ 8", flush=True, file=sys.stderr)
    active = []
    check(len(plan["controls"]) == len(report["controls"]) == 5, model + " all five controls")
    for (name, path, value), planned, actual in zip(CONTROL, plan["controls"], report["controls"], strict=True):
        skipped = plan["baseline_backend"][path] == value
        expected = {"name": name, "path": path, "value": value, "before": plan["baseline_backend"][path],
                    "status": "skipped" if skipped else "planned",
                    "reason": "already set; no distinct intervention" if skipped else None}
        check(planned == expected, model + " planned control " + name)
        check(actual == {**expected, "status": "skipped" if skipped else "executed",
                         "suites_completed": 0 if skipped else 8}, model + " executed/skipped " + name)
        if not skipped:
            active.append(name)
    check(active == ["matmul_ieee", "matmul_tf32", "cudnn_conv_ieee", "cudnn_deterministic"],
          model + " four active controls")
    phases = ["baseline", "repeat"] + active
    executions = {(e["root_id"], e["phase"]): e for e in report["executions"]}
    check(len(executions) == len(report["executions"]) == 48
          and set(executions) == {(x["root_id"], phase) for x in roster for phase in phases}, model + " 48 suites")
    baseline = report["baseline_backend"]
    ancillary = {"matmul_tf32": {"cuda.matmul.allow_tf32", "float32_matmul_precision"},
                 "cudnn_conv_ieee": {"cudnn.allow_tf32"}}
    for (root_id, phase), execution in executions.items():
        check(execution["parameter_buffer_sha256"] == before, model + " suite model hash " + root_id + "/" + phase)
        backend = execution["backend"]
        allowed = ancillary.get(phase, set())
        changed = {key for key in baseline if baseline[key] != backend[key]}
        requested = set() if phase in ("baseline", "repeat") else {next(path for name, path, _ in CONTROL if name == phase)}
        check(changed == requested | allowed, model + " sole property/derived getters " + root_id + "/" + phase)
        if requested:
            _, path, value = next(x for x in CONTROL if x[0] == phase)
            check(backend[path] == value, model + " applied setting " + phase)
        for key in allowed:
            check(isinstance(backend[key], dict) and "unavailable" in backend[key], model + " legacy getter unavailability")
        fingerprints = execution["tensor_sha256"]
        check(set(fingerprints) == set(VAR) | {"new_feedback_scalar"}, model + " variant hash set")
        for hashes in fingerprints.values():
            check(set(hashes) == {"context", "future_common", "rollout_all"}
                  and all(re.fullmatch("[0-9a-f]{64}", h) for h in hashes.values()), model + " tensor hash fields")
        if phase in ("repeat", "matmul_ieee", "cudnn_deterministic"):
            check(fingerprints == executions[(root_id, "baseline")]["tensor_sha256"],
                  model + " exact output hash " + phase + " " + root_id)
        calls = execution["calls"]
        check(len(calls) == 76, model + " actual call count")
        old = [c for c in calls if c["phase"] == "old_observed"]
        new = [c for c in calls if c["phase"] == "new_observed"]
        bridge = [c for c in calls if c["phase"] == "crossed_public_bridges"]
        check([c["inputs"][0]["shape"][0] for c in old if c["method"] == "encode"] == [4, 2] + [3] * 16
              and len([c for c in old if c["method"] == "rollout"]) == 32, model + " old call layout")
        check([c["inputs"][0]["shape"][0] for c in new if c["method"] == "encode"] == [6, 1, 32, 32, 32, 32]
              and len([c for c in new if c["method"] == "rollout"]) == 2 and len(bridge) == 18, model + " new/bridge calls")
        coordinates = math.prod(report["latent_shape"])
        for call in new + bridge:
            if call["method"] == "rollout" and call["inputs"][0]["shape"][0] == 8:
                check(call["inputs"][0]["stride_elements"][0] == 11 * coordinates, model + " bank history stride")
    groups = defaultdict(list)
    numeric_fields = [f"{tensor}_{stat}" for tensor in TF for stat in ("max_abs", "mean_abs")]
    numeric_fields += [f"{prefix}{metric}{suffix}" for metric in MF
                       for prefix, suffix in (("lhs_", ""), ("rhs_", ""), ("", "_delta"))]
    with (output / "records.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        check(len(reader.fieldnames) == len(set(reader.fieldnames)) == 32, model + " CSV header")
        for line, row in enumerate(reader, 2):
            for name in ("manifest_ordinal", "branch", "candidate", "horizon"):
                row[name] = int(row[name])
            for name in numeric_fields:
                row[name] = float(row[name])
                check(math.isfinite(row[name]), model + " nonfinite record " + str(line))
            for metric in MF:
                check(row[metric + "_delta"] == row["rhs_" + metric] - row["lhs_" + metric],
                      model + " signed delta " + str(line) + " " + metric)
            for tensor in TF:
                check(0 <= row[tensor + "_mean_abs"] <= row[tensor + "_max_abs"], model + " tensor bounds " + str(line))
            check(row["target_h8_max_abs"] <= row["future_common_max_abs"]
                  and row["rollout_h8_max_abs"] <= row["rollout_all_max_abs"], model + " tensor horizon bounds")
            groups[group_key(row)].append(row)
    expected_groups = {(phase, comp, lhs, rhs) for phase in ["baseline"] + active for comp, lhs, rhs in COMPS}
    expected_groups |= {("repeat", "identical_layout_repeat", "baseline/" + v, "repeat/" + v) for v in VAR}
    expected_groups |= {(phase, "backend_only", "baseline/" + v, phase + "/" + v) for phase in active for v in VAR}
    check(set(groups) == expected_groups and len(groups) == 60 and sum(map(len, groups.values())) == 7680,
          model + " every comparison group")
    full_ids = {(x["root_id"], "original", branch, candidate, 8) for x in roster for branch in (0, 1) for candidate in range(8)}
    ordinals = {x["root_id"]: x["manifest_ordinal"] for x in roster}
    published = {group_key(s): s for s in report["summary"]}
    check(len(published) == len(report["summary"]) == 60 and set(published) == expected_groups, model + " summary keys")
    summaries = {}
    max_rounding = 0.0
    checked_numbers = 0
    variant_values = {}
    for key, rows in groups.items():
        check(len(rows) == 128 and len({identity(x) for x in rows}) == 128
              and {identity(x) for x in rows} == full_ids, model + " complete 8x2x8 bank " + str(key))
        check(all(x["manifest_ordinal"] == ordinals[x["root_id"]] for x in rows), model + " manifest ordinals " + str(key))
        tensors = {t: {"max_abs": max(x[t + "_max_abs"] for x in rows),
                       "mean_abs": avg([x[t + "_mean_abs"] for x in rows])} for t in TF}
        metrics = {m: {"lhs_mean": avg([x["lhs_" + m] for x in rows]),
                       "rhs_mean": avg([x["rhs_" + m] for x in rows]),
                       "mean_delta": avg([x[m + "_delta"] for x in rows]),
                       "max_abs_delta": max(abs(x[m + "_delta"]) for x in rows),
                       "mean_abs_delta": avg([abs(x[m + "_delta"]) for x in rows])} for m in MF}
        summaries[key] = {"control": key[0], "comparison": key[1], "lhs": key[2], "rhs": key[3],
                          "roots": 8, "records": 128, "tensors": tensors, "metrics": metrics}
        check(set(published[key]) == set(summaries[key]) and published[key]["roots"] == 8
              and published[key]["records"] == 128, model + " summary metadata")
        for block in ("tensors", "metrics"):
            check(set(published[key][block]) == set(summaries[key][block]), model + " summary fields")
            for field, stats in summaries[key][block].items():
                check(set(stats) == set(published[key][block][field]), model + " summary statistic keys")
                for stat, value in stats.items():
                    reported = published[key][block][field][stat]
                    max_rounding = max(max_rounding, abs(value - reported))
                    checked_numbers += 1
                    check(math.isfinite(reported) and math.isclose(value, reported, rel_tol=1e-12, abs_tol=1e-15),
                          model + " recomputed summary " + str(key) + "/" + block + "/" + field + "/" + stat)
        for row in rows:
            for side in ("lhs", "rhs"):
                variant, phase = row[side], key[0]
                if "/" in variant:
                    phase, variant = variant.split("/")
                variant_key = (phase, variant, identity(row))
                values = tuple(row[side + "_" + m] for m in MF)
                check(variant_key not in variant_values or variant_values[variant_key] == values,
                      model + " matched variant metric consistency")
                variant_values[variant_key] = values
    for phase in ["baseline"] + active:
        by_comp = {comp: {identity(x): x for x in groups[(phase, comp, lhs, rhs)]} for comp, lhs, rhs in COMPS}
        for record_id in full_ids:
            for metric in MF:
                observed = by_comp["observed_paths"][record_id][metric + "_delta"]
                for path in (("encoding_at_single", "rollout_at_new_encoding", "feedback_vs_public", "metric_reduction_only"),
                             ("rollout_at_old_encoding", "encoding_at_bank", "feedback_vs_public", "metric_reduction_only")):
                    total = math.fsum(by_comp[comp][record_id][metric + "_delta"] for comp in path)
                    check(math.isclose(observed, total, rel_tol=1e-12, abs_tol=1e-15), model + " recordwise crossed decomposition")
    canonical = json.dumps([summaries[key] for key in sorted(summaries)], sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()
    result = {
        "counts": {"rows": 7680, "groups": 60, "records_per_group": 128,
                   "numeric_summary_statistics_checked": checked_numbers,
                   "max_summary_rounding_difference": max_rounding, "suites": 48,
                   "inventoried_files": 65, "input_files_rehashed": 11,
                   "source_files_rehashed": 17, "input_array_receipts_recomputed": 32},
        "hashes": {"archive_sha256": archive_sha, **artifacts, "backup_sha256": sha(root / "BACKUP.json"),
                   "restore_marker_sha256": sha(root / "RESTORE-VERIFIED.json"),
                   "independently_recomputed_summaries_sha256": hashlib.sha256(canonical).hexdigest()},
        "controls": report["controls"],
        "baseline": {key[1]: compact(s) for key, s in summaries.items() if key[0] == "baseline"},
        "convolution_ieee_observed": compact(next(s for key, s in summaries.items()
                                                  if key[:2] == ("cudnn_conv_ieee", "observed_paths"))),
        "control_same_layout": {},
        "wrapper_reduction_all_settings": {},
    }
    for phase in ["repeat"] + active:
        matches = [s for key, s in summaries.items() if key[0] == phase
                   and key[1] in ("backend_only", "identical_layout_repeat")]
        result["control_same_layout"][phase] = {
            "variant_native_f32_max_abs_delta": {s["rhs"].split("/")[1]: s["metrics"]["native_f32"]["max_abs_delta"] for s in matches},
            "native_f64_max_abs_delta": max(s["metrics"]["native_f64"]["max_abs_delta"] for s in matches),
            "all_tensor_max_abs": max(s["tensors"][t]["max_abs"] for s in matches for t in TF),
        }
    for phase in ["baseline"] + active:
        result["wrapper_reduction_all_settings"][phase] = {}
        for comparison in ("feedback_vs_public", "metric_reduction_only"):
            summary = next(s for key, s in summaries.items() if key[:2] == (phase, comparison))
            result["wrapper_reduction_all_settings"][phase][comparison] = {
                "native_f32_max_abs_delta": summary["metrics"]["native_f32"]["max_abs_delta"],
                "native_f64_max_abs_delta": summary["metrics"]["native_f64"]["max_abs_delta"],
                "all_tensor_max_abs": max(x["max_abs"] for x in summary["tensors"].values()),
            }
    print(model, "verification complete; cumulative failed checks:", len(FAIL), flush=True, file=sys.stderr)
    return report, result


def main():
    check(sys.executable == str(WORK / ".venv/bin/python"), "Exact local .venv interpreter")
    reports, models = {}, {}
    for model in ("lewm", "dinowm"):
        try:
            reports[model], models[model] = verify_model(model)
        except Exception as error:
            FAIL.append(model + ": verification aborted: " + type(error).__name__ + ": " + str(error))
    if len(reports) == 2:
        check(reports["lewm"]["input_sha256"] == reports["dinowm"]["input_sha256"]
              and reports["lewm"]["inputs"] == reports["dinowm"]["inputs"], "Across-model input identity")
    receipt = {
        "schema": "wm_audit.numerical_independent_verification.v1",
        "cpu_only": True,
        "interpreter": sys.executable,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "verifier_sha256": sha(Path(__file__)),
        "method": "Independent csv.DictReader identity joins, math.fsum means, full archive/restored SHA and CRC checks; no audit code imports or model execution",
        "comparison_tolerance": {"relative": 1e-12, "absolute": 1e-15},
        "models": models,
        "failed_checks": FAIL,
        "warnings": WARN,
        "passed": not FAIL,
        "limits": [
            "No raw latent tensors are archived: tensor deltas and CPU float64 MSE are reaggregated from records, not recalculated from full tensors.",
            "Model and final backend restoration hashes are checked for report/suite consistency, not reproduced by loading models.",
            "The first eight original-manifest roots do not conclusively explain historical 128-root maxima or historical backend state.",
            "Archive SHA receipts establish integrity against supplied receipts, not an independent external signature.",
        ],
    }
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return int(bool(FAIL))


if __name__ == "__main__":
    raise SystemExit(main())
