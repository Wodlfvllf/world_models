"""Read-only, independent CPU/data verification. Prints JSON; never writes files.

Only stdlib and NumPy are imported. No production analysis, model, checkpoint,
torch, simulator, network, or archive extraction is executed. Git is read-only.
All malformed records fail closed: no partial-root analysis or draw filtering.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import stat
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

BASE = Path(__file__).absolute().parent
WORK = BASE.parents[1]
REP = BASE / "replication-restored/replication"
OPS = BASE / "operations-restored"
OLD = WORK / "runs/recovered-20261031/runs/pilot-gpu-01/data"
PIN = "36d6c1f4b744d808d1955ed50a3f7bd72280ba31"
OLD_SHA = "fc759e41bd66c48954f3d41568a9e48893ece47ae0a9719229109d08a8086485"
SPLITS = ("editor", "scorer", "validation", "pilot", "confirmation")
SEEDS = dict(zip(SPLITS, (10201, 20323, 30449, 40577, 50707)))
OLD_COUNTS = dict(zip(SPLITS, (256, 128, 64, 128, 0)))
RESERVED = dict(zip(SPLITS, (2000, 600, 256, 128, 400)))
NEW_COUNTS = dict(zip(SPLITS, (0, 128, 64, 128, 0)))
APPEARANCES = ("original", "color_only", "background_only")
STREAMS = ("anchor", "intervention", "goal", "actions", "appearance", "environment")
PRIMARY = (("lewm", "effect_error_minus_zero"), ("lewm", "matched_minus_all_wrong"),
           ("dinowm", "effect_error_minus_zero"), ("dinowm", "matched_minus_all_wrong"),
           ("dinowm", "forecast_minus_endpoint_regret"))
ARCHIVES = {
    "numerical-lewm": ("lewm", 967237, 65, "d244e69da1a693725b94491612e798bf3091ae8acbb230aa92eed610fafb98ee"),
    "numerical-dinowm": ("dinowm", 940404, 65, "4c84faef80b1557a9edfd836cea163c9304326268965ce9a6e05664d0d47c7ea"),
    "replication": ("replication-restored", 350515773, 415, "7e4ffb90d484e7fc5f736715d23ebb22e0af9da0ce83d38494a79748a557e27d"),
    "operations": ("operations-restored", 351221, 65, "22d49add52737c4cf9258f671e40c7eeced2d508377e07be444c6d45983dc31e"),
}
FAIL, WARN, COUNTS, ROUNDING = [], [], Counter(), {}
START = time.monotonic()


def check(ok, label):
    COUNTS["checks"] += 1
    if not bool(ok):
        FAIL.append(label)
    return bool(ok)


def require(ok, label):
    if not check(ok, label):
        raise ValueError(label)


def progress(label, done=None, total=None, began=None):
    elapsed = time.monotonic() - (START if began is None else began)
    eta = None if not done or total is None else elapsed * (total - done) / done
    print(f"CPU/data {label}: {done or 0}/{total or '?'} elapsed={elapsed:.1f}s"
          + (f" ETA={eta:.1f}s" if eta is not None else ""), file=sys.stderr, flush=True)


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def reject(value):
    raise ValueError("nonfinite JSON token: " + value)


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def read(path):
    return parse(path.read_bytes())


def sha(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "ordinary file: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(root, name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and p.as_posix() == name, "safe inventory path: " + str(name))
    return root / name


def files(root):
    paths = list(root.rglob("*"))
    check(not any(p.is_symlink() for p in paths), "no symlinks in " + str(root))
    check(all(p.is_file() or p.is_dir() for p in paths), "no special files in " + str(root))
    return {p.relative_to(root).as_posix() for p in paths if p.is_file()}


def avg(values):
    return math.fsum(float(v) for v in values) / len(values)


def numeric(values, shape, label):
    raw = np.asarray(values, dtype=object)
    require(raw.shape == shape and all(type(x) in (int, float) for x in raw.flat), label + " shape/types")
    result = np.asarray(values, dtype=np.float64)
    require(np.isfinite(result).all(), label + " finite; no filtering")
    return result


def close(actual, expected, label, *, rtol=1e-10, atol=1e-12, category="record"):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return check(False, label + " shape/nonfinite")
    delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    ROUNDING[category] = max(ROUNDING.get(category, 0.0), delta)
    return check(np.allclose(a, b, rtol=rtol, atol=atol), label)


def guarded(label, function, *args):
    try:
        return function(*args)
    except Exception as error:
        FAIL.append(label + ": aborted (not silently omitted): " + type(error).__name__ + ": " + str(error))
        return None


def archive(stage):
    folder, size, count, expected = ARCHIVES[stage]
    root, path = BASE / folder, BASE / (stage + ".zip")
    actual = sha(path)
    check(actual == expected and path.stat().st_size == size, stage + " pinned archive SHA/size")
    check((BASE / (stage + ".zip.sha256")).read_text().split() == [actual, path.name], stage + " SHA receipt")
    backup, marker = read(root / "BACKUP.json"), read(root / "RESTORE-VERIFIED.json")
    inventory = backup["files"]
    require(backup["schema"] == "wm_audit.backup.v1" and backup["stage"] == stage
            and len(inventory) == count, stage + " complete backup scope")
    check(marker == {"archive_sha256": actual, "files_verified": count,
                     "completed_roots": backup["completed_roots"]}, stage + " restore receipt")
    check(files(root) == set(inventory) | {"BACKUP.json", "RESTORE-VERIFIED.json"}, stage + " restored exact file set")
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        check(len(members) == len({m.filename for m in members}), stage + " no duplicate ZIP members")
        check({m.filename for m in members} == set(inventory) | {"BACKUP.json"}, stage + " ZIP exact file set")
        check(z.read("BACKUP.json") == (root / "BACKUP.json").read_bytes(), stage + " archived manifest bytes")
        began = time.monotonic()
        for i, (name, rec) in enumerate(inventory.items(), 1):
            def entry():
                p = safe(root, name)
                check(p.stat().st_size == rec["size_bytes"] and sha(p) == rec["sha256"], stage + " restored " + name)
                info = z.getinfo(name)
                check(info.file_size == rec["size_bytes"] and not stat.S_ISLNK(info.external_attr >> 16), stage + " member type/size " + name)
                h = hashlib.sha256()
                with z.open(info) as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                check(h.hexdigest() == rec["sha256"], stage + " member SHA/CRC " + name)
            guarded(stage + " inventory " + name, entry)
            if i % 100 == 0 or i == count:
                progress(stage + " archive/restored inventory", i, count, began)
    COUNTS["archive_inventory_entries"] += count
    return {"sha256": actual, "bytes": path.stat().st_size, "inventory_entries": count}


def git(*args):
    return subprocess.run(["git", *args], cwd=WORK, check=True, capture_output=True).stdout


def sources(plan):
    require(git("rev-parse", "--verify", PIN + "^{commit}").decode().strip() == PIN, "exact local commit object")
    launch = read(OPS / "launch.json")
    require(launch["source_commit"] == PIN and launch["completed_discovery_rerun"] is False
            and launch["implicit_resume"] is False, "launch exact source pin/no discovery rerun")
    for remote, expected in launch["source_sha256"].items():
        relative = remote.split("/world_models_restored_20261103/", 1)[1]
        p = safe(OPS / "sources", relative)
        check(sha(p) == expected, "launch source SHA " + relative)
        check(hashlib.sha256(git("show", PIN + ":" + relative)).hexdigest() == expected, "commit source bytes " + relative)
    for name, rec in plan["source_receipts"].items():
        check(sha(safe(REP / "sources", name)) == rec["sha256"], "replication source snapshot " + name)
        check(hashlib.sha256(git("show", PIN + ":" + name)).hexdigest() == rec["sha256"], "replication exact commit snapshot " + name)
    COUNTS["launch_sources_matched_to_commit"] = len(launch["source_sha256"])
    COUNTS["replication_sources_matched_to_commit"] = len(plan["source_receipts"])
    # All immutable archive copies of the launch refer to identical pinned bytes.
    for folder, _, _, _ in ARCHIVES.values():
        root = BASE / folder
        check((root / "launch.json").read_bytes() == (OPS / "launch.json").read_bytes(), folder + " identical launch")
        for p in (root / "sources").rglob("*"):
            if p.is_file():
                rel = p.relative_to(root / "sources").as_posix()
                check(hashlib.sha256(git("show", PIN + ":" + rel)).hexdigest() == sha(p), folder + " commit snapshot " + rel)
    try:
        remote_ref = git("rev-parse", "refs/remotes/origin/main").decode().strip()
        publication = {"local_origin_main": remote_ref, "pin_reachable_from_local_origin_main":
                       subprocess.run(["git", "merge-base", "--is-ancestor", PIN, remote_ref], cwd=WORK).returncode == 0}
        check(publication["pin_reachable_from_local_origin_main"], "pin in locally recorded published history")
    except subprocess.CalledProcessError:
        publication = {"local_origin_main": None}
        WARN.append("No local origin/main reference; remote publication cannot be independently attested offline.")
    return publication


def identity(split, split_seed, index):
    def derive(seed, *labels):
        raw = json.dumps([seed, *labels], separators=(",", ":"), ensure_ascii=True).encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")
    seed = derive(split_seed, "wm_audit.pusht.paired.v1", split, index)
    return {"root_id": f"{split}-{index:06d}-{seed:032x}", "root_index": index,
            "root_seed": seed, "split_seed": split_seed, "environment_seed": derive(seed, "environment")}, derive


ARRAYS = {
    "context": ((3, 2, 3, 224, 224, 3), "uint8"), "future": ((3, 2, 8, 8, 224, 224, 3), "uint8"),
    "goal_pixels": ((3, 224, 224, 3), "uint8"), "states_context": ((2, 3, 6), "float64"),
    "states_future": ((2, 8, 8, 6), "float64"), "actions": ((8, 10, 10), "float32"),
    "costs": ((2, 8), "float64"), "request": ((6,), "float64"),
    "coverage_context": ((2, 3), "float64"), "coverage_future": ((2, 8, 8), "float64"),
}


def inspect_npz(path, expected_id, *, pilot=False):
    # Authenticate inner ZIP dimensions and payload lengths without loading RGB.
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist()) == 11 and set(z.namelist()) == {k + ".npy" for k in (*ARRAYS, "metadata")}, str(path) + " exact NPZ members")
        for name in z.namelist():
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                require(version in ((1, 0), (2, 0)), str(path) + " supported NPY header")
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                                         else np.lib.format.read_array_header_2_0(f))
                field = name[:-4]
                require(not dtype.hasobject and not fortran, str(path) + " safe ordinary array " + field)
                if field == "metadata":
                    require(shape == () and dtype.kind == "U", str(path) + " Unicode metadata")
                else:
                    require((shape, str(dtype)) == ARRAYS[field], str(path) + " schema " + field)
                require(z.getinfo(name).file_size == f.tell() + math.prod(shape) * dtype.itemsize, str(path) + " exact NPY payload length " + field)
    with np.load(path, allow_pickle=False) as z:
        meta = parse(z["metadata"].item())
        actual, derive = identity(meta["split"], meta["split_seed"], meta["root_index"])
        require(all(type(meta[k]) is int for k in ("root_index", "root_seed", "split_seed"))
            and all(type(v) is int for v in meta["stream_seeds"].values()), path.stem + " strict identity integer types")
        observed = {k: meta[k] for k in ("root_id", "root_index", "root_seed", "split_seed")}
        observed["environment_seed"] = meta["stream_seeds"]["environment"]
        require(observed == actual == expected_id and path.stem == actual["root_id"], str(path) + " independent identity")
        check(meta["schema"] == "wm_audit.pusht.paired.v1" and meta["stream_seeds"] == {s: derive(actual["root_seed"], s) for s in STREAMS}, path.stem + " all independent seed streams")
        check(meta["config"] == {"horizon": 8, "candidates": 8, "anchor_clearance": 2.0, "max_anchor_attempts": 256}, path.stem + " unchanged generation parameters")
        sampling = meta["sampling"]
        check(meta["actions_source"] == "fixed_bank_v1" and sampling["anchor_source"] == "sampled"
              and sampling["eligibility"] == "anchor_only" and sampling["future_filtering"] is False
              and sampling["rng"] == "numpy.PCG64" and 1 <= sampling["attempts"] <= 256
              and sampling["rejected_anchors"] == sampling["attempts"] - 1, path.stem + " no outcome filtering receipt")
        arrays = {k: z[k] for k in ARRAYS if k not in ("context", "future", "goal_pixels")}
        check(all(np.isfinite(v).all() for v in arrays.values()), path.stem + " finite numeric arrays")
        check(not np.any(arrays["actions"][:, :2]) and not np.any(arrays["actions"][0])
              and np.max(np.abs(arrays["actions"])) <= 1, path.stem + " resting prefix/hold/native bounds")
        for k in ("costs", "coverage_context", "coverage_future"):
            check(np.all((arrays[k] >= 0) & (arrays[k] <= 1)), path.stem + " bounded " + k)
        close(arrays["costs"], 1 - arrays["coverage_future"][:, :, -1], path.stem + " endpoint coverage costs", rtol=0)
        for k in ("states_context", "states_future"):
            close(np.sum(arrays[k][..., 4:6] ** 2, axis=-1), np.ones(arrays[k].shape[:-1]), path.stem + " unit orientation", rtol=0, atol=1e-10)
        close(arrays["request"], arrays["states_context"][1, 0] - arrays["states_context"][0, 0], path.stem + " anchor intervention delta", rtol=0, atol=1e-10)
        raw = arrays["actions"]
        h = hashlib.sha256(json.dumps([raw.dtype.str, raw.shape], separators=(",", ":")).encode() + raw.tobytes()).hexdigest()
        check(h == meta["audit"]["actions_sha256"], path.stem + " action-array digest")
        check([a["name"] for a in meta["appearances"]] == list(APPEARANCES), path.stem + " appearance roster")
        check(all(recipe["seed"] == actual["environment_seed"] for recipe in meta["recipes"])
              and len(meta["recipes"]) == 2 and meta["recipes"][0]["goal"] == meta["recipes"][1]["goal"], path.stem + " matched branch goal/environment")
        # Retain only small physical/coverage arrays, never raw latent tensors.
        if pilot:
            arrays["source"] = meta["source"]
            return arrays
    return None


def dataset(plan, analysis):
    original = read(REP / "original-manifest-development.json")
    manifest = read(REP / "data/manifest-development.json")
    done = read(REP / "data-complete.json")
    require(sha(REP / "original-manifest-development.json") == sha(OLD / "manifest-development.json") == OLD_SHA
            == plan["original_manifest_sha256"] == done["original_manifest_sha256"], "original manifest exact authenticated bytes")
    fresh = [identity("pilot", 2026110201, i)[0] for i in range(128)]
    original_ids = {s: [identity(s, SEEDS[s], i)[0] for i in range(OLD_COUNTS[s])] for s in SPLITS}
    require(plan["pilot_split_seed"] == 2026110201 and plan["pilot_roots"] == fresh, "all 128 fresh pilot IDs/seeds/indices")
    require(plan["original_identities"] == original_ids and plan["reserved_counts"] == RESERVED
            and plan["preflight_seed_excluded"] == 987654321, "full old/default/preflight saved exclusion plan")
    exclusions = [x for roster in original_ids.values() for x in roster]
    exclusions += [identity(s, SEEDS[s], i)[0] for s in SPLITS for i in range(RESERVED[s])]
    exclusions += [identity("pilot", 987654321, 0)[0]] + plan["additional_excluded_identities"]
    for key in ("root_id", "root_seed", "environment_seed"):
        a = [r[key] for r in fresh]
        check(len(set(a)) == 128 and not set(a) & {r[key] for r in exclusions}, "all historical/reserved separation " + key)
    require(not plan["additional_excluded_identities"] and not plan["additional_exclusion_sha256"], "this run has no supplied additional historical exclusions")
    root_ids = sorted(r["root_id"] for r in fresh)
    require(plan["resampling"] == {"seed": 20261102, "resamples": 10000, "rng": "numpy.PCG64",
            "root_order": root_ids, "ci99_quantiles": [0.005, 0.995], "quantile_method": "linear"}
            and analysis["root_ids"] == root_ids, "fixed complete paired bootstrap/root order")
    check(plan["primary_family"] == [{"model": m, "metric": k} for m, k in PRIMARY], "fixed primary family plan")
    require(original["config"]["counts"] == OLD_COUNTS and manifest["config"] == plan["config"]
            and manifest["config"]["counts"] == NEW_COUNTS, "old/new exact split counts and config")
    provenance = manifest["replication_provenance"]
    check(provenance["plan_sha256"] == sha(REP / "plan.json") and provenance["original_manifest_sha256"] == OLD_SHA
          and provenance["reused_splits"] == ["scorer", "validation"]
          and provenance["generated_counts"] == {s: (128 if s == "pilot" else 0) for s in SPLITS}
          and provenance["generated_split_seeds"] == dict(SEEDS, pilot=2026110201), "composite manifest actual generation/reuse provenance")
    old_remote = read(OPS / "launch.json")["original_data"]
    expected_original_inputs = {old_remote + "/" + rel: h for s in SPLITS for rel, h in original["files"][s].items()}
    expected_original_inputs[old_remote + "/manifest-development.json"] = OLD_SHA
    require(plan["original_input_sha256"] == expected_original_inputs, "entire original 577-file input receipt")
    pilot_arrays, expected_new_inputs = {}, {}
    for root, man, counts, identities in ((OLD, original, OLD_COUNTS, original_ids),
                                         (REP / "data", manifest, NEW_COUNTS, dict(original_ids, pilot=fresh, editor=[], confirmation=[]))):
        require(set(man["files"]) == set(SPLITS), str(root) + " all five splits")
        wanted = {rel for s in SPLITS for rel in man["files"][s]}
        check(files(root) == wanted | {"manifest-development.json"}, str(root) + " full dataset exact file set")
        began, total, i = time.monotonic(), sum(counts.values()), 0
        for split in SPLITS:
            roster = man["files"][split]
            require(len(roster) == counts[split] and set(roster) == {split + "/" + x["root_id"] + ".npz" for x in identities[split]}, str(root) + " exact contiguous roster " + split)
            by_id = {x["root_id"]: x for x in identities[split]}
            for rel, expected in sorted(roster.items()):
                p = safe(root, rel)
                check(sha(p) == expected, "NPZ manifest hash " + str(p))
                # Keyword argument is explicit; no invalid archives are discarded.
                try:
                    arr = inspect_npz(p, by_id[p.stem], pilot=root != OLD and split == "pilot")
                except Exception as error:
                    FAIL.append("NPZ " + str(p) + ": invalid, not omitted: " + type(error).__name__ + ": " + str(error))
                    arr = None
                if root != OLD:
                    expected_new_inputs[plan["output"] + "/data/" + rel] = expected
                    if split in ("scorer", "validation"):
                        check(roster[rel] == original["files"][split][rel] == sha(OLD / rel), "byte-identical original reused root " + rel)
                    elif arr is not None:
                        pilot_arrays[p.stem] = arr
                i += 1
                if i % 128 == 0 or i == total:
                    progress("original" if root == OLD else "composite" + " root bytes/schema", i, total, began)
        COUNTS["original_npzs" if root == OLD else "materialized_npzs"] = total
    expected_new_inputs[plan["output"] + "/data/manifest-development.json"] = sha(REP / "data/manifest-development.json")
    require(done["input_sha256"] == expected_new_inputs and done["identities"] == dict(original_ids, pilot=fresh, editor=[], confirmation=[]), "complete data validation receipt matches independently derived roster")
    draws = np.random.Generator(np.random.PCG64(20261102)).integers(128, size=(10000, 128), dtype=np.int64)
    saved = np.load(REP / "bootstrap-indices.npy", allow_pickle=False)
    require(saved.dtype == np.int64 and np.array_equal(saved, draws), "all 1,280,000 fixed PCG64 bootstrap indices")
    buffer = io.BytesIO()
    np.save(buffer, draws, allow_pickle=False)
    require(hashlib.sha256(buffer.getvalue()).hexdigest() == sha(REP / "bootstrap-indices.npy")
            == plan["bootstrap_indices_sha256"] == analysis["bootstrap_indices_sha256"], "independently regenerated bootstrap bytes/hash")
    counts = np.asarray([np.bincount(row, minlength=128) for row in draws], dtype=np.float64)
    COUNTS["bootstrap_draws"] = len(draws)
    COUNTS["bootstrap_indices"] = draws.size
    require(len(pilot_arrays) == 128, "all fresh NPZ data valid; no partial-root inference")
    return root_ids, counts, pilot_arrays


def operations(analysis):
    status, complete = read(OPS / "status.json"), read(OPS / "complete.json")
    require(status["state"] == "completed" and status["completed"] is True and status["source_commit"] == PIN,
            "terminal operational status/source pin")
    require(complete["schema"] == "wm_audit.tracked_studies.complete.v1" and complete["source_commit"] == PIN
            and complete["status_sha256"] == sha(OPS / "status.json") and complete["stages"] == status["stages"], "terminal completion authenticated status hash/stages")
    check((BASE / "status.json").read_bytes() == (OPS / "status.json").read_bytes()
          and (BASE / "complete.json").read_bytes() == (OPS / "complete.json").read_bytes(), "top-level final status/completion identical")
    check(read(BASE / "operations-receipt.json") == {"path": read(OPS / "launch.json")["output"] + "/operations.zip",
          "sha256": ARCHIVES["operations"][3], "bytes": ARCHIVES["operations"][1], "inventory_entries": 65}, "operations archive receipt")
    seal = read(REP / "complete.json")
    require(seal["schema"] == "wm_audit.replication.complete.v1", "replication completion schema")
    check(files(REP) == set(seal["files"]) | {"complete.json"}, "recursive replication completion exact file set")
    for rel, rec in seal["files"].items():
        def entry():
            p = safe(REP, rel)
            check(p.stat().st_size == rec["bytes"] and sha(p) == rec["sha256"], "replication completion inventory " + rel)
        guarded("completion inventory " + rel, entry)
    COUNTS["replication_completion_entries"] = len(seal["files"])
    require(analysis["plan_sha256"] == sha(REP / "plan.json")
            and analysis["data_manifest_sha256"] == sha(REP / "data/manifest-development.json"), "joint report plan/data hash links")
    events = [parse(line) for line in (OPS / "events.jsonl").read_text().splitlines()]
    stamps = [datetime.fromisoformat(e["utc"]) for e in events]
    check(stamps == sorted(stamps), "operational events chronological")
    stages, last = {}, None
    for stage in ("numerical-lewm", "numerical-dinowm", "replication"):
        rec = status["stages"][stage]
        require(rec["state"] == "completed" and rec["exit_code"] == 0, stage + " terminal success")
        check(rec["archive"]["sha256"] == ARCHIVES[stage][3] and rec["archive"]["bytes"] == ARCHIVES[stage][1]
              and rec["archive"]["inventory_entries"] == ARCHIVES[stage][2], stage + " terminal archive ledger")
        target = REP / "replication-results.json" if stage == "replication" else BASE / stage.removeprefix("numerical-") / stage / "report.json"
        check(rec["verified"]["report_sha256"] == sha(target), stage + " terminal report hash")
        began, ended = map(datetime.fromisoformat, (rec["started_utc"], rec["finished_utc"]))
        check(ended > began and (last is None or began >= last), stage + " positive serialized elapsed time")
        last = ended
        selected = [e for e in events if e["stage"] == stage]
        for kind in ("started", "process_exit", "verified", "archived"):
            check(sum(e["event"] == kind for e in selected) == 1, stage + " exactly one event " + kind)
        check(all(e["exit_code"] == 0 for e in selected if e["event"] == "process_exit"), stage + " logged exit zero")
        stages[stage] = (ended - began).total_seconds()
    check(status["stages"]["replication"]["verified"]["primary_claims"] == analysis["primary_family"]["claims"]
          and status["stages"]["replication"]["verified"]["decisions"] == analysis["decisions"], "terminal primary claims/decisions unchanged")
    check((BASE / "replication-restored/replication.log").read_bytes() == (OPS / "replication.log").read_bytes(), "final replication log identical")
    snapshots = {}
    for folder, _, _, _ in ARCHIVES.values():
        s = read(BASE / folder / "status.json")
        snapshots[folder] = {"state": s["state"], "completed": s["completed"]}
    WARN.append("Immutable per-stage archives preserve earlier overall status; final operations-restored status/completion is authoritative.")
    COUNTS["operational_events"] = len(events)
    return {"final_status_sha256": sha(OPS / "status.json"), "final_complete_sha256": sha(OPS / "complete.json"),
            "replication_complete_sha256": sha(REP / "complete.json"), "replication_results_sha256": sha(REP / "replication-results.json"),
            "stage_snapshots": snapshots, "timing_seconds": {"stage_wall": stages,
            "batch_launch_to_final": (datetime.fromisoformat(complete["utc"]) - datetime.fromisoformat(read(OPS / "launch.json")["utc"])).total_seconds(),
            "replication_sealed_elapsed": seal["seconds"], "replication_reported_components": analysis["timing_seconds"]}}


def ranking(row):
    end = numeric(row["endpoint_scores"], (8,), "endpoint scores")
    fore = numeric(row["forecast_scores"], (8,), "forecast scores")
    costs = numeric(row["true_costs"], (8,), "true costs")
    require(np.all((costs >= 0) & (costs <= 1)), "true cost bounds")
    choices = {"endpoint": min(range(8), key=lambda k: (end[k], k)),
               "forecast": min(range(8), key=lambda k: (fore[k], k))}
    require(row["chosen"] == choices and all(type(v) is int for v in row["chosen"].values()), "smallest-index exact ties/choices")
    best = min(costs)
    result = {name + "_regret": float(costs[index] - best) for name, index in choices.items()}
    result.update(hold_regret=float(costs[0] - best), uniform_regret=avg(costs) - best,
                  best_cost=float(best), best_coverage=1 - float(best), cost_range=float(max(costs) - best),
                  no_positive_coverage=float(all(x == 1 for x in costs)))
    for name in choices:
        for baseline in ("hold", "uniform"):
            result[name + "_minus_" + baseline] = result[name + "_regret"] - result[baseline + "_regret"]
    result["forecast_minus_endpoint_regret"] = result["forecast_regret"] - result["endpoint_regret"]
    COUNTS["choices_checked"] += 2
    return result


def bank(row, model, physical):
    modes = ("free_none", "actual_none") if model == "lewm" else ("free_recurrent", "free_reset", "actual_recurrent", "actual_reset")
    effect_keys = ("effect_error", "true_effect_energy", "predicted_effect_energy", "effect_dot", "effect_error_minus_zero")
    extra = ("actual_minus_free_none",) if model == "lewm" else (
        "actual_minus_free_recurrent", "actual_minus_free_reset", "free_reset_minus_recurrent", "actual_reset_minus_recurrent", "visual_aux_interaction")
    ckeys = set(modes) | {"initial_persistence", "local_persistence", "matched_minus_all_wrong"} | set(effect_keys) | set(extra) | {m + "_minus_baseline" for m in modes}
    require(set(row) == {"cluster", "appearance", "branch", "horizon", "metrics", "candidates", "prediction_to_actual_mse", "chosen", "endpoint_scores", "forecast_scores", "true_costs"}, "complete record fields")
    require(set(row["candidates"]) == ckeys, "complete candidate metric roster")
    arrays = {k: numeric(v, (7 if k in effect_keys else 8,), k) for k, v in row["candidates"].items()}
    for k, values in arrays.items():
        close(avg(values), row["metrics"][k], "candidate -> bank mean " + k)
        COUNTS["candidate_numbers_checked"] += len(values)
    result = {k: avg(v) for k, v in arrays.items()}
    # The archived F2 candidate contrast arithmetic was float32 before export.
    f32 = {k: arrays[k].astype(np.float32) for k in (*modes, "initial_persistence", "local_persistence")}
    check(all(np.array_equal(f32[k].astype(np.float64), arrays[k]) for k in f32), "native F2 float32 scalar representations")
    reconstructed = {m + "_minus_baseline": f32[m] - f32["initial_persistence" if m.startswith("free") else "local_persistence"] for m in modes}
    for aux in ("none", "recurrent", "reset"):
        if "actual_" + aux in modes:
            reconstructed["actual_minus_free_" + aux] = f32["actual_" + aux] - f32["free_" + aux]
    if model == "dinowm":
        for visual in ("free", "actual"):
            reconstructed[visual + "_reset_minus_recurrent"] = f32[visual + "_reset"] - f32[visual + "_recurrent"]
        reconstructed["visual_aux_interaction"] = f32["actual_reset"] - f32["free_reset"] - f32["actual_recurrent"] + f32["free_recurrent"]
    for k, values in reconstructed.items():
        close(values.astype(np.float64), arrays[k], "independent F2 contrast " + k, rtol=0, atol=0)
        result[k] = avg(values)
    if row["horizon"] == 1:
        check(all(np.array_equal(arrays[m], arrays[modes[0]]) for m in modes), "step-one variant metric parity")
    matrix = numeric(row["prediction_to_actual_mse"], (8, 8), "complete all-wrong assignment matrix")
    require(np.all(matrix >= 0), "nonnegative assignment MSE")
    matched = [matrix[k, k] - avg([matrix[k, j] for j in range(8) if j != k]) for k in range(8)]
    close(matched, arrays["matched_minus_all_wrong"], "all 56 off-diagonal assignments")
    result["matched_minus_all_wrong"] = avg(matched)
    native = "free_none" if model == "lewm" else "free_recurrent"
    close(np.diag(matrix), arrays[native], "float64 assignment diagonal versus native float32 MSE", rtol=2e-6, atol=2e-8, category="f64_matrix_vs_f32_native")
    error, true, predicted, dot = [arrays[k] for k in effect_keys[:4]]
    require(np.all(error >= 0) and np.all(true >= 0) and np.all(predicted >= 0), "nonnegative action-effect energies/error")
    contrast = error - true
    close(contrast, arrays["effect_error_minus_zero"], "independent zero-effect contrast")
    close(error, predicted + true - 2 * dot, "effect squared-error energy/dot identity")
    check(np.all(dot ** 2 <= predicted * true + 1e-12), "effect Cauchy-Schwarz bounds")
    result["effect_error_minus_zero"] = avg(contrast)
    result.update(ranking(row))
    branch, step = row["branch"], row["horizon"] - 1
    close(row["true_costs"], 1 - physical["coverage_future"][branch, :, step], "report candidate costs from saved NPZ coverage", rtol=0, atol=0)
    state = physical["states_future"][branch, :, step]
    result["block_effect_units"] = avg([math.hypot(*(s[2:4] - state[0, 2:4])) for s in state[1:]])
    result["pusher_effect_units"] = avg([math.hypot(*(s[:2] - state[0, :2])) for s in state[1:]])
    angles = [math.atan2(s[4], s[5]) for s in state]
    result["angle_effect_degrees"] = avg([abs(math.atan2(math.sin(a - angles[0]), math.cos(a - angles[0]))) for a in angles[1:]]) * 180 / math.pi
    result["rgb_effect_mse"] = numeric([row["metrics"]["rgb_effect_mse"]], (1,), "RGB effect scalar")[0]
    require(0 <= result["rgb_effect_mse"] <= 1, "bounded normalized RGB effect scalar")
    # RGB is integrity checked by whole-NPZ hashes; no latent/RGB encoding occurs.
    require(set(row["metrics"]) == set(result), "complete bank-level metric roster")
    for k, value in result.items():
        close(value, row["metrics"][k], "independently reconstructed bank metric " + k)
    COUNTS["assignment_matrix_numbers"] += 64
    COUNTS["bank_records"] += 1
    return result


def decision(ci):
    return "supports_prespecified_negative_direction" if ci[1] < 0 else ("opposite_direction" if ci[0] > 0 else "inconclusive")


def model_analysis(model, plan, analysis, root_ids, multiplicities, physical):
    report = read(REP / "dynamics" / model / "report.json")
    require(report["schema"] == "wm_audit.diagnostic.v1" and report["kind"] == "followup_dynamics"
            and report["model"]["name"] == model and report["config"] == plan["config"]
            and report["spec"] == plan["spec"] and report["confirmatory"] is False
            and report["protocol"]["world_model_trained"] is False and report["protocol"]["readout_used"] is False,
            model + " report model/config/spec/no training/no readout")
    core = {PurePosixPath(k).name: v["sha256"] for k, v in plan["source_receipts"].items() if k.startswith("src/wm_audit/")}
    require(report["code_sha256"] == core, model + " qualified source receipts")
    expected_inputs = {k: v for k, v in read(REP / "data-complete.json")["input_sha256"].items() if "/pilot/" in k or k.endswith("/manifest-development.json")}
    require(report["input_sha256"] == expected_inputs, model + " all pilot report input hashes")
    require(sha(REP / "dynamics" / model / "report.json") == analysis["dynamics_report_sha256"][model], model + " joint report SHA")
    require(set(report["artifacts"]) == {"index.html"} and sha(REP / "dynamics" / model / "index.html") == report["artifacts"]["index.html"], model + " report artifact integrity")
    require(len(report["records"]) == 6144, model + " full 128x3x2x8 record count")
    cells, began, failures_before = {}, time.monotonic(), len(FAIL)
    for i, row in enumerate(report["records"], 1):
        try:
            key = (row["appearance"], row["horizon"], row["cluster"], row["branch"])
            require(row["cluster"] in root_ids and row["appearance"] in APPEARANCES
                    and type(row["branch"]) is int and row["branch"] in (0, 1)
                    and type(row["horizon"]) is int and row["horizon"] in range(1, 9) and key not in cells,
                    model + " unique valid bank identity " + str(i))
            cells[key] = bank(row, model, physical[row["cluster"]])
        except Exception as error:
            FAIL.append(model + " invalid record " + str(i) + ": no partial inference: " + type(error).__name__ + ": " + str(error))
        if i % 2048 == 0:
            progress(model + " all candidate banks", i, 6144, began)
    wanted = {(a, h, r, b) for a in APPEARANCES for h in range(1, 9) for r in root_ids for b in (0, 1)}
    require(set(cells) == wanted and len(FAIL) == failures_before, model + " all banks valid; no bad data/roots dropped")
    stored_roots = {(x["appearance"], x["horizon"]): x["metrics"] for x in analysis["root_means"][model]}
    stored_stats = {(x["appearance"], x["horizon"]): x["metrics"] for x in analysis["exploratory_pointwise95"][model]}
    require(len(stored_roots) == len(analysis["root_means"][model]) == len(stored_stats) == len(analysis["exploratory_pointwise95"][model]) == 24
            and set(stored_roots) == set(stored_stats) == {(a, h) for a in APPEARANCES for h in range(1, 9)}, model + " every stored root/secondary slice")
    primary, secondary, canonical_roots = [], {}, []
    for a in APPEARANCES:
        for h in range(1, 9):
            keys = sorted(cells[a, h, root_ids[0], 0])
            # No stored root means are used as input: equal branch means rebuilt.
            matrix = np.asarray([[avg([cells[a, h, r, 0][k], cells[a, h, r, 1][k]]) for k in keys] for r in root_ids])
            require(set(stored_roots[a, h]) == set(keys), model + " root metric roster")
            for j, k in enumerate(keys):
                close(matrix[:, j], numeric(stored_roots[a, h][k], (128,), "stored root comparison"), model + " independent equal-branch roots " + str((a, h, k)), category="root_vectors")
            points = np.asarray([avg(matrix[:, j]) for j in range(len(keys))])
            # Independent bootstrap implementation: draw multiplicities, not the
            # production gather/mean routine. Same 10,000 complete root draws.
            boot = multiplicities @ matrix / 128
            require(np.isfinite(boot).all(), model + " all unconditional bootstrap values finite")
            ci95 = np.quantile(boot, [0.025, 0.975], axis=0, method="linear")
            stats = {k: {"estimate": float(points[j]), "ci95": ci95[:, j].tolist(), "clusters": 128} for j, k in enumerate(keys)}
            numerator, denominator = keys.index("effect_dot"), keys.index("true_effect_energy")
            undefined = int(np.count_nonzero(boot[:, denominator] <= 0))
            gain = float(points[numerator] / points[denominator]) if points[denominator] > 0 else None
            gain_ci = None if undefined else np.quantile(boot[:, numerator] / boot[:, denominator], [0.025, 0.975], method="linear").tolist()
            stats["aggregate_effect_gain"] = {"estimate": gain, "ci95": gain_ci, "clusters": 128, "undefined_resamples": undefined}
            require(set(stats) == set(stored_stats[a, h]), model + " complete exploratory metric roster")
            for k, rec in stats.items():
                old = stored_stats[a, h][k]
                check(set(rec) == set(old) and old["clusters"] == 128, model + " interval metadata " + k)
                if rec["estimate"] is None or rec["ci95"] is None:
                    check(rec == old, model + " unconditional undefined gain; no draw filtering")
                else:
                    close([rec["estimate"], *rec["ci95"]], [old["estimate"], *old["ci95"]], model + " recomputed 95% " + str((a, h, k)), category="estimates_intervals")
                    if "undefined_resamples" in rec:
                        check(rec["undefined_resamples"] == old["undefined_resamples"], model + " undefined gain draws")
                COUNTS["exploratory_estimates_intervals"] += 1
            canonical_roots.append({"appearance": a, "horizon": h, "metrics": {k: matrix[:, j].tolist() for j, k in enumerate(keys)}})
            if a == "original" and h == 8:
                secondary = {k: [v["estimate"], *(v["ci95"] or [None, None])] for k, v in stats.items()}
                secondary["aggregate_effect_gain_undefined_resamples"] = undefined
                secondary["zero_coverage_banks"] = int(sum(cells[a, h, r, b]["no_positive_coverage"] for r in root_ids for b in (0, 1)))
                for m, k in PRIMARY:
                    if m == model:
                        j = keys.index(k)
                        ci = np.quantile(boot[:, j], [0.005, 0.995], method="linear").tolist()
                        primary.append({"model": model, "metric": k, "estimate": float(points[j]), "ci99": ci, "clusters": 128, "decision": decision(ci)})
            COUNTS["root_means_recomputed"] += matrix.size
        progress(model + " reconstructed root intervals", APPEARANCES.index(a) + 1, 3, began)
    canonical = json.dumps(canonical_roots, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return primary, secondary, hashlib.sha256(canonical).hexdigest()


def main():
    progress("independent verification start")
    check(sys.executable == str(WORK / ".venv/bin/python"), "exact configured local .venv interpreter")
    archives = {stage: guarded(stage, archive, stage) for stage in ARCHIVES}
    plan = guarded("saved plan", read, REP / "plan.json")
    analysis = guarded("joint analysis receipt", read, REP / "replication-results.json")
    publication = guarded("source pin", sources, plan) if plan is not None else None
    terminal = guarded("final operations/completion", operations, analysis) if analysis is not None else None
    data = guarded("entire dataset/identities/resamples", dataset, plan, analysis) if plan is not None and analysis is not None else None
    primary, secondary, roots = [], {}, {}
    if data is not None:
        for model in ("lewm", "dinowm"):
            result = guarded(model + " independent inference", model_analysis, model, plan, analysis, *data)
            if result is not None:
                claims, secondary[model], roots[model] = result
                primary.extend(claims)
    if analysis is not None:
        require_primary = [(p["model"], p["metric"]) for p in primary] == list(PRIMARY)
        check(require_primary, "all five independently recomputed primary claims; incomplete family cannot pass")
        if require_primary:
            published = analysis["primary_family"]
            check(published["size"] == 5 and published["fwer_alpha"] == 0.05 and published["individual_confidence"] == 0.99
                  and published["quantiles"] == [0.005, 0.995] and published["resamples"] == 10000
                  and published["seed"] == 20261102 and published["appearance"] == "original" and published["horizon"] == 8
                  and published["shared_resamples_across_models_and_metrics"] is True, "fixed five-member primary interval design")
            check(len(published["claims"]) == 5, "exact published primary count")
            for p, q in zip(primary, published["claims"], strict=True):
                check(all(p[k] == q[k] for k in ("model", "metric", "clusters", "decision")), "independent primary decision/identity")
                close([p["estimate"], *p["ci99"]], [q["estimate"], *q["ci99"]], "independent primary 99% " + p["model"] + "/" + p["metric"], category="estimates_intervals")
            support = {(p["model"], p["metric"]): p["decision"] == "supports_prespecified_negative_direction" for p in primary}
            check(analysis["decisions"]["all_five_supported"] == all(support.values()) and
                  analysis["decisions"]["dino_forecast_beats_endpoint_ranking"] == support["dinowm", "forecast_minus_endpoint_regret"] and
                  analysis["decisions"]["action_fidelity_and_specificity_by_model"] == {m: all(support[m, k] for k in ("effect_error_minus_zero", "matched_minus_all_wrong")) for m in ("lewm", "dinowm")}, "joint decisions independently reproduced")
    check(not any(name == "torch" or name.startswith("wm_audit") or name in ("pygame", "pymunk") for name in sys.modules), "no model/production/simulator imports")
    result = {"schema": "wm_audit.replication_independent_verification.v1", "utc": datetime.now(timezone.utc).isoformat(),
              "cpu_data_only": True, "read_only_verifier": True, "interpreter": sys.executable,
              "python": sys.version.split()[0], "numpy": np.__version__, "source_pin": PIN,
              "verifier_sha256": sha(Path(__file__).absolute()), "publication_evidence": publication,
              "archives": archives, "operations": terminal, "counts": dict(COUNTS),
              "maximum_absolute_comparison_differences": ROUNDING, "primary": primary,
              "secondary_original_h8": secondary, "secondary_value_format": ["estimate", "pointwise_ci95_low", "pointwise_ci95_high"],
              "recomputed_root_vectors_sha256": roots, "bootstrap": {"rng": "numpy.PCG64", "seed": 20261102,
              "resamples": 10000, "roots_per_draw": 128, "root_order": "sorted full root ID", "quantile_method": "linear",
              "sha256": plan["bootstrap_indices_sha256"] if plan else None},
              "method": "Independent SHA/size/ZIP CRC and exact inventories; strict JSON and NPZ headers/numeric data; SHA-derived identities and old/default/preflight exclusions; candidate reconstructions, all-wrong fsum, argmin ties and NPZ coverage costs; equal two-branch then equal-root fsum; shared draw-multiplicity bootstrap. No stored root mean is an input.",
              "limits": ["No raw latent tensors: stored candidate errors/energies/dot products/assignment matrices/goal scores cannot be regenerated from model outputs. Algebra, identities, choices and aggregation are verified, not raw latent measurements.",
                         "F2 float32 subtraction is reconstructed from exported scalar errors; latent persistence errors themselves are not reproduced.",
                         "RGB arrays have whole-NPZ integrity and exact header/payload checks; RGB effect scalar is reaggregated, not recalculated from pixels. No encoder is run.",
                         "Coverage costs and physical state-effect magnitudes are checked from saved NPZ arrays, not simulator replay or independent geometric coverage reconstruction.",
                         "Source pin is compared with local Git commit objects, recorded origin/main and archived launch bytes; no remote publication signature or runtime attestation is asserted.",
                         "All original 576 NPZs are rehashed and structurally/numerically checked; unknown nonstandard historical rosters cannot be certified.",
                         "Archive receipts are supplied checksums, not independent external signatures. Model/checkpoint/backend unchanged receipts and reported times are not reproduced by execution.",
                         "All 48 slices/metrics are independently recomputed and compared; only original/h8 secondary values are retained here. Five 99% primary intervals; all 95% intervals exploratory. Nominal/asymptotic coverage.",
                         "Integrity failures invalidate overall verification; malformed banks prohibit partial-root inference. No roots or undefined bootstrap draws are dropped."],
              "failed_checks": FAIL, "warnings": WARN, "passed": not FAIL,
              "verification_elapsed_seconds": time.monotonic() - START}
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return int(bool(FAIL))


if __name__ == "__main__":
    raise SystemExit(main())