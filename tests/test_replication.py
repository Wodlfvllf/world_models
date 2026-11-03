"""CPU SYNTHETIC contracts only: no release weights, native simulation or research.

Schema fixtures contain invented RGB/state trajectories and say so in their
runtime metadata. Stub generation below is ONLY a test of wiring/provenance.
It is never an alternative production generator or empirical replication.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wm_audit import data, followup_probes, models, pipeline
from wm_audit import followup_dynamics as f
from wm_audit.provenance import digest, read_config
from wm_audit.simulator import EpisodeRecipe, array_digest
from wm_audit.statistics import APPEARANCES
from test_followup_dynamics import _payload_model
from test_pipeline import _synthetic_root


ROOT = Path(__file__).resolve().parents[1]
_module_spec = importlib.util.spec_from_file_location("replication_contract_target", ROOT / "scripts/run_replication.py")
r = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(r)


def forbidden(*args, **kwargs):
    raise AssertionError("CPU contract tests must not load released models, simulate roots, download, fit or run research.")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def no_research(monkeypatch):
    monkeypatch.setenv("WM_AUDIT_PROGRESS", "0")
    monkeypatch.setattr(models, "load_model", forbidden)
    monkeypatch.setattr(models, "_download", forbidden)
    monkeypatch.setattr(r, "PushTSimulator", forbidden)
    monkeypatch.setattr(data, "generate_splits", forbidden)
    monkeypatch.setattr(data, "generate_root", forbidden)
    monkeypatch.setattr(followup_probes, "probe_followup", forbidden)
    monkeypatch.setattr(followup_probes, "_fit_probes", forbidden)
    monkeypatch.setattr(pipeline, "fit", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def tiny_config():
    config = read_config(ROOT / "configs/pilot.json")
    config.update(name="SYNTHETIC-replication-contract-not-research", horizon=2, candidates=3, horizons=[1, 2])
    config["counts"] = dict(editor=1, scorer=2, validation=1, pilot=2, confirmation=0)
    return config


def fixture_root(split, index, config, *, split_seed=None):
    """Invented schema fixture with a reproducible bank, NOT physically replayed."""
    root = _synthetic_root(split, index, config, split_seed=split_seed)
    recipe = EpisodeRecipe.from_dict(root.metadata["recipes"][0])
    root.actions[:] = data.make_candidate_bank(recipe.anchor, recipe.goal,
                                               root.metadata["stream_seeds"]["actions"],
                                               data.GenerationConfig(horizon=config["horizon"], candidates=config["candidates"]))
    root.metadata["actions_source"] = "fixed_bank_v1"
    root.metadata["sampling"]["anchor_source"] = "sampled"
    root.metadata["audit"]["actions_sha256"] = array_digest(root.actions)
    assert "synthetic_fixture" in root.metadata["runtime"]
    root.validate()
    return root


def write_fixture(directory, config):
    directory = directory.resolve()
    files, identities = {s: {} for s in data.SPLITS}, {s: [] for s in data.SPLITS}
    for split in data.SPLITS:
        for index in range(config["counts"][split]):
            root = fixture_root(split, index, config)
            path = data.save_root(directory / split / f"{root.metadata['root_id']}.npz", root)
            files[split][path.relative_to(directory).as_posix()] = digest(path)
            identities[split].append(pipeline._identity(root))
    manifest = {"config": config, "files": files, "runtime": {"synthetic": "not a research run"}}
    (directory / "manifest-development.json").write_text(json.dumps(manifest))
    hashes = {str(directory / relative): value for split in files.values() for relative, value in split.items()}
    hashes[str(directory / "manifest-development.json")] = digest(directory / "manifest-development.json")
    return {"path": str(directory), "manifest": manifest, "input_sha256": hashes, "identities": identities}


@pytest.fixture(scope="module")
def original_fixture(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("SYNTHETIC-original"), tiny_config())


def small_settings(monkeypatch, original):
    config = deepcopy(original["manifest"]["config"])
    config["counts"]["editor"] = 0
    config.update(seed=r.BOOTSTRAP_SEED, bootstrap_resamples=r.RESAMPLES)
    spec = json.loads((ROOT / "configs/followup_20261101.json").read_text())
    spec.update(seed=r.BOOTSTRAP_SEED, bootstrap_resamples=r.RESAMPLES, primary_horizon=2, encode_batch_size=4)
    monkeypatch.setattr(r, "ROOTS", 2)
    monkeypatch.setattr(r, "HORIZON", 2)
    monkeypatch.setattr(r, "CANDIDATES", 3)
    return config, spec


def install_synthetic_generator(monkeypatch, config, calls):
    def generate(output, *, counts, split_seeds, config: data.GenerationConfig, simulator):
        calls.append({"counts": counts.copy(), "seeds": split_seeds.copy(), "config": asdict(config)})
        assert counts == {s: 2 if s == "pilot" else 0 for s in data.SPLITS}
        assert split_seeds == dict(data.DEFAULT_SPLIT_SEEDS, pilot=2026110201)
        assert asdict(config) == dict(horizon=2, candidates=3, anchor_clearance=2., max_anchor_attempts=256)
        assert simulator.source == {"TEST_ONLY": "synthetic generator surrogate, no native simulation"}
        result = {s: [] for s in data.SPLITS}
        for index in range(2):
            # Test data explicitly retains its synthetic runtime provenance.
            root = fixture_root("pilot", index, generation_config, split_seed=split_seeds["pilot"])
            result["pilot"].append(data.save_root(output / "pilot" / f"{root.metadata['root_id']}.npz", root))
        return result

    generation_config = config
    monkeypatch.setattr(data, "generate_splits", generate)


def dummy_simulator():
    return SimpleNamespace(source={"TEST_ONLY": "synthetic generator surrogate, no native simulation"})


def test_shared_config_schema_exact_fixed_budget_and_inert_fit_settings():
    config, spec = r.checked_settings(ROOT / "configs/replication_20261102.json")
    assert config == read_config(ROOT / "configs/replication_20261102.json")
    assert config["counts"] == dict(editor=0, scorer=128, validation=64, pilot=128, confirmation=0)
    assert config["horizon"] == config["candidates"] == 8
    assert config["horizons"] == [1, 4, 8]
    assert config["bootstrap_resamples"] == spec["bootstrap_resamples"] == 10000
    assert config["seed"] == spec["seed"] == 20261102
    assert spec["primary_horizon"] == 8 and spec["encode_batch_size"] == 32
    assert spec["confirmatory"] is False
    assert "not run" in spec["recorded_windows"] and "dynamics only" in spec["replication_scope"]
    prior = read_config(ROOT / "configs/pilot.json")
    for key in set(config) - {"name", "counts", "seed", "bootstrap_resamples"}:
        assert config[key] == prior[key]


@pytest.mark.parametrize("field,value", [("seed", 3), ("bootstrap_resamples", 5000), ("horizon", 9),
                                         ("candidates", 9), ("batch_size", 32), ("horizons", [8]),
                                         ("steps", 0), ("name", "changed"), ("seed", True)])
def test_research_config_cannot_be_tuned(tmp_path, field, value):
    config = json.loads((ROOT / "configs/replication_20261102.json").read_text())
    config[field] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        r.checked_settings(path)


@pytest.mark.parametrize("split", data.SPLITS)
def test_every_split_count_is_fixed(tmp_path, split):
    config = json.loads((ROOT / "configs/replication_20261102.json").read_text())
    config["counts"][split] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exact fixed"):
        r.checked_settings(path)


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'])
def test_json_duplicate_keys_and_nonfinite_constants_are_rejected(tmp_path, text):
    path = tmp_path / "input.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        r.read_json(path)


def test_source_pins_capture_executed_source_and_all_protocol_inputs():
    receipts = r.source_receipt(ROOT / "configs/replication_20261102.json")
    assert len(r.SOURCE_PINS) == 16
    for name, value in r.SOURCE_PINS.items():
        assert receipts[f"src/wm_audit/{name}"]["sha256"] == value
    assert {"scripts/run_replication.py", "REPLICATION_PROTOCOL_20261102.md", "configs/replication_20261102.json",
            "configs/followup_20261101.json", "configs/pilot.json", "pyproject.toml"} <= receipts.keys()
    r.verify_hashes({v["path"]: v["sha256"] for v in receipts.values()})


def test_changed_core_source_rejected_before_publication(monkeypatch):
    changed = dict(r.SOURCE_PINS, **{"data.py": "0" * 64})
    monkeypatch.setattr(r, "code_hashes", lambda: changed)
    with pytest.raises(ValueError, match="Core source differs"):
        r.source_receipt(ROOT / "configs/replication_20261102.json")


def test_new_128_roots_independent_derivation_and_full_historical_exclusion():
    fresh, reserved = r.planned_roots(), r.reserved_roots()
    assert len(fresh) == 128 and len(reserved) == 3385
    assert r.RESERVED_COUNTS == dict(editor=2000, scorer=600, validation=256, pilot=128, confirmation=400)
    assert r.PILOT_SPLIT_SEED == 2026110201
    for index, row in enumerate(fresh):
        encoded = json.dumps([2026110201, "wm_audit.pusht.paired.v1", "pilot", index], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")
        encoded_env = json.dumps([seed, "environment"], separators=(",", ":")).encode()
        assert row == {"root_id": f"pilot-{index:06d}-{seed:032x}", "root_index": index, "root_seed": seed,
                       "split_seed": 2026110201, "environment_seed": int.from_bytes(hashlib.sha256(encoded_env).digest()[:16], "big")}
    r.require_disjoint(fresh, reserved)
    for key in ("root_id", "root_seed", "environment_seed"):
        assert not {x[key] for x in fresh} & {x[key] for x in reserved}


@pytest.mark.parametrize("key", ["root_id", "root_seed", "environment_seed"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_no_root_or_environment_overlap_can_be_hidden_by_new_filename(key, duplicate):
    fresh, excluded = r.planned_roots(), r.reserved_roots()
    if duplicate:
        fresh[1][key] = fresh[0][key]
    else:
        excluded[-1][key] = fresh[97][key]
    with pytest.raises(ValueError, match="separation"):
        r.require_disjoint(fresh, excluded)


def test_split_seed_cannot_reuse_an_old_namespace(monkeypatch):
    monkeypatch.setattr(r, "PILOT_SPLIT_SEED", data.DEFAULT_SPLIT_SEEDS["scorer"])
    with pytest.raises(ValueError, match="split seed overlaps"):
        r.require_disjoint(r.planned_roots(), [])


def test_complete_original_roster_including_editor_is_validated(original_fixture):
    original = original_fixture
    config = original["manifest"]["config"]
    paths, hashes = r.manifest_paths(original["path"], original["manifest"], config)
    identities = r.scan_archives(paths, hashes, config, split_seeds=data.DEFAULT_SPLIT_SEEDS)
    assert identities == original["identities"]
    assert set(paths) == set(data.SPLITS) and len(paths["editor"]) == 1
    assert sum(map(len, paths.values())) == 6


def test_original_manifest_authentication_cannot_be_replaced_by_similar_config(original_fixture):
    with pytest.raises(ValueError, match="authenticated original"):
        r.authenticate_original(original_fixture["path"])


def test_original_authenticator_complete_fixture_under_explicit_test_pin(original_fixture, monkeypatch):
    original = original_fixture
    monkeypatch.setattr(r, "MANIFEST_SHA256", digest(Path(original["path"]) / "manifest-development.json"))
    monkeypatch.setattr(r, "ORIGINAL_COUNTS", original["manifest"]["config"]["counts"])
    monkeypatch.setattr(r, "read_config", lambda path: original["manifest"]["config"])
    assert r.authenticate_original(original["path"]) == original


@pytest.mark.parametrize("bad", ["missing_split", "count", "config", "traversal", "nested", "absolute", "hash"])
def test_manifest_contracts_reject_partial_or_escaping_rosters(original_fixture, bad):
    manifest = deepcopy(original_fixture["manifest"])
    config = original_fixture["manifest"]["config"]
    if bad == "missing_split":
        del manifest["files"]["confirmation"]
    elif bad == "config":
        manifest["config"]["seed"] += 1
    elif bad == "count":
        manifest["files"]["editor"] = {}
    else:
        old, value = next(iter(manifest["files"]["editor"].items()))
        if bad == "hash":
            manifest["files"]["editor"][old] = "z" * 64
        else:
            name = {"traversal": "editor/../victim.npz", "nested": "editor/extra/root.npz", "absolute": "/editor/root.npz"}[bad]
            manifest["files"]["editor"] = {name: value}
    with pytest.raises(ValueError):
        r.manifest_paths(original_fixture["path"], manifest, config)


def test_extra_or_nested_archives_fail_instead_of_being_ignored(tmp_path):
    original = write_fixture(tmp_path / "data", tiny_config())
    extra = Path(original["path"]) / "other/unlisted.npz"
    extra.parent.mkdir()
    extra.write_bytes(b"not a root")
    with pytest.raises(ValueError, match="extra archives"):
        r.manifest_paths(original["path"], original["manifest"], original["manifest"]["config"])


@pytest.mark.parametrize("bad", ["hash", "filename", "supplied", "prefix", "split", "sparse"])
def test_full_archive_scan_rejects_changed_distribution_or_identity(tmp_path, bad):
    config = tiny_config()
    root = fixture_root("pilot", 0, config)
    if bad == "supplied":
        root.metadata["actions_source"] = "supplied_before_rollout"
    elif bad == "prefix":
        root.actions[:, :2] = .1
        root.metadata["actions_source"] = "supplied_before_rollout"
        root.metadata["audit"]["actions_sha256"] = array_digest(root.actions)
    elif bad == "sparse":
        root = fixture_root("pilot", 1, config)
    split = "scorer" if bad == "split" else "pilot"
    path = tmp_path / split / (("wrong" if bad == "filename" else root.metadata["root_id"]) + ".npz")
    data.save_root(path, root)
    hashes = {str(path): "0" * 64 if bad == "hash" else digest(path)}
    with pytest.raises(ValueError):
        r.scan_archives({split: [path]}, hashes, config, split_seeds=data.DEFAULT_SPLIT_SEEDS)


def test_additional_historical_rosters_validate_all_npzs_and_capture_real_hashes(original_fixture):
    rows, hashes = r.additional_exclusions([original_fixture["path"]])
    expected = [item for entries in original_fixture["identities"].values() for item in entries]
    assert {x["root_id"] for x in rows} == {x["root_id"] for x in expected}
    assert len(rows) == len(hashes) == 6
    r.verify_hashes(hashes)
    r.require_disjoint(r.planned_roots(), rows)


def test_empty_additional_roster_is_not_silently_accepted(tmp_path):
    with pytest.raises(ValueError, match="must contain"):
        r.additional_exclusions([tmp_path])


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_exclusive_copy_never_overwrites_or_follows_symlinks(tmp_path, kind):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original provenance bytes")
    if kind == "file":
        target.write_bytes(b"preserve this")
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises((FileExistsError, IsADirectoryError)):
        r.copy_exclusive(source, target, digest(source))
    assert source.read_bytes() == b"original provenance bytes"
    if kind == "file":
        assert target.read_bytes() == b"preserve this"
    if kind == "dangling_symlink":
        assert target.is_symlink() and not (tmp_path / "absent").exists()


def test_copied_bytes_are_independent_not_mutable_hardlinks(tmp_path):
    source, target = tmp_path / "source", tmp_path / "nested/target"
    source.write_bytes(b"original")
    before = digest(source)
    r.copy_exclusive(source, target, before)
    assert not source.samefile(target) and digest(target) == before
    target.write_bytes(b"later corruption")
    assert source.read_bytes() == b"original"
    with pytest.raises(ValueError, match="changed"):
        r.verify_hashes({str(target): before})


def test_pilot_only_generation_reuses_exact_original_scorer_validation(original_fixture, tmp_path, monkeypatch):
    config, _ = small_settings(monkeypatch, original_fixture)
    calls = []
    install_synthetic_generator(monkeypatch, config, calls)
    out = tmp_path / "new-data"
    manifest, hashes, identities = r.generate_dataset(out, original_fixture, config, dummy_simulator(), "f" * 64)
    assert len(calls) == 1
    assert len(list(out.rglob("*.npz"))) == 5
    assert not (out / "editor").exists() and not (out / "confirmation").exists()
    assert identities["pilot"] == r.planned_roots()
    assert manifest["config"] == config
    provenance = manifest["replication_provenance"]
    assert provenance["generated_counts"] == dict(editor=0, scorer=0, validation=0, pilot=2, confirmation=0)
    assert provenance["original_manifest_sha256"] == r.MANIFEST_SHA256
    assert provenance["plan_sha256"] == "f" * 64
    for split in ("scorer", "validation"):
        assert identities[split] == original_fixture["identities"][split]
        assert manifest["files"][split] == original_fixture["manifest"]["files"][split]
        for relative, expected in manifest["files"][split].items():
            old, new = Path(original_fixture["path"]) / relative, out / relative
            assert old.read_bytes() == new.read_bytes() and not old.samefile(new)
            assert data.load_root(old).metadata == data.load_root(new).metadata
    assert not {p["root_id"] for p in identities["pilot"]} & {
        p["root_id"] for rows in original_fixture["identities"].values() for p in rows}
    r.verify_hashes(original_fixture["input_sha256"] | hashes)
    with pytest.raises(FileExistsError):
        r.generate_dataset(out, original_fixture, config, dummy_simulator(), "f" * 64)
    assert len(calls) == 1


def test_generation_failure_does_not_replace_roots_or_publish_completion(original_fixture, tmp_path, monkeypatch):
    config, _ = small_settings(monkeypatch, original_fixture)
    calls = []

    def broken(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic technical failure; no retry")

    monkeypatch.setattr(data, "generate_splits", broken)
    out = tmp_path / "failed-data"
    with pytest.raises(RuntimeError, match="technical failure"):
        r.generate_dataset(out, original_fixture, config, dummy_simulator(), "f" * 64)
    assert len(calls) == 1 and out.is_dir()
    assert not (out / "manifest-development.json").exists()
    assert len(list((out / "scorer").glob("*.npz"))) == 2
    r.verify_hashes(original_fixture["input_sha256"])


def synthetic_records(name, root_ids, *, horizon=8, candidates=8, zero_energy=False):
    """Invented finite candidate banks, never checkpoint inference or simulation."""
    records = []
    for index, root_id in enumerate(root_ids):
        for branch in (0, 1):
            # Retain a completely zero-effect/zero-coverage root.
            magnitude = 0. if zero_energy or index == 0 else (index % 11 + 1) / 11 + branch / 10
            a = torch.arange(candidates, dtype=torch.float64)[:, None, None] * magnitude
            actual = a * torch.arange(horizon + 3, dtype=torch.float64)[None, :, None]
            free = .75 * actual[:, 3:] + .1
            outputs = ({"free_none": free, "actual_none": actual[:, 3:] + .02} if name == "lewm" else
                       {"free_recurrent": free, "free_reset": free + 1e-5,
                        "actual_recurrent": actual[:, 3:] + .02, "actual_reset": actual[:, 3:] + .02001})
            endpoint = np.arange(candidates, dtype=float)
            forecast = endpoint[::-1].copy()
            if name == "lewm":
                endpoint, forecast = forecast, endpoint
            costs = np.ones(candidates) if index == 0 else np.linspace(.8, .2, candidates)
            for step in range(1, horizon + 1):
                errors = f.feedback_errors(outputs, actual, step)
                effects, matrix = f.action_effects(free[:, step - 1], actual[:, step + 2])
                ranking, chosen = f.ranking_metrics(endpoint, forecast, costs)
                values = {**errors, **effects}
                metrics = {k: float(v.mean()) for k, v in values.items()}
                metrics.update(ranking, block_effect_units=magnitude, pusher_effect_units=magnitude,
                               angle_effect_degrees=magnitude, rgb_effect_mse=magnitude)
                row = {"cluster": root_id, "branch": branch, "horizon": step, "metrics": metrics,
                       "candidates": {k: v.tolist() for k, v in values.items()},
                       "prediction_to_actual_mse": matrix.tolist(), "chosen": chosen,
                       "endpoint_scores": endpoint.tolist(), "forecast_scores": forecast.tolist(), "true_costs": costs.tolist()}
                for appearance in APPEARANCES:
                    records.append({**deepcopy(row), "appearance": appearance})
    return records


@pytest.fixture(scope="module")
def fixed_statistics():
    ids = [row["root_id"] for row in r.planned_roots()]
    reports = {name: {"records": synthetic_records(name, ids)} for name in r.MODELS}
    indices = r.bootstrap_indices(ids)
    result = r.analyze_reports(reports, ids, indices)
    return ids, reports, indices, result


def test_exact_10000_shared_resamples_and_all_five_bonferroni_intervals(fixed_statistics):
    ids, reports, indices, result = fixed_statistics
    expected_indices = np.random.default_rng(20261102).integers(128, size=(10000, 128), dtype=np.int64)
    np.testing.assert_array_equal(indices, expected_indices)
    family = result["primary_family"]
    assert family["size"] == 5 and family["fwer_alpha"] == .05
    assert family["individual_confidence"] == .99 and family["quantiles"] == [.005, .995]
    assert family["shared_resamples_across_models_and_metrics"] is True
    assert [(row["model"], row["metric"]) for row in family["claims"]] == list(r.PRIMARY)
    for claim in family["claims"]:
        rows = [row for row in reports[claim["model"]]["records"]
                if row["appearance"] == "original" and row["horizon"] == 8]
        values = np.asarray([np.mean([row["metrics"][claim["metric"]] for row in rows if row["cluster"] == root]) for root in ids])
        independent = values[expected_indices].mean(1)
        # Row-major versus vectorized float64 reductions can differ by a few ULPs.
        assert claim["estimate"] == pytest.approx(values.mean(), rel=1e-14, abs=1e-13)
        assert claim["ci99"] == pytest.approx(np.quantile(independent, [.005, .995], method="linear"), rel=1e-14, abs=1e-13)
        assert claim["clusters"] == 128
    assert all(claim["decision"] == "supports_prespecified_negative_direction" for claim in family["claims"])
    assert result["decisions"]["all_five_supported"] is True
    assert result["decisions"]["action_fidelity_and_specificity_by_model"] == {"lewm": True, "dinowm": True}


def test_every_secondary_slice_metric_gain_and_zero_coverage_root_is_retained(fixed_statistics):
    ids, reports, indices, result = fixed_statistics
    for name in r.MODELS:
        entries = result["exploratory_pointwise95"][name]
        assert len(entries) == 24
        assert {(row["appearance"], row["horizon"]) for row in entries} == {(a, h) for a in APPEARANCES for h in range(1, 9)}
        assert len(result["root_means"][name]) == 24
        primary_slice = next(row for row in entries if row["appearance"] == "original" and row["horizon"] == 8)
        root_slice = next(row for row in result["root_means"][name] if row["appearance"] == "original" and row["horizon"] == 8)
        assert set(primary_slice["metrics"]) == set(root_slice["metrics"]) | {"aggregate_effect_gain"}
        assert all(len(vector) == 128 for vector in root_slice["metrics"].values())
        assert root_slice["metrics"]["true_effect_energy"][0] == 0
        assert root_slice["metrics"]["no_positive_coverage"][0] == 1
        assert root_slice["metrics"]["forecast_regret"][0] == 0
        dot, energy = [np.asarray(root_slice["metrics"][key]) for key in ("effect_dot", "true_effect_energy")]
        gain = primary_slice["metrics"]["aggregate_effect_gain"]
        assert gain["estimate"] == pytest.approx(dot.mean() / energy.mean())
        assert gain["ci95"] == pytest.approx(np.quantile(dot[indices].mean(1) / energy[indices].mean(1), [.025, .975]))
        assert gain["undefined_resamples"] == 0
        assert all(v["clusters"] == 128 for v in primary_slice["metrics"].values())
        for metric in ("actual_none_minus_baseline",) if name == "lewm" else ("visual_aux_interaction", "free_reset_minus_recurrent"):
            assert metric in primary_slice["metrics"]
    assert ("lewm", "forecast_minus_endpoint_regret") not in r.PRIMARY


def test_root_aggregation_is_order_invariant_and_equal_branch_weight():
    ids = ["a", "b"]
    records = synthetic_records("lewm", ids, horizon=1, candidates=2)
    for row in records:
        row["metrics"]["block_effect_units"] = {("a", 0): 2., ("a", 1): 6., ("b", 0): 100., ("b", 1): 200.}[row["cluster"], row["branch"]]
    result = r.aggregate_records(records, "lewm", ids, horizon=1, candidates=2)
    reverse = r.aggregate_records(list(reversed(records)), "lewm", ids, horizon=1, candidates=2)
    for group in result:
        for metric in result[group]:
            np.testing.assert_array_equal(result[group][metric], reverse[group][metric])
        np.testing.assert_array_equal(result[group]["block_effect_units"], [4., 150.])


@pytest.mark.parametrize("bad", ["short", "duplicate", "root", "appearance", "branch_bool", "horizon_float", "nan",
                                  "bool_metric", "missing_metric", "extra_metric", "missing_candidate", "candidate_count",
                                  "candidate_bool", "candidate_mean", "negative_matrix", "wrong_matrix", "wrong_effect",
                                  "bad_cost", "wrong_ranking", "chosen_bool", "tie_rule"])
def test_incomplete_or_corrupt_records_never_trigger_filtering(bad):
    records = synthetic_records("dinowm", ["a", "b"], horizon=1, candidates=2)
    row = records[-1]
    if bad == "short":
        records.pop()
    elif bad == "duplicate":
        records[-1] = deepcopy(records[0])
    elif bad in ("root", "appearance", "branch_bool", "horizon_float"):
        key, value = {"root": ("cluster", "outside"), "appearance": ("appearance", "unknown"),
                      "branch_bool": ("branch", True), "horizon_float": ("horizon", 1.)}[bad]
        row[key] = value
    elif bad in ("nan", "bool_metric"):
        row["metrics"]["block_effect_units"] = float("nan") if bad == "nan" else True
    elif bad == "missing_metric":
        del row["metrics"]["local_persistence"]
    elif bad == "extra_metric":
        row["metrics"]["posthoc"] = 1.
    elif bad == "missing_candidate":
        del row["candidates"]["matched_minus_all_wrong"]
    elif bad == "candidate_count":
        row["candidates"]["effect_error"].append(0.)
    elif bad == "candidate_bool":
        row["candidates"]["free_recurrent"][0] = True
    elif bad == "candidate_mean":
        row["metrics"]["effect_error"] += 1.
    elif bad == "negative_matrix":
        row["prediction_to_actual_mse"][0][0] = -1.
    elif bad == "wrong_matrix":
        row["prediction_to_actual_mse"][0][1] += 3.
    elif bad == "wrong_effect":
        row["candidates"]["effect_error_minus_zero"][0] += 1.
        row["metrics"]["effect_error_minus_zero"] += 1.
    elif bad == "bad_cost":
        row["true_costs"][0] = 1.1
    elif bad == "wrong_ranking":
        row["metrics"]["forecast_minus_endpoint_regret"] += .1
    elif bad == "chosen_bool":
        row["chosen"]["forecast"] = True
    else:
        row["forecast_scores"] = [0., 0.]
        row["chosen"]["forecast"] = 1
    with pytest.raises(ValueError):
        r.aggregate_records(records, "dinowm", ["a", "b"], horizon=1, candidates=2)


@pytest.mark.parametrize("ci,expected", [([-2., -1.], "supports_prespecified_negative_direction"),
                                        ([1., 2.], "opposite_direction"), ([-1., 1.], "inconclusive"),
                                        ([-1., 0.], "inconclusive"), ([0., 1.], "inconclusive"),
                                        ([0., 0.], "inconclusive")])
def test_strict_directional_decisions_do_not_call_touching_zero_a_pass(ci, expected):
    assert r.decision(ci) == expected


@pytest.mark.parametrize("ids,seed,resamples", [([], 0, 10), (["b", "a"], 0, 10), (["a", "a"], 0, 10),
                                               (["a"], True, 10), (["a"], 0, True), (["a"], 0, 0), ([1], 0, 10)])
def test_bootstrap_roster_and_integer_contracts(ids, seed, resamples):
    with pytest.raises(ValueError):
        r.bootstrap_indices(ids, seed=seed, resamples=resamples)


@pytest.mark.parametrize("bad", ["missing_model", "partial_roots", "wrong_seed", "fewer_draws", "float_indices"])
def test_primary_analysis_requires_both_models_full_roots_and_exact_resamples(fixed_statistics, bad):
    ids, reports, indices, _ = fixed_statistics
    reports, ids = dict(reports), ids.copy()
    if bad == "missing_model":
        reports.pop("lewm")
    elif bad == "partial_roots":
        ids.pop()
    elif bad == "wrong_seed":
        indices = r.bootstrap_indices(ids, seed=1)
    elif bad == "fewer_draws":
        indices = indices[:9999]
    else:
        indices = indices.astype(float)
    with pytest.raises(ValueError):
        r.analyze_reports(reports, ids, indices)


@pytest.mark.parametrize("zero_energy", [True, False])
def test_gain_keeps_zero_energy_roots_and_reports_undefined_resamples(monkeypatch, zero_energy):
    monkeypatch.setattr(r, "ROOTS", 2)
    monkeypatch.setattr(r, "HORIZON", 1)
    monkeypatch.setattr(r, "CANDIDATES", 2)
    ids = ["a", "b"]
    indices = r.bootstrap_indices(ids)
    reports = {name: {"records": synthetic_records(name, ids, horizon=1, candidates=2, zero_energy=zero_energy)} for name in r.MODELS}
    result = r.analyze_reports(reports, ids, indices)
    for name in r.MODELS:
        gain = result["exploratory_pointwise95"][name][0]["metrics"]["aggregate_effect_gain"]
        assert gain["estimate"] is None if zero_energy else gain["estimate"] == pytest.approx(.75)
        assert gain["ci95"] is None
        assert gain["undefined_resamples"] == (10000 if zero_energy else int((indices == 0).all(1).sum()))
        assert gain["clusters"] == 2


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_existing_output_fails_before_reading_sources_or_loading_models(tmp_path, monkeypatch, kind):
    output = tmp_path / "out"
    if kind == "file":
        output.write_text("preserve")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent")
    monkeypatch.setattr(r, "checked_settings", forbidden)
    with pytest.raises(FileExistsError, match="Output exists"):
        r.execute(SimpleNamespace(output=output))


def test_plan_only_does_not_create_outputs_generate_or_load(original_fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(r, "authenticate_original", lambda _: original_fixture)
    monkeypatch.setattr(r, "hosted_preflight", forbidden)
    args = SimpleNamespace(config=ROOT / "configs/replication_20261102.json", original_data=Path(original_fixture["path"]),
                           output=tmp_path / "absent-output", cache=tmp_path / "absent-cache", exclude_data=[], execute=False)
    result = r.execute(args)
    assert result["executed"] is False and not args.output.exists()
    assert len(result["plan"]["pilot_roots"]) == 128 and len(result["plan"]["primary_family"]) == 5
    assert result["plan"]["resampling"]["resamples"] == 10000


@pytest.mark.parametrize("options", [["--device", "cpu"], ["--device", "mps"], ["--roots", "8"],
                                     ["--seed", "1"], ["--resume"], ["--model", "lewm"], ["--stage", "probes"]])
def test_cli_has_no_local_research_tuning_partial_model_or_resume_escape(tmp_path, monkeypatch, capsys, options):
    monkeypatch.setattr(r, "execute", forbidden)
    with pytest.raises(SystemExit) as exc:
        r.main(["--original-data", "absent", "--cache", "absent", "--output", str(tmp_path / "out"), *options])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_research_preflight_never_falls_back_to_cpu(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(r.sys, "platform", platform)
    monkeypatch.setattr(r, "verify_hashes", forbidden)
    with pytest.raises(RuntimeError, match="hosted Linux CUDA"):
        r.hosted_preflight(tmp_path, trust_checkpoint=True, threads=2)


def test_research_preflight_requires_literal_checkpoint_consent_before_io(tmp_path, monkeypatch):
    monkeypatch.setattr(r.sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(r, "verify_hashes", forbidden)
    with pytest.raises(PermissionError, match="pinned DINO"):
        r.hosted_preflight(tmp_path, trust_checkpoint=1, threads=2)


def setup_synthetic_execution(monkeypatch, tmp_path, original):
    config, spec = small_settings(monkeypatch, original)
    config_path = tmp_path / "test-config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(r, "checked_settings", lambda _: (config, spec))
    monkeypatch.setattr(r, "authenticate_original", lambda _: original)
    # The synthetic manifest is not the released 576-root manifest; production
    # keeps its immutable real pin, while this CPU-only fixture pins its own bytes.
    monkeypatch.setattr(r, "MANIFEST_SHA256", digest(Path(original["path"]) / "manifest-development.json"))
    monkeypatch.setattr(r, "hosted_preflight", lambda *a, **k: {})
    monkeypatch.setattr(r, "PushTSimulator", lambda _: dummy_simulator())
    generation_calls, model_calls = [], []
    install_synthetic_generator(monkeypatch, config, generation_calls)

    def load(name, cache, device, **kwargs):
        assert device == "cuda"  # Production dispatch; actual TEST adapter remains CPU.
        assert kwargs == {"trust_checkpoint": True}
        model_calls.append(name)
        model = _payload_model(name)
        assert model.device.type == "cpu"
        return model

    monkeypatch.setattr(models, "load_model", load)
    args = SimpleNamespace(config=config_path, original_data=Path(original["path"]), cache=tmp_path / "unused-cache",
                           output=tmp_path / "SYNTHETIC-output", exclude_data=[], execute=True,
                           trust_checkpoint=True, threads=2, device="cuda")
    return args, generation_calls, model_calls


def test_full_pipeline_cpu_test_doubles_data_then_both_existing_qualified_dynamics(original_fixture, tmp_path, monkeypatch):
    args, generation_calls, model_calls = setup_synthetic_execution(monkeypatch, tmp_path, original_fixture)
    plan_seen, original_generate = [], data.generate_splits

    def assert_preoutcome_publication(*a, **kw):
        assert (args.output / "plan.json").is_file()
        assert (args.output / "bootstrap-indices.npy").is_file()
        assert not (args.output / "dynamics").exists()
        plan_seen.append(True)
        return original_generate(*a, **kw)

    monkeypatch.setattr(data, "generate_splits", assert_preoutcome_publication)
    result = r.execute(args)
    assert result["executed"] is True and plan_seen == [True]
    assert len(generation_calls) == 1 and model_calls == ["lewm", "dinowm"]
    analysis = json.loads((args.output / "replication-results.json").read_text())
    assert analysis["primary_family"]["size"] == 5
    assert len(analysis["primary_family"]["claims"]) == 5
    assert len(analysis["root_ids"]) == 2  # Explicitly reduced TEST ONLY internal budget.
    for name in r.MODELS:
        report_path = args.output / "dynamics" / name / "report.json"
        report = json.loads(report_path.read_text())
        assert report["kind"] == "followup_dynamics" and report["confirmatory"] is False
        assert len(report["records"]) == 2 * 3 * 2 * 2
        assert report["protocol"]["readout_used"] is False
        assert analysis["dynamics_report_sha256"][name] == digest(report_path)
    indices = np.load(args.output / "bootstrap-indices.npy", allow_pickle=False)
    np.testing.assert_array_equal(indices, r.bootstrap_indices(analysis["root_ids"]))
    inventory = json.loads((args.output / "complete.json").read_text())["files"]
    actual = {p.relative_to(args.output).as_posix() for p in args.output.rglob("*") if p.is_file()} - {"complete.json"}
    assert set(inventory) == actual
    for relative, value in inventory.items():
        path = args.output / relative
        assert value == {"sha256": digest(path), "bytes": path.stat().st_size}
    assert not list(args.output.rglob("*.pt"))  # No fitted/selected model artifact is created.
    r.verify_hashes(original_fixture["input_sha256"])
    with pytest.raises(FileExistsError):
        r.execute(args)
    assert model_calls == ["lewm", "dinowm"]


@pytest.mark.parametrize("changed", ["original", "generated", "source_snapshot"])
def test_midrun_input_change_blocks_joint_results_and_completion(tmp_path, monkeypatch, changed):
    original = write_fixture(tmp_path / "old", tiny_config())
    args, _, model_calls = setup_synthetic_execution(monkeypatch, tmp_path, original)
    original_dynamics = f.dynamics_followup

    def mutate(*a, **kw):
        result = original_dynamics(*a, **kw)
        path = ({"original": next((Path(original["path"]) / "validation").glob("*.npz")),
                 "generated": next((args.output / "data/pilot").glob("*.npz")),
                 "source_snapshot": args.output / "sources/scripts/run_replication.py"})[changed]
        with path.open("ab") as stream:
            stream.write(b"SYNTHETIC mutation after first model")
        return result

    monkeypatch.setattr(f, "dynamics_followup", mutate)
    with pytest.raises(ValueError, match="changed"):
        r.execute(args)
    assert model_calls == ["lewm"]
    assert not (args.output / "replication-results.json").exists()
    assert not (args.output / "complete.json").exists()


@pytest.mark.parametrize("field", ["kind", "config", "spec", "code_sha256", "input_sha256", "model", "parameter_buffer_sha256", "protocol"])
def test_report_must_bind_actual_data_models_sources_and_spec(field):
    config, spec, metadata = {"config": "fixture"}, {"spec": "fixture"}, {"model": "fixture"}
    hashes = {"/ordinary/data/manifest-development.json": "a" * 64, "/ordinary/data/pilot/p.npz": "b" * 64,
              "/ordinary/data/scorer/s.npz": "c" * 64}
    report = {"schema": "wm_audit.diagnostic.v1", "kind": "followup_dynamics", "config": config, "spec": spec,
              "code_sha256": r.SOURCE_PINS, "input_sha256": {k: v for k, v in hashes.items() if "/scorer/" not in k},
              "model": metadata, "parameter_buffer_sha256": "d" * 64,
              "protocol": {"world_model_trained": False, "readout_used": False}}
    r.validate_report(report, "lewm", config, spec, hashes, metadata, "d" * 64)
    report[field] = {} if field in ("protocol", "code_sha256") else "wrong"
    with pytest.raises(ValueError, match="provenance"):
        r.validate_report(report, "lewm", config, spec, hashes, metadata, "d" * 64)