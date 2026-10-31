"""Local provenance contracts; opaque fixture bytes are NOT trained models/data.

No checkpoint, simulator, network, GPU, or research experiment is executed.
Source-hash tests use a temporary source tree, never production source edits.
Regression assertions intentionally expose missing fail-closed checks rather
than accepting the current implementation's unsafe behavior or using xfail.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import urllib.request
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from wm_audit import provenance
from wm_audit.data import SPLITS


ROOT = Path(__file__).resolve().parents[1]


def _forbidden(*args, **kwargs):
    raise AssertionError("Provenance unit tests must not use the network or a GPU.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", _forbidden)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs" / "smoke.json").read_text())


def _dump(path, value):
    """Test-only replacement writer, also used to inject malformed/tampered JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(config):
    loss = {"inverse": 2.0, "ridge": 0.25, "mlp": 3.0, "rollout": 4.0}
    return [{"root_id": f"validation-{index}", "split": "validation",
             "appearance": "original", "method": method,
             "history_fidelity": loss[method] + index, "specificity_loss": 0.5}
            for index in range(config["counts"]["validation"])
            for method in config["editors"]]


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    source = tmp_path / "isolated-source"
    source.mkdir()
    for name in ("provenance.py", "cli.py", "models.py"):
        (source / name).write_text(f"# Synthetic hash fixture for {name}; never executed.\n")
    monkeypatch.setattr(provenance, "__file__", str(source / "provenance.py"))
    return source


@pytest.fixture
def prepared_run(tmp_path, config, source_tree, monkeypatch):
    run, data = tmp_path / "run", tmp_path / "data"
    (run / "editors").mkdir(parents=True)
    for method in config["editors"]:
        (run / "editors" / f"{method}.pt").write_bytes(f"opaque test editor: {method}".encode())
    (run / "scorer.pt").write_bytes(b"opaque test scorer; NOT a torch checkpoint")
    _dump(run / "training.json", {"config": config, "synthetic_fixture": True})
    records = _records(config)
    _dump(run / "validation.json", {"config": config, "primary_method": "ridge", "records": records})
    for split in SPLITS[:-1]:
        (data / split).mkdir(parents=True)
        for index in range(config["counts"][split]):
            (data / split / f"{index}.npz").write_bytes(f"hash-only fixture: {split}/{index}".encode())
    monkeypatch.setattr(provenance, "runtime", lambda: {"synthetic_fixture": "not an experiment"})
    return SimpleNamespace(config=config, run=run, data=data, source=source_tree, records=records)


@pytest.fixture
def locked_run(prepared_run):
    prepared_run.lock = provenance.create_lock(prepared_run.config, prepared_run.run, prepared_run.data)
    return prepared_run


def test_smoke_config_round_trips_without_modification(config, tmp_path):
    path = _dump(tmp_path / "config.json", config)
    before = path.read_bytes()
    assert provenance.read_config(path) == config
    assert provenance.read_config(ROOT / "configs" / "smoke.json") == config
    assert path.read_bytes() == before


@pytest.mark.parametrize("missing", ["name", "counts", "horizons", "regret_margin"])
def test_config_requires_all_keys(config, tmp_path, missing):
    del config[missing]
    with pytest.raises(ValueError, match="keys"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


def test_config_rejects_unknown_keys(config, tmp_path):
    config["typo_instead_of_setting"] = 1
    with pytest.raises(ValueError, match="keys"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("missing", SPLITS)
def test_config_requires_explicit_count_for_every_split(config, tmp_path, missing):
    del config["counts"][missing]
    with pytest.raises(ValueError, match="five split counts"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


def test_config_rejects_unknown_split(config, tmp_path):
    config["counts"]["test"] = 4
    with pytest.raises(ValueError, match="five split counts"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("field", ["horizon", "candidates", "rank", "batch_size", "bootstrap_resamples"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_config_positive_integer_fields_are_strict(config, tmp_path, field, value):
    config[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("field", ["seed", "steps", *SPLITS])
@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_config_nonnegative_integer_fields_are_strict(config, tmp_path, field, value):
    target = config["counts"] if field in SPLITS else config
    target[field] = value
    with pytest.raises(ValueError, match="nonnegative integers"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


def test_zero_steps_seed_and_explicit_empty_splits_are_configurable(config, tmp_path):
    config.update(seed=0, steps=0)
    config["counts"] = dict.fromkeys(SPLITS, 0)
    assert provenance.read_config(_dump(tmp_path / "config.json", config)) == config


@pytest.mark.parametrize("value", ["", None, 1, True, []])
def test_config_name_is_nonempty_text(config, tmp_path, value):
    config["name"] = value
    with pytest.raises(ValueError, match="name"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("value", [[], [1, 1], [0], [3], [True], [1.0], ["1"]])
def test_config_horizons_are_unique_bounded_integer_steps(config, tmp_path, value):
    config["horizons"] = value
    with pytest.raises(ValueError, match="Horizons"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_config_scales_must_be_finite_positive(config, tmp_path, value):
    config["scales"][2] = value
    with pytest.raises(ValueError, match="scales"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("length", [0, 5, 7])
def test_config_requires_exactly_six_scales(config, tmp_path, length):
    config["scales"] = [1] * length
    with pytest.raises(ValueError, match="Six"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("value", [[], ["ridge", "ridge"], ["unsupported"], [None]])
def test_config_editors_are_unique_supported_names(config, tmp_path, value):
    config["editors"] = value
    with pytest.raises(ValueError, match="editor"):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("field", ["native_margin", "readout_margin", "regret_margin"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_config_margins_must_be_finite_positive(config, tmp_path, field, value):
    config[field] = value
    with pytest.raises(ValueError, match=field):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("field,value", [
    (None, None), (None, 42), ("counts", list(SPLITS)), ("horizons", True),
    ("scales", None), ("editors", {"ridge": True}), ("native_margin", "0.05"),
])
def test_malformed_config_shapes_raise_actionable_value_errors(config, tmp_path, field, value):
    """Malformed JSON values must not escape main() as TypeError/AttributeError."""
    if field is None:
        config = value
    else:
        config[field] = value
    with pytest.raises(ValueError):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("field", ["scales", "native_margin", "readout_margin", "regret_margin"])
def test_physical_scales_and_margins_reject_boolean_numbers(config, tmp_path, field):
    if field == "scales":
        config[field][0] = True
    else:
        config[field] = True
    with pytest.raises(ValueError):
        provenance.read_config(_dump(tmp_path / "config.json", config))


@pytest.mark.parametrize("text", ["{", '{"name":', ""])
def test_config_invalid_json_fails(tmp_path, text):
    path = tmp_path / "config.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        provenance.read_config(path)


def test_selection_uses_only_clean_validation_immediate_losses(config):
    records = _records(config)
    for record in records:
        # Future outcomes favor a different method and MUST NOT select it.
        record.update(effect_loss=0 if record["method"] == "inverse" else 1000, regret=123)
    for method in config["editors"]:
        for appearance in ("color_only", "background_only"):
            records.append({"method": method, "appearance": appearance, "split": "confirmation"})
    records.append({"method": "native", "appearance": "original", "split": "pilot"})
    before = deepcopy((config, records))
    selected, scores = provenance.select_primary(config, records)
    assert selected == "ridge"
    assert scores == pytest.approx({"inverse": 4.0, "ridge": 2.25, "mlp": 5.0, "rollout": 6.0})
    assert (config, records) == before


@pytest.mark.parametrize("reverse", [False, True])
def test_selection_ties_are_lexical_not_editor_or_row_order(config, reverse):
    config["editors"] = ["rollout", "mlp", "ridge", "inverse"]
    records = _records(config)
    for record in records:
        record.update(history_fidelity=2.0, specificity_loss=1.0)
    if reverse:
        config["editors"].reverse()
        records.reverse()
    selected, scores = provenance.select_primary(config, records)
    assert selected == "inverse" and set(scores.values()) == {3.0}


@pytest.mark.parametrize("split", ["editor", "scorer", "pilot", "confirmation"])
def test_selection_rejects_any_clean_nonvalidation_row(config, split):
    records = _records(config)
    records.append({**records[0], "split": split})
    with pytest.raises(ValueError, match="clean validation"):
        provenance.select_primary(config, records)


@pytest.mark.parametrize("appearance", [None, "color_only", "background_only"])
def test_selection_requires_clean_rows_for_every_configured_editor(config, appearance):
    records = [r for r in _records(config) if r["method"] != "mlp"]
    if appearance is not None:
        records.append({"method": "mlp", "appearance": appearance, "split": "validation"})
    with pytest.raises(ValueError, match="mlp"):
        provenance.select_primary(config, records)


@pytest.mark.parametrize("field", ["history_fidelity", "specificity_loss"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True])
def test_selection_rejects_nonfinite_negative_or_boolean_clean_metrics(config, field, value):
    records = _records(config)
    records[0][field] = value
    with pytest.raises(ValueError):
        provenance.select_primary(config, records)


@pytest.mark.parametrize("mutation", ["missing-root", "duplicate-root", "different-root"])
def test_selection_requires_the_same_complete_unique_clean_root_roster(config, mutation):
    records = _records(config)
    rows = [r for r in records if r["method"] == "ridge"]
    if mutation == "missing-root":
        records.remove(rows[-1])
    elif mutation == "duplicate-root":
        rows[-1]["root_id"] = rows[0]["root_id"]
    else:
        rows[-1]["root_id"] = "not-in-the-other-editors-roster"
    with pytest.raises(ValueError):
        provenance.select_primary(config, records)


def test_digest_is_sha256_of_exact_bytes_and_missing_files_fail(tmp_path):
    path = tmp_path / "binary"
    path.write_bytes(b"\x00\xff\nopaque fixture")
    assert provenance.digest(path) == _sha(path)
    with pytest.raises(FileNotFoundError):
        provenance.digest(tmp_path / "missing")


def test_json_digest_canonicalizes_mapping_order_but_not_values_or_list_order():
    a = {"b": [1, 2], "a": {"y": 1, "x": 2}}
    b = {"a": {"x": 2, "y": 1}, "b": [1, 2]}
    expected = hashlib.sha256(json.dumps(a, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert provenance.json_digest(a) == provenance.json_digest(b) == expected
    assert provenance.json_digest({**a, "b": [2, 1]}) != expected
    assert provenance.json_digest({**a, "b": [1, 3]}) != expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_json_digest_and_writer_reject_nonfinite_values_without_publishing(tmp_path, value):
    path = tmp_path / "nested" / "result.json"
    with pytest.raises(ValueError):
        provenance.json_digest({"metric": value})
    with pytest.raises(ValueError):
        provenance.write_json(path, {"metric": value})
    assert not path.exists()


def test_json_writer_creates_parents_and_never_overwrites(tmp_path):
    path = tmp_path / "nested" / "result.json"
    value = {"z": [1, 2], "a": "synthetic"}
    provenance.write_json(path, value)
    expected = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert path.read_text() == expected
    with pytest.raises(FileExistsError):
        provenance.write_json(path, {"replacement": True})
    assert path.read_text() == expected


@pytest.mark.parametrize("target_exists", [False, True])
def test_json_writer_never_follows_existing_or_dangling_output_symlinks(tmp_path, target_exists):
    target, link = tmp_path / "target.json", tmp_path / "result.json"
    if target_exists:
        target.write_bytes(b"preserve this target")
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        provenance.write_json(link, {"replacement": True})
    assert link.is_symlink()
    assert target.exists() is target_exists
    if target_exists:
        assert target.read_bytes() == b"preserve this target"


def test_code_hashes_are_relative_sorted_and_cover_package_python_files(source_tree):
    (source_tree / "ignored.txt").write_text("not Python")
    result = provenance.code_hashes()
    assert list(result) == ["cli.py", "models.py", "provenance.py"]
    assert result == {p.name: _sha(p) for p in source_tree.glob("*.py")}


def test_fitted_hashes_cover_training_scorer_and_all_editor_checkpoints(prepared_run):
    p = prepared_run
    (p.run / "unrelated.log").write_text("not a fitted artifact")
    expected = [p.run / "training.json", p.run / "scorer.pt", *sorted((p.run / "editors").glob("*.pt"))]
    assert provenance.fitted_hashes(p.run) == {str(path.relative_to(p.run)): _sha(path) for path in expected}


def test_fitted_hashes_require_at_least_one_editor(prepared_run):
    for path in (prepared_run.run / "editors").glob("*.pt"):
        path.unlink()
    with pytest.raises(ValueError, match="No fitted editors"):
        provenance.fitted_hashes(prepared_run.run)


@pytest.mark.parametrize("name", ["training.json", "scorer.pt"])
def test_fitted_hashes_fail_on_missing_required_artifacts(prepared_run, name):
    (prepared_run.run / name).unlink()
    with pytest.raises(FileNotFoundError):
        provenance.fitted_hashes(prepared_run.run)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_runtime_records_versions_and_utc_without_probing_unavailable_gpu(monkeypatch, cuda_available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    gpu = Mock(return_value="mock GPU, not real qualification") if cuda_available else Mock(side_effect=_forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_name", gpu)
    result = provenance.runtime()
    assert datetime.fromisoformat(result["utc"]).utcoffset() == timedelta(0)
    assert result["python"] and result["platform"]
    assert result["numpy"] == np.__version__ and result["torch"] == str(torch.__version__)
    assert result["cuda_build"] == torch.version.cuda
    if cuda_available:
        gpu.assert_called_once_with(0)
        assert result["gpu"] == "mock GPU, not real qualification"
    else:
        gpu.assert_not_called()
        assert result["gpu"] is None


def test_create_lock_commits_config_code_fits_selection_and_all_development_bytes(prepared_run):
    p = prepared_run
    result = provenance.create_lock(p.config, p.run, p.data)
    assert json.loads((p.run / "protocol-lock.json").read_text()) == result
    assert result["schema"] == 1 and result["config"] == p.config
    assert result["config_hash"] == provenance.json_digest(p.config)
    assert result["code"] == {path.name: _sha(path) for path in p.source.glob("*.py")}
    assert result["fitted"] == provenance.fitted_hashes(p.run)
    assert result["validation_hash"] == _sha(p.run / "validation.json")
    assert result["primary_method"] == "ridge"
    assert result["selection_scores"] == pytest.approx({"inverse": 4, "ridge": 2.25, "mlp": 5, "rollout": 6})
    assert result["development_data"] == {str(path.relative_to(p.data)): _sha(path) for path in p.data.glob("*/*.npz")}
    assert len(result["development_data"]) == sum(p.config["counts"][split] for split in SPLITS[:-1])
    assert "LOCAL" in result["status"] and "NOT EXTERNAL" in result["status"]


def test_create_lock_refuses_confirmation_before_reading_validation(prepared_run):
    p = prepared_run
    (p.data / "confirmation").mkdir()
    (p.data / "confirmation" / "already-observed.npz").write_bytes(b"opaque fixture")
    (p.run / "validation.json").unlink()
    with pytest.raises(ValueError, match="Lock before generating"):
        provenance.create_lock(p.config, p.run, p.data)
    assert not (p.run / "protocol-lock.json").exists()


@pytest.mark.parametrize("count", [0, 1])
def test_create_lock_requires_at_least_two_confirmation_roots(prepared_run, count):
    p = prepared_run
    p.config["counts"]["confirmation"] = count
    _dump(p.run / "training.json", {"config": p.config})
    _dump(p.run / "validation.json", {"config": p.config, "records": p.records})
    with pytest.raises(ValueError, match="confirmation roster"):
        provenance.create_lock(p.config, p.run, p.data)
    assert not (p.run / "protocol-lock.json").exists()


@pytest.mark.parametrize("artifact", ["training.json", "validation.json"])
def test_create_lock_requires_matching_training_and_validation_configs(prepared_run, artifact):
    p = prepared_run
    payload = json.loads((p.run / artifact).read_text())
    payload["config"]["seed"] += 1
    _dump(p.run / artifact, payload)
    with pytest.raises(ValueError, match="Config must match"):
        provenance.create_lock(p.config, p.run, p.data)
    assert not (p.run / "protocol-lock.json").exists()


def test_create_lock_is_immutable(locked_run):
    p = locked_run
    before = (p.run / "protocol-lock.json").read_bytes()
    with pytest.raises(FileExistsError):
        provenance.create_lock(p.config, p.run, p.data)
    assert (p.run / "protocol-lock.json").read_bytes() == before


def test_create_lock_requires_every_configured_editor_checkpoint(prepared_run):
    p = prepared_run
    (p.run / "editors" / "mlp.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        provenance.create_lock(p.config, p.run, p.data)
    assert not (p.run / "protocol-lock.json").exists()


def test_verify_lock_allows_confirmation_to_be_generated_after_the_commitment(locked_run):
    p = locked_run
    assert provenance.verify_lock(p.config, p.run, p.data) == p.lock
    (p.data / "confirmation").mkdir()
    (p.data / "confirmation" / "new.npz").write_bytes(b"post-lock opaque fixture")
    assert provenance.verify_lock(p.config, p.run, p.data) == p.lock


@pytest.mark.parametrize("artifact", ["training.json", "validation.json", "scorer.pt", "editors/ridge.pt"])
def test_verify_lock_rejects_mutated_fitted_artifacts_and_validation(locked_run, artifact):
    p = locked_run
    path = p.run / artifact
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="changed after lock"):
        provenance.verify_lock(p.config, p.run, p.data)


@pytest.mark.parametrize("mutation", ["modified", "added", "deleted"])
def test_verify_lock_rejects_changed_code_without_editing_production(locked_run, mutation):
    p = locked_run
    path = p.source / "cli.py"
    if mutation == "modified":
        path.write_text("# Changed synthetic code\n")
    elif mutation == "added":
        (p.source / "new_module.py").write_text("# New synthetic code\n")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="changed after lock"):
        provenance.verify_lock(p.config, p.run, p.data)


def test_verify_lock_rejects_changed_requested_config(locked_run):
    p = locked_run
    changed = deepcopy(p.config)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="changed after lock"):
        provenance.verify_lock(changed, p.run, p.data)


@pytest.mark.parametrize("mutation", ["modified", "deleted", "renamed"])
def test_verify_lock_rejects_changed_or_missing_development_bytes(locked_run, mutation):
    p = locked_run
    path = p.data / "editor" / "0.npz"
    if mutation == "modified":
        path.write_bytes(b"changed root bytes")
    elif mutation == "deleted":
        path.unlink()
    else:
        path.rename(path.with_name("renamed.npz"))
    with pytest.raises((ValueError, FileNotFoundError)):
        provenance.verify_lock(p.config, p.run, p.data)


@pytest.mark.parametrize("split", SPLITS[:-1])
def test_verify_lock_rejects_added_development_roots(locked_run, split):
    p = locked_run
    (p.data / split / "uncommitted.npz").write_bytes(b"uncommitted root")
    with pytest.raises(ValueError):
        provenance.verify_lock(p.config, p.run, p.data)


@pytest.mark.parametrize("field", ["primary_method", "config", "selection_scores", "schema"])
def test_verify_lock_rejects_internally_inconsistent_lock_metadata(locked_run, field):
    """This tests consistency, not authentication against rewriting every hash."""
    p = locked_run
    path = p.run / "protocol-lock.json"
    payload = json.loads(path.read_text())
    if field == "primary_method":
        payload[field] = "mlp"  # Different configured editor, inconsistent with pinned validation.
    elif field == "config":
        payload[field]["seed"] += 1
    elif field == "selection_scores":
        payload[field]["ridge"] += 100
    else:
        payload[field] = 999
    _dump(path, payload)
    with pytest.raises(ValueError):
        provenance.verify_lock(p.config, p.run, p.data)