"""CPU-only synthetic contracts. NEVER load real checkpoints or run research.

The tiny encoder decodes synthetic RGB markers. Its deliberate batch offsets
test attribution and matching, not the numerics of either released model.
"""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wm_audit import models, pipeline as p
from wm_audit.followup_dynamics import _state_digest
from wm_audit.models import DinoWMAdapter, LeWMAdapter
from wm_audit.provenance import digest
from test_models import SYNTHETIC_STATS, _ToyDinoWM


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("execution_layout_audit", ROOT / "scripts/audit_execution_layout.py")
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)


def forbidden(*args, **kwargs):
    raise AssertionError("Unit tests must not load/download real weights, initialize CUDA or run research.")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def no_research(monkeypatch):
    monkeypatch.setenv("WM_AUDIT_PROGRESS", "0")
    for name in ("load_model", "_load_lewm", "_load_dinowm", "_download"):
        monkeypatch.setattr(models, name, forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)


def settings():
    config = json.loads((ROOT / "configs/pilot.json").read_text())
    spec = json.loads((ROOT / "configs/followup_20261101.json").read_text())
    return config, spec


def pixels(markers):
    markers = np.asarray(markers, dtype=np.uint8)
    return np.broadcast_to(markers[..., None, None, :], (*markers.shape[:-1], 224, 224, 3))


def synthetic_root(index=0):
    context = np.zeros((3, 2, 3, 3), dtype=np.uint8)
    future = np.zeros((3, 2, 8, 8, 3), dtype=np.uint8)
    goals = np.zeros((3, 3), dtype=np.uint8)
    for appearance in range(3):
        context[appearance, ..., 0] = 10 + index
        future[appearance, ..., 0] = 20 + index
        context[appearance, ..., 2] = appearance
        future[appearance, ..., 2] = appearance
        goals[appearance] = [100 + index, 200, appearance]
        for branch in range(2):
            context[appearance, branch, :, 1] = branch * 40 + np.arange(3)
            future[appearance, branch, :, :, 1] = branch * 100 + 8 * np.arange(8)[:, None] + np.arange(8)
    actions = np.linspace(-.75, .75, 800, dtype=np.float32).reshape(8, 10, 10)
    actions[:, :2] = np.tile([.1, -.2], (2, 5))
    actions[0, 2:] = 0
    return SimpleNamespace(context=pixels(context), future=pixels(future), goal_pixels=pixels(goals),
                           actions=actions, metadata={"root_id": f"pilot-{index:06d}-synthetic",
                                                      "root_index": index, "synthetic_only": True})


class TinyPredictor(nn.Module):
    def __init__(self, batch_sensitive=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.register_buffer("nonpersistent_mask", torch.tensor(1.), persistent=False)
        self.action_encoder = nn.Identity()
        self.dropout = nn.Dropout(.9)
        self.batch_sensitive = batch_sensitive

    def predict(self, history, actions):
        shift = actions[..., :1] * .1 + actions[..., 1:2] * .03
        result = (history * .25 + history.cumsum(1) * .125 + shift) * self.scale
        if self.batch_sensitive:
            result = result + len(history) * .01
        return self.dropout(result)


class MarkerEncoder:
    def encode(self, frames):
        assert self.device.type == "cpu" and not torch.is_grad_enabled()
        assert not any(m.training for m in self.modules())
        assert all(not parameter.requires_grad for parameter in self.parameters())
        marker = np.array(frames[:, :, 0, 0], dtype=np.float32, copy=True)
        value = torch.from_numpy(marker) / 255
        value = value.repeat(1, 1, self.latent_shape[-1] // 3)
        if len(self.latent_shape) == 2:
            value = value[:, :, None].expand(-1, -1, self.latent_shape[0], -1)
        if self.batch_sensitive:
            value = value + len(frames) * .001
        self.encode_count += 1
        if self.repeat_drift:
            value = value + self.encode_count * .0001
        return value

    def rollout(self, history, actions, **kwargs):
        self.rollout_calls.append({"history": history.detach().clone(), "stride": history.stride(),
                                   "actions": np.array(torch.as_tensor(actions).cpu(), copy=True),
                                   "grad_enabled": torch.is_grad_enabled()})
        return super().rollout(history, actions, **kwargs)


class TinyLeWM(MarkerEncoder, LeWMAdapter):
    latent_shape = (6,)

    def __init__(self, *, batch_sensitive=False, repeat_drift=False):
        super().__init__(TinyPredictor(batch_sensitive), SYNTHETIC_STATS,
                         metadata={"synthetic_unit_test": True})
        self.batch_sensitive, self.repeat_drift = batch_sensitive, repeat_drift
        self.encode_count, self.rollout_calls = 0, []


class MarkerDino(MarkerEncoder, DinoWMAdapter):
    def __init__(self):
        super().__init__(_ToyDinoWM(), SYNTHETIC_STATS, metadata={"synthetic_unit_test": True})
        self.batch_sensitive = self.repeat_drift = False
        self.encode_count, self.rollout_calls = 0, []


def run_suite(model=None, root=None):
    model, root = model or TinyLeWM(), root or synthetic_root()
    config, spec = settings()
    with p._frozen(model):
        return a.layout_suite(model, root, config, spec)


def comparison_rows(variants, lhs, rhs):
    return a.compare(variants[lhs], variants[rhs], control="unit", comparison="unit", lhs=lhs, rhs=rhs, ordinal=0)


def test_observed_exact_frame_batches_actions_candidates_and_history_strides():
    model, root = TinyLeWM(batch_sensitive=True), synthetic_root(2)
    variants, receipt = run_suite(model, root)
    calls = receipt["calls"]
    batches = lambda phase: [c["inputs"][0]["shape"][0] for c in calls if c["phase"] == phase and c["method"] == "encode"]
    assert batches("old_observed") == [4, 2] + [3] * 16
    assert batches("new_observed") == [6, 1, 32, 32, 32, 32]
    assert batches("crossed_public_bridges") == []  # Reuse EXACT encoded tensors.
    assert len(model.rollout_calls) == 32 + 2 + 18
    for i, (native, zero) in enumerate(zip(model.rollout_calls[:32:2], model.rollout_calls[1:32:2], strict=True)):
        candidate, branch = i % 8, i // 8
        np.testing.assert_array_equal(native["actions"], root.actions[candidate:candidate + 1])
        np.testing.assert_array_equal(zero["actions"][:, :2], native["actions"][:, :2])
        assert not zero["actions"][:, 2:].any()
        torch.testing.assert_close(native["history"], zero["history"], rtol=0, atol=0)
        key = (root.metadata["root_id"], "original", branch, candidate, 8)
        torch.testing.assert_close(native["history"][0], variants["old_single"][key]["context"], rtol=0, atol=0)
        assert not native["grad_enabled"] and native["history"].shape == (1, 3, 6)
    for index in (32, 33, 34, 43):
        call = model.rollout_calls[index]
        assert call["history"].shape == (8, 3, 6)
        assert call["stride"] == (11 * 6, 6, 1)  # NOT zero-stride expand or contiguous 3*D.
        np.testing.assert_array_equal(call["actions"], root.actions)
    for index, branch in ((35, 0), (44, 1)):
        for candidate in range(8):
            call = model.rollout_calls[index + candidate]
            key = (root.metadata["root_id"], "original", branch, candidate, 8)
            torch.testing.assert_close(call["history"][0], variants["new_single"][key]["context"], rtol=0, atol=0)
            np.testing.assert_array_equal(call["actions"], root.actions[candidate:candidate + 1])
    assert set(variants) == {*a.VARIANTS, "new_feedback_scalar"}


def test_crossed_controls_isolate_encoding_rollout_recurrence_and_reduction():
    variants, receipt = run_suite(TinyLeWM(batch_sensitive=True))
    for lhs, rhs in (("old_single", "old_bank"), ("new_single", "new_bank")):
        rows = comparison_rows(variants, lhs, rhs)
        assert all(row["context_max_abs"] == row["future_common_max_abs"] == row["target_h8_max_abs"] == 0 for row in rows)
        assert all(row["rollout_all_max_abs"] > 0 for row in rows)
        assert receipt["tensor_sha256"][lhs]["context"] == receipt["tensor_sha256"][rhs]["context"]
        assert receipt["tensor_sha256"][lhs]["future_common"] == receipt["tensor_sha256"][rhs]["future_common"]
    for lhs, rhs in (("old_single", "new_single"), ("old_bank", "new_bank")):
        rows = comparison_rows(variants, lhs, rhs)
        assert all(row["context_max_abs"] > 0 and row["target_h8_max_abs"] > 0 for row in rows)
    for lhs, rhs in (("new_bank", "new_feedback_scalar"), ("new_feedback_scalar", "new_feedback")):
        assert all(row[name + "_max_abs"] == 0 for row in comparison_rows(variants, lhs, rhs) for name in a.TENSOR_FIELDS)
    assert ("feedback_vs_public", "new_bank", "new_feedback_scalar") in a.COMPARISONS


@pytest.mark.parametrize("factory", [TinyLeWM, MarkerDino])
def test_deterministic_inputs_public_native_parity_and_no_model_rng_mutation(factory):
    model, root = factory(), synthetic_root()
    before_input, before_state = a.input_receipt(root), _state_digest(model)
    torch_rng = torch.random.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    variants, _ = run_suite(model, root)
    assert a.input_receipt(root) == before_input and _state_digest(model) == before_state
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and numpy_rng[2:] == after_numpy[2:]
    np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
    for lhs, rhs in (("old_single", "new_single"), ("old_single", "old_bank"), ("new_bank", "new_feedback_scalar")):
        rows = comparison_rows(variants, lhs, rhs)
        assert all(r[name + "_max_abs"] == 0 for r in rows for name in a.TENSOR_FIELDS)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_first_repeat_is_independent_reexecution_not_cached_or_rng_reset():
    model, root = TinyLeWM(repeat_drift=True), synthetic_root()
    first, first_receipt = run_suite(model, root)
    count = model.encode_count
    second, second_receipt = run_suite(model, root)
    assert model.encode_count == 2 * count and first_receipt["calls"] == second_receipt["calls"]
    for name in a.VARIANTS:
        rows = a.compare(first[name], second[name], control="repeat", comparison="repeat", lhs=name, rhs=name, ordinal=0)
        assert all(row["context_max_abs"] > 0 for row in rows)
        assert any(row["native_f32_delta"] != 0 for row in rows)


def test_native_prediction_never_reads_future_padding_or_goal_pixels():
    root = synthetic_root()
    before, _ = run_suite(root=root)
    changed = deepcopy(root)
    changed.future = changed.future.copy()
    changed.future[0, :, :, 1] = 255  # h2: not an old target or any native history.
    changed.goal_pixels = np.full_like(root.goal_pixels, 255)
    after, _ = run_suite(root=changed)
    for name in a.VARIANTS:
        rows = a.compare(before[name], after[name], control="unit", comparison="unit", lhs=name, rhs=name, ordinal=0)
        assert all(row[field + "_max_abs"] == 0 for row in rows for field in a.TENSOR_FIELDS)


def test_identity_matching_is_order_independent_and_rejects_missing_or_wrong_candidate():
    variants, _ = run_suite()
    records = variants["old_single"]
    reversed_records = dict(reversed(list(records.items())))
    rows = a.compare(records, reversed_records, control="unit", comparison="unit", lhs="x", rhs="x", ordinal=0)
    assert len(rows) == 16 and all(r["native_f32_delta"] == 0 for r in rows)
    changed = dict(records)
    key, value = changed.popitem()
    for candidate in (None, 99):
        wrong = dict(changed)
        if candidate is not None:
            wrong[(*key[:3], candidate, key[-1])] = value
        with pytest.raises(ValueError, match="identity sets"):
            a.compare(records, wrong, control="unit", comparison="unit", lhs="x", rhs="y", ordinal=0)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "empty"])
def test_actual_tensor_comparisons_fail_closed(bad):
    left = torch.tensor([1., 2.])
    right = {"shape": torch.ones(1, 2), "dtype": left.double(),
             "nan": torch.tensor([1., float("nan")]), "empty": torch.empty(0)}[bad]
    with pytest.raises(ValueError):
        a.tensor_delta(left, right)


def test_tensor_and_metric_max_mean_do_not_hide_cancelling_record_drift():
    variants, _ = run_suite()
    base = comparison_rows(variants, "old_single", "old_single")
    rows = []
    for index in range(8):
        for original in base:
            row = dict(original, root_id=f"root-{index}")
            sign = 1 if index % 2 else -1
            row.update(native_f32_delta=sign * 2., lhs_native_f32=5., rhs_native_f32=5. + sign * 2.,
                       rollout_h8_max_abs=3., rollout_h8_mean_abs=.5)
            rows.append(row)
    summary = a.summarize(rows)[0]
    assert summary["records"] == 128 and summary["roots"] == 8
    assert summary["metrics"]["native_f32"] == dict(lhs_mean=5., rhs_mean=5., mean_delta=0., max_abs_delta=2., mean_abs_delta=2.)
    assert summary["tensors"]["rollout_h8"] == {"max_abs": 3., "mean_abs": .5}
    for bad in (rows[:-1], rows + rows[:1], [dict(r, candidate=99) if i == 0 else r for i, r in enumerate(rows)]):
        with pytest.raises(ValueError, match="complete root banks"):
            a.summarize(bad)


def test_metrics_are_measured_from_actual_captured_tensors_not_hashes_or_pooled_features():
    variants, _ = run_suite(TinyLeWM(batch_sensitive=True))
    for records in variants.values():
        for record in records.values():
            prediction = record["rollout_h8"].numpy().astype(np.float64)
            target = record["target_h8"].numpy().astype(np.float64)
            context = record["context"][-1].numpy().astype(np.float64)
            assert record["native_f64"] == pytest.approx(np.square(prediction - target).mean(), abs=1e-14)
            assert record["persistence_f64"] == pytest.approx(np.square(context - target).mean(), abs=1e-14)
    assert a.tensor_delta(torch.tensor([0., 2., 4.]), torch.tensor([1., 0., 7.])) == (3., 2.)


def test_hashes_include_scalar_bfloat16_aliases_and_nonpersistent_buffers():
    model = TinyLeWM()
    model.register_buffer("counter", torch.tensor(3))
    model.register_buffer("bf16", torch.tensor([.5, 2.], dtype=torch.bfloat16), persistent=False)
    initial = a.model_hashes(model)
    assert initial["parameter_buffer_sha256"] == _state_digest(model)
    with torch.no_grad():
        for tensor, kind in ((model.model.scale, "parameters"), (model.model.nonpersistent_mask, "buffers"),
                             (model.counter, "buffers"), (model.bf16, "buffers")):
            saved = tensor.clone()
            tensor.add_(1)
            changed = a.model_hashes(model)
            assert changed[kind + "_sha256"] != initial[kind + "_sha256"]
            other = "buffers" if kind == "parameters" else "parameters"
            assert changed[other + "_sha256"] == initial[other + "_sha256"]
            tensor.copy_(saved)
            assert a.model_hashes(model) == initial
    model.register_buffer("alias", model.bf16, persistent=False)
    assert a.model_hashes(model)["buffers_sha256"] != initial["buffers_sha256"]
    assert a.array_sha256(torch.arange(6).reshape(2, 3).T) == a.array_sha256(torch.arange(6).reshape(2, 3).T.contiguous())


@pytest.mark.parametrize("field,value", [("horizon", 7), ("candidates", 7), ("horizons", [8, 4, 1]),
                                         ("horizons", [8]), ("batch_size", 32), ("batch_size", True)])
def test_changed_research_layout_configuration_is_not_silently_accepted(field, value):
    config, spec = settings()
    config[field] = value
    with pytest.raises(ValueError):
        a.checked_settings(config, spec)


@pytest.mark.parametrize("value", [4, True, 32.0, 0])
def test_followup_encoder_batch_is_observed_not_arbitrary(value):
    config, spec = settings()
    spec["encode_batch_size"] = value
    with pytest.raises(ValueError):
        a.checked_settings(config, spec)


@pytest.fixture
def roster(tmp_path, monkeypatch):
    """Manifest/authentication fixture, not a purported PairedRoot NPZ.

    Strict production schema loading is delegated to the existing pipeline; this
    fixture replaces ONLY that boundary with labeled, deterministic test doubles.
    """
    config, spec = settings()
    config["counts"]["pilot"] = 10
    data = tmp_path / "data"
    (data / "pilot").mkdir(parents=True)
    order, files, roots = [7, 2, 9, 0, 6, 5, 8, 1, 4, 3], {}, {}
    for index in order:
        root = synthetic_root(index)
        relative = "pilot/" + root.metadata["root_id"] + ".npz"
        path = data / relative
        path.write_bytes(f"synthetic boundary fixture {index}".encode())
        files[relative], roots[str(path)] = digest(path), root
    manifest = data / "manifest-development.json"
    manifest.write_text(json.dumps({"config": config, "files": {"pilot": files}}))
    monkeypatch.setattr(a, "MANIFEST_SHA256", digest(manifest))

    def scan(paths, passed_config):
        assert passed_config == config and list(paths) == ["pilot"]
        return {"pilot": [dict(root_id=roots[str(path)].metadata["root_id"],
                               root_index=roots[str(path)].metadata["root_index"]) for path in paths["pilot"]]}

    def load(path, split, passed_config, hashes):
        assert split == "pilot" and passed_config == config
        assert digest(path) == hashes[str(path)]
        return roots[str(path)]

    monkeypatch.setattr(a.p, "_scan", scan)
    monkeypatch.setattr(a.d, "_load", load)
    return config, spec, data, files, roots


def test_first_eight_manifest_order_not_lexical_or_outcome_selected(roster):
    config, _, data, files, _ = roster
    for name in list(files)[8:]:
        (data / name).unlink()  # Unselected payloads need not even be present.
    selected, hashes = a.select_roster(data, config)
    assert [s["root_index"] for s in selected] == [7, 2, 9, 0, 6, 5, 8, 1]
    assert [s["relative_path"] for s in selected] == list(files)[:8]
    assert [s["manifest_ordinal"] for s in selected] == list(range(8))
    assert len(hashes) == 9


@pytest.mark.parametrize("bad", ["manifest", "payload", "config", "short", "duplicate", "traversal", "symlink"])
def test_input_authentication_and_manifest_validation_fail_closed(roster, monkeypatch, tmp_path, bad):
    config, _, data, files, _ = roster
    manifest = data / "manifest-development.json"
    if bad == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    elif bad == "payload":
        (data / next(iter(files))).write_bytes(b"changed")
    elif bad == "config":
        config = deepcopy(config)
        config["seed"] += 1
    elif bad == "symlink":
        first = data / next(iter(files))
        target = tmp_path / "outside.npz"
        target.write_bytes(first.read_bytes())
        first.unlink()
        first.symlink_to(target)
    else:
        value = json.loads(manifest.read_text())
        if bad == "short":
            value["files"]["pilot"] = dict(list(files.items())[:7])
        elif bad == "traversal":
            value["files"]["pilot"]["pilot/../outside.npz"] = value["files"]["pilot"].pop(next(iter(files)))
        if bad == "duplicate":
            manifest.write_text('{"config": {}, "config": {}}')
        else:
            manifest.write_text(json.dumps(value))
        monkeypatch.setattr(a, "MANIFEST_SHA256", digest(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        a.select_roster(data, config)


def test_source_pins_match_inspected_current_files_and_fail_closed(monkeypatch):
    receipt = a.source_receipt()
    assert receipt[str(ROOT / "scripts/audit_execution_layout.py")] == digest(ROOT / "scripts/audit_execution_layout.py")
    hashes = a.code_hashes()
    hashes["models.py"] = "0" * 64
    monkeypatch.setattr(a, "code_hashes", lambda: hashes)
    with pytest.raises(ValueError, match="inspected generated paths"):
        a.source_receipt()


def test_backend_default_has_no_setters_and_control_changes_only_one_property(monkeypatch):
    owner = SimpleNamespace(precision="none", unrelated=True)
    monkeypatch.setattr(a, "backend_state", lambda: dict(precision=owner.precision, unrelated=owner.unrelated))
    accesses = []

    def property_(path):
        accesses.append(path)
        return owner, "precision"

    monkeypatch.setattr(a, "backend_property", property_)
    with a.backend_control() as state:
        assert state == {"precision": "none", "unrelated": True}
    assert accesses == []
    with a.backend_control({"path": "test", "value": "ieee"}) as state:
        assert state == {"precision": "ieee", "unrelated": True}
    assert owner.precision == "none" and accesses == ["test"]


@pytest.mark.parametrize("raises", [False, True])
def test_actual_pytorch_controls_restore_cpu_host_flags_even_on_failure(raises):
    # Backend FLAGS only, not a CUDA kernel or real model execution.
    before = a.backend_state()
    for control in a.control_plan(before):
        if control["status"] == "skipped":
            continue
        try:
            with a.backend_control(control) as observed:
                assert observed[control["path"]] == control["value"]
                if raises:
                    raise LookupError("synthetic failure")
        except LookupError:
            assert raises
        assert a.backend_state() == before


def test_backend_control_setup_failure_restores_partially_set_property(monkeypatch):
    class PartialSetter:
        value = "none"

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if value == "ieee":
                raise RuntimeError("synthetic unsupported setter")

    owner = PartialSetter()
    monkeypatch.setattr(a, "backend_state", lambda: {"precision": owner.value})
    monkeypatch.setattr(a, "backend_property", lambda path: (owner, "value"))
    with pytest.raises(RuntimeError, match="unsupported setter"):
        with a.backend_control({"path": "test", "value": "ieee"}):
            pytest.fail("must not execute a suite after a failed setter")
    assert owner.value == "none"


def test_backend_plan_exposes_noops_unavailable_and_disabled_not_fake_controls():
    state = {"cuda.matmul.fp32_precision": "ieee", "cudnn.conv.fp32_precision": {"unavailable": "test"},
             "cudnn.deterministic": True, "cudnn.benchmark": False}
    planned = a.control_plan(state)
    assert [c["name"] for c in planned if c["status"] == "planned"] == ["matmul_tf32"]
    assert next(c for c in planned if c["name"] == "cudnn_conv_ieee")["reason"] == "API unavailable"
    assert all(c["status"] == "skipped" and c["reason"] == "disabled by CLI" for c in a.control_plan(state, False))


def test_legacy_precision_getter_failure_is_exposed_not_a_control_failure(monkeypatch):
    def unsupported():
        raise RuntimeError("legacy getter cannot describe a per-backend override")

    monkeypatch.setattr(torch, "get_float32_matmul_precision", unsupported)
    state = a.backend_state()
    assert state["float32_matmul_precision"] == {
        "unavailable": "RuntimeError: legacy getter cannot describe a per-backend override"}
    with a.backend_control():
        assert a.backend_state() == state


def test_publication_compact_matching_csv_provenance_repeat_and_exclusive_output(roster, tmp_path):
    config, spec, data, _, _ = roster
    selected, hashes = a.select_roster(data, config)
    output, model = tmp_path / "audit", TinyLeWM(batch_sensitive=True)
    model.model.scale.requires_grad_(True)
    model.model.dropout.train()
    original, backend = a.model_hashes(model), a.backend_state()
    result = a.execute_audit(model, config, spec, selected, output, hashes, a.source_receipt(), backend_controls=False)
    report = json.loads((output / "report.json").read_text())
    plan = json.loads((output / "selection.json").read_text())
    with (output / "records.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert result["roots"] == 8 and result["comparisons"] == 12
    assert len(rows) == report["csv_records"] == 128 * (7 + 5)
    assert len(report["executions"]) == 16 and len(report["summary"]) == 12
    assert [r["root_index"] for r in plan["roots"]] == [7, 2, 9, 0, 6, 5, 8, 1]
    assert report["model_state"] == original == a.model_hashes(model)
    assert report["parameter_buffer_sha256_before"] == report["parameter_buffer_sha256_after"] == _state_digest(model)
    assert report["backend_restored"] and a.backend_state() == backend
    assert model.model.scale.requires_grad and model.model.dropout.training
    assert model.model.scale.grad is None
    assert report["input_sha256"] == hashes and report["code_sha256"] == a.source_receipt()
    assert report["model"]["synthetic_unit_test"] and report["runtime"]["device"] == "cpu"
    assert not report["confirmatory"] and report["model_unchanged"]
    assert set(report["artifacts"]) == {"selection.json", "records.csv"}
    assert all(digest(output / name) == expected for name, expected in report["artifacts"].items())
    assert all(s["records"] == 128 and s["roots"] == 8 for s in report["summary"])
    for row in report["summary"]:
        if row["comparison"] == "identical_layout_repeat":
            assert all(value["max_abs"] == 0 for value in row["tensors"].values())
    assert len({(r["control"], r["comparison"], r["lhs"], r["rhs"], r["root_id"], r["branch"], r["candidate"], r["horizon"])
                for r in rows}) == len(rows)
    with pytest.raises(FileExistsError):
        a.execute_audit(model, config, spec, selected, output, hashes, {}, backend_controls=False)
    assert digest(output / "report.json") == digest(Path(result["report"]))


def test_backend_matrix_is_matched_to_baseline_with_actual_settings_and_immutable_plan(roster, tmp_path, monkeypatch):
    config, spec, data, _, _ = roster
    selected, hashes = a.select_roster(data, config)
    before = a.backend_state()
    chosen = next(control for control in a.control_plan(before) if control["status"] == "planned")
    monkeypatch.setattr(a, "control_plan", lambda state, enabled: [chosen.copy()])
    output = tmp_path / "backend"
    a.execute_audit(TinyLeWM(), config, spec, selected, output, hashes, {}, backend_controls=True)
    report = json.loads((output / "report.json").read_text())
    plan = json.loads((output / "selection.json").read_text())
    assert plan["controls"][0]["status"] == "planned"
    assert report["controls"][0]["status"] == "executed" and report["controls"][0]["suites_completed"] == 8
    assert len(report["executions"]) == len(report["summary"]) == 24
    assert report["csv_records"] == 3072 and a.backend_state() == before
    for execution in report["executions"]:
        expected = chosen["value"] if execution["phase"] == chosen["name"] else before[chosen["path"]]
        assert execution["backend"][chosen["path"]] == expected
    backend_rows = [row for row in report["summary"] if row["comparison"] == "backend_only"]
    assert len(backend_rows) == 5
    for row in backend_rows:
        assert row["lhs"].split("/")[1] == row["rhs"].split("/")[1]
        assert row["lhs"].startswith("baseline/") and row["rhs"].startswith(chosen["name"] + "/")
        # This algebraic CPU double does not invoke CUDA matmul. Zero sensitivity
        # is reported, not invented from a changed permission flag.
        assert all(value["max_abs"] == 0 for value in row["tensors"].values())


@pytest.mark.parametrize("kind", ["parameter", "buffer", "input", "metadata", "failure"])
def test_mutation_or_failure_forbids_completion_and_restores_modes_backend(roster, tmp_path, monkeypatch, kind):
    config, spec, data, _, _ = roster
    selected, hashes = a.select_roster(data, config)
    output, model = tmp_path / "failed", TinyLeWM()
    model.model.scale.requires_grad_(True)
    model.model.dropout.train()
    original_suite, backend = a.layout_suite, a.backend_state()

    def broken(adapter, root, *args):
        assert (output / "selection.json").is_file() and not (output / "report.json").exists()
        if kind == "failure":
            raise RuntimeError("synthetic inference failure")
        result = original_suite(adapter, root, *args)
        if kind == "parameter":
            adapter.model.scale.add_(1)
        elif kind == "buffer":
            adapter.model.nonpersistent_mask.add_(1)
        elif kind == "input":
            root.actions[1, 2, 0] += .01
        else:
            root.metadata["changed"] = True
        return result

    monkeypatch.setattr(a, "layout_suite", broken)
    with pytest.raises((ValueError, RuntimeError), match="changed|synthetic inference"):
        a.execute_audit(model, config, spec, selected, output, hashes, {}, backend_controls=False)
    assert not (output / "report.json").exists() and not (output / "records.csv").exists()
    assert model.model.scale.requires_grad and model.model.dropout.training
    assert a.backend_state() == backend and "encode" not in model.__dict__ and "rollout" not in model.__dict__


def test_final_reauthentication_detects_changed_config_or_source(roster, tmp_path, monkeypatch):
    config, spec, data, _, _ = roster
    selected, hashes = a.select_roster(data, config)
    source = tmp_path / "test-source.txt"
    source.write_text("original")
    sources = {str(source): digest(source)}
    original_suite = a.layout_suite

    def changed(*args):
        result = original_suite(*args)
        source.write_text("changed during execution")
        return result

    monkeypatch.setattr(a, "layout_suite", changed)
    with pytest.raises(ValueError, match="Input/source changed"):
        a.execute_audit(TinyLeWM(), config, spec, selected, tmp_path / "out", hashes, sources, backend_controls=False)
    assert not (tmp_path / "out/report.json").exists()


def test_capture_restores_instance_methods_and_class_methods_on_error():
    model, trace = TinyLeWM(), []
    original = model.encode
    model.encode = lambda *args: (_ for _ in ()).throw(LookupError("test encode failure"))
    local = model.encode
    with pytest.raises(LookupError):
        with a.capture_calls(model, "unit", trace):
            model.encode(np.zeros((1, 1, 224, 224, 3), dtype=np.uint8))
    assert model.encode is local and "rollout" not in model.__dict__
    del model.encode
    assert model.encode == original


def cli_args(tmp_path, model="lewm"):
    return ["--config", str(tmp_path / "config"), "--spec", str(tmp_path / "spec"),
            "--data", str(tmp_path / "data"), "--cache", str(tmp_path / "cache"),
            "--output", str(tmp_path / "out"), "--model", model]


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "not-a-device"])
def test_cli_cannot_execute_local_real_model_or_fall_back(tmp_path, monkeypatch, capsys, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        a.main(cli_args(tmp_path) + ["--device", device])
    assert exc.value.code == 2 and "error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_trust_required_before_file_access_or_checkpoint_load(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(a, "read_json", forbidden)
    with pytest.raises(SystemExit) as exc:
        a.main(cli_args(tmp_path, "dinowm"))
    assert exc.value.code == 2 and "trust-checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_existing_or_dangling_output_rejected_before_cuda_or_io(tmp_path, monkeypatch, capsys, symlink):
    output = tmp_path / "out"
    if symlink:
        output.symlink_to(tmp_path / "missing")
    else:
        output.mkdir()
        (output / "keep").write_text("preserve")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(SystemExit) as exc:
        a.main(cli_args(tmp_path))
    assert exc.value.code == 2 and "Output exists" in capsys.readouterr().err
    assert output.is_symlink() if symlink else (output / "keep").read_text() == "preserve"


def test_cli_parser_defaults_gpu_controls_and_no_root_or_candidate_selection(tmp_path):
    options = a.parser().parse_args(cli_args(tmp_path))
    assert options.device == "cuda" and options.backend_controls is True
    assert options.trust_checkpoint is False
    assert not {"roots", "candidates", "seed", "horizon", "offset"} & vars(options).keys()
    assert a.parser().parse_args(cli_args(tmp_path) + ["--no-backend-controls"]).backend_controls is False


def test_cli_forwards_config_cache_data_device_and_trust_without_local_model_execution(tmp_path, monkeypatch, capsys):
    config, spec = settings()
    (tmp_path / "config").write_text(json.dumps(config))
    (tmp_path / "spec").write_text(json.dumps(spec))
    selected, calls = [{"synthetic": True}], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(a, "source_receipt", lambda: {})

    def select(data, passed_config):
        assert data == tmp_path / "data" and passed_config == config
        return selected, {}

    def load(name, cache, device, *, trust_checkpoint):
        assert (name, cache, str(device), trust_checkpoint) == ("lewm", tmp_path / "cache", "cuda:2", True)
        calls.append("load-stub")
        return "no-real-model"

    def execute(model, passed_config, passed_spec, roster, output, hashes, sources, *, backend_controls):
        assert model == "no-real-model" and (passed_config, passed_spec, roster) == (config, spec, selected)
        assert output == tmp_path / "out" and not backend_controls and not sources
        assert hashes == {str(tmp_path / "config"): digest(tmp_path / "config"), str(tmp_path / "spec"): digest(tmp_path / "spec")}
        calls.append("execute-stub")
        return {"unit_test": True}

    monkeypatch.setattr(a, "select_roster", select)
    monkeypatch.setattr(models, "load_model", load)
    monkeypatch.setattr(a, "execute_audit", execute)
    a.main(cli_args(tmp_path) + ["--device", "cuda:2", "--trust-checkpoint", "--no-backend-controls"])
    assert calls == ["load-stub", "execute-stub"] and json.loads(capsys.readouterr().out) == {"unit_test": True}
    assert not (tmp_path / "out").exists()


def test_cli_refuses_implicit_dinov2_downloads(tmp_path, monkeypatch, capsys):
    config, spec = settings()
    (tmp_path / "config").write_text(json.dumps(config))
    (tmp_path / "spec").write_text(json.dumps(spec))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(a, "source_receipt", lambda: {})
    monkeypatch.setattr(a, "select_roster", lambda *args: ([], {}))
    with pytest.raises(SystemExit) as exc:
        a.main(cli_args(tmp_path, "dinowm") + ["--trust-checkpoint"])
    assert exc.value.code == 2 and "no implicit downloads" in capsys.readouterr().err