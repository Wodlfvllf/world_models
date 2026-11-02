"""CPU/JSON plotting contracts. All temporary fixtures are SYNTHETIC, not research.

The optional local-artifact test reads only the already retrieved JSON/ZIP files.
No model imports, experiment launchers, fitting, downloads, or bootstrapping.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plotter():
    spec = importlib.util.spec_from_file_location("followup_plot", ROOT / "scripts/plot_followup_summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stat(value, width=.005):
    return {"estimate": value, "ci95": [value - width, value + width], "clusters": 128}


def synthetic_report(plotter, stage="probes", model="lewm"):
    report = {
        "schema": "wm_audit.diagnostic.v1", "kind": plotter.BUNDLES[stage]["kind"],
        "model": {"name": model}, "confirmatory": False, "caveat": "SYNTHETIC TEST FIXTURE ONLY",
        "spec": {"schema": "wm_audit.followup.v1", "confirmatory": False, "primary_horizon": 8,
                 "bootstrap_resamples": 5000, "mlp_seeds": [11, 29, 47]},
        "config": {"horizon": 8, "candidates": 8, "counts": dict(plotter.COUNTS)},
        "protocol": {"confirmatory": False, "uncertainty": "equal-root/episode pointwise bootstrap"},
    }
    if stage == "probes":
        report.update(followup_schema="wm_audit.followup.probes.v1", training_roots=128,
                      validation_roots=64, evaluation_roots=128, methods=[*plotter.PROBES, "train_mean"])
        report["protocol"]["canonical_roots"] = {"scorer": 128, "validation": 64, "pilot": 128}
        report["selection"] = {"spec": deepcopy(report["spec"]),
                               "fit_root_ids": [f"scorer-{i}" for i in range(128)],
                               "validation_root_ids": [f"validation-{i}" for i in range(64)]}
        report["summary"] = [{"appearance": "original", "surface": "horizon_8", "method": method,
                              "roots": 128, "samples": 2048, "pairs": 1024,
                              "metrics": {"block_distance": stat(10 + i)}}
                             for i, method in enumerate(report["methods"])]
        report["records"] = [{"appearance": "original", "surface": "horizon_8", "method": method,
                              "root_id": f"pilot-{i}"} for method in report["methods"] for i in range(128)]
        decoy = deepcopy(report["summary"][0])
        decoy.update(surface="all", metrics={"block_distance": stat(999)})
        report["summary"].append(decoy)
    else:
        actual = "actual_none" if model == "lewm" else "actual_recurrent"
        metrics = {actual: stat(.2), "local_persistence": stat(.3), actual + "_minus_baseline": stat(-.1)}
        if stage == "dynamics":
            metrics.update(effect_error=stat(.2), true_effect_energy=stat(.5), effect_error_minus_zero=stat(-.3),
                           best_coverage=stat(.1234), **{key: stat(.02 + i / 100) for i, key in enumerate(plotter.REGRETS)})
            report["records"] = [{"appearance": "original", "horizon": 8, "cluster": f"pilot-{i}", "branch": branch}
                                 for i in range(128) for branch in (0, 1)]
        else:
            report["windows"] = [{"episode": i, "start": i + 2} for i in range(128)]
            report["records"] = [{"appearance": "original", "horizon": 8, "cluster": str(i), "start": i + 2}
                                 for i in range(128)]
        report["summary"] = [{"appearance": "original", "horizon": 8, "metrics": metrics}]
        decoy = deepcopy(report["summary"][0])
        decoy["appearance"] = "background_only"
        report["summary"].append(decoy)
    return report


@pytest.fixture
def authenticated_root(plotter, tmp_path, monkeypatch):
    """Test-only archive pins; production pins cannot be overridden by a CLI flag."""
    for stage, bundle in list(plotter.BUNDLES.items()):
        files = {}
        for model in plotter.MODELS:
            relative = f"runs/{bundle['run']}/{model}/{stage}/report.json"
            files[relative] = json.dumps(synthetic_report(plotter, stage, model)).encode()
        inventory = {"schema": "wm_audit.backup.v1", "files": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)} for name, raw in files.items()}}
        files["BACKUP.json"] = json.dumps(inventory).encode()
        archive = tmp_path / bundle["archive"]
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
            for name, raw in files.items():
                path = tmp_path / stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                zipped.writestr(name, raw)
        digest = plotter.sha256(archive)
        monkeypatch.setitem(plotter.BUNDLES, stage, {**bundle, "sha256": digest})
        Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n")
        (tmp_path / stage / "RESTORE-VERIFIED.json").write_text(json.dumps({"archive_sha256": digest, "files_verified": 2}))
    return tmp_path


@pytest.mark.parametrize("path,value", [
    (("schema",), "unknown"), (("kind",), "followup_dynamics"), (("model", "name"), "other"),
    (("followup_schema",), "unknown"), (("confirmatory",), True), (("confirmatory",), 0),
    (("spec", "confirmatory"), True), (("protocol", "confirmatory"), 0),
    (("selection", "spec", "confirmatory"), True), (("spec", "primary_horizon"), 8.0),
    (("spec", "bootstrap_resamples"), 4999), (("spec", "mlp_seeds"), [11, 29]),
    (("config", "candidates"), 7), (("config", "counts", "pilot"), 127),
    (("protocol", "canonical_roots", "scorer"), 127), (("evaluation_roots",), 127),
    (("validation_roots",), True), (("selection", "fit_root_ids"), ["duplicate"] * 128),
    (("summary", 0, "roots"), 127), (("summary", 0, "samples"), 2047),
    (("summary", 0, "metrics", "block_distance", "estimate"), True),
    (("summary", 0, "metrics", "block_distance", "estimate"), float("nan")),
    (("summary", 0, "metrics", "block_distance", "ci95"), [2, 1]),
    (("summary", 0, "metrics", "block_distance", "ci95"), [0, float("inf")]),
    (("summary", 0, "metrics", "block_distance", "ci95"), None),
    (("summary", 0, "metrics", "block_distance", "clusters"), 128.0),
])
def test_invalid_schema_counts_flags_and_intervals_fail_closed(plotter, path, value):
    report = synthetic_report(plotter)
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        plotter.select_report(report, "probes", "lewm")


@pytest.mark.parametrize("stage", ["probes", "dynamics", "recorded-feedback"])
def test_missing_or_duplicate_primary_roots_rejected(plotter, stage):
    report = synthetic_report(plotter, stage)
    report["records"][0] = deepcopy(report["records"][1])
    with pytest.raises(ValueError):
        plotter.select_report(report, stage, "lewm")


@pytest.mark.parametrize("change", ["missing_seed", "duplicate_summary", "missing_summary", "missing_ci"])
def test_seed_and_summary_selection_never_silent(plotter, change):
    report = synthetic_report(plotter)
    if change == "missing_seed":
        report["methods"].remove("mlp_seed29")
    elif change == "duplicate_summary":
        report["summary"].append(deepcopy(report["summary"][0]))
    elif change == "missing_summary":
        report["summary"].pop(0)
    else:
        del report["summary"][0]["metrics"]["block_distance"]["ci95"]
    with pytest.raises(ValueError):
        plotter.select_report(report, "probes", "lewm")


@pytest.mark.parametrize("metric", ["actual_none_minus_baseline", "effect_error_minus_zero"])
def test_wrong_contrast_baseline_rejected(plotter, metric):
    report = synthetic_report(plotter, "dynamics")
    report["summary"][0]["metrics"][metric] = stat(-.99)
    with pytest.raises(ValueError, match="baseline|LOCAL"):
        plotter.select_report(report, "dynamics", "lewm")


@pytest.mark.parametrize("raw", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '[]'])
def test_strict_json(plotter, raw):
    with pytest.raises(ValueError):
        plotter.read_json(raw)


@pytest.mark.parametrize("target", ["report", "archive", "inventory", "receipt", "restore"])
def test_authentication_tampering_rejected(plotter, authenticated_root, target):
    bundle = plotter.BUNDLES["probes"]
    root = authenticated_root
    paths = {"archive": root / bundle["archive"], "inventory": root / "probes/BACKUP.json",
             "report": next((root / "probes").glob("runs/*/lewm/*/report.json")),
             "receipt": root / (bundle["archive"] + ".sha256"), "restore": root / "probes/RESTORE-VERIFIED.json"}
    path = paths[target]
    if target == "restore":
        data = json.loads(path.read_text())
        data["files_verified"] = 3
        path.write_text(json.dumps(data))
    elif target == "receipt":
        path.write_text("0" * 64 + "  " + bundle["archive"])
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256|inventory|receipt|count"):
        plotter.load_summary(root)


def test_selection_provenance_exactly_matches_sources(plotter, authenticated_root):
    data = plotter.load_summary(authenticated_root)
    assert {k: len(v) for k, v in data["selected_values"].items()} == {
        "physical_block_error": 8, "feedback_minus_local": 4, "action_effect_minus_zero": 2,
        "regret": 8, "bank_best_coverage": 2}
    reports = {s["source_id"]: json.loads((authenticated_root / s["path"]).read_text()) for s in data["reports"]}
    for points in data["selected_values"].values():
        for point in points:
            for selected in (point, *point.get("components", {}).values()):
                value = reports[point["source_id"]]
                for key in selected["json_pointer"].split("/")[1:]:
                    value = value[int(key)] if isinstance(value, list) else value[key]
                assert {k: selected[k] for k in ("estimate", "ci95", "clusters")} == value
            assert point["appearance"] == "original" and point["horizon"] == 8
    assert [p["method"] for p in data["selected_values"]["physical_block_error"]] == list(plotter.PROBES) * 2
    assert data["bank_best_coverage_label"] == "12.34%"  # Synthetic; derived, never hard-coded to real results.


def test_render_and_explicit_overwrite_cli(plotter, authenticated_root):
    output = authenticated_root / "figures"
    assert plotter.main(["--input-root", str(authenticated_root)]) is None
    assert (output / plotter.OUTPUTS[0]).read_bytes().startswith(b"%PDF-")
    assert b"/Subtype /Type3" not in (output / plotter.OUTPUTS[0]).read_bytes()
    assert (output / plotter.OUTPUTS[1]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    provenance = json.loads((output / plotter.OUTPUTS[2]).read_text())
    assert provenance["confirmatory"] is False and provenance["backend"].lower() == "agg"
    assert len(provenance["reports"]) == 6
    for entry in provenance["outputs"]:
        assert plotter.sha256(output / entry["path"]) == entry["sha256"]
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit) as exc:
        plotter.main(["--input-root", str(authenticated_root)])
    assert exc.value.code == 2
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    plotter.main(["--input-root", str(authenticated_root), "--overwrite"])


@pytest.mark.parametrize("name", ["followup_summary.pdf", "followup_summary.png", "followup_summary.sources.json"])
def test_any_existing_output_blocks_the_whole_publication(plotter, tmp_path, name):
    (tmp_path / name).write_bytes(b"preserve me")
    with pytest.raises(FileExistsError, match="--overwrite"):
        plotter.write_summary({}, tmp_path)
    assert [(p.name, p.read_bytes()) for p in tmp_path.iterdir()] == [(name, b"preserve me")]


def test_latent_panels_have_independent_axes_and_labels(plotter, authenticated_root):
    data = plotter.load_summary(authenticated_root)
    fig = plotter.make_figure(data)
    import matplotlib.pyplot as plt
    try:
        assert len(fig.axes) == 8  # A, four B facets, two C facets, D.
        for axis in fig.axes[1:7]:
            assert list(axis.get_shared_x_axes().get_siblings(axis)) == [axis]
        text = " ".join(t.get_text() for t in fig.texts)
        for label in ("exploratory", "LOCAL", "128 generated roots", "128 recorded episodes", "pointwise 95%", "12.34%"):
            assert label in text
    finally:
        plt.close(fig)


def test_import_does_not_load_research_stack():
    script = str(ROOT / "scripts/plot_followup_summary.py")
    code = (f"import runpy, sys; runpy.run_path({script!r}); "
            "assert not any(k.split('.')[0] in {'torch', 'wm_audit', 'gymnasium'} for k in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_retrieved_real_json_if_available(plotter):
    if not all((plotter.DEFAULT_ROOT / b["archive"]).is_file() for b in plotter.BUNDLES.values()):
        pytest.skip("Authenticated retrieval archives are not present; no download attempted")
    data = plotter.load_summary(plotter.DEFAULT_ROOT)
    assert len(data["reports"]) == 6
    assert data["bank_best_coverage_label"] == "11.28%"
    feedback = next(p for p in data["selected_values"]["feedback_minus_local"]
                    if p["model"] == "dinowm" and p["domain"] == "generated")
    assert feedback["ci95"][0] < 0 < feedback["ci95"][1]
    assert all(p["clusters"] == 128 for points in data["selected_values"].values() for p in points)