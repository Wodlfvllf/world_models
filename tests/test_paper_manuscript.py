"""CPU-only manuscript source/provenance contracts; no research execution."""
import importlib.util
import json
from pathlib import Path
import re

import pytest

pytest.importorskip("bibtexparser")
pytest.importorskip("pymupdf")

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


a = module("paper_audit", ROOT / "paper/audit_manuscript.py")
g = module("paper_assets", ROOT / "paper/generate_assets.py")


def test_all_manuscript_citations_resolve():
    cited, count = a.citations([ROOT / "paper/main.tex", *sorted((ROOT / "paper/sections").glob("*.tex"))],
                              ROOT / "paper/references.bib")
    assert count == 29 and len(cited) == 21
    assert {"resani2026dojepa", "liu2026djepa", "vakalis2026interventiongap"} <= set(cited)


def test_bibtex_comment_does_not_contain_entry_marker():
    # Native BibTeX scans '@' even in nominal percent-comment prose.
    assert not re.search(r"^%.*@", (ROOT / "paper/references.bib").read_text(), re.M)


def test_missing_citation_fails_closed(tmp_path):
    tex, bib = tmp_path / "test.tex", tmp_path / "test.bib"
    tex.write_text(r"\citep{missing}")
    bib.write_text("@misc{present, title={Synthetic}, year={2026}}")
    with pytest.raises(ValueError, match="Unresolved"):
        a.citations([tex], bib)


@pytest.mark.parametrize("path,expected", [("/a/0/value", 7), ("/a/1/ci99/1", 2), ("/escaped~1key", 3)])
def test_statistic_pointer_resolution(path, expected):
    assert a.pointer({"a": [{"value": 7}, {"ci99": [1, 2]}], "escaped/key": 3}, path) == expected


@pytest.mark.parametrize("rows", [[], [{"horizon": 8}, {"horizon": 8}]])
def test_selection_refuses_absent_or_duplicate_statistic(rows):
    with pytest.raises(ValueError, match="unique"):
        g.one(rows, horizon=8)


def test_99_and_95_intervals_are_not_interchanged():
    value = {"estimate": 1., "ci95": [.5, 1.5], "ci99": [0., 2.]}
    assert g.interval_text(value, "ci99", 2) == "1.00 [0.00, 2.00]"
    assert g.interval_text(value, "ci95", 2) == "1.00 [0.50, 1.50]"


def test_primary_family_and_all_seeds_are_preserved():
    manifest = json.loads((ROOT / "paper/generated/assets.sources.json").read_text())
    assert manifest["all_mlp_seeds"] == [11, 29, 47]
    assert [tuple(row) for row in manifest["primary_family"]] == g.PRIMARY
    assert manifest["primary_confidence"] == .99 and manifest["secondary_confidence"] == .95
    assert manifest["no_model_execution"] is True
    assert len(manifest["source_reports"]) == 9
    assert len(manifest["displayed_statistics"]) == 206


@pytest.mark.parametrize("name", ["probes_rows.tex", "feedback_rows.tex", "primary_rows.tex",
                                  "ranking_rows.tex", "numerical_rows.tex", "slice_rows.tex"])
def test_generated_tables_are_complete_and_authenticated(name):
    text = (ROOT / "paper/generated" / name).read_text()
    assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1
    assert text.rstrip().endswith(r"\end{tabular}")
    manifest = json.loads((ROOT / "paper/generated/assets.sources.json").read_text())
    assert a.sha(ROOT / "paper/generated" / name) == manifest["output_sha256"][name]


def test_new_generation_requires_explicit_overwrite(tmp_path):
    with pytest.raises(FileExistsError, match="explicit"):
        g.build_assets(tmp_path)


def test_hypothesis_insertion_labels_exist():
    body = (ROOT / "paper/sections/body.tex").read_text()
    appendix = (ROOT / "paper/sections/appendix.tex").read_text()
    for label in ("sec:measurementmethods", "sec:measurement", "sec:action", "sec:replication",
                  "sec:ranking", "sec:decisionresults", "sec:numerics", "app:numerics"):
        assert r"\label{" + label + "}" in body + appendix


def test_working_notice_does_not_claim_review_or_acceptance():
    text = (ROOT / "paper/main.tex").read_text()
    assert "not submitted or accepted" in text
    assert r"\usepackage[accepted]" not in text
    assert r"\icmlcorrespondingauthor" in text