"""CPU-only paper assets from authenticated reports; no models or new tests.

Copy published statistics, retain every probe seed, and distinguish 99% primary
intervals from exploratory 95% intervals. All displayed values have JSON pointers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OLD = ROOT / "runs/molab-20261102-retrieved"
NEW = ROOT / "runs/paper-studies-20261103-retrieved"
MODELS = ("lewm", "dinowm")
LABELS = {"lewm": "LeWM", "dinowm": "DINO (Point)"}
JOINT_SHA = "a9f77137c2480c07420ce2d75458af86b076f8934cb44ad91e4bff2f0e987fbc"
ARCHIVES = {
    "probes": (OLD / "followup-probes-20261102-verified.zip", "678ed416c59dfb383b0b64b92de234a82d99b872187bb2ac3514f37d74eb1fe7"),
    "dynamics": (OLD / "dynamics-20261102-verified.zip", "538e62131fcf4f4139cfb86fa8abea4817297e18a9e02d76bbb08805208482d9"),
    "recorded-feedback": (OLD / "recorded-feedback-20261102-verified.zip", "71892b690d0d365f2e56483af28bd6e97a24fcee9deb0db0b6166c09a9f1f227"),
    "numerical-lewm": (NEW / "numerical-lewm.zip", "d244e69da1a693725b94491612e798bf3091ae8acbb230aa92eed610fafb98ee"),
    "numerical-dinowm": (NEW / "numerical-dinowm.zip", "4c84faef80b1557a9edfd836cea163c9304326268965ce9a6e05664d0d47c7ea"),
    "replication": (NEW / "replication.zip", "7e4ffb90d484e7fc5f736715d23ebb22e0af9da0ce83d38494a79748a557e27d"),
}
PRIMARY = [("lewm", "effect_error_minus_zero"), ("lewm", "matched_minus_all_wrong"),
           ("dinowm", "effect_error_minus_zero"), ("dinowm", "matched_minus_all_wrong"),
           ("dinowm", "forecast_minus_endpoint_regret")]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(rows, **selector):
    found = [(i, row) for i, row in enumerate(rows) if all(row.get(k) == v for k, v in selector.items())]
    require(len(found) == 1, f"Expected unique statistic: {selector}")
    return found[0]


def authenticated_member(stage, name, sources):
    archive, expected = ARCHIVES[stage]
    require(sha(archive) == expected, f"Archive changed: {stage}")
    require(Path(str(archive) + ".sha256").read_text().split()[0] == expected, "Receipt changed")
    with zipfile.ZipFile(archive) as zipped:
        inventory = json.loads(zipped.read("BACKUP.json"))["files"]
        require(name in inventory, f"Uninventoried report: {name}")
        raw = zipped.read(name)
        item = inventory[name]
        require(len(raw) == item["size_bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "Report integrity failed")
    key = stage + "/" + name
    sources[key] = {"archive": archive.relative_to(ROOT).as_posix(), "archive_sha256": expected,
                    "member": name, "sha256": item["sha256"]}
    return json.loads(raw), key


def load_data():
    sources, reports = {}, {}
    run_ids = {"probes": "followup-probes-20261102-01", "dynamics": "dynamics-20261102-01",
               "recorded-feedback": "recorded-feedback-20261102-01"}
    for stage, run_id in run_ids.items():
        for model in MODELS:
            name = f"runs/{run_id}/{model}/{stage}/report.json"
            report, source = authenticated_member(stage, name, sources)
            require(report["schema"] == "wm_audit.diagnostic.v1" and report["confirmatory"] is False, "Discovery schema/status changed")
            reports[stage, model] = (report, source)
    numerical = {}
    for model in MODELS:
        stage = "numerical-" + model
        report, source = authenticated_member(stage, stage + "/report.json", sources)
        require(report["schema"] == "wm_audit.execution_layout.v1" and report["model_unchanged"] and report["backend_restored"], "Numerical qualification missing")
        numerical[model] = (report, source)
    joint, source = authenticated_member("replication", "replication/replication-results.json", sources)
    require(sources[source]["sha256"] == JOINT_SHA, "Joint report changed")
    require(joint["schema"] == "wm_audit.replication.analysis.v1" and len(joint["root_ids"]) == 128, "Replication incomplete")
    family = joint["primary_family"]
    require([(r["model"], r["metric"]) for r in family["claims"]] == PRIMARY and family["individual_confidence"] == .99, "Primary family changed")
    require(joint["decisions"]["all_five_supported"] is False, "Do not replace the inconclusive result")
    return reports, numerical, joint, source, sources


def interval_text(stat, confidence="ci95", digits=3):
    a, b = stat[confidence]
    return f"{stat['estimate']:.{digits}f} [{a:.{digits}f}, {b:.{digits}f}]"


def build_assets(output, overwrite=False):
    if output.exists() and not overwrite:
        raise FileExistsError("Generated directory exists; explicit --overwrite required")
    reports, numerical, joint, joint_source, sources = load_data()
    output.mkdir(parents=True, exist_ok=True)
    displays = []

    def select(report, source, field, selector, metric, confidence="ci95"):
        index, row = one(report[field], **selector)
        stat = row["metrics"][metric]
        require(stat["clusters"] == 128 and len(stat[confidence]) == 2, "Interval/cluster mismatch")
        displays.append({"source": source, "pointer": f"/{field}/{index}/metrics/{metric}",
                         "estimate": stat["estimate"], confidence: stat[confidence]})
        return stat

    def fresh(model, metric, appearance="original", horizon=8):
        field = f"exploratory_pointwise95/{model}"
        wrapper = {field: joint["exploratory_pointwise95"][model]}
        return select(wrapper, joint_source, field, {"appearance": appearance, "horizon": horizon}, metric)

    def discovery(stage, model, metric, **selector):
        report, source = reports[stage, model]
        return select(report, source, "summary", {"appearance": "original", **selector}, metric)

    def write(name, lines):
        formats = {
            "probes_rows.tex": ("llrrr", r"Model & Probe & Block & Angle & Effect angle \\"),
            "feedback_rows.tex": ("lll", r"Sample & Model & MSE contrast [95\% CI] \\"),
            "primary_rows.tex": ("llll", r"Model & Contrast & Estimate [99\% CI] & Prespecified decision \\"),
            "ranking_rows.tex": ("lrrl", r"Model & Endpoint & Forecast & Forecast--hold [95\%] \\"),
            "numerical_rows.tex": ("lrr", r"Comparison & LeWM & DINO \\"),
            "slice_rows.tex": ("lllrrr", r"Model & Appearance & $h$ & Oracle--local persistence & Effect--zero & Forecast--hold \\"),
        }
        columns, header = formats[name]
        body = [r"\begin{tabular}{" + columns + "}", r"\toprule", header, r"\midrule",
                *lines, r"\bottomrule", r"\end{tabular}"]
        (output / name).write_text("% Generated from authenticated reports; do not edit.\n" + "\n".join(body) + "\n")

    probe_rows = []
    for model in MODELS:
        for method, label in [("ridge_group_selected", "Ridge"), ("mlp_seed11", "MLP 11"),
                              ("mlp_seed29", "MLP 29"), ("mlp_seed47", "MLP 47")]:
            values = [discovery("probes", model, m, surface="horizon_8", method=method)
                      for m in ("block_distance", "angle_mae_deg", "paired_angle_effect_mae_deg")]
            probe_rows.append(f"{LABELS[model]} & {label} & " + " & ".join(f"{s['estimate']:.2f}" for s in values) + r" \\")
    zero = discovery("probes", "lewm", "paired_angle_effect_mae_deg", surface="horizon_8", method="train_mean")
    probe_rows += [r"\midrule", f"Zero-change & -- & -- & -- & {zero['estimate']:.2f}" + r" \\"]
    write("probes_rows.tex", probe_rows)

    feedback = []
    for domain, stage in [("Discovery", "dynamics"), ("Recorded", "recorded-feedback")]:
        for model in MODELS:
            suffix = "none" if model == "lewm" else "recurrent"
            stat = discovery(stage, model, "actual_" + suffix + "_minus_baseline", horizon=8)
            feedback.append(f"{domain} & {LABELS[model]} & {interval_text(stat, digits=4)}" + r" \\")
    for model in MODELS:
        suffix = "none" if model == "lewm" else "recurrent"
        feedback.append(f"Fresh roots & {LABELS[model]} & {interval_text(fresh(model, 'actual_' + suffix + '_minus_baseline'), digits=4)}" + r" \\")
    write("feedback_rows.tex", feedback)

    primary_rows = []
    for index, stat in enumerate(joint["primary_family"]["claims"]):
        displays.append({"source": joint_source, "pointer": f"/primary_family/claims/{index}",
                         "estimate": stat["estimate"], "ci99": stat["ci99"]})
        label = {"effect_error_minus_zero": "Effect vs. zero", "matched_minus_all_wrong": "Matched vs. wrong",
                 "forecast_minus_endpoint_regret": "Forecast vs. endpoint"}[stat["metric"]]
        supported = stat["ci99"][1] < 0
        decision = "Supported" if supported else r"\textbf{Inconclusive}"
        primary_rows.append(f"{LABELS[stat['model']]} & {label} & {interval_text(stat, 'ci99', 4)} & {decision}" + r" \\")
    write("primary_rows.tex", primary_rows)

    ranking = []
    for model in MODELS:
        values = [fresh(model, m) for m in ("endpoint_regret", "forecast_regret", "forecast_minus_hold", "best_coverage")]
        ranking.append(f"{LABELS[model]} & " + " & ".join(f"{v['estimate']:.4f}" for v in values[:2])
                       + " & " + interval_text(values[2], digits=4) + r" \\")
    write("ranking_rows.tex", ranking)

    numerical_rows = []
    labels = [("baseline", "observed_paths", "Observed paths"),
              ("baseline", "encoding_at_single", "Encoding only (single)"),
              ("baseline", "rollout_at_new_encoding", "Rollout layout (new encoding)"),
              ("baseline", "feedback_vs_public", "Wrapper vs. public"),
              ("baseline", "metric_reduction_only", "Metric reduction only"),
              ("cudnn_conv_ieee", "observed_paths", "Observed paths, conv. IEEE")]
    numeric_stats = {}
    for control, comparison, label in labels:
        vals = []
        for model in MODELS:
            report, source = numerical[model]
            index, row = one(report["summary"], control=control, comparison=comparison)
            val = row["metrics"]["native_f32"]["max_abs_delta"]
            displays.append({"source": source, "pointer": f"/summary/{index}/metrics/native_f32/max_abs_delta", "value": val})
            numeric_stats[model, control, comparison] = val
            vals.append("0" if val == 0 else f"{val:.2e}")
        numerical_rows.append(label + " & " + " & ".join(vals) + r" \\")
    write("numerical_rows.tex", numerical_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"lewm": "#0072B2", "dinowm": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1, 1, 1.13]})
    for ax, model in zip(axes[:2], MODELS):
        for y, metric in enumerate(("effect_error_minus_zero", "matched_minus_all_wrong")):
            _, s = one(joint["primary_family"]["claims"], model=model, metric=metric)
            lo, hi = s["ci99"]
            ax.plot([lo, hi], [y, y], color=colors[model], lw=2)
            ax.plot(s["estimate"], y, "o", color=colors[model], ms=6)
        ax.axvline(0, color="#555555", lw=.8, ls="--")
        ax.set_yticks([0, 1], ["Effect vs. zero", "Matched vs. wrong"])
        ax.set_ylim(1.6, -.6)
        ax.set_title(LABELS[model])
        ax.set_xlabel("Latent-MSE contrast (99% CI)")
        ax.grid(axis="x", alpha=.2)
    ax = axes[2]
    _, s = one(joint["primary_family"]["claims"], model="dinowm", metric="forecast_minus_endpoint_regret")
    ax.plot(s["ci99"], [0, 0], color=colors["dinowm"], lw=2)
    ax.plot(s["estimate"], 0, "o", color=colors["dinowm"], ms=6)
    ax.axvline(0, color="#555555", lw=.8, ls="--")
    ax.set_yticks([])
    ax.set_title("DINO native-goal ranking")
    ax.set_xlabel("Forecast minus endpoint regret")
    ax.text(.5, .2, "Inconclusive", transform=ax.transAxes, ha="center", color="#444444")
    ax.set_ylim(-1, 1)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout(pad=.8, w_pad=1.3)
    fig.savefig(output / "replication_primary.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "replication_primary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    appearances = ("original", "color_only", "background_only")
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.1), sharex=True)
    for row, model in enumerate(MODELS):
        for col, appearance in enumerate(appearances):
            ax = axes[row, col]
            for metric, marker, style, label in [("effect_error_minus_zero", "o", "-", "Effect vs. zero"),
                                                 ("matched_minus_all_wrong", "s", "--", "Matched vs. wrong")]:
                values = [fresh(model, metric, appearance, h) for h in range(1, 9)]
                x = np.arange(1, 9)
                ax.plot(x, [s["estimate"] for s in values], marker=marker, linestyle=style,
                        color=colors[model], ms=3, lw=1, label=label)
                ax.fill_between(x, [s["ci95"][0] for s in values], [s["ci95"][1] for s in values],
                                color=colors[model], alpha=.10 if marker == "o" else .22)
            ax.axhline(0, color="#555555", lw=.8)
            ax.set_title(LABELS[model] + ": " + appearance.replace("_only", "").replace("_", " "))
            ax.set_xticks([1, 4, 8])
            ax.grid(alpha=.15)
            if col == 0:
                ax.set_ylabel("Latent-MSE contrast")
            if row == 1:
                ax.set_xlabel("Future horizon")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(pad=.8)
    fig.savefig(output / "replication_slices.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "replication_slices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # A concise appendix records all horizons/appearances without selecting wins.
    slice_rows = []
    for model in MODELS:
        for appearance in appearances:
            for h in (1, 4, 8):
                suffix = "none" if model == "lewm" else "recurrent"
                stats = [fresh(model, m, appearance, h) for m in
                         ("actual_" + suffix + "_minus_baseline", "effect_error_minus_zero", "forecast_minus_hold")]
                slice_rows.append(f"{LABELS[model]} & {appearance.replace('_only','').replace('_',' ')} & {h} & "
                                  + " & ".join(interval_text(s, digits=4) for s in stats) + r" \\")
    write("slice_rows.tex", slice_rows)
    prior_figure = OLD / "figures/followup_summary.pdf"
    prior_manifest = json.loads((OLD / "figures/followup_summary.sources.json").read_text())
    # The new manuscript's manifest pins the exact prior PDF and source manifest;
    # the existing generator independently authenticates its six report sources.
    require(prior_manifest["schema"] == "wm_audit.followup_summary.v1", "Prior figure manifest schema changed")
    shutil.copyfile(prior_figure, output / "discovery_summary.pdf")
    manifest = {"schema": "wm_audit.paper_assets.v1", "source_reports": sources,
                "displayed_statistics": displays, "no_model_execution": True,
                "primary_confidence": .99, "secondary_confidence": .95,
                "all_mlp_seeds": [11, 29, 47], "primary_family": PRIMARY,
                "prior_figure_sha256": sha(prior_figure),
                "prior_figure_manifest_sha256": sha(OLD / "figures/followup_summary.sources.json"),
                "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.suffix in (".tex", ".pdf", ".png")}}
    (output / "assets.sources.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Paper assets: {len(displays)} source-linked statistics from {len(sources)} reports; no new experiments")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "generated")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_assets(args.output, args.overwrite)