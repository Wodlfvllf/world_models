"""Plot the authenticated 2026-11-02 follow-ups; JSON + CPU/Agg only.

No wm_audit/model imports, checkpoint loading, fitting, or new bootstraps.
Published intervals are copied, not reconstructed from marginal intervals.
The archive pins are the existing retrieval receipts, not digital signatures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from zipfile import ZipFile, BadZipFile


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runs/molab-20261102-retrieved"
MODELS = {"lewm": "LeWM", "dinowm": "DINO-WM (Point)"}
PROBES = ("ridge_group_selected", "mlp_seed11", "mlp_seed29", "mlp_seed47")
REGRETS = ("endpoint_regret", "forecast_regret", "hold_regret", "uniform_regret")
COUNTS = {"scorer": 128, "validation": 64, "pilot": 128, "editor": 256, "confirmation": 0}
BUNDLES = {
    "probes": {
        "archive": "followup-probes-20261102-verified.zip",
        "sha256": "678ed416c59dfb383b0b64b92de234a82d99b872187bb2ac3514f37d74eb1fe7",
        "run": "followup-probes-20261102-01", "kind": "followup_probes",
    },
    "dynamics": {
        "archive": "dynamics-20261102-verified.zip",
        "sha256": "538e62131fcf4f4139cfb86fa8abea4817297e18a9e02d76bbb08805208482d9",
        "run": "dynamics-20261102-01", "kind": "followup_dynamics",
    },
    "recorded-feedback": {
        "archive": "recorded-feedback-20261102-verified.zip",
        "sha256": "71892b690d0d365f2e56483af28bd6e97a24fcee9deb0db0b6166c09a9f1f227",
        "run": "recorded-feedback-20261102-01", "kind": "followup_recorded_feedback",
    },
}
OUTPUTS = ("followup_summary.pdf", "followup_summary.png", "followup_summary.sources.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "JSON document must be an object")
    return value


def count(value, expected, label):
    require(type(value) is int and value == expected, f"{label}: expected integer {expected}")


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Expected a finite, non-boolean number")
    return value


def matches(row, selector):
    return all(type(row.get(k)) is type(v) and row[k] == v for k, v in selector.items())


def interval(report, index, metric):
    stat = report["summary"][index]["metrics"][metric]
    estimate, ci = number(stat["estimate"]), stat["ci95"]
    require(isinstance(ci, list) and len(ci) == 2, f"{metric}: need a two-sided ci95")
    low, high = map(number, ci)
    require(low <= high, f"{metric}: reversed ci95")
    count(stat["clusters"], 128, f"{metric} clusters")
    return {"estimate": estimate, "ci95": [low, high], "clusters": 128,
            "json_pointer": f"/summary/{index}/metrics/{metric}"}


def unique_roster(ids, expected, label):
    require(isinstance(ids, list) and all(isinstance(x, str) and x for x in ids), f"Invalid {label}")
    require(len(ids) == len(set(ids)) == expected, f"{label}: expected {expected} unique entries")
    return set(ids)


def _select_report(report, stage, model):
    require(report["schema"] == "wm_audit.diagnostic.v1", "Unsupported report schema")
    require(report["kind"] == BUNDLES[stage]["kind"], "Report kind mismatch")
    require(report["model"]["name"] == model, "Report model mismatch")
    spec, config, protocol = report["spec"], report["config"], report["protocol"]
    require(spec["schema"] == "wm_audit.followup.v1", "Unsupported follow-up schema")
    require(report["confirmatory"] is False and spec["confirmatory"] is False,
            "Report and specification must be explicitly confirmatory=false")
    selection = report.get("selection", {})
    for container in (protocol, selection.get("spec", {})):
        if "confirmatory" in container:
            require(container["confirmatory"] is False, "Nested confirmatory must be false")
    count(spec["primary_horizon"], 8, "primary horizon")
    count(spec["bootstrap_resamples"], 5000, "bootstrap resamples")
    count(config["horizon"], 8, "config horizon")
    count(config["candidates"], 8, "candidate bank")
    for field in (config["counts"], protocol.get("canonical_roots", {})):
        require(isinstance(field, dict), "Roster counts must be an object")
        for key, expected in COUNTS.items():
            if key in field:
                count(field[key], expected, f"{key} roster")
    for key, expected in (("training_roots", 128), ("validation_roots", 64), ("evaluation_roots", 128)):
        if key in report:
            count(report[key], expected, key)
    require(isinstance(protocol["uncertainty"], str), "Missing uncertainty description")
    for key in ("summary", "records"):
        require(isinstance(report[key], list) and all(isinstance(r, dict) for r in report[key]),
                f"{key} must be a list of objects")
    domain = "recorded" if stage == "recorded-feedback" else "generated"
    selector = {"appearance": "original", **({"surface": "horizon_8"} if stage == "probes" else {"horizon": 8})}
    records = [r for r in report["records"] if matches(r, selector)]
    rosters, panels = {}, {}

    def row_index(**extra):
        found = [i for i, row in enumerate(report["summary"]) if matches(row, {**selector, **extra})]
        require(len(found) == 1, f"Need exactly one original h8 summary: {extra}")
        return found[0]

    def point(index, metric, *, bounded=False, **extra):
        value = interval(report, index, metric)
        if bounded:
            require(all(0 <= v <= 1 for v in (value["estimate"], *value["ci95"])), f"Invalid {metric} range")
        return {"source_id": f"{stage}/{model}", "model": model, "domain": domain,
                "appearance": "original", "horizon": 8, "metric": metric, **value, **extra}

    if stage == "probes":
        require(report["followup_schema"] == "wm_audit.followup.probes.v1", "Unsupported probe schema")
        require(spec["mlp_seeds"] == [11, 29, 47] and all(type(s) is int for s in spec["mlp_seeds"]),
                "Require all three predeclared MLP seeds, not a best seed")
        methods = [*PROBES, "train_mean"]
        require(sorted(report["methods"]) == sorted(methods), "Incomplete or duplicate probe method roster")
        for field, split in (("fit_root_ids", "scorer"), ("validation_root_ids", "validation")):
            if field in selection:
                rosters[split] = unique_roster(selection[field], COUNTS[split], field)
        panels["physical_block_error"] = []
        for method in methods:
            index = row_index(method=method)
            row = report["summary"][index]
            for key, expected in (("roots", 128), ("samples", 2048), ("pairs", 1024)):
                if key in row:
                    count(row[key], expected, f"h8 {key}")
            subset = [r for r in records if r["method"] == method]
            roots = unique_roster([r["root_id"] for r in subset], 128, f"{method} pilot roster")
            require("pilot" not in rosters or roots == rosters["pilot"], "Probe methods have different pilot rosters")
            rosters["pilot"] = roots
            if method in PROBES:
                value = point(index, "block_distance", method=method, units="native simulator units")
                require(min(value["estimate"], *value["ci95"]) >= 0, "Negative physical distance")
                panels["physical_block_error"].append(value)
        for split in ("scorer", "validation"):
            require(not (rosters.get(split, set()) & rosters["pilot"]), "Training/validation overlaps pilot")
        require(not (rosters.get("scorer", set()) & rosters.get("validation", set())), "Fit/validation overlap")
    else:
        index = row_index()
        if stage == "dynamics":
            keys = [(r["cluster"], r["branch"]) for r in records]
            require(all(isinstance(root, str) and root and type(branch) is int and branch in (0, 1)
                        for root, branch in keys), "Invalid root/branch roster")
            roots = {root for root, _ in keys}
            require(len(roots) == 128 and len(keys) == len(set(keys)) == 256
                    and set(keys) == {(root, branch) for root in roots for branch in (0, 1)},
                    "Require 128 complete pilot roots / 256 root-branch banks")
            rosters["pilot"] = roots
        else:
            windows = report["windows"]
            require(isinstance(windows, list) and len(windows) == 128, "Require 128 recorded windows")
            require(all(set(w) == {"episode", "start"} and all(type(w[k]) is int and w[k] >= 0 for k in w)
                        for w in windows), "Invalid recorded episode/start roster")
            unique_roster([str(w["episode"]) for w in windows], 128, "recorded episodes")
            expected = {(str(w["episode"]), w["start"]) for w in windows}
            observed = [(r["cluster"], r["start"]) for r in records]
            require(len(observed) == 128 and set(observed) == expected, "Recorded h8 roster differs from windows")
            rosters["episode_windows"] = expected
        actual = "actual_none" if model == "lewm" else "actual_recurrent"
        value = point(index, actual + "_minus_baseline", units="checkpoint latent MSE difference")
        value["components"] = {key: interval(report, index, key) for key in (actual, "local_persistence")}
        difference = value["components"][actual]["estimate"] - value["components"]["local_persistence"]["estimate"]
        require(math.isclose(value["estimate"], difference, rel_tol=1e-5, abs_tol=2e-7),
                "Actual-history contrast is not against LOCAL persistence")
        panels["feedback_minus_local"] = [value]
        if stage == "dynamics":
            value = point(index, "effect_error_minus_zero", units="checkpoint latent effect-MSE difference")
            value["components"] = {key: interval(report, index, key) for key in ("effect_error", "true_effect_energy")}
            difference = value["components"]["effect_error"]["estimate"] - value["components"]["true_effect_energy"]["estimate"]
            require(math.isclose(value["estimate"], difference, rel_tol=1e-5, abs_tol=2e-7), "Wrong zero-effect baseline")
            panels["action_effect_minus_zero"] = [value]
            panels["regret"] = [point(index, metric, bounded=True, units="coverage-cost regret") for metric in REGRETS]
            panels["bank_best_coverage"] = [point(index, "best_coverage", bounded=True, units="coverage fraction")]
    return panels, rosters


def select_report(report, stage, model):
    """Validate the schema and complete primary-slice rosters before selection."""
    try:
        return _select_report(report, stage, model)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"{stage}/{model}: malformed report schema ({exc})") from exc


def load_summary(root):
    """Rehash each archive, its inventory, and the six local report byte streams."""
    root = Path(root).resolve()
    result = {"schema": "wm_audit.followup_summary.v1", "confirmatory": False,
              "input_root": str(root), "appearance": "original", "horizon": 8,
              "generated_roots": 128, "recorded_episodes": 128, "bootstrap_resamples": 5000,
              "intervals": "Reported pointwise 95% equal-root/episode bootstrap intervals; not simultaneous",
              "selection": "Group-selected ridge and all MLP seeds 11, 29, 47; no best-seed selection",
              "authentication_scope": "Pinned retrieval SHA256 -> ZIP BACKUP.json -> six report hashes/sizes; not a new verification of every extracted file",
              "archives": [], "reports": [], "selected_values": {}}
    seen_rosters = {}
    for i, (stage, bundle) in enumerate(BUNDLES.items(), 1):
        print(f"[{i}/3] Authenticate {stage} archive and two JSON reports (CPU only)", flush=True)
        archive = root / bundle["archive"]
        receipt = Path(str(archive) + ".sha256")
        require(receipt.read_text().split() == [bundle["sha256"], archive.name], "Archive receipt disagrees with pinned retrieval")
        require(sha256(archive) == bundle["sha256"], f"Archive SHA256 mismatch: {archive.name}")
        inventory_bytes = (root / stage / "BACKUP.json").read_bytes()
        with ZipFile(archive) as zipped:
            require(len(zipped.namelist()) == len(set(zipped.namelist())), "Duplicate ZIP members")
            require(zipped.read("BACKUP.json") == inventory_bytes, "Local inventory differs from authenticated archive")
        inventory = read_json(inventory_bytes)
        require(inventory["schema"] == "wm_audit.backup.v1" and isinstance(inventory["files"], dict), "Invalid backup schema")
        restore_path = root / stage / "RESTORE-VERIFIED.json"
        restore = read_json(restore_path.read_bytes())
        require(restore["archive_sha256"] == bundle["sha256"], "Restore receipt archive mismatch")
        count(restore["files_verified"], len(inventory["files"]), "Restore inventory count")
        result["archives"].append({"path": archive.name, "sha256": bundle["sha256"],
                                   "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                                   "receipt_sha256": sha256(receipt), "restore_receipt_sha256": sha256(restore_path),
                                   "inventory_files": len(inventory["files"]), "source_commit": inventory.get("source_commit")})
        for model in MODELS:
            relative = f"runs/{bundle['run']}/{model}/{stage}/report.json"
            path = root / stage / relative
            raw, expected = path.read_bytes(), inventory["files"][relative]
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == expected["sha256"], f"Report SHA256 mismatch: {stage}/{model}")
            count(expected["size_bytes"], len(raw), "Report size")
            report = read_json(raw)
            panels, rosters = select_report(report, stage, model)
            for key, roster in rosters.items():
                require(key not in seen_rosters or roster == seen_rosters[key], f"Cross-report {key} roster mismatch")
                seen_rosters[key] = roster
            result["reports"].append({"source_id": f"{stage}/{model}", "path": f"{stage}/{relative}",
                                      "sha256": digest, "size_bytes": len(raw), "schema": report["schema"],
                                      "kind": report["kind"], "model": model, "confirmatory": False,
                                      "roster_counts": {key: len(value) for key, value in rosters.items()},
                                      "uncertainty": report["protocol"]["uncertainty"]})
            for panel, values in panels.items():
                result["selected_values"].setdefault(panel, []).extend(values)
            del report, raw
    coverage = result["selected_values"]["bank_best_coverage"]
    require(coverage[0]["estimate"] == coverage[1]["estimate"], "Models do not share the same bank coverage")
    result["bank_best_coverage_label"] = f"{100 * coverage[0]['estimate']:.2f}%"
    return result


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    colors, markers = {"lewm": "#0072B2", "dinowm": "#D55E00"}, {"lewm": "o", "dinowm": "s"}
    values = data["selected_values"]
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
             "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#888888",
             "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12.8, 9.6))
        grid = fig.add_gridspec(2, 2, left=.115, right=.97, top=.82, bottom=.20, wspace=.48, hspace=.83)
        fig.suptitle("Frozen world-model follow-ups | exploratory", x=.115, y=.975, ha="left", fontsize=18, weight="bold")
        fig.text(.115, .936, "Original appearance · horizon 8 · 128 generated roots / 128 recorded episodes", fontsize=11, color="#444444")

        def heading(slot, title, subtitle=None):
            box = slot.get_position(fig)
            fig.text(box.x0, box.y1 + .071, title, fontsize=11, weight="bold")
            if subtitle:
                fig.text(box.x0, box.y1 + .045, subtitle, fontsize=8.5, color="#555555")

        def base(ax):
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e5e5e5", linewidth=.6)
            ax.tick_params(axis="both", length=0, pad=6, labelsize=9)

        def whisker(ax, point, y, label=None):
            low, high = point["ci95"]
            color, marker = colors[point["model"]], markers[point["model"]]
            ax.hlines(y, low, high, color=color, linewidth=1.6)
            ax.vlines([low, high], y - .045, y + .045, color=color, linewidth=1)
            ax.plot(point["estimate"], y, marker=marker, color=color, markersize=5, label=label, linestyle="none")

        def grouped(slot, points, keys, field, labels, xlabel, legend=False):
            ax = fig.add_subplot(slot)
            base(ax)
            for m, model in enumerate(MODELS):
                for j, key in enumerate(keys):
                    point = next(p for p in points if p["model"] == model and p[field] == key)
                    whisker(ax, point, len(keys) - 1 - j + (.13 if m == 0 else -.13), MODELS[model] if j == 0 else None)
            ax.set_yticks(range(len(keys)), labels[::-1])
            ax.set_ylim(-.5, len(keys) - .5)
            ax.set_xlim(0, max(p["ci95"][1] for p in points) * 1.10)
            ax.set_xlabel(xlabel, labelpad=10)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if legend:
                ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.035), ncol=2, fontsize=9, borderaxespad=0)
            return ax

        def contrast(slot, point, title):
            ax = fig.add_subplot(slot)
            base(ax)
            whisker(ax, point, 0)
            low, high = min(0, point["ci95"][0], point["estimate"]), max(0, point["ci95"][1], point["estimate"])
            padding = max(high - low, 1e-6) * .10
            ax.set_xlim(low - padding, high + padding)
            ax.set_ylim(-.6, .9)
            ax.axvline(0, color="#777777", linestyle="--", linewidth=.8)
            ax.set_yticks([])
            ax.set_title(title, loc="left", fontsize=9, pad=13)
            ax.text(0, .91, f"{point['estimate']:+.4f} [{point['ci95'][0]:+.4f}, {point['ci95'][1]:+.4f}]",
                    transform=ax.transAxes, fontsize=8, va="top")
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            return ax

        heading(grid[0, 0], "A  Physical block-position error")
        grouped(grid[0, 0], values["physical_block_error"], PROBES, "method",
                ["Ridge\n(group-selected)", "MLP seed 11", "MLP seed 29", "MLP seed 47"],
                "Mean error (native units; 512-unit arena)", legend=True)
        heading(grid[0, 1], "B  Actual history − LOCAL persistence", "Own axes per model/domain; negative favors actual history")
        feedback = grid[0, 1].subgridspec(2, 2, hspace=1.25, wspace=.50)
        for m, model in enumerate(MODELS):
            for d, domain in enumerate(("generated", "recorded")):
                point = next(p for p in values["feedback_minus_local"] if p["model"] == model and p["domain"] == domain)
                contrast(feedback[d, m], point, f"{MODELS[model]} · {domain}")
        heading(grid[1, 0], "C  Action-effect error − zero-effect baseline", "Own axes per model; negative favors predicted action effects")
        effects = grid[1, 0].subgridspec(2, 1, hspace=1.25)
        for m, model in enumerate(MODELS):
            point = next(p for p in values["action_effect_minus_zero"] if p["model"] == model)
            contrast(effects[m], point, MODELS[model] + " · generated")
        heading(grid[1, 1], "D  Finite-bank candidate regret",
                f"Mean bank best coverage: {data['bank_best_coverage_label']} · 8 candidates")
        grouped(grid[1, 1], values["regret"], REGRETS, "metric",
                ["True endpoint", "Forecast", "Hold", "Expected uniform"], "Mean coverage-cost regret (lower is better)")
        footnotes = [
            "Points and whiskers: pointwise 95% bootstrap intervals (5,000 resamples), equal-root / episode weighting. All three MLP seeds retained.",
            "B/C: checkpoint-specific latent MSE differences; raw scales cannot rank models. Actual history is privileged, not deployable prediction.",
            "Recorded data are pretraining-distribution diagnostics, not held-out generalization. Finite-bank regret is not closed-loop control success.",
        ]
        for y, text in zip((.112, .085, .058), footnotes):
            fig.text(.115, y, text, fontsize=8.5, color="#444444")
    return fig


def check_outputs(output, overwrite):
    paths = [Path(output) / name for name in OUTPUTS]
    if not overwrite:
        require_new = [str(p) for p in paths if p.exists() or p.is_symlink()]
        if require_new:
            raise FileExistsError("Refusing to overwrite; use --overwrite explicitly: " + ", ".join(require_new))
    return paths


def write_summary(data, output, *, overwrite=False):
    paths = check_outputs(output, overwrite)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig = make_figure(data)
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        with tempfile.TemporaryDirectory(prefix=".followup-plot-", dir=output) as temp:
            staged = [Path(temp) / name for name in OUTPUTS]
            # PDF font embedding is consulted at save time, after figure styling.
            with matplotlib.rc_context({"pdf.fonttype": 42}):
                fig.savefig(staged[0], format="pdf", bbox_inches="tight",
                            metadata={"Title": "Exploratory frozen world-model follow-ups", "CreationDate": None, "ModDate": None})
            fig.savefig(staged[1], format="png", dpi=300, bbox_inches="tight")
            provenance = {**data, "plotter_sha256": sha256(Path(__file__)), "python": platform.python_version(),
                          "matplotlib": matplotlib.__version__, "backend": matplotlib.get_backend(),
                          "execution": "CPU JSON plotting only; no models, fitting, experiment reruns, or resampling",
                          "outputs": [{"path": p.name, "sha256": sha256(p), "size_bytes": p.stat().st_size} for p in staged[:2]]}
            staged[2].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            # Stage all files first; exclusive hard links prevent even a late overwrite race.
            published = []
            try:
                for source, target in zip(staged, paths):
                    if overwrite:
                        os.replace(source, target)
                    else:
                        os.link(source, target)
                        published.append(target)
            except OSError:
                for path in published:
                    path.unlink()
                raise
    finally:
        plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: INPUT_ROOT/figures")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing figures and source provenance")
    args = parser.parse_args(argv)
    output = args.output_dir or args.input_root / "figures"
    try:
        check_outputs(output, args.overwrite)
        data = load_summary(args.input_root)
        print("Validated 6 reports; rendering PDF, PNG and source provenance (CPU/Agg)", flush=True)
        paths = write_summary(data, output, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()