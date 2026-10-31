"""NumPy-only, paired analysis of the physical-state-edit audit.

No fitting, method selection, outcome filtering, or simulator/model execution
occurs here. Analyze ONE split, checkpoint, candidate-bank definition and fitted
editor/scorer seed roster per call. Repeated candidates/horizons/fitting seeds
must already be averaged within each root/appearance/method record. Every root
must have the exact 3-appearance x 9-method roster, including failed native
representations and tied candidate banks. A missing/nonfinite record raises;
it is never silently removed or replaced by a data-dependent failure penalty.

``analyze_records`` returns plain JSON-compatible values. Each numerical summary
has ``estimate``, ``ci95`` and ``ci90`` (two-element [lower, upper] lists). Top-level
``means`` reports all methods; ``contrasts[metric][method]`` contains paired
``minus_native``, ``transport`` and ``deterioration`` summaries. ``native``,
``readout`` and ``decision`` contain parallel qualification diagnostics, not
inclusion gates. ``conclusions[condition]`` refers ONLY to ``primary_method``.
``shifted`` always means an equal within-root average of color/background shifts.

Intervals are pointwise percentile cluster-bootstrap intervals, conditional on
the fitted models/seeds and, if reused, the appearance roster. They are NOT
simultaneous intervals or automatic confirmatory tests. Equivalence uses a
strictly contained TWO-SIDED 90% interval (the nominal 5% TOST inclusion rule,
with approximate bootstrap coverage), not failure to reject a difference.
There are no analytic TOST p-values or multiplicity adjustments in this module.
"""

from __future__ import annotations

from numbers import Integral, Real
import warnings

import numpy as np


__all__ = ["analyze_records", "physical_effect_loss", "circular_error_deg", "candidate_regret"]

APPEARANCES = ("original", "color_only", "background_only")
METHODS = ("native", "identity", "ridge", "inverse", "mlp", "rollout", "random", "wrong", "zero")
EDITORS = ("ridge", "inverse", "mlp", "rollout")
METRICS = ("effect_loss", "absolute_loss", "angle_error_deg", "regret",
           "endpoint_oracle_regret", "zero_effect_loss", "offset_effect_loss", "readout_loss")
CONDITIONS = (*APPEARANCES, "shifted")
SHIFTS = (*APPEARANCES[1:], "shifted")


def _real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    # NumPy otherwise turns mixed Python [True, 2] into indistinguishable ints.
    mixed_bool = (not isinstance(value, np.ndarray)
                  and any(isinstance(item, (bool, np.bool_))
                          for item in np.asarray(value, dtype=object).flat))
    if array.dtype.kind not in "iuf" or mixed_bool:
        raise ValueError(f"{name} must contain real numbers, not booleans, strings or complex values.")
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = array.astype(np.float64, copy=False)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as finite float64 values.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers.")
    return array


def _scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar, not a boolean or numeric string.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite in float64.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, not a boolean.")
    return int(value)


def _numeric_result(value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("The numerical result is not finite; no observations were discarded.")
    return float(value) if value.ndim == 0 else value


def physical_effect_loss(pred_factual, pred_cf, true_factual, true_cf, scales, horizons):
    """Mean of SUMMED squared, scaled effect errors over candidates/horizons.

    All four arrays have identical shape ``(..., K, T, D)``; no broadcasting
    between branches is permitted. ``scales`` is a positive vector of D fixed
    physical-unit DIVISORS (W = diag(1/scales)), not squared-error multipliers.
    ``horizons`` is a nonempty collection of distinct 1-indexed future steps in
    [1, T]; the arrays contain future steps only, not the context. K and the
    selected horizons have equal weights. Features are summed, NOT averaged.

    Returns a float for [K,T,D], otherwise an array with the leading batch shape.
    Common factual/counterfactual bias cancels by design; absolute error must be
    reported separately. Zero realized effects are retained without division by
    their magnitude. Scales/horizons must be frozen before confirmation.
    """
    arrays = [_real_array(value, name) for value, name in zip(
        (pred_factual, pred_cf, true_factual, true_cf),
        ("pred_factual", "pred_cf", "true_factual", "true_cf"),
    )]
    shape = arrays[0].shape
    if len(shape) < 3 or any(size == 0 for size in shape) or any(a.shape != shape for a in arrays):
        raise ValueError("All branches must have identical, nonempty (..., K, T, D) shapes.")
    scales = _real_array(scales, "scales")
    if scales.shape != (shape[-1],) or np.any(scales <= 0):
        raise ValueError("scales must be a positive vector with one divisor per feature.")
    steps = np.asarray(horizons)
    if (steps.ndim != 1 or steps.size == 0 or steps.dtype.kind not in "iu"
            or any(isinstance(step, (bool, np.bool_)) for step in horizons)
            or np.any(steps < 1) or np.any(steps > shape[-2])
            or np.unique(steps).size != steps.size):
        raise ValueError("horizons must be distinct 1-indexed integer future steps in [1, T].")
    indices = np.sort(steps).astype(np.intp) - 1
    pf, pc, tf, tc = (a[..., indices, :] for a in arrays)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = ((pc - pf) - (tc - tf)) / scales
            return _numeric_result(np.mean(np.sum(error ** 2, axis=-1), axis=(-2, -1)))
    except FloatingPointError as exc:
        raise ValueError("physical_effect_loss overflowed; no observations were discarded.") from exc


def circular_error_deg(pred_sincos, true_sincos):
    """Absolute circular error in [0,180], for matching ``(..., 2)`` sin/cos pairs.

    Order is (sin(theta), cos(theta)). Nonunit, nonzero pairs are allowed: atan2
    uses their direction. A zero pair is undefined and raises, rather than
    inventing an angle or dropping the measurement. Returns a float or the
    leading-shape array, without averaging samples.
    """
    pred = _real_array(pred_sincos, "pred_sincos")
    true = _real_array(true_sincos, "true_sincos")
    if (pred.ndim < 1 or pred.shape != true.shape or pred.shape[-1] != 2
            or any(size == 0 for size in pred.shape)):
        raise ValueError("Circular inputs must have identical nonempty (..., 2) shapes.")
    if np.any(np.all(pred == 0, axis=-1)) or np.any(np.all(true == 0, axis=-1)):
        raise ValueError("A zero sin/cos pair has undefined orientation.")
    delta = np.arctan2(pred[..., 0], pred[..., 1]) - np.arctan2(true[..., 0], true[..., 1])
    return _numeric_result(np.abs(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta)))))


def candidate_regret(costs, pred_costs):
    """Executed cost of argmin(pred_costs) minus the minimum executed bank cost.

    Both inputs have identical nonempty shape ``(..., K)``. Lower is better;
    costs need not be nonnegative or normalized. Exact predicted-cost ties use
    the LOWEST candidate index (NumPy argmin), with no random jitter or tie
    tolerance. All-tied/failed banks remain in the population. Returns a float
    for one bank, otherwise an array of bank regrets. This is not global-policy
    regret, and a zero regret on an all-tied bank does not demonstrate utility.
    """
    actual = _real_array(costs, "costs")
    predicted = _real_array(pred_costs, "pred_costs")
    if (actual.ndim < 1 or actual.shape != predicted.shape
            or any(size == 0 for size in actual.shape)):
        raise ValueError("Costs must have identical nonempty (..., K) shapes.")
    selected = np.argmin(predicted, axis=-1)
    chosen = np.take_along_axis(actual, np.expand_dims(selected, -1), axis=-1)[..., 0]
    try:
        with np.errstate(over="raise", invalid="raise"):
            return _numeric_result(chosen - np.min(actual, axis=-1))
    except FloatingPointError as exc:
        raise ValueError("candidate_regret overflowed; no banks were discarded.") from exc


def _label(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return str(value)  # Preserve spelling/whitespace; do not merge identities.


def _identifier(value, name: str):
    if isinstance(value, str):
        return _label(value, name)
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError(f"{name} must be a nonempty string or an integer, not a boolean/float/null.")


def _id_order(value):
    return ("int", value) if isinstance(value, int) else ("str", value)


def _validate_records(records: list[dict]):
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list of dictionaries.")
    required = {"root_id", "split", "appearance", "method", *METRICS}
    cells, root_clusters, rosters, splits = {}, {}, {}, set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a dictionary.")
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Record {index} is missing required fields: {sorted(missing)}.")
        root = _identifier(record["root_id"], "root_id")
        splits.add(_label(record["split"], "split"))
        appearance = _label(record["appearance"], "appearance")
        method = _label(record["method"], "method")
        if appearance not in APPEARANCES or method not in METHODS:
            raise ValueError(f"Unknown appearance/method: {appearance!r}/{method!r}.")
        key = (root, appearance, method)
        if key in cells:
            raise ValueError(f"Duplicate root/appearance/method identity: {key!r}.")
        cluster = (("episode_id", _identifier(record["episode_id"], "episode_id"))
                   if "episode_id" in record else ("root_id", root))
        if root in root_clusters and root_clusters[root] != cluster:
            raise ValueError(f"Inconsistent episode_id presence/value for root {root!r}.")
        root_clusters[root] = cluster
        rosters.setdefault(root, set()).add((appearance, method))
        values = tuple(_scalar(record[name], f"record {index} {name}") for name in METRICS)
        if any(value < 0 for value in values):
            raise ValueError(f"Record {index} losses, absolute angle errors and regrets must be nonnegative.")
        if values[METRICS.index("angle_error_deg")] > 180:
            raise ValueError(f"Record {index} angle_error_deg must be in [0, 180].")
        cells[key] = values
    if len(splits) != 1:
        raise ValueError("Analyze exactly one split per call; never pool fitting/validation/confirmation.")
    expected = {(appearance, method) for appearance in APPEARANCES for method in METHODS}
    roots = sorted(rosters, key=_id_order)
    for root in roots:
        missing = expected - rosters[root]
        if missing:
            raise ValueError(f"Root {root!r} is missing roster cells: {sorted(missing)}.")
    data = np.array([[[cells[root, appearance, method] for method in METHODS]
                      for appearance in APPEARANCES] for root in roots], dtype=np.float64)
    # Physical baselines are appearance-independent; real-endpoint diagnostics
    # can depend on appearance but not on the editor whose row carries them.
    for field in ("zero_effect_loss", "offset_effect_loss"):
        values = data[..., METRICS.index(field)]
        if not np.all(values == values[:, :1, :1]):
            raise ValueError(f"{field} must agree exactly across appearances/methods within each root.")
    for field in ("endpoint_oracle_regret", "readout_loss"):
        values = data[..., METRICS.index(field)]
        if not np.all(values == values[:, :, :1]):
            raise ValueError(f"{field} must agree exactly across methods within each root/appearance.")
    cluster_keys = sorted(set(root_clusters.values()), key=lambda c: (c[0], *_id_order(c[1])))
    if len(cluster_keys) < 2:
        raise ValueError("At least 2 independent root/episode clusters are required.")
    lookup = {key: index for index, key in enumerate(cluster_keys)}
    codes = np.array([lookup[root_clusters[root]] for root in roots], dtype=np.intp)
    sizes = np.bincount(codes, minlength=len(cluster_keys))
    clusters = [{"kind": key[0], "id": key[1], "n_roots": int(size)}
                for key, size in zip(cluster_keys, sizes)]
    return data, next(iter(splits)), codes, clusters


class _Summaries:
    """Register root-level contrasts BEFORE applying shared cluster draws."""

    def __init__(self):
        self.values = []
        self.outputs = []

    def add(self, values: np.ndarray) -> dict:
        if not np.isfinite(values).all():
            raise ValueError("A derived contrast is nonfinite; no roots were discarded.")
        output = {}
        self.values.append(values)
        self.outputs.append(output)
        return output

    def finish(self, codes: np.ndarray, n_clusters: int, resamples: int, seed: int) -> None:
        values = np.column_stack(self.values)
        # Column scaling avoids overflowing sums of large but finite losses.
        scales = np.maximum(1.0, np.max(np.abs(values), axis=0))
        values = values / scales
        point = values.mean(axis=0) * scales
        totals = np.zeros((n_clusters, values.shape[1]), dtype=np.float64)
        np.add.at(totals, codes, values)
        sizes = np.bincount(codes, minlength=n_clusters)
        rng = np.random.default_rng(seed)
        samples = np.empty((resamples, values.shape[1]), dtype=np.float64)
        for start in range(0, resamples, 128):
            stop = min(start + 128, resamples)
            draws = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
            counts = np.zeros((stop - start, n_clusters), dtype=np.float64)
            np.add.at(counts, (np.arange(stop - start)[:, None], draws), 1)
            # Ratio of sampled root totals to sampled root counts: NOT an
            # equal-weight mean of episode means when episodes differ in size.
            samples[start:stop] = (counts @ totals) / (counts @ sizes)[:, None]
        quantiles = np.percentile(samples, [2.5, 97.5, 5, 95], axis=0, method="linear") * scales
        if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
            raise ValueError("Bootstrap summaries are not finite; no roots were discarded.")
        for index, output in enumerate(self.outputs):
            output.update(estimate=float(point[index]),
                          ci95=[float(quantiles[0, index]), float(quantiles[1, index])],
                          ci90=[float(quantiles[2, index]), float(quantiles[3, index])])


def _equivalence(summary: dict, margin: float) -> None:
    summary.update(margin=margin,
                   equivalent90=bool(summary["ci90"][0] > -margin and summary["ci90"][1] < margin))


def _conclusions(report: dict) -> dict:
    method = report["primary_method"]
    native, readout, decision = (report[key] for key in ("native", "readout", "decision"))
    output = {}
    for shift in SHIFTS:
        shifts = SHIFTS if shift == "shifted" else (shift,)
        conditions = ("original", *shifts)
        effect = report["contrasts"]["effect_loss"][method]
        regret = report["contrasts"]["regret"][method]
        physical_checks = {
            "positive_transport95": effect["transport"][shift]["ci95"][0] > 0,
            "positive_editor_deterioration95": effect["deterioration"][shift]["ci95"][0] > 0,
            "native_effect_equivalent90": all(native["effect_shift"][c]["equivalent90"] for c in shifts),
            "native_beats_both_baselines95": all(native["accuracy"][c]["beats_both95"] for c in conditions),
            "readout_absolute_valid95": all(readout["loss"][c]["upper95_below_margin"] for c in conditions),
            "readout_transfer_valid95": all(readout["shift"][c]["noninferior95"] for c in shifts),
            "clean_editor_beats_identity95": report["clean_primary"]["minus_identity"]["ci95"][1] < 0,
            "clean_editor_beats_random_and_wrong95": all(
                report["clean_primary"]["minus_" + c]["ci95"][1] < 0 for c in ("random", "wrong")),
        }
        physical_ok = all(physical_checks.values())
        regret_checks = {
            "transport_exceeds_margin95": regret["transport"][shift]["ci95"][0] > report["margins"]["regret"],
            "editor_deterioration_exceeds_margin95": (
                regret["deterioration"][shift]["ci95"][0] > report["margins"]["regret"]),
            "native_regret_equivalent90": all(native["regret_shift"][c]["equivalent90"] for c in shifts),
            "endpoint_oracle_adequate95": all(
                decision["endpoint_oracle"]["regret"][c]["upper95_below_margin"] for c in conditions),
            "endpoint_oracle_equivalent90": all(
                decision["endpoint_oracle"]["shift"][c]["equivalent90"] for c in shifts),
            "native_beats_random_regret95": all(
                decision["native_minus_random"][c]["ci95"][1] < 0 for c in conditions),
            "physical_effect_pattern_qualified": physical_ok,
        }
        regret_ok = all(regret_checks.values())
        output[shift] = {
            "method": method,
            "conditions_checked": list(conditions),
            "physical_effect": {
                "criteria": physical_checks,
                "unmet_criteria": [key for key, value in physical_checks.items() if not value],
                "nominal_pattern_supported": physical_ok,
                "statement": (
                    "Nominal intervals support excess physical-effect error and within-editor worsening, "
                    "conditional on upstream replay, absolute-accuracy and scorer-interface qualification. "
                    "Physical materiality is not established by a positive interval alone."
                    if physical_ok else
                    "An editor-specific physical-effect interpretation is not established. Retain all "
                    "roots and report unmet criteria; this is not evidence of no effect."),
            },
            "decision_regret": {
                "criteria": regret_checks,
                "unmet_criteria": [key for key, value in regret_checks.items() if not value],
                "nominal_pattern_supported": regret_ok,
                "statement": (
                    "Nominal intervals support corroborating candidate-bank regret increases beyond the "
                    "declared margin, conditional on the qualification checks. This is not closed-loop "
                    "policy improvement or a uniquely identified dynamics mechanism."
                    if regret_ok else
                    "Editor-specific material decision damage is not established. Scoring/representation "
                    "and forecast-path alternatives are not ruled out; native/oracle failures remain included."),
            },
            "inference_limit": "Descriptive, pointwise and conditional; not multiplicity-controlled confirmation.",
        }
    return output


def _analyze(data, codes, report):
    summaries = _Summaries()

    def series(method, condition, metric):
        values = data[:, :, METHODS.index(method), METRICS.index(metric)]
        # Half first: averaging two large finite losses must not overflow a sum.
        return (values[:, 1] / 2 + values[:, 2] / 2 if condition == "shifted"
                else values[:, APPEARANCES.index(condition)])

    def changes(method, metric):
        return {c: summaries.add(series(method, c, metric) - series(method, "original", metric)) for c in SHIFTS}

    report["means"] = {
        method: {c: {metric: summaries.add(series(method, c, metric))
                     for metric in ("effect_loss", "absolute_loss", "angle_error_deg", "regret")}
                 for c in CONDITIONS} for method in METHODS
    }
    report["contrasts"] = {}
    for metric in ("effect_loss", "regret"):
        methods = {}
        for method in METHODS[1:]:
            excess = {c: series(method, c, metric) - series("native", c, metric) for c in CONDITIONS}
            methods[method] = {
                "minus_native": {c: summaries.add(excess[c]) for c in CONDITIONS},
                "transport": {c: summaries.add(excess[c] - excess["original"]) for c in SHIFTS},
                "deterioration": changes(method, metric),
            }
        report["contrasts"][metric] = methods
    report["baselines"] = {field: summaries.add(series("native", "original", field))
                           for field in ("zero_effect_loss", "offset_effect_loss")}
    report["native"] = {
        "effect_shift": changes("native", "effect_loss"),
        "regret_shift": changes("native", "regret"),
        "accuracy": {c: {
            "minus_zero": summaries.add(series("native", c, "effect_loss") - series("native", c, "zero_effect_loss")),
            "minus_offset": summaries.add(series("native", c, "effect_loss") - series("native", c, "offset_effect_loss")),
        } for c in CONDITIONS},
    }
    report["readout"] = {
        "loss": {c: summaries.add(series("native", c, "readout_loss")) for c in CONDITIONS},
        "shift": changes("native", "readout_loss"),
    }
    oracle = {c: series("native", c, "endpoint_oracle_regret") for c in CONDITIONS}
    report["decision"] = {
        "endpoint_oracle": {
            "regret": {c: summaries.add(oracle[c]) for c in CONDITIONS},
            "shift": changes("native", "endpoint_oracle_regret"),
        },
        "minus_endpoint_oracle": {
            m: {c: summaries.add(series(m, c, "regret") - oracle[c]) for c in CONDITIONS} for m in METHODS},
        "transport_minus_endpoint_oracle": {
            m: {c: summaries.add((series(m, c, "regret") - oracle[c])
                                  - (series(m, "original", "regret") - oracle["original"]))
                for c in SHIFTS} for m in METHODS},
        "native_minus_random": {
            c: summaries.add(series("native", c, "regret") - series("random", c, "regret")) for c in CONDITIONS},
    }
    report["clean_primary"] = {
        "minus_" + m: summaries.add(series(report["primary_method"], "original", "effect_loss")
                                    - series(m, "original", "effect_loss")) for m in ("identity", "random", "wrong")}
    summaries.finish(codes, report["n_clusters"], report["resamples"], report["seed"])

    for c in SHIFTS:
        _equivalence(report["native"]["effect_shift"][c], report["margins"]["native"])
        _equivalence(report["native"]["regret_shift"][c], report["margins"]["regret"])
        _equivalence(report["decision"]["endpoint_oracle"]["shift"][c], report["margins"]["regret"])
        transfer = report["readout"]["shift"][c]
        transfer.update(margin=report["margins"]["readout"],
                        noninferior95=transfer["ci95"][1] < report["margins"]["readout"])
    for c in CONDITIONS:
        accuracy = report["native"]["accuracy"][c]
        accuracy["beats_zero95"] = accuracy["minus_zero"]["ci95"][1] < 0
        accuracy["beats_offset95"] = accuracy["minus_offset"]["ci95"][1] < 0
        accuracy["beats_both95"] = accuracy["beats_zero95"] and accuracy["beats_offset95"]
        for summary, margin in ((report["readout"]["loss"][c], report["margins"]["readout"]),
                                (report["decision"]["endpoint_oracle"]["regret"][c], report["margins"]["regret"])):
            summary.update(margin=margin, upper95_below_margin=summary["ci95"][1] < margin)
    report["conclusions"] = _conclusions(report)
    return report


def analyze_records(records: list[dict], *, resamples=5000, seed=0, native_margin=0.05,
                    readout_margin=0.05, regret_margin=0.02, primary_method="ridge") -> dict:
    """Validate the full roster and return paired root-estimand statistics.

    Required keys are root_id, split, appearance, method and every METRICS field.
    IDs are nonempty strings or integers (1 and "1" remain distinct). Optional
    episode_id must be consistently present/absent and identical within a root.
    Roots sharing it are resampled together; roots without it are assumed
    independent, in a separate ID namespace. No episode may be split into
    independent roots upstream. Additional metadata is ignored, not validated.
    Diagnostic copies must agree EXACTLY: zero/offset baselines across all rows
    of a root, readout/endpoint-oracle losses across its methods per appearance.

    Select primary_method from ridge/inverse/mlp/rollout on CLEAN validation
    upstream; this function never selects it. At least two independent clusters
    and two resamples are required. Fewer than 5000 resamples warns and is for
    development/synthetic tests, not reported inferential results.

    native_margin is the effect-loss equivalence half-width, NOT a material
    editor-damage threshold. readout_margin is both the real-endpoint mean-loss
    tolerance and allowed transfer deterioration (upper two-sided 95% bound).
    regret_margin is the native/oracle regret-equivalence half-width, oracle
    adequacy tolerance, and minimum corroborating regret increase. All three
    must be positive, finite, development-justified values in the reported units.
    These defaults are conveniences, not scientifically validated thresholds.

    Cluster bootstrap draws C clusters with replacement, copies every paired
    sibling, and divides sampled root totals by sampled root counts. This
    preserves equal ROOT weighting even for unequal episode sizes. Combined
    descriptive conclusions require qualification of EACH shift as well as
    their average; an average must not hide a failed appearance condition.
    """
    resamples = _integer(resamples, "resamples", 2)
    seed = _integer(seed, "seed", 0)
    margins = {name: _scalar(value, name + "_margin") for name, value in
               (("native", native_margin), ("readout", readout_margin), ("regret", regret_margin))}
    if any(value <= 0 for value in margins.values()):
        raise ValueError("All margins must be strictly positive.")
    if not isinstance(primary_method, str) or primary_method not in EDITORS:
        raise ValueError(f"primary_method must be one of {EDITORS}, selected on clean validation upstream.")
    data, split, codes, clusters = _validate_records(records)
    notices = []
    if resamples < 5000:
        notice = "resamples < 5000: development/synthetic-test intervals only; use at least 5000 for reporting."
        notices.append(notice)
        warnings.warn(notice, UserWarning, stacklevel=2)
    if len(clusters) < 10:
        notices.append("Fewer than 10 independent clusters: percentile coverage can be unreliable; "
                       "two clusters is only an API minimum, not a power guarantee.")
    report = {
        "schema_version": "wm_audit.statistics.v1", "split": split,
        "primary_method": str(primary_method),
        "primary_contrast": f"contrasts.effect_loss.{primary_method}.transport.shifted",
        "seed": seed, "resamples": resamples, "n_records": len(records),
        "n_roots": int(data.shape[0]), "n_clusters": len(clusters), "clusters": clusters,
        "appearance_weights": {"color_only": 0.5, "background_only": 0.5},
        "margins": margins, "warnings": notices, "outcome_filtering": False,
        "inference": {
            "interval_method": "paired cluster percentile bootstrap; linear quantiles",
            "ci95_level": 0.95, "ci90_level": 0.90, "equivalence_alpha": 0.05,
            "equivalence_rule": "ci90 strictly inside (-margin, margin); approximate bootstrap TOST-style rule",
            "accuracy_rule": "paired difference upper two-sided 95% bound < 0",
            "multiplicity": "Pointwise intervals only; no multiplicity correction or automatic significance "
                            "across methods, conditions, endpoints or checkpoints. Prespecify/adjust confirmatory families.",
            "confirmatory": False, "reporting_resamples_met": resamples >= 5000,
            "physical_materiality": "Not tested: native_margin is not an editor-effect threshold.",
        },
        "assumptions": [
            "Primary method was selected by the caller on clean validation; no selection is performed here.",
            "One split/checkpoint/fitted editor and scorer seed roster per call; intervals are conditional on "
            "these fits/seeds, not arbitrary retraining or world-model training-seed robustness.",
            "Roots are equally weighted. Episodes are independent sampling clusters; supply episode_id for "
            "every root sharing a source. Missing episode IDs assert independent roots, not unknown dependence.",
            "Candidates, horizons and repeated fitting seeds were aggregated within roots upstream, never "
            "counted as independent observations. Nuisance weights are fixed at one half per shift.",
            "Inference is conditional on a reused fixed appearance roster; a sampled appearance distribution "
            "must have been fixed independently of confirmation outcomes.",
            "All methods use the same physical roots, actions, goals, scoring surface and candidate bank. "
            "These upstream contracts cannot be verified from aggregate records alone.",
            "All valid roots, zero-effect cases, candidate ties and native/oracle representation failures "
            "are retained upstream. Nonfinite data raises for the whole call; failure penalties/endpoints "
            "must be prespecified upstream, never chosen here after observing failures.",
            "Readout loss measures independently fitted scorer error on re-encoded real endpoints at the "
            "exact rollout scoring interface. Passing it does not prove validity on off-manifold edited latents.",
            "Native baseline comparisons establish relative effect accuracy only. Absolute/angle errors, "
            "replay, clean semantic validity and physical effect materiality need upstream qualification.",
            "Regret is relative only to the tested bank. Zero regret, including all-tied banks, is not alone "
            "evidence of useful ranking. Oracle/forecast/editor contrasts are not a nonnegative causal decomposition.",
            "Roster validation detects missing cells for supplied roots, not roots wholly omitted upstream; "
            "compare against the frozen external root manifest. No outcome-based inclusion is allowed.",
            "Nominal bootstrap criteria and descriptive conjunctions do not resolve multiplicity, low power, "
            "small-cluster coverage, optional stopping, or the absence of a locked preregistration.",
        ],
    }
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _analyze(data, codes, report)
    except FloatingPointError as exc:
        raise ValueError("Analysis arithmetic overflowed; no records were filtered or discarded.") from exc