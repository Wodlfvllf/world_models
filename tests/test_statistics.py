"""Synthetic mathematical/validation tests, NOT tests of research hypotheses.

No simulator, pretrained model, dataset or research result is used. Small
bootstrap counts intentionally exercise the development API; they do not
validate inferential coverage or establish real-world competence/significance.
"""

from copy import deepcopy
import json
import warnings

import numpy as np
import pytest

from wm_audit import statistics as stats
from wm_audit.statistics import (
    analyze_records, candidate_regret, circular_error_deg, physical_effect_loss,
)


pytestmark = pytest.mark.filterwarnings("ignore:resamples < 5000:UserWarning")

APPEARANCES = ("original", "color_only", "background_only")
METHODS = ("native", "identity", "ridge", "inverse", "mlp", "rollout", "random", "wrong", "zero")
METRICS = ("effect_loss", "absolute_loss", "angle_error_deg", "regret",
           "endpoint_oracle_regret", "zero_effect_loss", "offset_effect_loss", "readout_loss")


def synthetic_records(n_roots=6, episode_ids=None):
    """Complete synthetic roster with known paired differences and good controls."""
    clean_effect = dict(zip(METHODS, (0.10, 0.90, 0.12, 0.25, 0.16, 0.14, 0.80, 0.75, 0.90)))
    clean_regret = dict(zip(METHODS, (0.005, 0.30, 0.01, 0.04, 0.02, 0.015, 0.40, 0.35, 0.30)))
    records = []
    for root in range(n_roots):
        for appearance in APPEARANCES:
            shifted = appearance != "original"
            for index, method in enumerate(METHODS):
                effect_shift, regret_shift = 0.0, 0.0
                if method in ("ridge", "inverse", "mlp", "rollout") and shifted:
                    effect_shift = 0.20 if appearance == "color_only" else 0.40
                    regret_shift = 0.08 if appearance == "color_only" else 0.12
                record = {
                    "root_id": root, "split": "confirmation", "appearance": appearance, "method": method,
                    "effect_loss": clean_effect[method] + 0.005 * root + effect_shift,
                    "absolute_loss": 0.2 + 0.01 * index + 0.001 * root + 0.02 * shifted,
                    "angle_error_deg": 2.0 + index + 0.1 * root + 0.1 * shifted,
                    "regret": clean_regret[method] + 0.0001 * root + regret_shift,
                    "endpoint_oracle_regret": 0.002 + 0.0001 * root + 0.0005 * shifted,
                    "zero_effect_loss": 1.0 + 0.005 * root,
                    "offset_effect_loss": 0.8 + 0.005 * root,
                    "readout_loss": 0.002 + 0.0001 * root + 0.001 * shifted,
                }
                if episode_ids is not None:
                    record["episode_id"] = episode_ids[root]
                records.append(record)
    return records


def analyze(records, **kwargs):
    options = {"resamples": 127, "seed": 11}
    options.update(kwargs)
    return analyze_records(records, **options)


def constant_summary(summary, value):
    assert summary["estimate"] == pytest.approx(value)
    assert summary["ci95"] == pytest.approx([value, value])
    assert summary["ci90"] == pytest.approx([value, value])


def sincos(degrees):
    radians = np.deg2rad(degrees)
    return np.stack((np.sin(radians), np.cos(radians)), axis=-1)


def test_physical_loss_feature_sum_scaling_equal_weights_and_one_indexed_horizons():
    factual = np.zeros((2, 8, 2))
    true_cf = np.arange(32, dtype=float).reshape(2, 8, 2) / 4
    predicted_cf = true_cf.copy()
    predicted_cf[0, [0, 3, 7]] += [[2, 0], [0, 8], [2, 4]]
    predicted_cf[1, [0, 3, 7]] += [[4, 4], [6, 0], [0, 4]]
    predicted_cf[:, 1] += 1e6  # Unselected FUTURE step 2 must not affect the mean.
    actual = physical_effect_loss(factual, predicted_cf, factual, true_cf, [2, 4], [1, 4, 8])
    # Selected, feature-SUMMED losses: 1,4,2,5,9,1. No extra division by D.
    assert isinstance(actual, float)
    assert actual == pytest.approx(22 / 6)
    assert physical_effect_loss(factual, predicted_cf, factual, true_cf, [2, 4], [8, 1, 4]) == actual


def test_physical_loss_preserves_batch_axes_and_does_not_mutate():
    true_factual = np.zeros((2, 3, 4, 2))
    true_cf = np.ones_like(true_factual)
    pred_factual = true_factual.copy()
    pred_cf = true_cf.copy()
    pred_cf[0] += [2, 4]
    pred_cf[1] += [4, 8]
    before = [value.copy() for value in (pred_factual, pred_cf, true_factual, true_cf)]
    result = physical_effect_loss(pred_factual, pred_cf, true_factual, true_cf, [2, 4], [1, 4])
    np.testing.assert_allclose(result, [2, 8])
    for old, new in zip(before, (pred_factual, pred_cf, true_factual, true_cf)):
        np.testing.assert_array_equal(old, new)
    expanded = [np.stack((a, a)) for a in before]
    np.testing.assert_allclose(physical_effect_loss(*expanded, [2, 4], [1, 4]), [[2, 8], [2, 8]])


def test_effect_loss_common_absolute_bias_cancels_but_zero_effect_roots_remain():
    factual = np.arange(12.0).reshape(2, 3, 2)
    counterfactual = factual + [1, -2]
    assert physical_effect_loss(factual + 100, counterfactual + 100,
                                factual, counterfactual, [1, 1], [1, 3]) == 0
    zero = np.zeros((2, 1, 1))
    predicted = np.array([[[0.0]], [[2.0]]])
    # The zero-error, zero-effect candidate is included in the denominator.
    assert physical_effect_loss(zero, predicted, zero, zero, [1], [1]) == 2


@pytest.mark.parametrize("horizons", [[], [0], [-1], [4], [1, 1], [1.0], [True],
                                      [True, 2], [[1]], 1, [np.nan], ["1"]])
def test_physical_loss_rejects_bad_horizons(horizons):
    values = np.zeros((2, 3, 2))
    with pytest.raises(ValueError, match="horizons"):
        physical_effect_loss(values, values, values, values, [1, 1], horizons)


@pytest.mark.parametrize("scales", [[0, 1], [-1, 1], [np.nan, 1], [np.inf, 1],
                                    [1], [[1, 1]], 1, [True, False], [1, True], ["1", "1"], [1j, 1]])
def test_physical_loss_rejects_bad_scales(scales):
    values = np.zeros((2, 3, 2))
    with pytest.raises(ValueError, match="scales"):
        physical_effect_loss(values, values, values, values, scales, [1])


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.zeros((1, 3, 2)),
                                 np.zeros((0, 3, 2)), np.zeros((2, 0, 2)), np.zeros((2, 3, 0))])
def test_physical_loss_rejects_shapes_and_broadcasting(bad):
    values = np.zeros((2, 3, 2))
    with pytest.raises(ValueError, match="shapes"):
        physical_effect_loss(values, bad, values, values, [1, 1], [1])


@pytest.mark.parametrize("branch", range(4))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_physical_loss_checks_finite_values_even_at_unselected_steps(branch, bad):
    arrays = [np.zeros((2, 3, 2)) for _ in range(4)]
    arrays[branch][0, 1, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        physical_effect_loss(*arrays, [1, 1], [1, 3])


@pytest.mark.parametrize("dtype", [bool, complex, str, object])
def test_physical_loss_rejects_nonreal_dtypes(dtype):
    bad = np.zeros((2, 3, 2)).astype(dtype)
    good = np.zeros((2, 3, 2))
    with pytest.raises(ValueError):
        physical_effect_loss(bad, good, good, good, [1, 1], [1])


def test_physical_loss_rejects_arithmetic_overflow_instead_of_returning_infinity():
    zero = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="overflow"):
        physical_effect_loss(zero, np.full_like(zero, 1e308), zero, zero, [1], [1])


def test_circular_error_wraparound_sincos_order_and_nonunit_vectors():
    pred = sincos(np.array([179.0, -179.0, 0.0, 90.0, 180.0]))
    true = sincos(np.array([-179.0, 179.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(circular_error_deg(pred * 7, true * 0.25), [2, 2, 0, 90, 180], atol=1e-12)
    assert circular_error_deg([1, 0], [0, 1]) == 90
    assert isinstance(circular_error_deg([0, 1], [0, 1]), float)
    assert circular_error_deg(np.ones(2) * 1e308, np.ones(2) * 1e-300) == 0
    assert circular_error_deg(np.tile(pred, (2, 1, 1)), np.tile(true, (2, 1, 1))).shape == (2, 5)


@pytest.mark.parametrize("pred,true", [([0, 0], [0, 1]), ([0, 1], [0, 0]),
                                       ([], []), ([1], [1]), (1, 1), ([0, 1, 2], [0, 1, 2]),
                                       ([[0, 1]], [0, 1]), ([np.nan, 1], [0, 1]),
                                       ([0, 1], [np.inf, 1]), ([1j, 1], [0, 1]),
                                       ([True, False], [0, 1]), ([True, 1], [0, 1]),
                                       (["0", "1"], [0, 1])])
def test_circular_error_rejects_undefined_nonfinite_and_mismatched_inputs(pred, true):
    with pytest.raises(ValueError):
        circular_error_deg(pred, true)


def test_candidate_regret_lowest_index_ties_and_true_cost_ranking():
    # Predicted tie chooses candidate 0, NOT the tied candidate with best true cost.
    assert candidate_regret([5, 1, 2], [0, 0, 3]) == 4
    assert candidate_regret([5, 1, 2], [0, -np.finfo(float).eps, 3]) == 0
    assert candidate_regret([-5, -2, -3], [3, 1, 0]) == 2
    assert candidate_regret([7], [-8]) == 0
    assert isinstance(candidate_regret([1, 2], [0, 1]), float)


def test_candidate_regret_retains_all_tied_banks_and_leading_axes():
    actual = np.array([[5.0, 1, 2], [4, 4, 4]])
    predicted = np.array([[0.0, 0, 2], [2, 1, 1]])
    before = actual.copy(), predicted.copy()
    np.testing.assert_array_equal(candidate_regret(actual, predicted), [4, 0])
    np.testing.assert_array_equal(candidate_regret(actual[None], predicted[None]), [[4, 0]])
    np.testing.assert_array_equal(actual, before[0])
    np.testing.assert_array_equal(predicted, before[1])
    assert candidate_regret([0.6, 0.6, 0.6], [0, 0, 0]) == 0


@pytest.mark.parametrize("costs,predicted", [([], []), (1, 1), ([1, 2], [1]),
                                             ([[1, 2]], [1, 2]), ([[1, 2]], [[1, 2], [3, 4]]),
                                             ([1, np.nan], [0, 1]), ([1, 2], [0, np.inf]),
                                             ([np.inf, 1], [1, 0]), ([1, 2], [1j, 0]),
                                             ([True, False], [0, 1]), ([1, True], [0, 1]),
                                             ([1, 2], ["0", "1"])])
def test_candidate_regret_strict_finite_shape_validation(costs, predicted):
    with pytest.raises(ValueError):
        candidate_regret(costs, predicted)


def test_candidate_regret_rejects_overflow():
    with pytest.raises(ValueError, match="overflow"):
        candidate_regret([-1e308, 1e308], [1, 0])


def test_full_roster_paired_math_and_companion_means():
    result = analyze(synthetic_records())
    assert result["n_records"] == 6 * 3 * 9
    assert result["n_roots"] == result["n_clusters"] == 6
    assert result["split"] == "confirmation"
    assert set(result["means"]) == set(METHODS)
    assert result["appearance_weights"] == {"color_only": 0.5, "background_only": 0.5}
    effect = result["contrasts"]["effect_loss"]["ridge"]
    regret = result["contrasts"]["regret"]["ridge"]
    for condition, delta, regret_delta in (("color_only", 0.2, 0.08),
                                           ("background_only", 0.4, 0.12), ("shifted", 0.3, 0.10)):
        constant_summary(effect["transport"][condition], delta)
        constant_summary(effect["deterioration"][condition], delta)
        constant_summary(effect["minus_native"][condition], delta + 0.02)
        constant_summary(regret["transport"][condition], regret_delta)
        constant_summary(regret["deterioration"][condition], regret_delta)
        constant_summary(result["native"]["effect_shift"][condition], 0)
        constant_summary(result["native"]["regret_shift"][condition], 0)
        constant_summary(result["readout"]["shift"][condition], 0.001)
        assert result["conclusions"][condition]["physical_effect"]["nominal_pattern_supported"]
        assert result["conclusions"][condition]["decision_regret"]["nominal_pattern_supported"]
    constant_summary(effect["minus_native"]["original"], 0.02)
    for condition in (*APPEARANCES, "shifted"):
        accuracy = result["native"]["accuracy"][condition]
        constant_summary(accuracy["minus_zero"], -0.9)
        constant_summary(accuracy["minus_offset"], -0.7)
        assert accuracy["beats_both95"]
        assert result["readout"]["loss"][condition]["upper95_below_margin"]
    means = result["means"]["ridge"]["original"]
    assert means["effect_loss"]["estimate"] == pytest.approx(0.1325)
    assert means["absolute_loss"]["estimate"] == pytest.approx(0.2225)
    assert means["angle_error_deg"]["estimate"] == pytest.approx(4.25)
    assert means["regret"]["estimate"] == pytest.approx(0.01025)


def test_native_subtraction_is_distinct_from_within_editor_deterioration():
    records = synthetic_records()
    for row in records:
        if row["method"] == "native" and row["appearance"] != "original":
            row["effect_loss"] += 0.04
            row["regret"] += 0.01
    result = analyze(records)
    effect = result["contrasts"]["effect_loss"]["ridge"]
    regret = result["contrasts"]["regret"]["ridge"]
    constant_summary(effect["transport"]["shifted"], 0.26)
    constant_summary(effect["deterioration"]["shifted"], 0.30)
    constant_summary(regret["transport"]["shifted"], 0.09)
    constant_summary(regret["deterioration"]["shifted"], 0.10)
    constant_summary(result["native"]["effect_shift"]["shifted"], 0.04)
    constant_summary(result["native"]["regret_shift"]["shifted"], 0.01)


def test_episode_bootstrap_preserves_pairing_and_root_weighted_ratio_for_unequal_sizes():
    records = synthetic_records(6, episode_ids=["a", "a", "b", "c", "c", "c"])
    effects = np.array([0.0, 0.4, 1.4, 2.0, 2.2, 2.6])
    for row in records:
        if row["method"] == "ridge":
            factor = {"original": 0, "color_only": 1, "background_only": 2}[row["appearance"]]
            row["effect_loss"] = 0.2 + factor * effects[row["root_id"]]
    resamples, seed = 257, 23  # Exercises multiple bootstrap batches.
    result = analyze(records, resamples=resamples, seed=seed)
    assert result["n_clusters"] == 3 and result["n_roots"] == 6
    assert [cluster["n_roots"] for cluster in result["clusters"]] == [2, 1, 3]
    draws = np.random.default_rng(seed).integers(0, 3, size=(resamples, 3))
    totals = np.array([effects[:2].sum(), effects[2], effects[3:].sum()])
    sizes = np.array([2, 1, 3])
    boot = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    wrong_equal_episode_boot = (totals / sizes)[draws].mean(axis=1)
    assert not np.allclose(np.percentile(boot, [5, 95]), np.percentile(wrong_equal_episode_boot, [5, 95]))
    transport = result["contrasts"]["effect_loss"]["ridge"]["transport"]
    for condition, factor in (("color_only", 1), ("background_only", 2), ("shifted", 1.5)):
        summary = transport[condition]
        assert summary["estimate"] == pytest.approx(factor * effects.mean())
        assert summary["ci95"] == pytest.approx(factor * np.percentile(boot, [2.5, 97.5]))
        assert summary["ci90"] == pytest.approx(factor * np.percentile(boot, [5, 95]))


def test_seed_reproducibility_record_order_invariance_and_no_mutation_or_global_rng_use():
    records = synthetic_records(7)
    for row in records:
        if row["method"] == "ridge" and row["appearance"] != "original":
            row["effect_loss"] += row["root_id"] ** 2 / 10
    before = deepcopy(records)
    global_state = np.random.get_state()
    first = analyze(records, resamples=211, seed=123)
    second = analyze(records, resamples=211, seed=123)
    shuffled = [records[i] for i in np.random.default_rng(42).permutation(len(records))]
    assert first == second == analyze(shuffled, resamples=211, seed=123)
    third = analyze(records, resamples=211, seed=456)
    key = lambda r: r["contrasts"]["effect_loss"]["ridge"]["transport"]["shifted"]
    assert key(first)["estimate"] == key(third)["estimate"]
    assert key(first)["ci95"] != key(third)["ci95"]
    assert records == before
    new_state = np.random.get_state()
    assert global_state[0] == new_state[0] and global_state[2:] == new_state[2:]
    np.testing.assert_array_equal(global_state[1], new_state[1])


def test_json_serializable_metadata_and_nominal_not_automatic_significance():
    records = synthetic_records()
    for row in records:
        row["root_id"] = np.int64(row["root_id"])
        row["effect_loss"] = np.float64(row["effect_loss"])
        row["additional_metadata"] = "ignored, not a selection input"
    result = analyze(records, seed=np.int64(9), resamples=np.int64(31), native_margin=np.float64(0.05))
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded) == result
    assert result["seed"] == 9 and result["resamples"] == 31
    assert result["primary_method"] == "ridge"
    assert not result["outcome_filtering"]
    assert not result["inference"]["confirmatory"]
    assert "no multiplicity correction" in result["inference"]["multiplicity"]
    assert "conditional" in " ".join(result["assumptions"])
    assert "clean validation" in " ".join(result["assumptions"])
    assert "not an editor-effect threshold" in result["inference"]["physical_materiality"]


def test_small_resamples_warn_but_two_are_allowed():
    with pytest.warns(UserWarning, match="resamples < 5000"):
        result = analyze_records(synthetic_records(2), resamples=2)
    assert result["resamples"] == 2
    assert not result["inference"]["reporting_resamples_met"]
    assert any("resamples < 5000" in item for item in result["warnings"])
    assert any("power guarantee" in item for item in result["warnings"])


def test_default_5000_resamples_and_ninety_percent_not_ninety_five_for_equivalence():
    deviations = [-0.16, -0.10, -0.06, -0.02, 0.02, 0.06, 0.10, 0.16]
    records = synthetic_records(8)
    for row in records:
        if row["method"] == "native":
            row["effect_loss"] = 0.5 + (deviations[row["root_id"]] if row["appearance"] != "original" else 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyze_records(records, native_margin=0.061)
    assert not caught
    assert result["seed"] == 0 and result["resamples"] == 5000
    assert result["inference"]["reporting_resamples_met"]
    summary = result["native"]["effect_shift"]["shifted"]
    assert summary["ci90"][0] > -0.061 and summary["ci90"][1] < 0.061
    assert summary["ci95"][0] < -0.061 or summary["ci95"][1] > 0.061
    assert summary["equivalent90"]


@pytest.mark.parametrize("shift", [-0.125, 0.125])
def test_equivalence_margin_boundaries_are_strict(shift):
    records = synthetic_records()
    for row in records:
        if row["method"] == "native":
            row["effect_loss"] = 0.5 + (shift if row["appearance"] != "original" else 0)
    result = analyze(records, native_margin=0.125)
    constant_summary(result["native"]["effect_shift"]["shifted"], shift)
    assert not result["native"]["effect_shift"]["shifted"]["equivalent90"]


def test_equivalence_can_hold_for_small_nonzero_change_and_fail_for_wide_zero_mean():
    records = synthetic_records(2)
    for row in records:
        if row["method"] == "native":
            row["effect_loss"] = 0.5 + (0.01 if row["appearance"] != "original" else 0)
    small = analyze(records)
    assert small["native"]["effect_shift"]["shifted"]["ci95"][0] > 0
    assert small["native"]["effect_shift"]["shifted"]["equivalent90"]
    for row in records:
        if row["method"] == "native" and row["appearance"] != "original":
            row["effect_loss"] = 0.5 + (-0.25 if row["root_id"] == 0 else 0.25)
    wide = analyze(records)["native"]["effect_shift"]["shifted"]
    assert wide["estimate"] == pytest.approx(0)
    assert wide["ci95"][0] < 0 < wide["ci95"][1]
    assert not wide["equivalent90"]  # Nonsignificant is NOT equivalent.


def test_native_accuracy_requires_upper_bound_below_zero_for_both_baselines():
    records = synthetic_records(2)
    for row in records:
        row["zero_effect_loss"] = 0.5
        row["offset_effect_loss"] = 0.8
        if row["method"] == "native":
            row["effect_loss"] = [0.0, 0.9][row["root_id"]]
    result = analyze(records)
    accuracy = result["native"]["accuracy"]["original"]
    assert accuracy["minus_zero"]["estimate"] < 0 < accuracy["minus_zero"]["ci95"][1]
    assert not accuracy["beats_both95"]
    for row in records:
        if row["method"] == "native":
            row["effect_loss"] = 0.5
    equal = analyze(records)["native"]["accuracy"]["original"]
    constant_summary(equal["minus_zero"], 0)
    assert not equal["beats_zero95"] and equal["beats_offset95"] and not equal["beats_both95"]


def test_native_and_oracle_representation_failures_are_kept_not_qualification_filtered():
    records = synthetic_records()
    for row in records:
        if row["root_id"] == 0:
            row["endpoint_oracle_regret"] = 0.9
            if row["method"] == "native":
                row["effect_loss"] = 100.0
    result = analyze(records)
    assert result["n_roots"] == result["n_clusters"] == 6
    assert result["n_records"] == len(records)
    assert result["means"]["native"]["original"]["effect_loss"]["estimate"] > 16
    assert not result["native"]["accuracy"]["shifted"]["beats_both95"]
    assert not result["decision"]["endpoint_oracle"]["regret"]["shifted"]["upper95_below_margin"]
    constant_summary(result["contrasts"]["effect_loss"]["ridge"]["transport"]["shifted"], 0.3)
    assert not result["conclusions"]["shifted"]["physical_effect"]["nominal_pattern_supported"]
    assert not result["conclusions"]["shifted"]["decision_regret"]["nominal_pattern_supported"]


def test_readout_absolute_validity_not_just_stable_transfer():
    records = synthetic_records()
    for row in records:
        row["readout_loss"] = 0.25
    result = analyze(records)
    assert result["readout"]["shift"]["shifted"]["noninferior95"]
    assert not result["readout"]["loss"]["original"]["upper95_below_margin"]
    assert not result["conclusions"]["shifted"]["physical_effect"]["nominal_pattern_supported"]


def test_combined_qualification_cannot_hide_a_bad_appearance_in_an_average():
    records = synthetic_records()
    for row in records:
        row["readout_loss"] = 0.08 if row["appearance"] == "color_only" else 0.005
        if row["method"] == "native" and row["appearance"] != "original":
            row["effect_loss"] += 0.08 if row["appearance"] == "color_only" else -0.08
    result = analyze(records)
    assert result["readout"]["loss"]["shifted"]["upper95_below_margin"]
    assert result["native"]["effect_shift"]["shifted"]["equivalent90"]
    assert not result["readout"]["loss"]["color_only"]["upper95_below_margin"]
    assert not result["readout"]["shift"]["color_only"]["noninferior95"]
    assert not result["native"]["effect_shift"]["color_only"]["equivalent90"]
    conclusion = result["conclusions"]["shifted"]["physical_effect"]
    assert not conclusion["criteria"]["native_effect_equivalent90"]
    assert not conclusion["criteria"]["readout_absolute_valid95"]
    assert not conclusion["nominal_pattern_supported"]


def test_positive_transport_is_not_editor_damage_when_both_routes_improve():
    records = synthetic_records()
    for row in records:
        shifted = row["appearance"] != "original"
        if row["method"] == "ridge":
            row["effect_loss"] = 0.3 if shifted else 0.4
            row["regret"] = 0.08 if shifted else 0.10
        elif row["method"] == "native":
            row["effect_loss"] = 0.05 if shifted else 0.3
            row["regret"] = 0.0 if shifted else 0.15
    result = analyze(records, native_margin=0.3)
    for metric, expected in (("effect_loss", 0.15), ("regret", 0.13)):
        contrast = result["contrasts"][metric]["ridge"]
        constant_summary(contrast["transport"]["shifted"], expected)
        assert contrast["deterioration"]["shifted"]["ci95"][1] < 0
    conclusion = result["conclusions"]["shifted"]
    assert not conclusion["physical_effect"]["criteria"]["positive_editor_deterioration95"]
    assert not conclusion["physical_effect"]["nominal_pattern_supported"]
    assert not conclusion["decision_regret"]["criteria"]["editor_deterioration_exceeds_margin95"]


def test_clean_editor_qualification_and_primary_selection_are_not_chosen_from_shift_results():
    records = synthetic_records()
    for row in records:
        if row["method"] == "mlp":
            row["effect_loss"] = 0.05 + row["root_id"] * 0.001  # Cleanest editor, no transport.
    result = analyze(records, primary_method="mlp")
    assert result["primary_method"] == "mlp"
    assert result["primary_contrast"] == "contrasts.effect_loss.mlp.transport.shifted"
    assert result["contrasts"]["effect_loss"]["ridge"]["transport"]["shifted"]["estimate"] > 0
    constant_summary(result["contrasts"]["effect_loss"]["mlp"]["transport"]["shifted"], 0)
    assert result["conclusions"]["shifted"]["method"] == "mlp"
    assert not result["conclusions"]["shifted"]["physical_effect"]["nominal_pattern_supported"]
    assert analyze(records)["primary_method"] == "ridge"  # Never auto-select the cleanest either.
    for row in records:
        if row["method"] == "ridge":
            row["effect_loss"] += 2
    poor = analyze(records)
    assert not poor["conclusions"]["shifted"]["physical_effect"]["criteria"]["clean_editor_beats_identity95"]


def test_decision_oracle_contrasts_and_negative_gaps_are_not_an_additive_error_decomposition():
    result = analyze(synthetic_records())
    decision = result["decision"]
    constant_summary(decision["minus_endpoint_oracle"]["native"]["original"], 0.003)
    constant_summary(decision["transport_minus_endpoint_oracle"]["native"]["shifted"], -0.0005)
    constant_summary(decision["transport_minus_endpoint_oracle"]["ridge"]["shifted"], 0.0995)
    constant_summary(decision["endpoint_oracle"]["shift"]["shifted"], 0.0005)
    records = synthetic_records()
    for row in records:
        row["endpoint_oracle_regret"] = 0.6
    bad_oracle = analyze(records)
    assert bad_oracle["n_roots"] == 6
    assert bad_oracle["decision"]["minus_endpoint_oracle"]["native"]["original"]["estimate"] < 0
    assert not bad_oracle["conclusions"]["shifted"]["decision_regret"]["criteria"]["endpoint_oracle_adequate95"]


def test_regret_material_margin_is_not_just_positive_change():
    records = synthetic_records()
    for row in records:
        if row["method"] == "ridge":
            row["regret"] = 0.01 if row["appearance"] == "original" else 0.015
    result = analyze(records)
    assert result["contrasts"]["regret"]["ridge"]["transport"]["shifted"]["ci95"][0] > 0
    decision = result["conclusions"]["shifted"]["decision_regret"]
    assert not decision["criteria"]["transport_exceeds_margin95"]
    assert not decision["nominal_pattern_supported"]


def test_all_tied_regret_records_stay_in_roster_without_claiming_ranking_utility():
    records = synthetic_records()
    for row in records:
        row["regret"] = row["endpoint_oracle_regret"] = 0.0
    result = analyze(records)
    assert result["n_roots"] == 6 and result["n_records"] == 162
    for method in METHODS:
        constant_summary(result["means"][method]["original"]["regret"], 0)
    constant_summary(result["contrasts"]["regret"]["ridge"]["transport"]["shifted"], 0)
    assert not result["conclusions"]["shifted"]["decision_regret"]["criteria"]["native_beats_random_regret95"]


@pytest.mark.parametrize("field", ["root_id", "split", "appearance", "method", *METRICS])
def test_every_required_record_field_is_required(field):
    records = synthetic_records()
    del records[0][field]
    with pytest.raises(ValueError, match="missing required fields"):
        analyze(records)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("appearance", APPEARANCES)
def test_every_root_requires_every_appearance_method_cell(method, appearance):
    records = [row for row in synthetic_records() if not
               (row["root_id"] == 0 and row["method"] == method and row["appearance"] == appearance)]
    with pytest.raises(ValueError, match="missing roster cells"):
        analyze(records)


def test_duplicates_raise_even_if_identical_or_total_row_count_looks_complete():
    records = synthetic_records()
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(records + [deepcopy(records[0])])
    records[1] = deepcopy(records[0])  # One missing plus one duplicate must not cancel.
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(records)


@pytest.mark.parametrize("field,value", [("method", "best"), ("method", "Native"),
                                         ("appearance", "shifted"), ("appearance", "color_background"),
                                         ("split", ""), ("split", None), ("method", None),
                                         ("appearance", None), ("root_id", None), ("root_id", True),
                                         ("root_id", 1.0), ("root_id", np.nan), ("root_id", " "),
                                         ("root_id", []), ("episode_id", None), ("episode_id", False),
                                         ("episode_id", 1.0), ("episode_id", np.inf)])
def test_record_identity_validation(field, value):
    records = synthetic_records()
    records[0][field] = value
    with pytest.raises(ValueError):
        analyze(records)


def test_integer_and_string_ids_remain_distinct_and_fallback_ids_do_not_alias_episodes():
    records = synthetic_records(2)
    for row in records:
        row["root_id"] = 1 if row["root_id"] == 0 else "1"
    result = analyze(records)
    assert result["n_roots"] == result["n_clusters"] == 2
    assert [c["id"] for c in result["clusters"]] == [1, "1"]
    records = synthetic_records(2)
    for row in records:
        if row["root_id"] == 1:
            row["episode_id"] = 0
    result = analyze(records)
    assert result["n_clusters"] == 2
    assert {c["kind"] for c in result["clusters"]} == {"root_id", "episode_id"}


def test_episode_identity_must_be_consistent_for_all_sibling_records():
    records = synthetic_records(3, episode_ids=["a", "a", "b"])
    records[0]["episode_id"] = "another"
    with pytest.raises(ValueError, match="Inconsistent episode_id"):
        analyze(records)
    del records[0]["episode_id"]
    with pytest.raises(ValueError, match="Inconsistent episode_id"):
        analyze(records)


def test_minimum_two_independent_clusters_not_two_sibling_roots():
    with pytest.raises(ValueError, match="At least 2"):
        analyze(synthetic_records(1))
    with pytest.raises(ValueError, match="At least 2"):
        analyze(synthetic_records(6, episode_ids=["one"] * 6))
    assert analyze(synthetic_records(6, episode_ids=["one"] * 5 + ["two"]))["n_clusters"] == 2


def test_mixed_splits_cannot_be_pooled_or_used_for_method_selection():
    records = synthetic_records()
    for row in records:
        if row["root_id"] == 0:
            row["split"] = "validation"
    with pytest.raises(ValueError, match="one split"):
        analyze(records)


@pytest.mark.parametrize("field", METRICS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, None, "0.1", True, 1j, [0.1], np.array(0.1)])
def test_metric_fields_require_finite_real_scalars_without_coercion_or_dropping(field, bad):
    records = synthetic_records()
    records[-1][field] = bad
    with pytest.raises(ValueError, match="finite|scalar"):
        analyze(records)


@pytest.mark.parametrize("field", METRICS)
def test_metric_fields_are_nonnegative(field):
    records = synthetic_records()
    records[0][field] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        analyze(records)


def test_absolute_angle_errors_are_bounded_by_180():
    records = synthetic_records()
    records[0]["angle_error_deg"] = 180.01
    with pytest.raises(ValueError, match="angle_error_deg"):
        analyze(records)


@pytest.mark.parametrize("field", ["zero_effect_loss", "offset_effect_loss", "readout_loss", "endpoint_oracle_regret"])
def test_shared_diagnostic_copies_must_match_exactly(field):
    records = synthetic_records()
    records[1][field] = np.nextafter(records[1][field], np.inf)
    with pytest.raises(ValueError, match=field + " must agree exactly"):
        analyze(records)


@pytest.mark.parametrize("field", ["zero_effect_loss", "offset_effect_loss"])
def test_physical_baselines_cannot_vary_with_appearance_even_if_all_methods_agree(field):
    records = synthetic_records()
    for row in records:
        if row["appearance"] == "color_only":
            row[field] += 0.1
    with pytest.raises(ValueError, match=field + " must agree exactly"):
        analyze(records)


@pytest.mark.parametrize("name,bad", [("resamples", 0), ("resamples", 1), ("resamples", -1),
                                      ("resamples", 10.0), ("resamples", True), ("resamples", np.nan),
                                      ("seed", -1), ("seed", True), ("seed", 0.0), ("seed", np.inf)])
def test_integer_analysis_parameters_are_strict(name, bad):
    with pytest.raises(ValueError, match=name):
        analyze(synthetic_records(), **{name: bad})


@pytest.mark.parametrize("name", ["native_margin", "readout_margin", "regret_margin"])
@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf, True, "0.05", [0.05], 1j])
def test_margin_analysis_parameters_are_positive_finite_real_scalars(name, bad):
    with pytest.raises(ValueError, match="margin"):
        analyze(synthetic_records(), **{name: bad})


@pytest.mark.parametrize("method", ["native", "identity", "random", "wrong", "zero", "best", "Ridge", None, 2])
def test_primary_method_must_be_an_explicit_editor(method):
    with pytest.raises(ValueError, match="primary_method"):
        analyze(synthetic_records(), primary_method=method)


@pytest.mark.parametrize("bad", [None, [], {}, (), [{}], [None]])
def test_records_container_and_rows_are_strict(bad):
    with pytest.raises(ValueError):
        analyze(bad)


def test_analysis_fails_closed_on_nonrepresentable_derived_contrasts():
    records = synthetic_records()
    for row in records:
        if row["appearance"] == "original":
            row["effect_loss"] = 1e308 if row["method"] == "native" else 0.0
        else:
            row["effect_loss"] = 0.0 if row["method"] == "native" else 1e308
    with pytest.raises(ValueError, match="overflow|nonfinite"):
        analyze(records)


def test_finite_large_losses_do_not_overflow_representable_pooled_or_bootstrap_means():
    records = synthetic_records()
    for row in records:
        row["effect_loss"] = row["zero_effect_loss"] = row["offset_effect_loss"] = 1e308
    result = analyze(records)
    constant_summary(result["means"]["native"]["shifted"]["effect_loss"], 1e308)
    constant_summary(result["contrasts"]["effect_loss"]["ridge"]["transport"]["shifted"], 0)
    json.dumps(result, allow_nan=False)


def test_documented_public_api_and_roster_names():
    assert set(stats.__all__) == {"physical_effect_loss", "circular_error_deg", "candidate_regret", "analyze_records"}
    assert stats.METHODS == METHODS and stats.APPEARANCES == APPEARANCES