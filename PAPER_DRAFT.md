# From latent prediction to physical claims: a diagnostic audit of frozen PushT world models

**Working manuscript — exploratory discovery plus fixed fresh-root replication; not submission-ready.**
Updated 2026-11-03 after tracked numerical audits and fresh-root replication,
authenticated download and independent recomputation. This draft preserves negative results and does not claim that
the originally proposed physical-editor hypothesis has been established.

## Abstract

Accurate latent prediction does not by itself establish accurate physical
measurement, action-effect fidelity, or useful goal-directed decisions. We audit
two frozen RGB world-model releases on paired PushT trajectories and a recorded
pretraining distribution. Across 128 generated pilot roots in original appearance,
LeWM improves over persistence at all tested horizons; the Point-LeWM-author DINO-WM release has no
clear improvement at the longest horizon. Actual-image linear readouts improve
over training-mean controls but retain substantial physical-coordinate errors
and appearance sensitivity. Both models improve over persistence on 128 recorded
episode windows; these are pretraining-distribution diagnostics, not held-out
generalization tests. These findings do not support the original physical-editor
qualification claim. Complementary follow-ups find that the tested nonlinear
probes do not rescue measurement, both models exhibit latent action-effect
fidelity, and DINO's generated oracle-history prediction has no clear advantage
over local persistence. A fixed 128-fresh-root replication supports action-effect
fidelity and specificity for both checkpoints using a five-contrast family of
99% intervals. DINO's exploratory forecast-over-endpoint ranking advantage is
inconclusive under independent primary-family inference. A bounded eight-root
numerical audit finds encoding-layout sensitivity strongly reduced by a
convolution-precision control, without establishing the cause of all historical
drift. The contribution currently supported
is a reproducible checkpoint-specific diagnostic case study, not a new general
theory of physical representation.

## 1. Motivation and scope

The initial project asked whether physical latent edits transfer across
appearance changes. Its interpretation requires reliable measurement and clean
editor competence first. Rather than silently replace those prerequisites with
latent prediction success, we retain failed or unresolved qualification outcomes
and test competing explanations. An unsuccessful readout can reflect limited
probe capacity, ambiguous observations, distribution mismatch, or inaccessible
information; it does not prove that the representation contains no physics.

The audit concerns the exact released checkpoints and interfaces documented in
the repository. In particular, the DINO release is the Point-LeWM-author RGB,
dummy-proprioception implementation, not every model called DINO-WM. General
claims about model families, closed-loop control, or causal understanding would
require broader evidence.

## 2. Methods

### Data and separation

The original checksum-verified pilot contains independent scorer, validation,
and pilot root rosters of 128, 64, and 128. Generated roots preserve factual and
counterfactual branches, eight executed candidates, and original/color/background
appearances. Readouts fit original scorer images, select on original validation,
and evaluate the locked choice on pilot images. Predictions are scored at horizons
1, 4, and 8 with the original observed-prefix controls. No pilot roots are removed
based on outcomes.

The recorded audit uses 128 identical metadata/seed-selected episode/start
windows for both models. The authenticated release contains finite terminal
action labels and some nonterminal controls outside the simulator unit box. We
preserve those controls without clipping and retain the released normalization;
terminal rows never become outgoing transitions. Recorded data overlaps the
pretraining distribution and cannot establish unseen-distribution generalization.

### Controls and uncertainty

Actual-image readouts are compared with the scorer training mean. Native rollout
is compared with persistence and zero-future-action rollout. Each root or recorded
episode receives equal weight after averaging correlated records. Intervals are
pointwise 95% clustered bootstrap intervals, not multiplicity-corrected
confirmatory tests. Absolute latent error scales are not compared across models.

### Prospective exploratory follow-ups

The [fixed follow-up protocol](FOLLOWUP_PROTOCOL_20261101.md) was written after
the first diagnostic outcomes and before these follow-ups. It is not a
retrospective preregistration. F1 compares target-specific ridge selection and
bounded nonlinear readouts using all predeclared seeds. F2 separates autonomous
feedback from oracle actual-history one-step prediction; DINO additionally
crosses visual history with recurrent versus encoded-zero auxiliary channels.
F3 compares predicted candidate effects with executed effects and all wrong
assignments. F4 compares native latent goal ranking of true versus predicted
endpoints, retaining uniform, hold, and no-goal-coverage controls.

## 3. Verified prerequisite results

### Original editor qualification was not established

The original pilot used a scorer fitted on a mixture of actual and predicted
surfaces. Its original-appearance readout losses were 71.6784 (LeWM) and 520.6246
(DINO), versus the draft 0.05 tolerance. Validation-selected MLP-minus-identity
effect losses were +1.0984 and +7.7819, respectively, with both reported intervals
above zero. These are poor scores under **unqualified measurement**, not a
demonstration that the edits harm physical behavior. Both original transport
interaction intervals include zero: neither appearance-specific editor failure
nor invariance is established. See the
[original pilot analysis](PILOT_BACKUP_ANALYSIS.md) for all estimates and limits.

### Actual-image-only readouts remain insufficiently qualified

The following later diagnostics fitted actual-image-only readouts. They did not
refit editors or retrospectively qualify the original pilot.

| Model | Pusher distance error | Block distance error | Angle MAE |
|---|---:|---:|---:|
| LeWM | 120.855 | 26.932 | 17.017° |
| DINO release | 99.765 | 35.157 | 39.480° |

Distances use native 512-arena simulator units, not resized-image pixels.
Readouts outperform their training-mean baselines, but improvement alone does not
provide the resolution needed for small physical-edit claims. Both exhibit
appearance-dependent degradation. The full estimates and intervals are retained
in the [verified diagnostic report](MOLAB_RESULTS_20261101.md).

A post-hoc geometric check finds that the pusher circle is wholly off-canvas in
134/1,536 sampled images. Errors remain substantial even when its center is
on-canvas. This is a descriptive observation bound, not a semantic visibility
test, a new exclusion rule, or a full-history information-theoretic result.

### Latent prediction is distribution- and horizon-dependent

Native-minus-persistence latent MSE at horizon 8, original appearance:

| Distribution | Model | Mean difference [pointwise 95% CI] |
|---|---|---|
| Generated | LeWM | −0.3464 [−0.4018, −0.2934] |
| Generated | DINO release | −0.0107 [−0.0667, 0.0423] |
| Recorded pretraining | LeWM | −1.6071 [−1.6953, −1.5134] |
| Recorded pretraining | DINO release | −0.8703 [−0.9309, −0.8084] |

Negative favors native rollout within each model. These magnitudes cannot rank
the models across their different feature spaces. DINO's generated horizon-8
interval includes zero in every tested appearance. Both models beat the
zero-future-action control at generated horizon 8, but action sensitivity is not
yet evidence of correct action-effect direction or gain.

## 4. Follow-up results

All six follow-up reports completed on the recovered hosted GPU and were
downloaded with 132 inventory entries verified across three archives. Full
estimates, all seeds, receipts, and caveats are in the
[follow-up results report](FOLLOWUP_RESULTS_20261102.md).

![Primary exploratory follow-up estimates with pointwise confidence intervals](runs/molab-20261102-retrieved/figures/followup_summary.png)

Figure 1. All predeclared MLP seeds, oracle-history versus local-persistence
contrasts, latent action-effect controls, and finite-bank candidate regret.
Latent comparisons use separate axes; raw scales cannot rank models. The
[vector PDF](runs/molab-20261102-retrieved/figures/followup_summary.pdf) and
[source-hash manifest](runs/molab-20261102-retrieved/figures/followup_summary.sources.json)
are generated from authenticated reports, not manually entered estimates.

### Bounded measurement capacity

At original appearance / horizon 8, group-selected ridge block errors are
29.317 units (LeWM) and 32.158 (DINO). Across all three MLP seeds, errors are
38.571–40.670 and 55.272–60.289 units, respectively. All six MLP-minus-ridge
block-error intervals are above zero. No tested probe approaches the specified
4-unit / 2.5° diagnostic resolution. Every fitted method also has greater paired
angular-effect error than a zero angular-change baseline. These are bounded
negative results, not evidence that physical information is absent.

There are 280 identical resized-RGB groups with differing labels across 6,912
original-appearance samples, implying empirical single-image ambiguity. This
does not bound full-history inference. The follow-up evaluates all candidates,
so its errors cannot be compared directly with the earlier selected-candidate,
mixed-time readout table.

### Feedback and auxiliary controls

Actual-history-minus-local-persistence MSE is −0.009385
[−0.013506, −0.005714] for generated LeWM, but −0.002820
[−0.013084, 0.007157] for generated DINO. DINO's large improvement from replacing
predicted history with actual history therefore does not establish a purely
autoregressive explanation: local prediction still has no clear advantage over
the aligned local baseline. DINO auxiliary resets change errors by very small
amounts and offer no meaningful rescue. Both models have clear local advantages
on the recorded pretraining distribution; these remain non-generalization tests.

### Action effects and candidate ranking

Action-effect error minus the zero-effect baseline is −0.377371
[−0.434371, −0.322418] for LeWM and −0.275802 [−0.321397, −0.231579] for DINO.
Matched candidates also beat the mean of all wrong assignments. Thus the models
show nontrivial action-effect fidelity in their latent metrics, despite failed
physical-readout qualification. This does not establish block-specific accuracy.

Both native forecast rankings outperform expected uniform choice, but neither
clearly beats hold. DINO forecast-minus-true-endpoint regret is −0.013560
[−0.022595, −0.005049]; LeWM's is +0.006081 [−0.001434, 0.014495]. Actual encoded
endpoints do not necessarily yield better native-metric rankings than forecasts.
Mean best available coverage is only 11.28%, limiting all decision claims.

Together these exploratory findings motivate a diagnostic case study of
measurement, action fidelity, and decision-score mismatch, not a confirmed
physical-editor transfer result. New mechanisms and generalizations require
fresh-root replication rather than further tuning on this pilot.

## 5. Fresh-root replication and numerical qualification

The [dated replication protocol](REPLICATION_PROTOCOL_20261102.md) fixed 128 new
pilot roots, seed 2026110201, before generation. It preserves the discovery
distribution and candidate bank, excludes original/reserved root identities,
and performs no fitting, filtering or seed search. The primary family contains
action-effect error minus zero and matched-minus-all-wrong assignment for each
model, plus DINO forecast-minus-endpoint native-goal regret. Equal-root clustered
bootstrap uses 10,000 shared draws and two-sided 99% Bonferroni intervals.
This is a prospective local design after discovery, not external registration.

| Primary contrast | LeWM estimate [99% CI] | DINO estimate [99% CI] |
|---|---|---|
| Effect error minus zero | −0.375268 [−0.452880, −0.303452] | −0.236408 [−0.288885, −0.183415] |
| Matched minus all wrong | −0.517972 [−0.584175, −0.454432] | −0.560457 [−0.596832, −0.524494] |
| Forecast minus endpoint regret | Not primary | −0.007112 [−0.018928, 0.005073] |

Four primary contrasts support the prespecified direction. **DINO's ranking
advantage remains inconclusive**, so the discovery observation is not promoted
to a replicated decision claim. Half the fresh banks lack positive coverage
(129/256); zero regret is not control success. F2 secondary intervals preserve
LeWM's useful local prediction and DINO's inconclusive local-persistence contrast.
F1 and recorded feedback were not independently repeated.

The [bounded numerical audit](NUMERICAL_AUDIT_PROTOCOL_20261102.md) compares both
observed execution paths, crossed encoding/rollout layouts, identical repeats,
wrapper/reduction contrasts and four distinct backend interventions on eight
preselected original roots. Baseline maximum native-MSE differences are
0.00489527 / 0.00153232 (LeWM/DINO), largely aligned with encoding-layout changes.
Convolution IEEE permission reduces these to 0.000002563 / 0.000006437; it also
changes forecasts and encoded references, so this is not physical-accuracy
improvement. Feedback/public tensors agree; unchanged same-process repeats are
exact. The historical larger maxima and portability across devices remain
unresolved. Parameters, nonpersistent buffers and backend state were checked.

All numerical CSV summaries and the replication's five primary plus all
secondary intervals were independently recomputed. See the
[complete tracked-results report](PAPER_STUDIES_RESULTS_20261103.md) for
full-precision receipts, all controls, source pins, timings and archive hashes.

## 6. Limitations and defensible claims

This is one environment, two frozen releases, a finite generated candidate bank,
an exploratory pilot inspected before follow-up design, and a fixed fresh sample
on the same distribution. Recorded performance
is not held-out evidence. Teacher forcing uses privileged observations and is
not a deployable forecast. Good actual-image decoding cannot qualify edited or
predicted states. A zero-regret bank with no positive coverage is not task success.
No closed-loop evaluation or new editor confirmation has been completed. In the
original appearance, 65/128 factual and 62/128 counterfactual candidate banks
had zero coverage for every candidate. Low regret must therefore be accompanied
by absolute coverage and candidate-bank quality, not interpreted alone.

The results support distinguishing measurement quality, prediction relative to
controls, and downstream decision validity. They do not yet uniquely identify
why an individual model fails, establish transfer of a qualified physical editor,
or justify a universal claim about world models.

Batching differs between old and new inference. Aggregate native/persistence
errors change by at most 0.000146, without changing conclusions, but some
individual native errors differ by up to 0.028922. The eight-root audit now
isolates sensitivity to encoding layout and a convolution-precision intervention
on that sample, but cannot establish the historical cause. The runtime adapter-parity checks
and immutable parameter/buffer checks support implementation consistency, not
bitwise replication across execution layouts.

## 7. Reproducibility and work remaining

The six-prerequisite archive contains 120 hash-inventoried files; its SHA256 and
source pins appear in the diagnostic report. Original pilot inputs remain in a
separate receipt-verified backup. Hosted recovery must preserve those bytes and
use fresh output directories. Code tests establish implementation invariants,
not scientific success.

Before submission: adapt the verified summary figure to the venue and add secondary-slice figures; identify any
missing evidence rather than tune on pilot outcomes; integrate the completed
fixed replication without broadening its claims; audit the related-work discussion against
primary sources in [LITERATURE.md](LITERATURE.md); add checked bibliographic
citations, checkpoint details, and venue-compliant reproducibility materials.