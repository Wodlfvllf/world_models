# Follow-up tranche: measurement, feedback, action effects, and goal geometry

**Specified 2026-11-01 after inspecting the six prerequisite diagnostics, before
executing this tranche.** This is a prospective specification of new exploratory
analyses on already inspected data, **not a retrospective preregistration**.
The original editor-transfer hypotheses and negative qualification outcomes are
retained. This protocol neither repairs their evidential status nor promises a
positive paper. Exact computational settings are in
[configs/followup_20261101.json](configs/followup_20261101.json).

## Scope and fixed budget

Two frozen checkpoints: LeWM projected CLS and the Point-LeWM-author RGB,
dummy-proprio DINO-WM release. Existing independent scorer/validation/pilot roots
remain 128/64/128. The optional recorded check reuses the **same** 128 authenticated
episode/start pairs, not a newly favorable sample. Recorded data remains a
pretraining-distribution diagnostic. No model pretraining, editor refit, or
adaptive test-root exclusion is included in this tranche.

The budget is one full measurement-capacity audit and one combined feedback /
action-fidelity / goal-geometry audit per model, plus one recorded feedback audit
per model. Six completed reports are expected. Fit seeds are robustness repeats,
not independent physical samples. Additional hypotheses suggested by outcomes
must be recorded separately with their timing and tested on new root families
before any claim of prospective replication.

## F1: physical measurement capacity and observation limits

**H-M:** current errors are partly attributable to linear accessibility, pooling,
or target-coupled hyperparameter selection. **Complement H-M0:** this bounded
capacity increase does not yield adequate measurement on the emitted surface.
Failure does not imply information absence or absence of physics.

- Fit features from original scorer actual images only; no predictions/editors.
- Retain the prior fixed-candidate sampling for fitting and selection. Evaluate
  all candidates at horizons 1/4/8 and all context frames in the full pilot.
- Ridge: LeWM CLS; DINO grids 4×4 and 7×7; four fixed regularizers.
- Nonlinear: standardized identical CLS / 4×4 features → 64-unit SiLU → six
  outputs; training-label mean and [512,512,512,512,1,1] scales. Two weight decays,
  checkpoints 50/100/200 epochs, three fixed seeds. No best-seed selection.
- Select regularization/grid or weight-decay/checkpoint separately for pusher XY,
  block XY, and orientation using original validation only. At fixed ridge alpha,
  dropping pusher columns cannot change the independent block coefficients;
  target-specific **selection**, not column deletion, is the control.
- Report all-sample and context/future errors in physical units and paired
  actual-endpoint effect errors. Keep the training-mean baseline.
- Report pusher-circle fully-in-canvas / partly clipped / wholly-outside geometry
  descriptively, never as an outcome-based inclusion gate. Pixel-hash collisions
  with differing labels are evidence of single-frame observation ambiguity, not
  necessarily a limitation of the full history-and-action interface.
- A small-edit operational resolution target is 4 native units for translation
  and 2.5° for orientation, one quarter of the minimum requested 16 units / 10°.
  These new tolerances are for diagnosis, not a retroactive pass of the pilot.
  Accurate actual-image decoding alone cannot qualify edited/predicted states.

## F2: autonomous feedback versus oracle actual-history prediction

**H-F:** useful local prediction is obscured by feeding predicted states back.
**Complement H-F0:** even real-history one-step prediction is weak on this support.

For 11 actual encoded frames Z and 10 packed action chunks A, at offset s use
Z[s:s+3], A[s:s+3] to predict Z[s+3]. Compare autonomous prediction against initial
persistence and actual-history one-step prediction against **local** persistence.
Never compare teacher forcing only with the easier stale initial persistence.

DINO has recurrent auxiliary channels despite dummy observation proprioception.
Run the full 2×2 control: free versus actual visual history, crossed with predicted
versus reset auxiliary channels. Reset means **encode_proprio(0)** at every
history position, not literal zero. Preserve all patch-varying predicted channels
in recurrent conditions. For LeWM only the visual factor exists.

Required invariants: no-reset diagnostic recurrence matches the public adapter;
all variants agree at step 1; altering later actual observations never changes
free predictions; original parameters/buffers stay frozen. Teacher forcing is an
oracle localization experiment using extra observations, **not** a deployable
forecast improvement. Report visual and auxiliary contributions separately.

## F3: action-effect fidelity, not only action sensitivity

**H-A:** predicted differences track real differences between executed candidates.
**Complement H-A0:** action-dependent output can still have wrong direction/gain.

Use candidate 0's real hold outcome as the reference. For every candidate k>0,
compare predicted effect (prediction k − prediction 0) with actual encoded effect
(endpoint k − endpoint 0). Primary contrast: squared effect error minus true
effect energy (the zero-effect baseline). No small-effect exclusions or unstable
per-sample normalization. Summarize aggregate energy/gain and dot products as
secondary quantities. Retain simulator-state and RGB effect magnitudes.

Construct prediction-to-actual candidate distance matrices. Compare the matched
diagonal with the mean over **all** off-diagonal assignments (no lucky shuffle).
Correct match beating wrong matches supports action specificity; ties or tiny
effects are retained. This is distinct from the old zero-action ablation, which
was scored against a nonzero-action outcome rather than its own physical target.

## F4: true-endpoint versus forecast goal ranking

**H-G:** the native goal metric ranks actual endpoints usefully, so errors in
forecast-based decisions implicate the forecast path. **Complement H-G0:** even
true-endpoint latent ranking is weak, consistent with goal/metric mismatch.

Use the unchanged checkpoint goal cost and same eight candidates. Compare actual
encoded endpoints and predicted endpoints; score chosen candidates with stored
true simulator coverage costs. Include expected uniform-choice regret, hold
regret, best available cost and full candidate cost range. Ties use the smallest
index. No physical readout enters the decision score.

Report all roots: zero regret in a bank with no goal coverage is **not** success.
Goal images also specify a pusher location whereas coverage concerns the block;
endpoint failure need not mean the block is unrepresented. The endpoint/native
comparison is not a guaranteed nonnegative additive causal decomposition or a
closed-loop planning evaluation.

## Uncertainty, reporting, and paper decision

Each root (or recorded episode) is equally weighted after averaging its correlated
branches/candidates. Report paired pointwise 95% bootstrap intervals with 5,000
resamples and the fixed follow-up seed. Horizon 8/original appearance is the
primary diagnostic slice; horizons 1/4 and shifted appearances are secondary.
All predeclared contrasts are retained. There is no multiple-testing-corrected
confirmatory significance claim, no optional stopping at a favorable interval,
and no comparison of raw latent MSE across the two feature spaces.

The paper may support an empirical diagnostic/negative-result contribution:
localizing which proposed physical claims survive measurement, feedback,
action-effect, and decision controls. Prior work already motivates these audit
principles; novelty must be restricted to verified checkpoint-specific findings.
If results remain inconclusive, report that explicitly rather than manufacture a
mechanism or keep tuning on the same pilot. Preserve code, settings, logs, every
fit seed, all metrics, and artifact checksums; keep data/model caches separate.