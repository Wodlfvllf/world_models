# Local pilot backup analysis — 2026-11-01

## Update: completed DINO backup verified locally

The newer **18:36 snapshot** is now available. Its ZIP is in the first backup
folder; its matching receipt is in the second folder (`world-model-backups 2`).
Files were inspected in place without moving, extracting or overwriting them.

- Archive: `20261031T183641Z-c29ccb098268.zip`.
- Captured: **2026-10-31 18:37:20.894167 UTC**.
- Size: **929,322,621 bytes**.
- SHA256: `b7176b64ab0207b46482c3df1e411cbf03a3803671e8f2cc770a3b3986e8d476`.
- Whole-archive SHA256 matches the downloaded receipt.
- **983 inventoried files** pass CRC, size and SHA256 checks, with no duplicate
   names or unexpected payload files.
- Contains DINO inference, training, validation, pilot and final analysis JSON,
   the physical scorer, and all four editor checkpoints under
   `runs/pilot-finish-01/dinowm`.
- Contains LeWM's final analysis and earlier completed model artifacts, plus
   the shared generated data under the original pilot run.

**The completed pilot results are backed up.** The zero data-root counters for
the continuation run do not indicate missing model artifacts.

For both models, training/validation/pilot/report configurations agree, the
analysis references the exact SHA256 of its archived evaluation, and recomputing
selection from clean validation chooses **MLP**, matching the pilot and report.
Each has **1,728 validation records** and **3,456 pilot records / 128 pilot roots**.
No model execution or retraining was needed for these checks.

### Final comparison

| Diagnostic | LeWM | RGB DINO-WM |
| --- | ---: | ---: |
| Actual-image readout loss, original | 71.6784 | 520.6246 |
| Actual-image readout loss, mean shifted | 74.2265 | 649.3029 |
| Clean MLP minus identity effect loss | +1.0984 [0.7795, 1.4899] | +7.7819 [5.5398, 10.4458] |
| Transport interaction | +0.1396 [-0.9960, 1.2747] | -2.6203 [-6.6503, 1.3440] |
| MLP shifted-minus-original effect loss | +0.0023 [-0.0992, 0.1084] | +0.7225 [-0.9100, 2.5286] |

Brackets are pointwise 95% bootstrap intervals. Both readouts fail the draft
0.05 tolerance. Positive clean-minus-identity values mean editing is worse
under the current measurement. Neither model qualifies the proposed
appearance-specific physical editing interpretation.

DINO's readout training losses are **0.05169** (actual context), **3.34068**
(actual future), **0.53736** (native rollout), and **1.30991** aggregate. Its
held-out original-appearance actual-image loss is **520.6246**. Training and
evaluation sample different surfaces/candidate sets, so these are not an
apples-to-apples generalization ratio. Nevertheless, the disparity is a strong
reason to investigate readout conditioning/overfitting and distribution or
coverage differences, rather than attribute the measured failure solely to
world-model dynamics. None of those causes is established yet.

DINO MLP history-latent training MSE improves from **0.23281 to 0.20643**, while
its clean pilot physical-effect loss is **11.38846**, versus identity **3.60654**.
Latent fitting improvement again does not establish physical utility. These
unqualified physical scores must not be used to claim that one world model is
intrinsically better than the other.

The sections below preserve the earlier local-only analysis and its historical
availability statements; this update supersedes the statements that DINO's
detailed report is not yet available locally.

## Scope and integrity

The newer locally available snapshot is `20261031T181252Z-d911acc9d9e7.zip`,
captured at **2026-10-31 18:13:30 UTC**. Its SHA256 is
`0cddf5e3d86841ceea71bc66541d30f5d2ecb78c1e0efc631d07c25a273ce8aa`.

- All **970 inventoried files** pass ZIP CRC, size and SHA256 checks.
- No duplicate member names or unlisted payload files were found.
- Relative to the earlier receipt-verified 18:00 snapshot: **0 added, 0 removed,
  0 changed payload files**. The backup metadata/archive itself differs.
- The newer archive's receipt is absent locally. Internal consistency is checked,
  but this is not a new verification against a Drive-upload receipt. The earlier
  archive was rechecked against its receipt, and all payload inventory hashes match.
- Neither local snapshot contains the DINO continuation run. They contain complete
  LeWM fitting, validation and pilot evaluation, plus the earlier smoke outputs.

No archive code/checkpoint was executed, no models were rerun, and no data were
regenerated. The existing locally generated LeWM analysis was reused only after
checking its recorded evaluation SHA256 against the evaluation bytes in this ZIP.
Candidate costs were independently reread for all 128 pilot roots from this ZIP.

## LeWM results: 128 pilot roots

The clean-validation-selected editor is **MLP**. These are exploratory pointwise
bootstrap intervals, not a multiplicity-controlled confirmation.

| Diagnostic | Result | Interpretation |
| --- | ---: | --- |
| Original-appearance actual-image readout loss | 71.6784 | Far above draft tolerance 0.05; measurement is unqualified |
| Mean shifted-appearance actual-image readout loss | 74.2265 | Measurement remains unqualified |
| Clean MLP minus identity effect loss | +1.0984; 95% CI [0.7795, 1.4899] | Editing is worse under the current measurement |
| MLP shifted-minus-original effect loss | +0.00235; 95% CI [-0.09917, 0.10842] | No clear editor deterioration |
| MLP transport interaction | +0.13963; 95% CI [-0.99600, 1.27465] | Does not establish appearance-specific editor failure |

The effect losses below are sums of scaled squared physical-feature errors,
averaged over configured horizons/candidates and then roots. They are **not
success rates**, raw pixel distances, or trustworthy physical diagnoses when
the physical readout has failed its own qualification.

| Method | Original-appearance physical-effect loss (lower is better) |
| --- | ---: |
| Identity / no edit | 3.60654 |
| Paired ridge editor | 4.68528 |
| MLP editor | 4.70497 |
| Rollout-supervised editor | 4.91721 |
| Native counterfactual initialization | 20.26603 |
| Inverse-probe editor | 27.64979 |

Native initialization performing poorly under this scorer is a diagnostic warning,
**not proof that the frozen world model itself lacks physical competence**. The
readout, generated-data distribution, dynamics and their interaction are not yet
separated. An editor doing better than this native baseline does not rescue the
claim when it is worse than identity and the measurement is unqualified.

## Fitting versus physical utility

LeWM training history-latent MSE starts at **0.14618** and ends at:

- Ridge: **0.11469**.
- MLP: **0.11477**.
- Rollout editor: **0.11610**.
- Inverse editor: **0.97711** (worse).

Thus fitting did occur and several editors improved their latent reconstruction
objective. That improvement did **not** establish physical editing utility on the
pilot. Latent MSE and scaled physical losses are different quantities and should
not be compared numerically as if they shared units.

The LeWM readout also has substantial in-sample error: context **6.3704**,
actual future **29.5206**, native rollout **27.2110**, aggregate **21.0340**.
This is not only a problem introduced by appearance changes. The training
readout mixes actual and predicted latent surfaces, so its aggregate loss is
not a clean estimate of either perception or dynamics accuracy.

## Candidate-bank limitation

| Diagnostic | Factual | Counterfactual |
| --- | ---: | ---: |
| All eight candidates have cost exactly 1 | 65 / 128 | 62 / 128 |
| Mean best available candidate cost | 0.88746 | 0.88690 |
| Mean uniform-random candidate cost | 0.94041 | 0.93798 |
| Mean random-minus-best available cost | 0.05295 | 0.05108 |

Cost 1 means zero endpoint goal coverage. Nearly half the banks give no successful
coverage from any candidate; zero selection regret on those banks does not mean
successful control. All roots were retained, including these failures.

## Newer remote backup and remaining analysis

The user-provided status reports `20261031T182442Z-9530cc79abc3.zip` as already
**verified on Drive at 18:26:41 UTC**, with size **929,064,079 bytes** and SHA256
`fcb4672ffbcf84420abc60a0014af65af59c2d7d8f3d875516633377bcc2a348`.
It was not present locally at inspection time. That status alone does not list
which DINO artifacts were captured or establish that capture followed completion.

Zero `completed_roots` for `pilot-finish-01` is expected: the continuation reads
the original `pilot-gpu-01/data` rather than duplicating its root archives.
Root counts do not count finished model evaluations.

When the newer ZIP and receipt arrive, verify them and check for the DINO training,
validation, pilot and analysis reports under `runs/pilot-finish-01/dinowm`.
The separate console log shows DINO completed with 1,728 validation records and
3,456 pilot records, selected MLP, and failed the current physical-effect
qualification criteria. Its detailed numerical report still needs local review.

## Recommended next scientific steps (not executed here)

1. Finish the combined LeWM/DINO audit before changing the experiment.
2. Fit an **actual-image-only** readout on development data; measure held-out
   per-coordinate errors and appearance transfer independently of native rollouts.
3. Compare frozen native predictions with persistence on recorded PushT windows
   and on generated windows, at identical horizons/action conventions. The available
   original recording is pretraining data, so do not call it held-out world-model
   generalization evidence.
4. Diagnose action-bank/goal utility on development roots before specifying a new
   evaluation protocol. Preserve all current pilot roots and results.
5. Do not start a large confirmation run or tune margins/filter roots to obtain a
   positive claim. The current pipeline has not passed the necessary prerequisites.