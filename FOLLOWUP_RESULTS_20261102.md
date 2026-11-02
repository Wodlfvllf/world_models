# Recovered Molab follow-ups — 2026-11-02

## Status

**All six fixed-budget follow-up reports completed and were downloaded and
restored locally with every inventoried file hash/size verified.** These add to,
not replace, the six prerequisite diagnostics. The original pilot was restored
unchanged; no regeneration, editor refit, or confirmation was performed.

- Host: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB reported memory;
  Python 3.11 / Torch 2.10.0+cu128. A real CUDA kernel passed during recovery.
- Original pilot ZIP: 929,322,621 bytes, SHA256
  `b7176b64ab0207b46482c3df1e411cbf03a3803671e8f2cc770a3b3986e8d476`.
  Fourteen transfer pieces were assembled and the original full hash verified
  before restoring. Interrupted uploads were retried, not accepted as complete.
- The prior six-diagnostic archive was restored with all 120 files verified.
- F1 source: `e828004dc9c0c7e71abd058de8b26211f933cb57`.
- F2–F4 and recorded-feedback source:
  `0ac3c719aa82eb852513704587b391a1b91a5617`, in a separate checkout.
- Both models ran concurrently within each batch. The attached GPU reached
  100% utilization during overlapping batches; no extra machines were allocated.
- Full implementation validation: **1,059 tests passed**, no failures/errors/skips,
  including synthetic and real-checkpoint recurrence/control tests.

The [follow-up protocol](FOLLOWUP_PROTOCOL_20261101.md) predates these runs but
postdates inspection of the original pilot. Findings are exploratory, not fresh
confirmation. Below, the primary slice is original appearance / horizon 8.
Intervals are pointwise 95% equal-root/episode bootstrap intervals, 5,000 resamples.
No best-seed selection or outcome-based root filtering was used.

## F1: bounded probe capacity did not rescue physical measurement

All 128 scorer / 64 validation / 128 pilot roots were retained. Selection was
fixed on original validation before pilot evaluation. Evaluation includes all
eight candidates: 6,912 images per appearance, of which 2,048 are horizon-8
images. This differs from the prerequisite readout's selected-candidate,
mixed-time 1,536-image summary; the tables are **not directly interchangeable**.

Native simulator units (512-unit arena), not resized pixels:

| Model / method | Pusher error | Block error | Angle MAE | Paired block-effect error | Paired angle-effect MAE |
|---|---:|---:|---:|---:|---:|
| LeWM group-selected ridge | 238.068 | 29.317 | 20.045° | 20.147 | 14.690° |
| LeWM MLP seed 11 | 245.868 | 40.670 | 18.919° | 27.121 | 13.978° |
| LeWM MLP seed 29 | 244.940 | 39.642 | 18.735° | 27.236 | 14.024° |
| LeWM MLP seed 47 | 250.180 | 38.571 | 19.218° | 26.110 | 14.597° |
| DINO group-selected ridge | 228.196 | 32.158 | 34.832° | 18.990 | 22.650° |
| DINO MLP seed 11 | 250.063 | 56.904 | 38.609° | 39.220 | 24.989° |
| DINO MLP seed 29 | 250.218 | 60.289 | 37.375° | 40.550 | 24.904° |
| DINO MLP seed 47 | 247.066 | 55.272 | 37.821° | 34.396 | 23.769° |
| Training mean, both | 304.812 | 139.466 | 92.866° | 32.603 | 11.733° |

Every MLP seed has worse block error than ridge (all paired intervals above
zero). Every fitted method has worse paired angular-effect error than the
zero-change prediction induced by the training mean. All remain far above the
new diagnostic 4-unit / 2.5° resolution targets. This supports the **bounded
H-M0** conclusion, not information absence or impossibility of better probes.

Observation ambiguity is real: 280 exact resized-RGB hash groups have differing
state labels, covering 630 samples. The empirical single-image scaled-MSE lower
bound is 3.6240 across all 6,912 original images. It is not a population bound,
an angular/distance-error bound, or a full-history limitation. At horizon 8,
650/2,048 pusher circles are wholly outside the canvas. Fully-in-canvas ridge
pusher errors still average 87.20 (LeWM) and 70.93 (DINO) units. No samples were
removed from primary analyses.

## F2: feedback localization requires a local-persistence control

| Data / model | Free MSE | Actual-history MSE | Local persistence MSE | Actual-history minus local [95% CI] |
|---|---:|---:|---:|---|
| Generated / LeWM | 0.336436 | 0.027945 | 0.037330 | −0.009385 [−0.013506, −0.005714] |
| Generated / DINO recurrent | 0.790017 | 0.155523 | 0.158343 | −0.002820 [−0.013084, 0.007157] |
| Recorded / LeWM | 0.177977 | 0.008404 | 0.208970 | −0.200566 [−0.239180, −0.164919] |
| Recorded / DINO recurrent | 0.121063 | 0.042103 | 0.260635 | −0.218533 [−0.242758, −0.195422] |

LeWM has useful local prediction under this control. DINO's large oracle/free
improvement is **not** sufficient evidence that useful local prediction is merely
hidden by autoregressive feedback: its generated local-persistence contrast
includes zero. Both models have clear local advantages on the recorded
pretraining distribution, not held-out generalization.

DINO's full visual-history × auxiliary-channel control gives generated MSE:

| History | Recurrent auxiliary | Encoded-zero reset auxiliary |
|---|---:|---:|
| Free | 0.790017 | 0.790038 |
| Actual | 0.155523 | 0.155538 |

Free reset-minus-recurrent is +0.00002046 [−0.00000693, 0.00005000]; actual
reset-minus-recurrent is +0.00001440 [0.00001122, 0.00001762]. The latter tiny
positive interval is not a practically meaningful rescue. The interaction
interval includes zero. Recorded reset effects are likewise small. Reset uses
`encode_proprio(0)`, not literal zero channels; recurrent conditions preserve
patch-varying predictions. Actual-history observations are privileged oracle
information, not a deployable forecast improvement.

## F3: latent action-effect fidelity survives stronger controls

| Model | Effect error minus zero-effect baseline [95% CI] | Signed aggregate gain [95% CI] | Matched minus all wrong assignments |
|---|---|---|---:|
| LeWM | −0.377371 [−0.434371, −0.322418] | 0.750347 [0.723617, 0.775084] | −0.525039 |
| DINO | −0.275802 [−0.321397, −0.231579] | 0.695333 [0.677847, 0.712799] | −0.562303 |

Both assignment intervals exclude zero. These results support action specificity
and nontrivial effect fidelity in each checkpoint's latent metric, beyond merely
changing output when controls change. Raw latent errors cannot rank the models.
Predicted/true aggregate effect energy ratios are approximately 1.017 / 1.089;
gains below one therefore do **not** imply a simple insufficient-amplitude error.
No block-specific physical correctness follows without qualified measurement.

## F4: better forecast error does not guarantee better candidate ranking

| Model | True-endpoint regret | Forecast regret | Forecast minus endpoint [95% CI] |
|---|---:|---:|---|
| LeWM | 0.020309 | 0.026390 | +0.006081 [−0.001434, 0.014495] |
| DINO | 0.039812 | 0.026252 | −0.013560 [−0.022595, −0.005049] |

Expected uniform-choice regret is 0.052019 and hold regret is 0.027938.
True-endpoint-minus-uniform is −0.031710 [−0.042918, −0.021103] for LeWM and
−0.012207 [−0.023840, −0.000516] for DINO; the latter is marginal in this
multi-contrast exploratory audit. Both forecast rankings beat uniform, but neither
establishes improvement over hold.

DINO's forecasts rank better than its actual encoded endpoints in this bank.
This contradicts a simple nonnegative decomposition in which good endpoint
geometry is degraded only by forecasting; it does not identify a causal mechanism.
Candidate quality is poor: 127/256 root–branch banks have no positive goal
coverage and mean best available coverage is only 11.28%. This is finite-bank
ranking, not closed-loop control success.

## Numerical reproducibility and interpretation limits

Native recurrence matches the public adapter at runtime on every evaluated bank;
model parameter and buffer digests remain unchanged. The new runner batches
candidates and reuses encoded trajectories, unlike the older diagnostic layout.
Matched individual native-error records differ from the prior run by as much as
0.028922 (LeWM) / 0.013884 (DINO), and individual persistence errors by up to
0.002766 / 0.001594. Aggregate native/persistence estimates differ by at most
0.000146, without changing conclusions. Backend/batching numerics are a plausible
but **not experimentally isolated** explanation. Do not claim bitwise replication
or conceal the per-record differences; changed bootstrap seeds explain interval
changes, not point-estimate drift.

The strongest supported synthesis is a separation between unqualified physical
measurement, useful latent action specificity, and limited goal-ranking evidence.
Neither the original editor-transfer hypothesis nor a universal absence-of-physics
claim is established. Fresh-root replication and stronger candidate-bank support
remain necessary for stronger claims. No adaptive extra fits were run to obtain
favorable outcomes.

## Durable artifacts

All three ZIPs and adjacent receipts are saved locally; generated artifacts are
not committed to Git. Original input backup and model/H5 caches are separate.

| Archive | Verified files | SHA256 |
|---|---:|---|
| [F1 probes](runs/molab-20261102-retrieved/followup-probes-20261102-verified.zip) | 48 | `678ed416c59dfb383b0b64b92de234a82d99b872187bb2ac3514f37d74eb1fe7` |
| [F2–F4 dynamics](runs/molab-20261102-retrieved/dynamics-20261102-verified.zip) | 41 | `538e62131fcf4f4139cfb86fa8abea4817297e18a9e02d76bbb08805208482d9` |
| [Recorded feedback](runs/molab-20261102-retrieved/recorded-feedback-20261102-verified.zip) | 43 | `71892b690d0d365f2e56483af28bd6e97a24fcee9deb0db0b6166c09a9f1f227` |

The inventories include reports, compact records/predictions, source snapshots,
selection/training artifacts, launchers and logs. A read-only recovery monitor is
provided in [notebooks/monitor_followup_20261102.ipynb](notebooks/monitor_followup_20261102.ipynb).

The [summary figure](runs/molab-20261102-retrieved/figures/followup_summary.pdf)
is reproducible with [scripts/plot_followup_summary.py](scripts/plot_followup_summary.py).
Its 36 selected statistics/components match the source reports; 52 plotting and
integrity tests passed in addition to the 1,059-test implementation run. The
final hosted check showed no active GPU processes; Molab was left open, not
shut down. The local verified archives—not notebook in-memory state—are the
durable record.