# Tracked numerical audit and fresh-root replication — 2026-11-03

All pending fixed-protocol studies completed on the recovered Molab GPU. No
discovery experiment, editor fitting, world-model training, seed search, or
recorded-data repeat was performed. An inconclusive primary result is retained.

## Execution and traceability

- Run ID: `paper-studies-20261103-01`.
- Source commit: `36d6c1f4b744d808d1955ed50a3f7bd72280ba31`, published and remotely
  verified before execution; all qualified core-module hashes unchanged.
- Host: https://sb-6a039c950594bc39.sb.molab.run/.
- Project: `/marimo/world_models_restored_20261103`.
- Batch directory: project-relative `runs/paper-studies-20261103-01`.
- Launch: 2026-11-03 07:17:46 UTC. Terminal status: 07:41:06 UTC.
- Order: LeWM numerical audit, DINO numerical audit, joint fresh-root replication.
  Numerical studies were isolated from concurrent GPU workers.
- Every stage exited successfully and passed its integrity checks. A separate
  final operations archive records the terminal batch state; earlier immutable
  stage archives intentionally retain the then-current global status.
- Local validation: 198 pending-study/tracking tests passed, followed by 19
  hardened-supervisor tests; then **1,312 full-suite tests passed**, zero failures,
  errors or skips. [Full-suite receipt](runs/verification-tests.xml).

The [central tracker](EXPERIMENT_TRACKER.md), [launcher](scripts/run_tracked_studies.py),
[numerical protocol](NUMERICAL_AUDIT_PROTOCOL_20261102.md), and
[replication protocol](REPLICATION_PROTOCOL_20261102.md) specify the scope.
Completion is separate from support for a hypothesis.

## N01–N02: Numerical execution-layout audit

Both models evaluated the same first eight roots in serialized original pilot
manifest order, original appearance, both branches and all eight candidates.
Each report retains 7,680 rows: 60 complete comparison groups, 128 paired records
per group. Baseline, unchanged repeat, and four distinct backend controls were
executed; benchmark-off was already set and explicitly skipped.

The table reports maximum absolute changes in horizon-8 native MSE. Distances
are model-specific latent distances, not physical errors or a cross-model score.

| Comparison | LeWM | DINO release |
|---|---:|---:|
| Observed old versus new paths, baseline | 0.00489527 | 0.00153232 |
| Encoding changes, single rollout fixed | 0.00489271 | 0.00152278 |
| Rollout layout changes, new encoding fixed | 0.000002563 | 0.000009537 |
| Feedback wrapper versus public rollout, same reduction | 0 | 0 |
| Reduction only, identical forecast tensors | 0 | 0.0000001192 |
| Observed old/new paths, convolution IEEE control | 0.000002563 | 0.000006437 |
| Unchanged-layout repeat, every compared variant | 0 | 0 |

Encoding layout dominates the baseline discrepancy on this bounded sample.
Changing convolution precision permission from TF32 to IEEE reduces the maximum
observed-path MSE disagreement by approximately **1,910× / 238×**, respectively.
This establishes sensitivity to that control, not improved physical accuracy:
both forecasts and actual encoded reference targets can change. Matmul IEEE and
cuDNN deterministic controls produced no observed changes; matmul TF32 did.
Backend properties were restored and parameters plus all buffers remained
unchanged. Same-process repeats do not establish fresh-process or cross-device
bitwise reproducibility.

The earlier historical maxima (0.028922 / 0.013884) cannot be conclusively
attributed by this eight-root audit. Full latent tensors were not archived;
independent verification checks the stored record-level measurements and their
aggregation, not a reconstruction of every raw tensor.

[Independent numerical verification](runs/paper-studies-20261103-retrieved/numerical-independent-verification.json)
recomputed all **1,800 summary statistics per model** from every CSV row,
checked exact identities, source/input receipts and archive inventories. Maximum
summary rounding discrepancy was $1.11\times10^{-16}$; no checks failed. The
[read-only verifier](runs/paper-studies-20261103-retrieved/numerical-independent-verifier.py)
does not load models or simulate trajectories.

## R01: Fixed independent fresh-root replication

Exactly **128 fresh pilot roots** were generated with split seed `2026110201`,
indices 0–127. Original scorer/validation archives were copied byte-identically;
they were not refitted or encoded for this dynamics-only run. Root identities
were checked against all original splits and the reserved original/default and
software-preflight rosters. No roots, branches, candidates, ties, zero effects,
out-of-frame cases or no-coverage banks were removed.

Both frozen models evaluated every appearance, branch, candidate and future
step: **6,144 bank/horizon records per model**. The five primary contrasts use
original appearance / horizon 8, equal-root weighting after bank and branch
means, shared 10,000 PCG64 root-bootstrap draws, and two-sided **99% intervals**
for the fixed five-member Bonferroni family. This is a prospective local design
after discovery, not external preregistration or a new task distribution.

| Primary contrast | Model | Estimate | 99% interval | Decision |
|---|---|---:|---|---|
| Action-effect error minus zero-effect baseline | LeWM | −0.375268 | [−0.452880, −0.303452] | Supported |
| Matched minus mean of all wrong action assignments | LeWM | −0.517972 | [−0.584175, −0.454432] | Supported |
| Action-effect error minus zero-effect baseline | DINO | −0.236408 | [−0.288885, −0.183415] | Supported |
| Matched minus mean of all wrong action assignments | DINO | −0.560457 | [−0.596832, −0.524494] | Supported |
| Forecast minus actual-endpoint native-goal regret | DINO | −0.007112 | [−0.018928, 0.005073] | **Inconclusive** |

Thus **four of five** primary contrasts support their prespecified negative
direction. Both models' latent action fidelity and action specificity replicate
on this distribution. DINO's exploratory discovery ranking advantage **does not
receive independent primary-family support**. A point estimate in the expected
direction is not a successful replication when its prespecified interval spans
zero. No claim that all five findings replicated is justified.

F2 and all nonprimary appearances/horizons/metrics remain exploratory pointwise
95% estimates, retained in the complete joint report. Latent action fidelity is
not block-specific physical correctness; finite-bank regret is not closed-loop
control. F1 and recorded feedback were not rerun or independently replicated.

### Complementary exploratory results on the fresh roots

At original appearance / horizon 8, actual-history minus local-persistence MSE
is −0.011180 [−0.014788, −0.007810] for LeWM and +0.002896
[−0.006993, 0.012992] for DINO recurrent auxiliary history. Thus LeWM retains
useful local prediction while DINO's analogous contrast remains inconclusive.
These are pointwise 95% intervals, not additions to the primary family.

Signed aggregate action-effect gains are 0.764917 [0.732540, 0.795694] and
0.664418 [0.644143, 0.683983]; neither bootstrap has an undefined denominator.
Forecast regrets are 0.030412 (LeWM) and 0.023694 (DINO), against hold 0.027010
and uniform 0.050835. These descriptive differences alone do not establish a
forecast-versus-hold advantage. **129/256 banks have zero positive goal
coverage**, and none were excluded.

[Independent replication verification](runs/paper-studies-20261103-retrieved/replication-independent-verification.json)
checked all four archives (610 inventoried entries), final status/completion,
352 replication completion-inventory files, source pins, all 320 materialized
roots and roster separation. It independently regenerated the bootstrap draw
matrix and recomputed all five primary and **1,680 exploratory estimates/intervals**
from the 12,288 bank records. Maximum numerical discrepancy was
$2.27\times10^{-13}$; no integrity checks failed. All 24,576 selected indices,
candidate means and assignment matrices were checked. Physical effects/coverage
were checked against stored NPZ arrays without replay; raw latent measurements
cannot be regenerated without the unarchived latent tensors.

The [read-only replication verifier](runs/paper-studies-20261103-retrieved/replication-independent-verifier.py)
and full-precision receipt preserve the analysis. Model evaluation stages took
131.57 seconds (LeWM) and 570.18 seconds (DINO); generation/preflight/validation
component 295.70 seconds; replication completion elapsed 1,013.50 seconds. The
entire tracked batch launch-to-terminal-state took approximately 23.34 minutes,
including audit, authentication, inference, verification and archive sealing.

## Archive ledger

| Archive | Bytes | Inventoried entries | SHA256 |
|---|---:|---:|---|
| Numerical LeWM | 967,237 | 65 | `d244e69da1a693725b94491612e798bf3091ae8acbb230aa92eed610fafb98ee` |
| Numerical DINO | 940,404 | 65 | `4c84faef80b1557a9edfd836cea163c9304326268965ce9a6e05664d0d47c7ea` |
| Fresh-root replication, including all 320 materialized NPZs | 350,515,773 | 415 | `7e4ffb90d484e7fc5f736715d23ebb22e0af9da0ce83d38494a79748a557e27d` |
| Terminal operations and all stage logs | 351,221 | 65 | `22d49add52737c4cf9258f671e40c7eeced2d508377e07be444c6d45983dc31e` |

Source snapshots, fixed selections/plans, raw CSV/bank records, generated data,
bootstrap draws, model/source/input receipts, status, timestamped events, exit
codes and stage logs are retained—not just manuscript tables. The older generic
backup helper was not used. Archive bytes and receipts must agree before use.
Local copies are held under the retrieved batch directory; sandbox storage is
not durable. No additional experiment is needed to keep an idle GPU occupied.

Final [status](runs/paper-studies-20261103-retrieved/status.json),
[completion receipt](runs/paper-studies-20261103-retrieved/complete.json), and
[exact launch record](runs/paper-studies-20261103-retrieved/launch.json) are
downloaded. Strict extraction markers verify 65 + 65 + 415 + 65 inventory entries
in four archives. The final operations ZIP, not the earlier snapshot statuses,
establishes terminal success. The GPU was idle with no active compute processes
after the run.