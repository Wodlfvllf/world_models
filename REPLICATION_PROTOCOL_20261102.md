# Fixed fresh-root replication — 2026-11-02

**Prospective design after inspecting the discovery results, before generating
these roots.** This is a local dated commitment, not an externally registered
preregistration. No replication result is asserted here. Implementation and CPU
contract tests do not constitute a research run.

Implementation: [scripts/run_replication.py](scripts/run_replication.py).
Fixed shared-schema configuration:
[configs/replication_20261102.json](configs/replication_20261102.json).
Contracts: [tests/test_replication.py](tests/test_replication.py).
Discovery: [FOLLOWUP_RESULTS_20261102.md](FOLLOWUP_RESULTS_20261102.md).

## 1. Scope, fixed budget, and unchanged distribution

- Exactly **128 new pilot roots**, indices **0–127**, pilot split seed
  **2026110201**. The seed is fixed here, not searched for favorable outcomes or
  replaced after a failure. The original seed-derivation algorithm determines
  every root ID, root seed and independent anchor/intervention/goal/action/
  appearance/environment stream.
- Both original frozen releases: LeWM projected CLS and the Point-LeWM-author
  RGB/dummy-proprio DINO-WM. No training, editor fitting, checkpoint selection,
  changed normalization, or best-seed selection.
- One existing qualified F2/F3/F4 dynamics pass per model. The primary slice is
  **original appearance, horizon 8**. Every root, both branches, all eight
  candidates, all three appearances, and all eight future steps remain in the
  stored dynamics records. F2 and every nonprimary slice remain exploratory.
- **No recorded-data repeat. No F1 run in this implementation.** Original F1
  findings remain exploratory, not newly replicated. Its selected probe weights
  must not be refitted or reselected on these roots. A future frozen-F1 extension
  would need a separate authenticated tensor-checkpoint/selection loader and
  model/roster linkage; it is deliberately not smuggled into this budget.
- Preserve the original generation distribution, including its limitations:
  horizon 8; K=8; clearance 2; 256 maximum initial-anchor attempts; resting
  ten-control prefix; the same 13-cell intervention grid; the same goals and
  appearance distribution; candidate 0 hold, 1 approach, 2 away, and the same
  smooth random/biased remaining candidates. Native success does not truncate
  the fixed trajectory. Only the original anchor-geometry rejection is allowed.
- **No bank improvement, root screening, outcome filtering, repeated-seed
  search, tuning, optional stopping, or adaptive sample-size extension.** Shams,
  ties, zero effects, no contact, out-of-frame pushers, misses, future wall
  contact, and no-goal-coverage banks remain. Numerical/parity failures abort;
  they do not justify replacing a root or reporting a partial primary family.

The original archived generator differs from the current pinned generator only
by a progress import and wrapping its root loop with progress reporting. Its
sampling, action-bank construction and execution code are unchanged. The native
simulator wrapper is byte-identical. Relevant SHA256 receipts:

| Source | SHA256 |
|---|---|
| Archived original generator | `1a7e1b21e7eb93f3620b2f7077e7576726da0baa998d7566d3d086a59b3ecc3e` |
| Current progress-enabled generator | `0c5c48525cde96c3c7c3edf3fe3d361667470872b673ec66b770378f0ffaa5b4` |
| Original/current simulator wrapper | `ea0d5845d889adc6397abb63de737e5186067464fa93a83e871ef04dc73bf455` |
| Qualified discovery dynamics | `016d2bbe293c0bfc1d6a3c303c76f83a3adef88ebe34061e279070e7ab5dd14f` |

The runner pins all 16 core module hashes to the qualified discovery snapshot,
not merely these four. It also checks the original configuration/specification,
captures its own source, this protocol, the new config and dependency manifest,
and rechecks watched inputs/sources before publication. No core source is edited.

## 2. Efficient generation with honest provenance and clean split constraints

The standard CLI development generator always uses the default split seeds and
regenerates every nonzero development count. It must **not** be used with the new
config: this would regenerate old scorer/validation data and the old pilot.

The dedicated runner instead:

1. Authenticates the **entire original 576-root manifest**, SHA256
   `fc759e41bd66c48954f3d41568a9e48893ece47ae0a9719229109d08a8086485`.
   This is the manifest referenced in the authenticated discovery reports.
   It validates every listed NPZ, its byte hash, full schema, filename/identity,
   native generation contract, all five split counts and the absence of extra
   archives. The original roster is 256 editor / 128 scorer / 64 validation /
   128 pilot / 0 confirmation. Unused editor roots are **not** skipped.
2. Checks the planned fresh IDs, 128-bit root seeds and environment seeds against
   **every original split**, against the entire originally planned 3,384-root
   default roster (2000/600/256/128/400, including ungenerated confirmation), and
   against the known software-preflight root. Earlier default-seed smoke and
   partial runs are subsets of the reserved roster. Additional nonstandard
   historical data directories can be supplied repeatedly through
   `--exclude-data`; every NPZ is then validated, inventoried, hashed and included
   in the exclusion check. These additions can only abort for overlap, never
   change the fresh sample. Unknown/unprovided nonstandard historical rosters
   cannot be certified by this implementation.
3. Publishes the full root plan, all original identities and input/source hashes,
   protocol/config/source snapshots, and the fixed bootstrap draw matrix **before
   generation or model evaluation**.
4. Exclusively copies the **original 128 scorer and 64 validation archives byte
   for byte** into the new dataset. Their identities, pixels, runtime/source
   metadata and hashes are unchanged. No old pilot/editor is copied as a new
   pilot; no metadata is relabeled, invented, or patched. Retained scorer and
   validation archives remain available for an authenticated frozen-F1 extension,
   but their images are not encoded and no fit is invoked in this run.
5. Calls the unchanged `generate_splits` API with generation counts
   **0/0/0/128/0**, all five distinct seeds explicitly present, and only the pilot
   seed changed. All default generation parameters are retained. Exactly 128
   fresh roots are physically simulated; existing fit/validation roots are not.
6. Writes a **new composite manifest**, with shared-schema config counts
   **0/128/64/128/0** and every actual NPZ hash. Separate provenance fields identify
   reused versus generated splits, the original manifest hash, actual generation
   arguments, simulator source and publication runtime. The original manifest
   itself is also retained byte-identically as evidence. This is a new manifest
   for a new dataset, not a copied old generation receipt with substituted roots.
7. Validates all 320 materialized roots and their cross-split separation, exact
   planned pilot identities and unchanged scorer/validation identities before
   loading either model. The qualified dynamics code independently validates its
   complete pilot manifest/schema again. All original and copied inputs, source
   snapshots, generated data, reports and fixed resamples are rehashed before
   the joint result is published. Extra/duplicate/unlisted files and changed
   inputs fail closed.

All outputs are exclusive, including pre-existing empty directories, interrupted
runs, files and dangling symlinks. No implicit resume or overwrite option exists.
Technical retries require a new directory, **the same fixed roots**, disclosure
of prior attempts, and no outcome-based decision about whether to retain a run.

## 3. Five-member primary family and exact estimands

Let $r$ index the 128 roots, $b\in\{0,1\}$ the two branches and $k=0,\ldots,7$
the candidates. At original appearance / h8, let $P_{rbk}$ be the frozen native
forecast, $A_{rbk}$ the actual encoded endpoint and $d(x,y)$ the mean squared
distance over **all** coordinates of that model's audited latent surface.

### Action-effect fidelity, separately within each model (two claims)

For $k>0$, set $\Delta P_{rbk}=P_{rbk}-P_{rb0}$ and
$\Delta A_{rbk}=A_{rbk}-A_{rb0}$. The bank contrast is

$$
F_{rb}=\frac{1}{7}\sum_{k=1}^{7}
\left[d(\Delta P_{rbk},\Delta A_{rbk})-d(0,\Delta A_{rbk})\right].
$$

The stored metric is `effect_error_minus_zero`. A negative population mean favors
the forecast's action effects over predicting no effect. Zero/tiny true effects
are retained without a normalization floor.

### Action specificity, separately within each model (two claims)

$$
S_{rb}=\frac{1}{8}\sum_{k=0}^{7}
\left[d(P_{rbk},A_{rbk})-\frac{1}{7}\sum_{j\ne k}d(P_{rbk},A_{rbj})\right].
$$

The stored metric is `matched_minus_all_wrong`. This includes the hold candidate
and **all** wrong assignments, not a selected shuffle. Negative favors matched
actions. The runner checks candidate counts, stored bank means, the complete
8×8 assignment matrix and both action-contrast definitions before aggregation.

### DINO forecast versus actual-endpoint candidate ranking (one claim)

Each ranking uses the original checkpoint goal score, with smallest-index exact
ties. Score its selected candidate by native coverage cost at the same horizon.
Regret is selected true cost minus the minimum true cost among all eight.
The contrast is forecast regret minus actual-endpoint regret, stored as
`forecast_minus_endpoint_regret`; negative favors forecast ranking. This is
**DINO only** in the primary family. LeWM's analogous contrast is exploratory.
The runner independently checks scores, chosen indices, true-cost bounds and
regret calculations. Zero-coverage banks remain and are explicitly counted.

For every contrast, average within the candidate bank as above, then average the
two branches for each root, then take the unweighted mean of the **128 root
means**. Neither branches nor candidates, appearances, frames, or bootstrap
replicates are independent experimental units. The same root order is used for
both models. No comparison of raw latent distances across models is primary.

## 4. Fixed uncertainty, decisions and complete secondary reporting

- Draw **10,000** bootstrap samples of 128 roots with replacement using
  **NumPy PCG64 seed 20261102**. Root order is sorted full root ID; the matrix is
  generated once with `Generator.integers(..., dtype=int64)`, saved before
  generation, checksummed, and reused across **both models and every metric**.
- The five primary contrasts receive two-sided percentile **99%** intervals:
  quantiles **0.005 and 0.995**, NumPy linear interpolation. This is
  Bonferroni $0.05/5=0.01$ per interval, targeting familywise error 0.05 for the
  one five-member family. Bootstrap coverage/FWER is nominal/asymptotic, not a
  finite-sample guarantee; 10,000 draws give about 50 draws per extreme tail.
- A claim supports its prespecified negative direction **only if its entire
  99% interval is strictly below zero**. An entirely positive interval is an
  opposite-direction result. Touching or spanning zero is inconclusive, not
  evidence of absence. Report all five estimates, intervals and decisions even
  when any fail. A model's combined fidelity/specificity statement requires both
  of its claims; an “all primary claims replicated” statement requires all five.
- All F2 controls, auxiliary resets/interactions, all other horizons/appearances,
  all F3 energies/dot products/gains/physical/RGB magnitudes, and all F4 costs,
  regrets, hold/uniform contrasts and coverage diagnostics receive **pointwise
  exploratory 95%** intervals (0.025/0.975), using the same paired root draws.
  Even the five primary metrics' pointwise views are labeled exploratory; only
  their 99% views drive primary decisions. No extra unadjusted primary tests.
- Aggregate effect gain remains the ratio of equal-root mean dot product to
  equal-root mean true effect energy, not a mean of per-root ratios. Zero-energy
  roots stay in the sample. If any bootstrap denominator is zero, the
  unconditional gain interval is null and the number of undefined draws is
  reported; undefined draws are not discarded or replaced.

The dedicated joint analysis is authoritative for primary inference. The
unchanged underlying dynamics reports and HTML retain their legacy exploratory
tags and “previously inspected pilot” boilerplate. **Those strings do not
describe the fresh roots.** They are preserved rather than rewritten; the new
plan, data manifest and joint analysis establish the new sample and primary
family. The legacy `confirmatory=false` follow-up schema is not weakened.

## 5. Hosted execution, source delivery and preservation

Research execution refuses macOS/CPU/MPS and CUDA fallbacks. Only CPU synthetic
tests are intended locally. Source hashes are those verified at repository
revision `e6f2423fc3469d216e27bb07d0bd73c92586e1c0` and in the completed discovery
dynamics reports. The runner, configuration, tests and this protocol are now
being published together with [the durable tracked launcher](scripts/run_tracked_studies.py).
Deployment must verify the exact published commit without replacing old runs.
The [experiment tracker](EXPERIMENT_TRACKER.md) records the actual execution and
archive receipts; this protocol alone is not evidence that a run completed.

The following paths are from the prior authenticated hosted run, not a claim
that its sandbox is still alive. If it has terminated, restore the authenticated
original backup, the pinned source tree, these additions, an isolated CUDA
environment and the pinned model/simulator caches first. No bootstrap, missing
artifact download, GPU allocation or archive restoration is implicit in this
runner. On a recovered host, substitute the verified locations only; do not
change scientific settings. Example on the prior layout:

```bash
cd /marimo/world_models_dynamics_20261102
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /marimo/world_models_followup_20261102/.venv/bin/python -u scripts/run_replication.py \
  --config configs/replication_20261102.json \
  --original-data /marimo/world_models_followup_20261102/runs/restored-pilot/runs/pilot-gpu-01/data \
  --cache /marimo/world_models_followup_20261102/cache \
  --output runs/replication-20261102-01 \
  --device cuda --threads 4 --trust-checkpoint --execute
```

Omitting `--execute` performs source/input authentication and prints the fixed
plan without creating outputs, generating roots, loading models or running GPU
inference. It still validates all original archives, rather than silently
trusting a partial roster. The two models execute sequentially on the attached
GPU; the runner never provisions another device or starts overlapping jobs.
Progress includes stage/device, completed/total, elapsed/ETA and heartbeats.
There is no training loss because no training occurs.

The output contains the pre-outcome plan/snapshots, original manifest, immutable
bootstrap draws, composite dataset and data-validation receipt, both untouched
dynamics reports, joint primary decisions plus every exploratory interval and
root vector, and a final recursive checksum/size inventory. **Only the final
completion inventory means the whole run completed.** Earlier per-model reports
are partial results. The older generic backup helper does not recognize these
new stage names; preserve the entire output tree and verify its inventory in a
separate durable archive, plus the original inputs and model cache separately.
A local completion marker is not a verified Drive snapshot.

## 6. Runtime scale and limitations

Generation is 128 new roots rather than 576 (22.2% as many), plus byte copies of
192 existing roots. That is 2,048 branch/candidate native trajectories and
102,400 primitive controls at the fixed 50-control length, with three render
conditions; no extra simulator episode is generated to choose a better bank.
Each model encodes **51,840 frames** (128 × 3 × (6 context + 1 goal + 128 future)),
evaluates 768 branch/appearance banks and publishes **6,144 bank/horizon
records**. Both models reuse the exact same generated data.

The previous same-sized hosted dynamics loops took approximately **325 seconds
LeWM / 643 seconds DINO** on an RTX PRO 6000 Blackwell, with concurrent workloads.
Their sum is a rough **16-minute sequential inference scale**, not a benchmark
of this new runner. Add generation, full 576-root authentication, 320-root
validation/copies, model loading, bootstrap and checksum time. Generation time
has not been measured for these new roots and must not be estimated by running
research locally. New NPZ compression may vary; the old pilot occupied about
122 MiB and old scorer/validation about 183 MiB, with about 7.3 GiB of raw RGB
represented by the new pilot alone. Plan storage for both full reports and an
independent verified backup, not only the compressed input NPZs.

This replication can support checkpoint-specific latent action fidelity,
specificity and DINO finite-bank ranking findings on the **same discovery
distribution**. It cannot establish physical readout qualification, information
absence, a causal decomposition of ranking error, robust closed-loop success,
generalization to richer candidate banks/tasks, or bitwise numerical portability.
Preserving the weak bank is deliberate replication, not an endorsement of its
coverage. F1, recorded feedback and the separate numerical audit are outside
this primary family. No paper or numerical-audit source is changed by this work.