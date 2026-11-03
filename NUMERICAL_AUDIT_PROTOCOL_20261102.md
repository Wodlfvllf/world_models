# Numerical execution-layout audit — 2026-11-02

## Status and scope

Implementation and CPU-only synthetic contracts; **no numerical research audit
has been run locally or on a GPU as part of this change**. This is a bounded
diagnostic of the previously disclosed maximum matched native-MSE differences
(LeWM 0.028922; DINO 0.013884; aggregate differences at most 0.000146). Those
historical numbers are context, not a result or acceptance threshold for this
audit. Batching/backend arithmetic is a hypothesis, not an established cause.

Only these new files are needed:

- [scripts/audit_execution_layout.py](scripts/audit_execution_layout.py)
- [tests/test_execution_layout.py](tests/test_execution_layout.py)
- [NUMERICAL_AUDIT_PROTOCOL_20261102.md](NUMERICAL_AUDIT_PROTOCOL_20261102.md)

No core-file changes, fitting, simulator execution, regenerated inputs, editor
evaluation, new roots, adaptive expansion, or outcome filtering. Publishing the
runner and its tracking infrastructure does not change the audited core paths.
The command line rejects CPU/MPS and unavailable CUDA; synthetic tests exercise
internal functions using explicitly labeled test doubles, never release weights.

## Fixed inputs, before looking at results

Use the **first eight entries in serialized pilot-manifest order**, original
appearance, both factual/counterfactual branches, all eight candidates, native
rollouts through horizon 8, with horizon-8 prediction/persistence metrics.
There is no root-count, offset, candidate, seed, or horizon-selection CLI option.
All ties, holds, low-effect and no-coverage cases remain. No outcomes enter
selection, including the decision to run backend controls.

The original [development manifest](runs/recovered-20261031/runs/pilot-gpu-01/data/manifest-development.json)
has SHA256
`fc759e41bd66c48954f3d41568a9e48893ece47ae0a9719229109d08a8086485`.
The CLI pins that entire file, requires its configuration to match, then
authenticates and schema-validates the first eight NPZ files. Their root indices
are 0–7 in the actual manifest. File iteration is **not** re-sorted by the audit.
Missing/corrupt selected roots fail; the runner never substitutes later roots.
Only those eight payloads are read, so this is not a new verification of all
128 pilot payloads. Selected roots retain the existing cross-root identity/
environment-disjointness validation.

Use the existing [pilot configuration](configs/pilot.json) and
[follow-up specification](configs/followup_20261101.json). Required observed
settings: eight candidates, horizon 8, old evaluated horizons `[1, 4, 8]`, old
encoder frame limit 4, new encoder frame limit 32. The whole configurations,
their file hashes, the manifest, selected NPZ hashes/identities and original
pixel/action-array hashes are retained. The whole root metadata is retained,
including simulator/generation provenance; it is not replayed or regenerated.

## Source-observed paths, not hypothetical layouts

The old path was inspected in the retrieved
[paired diagnostic snapshot](runs/molab-20261101-retrieved/complete/paired/src/wm_audit/diagnostics.py)
and its [pipeline helper](runs/molab-20261101-retrieved/complete/paired/src/wm_audit/pipeline.py).
Its source hashes also appear in the actual archived report. The new path was
inspected in the retrieved
[follow-up dynamics snapshot](runs/molab-20261102-retrieved/dynamics/src/wm_audit/followup_dynamics.py),
from commit `0ac3c719aa82eb852513704587b391a1b91a5617`.

| Source | Old paired snapshot SHA256 | Current/new executed SHA256 |
|---|---|---|
| [models](src/wm_audit/models.py) | `de993da1a611580b02aac0f72a17937c50718fd12499a6ccb741e9c6be9cbc1f` | `6bc28315d63ca1000bec79f4c2ee3d1cac143315a7f8e557a905706c13337b24` |
| [pipeline](src/wm_audit/pipeline.py) | `8de45cd7be3f75d786afe31b56b528fe7cc3177a44dc66c30608c24b88d6d9ad` | `5572870c6bd8ecde70ff61951cd4b28c55a7cf458b58c915276dacaaa546dbbd` |
| [diagnostics](src/wm_audit/diagnostics.py) | `c0d5eb84e8ce7f241887c97c1cc88c022a98291aa35de394546b5bd473474d60` | `ba9f703163f5d71b81d3b9b8f6978b680cbadb19ab6b8fafc1fd3202157b7c4b` |
| [follow-up dynamics](src/wm_audit/followup_dynamics.py) | Not present | `016d2bbe293c0bfc1d6a3c303c76f83a3adef88ebe34061e279070e7ab5dd14f` |

The old/current model and pipeline differences add the recorded-action opt-in
and validation; they do not change generated-action normalization, encoding or
native recurrence arithmetic. The old/current diagnostic differences add
recorded-data validation/provenance and raw-action diagnostics, not the generated
tensor path. The audit imports current functions rather than executing an old
source tree with outdated recorded-data support. A SHA guard rejects stale or
modified versions of the four executed core files; there is no bypass option.

### Observed old path

`prediction_diagnostic()` calls `_encode()` on original context `[2,3,...]`:
flatten six frames, adapter calls `[4,1,224,224,3]` then `[2,1,224,224,3]`.
For each branch/candidate, encode only future frames 1/4/8 together in one
three-frame adapter call. `_prediction_rows()` calls the public native adapter
with one candidate, then the zero-future-action control, preserving both prefix
chunks. Both calls are retained here, in their original order. Capture records
the **actual first returned rollout**, not a separately recomputed surrogate.
Reported metrics come from the original scalar `.square().mean()` expression.

### Observed new path

`dynamics_followup()` encodes original context as one six-frame adapter call,
then the original one-frame goal. Each branch's `[8,8,...]` future is flattened
candidate-major into 64 frames and encoded in two 32-frame calls. The goal call
is retained even though this audit does not use goal-ranking metrics.

`actual = cat(expanded_context, future, dim=1)` has 11 frames per candidate.
`feedback_rollouts()` runs all its existing free/oracle and DINO auxiliary
controls, including its **existing public-adapter parity call**. The audit
captures both the actual free-feedback tensor and that public call's tensor.
It does not replace the new recurrence with an assumed equivalent implementation.
The function's mandatory runtime parity assertions remain active and failures
prevent publication. DINO's recurrent patch-varying auxiliary channels are
unchanged. Reported new MSEs come from `feedback_errors()`'s batched float32
`.square().flatten(1).mean(1)` expression.

For the bank public call, history is `actual[:, :3]`, not merely a zero-stride
expanded context. Its first stride is `11 * D`, where `D` is the number of latent
coordinates per frame. The controlled bridge uses the same concatenation and
slice, matching the observed history layout. It uses existing new-future values
as **unread padding** for old-encoded histories; only the first three frames are
passed to the public adapter. Tests verify future/goal independence of native
predictions. The rollout factor therefore includes observed bank size **and**
associated memory layout, not a claim to isolate GEMM batch size alone.

## Exact comparison matrix

Within every suite, encode each observed path once and reuse those exact latent
tensors. `old_single` is the old native result; `new_bank` is the public parity
result captured inside the new feedback function. The crossed cells are
explicitly controlled bridges, not claimed historical executions:

| Encoding | Single candidate / public adapter | Eight-candidate bank / public adapter |
|---|---|---|
| Old observed | `old_single` | `old_bank` |
| New observed | `new_single` | `new_bank` |

All four public cells use the same scalar metric expression. Additional outputs
are `new_feedback` with the observed new metric, and `new_feedback_scalar` with
**the same feedback tensor** scored using the old scalar metric expression.

Every contrast below uses right minus left for scalar metrics; tensor deltas
report maximum and mean absolute coordinate difference:

| Comparison | Left | Right | What changes |
|---|---|---|---|
| `observed_paths` | `old_single` | `new_feedback` | Observed encoding, rollout layout, recurrence wrapper and metric call layout |
| `encoding_at_single` | `old_single` | `new_single` | Encoding layout only, public single-candidate rollout fixed |
| `encoding_at_bank` | `old_bank` | `new_bank` | Encoding layout only, public bank rollout fixed |
| `rollout_at_old_encoding` | `old_single` | `old_bank` | Public rollout layout only; exact old context and targets reused |
| `rollout_at_new_encoding` | `new_single` | `new_bank` | Public rollout layout only; exact new context and targets reused |
| `feedback_vs_public` | `new_bank` | `new_feedback_scalar` | Actual feedback/public recurrence outputs, same scalar metric expression |
| `metric_reduction_only` | `new_feedback_scalar` | `new_feedback` | Scalar versus row-batched metric kernel on identical latent values |

The **entire suite is independently re-executed twice under unchanged incoming
backend settings for each root**. No re-seeding, cached output substitution,
averaging away differences or forced deterministic defaults are used. Each of
the five main variants is compared with its own repeat. The repeat includes the
same encode, goal, zero-action, feedback and bridge call ordering.

## Backend controls and default-state protection

Baseline performs no backend setters. The report captures actual incoming
settings, every executed suite's settings, unavailable getters and skipped
controls. All controls are planned before inference; a property already at the
requested value is explicitly skipped, not presented as an independent
experiment. Applied properties can still have no numerical effect if an
environment override or kernel choice makes that permission irrelevant.

Each feasible control starts from and returns to the incoming state, changing
**one** property:

| Control | Sole requested change |
|---|---|
| `matmul_ieee` | `torch.backends.cuda.matmul.fp32_precision = "ieee"` |
| `matmul_tf32` | `torch.backends.cuda.matmul.fp32_precision = "tf32"` |
| `cudnn_conv_ieee` | `torch.backends.cudnn.conv.fp32_precision = "ieee"` |
| `cudnn_deterministic` | `torch.backends.cudnn.deterministic = True` |
| `cudnn_benchmark_off` | `torch.backends.cudnn.benchmark = False` |

Modern precision setters are used without mixing in deprecated TF32 setters.
Legacy getters and matmul precision are also recorded for visibility, including
API incompatibility messages if relevant. In Torch 2.10, even merely reading
`get_float32_matmul_precision()` can raise during a new per-backend override;
that getter's unavailability is recorded rather than mixing setter APIs.
Restoration includes the original
`"none"`/inherited precision value, not just an assumed effective boolean.
Missing APIs are marked unavailable; a setter or computation failure aborts
rather than silently dropping a control after seeing its results.

Each active setting executes the same seven within-setting contrasts and five
same-layout comparisons against the first baseline suite, holding pixels,
actions, weights and identities fixed. These backend comparisons change the
single property for **both encoding and rollout**; they are not a claim that
only a particular encoder, predictor or kernel used TF32. Repeats are the two
default-state suites, not additional repeats of every backend intervention.
`--no-backend-controls` intentionally requests only the baseline matrix and
its independent repeat and is recorded as such.

All touched flags are restored on success **and exceptions**, and restoration is
checked. No environment variables, global deterministic-algorithm settings,
SDPA flags or precision defaults are silently forced. Existing autocast is
rejected rather than silently altered. The functions also restore incoming
training/gradient flags; weights remain eval/frozen during inference. Joint and
separate parameter/buffer SHA256 receipts include scalar/integer/bfloat16 bytes,
aliases and nonpersistent DINO causal masks. The joint hash is checked after
every suite and again before publication. No checkpoint is saved or overwritten.

## Measurements, output and failure semantics

Each comparison has exactly **128 matched horizon-8 records**:
8 roots × 2 branches × 8 candidates. Matching uses the full root ID, appearance,
branch, candidate and horizon, never row position or a metric-based join.
Missing identities, duplicate/incomplete banks, shape mismatches or nonfinite
values fail closed.

For each matched record, measure actual tensor deltas for:

1. All three encoded context frames.
2. Actual encoded future frames at the common horizons 1/4/8.
3. The actual encoded horizon-8 target separately.
4. All eight predicted future latent tensors.
5. The predicted horizon-8 tensor separately.

Report both sides' native and initial-persistence MSEs, their signed difference,
and aggregate maximum/mean absolute differences. Retain the float32 metrics
from the observed paths **and** a common float64 CPU MSE computed from captured
float32 tensors on the GPU host, so changed model outputs can be distinguished
from a changed metric kernel. These are native per-coordinate MSEs, not the
LeWM summed goal cost. Raw errors cannot rank LeWM against DINO.

Tensor means have equal coordinate counts per record. The same context is
represented for each candidate; this duplication does not change its mean
because every bank is complete. There are exactly 16 records per root, so record
means also give equal-root means. Maxima never replace or disappear into signed
aggregate means. No bootstrap, significance claim or fitted interpretation is
added to this eight-root diagnostic.

The output directory must **not already exist**, even if empty or a dangling
symlink. Files are created exclusively:

- An immutable pre-inference selection/plan JSON.
- One compact per-comparison, per-record CSV (no full latent dump).
- A JSON completion report written **last**, with summaries, input/model/source
  receipts, actual call shapes/strides/dtypes/devices, output tensor hashes,
  actual backend settings, runtime/GPU information, output artifact checksums
  and limitations.

With controls disabled: 12 summaries and 1,536 CSV rows per model. Every active
backend control adds 12 summaries and 1,536 rows. Processing is sequential and
bounded to one root at a time; tensors are retained only for current-root
comparisons, not the whole experiment. Progress reports stage, device,
completed/total, elapsed time and ETA, with heartbeat messages around suites.
There is no training loss because there is no training.

The manifest, configuration/spec files, selected NPZ files and executed source
files are rehashed before publication. Model/input mutation or an exception
leaves no completion report. A partial directory is evidence of an incomplete
attempt and is never resumed or overwritten; investigate and use a new name.

## GPU-host deployment and command

1. Use an **idle, qualified Molab/Marimo CUDA host**, ideally the prior RTX PRO
   6000 Blackwell / Python 3.11 / Torch 2.10.0+cu128 environment. Record the driver
   and other GPU processes externally; the script reports its actual logical
   CUDA device, name, UUID text, capability, memory, CUDA/cuDNN/PyTorch builds,
   relevant TF32/workspace environment variables and backend flags. Do not
   interpret logical ordinals as physical GPU UUIDs.
2. Use the existing qualified checkout whose four core hashes match the table,
   with all project dependencies installed. Upload the three new files; they
   are not committed and **a prior Git clone or running notebook will not acquire
   them automatically**. Use the local checkout's source via `PYTHONPATH` to
   avoid a stale installed package. The SHA guard fails instead of guessing.
3. Set `DATA` to the authenticated restored original data directory and `CACHE`
   to the existing pinned release/source cache **on that host**. The whole
   manifest and first eight NPZs are sufficient input payloads. Do not use the
   macOS path as a remote path. No H5, fitted scorer or editor artifacts are
   needed. DINO needs the already populated DINOv2 source and checkpoint cache;
   implicit DINOv2 downloads are refused. The existing model loader verifies
   release/source hashes; `--trust-checkpoint` consents only to its known DINO
   pickle, not arbitrary local files.
4. Execute models **sequentially**, not concurrently with each other or other
   experiments. In Molab, launch from the existing native Marimo workflow using
   checked subprocess calls; do not assume an updated source tree or environment
   is already present. Do not change runtime-global precision in a live shared
   notebook and then call that state a historical default.

From the prepared GPU checkout, with `DATA` and `CACHE` set:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
for MODEL in lewm dinowm; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$PWD/src" \
  python -u scripts/audit_execution_layout.py \
    --model "$MODEL" --device cuda --trust-checkpoint \
    --config configs/pilot.json --spec configs/followup_20261101.json \
    --data "${DATA:?set DATA to the restored original data directory}" \
    --cache "${CACHE:?set CACHE to the verified model/source cache}" \
    --output "runs/execution-layout-${STAMP}/${MODEL}" || exit 1
done
```

The two thread-count settings above are explicit process inputs and are recorded;
they do not set TF32/cuDNN defaults. Choose `cuda:N` explicitly if necessary.
Every rerun needs a fresh output directory. Review every control's applied or
skipped status rather than presuming all flags were available or all TF32
permissions changed the chosen kernels.

Back up the **entire fresh audit output directories** with an explicit inventory
and checksums, plus the uploaded source/test/protocol files and launch logs.
The old generic diagnostic backup discovery does not know this new stage name;
a successful generic backup is not proof that this audit was included. This
change deliberately does not modify the backup system or create a notebook.

## CPU-only validation

The dedicated suite blocks real model loaders, downloads and CUDA initialization.
It tests deterministic synthetic inputs for both adapter families; exact encoder
call sizes and ordering; raw action/prefix preservation; singleton versus actual
strided-bank histories; common-horizon matching; independent repeats; encoding/
rollout/recurrence/metric isolation; tensor-derived metrics; max/mean aggregation;
no filtering; authenticated manifest-order selection; parameter/nonpersistent-
buffer/input mutation; backend no-op/unavailable reporting and exception-safe
restoration; pre-inference selection, final reauthentication, exclusive output,
and CUDA/trust CLI gates. It is not evidence that real GPU arithmetic agrees.

Focused local command, **not** the workspace's real-artifact/full-verification
task:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m pytest tests/test_execution_layout.py -q -p no:cacheprovider \
  --junitxml=/tmp/wm-audit-execution-layout-tests.xml
```

**Validation completed 2026-11-02: 58 passed, 0 failures, 0 errors, 0 skips**, in
55.36 seconds using the configured local Python 3.12.8 environment. A fresh
JUnit completion report and its timestamp were checked; an earlier incomplete
terminal snapshot was not treated as a pass. These are CPU-only synthetic
tests, including real PyTorch backend-flag restoration, not real checkpoint
inference. Static editor diagnostics and whitespace checks were also clean.

## Interpretation limits

These are reconstructed **native tensor paths**, not a replay of the entire
historical process. Other appearances, readout work, F3/F4 postprocessing,
other-horizon metric scoring, previous roots outside this prefix, allocator
state and concurrent GPU workloads are omitted. Capturing/scoring tensors adds
synchronization. Algorithm caches
are not reset, and the same-process fixed-order repeat is not a fresh-process or
cross-driver reproducibility study. Supported flags are permissions, not a
kernel profiler; the cuDNN precision intervention is convolution-specific and
does not disable every attention backend. Runtime kernels, GPU/driver/library
versions, process environment and past cache state can still matter.

The earlier reports did not save full tensors or backend-state snapshots, so
their historical maxima are not automatically attributable to a particular
factor. The first eight roots may not contain the historical worst-case root;
do not expand the subset after seeing that. A nonzero isolated contrast
demonstrates sensitivity under this run's controlled conditions, not proof of
the old cause. A zero contrast here does not rule out backend/layout effects
elsewhere. Do not retrofit outcome-selected roots, claim bitwise replication,
or use small aggregate drift to conceal larger per-record differences.