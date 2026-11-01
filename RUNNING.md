# Portable VM execution

For post-pilot actual-image readout and frozen-prediction checks, see
[DIAGNOSTICS.md](DIAGNOSTICS.md). These reuse existing data without refitting
editors or retraining world models and produce live progress plus offline plots.

## Live progress (enabled by default)

New processes show stage start/completion/failure, requested model device,
elapsed time, and progress bars with completed counts and estimated remaining
time. Generation and archive validation count roots; fitting shows encoding,
readout extraction, CPU algebraic initialization, optimizer steps and loss
snapshots; inference counts candidates; evaluation counts roots and names the
current appearance. Recovery distinguishes its printed plan from executing stages.

Interactive terminals use updating bars. Notebook/subprocess output and redirected
logs use readable newline snapshots (at most every five seconds per counter,
plus start/end). CLI commands emit a heartbeat every 30 seconds during long
operations, including loading, downloads and analysis. **A heartbeat means the
reporting thread is alive, not proof that computation is advancing.** ETA is
based on completed units and remains unavailable before the first completion.

Progress goes to **stderr**, while CLI result JSON stays on stdout (upstream
models may still print initialization messages to stdout). Capture both streams
when saving a combined log. Set `WM_AUDIT_PROGRESS=0` to disable project progress.
This changes logging only, not seeds, numerical settings, checkpoints or metrics.
It does not add checkpoint resume or keep the remote container alive.

Deploy the updated code/environment to the GPU machine before the **next** run.
Already-running Python processes do not acquire these changes; do not interrupt
an active experiment just to obtain progress bars.

## 1. Copy and install

Target: Linux x86-64, NVIDIA GPU/working driver (`nvidia-smi`), internet, Git,
and Python 3 with venv/ensurepip (or uv 0.9.28). The bootstrap installs a managed
Python 3.11 and project-local environment; it does not install an NVIDIA driver
or require sudo. Your driver must support CUDA 12.8. The 96 GB Blackwell card
is the intended target, **not locally verified hardware**.

From your Mac, replace the SSH destination with your actual VM:

```bash
rsync -av --exclude='.venv' --exclude='.bootstrap-venv' --exclude='.bootstrap-cache' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='cache' \
  --exclude='runs' --exclude='data' \
  /Users/shashank/Deep_Learning/Research_wcm/ USER@HOST:~/Research_wcm/
```

Then on the VM:

```bash
cd ~/Research_wcm
bash scripts/bootstrap.sh --download
bash scripts/run_experiment.sh --output runs/smoke-01 --trust-checkpoint
```

No activation is required. Run long experiments in your usual persistent SSH
session manager. Use a new output directory each time: outputs deliberately
refuse overwrites. An interrupted fit is **not resumable**. Preserve failed runs
for diagnosis rather than deleting individual artifacts to force continuation.

For CPU-only testing, use `bash scripts/bootstrap.sh --cpu --download`, then
add `--device cpu` to the runner. Do not copy virtual environments across hosts.
The bootstrap preserves and refuses incompatible existing environments; move
an incompatible environment aside yourself. Local development used Python 3.12;
the portable bootstrap deliberately creates Python 3.11.

Artifacts default to the project cache directory. Both scripts accept
`--cache /absolute/path`; set the same value for both. The CLI also accepts
`--cache` **before** its subcommand. Generated audit data does **not** need the
large original training dataset. Allow several GB for installation/model smoke
work, and substantially more for full runs. Measure disk use with the smoke run;
compressed RGB archives vary with content, and no full-run runtime/storage
benchmark is claimed.

## 2. Run the research pilot

```bash
bash scripts/run_experiment.sh --config configs/pilot.json \
  --output runs/pilot-01 --trust-checkpoint
```

Use `--model lewm` to run only the primary model without pickle consent, or
`--model dinowm --trust-checkpoint` for the RGB secondary. The secondary is
**Point-LeWM authors' RGB DINO-WM release**, not their point-cloud model and
not the original DINO-WM authors' checkpoint. The DINOv2 source revision is an
explicit reproducibility choice because the release did not record one.

The default smoke profile has 12 editor roots, 12 independent scorer roots,
4 validation and 4 pilot roots, 3 candidates, 2 future steps, and 4 fitting
steps. Its four confirmation roots are never generated automatically.
The exploratory pilot has 256/128/64/128 fitting/scoring/validation/pilot roots,
8 candidates and 8 future steps, scoring steps 1/4/8. All four editors are fit;
the primary is selected using **clean validation only**. All models share the
same generated physical roots and candidate banks within a runner invocation.

Outputs under your chosen run directory:

| Output | Meaning |
| --- | --- |
| console log | Every stage, errors and printed diagnostics |
| preflight report | Runtime/device, native rollout shapes and simulator checks |
| replay report | Fresh deterministic replay/render invariance check |
| data directory | Pickle-free paired NPZ roots and split manifests |
| model inference report | Native candidate ranking and executed bank costs |
| model training report | Model/config provenance, fitted editor/scorer metadata |
| model validation/pilot records | Full root × appearance × method evaluation roster |
| model pilot report | Paired confidence intervals and qualification diagnostics |

Each model directory retains fitted scorer/editor checkpoints and CPU memmapped
training histories. Model weights stay frozen. Inference uses float32, not
unverified mixed precision. Models run sequentially, not simultaneously.

## 3. Individual commands

Run these **instead of** the combined runner when you want individual stages.
All global options precede the subcommand:

```bash
.venv/bin/python -m wm_audit.cli --help
.venv/bin/python -m wm_audit.cli download
.venv/bin/python -m wm_audit.cli --trust-checkpoint preflight --output runs/check-01.json
.venv/bin/python -m wm_audit.cli generate --config configs/smoke.json --data data/smoke-01

ROOT=$(find data/smoke-01/pilot -name '*.npz' | sort | head -n 1)
.venv/bin/python -m wm_audit.cli infer --model lewm --root "$ROOT" --output runs/native-01.json
.venv/bin/python -m wm_audit.cli --trust-checkpoint infer --model dinowm --root "$ROOT" --output runs/dino-native-01.json

.venv/bin/python -m wm_audit.cli fit --model lewm --config configs/smoke.json --data data/smoke-01 --run runs/lewm-01
.venv/bin/python -m wm_audit.cli evaluate --model lewm --config configs/smoke.json --data data/smoke-01 --run runs/lewm-01 --split validation
.venv/bin/python -m wm_audit.cli evaluate --model lewm --config configs/smoke.json --data data/smoke-01 --run runs/lewm-01 --split pilot
.venv/bin/python -m wm_audit.cli analyze --config configs/smoke.json --data data/smoke-01 --run runs/lewm-01 --split pilot
```

Native inference ranks candidates from a stored paired root and reports real
simulator endpoint costs. It is not a closed-loop robot controller or a video
decoder. To use DINO for fitting/evaluation add the global trust flag, replace
the model name and use a separate run directory.

## 4. Original dataset: optional

```bash
bash scripts/bootstrap.sh --download --with-dataset
.venv/bin/python -m wm_audit.cli dataset-info cache/datasets/lewm-pusht/pusht_expert_train.h5
```

This explicitly opts into a **13.14 GB download**, about **46 GB extracted**;
reserve at least **65 GB free**, in addition to experiment storage. The archive
is retained. Download/extraction verifies the pinned archive hash and validates
the original scene-camera HDF5 schema. It does not substitute newer sensor-view
images. [src/wm_audit/offline_data.py](src/wm_audit/offline_data.py) exposes
`H5Windows` for lazy episode-contained context/future loading and `episode()`
for whole episodes; controls contain five ordered XY pairs per model step.
The generated intervention audit uses fresh paired simulations instead: logged
pose vectors are not complete states for restoring moving counterfactuals.

## 5. Read the results honestly

The report's `primary_method` identifies the clean-validation-selected editor.
Read `contrasts.effect_loss.<primary>.transport.shifted` for the interaction,
and its `deterioration.shifted` for actual editor degradation. Inspect `native`,
`readout`, `clean_primary` and `decision` alongside those contrasts. Actual
candidate regret and the true-endpoint oracle diagnose decision consequences.
The report retains null effects, tied/all-failed banks and native failures.
Nonfinite data stops the run, rather than silently dropping roots.

**A positive interaction alone does not validate the hypothesis.** The clean
editor, native baseline, independent measurement and actual regret must support
the interpretation. Poor native/readout performance is an unsupported result,
not evidence of an editor-specific causal failure. Smoke sample sizes cannot
support a scientific claim. Margins in the configs are draft conveniences.

Implemented scope is deliberately narrower than the research proposal:

- Original, color-only and background-only conditions; no held-out combined
  appearance condition yet.
- Inverse, paired-ridge, residual-MLP and rollout-supervised editors, plus
  identity, matched-random, reversed-request and zero-request controls.
- One independent ridge physical scorer; it includes native-rollout calibration
  which can absorb systematic forecast error. Real-image error is reported
  separately. A second MLP scorer/shuffled-label study is not implemented.
- Rollout supervision matches **native counterfactual latent futures**, not a
  guarantee of physically correct forecasts.
- Fixed candidate-bank evaluation, not global policy regret or online MPC.
  Default confirmation uses 8 candidates, not a separate 32-candidate study.
- One fitting seed per invocation; intervals are conditional on that fit.
  No augmentation×future-loss repair study, cube task, or new WM training.
- Pointwise paired bootstrap intervals, not multiplicity-corrected confirmatory
  inference. Reports explicitly mark `inference.confirmatory` as false.

## 6. Confirmation is manual and review-gated

Review [PREREGISTRATION.md](PREREGISTRATION.md) and
[configs/confirmation.json](configs/confirmation.json), freeze scientifically
justified scales/margins/seeds, then run development with that config in a new
directory. Do not change configs after fitting. For example, after running
development into `runs/confirm-dev` with the confirmation config:

```bash
.venv/bin/python -m wm_audit.cli lock --config configs/confirmation.json --data runs/confirm-dev/data --run runs/confirm-dev/lewm --acknowledge-draft
.venv/bin/python -m wm_audit.cli lock --config configs/confirmation.json --data runs/confirm-dev/data --run runs/confirm-dev/dinowm --acknowledge-draft
.venv/bin/python -m wm_audit.cli generate --config configs/confirmation.json --data runs/confirm-dev/data --run runs/confirm-dev/lewm --split confirmation
.venv/bin/python -m wm_audit.cli evaluate --model lewm --config configs/confirmation.json --data runs/confirm-dev/data --run runs/confirm-dev/lewm --split confirmation
.venv/bin/python -m wm_audit.cli analyze --config configs/confirmation.json --data runs/confirm-dev/data --run runs/confirm-dev/lewm --split confirmation
.venv/bin/python -m wm_audit.cli --trust-checkpoint evaluate --model dinowm --config configs/confirmation.json --data runs/confirm-dev/data --run runs/confirm-dev/dinowm --split confirmation
.venv/bin/python -m wm_audit.cli analyze --config configs/confirmation.json --data runs/confirm-dev/data --run runs/confirm-dev/dinowm --split confirmation
```

Lock **both models before generating** the shared confirmation roster. Locks
hash package code, configuration, fitted artifacts, validation and development
data. They are local commitments, **not external preregistration**, and do not
turn draft diagnostics into a fully confirmed scientific result.

## 7. Verification and troubleshooting

Local verification: **808 tests passed**, including real released-checkpoint
upstream rollout comparisons and native simulator checks. Both LeWM and DINO-WM
completed CPU smoke inference, fitting, validation, pilot evaluation and report
generation. The smoke reports are software evidence only, not research results.

```bash
.venv/bin/python -m pytest -q
WM_AUDIT_MODEL_CACHE="$PWD/cache" WM_AUDIT_POINT_ROOT="$PWD/cache/sources/point/stable-worldmodel" \
  WM_AUDIT_TRUST_DINOWM=1 .venv/bin/python -m pytest -q
```

The first form may skip real-artifact tests if caches are absent. The second
requires real cached model artifacts and authorizes the pinned DINO pickle;
model tests compare adapters with upstream rollouts. Synthetic tests are
explicitly labeled and are not evidence of physical/model competence.

CUDA unavailable: check the VM driver and use bootstrap's cu128 wheels, not
an old upstream CUDA environment. Hash mismatch: preserve the cache and inspect
the discrepancy; do not disable verification. Missing artifacts: run download
with the same cache used by the runner. Existing output: choose a new output
directory. Slow CPU DINO runs are expected; use the GPU pilot only after preflight.

The full training-data download, fresh Linux installation and Blackwell kernels
must be checked on the target VM. Local CPU tests cannot certify them.