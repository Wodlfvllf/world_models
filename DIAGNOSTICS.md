# Prerequisite diagnostics after the first pilot

These commands implement **steps 1–2**, not a new confirmation experiment:
actual-image-only physical readouts and frozen prediction versus controls.
They never train the world-model weights, refit latent editors, generate new
simulation roots, or overwrite the completed pilot. Existing pilot data is
already inspected: all new results on it remain **exploratory**.

## Install/update

Transfer the updated source to the GPU project before running. In an existing
working project environment, update the editable package (adds pinned tqdm and
matplotlib; retain the project's Torch/CUDA versions):

```bash
cd /marimo/world_models
.venv/bin/python -m pip install -e .
```

For a fresh machine use the existing bootstrap procedure instead. No original
recorded dataset is downloaded implicitly. Use the same cache as the pilot.

## Run both models on the existing paired data

```bash
.venv/bin/python -u scripts/run_diagnostics.py \
  --data runs/pilot-gpu-01/data --config configs/pilot.json \
  --output runs/diagnostics-01 --device cuda --trust-checkpoint
```

This runs readout and paired prediction diagnostics for LeWM, then DINO. Use
`--models lewm` to omit DINO and pickle consent, or `--models dinowm` to run only
DINO. Use `--device cpu` for local verification. Every output directory must be
new. There is **no automatic resume**: to add a stage later, specify `--stages`
and a fresh output directory rather than deleting files from an interrupted run.

The runner streams stage messages and combined logs to the terminal and its
console log. Commands provide progress, device labels and heartbeats. GPU is used
for frozen encoding/rollouts. Ridge eigensolves, plotting, archive validation
and hashing use CPU/disk; an idle GPU during those named stages is expected.

## Optional original recorded-data prediction diagnostic

If the original HDF5 dataset has not been downloaded, first run:

```bash
.venv/bin/python -m wm_audit.cli --cache cache download --dataset
```

This opts into approximately 13.14 GB compressed / 46 GB extracted data; allow
at least 65 GB free for archive plus extraction. Then run only the new stage:

```bash
.venv/bin/python -u scripts/run_diagnostics.py \
  --data runs/pilot-gpu-01/data --config configs/pilot.json \
  --output runs/diagnostics-recorded-01 --device cuda --trust-checkpoint \
  --stages prediction-recorded \
  --h5 cache/datasets/lewm-pusht/pusht_expert_train.h5 --episodes 128
```

Supplying `--h5` without `--stages` runs all three stages. Without `--h5`, the
runner explicitly reports recorded diagnostics as **not run**, not successful.

One window is selected per eligible episode using only episode metadata and
the fixed config seed. Windows never cross episode boundaries or use terminal
NaN actions. Both models receive the same episode/start selection. No learned
readout or interpretation of unspecified HDF5 state coordinates is used.

Recorded data is evaluated at **original appearance only**: the recording has no
validated semantic masks for faithful object/background rerendering. Generated
paired data supplies the exact color/background comparisons. The released H5 is
pretraining data, so good recorded performance is **not held-out world-model
generalization**. It is a distribution/compatibility diagnostic. The report hashes
the H5, but schema acceptance alone is not proof of camera provenance; use the
hash-pinned downloader above rather than an arbitrary similarly shaped file.

## Readout protocol

- **Fit:** scorer roots, original-appearance actual images only.
- **Select:** validation roots, original appearance only, lowest mean scaled
  physical loss. No pilot or shifted images enter fitting or selection.
- **Evaluate:** every pilot root under all three appearances, no refitting.
- Each split uses the same sampling: both branches, all three context frames,
  and configured future horizons for candidate `root_index % K`. This is fixed
  before outcomes, not a chosen successful action. These readout diagnostics do
  not measure all candidate futures; the separate prediction diagnostic does.
- Mean-state baseline is computed from fitting labels only.
- LeWM uses its CLS features. DINO compares spatial pooling grids 2x2 and 4x4;
  the latter retains finer location information, not the full 14x14 patch field.
- Ridge strengths: 0.01, 1, 100, 10000, under a **sum-squared** objective with an
  unpenalized intercept. Standardization is fitted on training features only.
  One CPU float64 primal/dual eigensolve per grid is reused across strengths.
- Positions are native simulator coordinates, not resized-image pixel distances.
  Report per-coordinate MAE/RMSE, pusher/block Euclidean errors, circular angle
  MAE, sin/cos errors, and the original scaled loss. Undefined predicted angular
  vectors receive 180 degrees error and are counted, never silently excluded.
- Confidence intervals resample root means, preserving correlated frames/branches.
  Context and future scores are reported separately as well as together.
  Paired selected-minus-mean and appearance-minus-original scaled-loss intervals
  are also reported. Existing root hashes and configuration must match the original
  development generation manifest; schema-valid replacements are not accepted.

Training data and original-pilot readout losses now have matching sampling rules;
do not equate these numbers with the old mixed actual/predicted readout scores.
Readout accuracy on actual images is necessary but does not by itself establish
validity on predicted or edited latent states.

Direct command (optional hyperparameter overrides, still development-only):

```bash
.venv/bin/python -u -m wm_audit.cli --device cuda --trust-checkpoint \
  diagnose-readout --model dinowm --config configs/pilot.json \
  --data runs/pilot-gpu-01/data --output runs/diagnostics-custom/dinowm/readout \
  --grids 2 4 --alphas 0.01 1 100 10000
```

## Prediction protocol

For every generated pilot root, branch, candidate and appearance, encode actual
future images at the configured horizons (pilot: 1/4/8) and compare:

1. Native frozen rollout with recorded raw actions.
2. Persistence: repeat the last actual context latent.
3. Frozen rollout with **future actions zeroed**, preserving the first two
   action chunks associated with the observed history.

Zero means native XY control zero, not normalized zero. The adapter applies its
released action normalization exactly once. Predictions use the same history and
are evaluated against the same actual future under the original action sequence.
This is an action-ablation diagnostic, not simulated zero-action ground truth.

Report coordinate-mean latent MSE and paired native-minus-control contrasts at
each horizon and appearance, with root/episode-cluster bootstrap intervals.
Negative contrasts favor native prediction. Raw latent MSE cannot rank LeWM
against DINO; compare against controls **within each model**. Near-static sequences
can make persistence strong; zero-action ties do not automatically imply a bug.

## Outputs and backups

Each run contains model subdirectories with `readout`, `prediction-paired`, and
optionally `prediction-recorded` directories. Open their HTML galleries locally
or download the PNGs. No web server or remote plotting service is needed.

- `index.html`: offline gallery; scatter plots, error distributions, readout
  selection curves, or horizon/paired-contrast prediction curves.
- `report.json`: completion marker, configuration, model/code/input hashes,
  per-root or per-window records, intervals and limitations. Written **last**.
- Readout: `selection.json`, tensor-only `readout.pt`, and `scatter-data.npz`
  with all plotted truth/prediction arrays, not selected examples.
- Recorded prediction: `windows.json` records exact episode/start selection.

The updated backup helper captures completed diagnostic reports and all their
hash-verified listed artifacts under the above directory layout. Partial
diagnostics without the final report are excluded. Existing notebook workers
retain the OLD imported helper; use a new helper session for future diagnostic
backups after the current upload finishes (do not run duplicate backup workers).
The backup root counters still count simulation roots, not completed diagnostics.
The helper does not follow arbitrary input paths: if diagnostics read a nested
restoration or data outside the immediate run/data layout, retain the original
data backup separately. A diagnostic-only snapshot is not a copy of that external
or nested dataset. Recorded HDF5 files are not included in these snapshots.

## Interpretation / next gates

1. If actual-image readout is inaccurate, investigate scaling, spatial features,
   regularization and label/feature coverage before interpreting physical edits.
2. If prediction fails on generated but works on recorded data, investigate
   distribution mismatch. If it fails on both, inspect alignment/normalization
   and native competence. These patterns suggest causes, not prove them.
3. Only after measurement and native prediction qualify should clean editing be
   revisited using the existing controls, with a predeclared useful-effect target.
4. A final appearance-transfer test needs fresh root families and a frozen protocol.
   None of these commands automatically approves confirmation or changes margins.

This implementation intentionally does **not** solve editor competence, redesign
the action bank, or launch a new confirmation run before the diagnostics exist.