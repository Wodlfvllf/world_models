# Inspect and finish the recovered pilot

## Snapshot findings (2026-10-31)

The downloaded archive with SHA256
`3cb806e41cc4837ae1e2ed20399897dc738d384c2ea255787540acc95d63f3d4`
was verified against its receipt and every restored file's internal hash/size.
It contains:

| Run | Data roots | Completed model stages |
| --- | ---: | --- |
| pilot-01 | 304 of 576 | Incomplete generation; preserve, do not use as complete data |
| pilot-gpu-01 | 576 of 576 | LeWM inference, fit, validation, pilot evaluation |
| smoke-01 | 32 development roots | Earlier software smoke results |

LeWM analysis was subsequently executed locally from the saved GPU evaluation.
No LeWM retraining or data regeneration was necessary. DINO-WM fitting/evaluation
are absent from this snapshot. This does not establish what may have completed
on the remote VM **after** the snapshot.

## Actual LeWM pilot findings

128 held-out root families; clean-validation-selected editor: MLP.

- Transport interaction estimate: **0.140**, paired 95% interval **[-0.996, 1.275]**.
- Real-image readout scaled squared loss: **71.68** original, **74.23** averaged
  shifted appearance. The draft tolerance is 0.05; measurement is not qualified.
- Clean editor minus identity physical-effect loss: **+1.098**, 95% interval
  **[0.780, 1.490]**. Positive means the editor is worse on this endpoint.
- **62/128 counterfactual banks** and **65/128 factual banks** have cost exactly
  1 for every candidate (zero goal coverage). These banks are retained; zero
  regret on them is not successful control.

These are exploratory results, not evidence that the hypothesis is true or that
there is no effect. Poor readout generalization, clean editing and candidate-bank
utility need investigation before confirmation. Do not remove failed banks or
relax margins to force a positive result.

## Restore and visualize elsewhere

Use the project environment. Destination directories must be new. No archived
Python code is executed; restoration verifies hashes and refuses traversal.

```bash
.venv/bin/python scripts/inspect_world_backup.py restore /path/to/backup.zip runs/recovered
.venv/bin/python scripts/inspect_world_backup.py visualize runs/recovered/runs/pilot-gpu-01 runs/recovered/gallery
```

Keep the matching checksum receipt adjacent to the ZIP. A successful restoration
publishes a RESTORE-VERIFIED marker; do not use a partial restoration lacking it.
The gallery contains HTML, PNG time-series/candidate grids and animated GIFs.
It uses the first three pilot roots by filename, not selected successes. Candidate
1 is fixed for time-series displays; all eight candidates are shown in endpoint
grids. Candidate 0 is a no-op control. The images are **simulator ground truth**,
not model-decoded predictions. The green T marks the target; the solid block is
the physical object and the small disk is the pusher. Appearance variants retain
the same physical states and action sequence within each branch.

## Run only the missing stages on your GPU VM

If the original complete pilot data still exists on Marimo:

```bash
cd /marimo/world_models
git pull --ff-only
.venv/bin/python scripts/finish_recovered_pilot.py \
  --source runs/pilot-gpu-01 \
  --config configs/pilot.json \
  --output runs/pilot-finish-01 \
  --device cuda --execute --trust-checkpoint
```

This verifies the full data manifest and config, computes the missing LeWM report
if necessary, and runs **DINO inference → fit → validation → pilot → analysis**
in a new output directory. It never regenerates the data or refits LeWM. Keep
the source run: the new directory references its data rather than copying it.
The Drive helper captures model artifacts in the new run as they complete.

If using a restored archive instead, replace `--source` with the restored complete
pilot run and `--config` with its restored pilot config. Download model caches
again if necessary. Omit `--execute` to only validate and print the plan. If DINO
is already fitted in the source, this helper refuses to refit it; inspect which
evaluation/report is missing instead. Interrupted fits still require a new output
directory; this is stage continuation, **not training-checkpoint resume**.

The local Mac has no CUDA connection to the Marimo instance. DINO continuation
was prepared and its plan checked locally, but was **not executed on the remote
GPU** by this recovery session.