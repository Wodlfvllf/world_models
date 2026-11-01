# Run on Molab, not locally

## Upload and start

Upload these three supplied files to your **GPU Molab session**:

1. `notebooks/run_prerequisites_marimo.py` — open/import this as the native Marimo notebook.
2. `dist/world-models-molab-source.zip` — updated project scripts, including unpushed changes.
3. `dist/world-models-molab-source.zip.sha256` — put beside the uploaded ZIP, with the same basename.

Use Molab's upload/file browser and copy the **remote paths** it assigns. The
default `/marimo/…` paths are examples; change them if your upload lands elsewhere.
You do not need a local Python environment and do not need to run a local server.
The notebook itself executes the shell scripts and Python commands remotely.

In **Cell 2**:

- `SOURCE_ZIP`: remote path of the source ZIP.
- `PROJECT`: a **new remote directory**, default `/marimo/world_models_diagnostics`.
  The previous project, results, and environment are not overwritten.
- `DATA`: actual remote path to the completed pilot's data, including its generation manifest.
  This may be nested inside a restored backup. No new pilot is generated.
- `EXISTING_PYTHON`: set to `/marimo/world_models/.venv/bin/python` if that
  previous working environment exists; otherwise leave `None` for a fresh CUDA
  environment installed **inside Molab**. The notebook kernel is not modified.
- `RUN_ID`: a fresh experiment name for each attempt.
- `TRUST_PINNED_DINO=True`: only after consenting to load the pinned released pickle.
- `EXECUTE=True`: authorize setup and experiments, then **Run all**.

Disable reactive auto-run while editing settings. Do not run the old and new
notebooks concurrently on the same GPU. Readout fitting is training of small
physical probes, **not training LeWM/DINO weights**.

## If Molab restarted and the pilot files disappeared

Upload the original full pilot backup ZIP and its matching `.zip.sha256`
receipt. Set `BACKUP_ZIP` in Cell 2 to that uploaded ZIP's absolute remote path.
The setup cell calls the existing checksum-verified restore script into a new
`PROJECT/runs/restored-pilot` directory and selects `runs/pilot-gpu-01/data`
inside it. The receipt must have the exact ZIP basename plus `.sha256`.
No code from the old backup replaces the new diagnostic source.

If your backup has a different run layout, restore it separately and set `DATA`
to its actual data path, leaving `BACKUP_ZIP=None`. A restored destination is
never overwritten: after restoration, set `BACKUP_ZIP=None` and use its data
path when restarting cells.

## What Run all does, in order

1. Verify and extract the uploaded source (reuses only an identical extraction).
2. Bootstrap/reuse the hosted environment; optionally restore pilot data.
3. Install the updated package; validate the original input hashes and configuration.
4. Download pinned model artifacts; check the GPU/model/simulator setup.
5. **Train/select/evaluate LeWM physical readout.**
6. Run LeWM prediction versus persistence and zero-future-action controls.
7. **Train/select/evaluate DINO physical readout.**
8. Run DINO prediction versus the same controls.
9. Optional recorded-data preparation and both models' recorded prediction checks.
10. Verify reports, show plots inline, and provide a results ZIP download button.

Every stage depends on the previous stage's completion token. A failure stops
downstream experiments. Live script output and 30-second waiting heartbeats
are shown in cells; detailed logs are also saved. Setup can use CPU/disk while
the GPU is idle. `RUN_RECORDED=False` explicitly skips recorded checks; enabling
the download requires a second opt-in and at least 65 GB of free disk.

Use a new `RUN_ID` after interruption; this notebook intentionally does not
guess how to resume partial scientific outputs. The download button supplies
**results only**, not model weights or original data. Retain the original backup.

## JSON cell version

`dist/molab_cells.json` provides the requested `cells` array, with
`metadata.language` and `metadata.id` for every cell. It contains the same
sequential setup/experiments, for copying cells into an existing hosted notebook.
For direct native Marimo import, use the `.py` notebook instead of JSON/Jupyter.

All of these remain exploratory prerequisites on previously inspected pilot
data. They do not approve editor qualification or launch fresh confirmation.
See [DIAGNOSTICS.md](DIAGNOSTICS.md) for the scientific protocol.