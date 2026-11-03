# Restored Molab session — 2026-11-03

Recovery is complete; no completed study was rerun and no new research result is claimed.

- Host: https://sb-6a039c950594bc39.sb.molab.run/
- Project: `/marimo/world_models_restored_20261103`
- Base source: `e6f2423fc3469d216e27bb07d0bd73c92586e1c0` (detached, preserving the original notebook checkout).
- Python: `/marimo/world_models_restored_20261103/.venv/bin/python`
- Cache: project-relative `cache`; pinned models and upstream sources downloaded and authenticated.
- Original input data: project-relative `runs/recovery-20261103/pilot/runs/pilot-gpu-01/data`.
- Restored root counts: editor 256, scorer 128, validation 64, pilot 128.

## Verified recovery

Only the two missing original backup chunks were transferred. All 14 chunks were reassembled and the original full-archive SHA256 was verified before strict inventory extraction.

| Restored archive | Inventoried files |
|---|---:|
| Original pilot | 983 |
| Original prerequisites | 120 |
| Follow-up probes | 48 |
| Follow-up dynamics | 41 |
| Recorded feedback | 43 |
| Unfinished source, bibliography, figures, old test receipts | 14 |
| **Total** | **1,249** |

All **12 diagnostic reports** were restored and their hashes rechecked. The published paper draft and results are present in the clean source checkout. Uncommitted replication/audit files were preserved both in a separate authenticated snapshot and at their working project paths. They remain **unfinished work**, not validated replication results. Unrelated local task settings and logs were not modified or transferred.

## GPU qualification

Both LeWM and DINO-WM passed the existing native-rollout CUDA preflight, including simulator qualification, a real CUDA matrix multiplication, finite rollout and goal-cost checks. This is a software check, not a repeat of the study. The host is an NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB). GPU was idle after qualification.

The original prerequisite notebook remains gated by `EXECUTE=False`; its old deployment settings are not the restored-work entry point. Use the restored project and the read-only [monitor](notebooks/monitor_restored_20261103.ipynb), also uploaded to `/marimo/monitor_restored_20261103.ipynb` and copied into the restored project's notebooks directory. Monitor JSON and Python syntax were checked locally and remotely; it was not run as a notebook.

## Durable receipt and remaining limitation

- [Downloaded final recovery receipt](runs/RECOVERY_READY_20261103.json), including archive hashes, report hashes, root counts, runtime and model preflight metadata.
- Receipt SHA256: `5a80e2379ea0ef6df8533f906cba2f9b7b466a0b7db2180423a51d54be67d0d0` (matches hosted output).
- Hosted receipt: project-relative `runs/recovery-20261103/RECOVERY_READY.json`.
- Bootstrap, restore and preflight logs remain in the restored project.

The optional **46 GB recorded training HDF5 was not redownloaded**. Its completed reports and provenance are restored. Generated-root replication can use the restored environment without it; rerunning recorded-data diagnostics requires downloading and authenticating that dataset first. No fresh-root replication has been run by this recovery operation. Local backups remain the durable source of truth; the sandbox is ephemeral.