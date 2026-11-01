# Google Drive backups from Marimo

Use [scripts/backup_world_models.py](scripts/backup_world_models.py) in the
**notebook kernel**, with the same already authenticated `fs` object used by
the Drive panel. No GPU environment or additional Python packages are needed.
Pass the **project root**, not the runs directory. The dedicated Drive folder
is `world-model-backups`.

Pull the latest project on the VM before starting the helper:

```bash
cd /marimo/world_models
git pull --ff-only
```

## Notebook cells

The following JSON describes three new cells. Paste each Python `source` into
its own Marimo cell (do not paste the JSON wrapper into a Python cell).
Cell 1 starts one worker and immediately attempts its first backup. It also
stores the worker in a process-local registry to prevent duplicate workers if
the cell is rerun. After stopping it, restart the kernel to start a new worker.
Do not restart the kernel while an upload is in progress.

```json
{
  "cells": [
    {
      "cell_type": "code",
      "metadata": {"language": "python"},
      "source": [
        "import runpy\n",
        "import sys\n",
        "import types\n",
        "\n",
        "_registry = sys.modules.setdefault('_world_model_backup_session', types.ModuleType('_world_model_backup_session'))\n",
        "if not hasattr(_registry, 'worker'):\n",
        "    _helper = runpy.run_path('/marimo/world_models/scripts/backup_world_models.py')\n",
        "    fs.makedirs('world-model-backups', exist_ok=True)\n",
        "    _registry.worker = _helper['DriveBackup'](fs, '/marimo/world_models', 'world-model-backups', interval=600).start()\n",
        "world_model_backup = _registry.worker\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {"language": "python"},
      "source": ["dict(world_model_backup.status)\n"]
    },
    {
      "cell_type": "code",
      "metadata": {"language": "python"},
      "source": [
        "# Leave commented while experiments run. Uncomment only to stop backups.\n",
        "# world_model_backup.stop()\n"
      ]
    }
  ]
}
```

Rerun cell 2 to refresh status; a background thread's changes do not trigger
Marimo's reactive cells automatically. `verified` means the archive was uploaded,
read back in full, SHA256 checked, and a checksum receipt written. A ZIP without
its matching `.sha256` receipt is **not** an acknowledged successful backup.
`error` reports an exception type without leaking HTTP/OAuth details. The worker
retries after the interval. `capturing`, `uploading`, and `verifying` are normal
active states. `stop()` takes effect after any current backup finishes.

## What is protected

- Completed prerequisite diagnostics under each model's `readout`,
  `prediction-paired`, and `prediction-recorded` directories: the final report,
  hash-listed plots, selection/window metadata, readout checkpoint and scatter
  arrays. A diagnostic without its final report is incomplete and is excluded.
  This requires the updated helper in a new notebook backup session; an existing
  worker does not reload changed Python source automatically.
- All run directories under the project's runs directory.
- Completed, atomically published root NPZ archives, **even when generation was
  interrupted before the final split manifest**. ZIP CRC is checked; this is not
  a replacement for the experiment's full physical/schema validation.
- Available valid generation manifests, with every referenced root SHA256 checked.
- Completed fit checkpoints (only after the training completion report exists),
  evaluation/inference/analysis reports and protocol locks.
- Configurations, package source, scripts and execution documentation.
- An internal `BACKUP.json` with counts, per-file hashes/sizes and skipped paths.

Excluded: unfinished/hidden root files, train-cache memmaps, incomplete fitted
checkpoints, pretrained model downloads, virtual environments, credentials,
arbitrary files, and **live console logs**. Logs are deliberately excluded because
they may change during capture; structured reports are included instead. Linked
artifact directories/files are refused. No checkpoint is deserialized.

Snapshots are full and independent, **not incremental synchronization**. Already
compressed NPZs are stored without recompression. Each backup temporarily needs
approximately the included data size on local disk, uploads that size, and reads
it back for verification. Full snapshots accumulate on Drive; no automatic
deletion occurs. For large pilots, consider `interval=3600` rather than 600.
The interval is measured after the preceding backup finishes; uploads never
overlap within one worker. Start only one notebook backup session per project.

## Recovery boundaries

Download a ZIP and its receipt, verify its SHA256 before extracting into a **new,
empty project/recovery directory**, and preserve internal paths. Do not overwrite
an active run. Recreate the environment and redownload pretrained model caches.
Archive integrity does not prove that the experiment was scientifically valid.

This helper preserves existing artifacts; it **does not implement automatic
resume** in the experiment runner. In particular, completed generation roots may
be recovered, but the current generator rejects an already-existing planned
root. Interrupted fits need a new run directory. Do not delete partial data to
force a rerun; retain it for a separately validated recovery workflow.

A stopped notebook/container cannot keep uploading. Unsynced work can be lost.
Live Google Drive authentication and upload must be verified in your notebook;
repository tests use a local fake filesystem, not your Drive account.