# Experiment tracker

This is the index of completed and pending work. Implementation, running,
verified completion, and a supported hypothesis are distinct states. A failed
or inconclusive result is retained. Hosted notebook state is not the record.

## Preserved discovery studies

Both frozen checkpoints were evaluated. These are exploratory results, not
independent replication. All twelve reports have been restored and authenticated.

| ID | Study | State | Evidence |
|---|---|---|---|
| D01 | Physical-state readout, both models | Completed | [Original diagnostics](MOLAB_RESULTS_20261101.md) |
| D02 | Generated native prediction, both models | Completed | [Original diagnostics](MOLAB_RESULTS_20261101.md) |
| D03 | Recorded native prediction, both models | Completed | [Original diagnostics](MOLAB_RESULTS_20261101.md) |
| D04 | F1 probe capacity, all three MLP seeds | Completed | [Follow-up results](FOLLOWUP_RESULTS_20261102.md) |
| D05 | F2 actual-history and auxiliary controls | Completed | [Follow-up results](FOLLOWUP_RESULTS_20261102.md) |
| D06 | F3 action-effect and assignment controls | Completed | [Follow-up results](FOLLOWUP_RESULTS_20261102.md) |
| D07 | F4 finite-bank native goal ranking | Completed | [Follow-up results](FOLLOWUP_RESULTS_20261102.md) |
| D08 | Recorded actual-history feedback | Completed | [Follow-up results](FOLLOWUP_RESULTS_20261102.md) |

F2/F3/F4 share a dynamics report per model; eight questions are not sixteen
independent reports. Earlier unsuccessful editor/appearance-transfer pilot
results remain in [the pilot analysis](PILOT_BACKUP_ANALYSIS.md).

## All pending fixed-protocol studies

| ID | Study | Budget and controls | Source |
|---|---|---|---|
| N01 | LeWM execution-layout numerical audit | **Completed and independently verified**; first 8 serialized original pilot roots, all applicable controls | [Numerical protocol](NUMERICAL_AUDIT_PROTOCOL_20261102.md) |
| N02 | DINO execution-layout numerical audit | **Completed and independently verified**; identical fixed roster, isolated GPU execution | [Numerical protocol](NUMERICAL_AUDIT_PROTOCOL_20261102.md) |
| R01 | Fresh-root joint F2/F3/F4 replication | **Completed and independently verified**; 128 roots, both models, five primary contrasts; four supported, ranking inconclusive | [Replication protocol](REPLICATION_PROTOCOL_20261102.md) |

No fresh F1, recorded-data repeat, editor refit, new world-model training,
improved candidate bank, or closed-loop experiment is included. Those would be
new scientific designs, not unimplemented parts of these protocols. The optional
46 GB recorded HDF5 is not needed for this pending batch.

## Durable execution contract

[The tracked launcher](scripts/run_tracked_studies.py) runs N01, N02, then R01
serially so backend comparisons are not confounded by concurrent GPU workers.
It refuses occupied GPUs, changed published source, existing destinations and
implicit resume. R01 keeps the qualified runner's model order unchanged.

Every batch contains:

- `launch.json`: exact commands, published commit, source hashes, input path,
  scheduling, and explicit no-discovery-rerun declaration.
- `sources/`: complete source/config/test/protocol snapshots before execution.
- `status.json`: atomic latest state, active stage, PID, progress line, timestamps,
  exit code, verification and archive receipt.
- `events.jsonl`: append-only timestamped start/progress/exit/failure/verification
  events. Per-stage logs retain every output byte and the runner's ETA/progress.
- Per-stage directories: pre-inference selection/plan, results, raw records,
  bootstrap draws, exact generated-root identities and data where applicable.
- Independent per-stage ZIP plus SHA256 receipt: explicitly inventories these
  new kinds, including failed attempts; the legacy backup helper is not used.
- `complete.json`: written only after all three stages pass integrity checks.
- A separately sealed `operations.zip` captures final success/failure status and
  all logs after the immutable stage archives are published. Failed ZIP assembly
  is kept separately and is never advertised as a verified backup.

Completion does not require a favored scientific outcome. Replication decisions
come from the five prespecified 99% intervals; touching/crossing zero is reported
as inconclusive. A numerical/parity/integrity failure stops the batch and retains
the partial attempt. It never authorizes replacing a root, changing the seed,
omitting a control, or claiming success from partial output.

All three stages ran under published commit
`36d6c1f4b744d808d1955ed50a3f7bd72280ba31`, run ID
`paper-studies-20261103-01`, 2026-11-03 07:17:46–07:41:06 UTC. All four ZIPs and
SHA receipts were downloaded and strictly restored; 610 inventoried entries
verified. [Results, all primary outcomes and archive ledger](PAPER_STUDIES_RESULTS_20261103.md).
[Final status](runs/paper-studies-20261103-retrieved/status.json) and
[completion receipt](runs/paper-studies-20261103-retrieved/complete.json).
No prescribed pending study remains unrun. Publication-readiness and broader
scientific questions are separate tasks; completion is not paper acceptance.

## Current recovery entry point

[Restored session receipt and paths](RESTORED_SESSION_20261103.md).
The original prerequisite notebook remains gated and must not be run-all to
launch this batch. A published runner and filesystem status are authoritative.