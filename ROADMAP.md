# Schedule, compute and execution decisions

**Prepared:** 2026-10-31. This is a work allocation, not a runtime benchmark or publication guarantee.

## 1. Two timelines, not one

The [official working-draft CFP](https://icwm.cc/call-for-papers) gives **2026-11-30** as the provisional submission date. From 2026-10-31 that is **30 elapsed days**. A full 70 elapsed days ends **2027-01-09**.

- **November:** attempt a complete, narrow submission only if the empirical and novelty gates pass.
- **December–January:** strengthen the study, prepare a preprint/revision, or target another suitable venue. Do not assume later results can replace a weak initial submission.
- **Acceptance/publication:** the listed decision date is February 8, 2027, beyond both research windows. Producing results or a manuscript within 70 days is not the same as securing publication within 70 days.

The inspected review policy is public and real-time on OpenReview. Submissions are anonymous during review, and author identities are revealed after **every** final decision, including rejection. Reviewers remain anonymous. Preserve author anonymity in submitted artifacts while planning for eventual public disclosure.

Before submission, verify the final template, page limit, required sections, time zone/cutoff, supplements, anonymity, originality/concurrent-submission, AI-use, and rebuttal-update rules. These were not fully specified on the inspected working-draft page.

## 2. First 72 hours

### A. Qualify the actual GPU host

The current VS Code workspace is on macOS; access to the NVIDIA machine is not established. Use a Linux CUDA host, potentially through the user's existing remote development setup. Do not install a CUDA experiment stack on the Mac and assume it will use the separate GPU.

Record:

- exact GPU model and whether its full 96 GB is available;
- driver, CUDA runtime, PyTorch build, Python, CPU/RAM, disk space and shared-use constraints;
- installed simulator/rendering libraries and headless rendering behavior;
- whether a small strict-load forward pass, backward pass for an editor, and paired replay pass.

Blackwell support is documented in [PyTorch's CUDA 12.8/2.7 release](https://pytorch.org/blog/pytorch-2-7/). Use a currently compatible build; do not blindly use an old DINO-WM CUDA 12.1-era environment or the Point-LeWM CUDA 12.6 lockfile. Pin successful versions after compatibility tests.

### B. Obtain only the first needed artifacts

1. [LeWM PushT weights/configuration](https://huggingface.co/quentinll/lewm-pusht/tree/main).
2. The matching [LeWM](https://github.com/lucas-maes/le-wm) and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) implementation revisions.
3. [RGB DINO-WM PushT checkpoint/configuration/action statistics](https://huggingface.co/fafraob/point-cloud-pusht/tree/main/dino-wm), with [Point-LeWM's documented adapter](https://github.com/fafraob/point-lewm).
4. The smallest required original-image dataset/sample for native replication. Generate fresh simulator roots for the new audit.

Inspect the loading path. Prefer weights-only/state-dictionary formats. Do not blindly enable arbitrary-object deserialization or execute scripts from a new checkpoint repository. Record licenses, upstream restrictions, source revisions and file hashes.

### C. Produce a small evidence bundle

- native baseline configuration and per-episode outcomes;
- exact replay/render-invariance checks;
- action/observation shape and normalization checks;
- timing at several batch sizes and horizons;
- 20–30 fresh-root smoke examples, including failed examples;
- explicit decision on which two checkpoint interfaces are qualified.

The smoke examples are development data, not final results. If the second model's adapter is not working promptly, continue the pipeline with LeWM instead of starting Dreamer training.

## 3. November submission schedule

| Dates | Deliverable | Gate |
| --- | --- | --- |
| Oct 31–Nov 3 | GPU/artifact contracts; replay; one native baseline | No silent architecture, action or simulator substitutions |
| Nov 4–7 | Readouts, clean editors, 128-root exploratory pilot | Decide whether an editor-specific phenomenon is plausible and novel |
| Nov 8–12 | Provisionally fix scope/splits/metrics; fit linear, nonlinear and temporal editor controls | No confirmation access; compute estimate fits remaining budget |
| Nov 13–17 | Qualify both checkpoint interfaces; complete development ablations; finalize and record the full protocol lock | No remaining adapter-dependent analysis choices before confirmation |
| Nov 18–23 | Locked fresh-root confirmation; executed candidate banks; endpoint oracle | Main claims pass, fail or remain uncertain without selective exclusions |
| Nov 24–27 | Robustness, statistics, full outcome accounting and writing | Audit each sentence against evidence; freeze main results |
| Nov 28–29 | Reproducibility/anonymity checks and submission package | Final venue requirements verified; retain buffer |
| Nov 30 | Submit only if ready | Exact cutoff is still provisional; do not assume AoE |

Start the methods and related-work notes while experiments run. Do not postpone all writing to the last two days, and do not write a results-driven conclusion before confirmation.

### November no-go conditions

- No reliable native rollout/replay by November 3–4: switch to the simplest qualified interface or re-scope the deadline.
- No credible clean editor/native competence by the pilot decision: stop the main hypothesis.
- Effects only appear after selecting favorable test roots: invalidate that analysis and re-register on fresh data.
- Main result is already covered by newly found prior art: reformulate before further compute.
- Too little evidence by November 23: do not claim a complete main-conference study merely to meet the date.

If the second model cannot be qualified before the final lock, a LeWM-only study must explicitly become a **one-checkpoint case study**, with a weaker publication case. Do not retain a two-model claim, and do not remove an already qualified checkpoint because its confirmation result contradicts the hypothesis.

## 4. Seventy-day extension

| Dates | Extension | What it adds |
| --- | --- | --- |
| Dec 1–7 | Qualify OGBench single-cube checkpoints and exact replay | A second physical task, not just another appearance variation |
| Dec 8–18 | Replicate the primary transport/regret experiment on cube | Cross-domain limits; includes counterexamples and nulls |
| Dec 19–28 | Matched 2 × 2 editor training: augmentation × future supervision | A controlled learning question, with frozen WMs |
| Dec 29–Jan 5 | Independently trained small LeWM seeds or one practical correction application, if budget permits | Training-seed robustness or real closed-loop relevance |
| Jan 6–9 | Consolidate paper, artifacts and limitations | Complete 70-day manuscript/results package |

Do not add all extensions just because weights exist. Prefer one well-qualified second domain and a useful mechanism/repair analysis over ten shallow benchmarks. Rebuttal use of additional results depends on the eventual conference policy.

## 5. Compute: the card is useful, but memory is not the main bottleneck

The proposed frozen models and small editors are reasonable candidates for a 96 GB GPU. Actual peak memory and throughput remain unmeasured. Use batch micro-splitting for candidate rollouts; avoid caching every activation from every layer.

### What the papers report, not measurements on this card

| Source | Reported context | How to use it |
| --- | --- | --- |
| [LeWM](https://arxiv.org/abs/2603.19312) | Approximately 15M parameters; a few hours of single-GPU training; L40S in implementation details | Supports choosing a small model, not an exact RTX runtime promise |
| [DINO-WM appendix](https://arxiv.org/html/2411.04983v2) | A6000 timing: 0.014 s single prediction step at batch 32; 53 s for the reported CEM setup | Planning can dominate; do not multiply this by an assumed Blackwell speed factor |
| [PLDM](https://arxiv.org/html/2502.14819v2) | Single-GPU experiments up to a day on V100 | Evidence small-model training is possible, not a PushT port budget |
| [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) | Reported 16 s/action on RTX 4090 under a particular horizon-one search setup | Large search costs exist even with frozen weights |
| [MMBench2 repository](https://github.com/nicklashansen/mmbench2) | Large training/preprocessing pipeline, with multi-H100 training recommendations | Use released frozen inference if selected; do not reproduce full pretraining |

### Proposed November spending caps

These are **management caps**, not predictions that the work will finish in these times.

| Activity | GPU-hour cap |
| --- | ---: |
| Loading, native replication, profiling | 20 |
| Data feature extraction and pair validation | 30 |
| Probe/editor development | 30 |
| Core frozen-rollout audit | 80 |
| Executed candidate-bank evaluations | 60 |
| Robustness/readout/seed checks | 50 |
| Recovery/rerun reserve | 80 |
| **Total initial cap** | **350** |

A continuously available GPU has 720 gross hours in 30 days and 1,680 in 70 days. Neither is usable experiment capacity by default: engineering, rendering, storage, failures, shared use and analysis consume time. Do not spend the reserve before the pilot passes.

Profile complete batches for each model, horizon and precision. Estimate total compute from actual planned batches, including factual/native reuse, fitting and rendering. Log wall time, GPU time when available, peak VRAM, CPU utilization, I/O and failures. Re-estimate before confirmation.

### Prevent combinatorial explosion

- Initial pilot: 128 roots, two intervention families, original appearance plus two nuisance families; small fixed candidate banks.
- Core prediction audit: target 400 locked roots, four render conditions, horizons 1/4/8 from a single maximum-horizon rollout, about eight action candidates.
- Decision analysis: a smaller prespecified root subset or the full roster as budget permits, with 32 candidates; subset selection must be fixed before outcomes.
- Reuse physical trajectories across rendering conditions when rendering is verified not to affect physics.
- Reuse factual/native predictions across matched editor comparisons.
- Run expensive full CEM/closed-loop evaluations only for prespecified leading comparisons, selected on development data—not for every editor × nuisance × horizon cell.

## 6. Storage and system requirements

Suggested planning allowances, to check against the actual host: 64 GB system RAM is a useful starting target, 128 GB may ease data handling; reserve roughly 100 GB scratch for an initial PushT audit and more if storing raw 3D episodes. These are engineering allowances, not verified minima.

Public listings inspected during the audit include:

- LeWM–PushT weights approximately 72.3 MB; its released compressed dataset approximately 13.1 GB.
- The selected RGB DINO-WM PushT package approximately 88.8 MB, excluding separately fetched DINOv2 and data.
- Original DINO-WM PushT archive approximately 2.79 GB in the inspected OSF dataset listing.
- LeWM single-cube dataset approximately 46.2 GB compressed.

Feature caching can exceed checkpoint/data sizes. At 196 × 384 FP16 elements, one DINO feature frame is about **150 kB**; 100,000 frames are about **15 GB**, before metadata. A 192D FP16 LeWM embedding is only 384 bytes. Cache only the exact interface/features needed; estimate volume before extraction.

## 7. Implementation and release checklist

The next coding phase should implement adapters and tests rather than a new world model:

1. Environment reset/replay, state capture and independent appearance rendering.
2. Native model encoding/rollout with strict history/action alignment.
3. Root-family data manifest and split checks.
4. Readout/editor interfaces with separate fitting data.
5. Paired rollout metrics and actual simulator candidate outcomes.
6. Clustered analysis and figure generation directly from raw per-root records.

For every run record: code commit, environment versions, checkpoint hashes, root/split manifest hash, exact config, seed roles, wall time/VRAM, outputs, exclusions/failures and their predefined reasons. Reconstruct paper tables from these records; never manually edit result values.

Publish only assets permitted by their licenses. Cite original model authors **and** re-release/checkpoint authors where applicable. Keep upstream checkpoints separate from generated audit data. Preserve unsuccessful runs and analysis decisions.

## 8. Expected final deliverable—not promised outcomes

A defensible paper package would contain a narrow, verified finding; methods sufficient to reproduce it; meaningful positive and negative controls; effects and uncertainty in physical units; actual action outcomes; a clear novelty comparison; and limits on what the tested checkpoints support.

The practical next step is **E00–E02 on the real Linux GPU host**. No full-scale training job is justified before those checks and the seven-day pilot decision.