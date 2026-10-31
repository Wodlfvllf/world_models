# World-model research: decision and execution plan

**Prepared:** 2026-10-31. **Status:** executable research pipeline plus proposed experiments; no established research result.

## Run on your Linux GPU VM

Copy this project **without** the local virtual environment. From the project directory:

```bash
bash scripts/bootstrap.sh --download
bash scripts/run_experiment.sh --output runs/smoke-01 --trust-checkpoint
```

The first command creates an isolated Python environment, installs CUDA 12.8
PyTorch and pinned dependencies, executes a GPU kernel, and downloads both
released models, the DINOv2 backbone and pinned simulator/model sources.
The second runs **real-model inference → paired simulation → editor/readout
fitting → validation → pilot evaluation → bootstrap analysis**, for both models.
DINO's upstream checkpoint contains Python pickle; the explicit trust flag
authorizes loading only the hash-pinned release. Use `--model lewm` without that
flag if you do not accept it. Downloading alone never unpickles it.

Next, run the larger exploratory pilot:

```bash
bash scripts/run_experiment.sh --config configs/pilot.json --output runs/pilot-01 --trust-checkpoint
```

**Start with smoke, not confirmation.** Read [RUNNING.md](RUNNING.md) for VM
requirements, copying, exact inference/data commands, output interpretation,
confirmation locking, limitations and troubleshooting. Smoke results are only
software checks. Linux/Blackwell execution must be verified on your VM.

## Recommendation

Use the available **single RTX PRO 6000 Blackwell, 96 GB VRAM** for controlled experiments on **frozen, released action-conditioned world models**, not foundation-model pretraining.

**Primary question:** Does an edit intended to change an object's physical state retain its meaning when only appearance changes, and does failure affect counterfactual action selection?

**Working title:** *Do Physical State Edits Survive Appearance Changes in Frozen World Models?*

- Start with **LeWorldModel (LeWM) on PushT**.
- Replicate on **DINO-WM**, preferably the explicitly identified RGB/no-proprioception checkpoint released by the Point-LeWM authors.
- Train only small probes and editors initially. Keep both world models frozen.
- Use **simulator-grounded physical interventions, complete observation histories, native counterfactual references, independent measurements, and executed action outcomes**.
- Commit initially to a **seven-day pilot**, not seventy days of blind execution.

This is the best balance found between interpretability relevance, artifact availability, and compute. **Novelty risk is high.** The broad ideas of causal probing, persistent editing, appearance disentanglement, and counterfactual fidelity already have close precedents. The narrower contribution must survive the comparisons in [PROPOSAL.md](PROPOSAL.md).

## Deadline correction

The [official ICWM call for papers](https://icwm.cc/call-for-papers), checked on 2026-10-31, identifies the event as **ICWM 2027** and lists:

| Milestone | Official provisional date |
| --- | --- |
| Paper submission | **2026-11-30** |
| Reviews released | 2027-01-11 |
| Rebuttal | 2027-01-11 through 2027-01-18 |
| Final decisions | 2027-02-08 |
| Conference | Late May 2027, San Francisco Bay Area |

The page says **all dates are provisional; exact deadline times are unconfirmed**. There are **30 elapsed days to submission**, not 70. Seventy elapsed days from October 31 ends **January 9, 2027**. Later experiments cannot be assumed admissible as replacements for a complete initial submission.

The inspected working-draft CFP does not establish a page limit, template, exact cutoff/time zone, complete required-section list, supplementary-material rules, or rebuttal experiment policy. Check these before preparing the final submission. Do not invent an eight-page limit or assume an AoE cutoff.

## Read in this order

1. [PROPOSAL.md](PROPOSAL.md): recommendation, closest competitors, alternatives, hypotheses, and the paper's claim boundary.
2. [EXPERIMENTS.md](EXPERIMENTS.md): exact intervention semantics, controls, metrics, splits, statistics, and stop conditions.
3. [ROADMAP.md](ROADMAP.md): the 30-day submission schedule, 70-day extension, GPU budget, and first 72 hours.
4. [LITERATURE.md](LITERATURE.md): 30 research-paper entries plus conference, software, checkpoint, and data sources.
5. [PREREGISTRATION.md](PREREGISTRATION.md): a draft checklist to lock before confirmatory experiments. It is **not yet a registered protocol**.
6. [experiment_matrix.csv](experiment_matrix.csv): machine-readable research queue; planned experiments are not completed research just because their software exists.

## The minimum useful finding

Establish all three, not just one:

1. A semantic editor works under the original appearance and is not merely changing its own scoring probe.
2. Appearance changes damage the editor more than the native counterfactual route, while native prediction and measurement remain sufficiently accurate on the same prespecified population.
3. The difference matters to **actual simulator outcomes of selected actions**, not just latent distance or a visualization.

If native prediction, measurement, or clean editing fails, the proposed interpretation is unsupported. A well-controlled null result is acceptable; it does not automatically make a publishable paper.

## What has and has not been done

**Implemented:** pinned downloads and source verification, original HDF5 loading,
frozen LeWM and RGB DINO-WM adapters, deterministic paired PushT generation,
appearance/replay checks, independent physical scoring, four editor families,
executed candidate regret, paired statistics and local protocol locks. See
[RUNNING.md](RUNNING.md) for the implemented scope and verification boundaries.

**Not established:** CUDA/Blackwell runtime or memory use, native task competence
on the audit population, a sufficiently accurate physical measurement system,
a transport failure, or a novel publishable result. Successful loading or a
passing smoke run does not establish any of these.

The current editor workspace is on **macOS**. No connection to the stated NVIDIA GPU host is established. Use a compatible Linux GPU host for the proposed CUDA experiments; do not assume the Mac exposes the Blackwell GPU.

**Decision:** begin with the reproducibility and simulator-replay checks in [ROADMAP.md](ROADMAP.md). Do not spend the first week building a new large model or writing an abstract that assumes the hypothesis is true.