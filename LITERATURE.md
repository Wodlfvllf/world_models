# Literature and artifact audit

**Audit date:** 2026-10-31. This is a targeted live-web review, not an exhaustive systematic review or a guarantee of novelty.

## Evidence conventions

- **Paper claim** means a claim made in the linked paper, not independently reproduced here.
- **Listed artifact** means a repository, configuration, or model-file listing was inspected. Nothing was loaded or execution-tested.
- **Not verified/not located** does not mean an artifact does not exist.
- Recent arXiv manuscripts remain prior art even without an established peer-reviewed venue. Their empirical claims still require scrutiny.
- The review used official venue pages, arXiv metadata/full texts, proceedings, project sites, GitHub, Hugging Face, and OSF listings. General-web searches had incomplete access; Google challenges and some OpenReview/search-page failures limited discovery. There was no bulk scrape of paywalled material.
- Several very recent papers are unusually close to plausible projects. Read them before allocating compute, and repeat forward/backward citation searches before claiming originality.

## A. Practical starting papers and model families

### P01 — DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

- **Date/status:** first preprint 2024-11-07; ICML 2025 is identified in the inspected literature.
- **Sources:** [paper](https://arxiv.org/abs/2411.04983), [project](https://dino-wm.github.io/), [official code](https://github.com/gaoyuezhou/dino_wm), [data/checkpoints](https://osf.io/bmw48/).
- **Contribution:** predicts future frozen DINOv2 patch features conditioned on controls; supports goal-image planning without a pixel reconstruction objective.
- **Practicality:** original release explicitly lists **PushT, PointMaze, and Wall** checkpoints. Do not infer released weights for all six environments in the paper.
- **Use here:** frozen patch-feature world-model case study. The original model's proprioceptive inputs must be preserved; dropping them is not a valid zero-cost adaptation.

### P02 — LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

- **Date/status:** first preprint 2026-03-13; inspected revision dated 2026-06-03.
- **Sources:** [paper](https://arxiv.org/abs/2603.19312), [project](https://le-wm.github.io/), [code](https://github.com/lucas-maes/le-wm), [model/data collection](https://huggingface.co/collections/quentinll/lewm).
- **Contribution:** approximately 15M-parameter, end-to-end JEPA with prediction loss and SIGReg; single 192D representation per image in the relevant release.
- **Practicality:** listed PushT, Reacher, TwoRoom, and single-cube checkpoints. Authors report training in a few hours on one GPU; this is **not** a measured Blackwell runtime.
- **Overlap:** already probes physical quantities and tests violation of expectation. Repeating those plots would not be a new contribution.

### P03 — Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations and Goals

- **Date/status:** 2026 preprint, arXiv 2608.29434; public code/model releases inspected.
- **Sources:** [paper](https://arxiv.org/abs/2608.29434), [code and release notes](https://github.com/fafraob/point-lewm), [RGB DINO-WM PushT checkpoint listing](https://huggingface.co/fafraob/point-cloud-pusht/tree/main/dino-wm).
- **Why relevant despite the title:** releases **RGB DINO-WM models trained with the original code, without proprioception**, on the LeWM task interfaces. These are the Point-LeWM authors' trained weights, not the original DINO-WM authors' weights.
- **Critical version issue:** October dataset changes replace some pixel views and repair point-cloud replay. Use the original scene-camera RGB contract for the released image baselines. Do not mix dataset versions silently.
- **Licensing:** code MIT; released datasets/checkpoints CC BY 4.0 according to the repository. Upstream components retain their own terms.

### P04 — Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models

- **Date/status:** first preprint 2025-02-20; inspected revision 2025-10-29. Method commonly called PLDM.
- **Sources:** [paper](https://arxiv.org/abs/2502.14819), [project](https://latent-planning.github.io/), [code](https://github.com/vladisai/PLDM).
- **Contribution:** compares model-based latent planning with offline goal-conditioned/model-free alternatives; navigation-focused.
- **Practicality:** small models and task/data-generation code. Original downloadable pretrained weights were not verified. Authors report experiments up to a day on a V100; not a new-task training estimate.
- **Boundary:** Two Rooms has displacement-style dynamics, not a rich contact-physics test. Do not make inertia claims from that environment.

### P05 — What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

- **Date/status:** first preprint 2025-12-30; revision 2026-09-02; arXiv comments report TMLR acceptance.
- **Sources:** [paper](https://arxiv.org/abs/2512.24497), [official implementation](https://github.com/facebookresearch/jepa-wms), [models](https://huggingface.co/facebook/jepa-wms), [data](https://huggingface.co/datasets/facebook/jepa-wms).
- **Contribution:** studies architecture, objectives, and planning choices with simulation and real robot data.
- **Practicality:** useful released action-conditioned models, but task-specific data access and native interfaces need qualification.
- **Overlap:** generic architecture/objective/planner sweeps are already occupied. Useful alternative substrate, not the first integration target.

### P06 — V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

- **Date/status:** 2025 preprint, arXiv 2506.09985; official repository contains subsequent releases.
- **Sources:** [paper](https://arxiv.org/abs/2506.09985), [official repository](https://github.com/facebookresearch/vjepa2), [V-JEPA 2-AC weights](https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt).
- **Distinction:** base V-JEPA 2/2.1 encoders are not interchangeable with **V-JEPA 2-AC**, the robot-action-conditioned system.
- **Practicality:** AC has a frozen approximately 1B encoder plus a large predictor, a Franka/DROID domain, and 7D action/proprioception conventions. Availability does not imply compatibility with PushT or a 5D OGBench controller.
- **Recommendation:** do not make adaptation to a new embodiment the critical path. A newly trained head on V-JEPA features studies that head, not the original predictor's causal use.

### P07 — TD-MPC2: Scalable, Robust World Models for Continuous Control

- **Date/status:** ICLR 2024.
- **Sources:** [project](https://www.tdmpc2.com/), [code](https://github.com/nicklashansen/tdmpc2), [checkpoint catalogue](https://www.tdmpc2.com/models), [HF models](https://huggingface.co/nicklashansen/tdmpc2).
- **Practicality:** extensive released task-specific checkpoints; small 5M single-task models are attractive controls.
- **Boundary:** standard released configurations are largely privileged-state models. RGB support in code does not establish a matched visual checkpoint. Not a drop-in PushT baseline.

### P08 — Mastering diverse control tasks through world models

- **Date/status:** Nature 2025; DreamerV3.
- **Sources:** [paper](https://www.nature.com/articles/s41586-025-08744-2), [author project](https://danijar.com/project/dreamerv3/), [author-maintained implementation](https://github.com/danijar/dreamerv3).
- **Contribution:** recurrent stochastic dynamics and policy learning through imagined trajectories.
- **Practicality:** mature implementation, but an official compatible manipulation checkpoint collection was not established in this audit.
- **Recommendation:** do not start a new RL training project merely to add one row to an interpretability table. Recurrent-memory interventions also have direct recent prior art below.

## B. Mandatory nearest-neighbor reading for the primary proposal

### P09 — Interpreting Physics in Video World Models

- **Date/status:** first preprint February 2026; **ICML 2026 proceedings verified**.
- **Sources:** [paper](https://arxiv.org/abs/2602.07050), [proceedings](https://proceedings.mlr.press/v306/joseph26a.html).
- **Already does:** layerwise physical probing, distributed direction geometry, targeted ablations and direction steering in video encoders.
- **Boundary:** steering a decoded direction is not the same as certifying physically correct autonomous action-conditioned rollouts.
- **Implication:** do not propose a V-JEPA layerwise probe/attention-map paper as new. Study-specific official code was not verified; upstream weights are available.

### P10 — What Do World Models Learn in RL? Probing Latent Representations in Learned Environment Simulators

- **Date/status:** 2026-03-23 preprint; ICLR 2026 world-model workshop listed in the inspected sources.
- **Source:** [paper](https://arxiv.org/abs/2603.21546).
- **Already does:** probes IRIS/DIAMOND environment state and perturbs hidden states along probe-derived directions.
- **Boundary:** output disruption or next-token KL does not establish a correct, specific physical intervention.
- **Artifacts:** exact study release not verified; [upstream IRIS checkpoints](https://huggingface.co/eloialonso/iris/tree/main/pretrained_models) are listed. These are not automatically the study's checkpoints.

### P11 — Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models

- **Date/status:** 2026-08-15 preprint; revision 2026-09-07.
- **Sources:** [paper](https://arxiv.org/abs/2608.15156), [code/artifacts](https://github.com/lysea8282/dynamic-effective-latent-carriers).
- **Already does:** simulator velocity interventions, native counterfactual hidden states, training-only low-rank editing maps, autonomous future rollouts, random/specificity controls, and fresh-checkpoint checks.
- **Boundary:** controlled two-object recurrent model with numerical state observations, including velocity, and counterfactual-supported training; not the proposed frozen pixel-input setting.
- **Implication:** native references plus low-rank physical editing are **not new**. Its artifact release is a useful positive-control resource, not a publication through replication alone.

### P12 — RolloutFaith: Auditing Persistent Internal Interventions in Visual World Models

- **Date/status:** 2026-09-29 preprint.
- **Sources:** [metadata](https://arxiv.org/abs/2609.36843), [full text](https://arxiv.org/html/2609.36843v1).
- **Already does:** reference activation patching, multiple fitted editors, delayed rollout effects, source transfer, norm controls, and memory/component restoration in DIAMOND, DreamerV3, and STORM. Also trains an editor through frozen future transitions.
- **Boundary:** chiefly factual repair and persistence, not certification of a specific state intervention under matched physical/appearance factors.
- **Implication:** “edits decay,” “memory matters,” and rollout-supervised editing cannot be headline novelty. Exact study code/checkpoints not located in this audit.

### P13 — The Intervention Gap in Latent World Models

- **Date/status:** 2026-08-30 preprint.
- **Sources:** [metadata](https://arxiv.org/abs/2608.29998), [full text](https://arxiv.org/html/2608.29998v1).
- **Already does:** capture-first evaluation, real-endpoint measurement controls, matched environment action interventions, native multi-step predictions, support/uncertainty analysis.
- **Boundary:** primarily environment action effects rather than a fixed internal semantic-state editor; explicitly limits decision-value claims.
- **Implication:** “good probes, wrong physical effects” is not new. Borrow the qualification discipline, not its central claim. Study-specific complete artifacts were not verified.

### P14 — GeoCo-SAVi: Geometry-Consistent Slot Attention for Explicitly Editable Object Representations

- **Date/status:** 2026-09-06 preprint; revision 2026-09-26.
- **Sources:** [paper](https://arxiv.org/abs/2609.06628), [full text](https://arxiv.org/html/2609.06628v3), [implementation](https://github.com/GeoCo-SAVi/GeoCo-SAVi).
- **Already does:** addresses geometric commands whose meanings change with appearance; appearance transplantation and calibrated position/scale semantics.
- **Boundary:** object-centric reconstruction/video editing, rather than a one-shot edit followed by autonomous action-conditioned physical rollout and executed counterfactual decisions.
- **Implication:** even “appearance changes edit meaning” has direct precedent. The proposed physical-dynamics consequence is essential. Pretrained weights were not located.

### P15 — Do-JEPA: From Masking to Intervention in Latent World Models

- **Date/status:** 2026-09-29 preprint.
- **Sources:** [paper](https://arxiv.org/abs/2609.37378), [full text](https://arxiv.org/html/2609.37378v1).
- **Already does:** exact simulator action forks, physical-effect prediction, same-state rendering-context twins, effect/invariance losses, and information-matched controls, including LeWM/PushT.
- **Boundary:** trains/adapts action-effect prediction rather than auditing the semantics of a frozen named physical-state editing interface.
- **Implication:** paired rendering plus counterfactual loss is not a new learning paradigm. Any small fine-tuning extension needs an explicit comparison. Dedicated artifacts were not located.

### P16 — D-JEPA: A Decision-Aligned Latent World Model

- **Date/status:** 2026-09-21 preprint.
- **Sources:** [paper](https://arxiv.org/abs/2609.24749), [code](https://github.com/NEBULIS-Lab/D-JEPA), [models](https://huggingface.co/Shuaijun/D-JEPA), [data](https://huggingface.co/datasets/Shuaijun/D-JEPA-Dataset).
- **Already does:** candidate-ranking failures, bounded relational corrections, executed decision outcomes, planning evaluation, and appearance-change studies.
- **Boundary:** decision alignment rather than semantic intervention transport. Adding regret plots alone is insufficient novelty.
- **Practicality:** real artifact listings exist, but cached candidate replay is not native simulator reproduction. Some profiles require upstream models/adapters. Model/data licensing was described as unfinished in inspected cards; verify before reuse.

### P17 — Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal Control

- **Date/status:** 2026-07-16 preprint.
- **Sources:** [paper](https://arxiv.org/abs/2607.14943), [code](https://github.com/trustworthyrobotics/steering_robustness_WAMs).
- **Already does:** contrastive activation geometry, training-free steering, reduced-order feedback control, and robustness to camera/gripper/image perturbations.
- **Boundary:** steering robustness is different from checking a named simulator state-edit correspondence, but it is close competing motivation.
- **Practicality:** large robot models and access/licensing requirements make it an expensive November baseline.

### P18 — CITRIS: Causal Identifiability from Temporal Intervened Sequences

- **Date/status:** 2022-02-07 preprint; ICML 2022.
- **Sources:** [paper](https://arxiv.org/abs/2202.03169), [code/data](https://github.com/phlippe/CITRIS).
- **Already does:** intervention-target-supervised causal representation learning, frozen-autoencoder variants, counterfactual factor manipulation, and unseen-factor generalization.
- **Implication:** causal factors, interchange consistency, and intervention generalization are established territory. Do not claim identifiability from an empirical editing audit.

### P19 — Addressing divergent representations from causal interventions on neural networks

- **Date/status:** first preprint 2025-11-06; inspected revision April 2026.
- **Sources:** [paper](https://arxiv.org/abs/2511.04638), [implementation](https://github.com/grantsrb/rep_divergence).
- **Already does:** explains how neural interventions can create misleading representations and uses natural counterfactual representation matching.
- **Implication:** native-reference alignment and warnings about off-manifold behavior are methodological precedent, not new contributions. Implementation inspection does not establish a compatible WM checkpoint.

### P20 — Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

- **Date/status:** 2020 preprint; TACL 2021.
- **Sources:** [paper](https://arxiv.org/abs/2006.00995), [code](https://github.com/yanaiela/amnesic_probing).
- **Already does:** distinguishes encoded information from behavioral use by erasure experiments.
- **Implication:** “decodable does not mean used” is foundational, not a new world-model discovery. Erasure remains vulnerable to collateral damage and redundant/nonlinear coding.

### P21 — LEACE: Perfect linear concept erasure in closed form

- **Date/status:** 2023 preprint, with later revisions.
- **Sources:** [paper](https://arxiv.org/abs/2306.03819), [implementation](https://github.com/EleutherAI/concept-erasure).
- **Contribution:** closed-form linear concept erasure with a specified distortion objective.
- **Use here:** optional diagnostic baseline only. It does not erase every nonlinear encoding or certify a valid physical intervention.

### P22 — Causal Abstractions of Neural Networks

- **Date/status:** 2021 preprint; NeurIPS 2021.
- **Sources:** [paper](https://arxiv.org/abs/2106.02997), [code](https://github.com/atticusg/Interchange).
- **Contribution:** correspondence between high-level variables and neural representations tested through interchange interventions.
- **Implication:** the proposed commutation/consistency framing is an application of established causal-abstraction methodology, not a new definition of causality.

## C. Alternatives and reasons not to select them first

### P23 — Calibrated for Which Policy? Conformal Uncertainty for Fixed World Models Under Intervention

- **Date/status:** 2026 preprint. **Metadata anomaly:** the inspected page displays 2026-08-20 despite identifier 2610.35877; retain this discrepancy rather than silently correcting it.
- **Sources:** [paper](https://arxiv.org/abs/2610.35877), [code/derived artifacts](https://github.com/noumenal-ai/conformal-realrobot-lab).
- **Already does:** weighted conformal calibration of a frozen DINO-feature model under policy-associated covariate shifts, assuming invariant conditional outcomes and support overlap.
- **Possible extension:** actual policy rollouts can alter hidden-state mixtures within an insufficient observable representation, violating that conditional-invariance assumption. This tests applicability, not the theorem.
- **Boundary:** its controlled physical-robot data study reweights a fixed pool; a new closed-loop study must do more than that.

### P24 — How Well Do Latent World Models Understand Partially Observable Safety Constraints?

- **Date/status:** first preprint 2025-10-07; inspected revision 2026-10-12.
- **Sources:** [paper](https://arxiv.org/abs/2510.06492), [project](https://cmu-intentlab.github.io/multisafe/).
- **Already does:** estimation versus prediction gaps, hidden safety variables, privileged multimodal supervision, and conformal risk mitigation on hardware.
- **Implication:** “partial observability breaks safety; add conformal” is occupied. Project code was marked coming soon during inspection.

### P25 — Hallucination in World Models is Predictable and Preventable

- **Date/status:** 2026-06-25 preprint.
- **Sources:** [paper](https://arxiv.org/abs/2606.27326), [interactive paper](https://www.nicklashansen.com/mmbench2), [code](https://github.com/nicklashansen/mmbench2), [models](https://huggingface.co/nicklashansen/mmbench2-models).
- **Already does:** three hallucination modes, data-centric failure signals, targeted collection/fine-tuning, and MPC evaluation.
- **Practicality:** listed 350M model and live simulator infrastructure are plausible frozen-inference resources. Full training is not the recommended single-GPU project; the reported pipeline is large.
- **Possible extension:** whether error signals detect decision-changing rather than visually large errors, with explicit controls for goal-metric failure. Crowded by P13/P16 as well.

### P26 — Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models

- **Date/status:** 2026-10-02 preprint; arXiv comments list a NeurIPS 2026 workshop publication claim, not independently verified here.
- **Sources:** [paper](https://arxiv.org/abs/2610.02860), [full text](https://arxiv.org/html/2610.02860v1), [advertised replication package](https://osf.io/t432p/).
- **Already does:** exact simulator forks, tracing physical effects through raster visibility, target embeddings, and prediction; action-path and identifiability controls.
- **Boundary:** controlled physical rasters rather than camera RGB; direction/control follow-up is explicitly suggested. Package contents were not independently downloaded.
- **Implication:** action shuffling or response-magnitude diagnostics alone are not new and need observation-sufficiency controls.

### P27 — Learning Counterfactual World Models for Embodied Reasoning under Partial Observability

- **Date/status:** 2026-09-05 preprint; revision 2026-10-25; comments report NeurIPS 2026 workshop acceptance.
- **Source:** [paper](https://arxiv.org/abs/2609.05834).
- **Already does:** recurrent beliefs, counterfactual separability, occluded manipulation, aliased navigation, and planning-exploitation analysis.
- **Boundary:** explicitly does not yet test large pretrained encoders. That omission alone is not enough novelty for a new paper. Exact code/checkpoints not located.

### P28 — DiLA: Disentangled Latent Action World Models

- **Date/status:** 2026-05-15 preprint; paper/project material identifies ICML 2026.
- **Sources:** [paper](https://arxiv.org/abs/2605.15725), [code](https://github.com/senngadaisuki/disentangled-latent-action-world-models), [weights](https://huggingface.co/senngadaisuki/disentangled-latent-action-world-models).
- **Already does:** content/structure separation, latent action abstraction, transfer and planning.
- **Implication:** action semantics across appearance and disentanglement are established objectives. Not a same-interface frozen PushT state-editor baseline.

### P29 — Learning to Act Robustly with View-Invariant Latent Actions

- **Date/status:** 2026-01-06 preprint; project material identifies CVPR 2026. Method: VILA.
- **Sources:** [paper](https://arxiv.org/abs/2601.02994), [code](https://github.com/joon-stack/vila-code).
- **Already does:** physically grounded action representations that transfer across viewpoints.
- **Implication:** camera-equivariant learning is not new. The first proposed experiment deliberately avoids viewpoint changes to avoid coordinate/visibility confounds.

### P30 — Pixels to Proofs: Probabilistically-Safe Latent World Model Control via Parallel Conformal Robust MPC

- **Date/status:** 2026-06-14 preprint.
- **Sources:** [paper](https://arxiv.org/abs/2606.15594), [code](https://github.com/trustworthyrobotics/SLS-squared).
- **Already does:** conformal error bounds, learned constraints, and robust latent MPC.
- **Implication:** a generic calibrated-planning wrapper is insufficient. Code/data-generation scripts are available; matching ready-to-use checkpoints were not verified.

## D. Artifact decision table

| Resource | What was actually verified | Main risk | Decision |
| --- | --- | --- | --- |
| [LeWM PushT](https://huggingface.co/quentinll/lewm-pusht/tree/main) | 72.3 MB weights listing plus configuration; MIT card | Loader/library version drift; three-frame/action-block alignment | **Primary** |
| [RGB/no-proprio DINO-WM PushT](https://huggingface.co/fafraob/point-cloud-pusht/tree/main/dino-wm) | 88.8 MB package listing with action statistics/configuration | Separate training provenance; original RGB data contract; backbone fetched separately | **Second case study** |
| [Original DINO-WM](https://github.com/gaoyuezhou/dino_wm) | Official release instructions for PushT/PointMaze/Wall | Old CUDA dependencies; proprioception; older simulator stack | Replication anchor/fallback |
| [LeWM Cube](https://huggingface.co/quentinll/lewm-cube/tree/main) | 72.3 MB weights/configuration listing | 3D contact/renderer/replay qualification | 70-day extension |
| [Point-LeWM release](https://github.com/fafraob/point-lewm) | Matching image baseline interfaces for several tasks | Point-cloud and image model names are easy to confuse | Shared task infrastructure, not all modalities |
| [JEPA-WMs](https://github.com/facebookresearch/jepa-wms) | Models/data/evaluation links | Additional integration and dataset terms | Alternative substrate |
| [TD-MPC2](https://www.tdmpc2.com/models) | Extensive task-specific checkpoint catalogue | Mostly privileged state; different objectives/task | Optional state-input control |
| [DreamerV3](https://github.com/danijar/dreamerv3) | Code, not a qualified matching pretrained manipulation agent | New RL training/integration | Not critical path |
| [V-JEPA 2-AC](https://github.com/facebookresearch/vjepa2) | AC checkpoint and robot interface | Embodiment and action-domain mismatch | Not critical path |
| [MMBench2](https://github.com/nicklashansen/mmbench2) | Frozen model/live-simulator release | Much larger evaluation stack; training/data footprint | Backup direction only |

**Software source:** [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) supplies environments, planning and visual/physical variation controls. Its APIs and data formats have changed; pin a known working commit rather than assuming the newest package matches every checkpoint.

**GPU source:** [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/) document Blackwell support with CUDA 12.8 builds. Use a currently compatible PyTorch/CUDA/driver combination and validate it on the actual device. An older paper's lockfile is not necessarily Blackwell-compatible.

## E. Reading/search order before the pilot

1. P02/P01/P03: reproduce the exact model and task contracts.
2. P11/P12/P13: understand what causal editing/rollout auditing already establishes.
3. P14/P15/P16/P17: test the proposed novelty against appearance semantics, rendering twins, decision alignment, and robustness steering.
4. P18/P19/P20/P22: write the limitations of causal interpretation correctly.
5. P23/P24/P25/P30 only if considering a backup.

Repeat searches combining world model with semantic editing, intervention transport, appearance transplantation, causal abstraction, counterfactual rollout, nuisance equivariance, and decision regret. Follow citations of P12–P17. Record new disconfirming prior art, not just papers supporting the desired story. A literature gap is **not** established by failure to find a keyword match.