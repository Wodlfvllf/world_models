# Research proposal and novelty decision

**Prepared:** 2026-10-31. All hypotheses below are proposals, not findings.

## 1. Choose one question, not a collection of benchmarks

### Recommended question

**When does a physical-state edit retain its meaning across appearance changes in a frozen visual world model?**

For example: a small editor claims to rotate a T-shaped object by 10 degrees inside a model. It appears successful when a probe reads the requested angle. Does the same editor still produce the right future contact and motion when the background or object color changes? Does it select actions appropriate to that rotated object?

This is not a request for a more photorealistic video. The reference is an independently executed simulator trajectory under an explicitly defined initial-condition intervention.

### Recommended implementation

| Component | Initial choice | Reason |
| --- | --- | --- |
| Main world model | Frozen LeWM–PushT | Small state, listed weights, simple simulator, cheap editor fitting |
| Second world model | Frozen RGB/no-proprioception DINO-WM–PushT from the Point-LeWM release | Distinct representation and predictor interface; matching image task infrastructure |
| Physical variables | Block translation and orientation | Observable and manipulable without hypothesizing hidden material properties |
| Appearance factors | Block color, background color/texture, held-out combinations | Can preserve physical state, visibility, camera and action semantics |
| Training | Small probes and state editors only | Avoids world-model pretraining or RL |
| Behavioral endpoint | Executed counterfactual candidate-bank regret | No dependence on a probe for the final outcome |
| Extension | OGBench single cube with matching frozen weights | Adds a second physical domain, not merely another color palette |

The DINO-WM checkpoint provenance must be explicit. The Point-LeWM release is not the original authors' training run. Two pretrained backbones are **case studies**, not a controlled comparison proving frozen-feature learning better/worse than end-to-end JEPA.

## 2. Why the obvious proposals are insufficient

The strongest overlaps are not peripheral:

| Tempting claim | Existing close work | Consequence |
| --- | --- | --- |
| Physical state is decodable | [Interpreting Physics](https://arxiv.org/abs/2602.07050), [LeWM](https://arxiv.org/abs/2603.19312), [RL probing](https://arxiv.org/abs/2603.21546) | A new probe table is insufficient |
| Decodable does not mean used | [Amnesic Probing](https://arxiv.org/abs/2006.00995), [Intervention Gap](https://arxiv.org/abs/2608.29998) | Not a new conceptual discovery |
| Small edits can launch counterfactual rollouts | [Latent Carriers](https://arxiv.org/abs/2608.15156) | Must go beyond low-rank editing/native-reference controls |
| Edits must persist; train through future steps | [RolloutFaith](https://arxiv.org/abs/2609.36843) | Persistence and rollout-supervised editors are baselines |
| Geometry edits depend on appearance | [GeoCo-SAVi](https://arxiv.org/abs/2609.06628) | Appearance-dependent semantics alone is already occupied |
| Use simulator forks and render twins | [Do-JEPA](https://arxiv.org/abs/2609.37378) | That data construction/loss is not a new paradigm |
| Low latent error can give bad decisions | [D-JEPA](https://arxiv.org/abs/2609.24749) | Regret plots alone do not rescue novelty |
| Steer a world model to improve robustness | [Robustness steering](https://arxiv.org/abs/2607.14943) | Generic activation steering is crowded |

### The remaining, conditional contribution

Test a **specific dissociation**:

> A physical counterfactual remains accessible through the frozen model's native observation interface, but a source-trained named state editor stops reproducing its physical consequences under appearance changes, with a measurable cost to counterfactual action selection.

The examined nearest neighbors separately cover most ingredients. None was identified as an exact implementation of this entire frozen-editor/physical-state/appearance/decision experiment. **That is a bounded search result, not proof of novelty.**

The paper must offer more than “OOD editors generalize poorly.” It should isolate whether the bottleneck is:

1. the original observation or representation;
2. the native dynamics;
3. the learned editing interface;
4. the physical-state measurement;
5. the goal-cost geometry.

The candidate is worth a pilot because these explanations can be separated with released models and modest training. It is not yet a defensible positive result.

## 3. Falsifiable hypotheses

### H1 — Editor-specific loss of physical meaning

Appearance changes increase a fixed editor's physical-effect error **more than they increase native counterfactual prediction error**, on an unfiltered, prespecified population of simulator roots.

- **Supporting evidence:** the paired native-subtracted transport contrast is materially positive **and the editor's own shifted-minus-clean error materially increases**; clean editing and native prediction pass independent competence/equivalence checks. A positive interaction alone could arise because native prediction improved more than editing.
- **Falsifier:** the contrast is practically negligible, or native degradation/readout failure explains the effect.
- **Not sufficient:** the edited latent lies far from a reference, a probe changes, or a few selected videos look wrong.

### H2 — Immediate semantic success is an incomplete indicator

Some edits satisfy a prespecified immediate semantic criterion yet fail at later physical consequences under appearance changes.

- Report the **joint frequency over all roots** of immediate success and later failure, with controls.
- Do not condition the main outcome analysis on whichever examples pass a test-time probe.
- If immediate failure explains everything, the narrower temporal dissociation is false. Report ordinary editor OOD failure rather than relabeling it.

### H3 — The error has decision consequences

Appearance-induced editor error raises the physical cost of the selected action sequence relative to native initialization, even when every method receives the same candidate bank and goal.

- Score the action actually selected by each route in the true counterfactual simulator.
- Compare with ranking true endpoints through the unchanged latent goal metric.
- If that endpoint-metric oracle already fails badly, do not attribute the entire decision gap to the editor or predictor.
- This establishes **counterfactual query utility**, not automatically improved ordinary closed-loop control.

### H4 — A modest repair targets the identified failure

An appearance-aware or augmentation-trained editor reduces the transport gap without degrading clean editing or non-target fidelity relative to the **true counterfactual**. Non-target variables are held fixed only at the physical intervention anchor; their subsequent causal changes are legitimate.

- This is optional until H1–H3 are credible.
- Strong controls receive identical pairs, labels, render twins, rollout supervision, parameter budgets, and tuning budgets.
- If simple nuisance augmentation/canonicalization performs equally well, prefer it. Do not invent a sophisticated method claim.

## 4. Small learning intervention, if a method is warranted

First fit three established editor families: inverse-probe steering, paired-native-difference ridge editing, and a small nonlinear residual editor. Include a variant trained through several frozen future transitions, following the motivation already established by RolloutFaith.

For the optional extension, use the same physical pair rendered in several contexts to supervise the editor's target latent history. A candidate loss is:

$$
\mathcal L(\phi)=\mathcal L_{\mathrm{native\ target}}
+\lambda_{\mathrm{future}}\mathcal L_{\mathrm{frozen\ rollout}}
+\lambda_{\mathrm{zero}}\mathcal L_{\mathrm{zero\ request}}.
$$

All terms use training-only pairs. World-model parameters remain fixed. A later real counterfactual frame can be a **training target**, never an input to the editor at evaluation.

This loss is deliberately not presented as new. The learning question is whether training exposure to nuisance changes, temporal consequences, or both is necessary to restore physical semantics. Use a matched **2 × 2 design**:

| | Immediate native-state target | Native-state target plus future rollout loss |
| --- | --- | --- |
| Original appearance only | Baseline editor | Temporal-supervision baseline |
| Same roots with nuisance augmentation | Augmentation baseline | Combined training condition |

Match total updates, sampled physical roots, and total example presentations. Re-rendered examples supply additional variation, not additional independent physical trajectories. Label the data budget accordingly.

A learned appearance-transport map is a further option, not a commitment. Unless its mathematical conditions are proved, call it **approximate transport**, not guaranteed equivariance. Count its parameters and any nuisance inference network in the total budget. An oracle nuisance label must be supplied to every relevant comparator or kept in a clearly privileged analysis.

## 5. Scope and publication bar

### Potential November submission

- One contact-rich manipulation environment, **two frozen model case studies**.
- Translation and rotation interventions; two appearance families plus their composition.
- Complete-context edits, native counterfactual references, strong editor controls.
- Independently measured multi-step physical effects and executed candidate-bank regret.
- Held-out root families and appearance settings; uncertainty at the root/episode level.
- A reproducible release of the audit and all outcomes, including failures.

This remains a **narrow one-environment study**. A second architecture does not substitute for a second physical domain. If the contribution is only a small effect on a single weak editor, do not force a main-conference story.

### Stronger 70-day study

- Replicate the central result on OGBench single cube or another qualified native task.
- Add independently trained small-model checkpoints if feasible.
- Test the matched 2 × 2 editor-training experiment.
- Include practical counterfactual planning or a well-defined closed-loop correction application.
- Explain when the failure disappears, not only when it occurs.

### Claims explicitly out of scope

- The model does or does not “understand physics” in general.
- Linear erasure removes all information about a variable.
- A failed editor proves the transition never uses that variable.
- A low-rank editor reveals the intrinsic dimension of the physical state.
- A native donor combined with incompatible history is automatically on-manifold.
- Frozen-feature learning is superior/inferior to end-to-end learning based on unmatched released checkpoints.
- Better counterfactual action selection automatically increases standard task success.

## 6. Ranked backup directions

Do not run these in parallel with the main study. A switch is a new exploratory decision and requires a fresh protocol/test population.

### Backup A — Policy-induced hidden-mixture shift breaks observable-space calibration

**Question:** Can two policies induce different future-outcome laws at the same available model input because the input aliases hidden physical states?

For full physical state $s$, model input/representation $z$, action $a$, and outcome $Y$:

$$
P_\pi(Y\mid z,a)=\int P(Y\mid s,a)P_\pi(s\mid z,a)\,ds.
$$

Unchanged physics does not guarantee unchanged observable-conditional outcomes. This identity is familiar; the contribution would be a practically significant, localized failure in a useful frozen-model interface.

**Experiment:** on native action-conditioned models, compare source calibration, observable importance weighting, history-aware weighting, privileged-state oracle weighting, and fresh target-policy calibration. Use both an analytically controlled overlapping mixture and actual new policy rollouts. Include all native context and actions in the observable interface; do not manufacture aliasing by truncating the model's history.

**Close work:** [Calibrated for Which Policy?](https://arxiv.org/abs/2610.35877), [Partially Observable Safety Constraints](https://arxiv.org/abs/2510.06492), [Pixels to Proofs](https://arxiv.org/abs/2606.15594), [Counterfactual WMs under Partial Observability](https://arxiv.org/abs/2609.05834).

**Falsifier:** observed-space correction works within practical tolerance once support, density-ratio estimation, and available history are handled correctly.

**Risk:** sophisticated identification/statistics and additional policy integration. Simulator access alone does not provide an exact image-history density ratio. Prediction-set coverage is not failure probability among accepted actions. Strong scientific question, less ready for November.

### Backup B — Do world-model error signals detect decision-changing mistakes?

**Question:** Do existing signals reject visually large but harmless changes while missing small errors that reverse action selection?

**Experiment:** frozen MMBench2 or qualified JEPA-WM, fixed candidate banks, paired appearance-only versus contact-boundary perturbations, actual simulator regret, endpoint-metric oracle, and a held-out risk–utility evaluation. Search-budget changes must be reported as extra compute and distribution changes.

**Close work:** [Hallucination in World Models](https://arxiv.org/abs/2606.27326), [D-JEPA](https://arxiv.org/abs/2609.24749), [Intervention Gap](https://arxiv.org/abs/2608.29998), [Observation Bottlenecks](https://arxiv.org/abs/2610.02860).

**Falsifier:** existing signals correctly rank decision risk, or goal-metric error explains the apparent mismatch.

**Risk:** particularly crowded. A correlation table or an AUROC improvement alone is not enough. Decoder-free models do not expose every uncertainty signal used by generative models.

### Not recommended as the initial project — “pixels versus latent prediction”

Off-the-shelf Dreamer versus DINO-WM versus LeWM changes pretraining, inputs, context, capacity, objectives, stochasticity, planner, and data. It cannot isolate a learning paradigm.

A defensible small 70-day extension could fix input encoder, transition architecture, data, context, planner, and evaluation cost while varying the training target/loss. A fixed-decoder pixel loss versus feature loss studies **prediction targets given that representation**, not the entire end-to-end-versus-frozen paradigm. P05 and P15 already provide relevant comparison methodology and close objective-based work.

## 7. Paper structure if the empirical gates pass

1. **Introduction:** semantic edit validity is not guaranteed by a successful probe or native world-model robustness.
2. **Related work:** explicitly separate the contribution from P11–P19, not merely older generic world-model papers.
3. **Protocol:** physical/history/appearance contract, information access, native references, outcome measurement.
4. **Main evidence:** full-population transport contrast and native-competence tests.
5. **Decision consequences:** actual counterfactual regret with endpoint-metric controls.
6. **Mechanism/repair:** matched editor training and alternative explanations, if supported.
7. **Limitations:** checkpoint-specific evidence, state/readout ambiguity, support restrictions, one-domain limits, and negative cases.

Planned figures: protocol schematic; physical-effect error across horizons; editor/native appearance interaction; immediate-success/later-failure joint rates; candidate regret with the endpoint oracle; repair and counterexample table. None should be populated with invented numbers.