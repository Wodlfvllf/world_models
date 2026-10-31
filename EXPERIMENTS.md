# Experimental protocol: physical-state edit transport

**Status:** proposed protocol, not executed or preregistered. Numerical budgets below are starting allocations, not power guarantees or observed performance.

## 1. Experimental unit and information contract

A root family contains:

- one independently generated simulator episode/root;
- a valid factual physical state and one requested initial-condition change;
- a common prefix-action sequence and complete model context;
- a goal and fixed candidate future-action bank;
- every rendering, intervention method, horizon, and random-seed replicate derived from that root.

**Independent unit:** root family, or source episode if multiple roots share an episode. Candidates, frames, horizons, appearance variants, and probe/editor seeds are repeated measurements, not independent samples.

The initial study uses a **frozen world model**. An editor receives the factual latent context and a named request, such as a translation vector or rotation amount. It receives no native counterfactual latent or future image at test time. The native-reference route receives the real counterfactual observation context and is an **oracle diagnostic**, not a fair deployable method competitor with identical information.

## 2. E00–E02: qualify the software, physics and measurement

### Checkpoint contract

For every checkpoint record source URL, exact revision, SHA-256 after download, model configuration, preprocessing, normalization, precision, and environment commit. Require strict weight loading; missing/unexpected keys must be explained, not ignored.

Known relevant contracts:

- LeWM–PushT has 224 × 224 image configuration, a 192D representation, a three-frame predictor context, and a 10D action embedding input in the inspected configuration.
- The relevant DINO-WM release uses three context frames and a frame skip of five. A model transition corresponds to a block of **five primitive 2D controls**, not one command repeated five times.
- Check the actual encoder's resize path. Nominal image size and DINO patch-token count are not interchangeable. Preserve the checkpoint's behavior.
- Confirm whether the selected implementation uses relative displacement controls and dataset-normalized actions. Do not mix absolute-target PushT conventions with relative-action variants.
- Convert every reported horizon into primitive simulator controls and physical time using the pinned environment. For a five-control block, horizons 1/4/8 mean 5/20/40 controls.
- Evaluation mode must disable dropout and update neither batch-normalization statistics nor world-model parameters.

**Exact scoring surface:** record the encoder → projector/normalizer → native rollout-output path for each checkpoint. Fit readouts on real observations transformed into the **same representation emitted by that rollout**, with identical token pooling and time alignment if scoring requires pooling. A raw-encoder probe cannot automatically score projected predictions. Native substitution, editor targets and the true-endpoint metric oracle must use their exact corresponding interfaces; representation dimensions alone do not establish parity.

### Replay and rendering acceptance tests

1. Reproduce the same unedited trajectory from the same complete state/history and action sequence.
2. Verify that changing only renderer appearance leaves physical states, contacts, rewards and termination unchanged.
3. Verify that sham/zero-request editing reproduces the unedited latent rollout to the declared numeric tolerance.
4. Replacing the complete rollout-entry context with the native counterfactual context must reproduce a directly initialized native counterfactual rollout.
5. Verify strict train/validation/test source-family disjointness and absence of future-observation access during rollout.
6. Save audit examples and machine-readable assertions before fitting an editor.

**PushT trap:** a commonly recorded seven-number state does not include every block velocity/physics variable. Pose alone may not restore a moving contact state. State setters may advance physics or change geometry depending on angle/position order. Prefer validated full reset/replay from episode start or a complete snapshot; test contacts as well as positions. No causal result is interpretable before this passes.

### Unmodified model replication

Run the published native preprocessing/planning configuration on a small sample first. Report published-protocol replication separately from the fresh-root causal audit. Check any success-based termination and set a finite action/planning budget.

At several horizons report native prediction quality relative to persistence, zero-effect, and simple kinematic baselines. Do not equate an executable forward pass with a competent world model. A large deviation from the reference behavior must be investigated before interpretability.

## 3. E03: physically valid counterfactual construction

### Start with initial-condition interventions, not mid-history teleportation

Apply the physical change **before the start of the observation context**. Generate both histories through the simulator under identical prefix controls. Initially use valid, contact-free resting prefixes so translation/rotation retains an unambiguous meaning. Subsequent candidate actions should include contact and non-contact behavior.

This changes an initial condition. It is **not** an instantaneous $do$-operation at the last context frame. Changing only that frame in a three-frame history would create an unintended velocity/teleportation cue.

Non-target physical variables match the factual branch **at the intervention anchor**, before the prefix. After that, legitimate dynamics may change them. Score later non-target fidelity against the native counterfactual trajectory, not permanent equality to the factual branch. “Immediate edit success” means success at rollout entry: compare with the true counterfactual endpoint of the prefix, which need not equal a simple offset from the factual endpoint after intervening dynamics.

Suggested development grid, to be fixed before confirmation:

- translations along positive/negative X or Y, initially 16 or 32 units in a 512-unit arena, rescaled if the pinned implementation differs;
- rotations of ±10 or ±20 degrees around the block's specified physical center;
- zero-request controls;
- geometrically valid starts with no imposed object/wall interpenetration.

Geometric validity rules may reject invalid initializations **before** model evaluation. Record rejection counts. Do not reject valid roots because a future action produces little effect, a method fails, or the goal is not reached.

Do not initially vary mass/friction or claim velocity editing: material variables may be unidentifiable, and an instantaneous velocity change can leave a rendered image unchanged. A reference with extra future frames would receive more information than the editor.

### Rendering factorial

For every physical branch render:

| Appearance condition | Physical state | Color/background |
| --- | --- | --- |
| C0 | Unchanged | Original |
| C1 | Unchanged | Color-only change |
| C2 | Unchanged | Background-only change |
| C3 | Unchanged | Held-out color/background combination |

Apply the same rendering choice to **all context frames, corresponding endpoints and goal images**. Keep silhouette, object geometry, camera, occlusion and adequate contrast fixed. Sample colors/textures independently of physical requests and state labels. Color is a nuisance only because the task does not use it to identify a goal/object.

Prefer nuisance parameters sampled independently per root from fixed development-chosen ranges, with frozen condition weights. If a fixed palette/texture roster is reused instead, state that inference is conditional on that roster. Primary editors are fitted on C0 only; nuisance-aware editors belong to the separate repair experiment. Evaluation readouts may see training-only nuisance examples to qualify measurement, but record that exposure explicitly. Keep composition settings designated for final generalization out of editor fitting **and** hyperparameter selection.

Start with mild, fixed variation. If it destroys native prediction, that is an ordinary model robustness problem, not the desired editor-specific dissociation. Choose the render distribution on development data and never escalate it based on test outcomes.

## 4. E04–E06: readouts and editor baselines

### Readout targets

Use pusher XY, block XY, and block orientation encoded as $(\sin\theta,\cos\theta)$. Report angular error in degrees using circular differences. Contact/proximity can be a secondary event label, not a replacement for continuous geometry.

Train readouts on **real encoded states**, with separate source families for fitting, validation and confirmation. Include ridge and a small MLP, shuffled-label controls, and a raw-image/simulator-label measurement check. If DINO features are reduced, fit dimensionality reduction only on fitting data and record its dimension.

An editor-construction probe and an evaluation readout must be fitted separately. A separately fitted readout is still an imperfect instrument, especially on edited or predicted latents. Agreement between readouts reduces one concern; it does not prove that off-manifold latents have a unique physical interpretation. Simulator action outcomes are the necessary corroboration.

Check readout error on re-encoded real endpoints **for every appearance condition**. Freeze capacity/regularization before test. A nominal starting tolerance is an error small relative to the smallest physical request—for example, one quarter of that request—but the actual tolerance must be justified and locked before confirmation.

### Required routes

| ID | Route | Information/access | Purpose |
| --- | --- | --- | --- |
| N | Native factual and native counterfactual initialization | True branch contexts | Available counterfactual competence; privileged reference |
| E0 | Identity/no edit | Factual context | Do-nothing baseline |
| E1 | Inverse-probe steering | Factual context + request | Standard weak semantic editor |
| E2 | Paired-difference ridge editor | Factual context + request; training-only native pair targets | Strong cheap linear baseline |
| E3 | Small residual MLP editor | Same fitting pairs/request | Nonlinear baseline |
| E4 | Rollout-supervised editor | Same access; future targets only in training | Temporal-supervision baseline, not a new invention |
| R | Norm/rank/location-matched random edit | Same realized perturbation budget | Generic-damage control |
| W | Shuffled/wrong request | Same request magnitude and allowed inputs | Specificity control |
| Z | Zero-request/sham | Zero requested physical change | Unsolicited-change control |

Fit each editor to replace/correct the **complete relevant rollout-entry latent history once**. For DINO, record the patch-token site precisely. A pooled probe does not justify replacing the native patch state with a pooled vector. A parameter-limited patchwise/shared residual module is preferable to an enormous flattened dense network.

No future real frame, repeated correction, or physical-state clamping is allowed in the autonomous rollout. Match random controls to the **realized** perturbation after clipping/precision rounding. Full native substitution is a reference, not evidence that a partial donor/recipient hybrid is on-manifold.

Suggested editor search budget: ranks 4/8/16 for low-rank variants and at most six prespecified tuning configurations per family. Select one main editor per family using **clean validation fidelity and specificity**, not the largest nuisance failure. If vulnerability-based selection is used in a separate exploratory analysis, disclose it and narrow that claim. Three editor-fitting seeds are a reasonable target; report their variability separately from root variability and world-model training seeds.

Optional LEACE/amnesic-erasure studies come later. They require rank/distortion controls, residual linear/nonlinear probing and rescue tests. A null effect from one eraser is not evidence that the variable is unused.

## 5. E07: primary physical-effect metric

Let $i$ index independent root families, $k$ a fixed candidate action sequence, $h$ rollout horizon, $c$ appearance, and $p\in\{0,1\}$ factual/counterfactual physical branches. Let $y_{ikh}^{p}$ be simulator physical features.

For native route $N$, both branches start from their real contexts. For editor $E$, branch 1 starts from the edited factual context and branch 0 from its untouched factual context. Let $\hat y_{ikh}^{m,p}(c)$ be predicted features from a frozen evaluation readout.

Define the error in the physical **effect**, not merely the absolute endpoint:

$$
L_i^m(c)=\frac{1}{K|\mathcal H|}\sum_{k=1}^{K}\sum_{h\in\mathcal H}
\left\|W\left[(\hat y_{ikh}^{m,1}(c)-\hat y_{ikh}^{m,0}(c))
-(y_{ikh}^{1}-y_{ikh}^{0})\right]\right\|_2^2.
$$

$\mathcal H$ is the fixed horizon set, initially proposed as $\{1,4,8\}$ with equal weights. $W$ fixes physical units/scales using development-only choices. Do not normalize each test example by its realized effect; that becomes unstable near zero and changes the estimand. Keep zero-effect cases. Freeze nuisance-condition and contact-stratum weights as well.

The primary transport interaction is:

$$
\tau_E=\mathbb E_{i,c\sim\Pi}\left[
\{L_i^E(c)-L_i^N(c)\}-\{L_i^E(C0)-L_i^N(C0)\}
\right],
$$

where $\Pi$ is the frozen nuisance distribution. Report each checkpoint and editor separately, plus the prespecified appearance breakdown.

**Interpretation:** positive $\tau_E$ means nuisance raises this editor's excess physical-effect error relative to native initialization. It is a paired functional contrast, not proof of a uniquely identified neural causal mechanism.

To claim actual appearance-induced **damage**, additionally require a materially positive within-editor contrast $\mathbb E[L_i^E(c)-L_i^E(C0)]$. Otherwise both routes could improve, with native prediction improving more. Set native-equivalence margins tighter than changes that could account for the claimed excess effect. Apply the same requirement to any regret-damage claim.

### Essential companion metrics

- Absolute state/trajectory error for both native physical branches; effect differences can cancel common bias.
- Native effect error against zero effect and an **initial-offset-carried-forward** baseline. A translation that is merely retained is not evidence of learned contact dynamics.
- Immediate target error and non-target changes.
- Effect direction agreement/cosine only where physically meaningful; retain absolute errors and all denominators.
- Feature-space distance to encoded true future observations as a supplementary, not sufficient, metric.
- Joint rate of immediate semantic success and later physical failure over the **entire** root population.
- Counts of invalid numerical outputs and measurement failures. Do not silently drop them; specify finite failure penalties or a separate failure endpoint before testing.

### No post-outcome qualification gate

Native competence, native robustness and readout validity are **parallel claims on the same population**. They are not per-example filters for the main test.

Development data may determine the task/horizon/appearance scope. Once frozen, confirmation retains every valid prespecified root, including native failures. If native competence fails in confirmation, report that failure and withdraw the editor-specific interpretation for that condition. Do not remove difficult examples until the desired conclusion appears.

## 6. E08–E09: executed decision consequences

Generate a shared candidate bank before inspecting outcomes. Prefer a small fixed bank initially over a different stochastic CEM search for every method. Use predetermined smooth random controls and geometry-based candidate construction, never future success filtering.

For each root execute **every** candidate from the actual counterfactual simulator context endpoint. Let $J_i(k)$ be physical task cost. For PushT, a suitable primary cost is $1-\text{goal coverage}$, with coverage/termination defined by the pinned environment. Position/angular goal error is secondary.

If route $m$ under appearance $c$ selects candidate $\hat k_i^m(c)$:

$$
R_i^m(c)=J_i(\hat k_i^m(c))-\min_{k\in\mathcal A_i}J_i(k).
$$

This is **regret relative to the tested candidate bank**, not the globally best policy. Keep banks where all candidates tie or fail. Use deterministic tie breaking. Report the same editor-minus-native, shifted-minus-original contrast for regret.

### Endpoint-metric oracle

Encode each **real simulator endpoint** and rank candidates with the unchanged native latent goal-distance metric.

| Result | Supported diagnostic |
| --- | --- |
| Endpoint-metric ranking is poor | Representation/goal-cost geometry can fail even with true endpoints |
| Endpoint metric is useful, native forecast ranking is poor | Forecast-path errors remain |
| Native ranking is useful, edited ranking is poor | Editing-interface error is implicated |

These are controlled comparisons, not a guaranteed nonnegative additive error decomposition.

Keep the goal and scoring rule unchanged between native and edited prediction routes. A separate cost-only perturbation control can test whether changes in action choice arise solely from goal-distance geometry. Do not count it as dynamics evidence.

**Deployment boundary:** this experiment asks a model to answer a hypothetical initial-state query. After executing actions, ordinary replanning receives real observations and may remove the edit's influence. Therefore counterfactual regret does not by itself establish sustained closed-loop success improvement. A later correction/partial-observation application needs its own information contract and full policy evaluation.

## 7. Data, seeds and statistics

### Proposed data allocation

| Split | Initial allocation | Allowed use |
| --- | --- | --- |
| Editor fitting | 2,000 root families | Fit small editors; native/future targets permitted |
| Independent scorer fitting | 600 additional families | Fit evaluation readouts; no editor selection |
| Validation | 256 families | Hyperparameters, scales, horizons, nuisance scope |
| Exploratory pilot | 128 additional families | Feasibility, variance, mechanism checks; never final confirmation |
| Locked confirmation | Initially plan 400 fresh families | Main inference only after protocol lock |

This allocation is **3,384 distinct root families in total**, plus separate smoke examples; it is not a 400-family total project. Reuse the same physical family roster across compatible checkpoints, while encoding and fitting per model. The pilot can use a smaller fitting bank before committing to the full 2,000-family allocation. E03 includes both the pilot pair generator and later full split-generation step.

All sibling renders, edit requests, physical branches and action sequences remain in the same split. Counts can be revised before confirmation to meet measured compute/power needs; record the revision. Fresh roots should not be recycled from a checkpoint's training-evaluation windows. Reproducing an original paper's split and running a new disjoint audit are different activities.

Balance translation/rotation and the prespecified contact-related regimes. If roots are intentionally sampled by stratum, state whether reported means use equal stratum weights or population prevalence. Do not make a natural-frequency claim from a balanced stress test.

### Sample-size planning

For a root-level paired contrast with pilot standard deviation $\hat\sigma_D$ and a scientifically chosen minimum effect $\delta$, a rough two-sided 5%/80%-power approximation is:

$$
n\approx\left(\frac{(1.96+0.84)\hat\sigma_D}{\delta}\right)^2.
$$

This is a planning approximation, not a replacement for cluster structure, heavy-tail checks, multiplicity adjustments or simulation-based power analysis. Choose $\delta$ from physical usefulness, not to make the desired result significant. If the required sample exceeds budget, narrow the claim or extend the timeline rather than count frames as independent data.

Power the physical-effect, native-equivalence and regret endpoints separately. The physical-error sample size does not establish power for regret, particularly when candidate costs tie. Freeze the exact decision-root subset, decision horizon and relation between the 8- and 32-candidate banks before testing. Retain physical-error measurements on the decision roots/candidates if making a claim connecting those errors to regret.

### Analysis commitments

- Pair all comparisons by root, action bank and rendering. Average repeated seeds/candidates within roots before inferential resampling.
- Use at least 5,000 root/episode-level paired bootstrap resamples for reported intervals; preserve all siblings.
- Name one primary editor contrast/horizon summary per checkpoint in the final registration. Treat other families/horizons as prespecified secondary analyses; adjust declared confirmatory families, e.g. Holm correction.
- Report raw effect sizes, task-unit errors, intervals, total roots, failures, and all registered cells. Not only p-values or best seeds.
- Use equivalence/noninferiority margins to support native stability. “No significant difference” does not establish invariance.
- Readout/editor seed variation is not world-model training-seed robustness. With one checkpoint per backbone, conclusions are checkpoint-specific.
- Do not pool incompatible architecture scales into a single leaderboard score.
- A negative result requires intervals narrow enough to exclude meaningful effects, not merely a nonsignificant test.

Freeze alternate scorers, seed rosters and subgroup analyses before confirmation. Root-bootstrap intervals are conditional on fitted editors and, if reused, the fixed nuisance roster; report fitting-seed sensitivity rather than implying those intervals cover arbitrary editor retraining. Post-result scorer/subgroup choices are exploratory and cannot repair a failed primary analysis without new confirmation.

## 8. Proposed experiment sequence

| ID | Experiment | Required conclusion or decision |
| --- | --- | --- |
| E00 | Artifact/runtime/shape checks | Exact model interface executes without silent substitutions |
| E01 | Simulator replay and render invariance | Counterfactual branches have trustworthy physical references |
| E02 | Native prediction/planning replication | Model has useful competence on the proposed support |
| E03 | Physical/history/nuisance pair construction | Intervention semantics and source splits are valid |
| E04 | Independent readout qualification | Effects can be measured on real encoded states |
| E05 | Clean editor qualification | At least one strong editor realizes the intended query cleanly |
| E06 | Sham/random/wrong-request and future-loss controls | Not generic perturbation damage or a weak-editor strawman |
| E07 | Locked transport interaction | Test H1/H2 with all roots retained |
| E08 | Candidate-bank regret | Test whether failure changes actual counterfactual decisions |
| E09 | Endpoint metric and cost-only controls | Separate editor, forecast and scoring explanations |
| E10 | Readout/seed/support sensitivity | Determine which conclusions survive alternative measurements |
| E11 | Matched 2 × 2 editor-training repair | Optional test of augmentation versus temporal supervision |
| E12 | Single-cube replication | Test cross-domain scope during the extension |
| E13 | New small-WM checkpoints/application | Optional training-seed or closed-loop extension, not a promised result |

The queue and dependencies are in [experiment_matrix.csv](experiment_matrix.csv).

## 9. Seven-day go/no-go rule

Proceed only if:

1. replay, strict loading and information-access tests pass;
2. native counterfactual prediction and measurement are adequate under the frozen candidate scope;
3. a strong editor works cleanly, not only an inverse-probe baseline;
4. the editor-specific appearance interaction appears large enough for a feasible confirmation study;
5. an independent physical action-outcome test can corroborate it;
6. the closest literature does not already establish the same result.

Illustrative practical-effect planning values: an interaction exceeding 10% of a fixed development do-nothing effect-error scale, and a regret interaction of 0.02 on a [0,1] physical cost. These are **not established universal thresholds**. Confirm/replace them using task significance before test lock, and record native-equivalence margins separately.

Stop or reframe if clean edits never work, native references fail under the proposed changes, results depend on outcome filtering, effects disappear under an independent scorer, or ordinary augmentation explains the whole method claim. A study of measurement failure may be worthwhile, but it is a different claim and needs fresh confirmation.