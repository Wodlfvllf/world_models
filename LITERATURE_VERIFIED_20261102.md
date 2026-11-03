# Primary-source literature and ICWM 2027 audit

**Verification date: 2026-11-02. Scope: literature, attribution, novelty, and venue rules only.**

**Verdict:** the defensible present contribution is an **exploratory,
checkpoint-specific PushT diagnostic case study applying established evaluation
ideas**. It is not the first separation of decoding, physical action effects,
and decision quality; not a new general causal-audit framework; and not a
successful demonstration of appearance-transfer failure in a qualified physical
latent editor. Citation preparation does not make the empirical paper
submission-ready.

This audit accompanies [paper/references.bib](paper/references.bib), containing
**29 verified bibliographic records with complete ordered author lists**. It does
not replace or edit [LITERATURE.md](LITERATURE.md), [PROPOSAL.md](PROPOSAL.md), or
[PAPER_DRAFT.md](PAPER_DRAFT.md). Those three documents, the two local results
reports, and the local artifact manifest were read for context. No experiment,
checkpoint download, external-study reproduction, manuscript edit, submission,
or Git commit was performed as part of this audit.

## 1. What “verified” means here

| Mark | Evidence inspected | What it does **not** establish |
|---|---|---|
| **M** | Primary arXiv metadata and abstract: title, ordered authors, identifier, submission/version dates, and displayed comments | Peer-review acceptance, correctness of results, or independent replication |
| **P** | Publisher/proceedings record and its bibliographic fields | Independent confirmation of the experiments |
| **F** | Relevant full-text methods/results/limitations, through official HTML, PDF, or author-submitted TeX | Exhaustive proof checking or execution of the study |
| **A** | Author-maintained project/repository or model-card/file listing | Artifact completeness, matching every paper result, or successful execution |

Every bibliography entry has primary-source metadata support. **M-only entries
are used for the broad claims actually stated in their abstracts, not for
uninspected implementation details.** For a verified proceedings article, its
publication year is used. Otherwise the verified preprint's first-submission
year is retained, with the inspected revision recorded separately. An arXiv DOI
is an identifier, not a peer-review credential.

### Search boundary and access limitations

- The search began with the closest entries in the existing survey, then followed
  references in D-JEPA, the latent-world interpretability perspective, PSG-JEPA,
  and the two closest diagnostic studies. A live
  [arXiv search for “world model intervention”](https://arxiv.org/search/?query=world+model+intervention&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
  was screened through its first 50 results; the page reported 2,062 results and
  included substantial unrelated material. This is **not** a systematic or
  exhaustive forward-citation search. No “first” claim follows from it.
- Newly checked relevant work absent from the original 30-paper survey includes
  PSG-JEPA, RC-aux, Temporal-Distance JEPA, A Control Theory of Predictability,
  WorldModelLens, and the October latent-world interpretability perspective.
- [UniState](https://arxiv.org/abs/2610.25414) (Jaeseok Jeong, Kyungmook Choi,
  Sohyun Chung, Youngsik Yun, Youngjung Uh; 2026-10-18) was screened at metadata/
  abstract level: it studies spatial memory, prescribed object motion and camera
  control in video generation. [Causal world models under physical parameter
  shifts](https://arxiv.org/abs/2610.29088) (Chu Lin, Yi Sun; the abstract record
  gives 2026-10-21) studies dependency masks and structured mass-dependent robot
  dynamics. Neither is used as evidence for the frozen-RGB PushT results, and
  neither is included in the selected bibliography. They should be revisited if
  the paper expands into controllable video generation or physical-parameter
  extrapolation.
- Automatic extraction failed for some official HTML/PDF pages. LeWM was checked
  through its [v3 TeX source](https://arxiv.org/src/2603.19312v3), including the
  experiment and probing-appendix sections. RolloutFaith was checked through its
  [v1 TeX source](https://arxiv.org/src/2609.36843v1), including framework,
  component-analysis, delayed-editor, and conclusion sections. Do-JEPA's
  [v1 HTML](https://arxiv.org/html/2609.37378v1) was read in the browser. The
  latent-carriers [v4 PDF](https://arxiv.org/pdf/2608.15156v4) was read directly,
  including Sections 3.1–3.4. Failed extraction is not evidence that a paper is
  unavailable.
- The attempted [DINOv2 OpenReview record](https://openreview.net/forum?id=a68SUt6zFt)
  returned a browser-verification challenge. The bibliography consequently uses
  the verified arXiv record, without asserting a journal publication from that
  inaccessible page. Terver et al.'s TMLR acceptance and Zhang's/Subramanian's
  workshop status remain **author-reported metadata**, as distinguished below.
- The remaining entries of [LITERATURE.md](LITERATURE.md) are not automatically
  certified by this audit. No external paper's quantitative claims or released
  artifact checksums were independently reproduced here.

## 2. ICWM 2027: official rules actually visible

Primary pages checked: [call for papers](https://icwm.cc/call-for-papers),
[conference home](https://icwm.cc/), and [about](https://icwm.cc/about).
The call was also inspected as a complete browser page, not just a search
snippet. These are the rules **displayed on the working draft**, not a guarantee
that the final call will be unchanged.

| Item | Verified official statement as inspected |
|---|---|
| Status | The CFP explicitly says **“Working draft.”** All dates are provisional; exact deadline times are to be confirmed. |
| Paper submission | **November 30, 2026**, provisional. No hour or timezone is specified. |
| Reviews released | **January 11, 2027**, provisional. |
| Author rebuttal | **January 11–18, 2027**, provisional. |
| Final decisions | **February 8, 2027**, provisional. |
| Conference | **Late May 2027, San Francisco Bay Area**; the inspected pages do not give exact event days. |
| Platform | A fully public, real-time review process **on OpenReview** is stated. The CFP does not expose a paper-submission invitation/portal link. |
| Author anonymity | Submissions are anonymous during review. **Every submission is de-anonymized after its final decision, whether accepted or rejected.** |
| Reviewer anonymity | Reviewer identities remain anonymous permanently. |
| Public visibility | Submissions, reviews, author responses, AC discussion and public comments are visible in real time. OpenReview account holders may comment; comments are distinguished from official reviews and do not count as reviews. |
| Phase 0 | Automated checks cover formatting, anonymity, required sections, file validity and other mechanical eligibility requirements; there is no human review in this phase. The page does not enumerate the actual formatting rules or required sections. |
| Initial reviews | Every paper surviving screening receives **two official reviews**. A score and substantive comments are requested. |
| Mixed reviews | Two additional reviewers are assigned, bringing the total to **four**. |
| Two negative reviews | The AC must explicitly confirm early rejection or rescue the paper. Rescue adds two official reviews and leads to rebuttal/public discussion; a short public justification is required. |
| Two positive reviews | Normally proceeds to rebuttal with two reviewers; the AC may require two more. Unanimity is a recommendation, not an automatic decision. |
| Rebuttal | Continuing papers enter public rebuttal/discussion; reviewers may revise assessments. Do not assume every early-rejected paper receives a rebuttal period. |

From November 2 to November 30 there are **28 elapsed calendar days**, not 70.
That is a date difference, not a precise remaining-hour countdown, because the
deadline time and timezone are unknown.

### Not specified on the inspected official pages

No verified rule was located for: main-text page/word limits; reference or
appendix limits; supplementary-file limits; a LaTeX/Word template; a list of
mandatory sections; abstract-registration deadlines; preprints or public code
during anonymous review; simultaneous submission/prior publication; code/data
release obligations; a reproducibility checklist; AI-assistance disclosure;
ethics/impact statements; submission licensing, archival proceedings or publisher;
author registration fees; or detailed conflict-of-interest rules.

**Do not substitute ICML, ICLR, or NeurIPS conventions for these missing rules.**
In particular, an AoE cutoff, an eight/nine-page limit, a mandatory code release,
and permission to expose identifying repository links are not established.
Recheck the CFP and the eventual OpenReview invitation before formatting or
uploading. The announced public-review/de-anonymization policy also means that
rejection should not be assumed to keep the submission private.

## 3. Verified bibliographic ledger

Author names below are abbreviated for readability; the companion bibliography
retains **all** authors in source order. Titles are linked to the inspected
primary records. Dates identify the version actually used; publication claims
are separated from preprint dates.

### Models, environment lineage, and backbone

| Citation key | Verified title / authors | Record, dates, and evidence |
|---|---|---|
| `zhou2025dinowm` | [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://proceedings.mlr.press/v267/zhou25t.html) — Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto | **ICML 2025**, PMLR 267:79115–79135 (**P**). [arXiv:2411.04983](https://arxiv.org/abs/2411.04983): first 2024-11-07, v2 2025-02-01 (**M/F**). |
| `maes2026lewm` | [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312) — Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero | **2026**, arXiv:2603.19312; first 2026-03-13, v3 2026-06-03. **M/F/A**; no independently verified conference attribution. |
| `oberweger2026pointlewm` | [Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations and Goals](https://arxiv.org/abs/2608.29434) — Fabio F. Oberweger, Michael Schwingshackl, Markus Murschitz | **2026**, arXiv:2608.29434; first 2026-08-29, v2 2026-09-28. **M/F/A**. |
| `maes2026stableworldmodel` | [stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation](https://arxiv.org/abs/2605.21800) — Lucas Maes et al. (12 authors) | **2026**, arXiv:2605.21800v1, submitted 2026-05-20. **M**. Cite the platform, not just the model using it. |
| `oquab2023dinov2` | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — Maxime Oquab et al. (26 authors) | Verified **2023 preprint**, arXiv:2304.07193; first 2023-04-14, inspected v2 2024-02-02. **M**. Journal record not verified in this pass. |
| `chi2023diffusionpolicy` | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://www.roboticsproceedings.org/rss19/p026.html) — Cheng Chi et al. (seven RSS authors) | **RSS 2023**, DOI **10.15607/RSS.2023.XIX.026** (**P**). Matching [arXiv:2303.04137v4](https://arxiv.org/abs/2303.04137v4), revised 2023-06-01 (**M**); [author project](https://diffusion-policy.cs.columbia.edu/) (**A**). |

### Closest diagnostic and latent-intervention precedents

| Citation key | Verified title / authors | Record, dates, and evidence |
|---|---|---|
| `joseph2026interpretingphysics` | [Interpreting Physics in Video World Models](https://proceedings.mlr.press/v306/joseph26a.html) — Sonia Joseph et al. (eight authors) | **ICML 2026**, PMLR 306:54687–54724 (**P**); [arXiv:2602.07050v1](https://arxiv.org/abs/2602.07050), 2026-02-04 (**M**). |
| `zhang2026probingworldmodels` | [What Do World Models Learn in RL? Probing Latent Representations in Learned Environment Simulators](https://arxiv.org/abs/2603.21546) — Xinyu Zhang | **2026**, arXiv:2603.21546v1, 2026-03-23. **M**. The arXiv journal-reference field lists the ICLR 2026 second Workshop on World Models; workshop proceedings were not independently checked. |
| `liu2026latentcarriers` | [Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models](https://arxiv.org/abs/2608.15156) — Yang Liu, Yuming Chen | **2026**, arXiv:2608.15156; first 2026-08-15, v4 2026-09-07. **M/F/A**. |
| `yao2026rolloutfaith` | [RolloutFaith: Auditing Persistent Internal Interventions in Visual World Model](https://arxiv.org/abs/2609.36843) — Junchi Yao, Ziyi Wang, Youling Huang, Lijie Hu | **2026**, arXiv:2609.36843v1, 2026-09-29. **M/F**. The title is singular **Model**, not **Models**. |
| `vakalis2026interventiongap` | [The Intervention Gap in Latent World Models](https://arxiv.org/abs/2608.29998) — Donna Vakalis | **2026**, arXiv:2608.29998v1, 2026-08-30. **M/F**. |
| `subramanian2026counterfactual` | [Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models](https://arxiv.org/abs/2610.02860) — Arjun Subramanian | **2026**, arXiv:2610.02860v1, 2026-10-02. **M/F**. Comments claim a NeurIPS 2026 Physical World AI workshop publication; this is not independently verified and is not a main-track attribution. |
| `huang2026geocosavi` | [GeoCo-SAVi: Geometry-Consistent Slot Attention for Explicitly Editable Object Representations](https://arxiv.org/abs/2609.06628) — Haoxiang Huang et al. (nine authors) | **2026**, arXiv:2609.06628; first 2026-09-06, v3 2026-09-26. **M/F**. |
| `resani2026dojepa` | [Do-JEPA: From Masking to Intervention in Latent World Models](https://arxiv.org/abs/2609.37378) — Hossein Resani, Javen Qinfeng Shi | **2026**, arXiv:2609.37378v1, 2026-09-29. **M/F**. |
| `liu2026djepa` | [D-JEPA: A Decision-Aligned Latent World Model](https://arxiv.org/abs/2609.24749) — Shuaijun Liu et al. (eight authors) | **2026**, arXiv:2609.24749v1, 2026-09-21. **M/F**. Do not confuse D-JEPA, Do-JEPA, or Delta-JEPA. |
| `hong2026steering` | [Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal Control](https://arxiv.org/abs/2607.14943) — Jihoon Hong, Julian Skifstad, Qiyue Dai, Alice Chan, Glen Chou | **2026**, arXiv:2607.14943v1, 2026-07-16. **M**. |
| `zhang2026deltajepa` | [Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding](https://arxiv.org/abs/2606.31232) — Zhenghao Zhang et al. (14 authors) | **2026**, arXiv:2606.31232v1, 2026-06-30. **M**. |
| `yan2026psgjepa` | [Is Forward Prediction Enough? Physical State Grounding for JEPA World Models](https://arxiv.org/abs/2608.06799) — Haodong Yan et al. (15 authors) | **2026**, arXiv:2608.06799v1, 2026-08-07. **M/F**. A particularly important omission from the older survey. |

### Foundations and additional directly relevant work

| Citation key | Verified title / authors | Record, dates, and evidence |
|---|---|---|
| `grant2025divergent` | [Addressing divergent representations from causal interventions on neural networks](https://arxiv.org/abs/2511.04638) — Satchel Grant, Simon Jerome Han, Alexa R. Tartaglini, Christopher Potts | **2025 preprint**, arXiv:2511.04638; first 2025-11-06, inspected v5 2026-04-22. **M**. |
| `lippe2022citris` | [CITRIS: Causal Identifiability from Temporal Intervened Sequences](https://proceedings.mlr.press/v162/lippe22a.html) — Phillip Lippe et al. (six authors) | **ICML 2022**, PMLR 162:13557–13603 (**P**); [arXiv:2202.03169](https://arxiv.org/abs/2202.03169), first 2022-02-07, v3 2022-06-15 (**M**). |
| `elazar2021amnesic` | [Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals](https://aclanthology.org/2021.tacl-1.10/) — Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg | **TACL 2021**, 9:160–175, DOI **10.1162/tacl_a_00359** (**P**); [arXiv:2006.00995](https://arxiv.org/abs/2006.00995), first 2020-06-01, v3 2021-02-19 (**M**). |
| `geiger2021causalabstractions` | [Causal Abstractions of Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) — Atticus Geiger, Hanson Lu, Thomas Icard, Christopher Potts | **NeurIPS 2021**, volume 34 (**P**); [arXiv:2106.02997](https://arxiv.org/abs/2106.02997), first 2021-06-06, v2 2021-10-27 (**M**). Unverified page numbers are omitted. |
| `lambert2020objectivemismatch` | [Objective Mismatch in Model-based Reinforcement Learning](https://proceedings.mlr.press/v120/lambert20a.html) — Nathan Lambert, Brandon Amos, Omry Yadan, Roberto Calandra | **L4DC 2020**, PMLR 120:761–770. **P**. No unverified arXiv identifier is supplied. |
| `terver2025jepawms` | [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/abs/2512.24497) — Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, Yann LeCun | **2025 preprint**, arXiv:2512.24497; first 2025-12-30, v4 2026-09-02. **M**. TMLR acceptance is stated in arXiv comments, not independently established by a journal record here. |
| `wang2026worldmodelinterpretability` | [World Model Interpretability: From Decoding Representations to Understanding Latent Worlds](https://arxiv.org/abs/2610.25296) — Jize Wang, Yongcheng Jing, Xinyi Le, Cailian Chen, Xinping Guan, Dacheng Tao | **2026**, arXiv:2610.25296v1, 2026-10-18. **M/F**. A perspective/research agenda, not an experimental validation of this project. |
| `you2026controltheory` | [A Control Theory of Predictability in Latent World Models](https://arxiv.org/abs/2607.10362) — Hanzhe You et al. (nine authors) | **2026**, arXiv:2607.10362v1, 2026-07-11. **M**. Theorems not independently checked here. |
| `bai2026tdjepa` | [Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control](https://arxiv.org/abs/2607.25337) — Jiaxin Bai, Jiaxuan Xiong | **2026**, arXiv:2607.25337; first 2026-07-28, v2 2026-07-29. **M**. The primary arXiv category is actually cs.CL; it has not been silently changed to cs.RO. |
| `challagundla2026worldmodellens` | [One Lens, Many Worlds : A Capability-Typed Interface for World-Model Interpretability](https://arxiv.org/abs/2606.09936) — Bhavith Chandra Challagundla et al. (ten authors) | **2026**, arXiv:2606.09936v1, 2026-06-07. **M/F**. Metadata punctuation retained. |
| `li2026rcaux` | [Predictive but Not Plannable: RC-aux for Latent World Models](https://arxiv.org/abs/2605.07278) — Wenyuan Li, Guang Li, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama | **2026**, arXiv:2605.07278v1, 2026-05-08. **M**. |

### Bibliographic corrections and version distinctions

1. The existing survey pluralizes the final word of RolloutFaith's title. The
   verified title and new BibTeX use **Visual World Model**. The old survey was
   left untouched as requested.
2. DINO-WM's **2024 preprint** and **2025 ICML publication** are the same work,
   not two independent precedents. The proceedings entry uses 2025. Its author
   surname capitalization follows arXiv/the author project (**LeCun**), rather
   than PMLR's **Lecun**.
3. Joseph et al.'s proceedings record uses **Blake Aaron Richards** and
   **Michael Rabbat**, whereas arXiv displays **Blake Richards** and **Mike
   Rabbat**. CITRIS uses **Stratis Gavves** in PMLR and **Efstratios Gavves** on
   arXiv. Publisher names are retained for these proceedings entries.
4. The RSS Diffusion Policy paper has **seven authors**; its later IJRR/v5
   version has a reordered **eight-author** list including Russ Tedrake. The
   RSS entry is tied to v4, and uses the publisher's **Benjamin CM Burchfiel**.
5. Grant et al. and Terver et al. have 2025 first-submission dates but inspected
   2026 revisions. Version notes preserve that distinction. Preprint status is
   not shorthand for “no prior art.” Conversely, a venue template, a keyword
   saying ICML, or an author-written acceptance comment is not an independently
   verified proceedings record.

## 4. Model attribution and the published-probe comparison

### Which DINO-WM is actually being audited?

The [Point-LeWM authors' release](https://github.com/fafraob/point-lewm) explicitly
states that its **RGB DINO-WM baselines** were trained through an adapter using
the unmodified official DINO-WM code, frozen DINOv2 ViT-S/14, **no
proprioception**, and no decoder. It asks users of those checkpoints to cite
both the original DINO-WM paper and the Point-LeWM paper. The
[PushT checkpoint folder](https://huggingface.co/fafraob/point-cloud-pusht/tree/main/dino-wm)
is an RGB baseline inside a repository whose name contains “point-cloud.” It is
**not** the Point-LeWM point-cloud predictor and **not** an original Zhou et al.
training run.

Recommended designation: **“the RGB/no-proprioception DINO-WM checkpoint trained
and released by Oberweger et al., using Zhou et al.'s model code.”** In the
implementation, “no proprioception” does not mean all auxiliary channels can be
deleted or reset during native rollout. Preserve the release's dummy-input and
predicted-channel contract; an auxiliary reset is a separately labeled diagnostic.

The [original DINO-WM full text](https://arxiv.org/html/2411.04983v2), Section
3.1.2, describes optional proprioceptive conditioning. Its planning results
cannot simply be assigned to this later, separately trained RGB checkpoint.
LeWM's [official repository](https://github.com/lucas-maes/le-wm) and
[PushT model card](https://huggingface.co/quentinll/lewm-pusht) identify the
other audited release. Two such releases are case studies, not a controlled
ablation of end-to-end versus frozen-feature learning.

### Immutable identities already recorded locally

The following are **reported by [cache/artifacts.json](cache/artifacts.json)**,
not newly rehashed by this literature audit:

| Artifact | Recorded revision | Recorded weight SHA256 |
|---|---|---|
| LeWM PushT | `22b330c28c27ead4bfd1888615af1340e3fe9052` | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| Point-LeWM-author RGB DINO-WM PushT | `664278a4c0b4c5f607d0993744986b2d10b4168e` | `14489cfdf4132a126fd1cb3cac39370a1fb2141503cd3e41aeb3c2dead271dd0` |
| DINOv2 ViT-S/14 | Source audit pin `85a24602099d397264d5b30461ad7f3bfd726ca1` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

The manifest explicitly calls the DINOv2 source revision an **audit pin**: the
original release's Hub-main revision was not recorded. Do not relabel it an
authenticated original-training source revision. The manuscript/release owner
must tie each reported experiment to its own archived manifest, rather than
assuming a local cache manifest alone proves the provenance of every run.

The Point-LeWM release notes also distinguish October **dataset v2 sensor-view
images** from the original scene-camera RGB used by the released image baselines;
its current image-evaluation configurations use the original HDF5 images. Do not
silently evaluate those checkpoints on the newer sensor-view pixel column. The
release declares code MIT and released data/checkpoints CC BY 4.0; the LeWM model
card declares MIT. These are inspected license declarations, **not** a complete
third-party redistribution audit. The Point-LeWM manuscript's CC BY-SA license
is a different asset license from its checkpoint license.

### The existing results do not directly refute the published probe tables

- LeWM's experiment section already includes **linear and nonlinear physical
  probes on PushT** and a violation-of-expectation contrast between teleportation
  and color changes. Its probing appendix explicitly reports **MSE and Pearson
  correlation $r$**, not $R^2$. The TwoRoom discussion already notes strong
  probing alongside weaker planning. Neither probing physics nor a probe/planner
  dissociation is new here.
- [Point-LeWM v2](https://arxiv.org/html/2608.29434v2), Section 4.2 and Appendix
  D.1, probes **Point-LeWM and Point-Delta-JEPA**, not its RGB DINO-WM baseline.
  It uses the post-projector planning latent, episode-disjoint probe splits,
  sensor-frame targets, heading-vector angles, and MSE/$R^2$. Those point-cloud
  results are not a directly matched replication target for the present RGB
  readouts.
- The present audit reports native 512-arena distance errors, angular MAE, and
  paired-effect errors on generated roots with off-canvas/aliased images.
  Published correlations and MSEs cannot be converted into those quantities by
  renaming a column. Match checkpoint, representation surface, observation/data
  distribution, target normalization, readout capacity, sample budget and split
  before claiming a failed reproduction. The inspected LeWM probing passages
  alone do not establish all of those matches.

## 5. Closest-precedent comparison: what is already occupied?

All statements in the middle column describe **what the cited source reports**,
not independently reproduced facts. Section references below are within the
linked external papers.

| Precedent and primary-text location | Existing contribution most relevant here | Narrow distinction and consequence for this manuscript |
|---|---|---|
| [Intervention Gap](https://arxiv.org/html/2608.29998v1), Sections 2.2, 6.4 and Appendix F | Capture-first qualification, real-endpoint effect resolution, native open-loop action contrasts, persistence/zero-effect and endpoint-readout controls, grouped uncertainty; it also reports audits that terminate at failed capture/effect-resolution gates. | This is the closest methodological precedent to the **current diagnostic paper**, not only to the abandoned editor hypothesis. Neither separating measurement from propagation nor reporting a failed gate is new. PushT, RGB release provenance, and the joint empirical outcomes are a bounded extension, not a new protocol principle. |
| [Observation Bottlenecks](https://arxiv.org/html/2610.02860v1), Sections 3–6 | Exact simulator forks traced through physical state, quantized observations, target features and predictions; upstream identifiability controls; caution that magnitude response is not correct direction or control. | Its observation is a density/velocity **physical raster**, not camera RGB. This project tests released RGB checkpoints and candidate-vector matching/decision routes. Those are different tests, but Subramanian explicitly proposes direction/ranking follow-ups; adding them is not, by itself, a strong originality claim. |
| [PSG-JEPA](https://arxiv.org/html/2608.06799v1), Sections 3 and 4.1–4.4 | Frozen single-latent and pairwise probes expose robot-state/state-change limitations in forward-trained LeWM; training-only state and transition grounding objectives modify representation learning; downstream planning/policy evaluations follow. | Physical measurement failure in a JEPA is already directly studied. Here the two released predictors remain frozen and small-edit resolution is tested on generated PushT roots. No grounding remedy, information-absence theorem, or improvement over PSG-JEPA has been demonstrated. |
| [D-JEPA](https://arxiv.org/html/2609.24749v1), Section 3, Appendices A.5 and C.1 | Executed candidate outcomes; predicted versus **realized encoded-endpoint** latent-cost correlations; decision-local inversions; learned relational ranking/adaptation; appearance studies. | Do **not** claim it lacks real-endpoint comparisons. The present F4 measures physical regret of the separate endpoint-selected and forecast-selected candidates, without a learned ranker. A signed reversal in one frozen checkpoint/bank is an empirical nuance, not discovery of prediction/control mismatch. Its selected/boundary-focused populations and this project's full, low-coverage bank also differ. |
| [Latent Carriers](https://arxiv.org/pdf/2608.15156v4), Sections 3.1–3.4 | Native counterfactual qualification, training-only low-rank carrier fitting, one-shot edits followed by autonomous rollouts, specificity controls and fresh-checkpoint checks. | The model receives **noisy numerical position/velocity observations**, and the addressable map receives factual physical state plus the requested edit; the main family has counterfactual-supported training. A frozen RGB setting is different, but the present study has not passed the gates needed to establish an RGB physical-editor result. Native references and low-rank editing are not new. |
| [RolloutFaith source](https://arxiv.org/src/2609.36843v1), framework, component-analysis and improvement sections | Reference activation patching versus fitted editors; immediate/later semantic gains; restoration of visual/recurrent/cache components; Delayed LoReFT trained through four frozen future transitions. | Its main task is correction toward real-observation references and persistence in DIAMOND/DreamerV3/STORM. A simulator-defined named initial-state intervention would be a different target, but no such qualified result exists here. Persistence, memory restoration, and future-supervised editing must be credited as precedents. |
| [Do-JEPA](https://arxiv.org/html/2609.37378v1), Sections 3.2–3.3 and 4.3 | Exact same-state action/reference forks, context twins, information-matched controls, effect supervision; released LeWM PushT fine-tuning; latent effect RMSE/cosine and physical-readout evaluations. | F3 is a frozen-model diagnostic, not an action-effect learning objective. Direction-sensitive evaluation and rendering twins are already present here. Differences between latent improvement and responsive physical holdout performance are also discussed; do not claim those distinctions are unstudied. |
| [GeoCo-SAVi](https://arxiv.org/html/2609.06628v3), Sections 3.4, 4.3–4.4, Appendices C.9 and F.6 | Geometry commands whose meanings remain calibrated under appearance transplantation; image/video editing and explicit non-target preservation measurements. | Its evaluated video edits operate through factual per-frame slot streams and decoding, rather than a single physical edit followed by autonomous action-conditioned simulator-matched dynamics. This bounds overlap, but **readout appearance sensitivity alone** is not evidence that the present study has achieved the stronger experiment. |
| [Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/), [Causal Abstractions](https://proceedings.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html), [CITRIS](https://proceedings.mlr.press/v162/lippe22a.html), [divergent interventions](https://arxiv.org/abs/2511.04638) | Encoded information versus behavioral use; interchange intervention correspondence; intervention-supervised identifiability under assumptions; natural-reference matching and off-distribution neural interventions. | “Decodable does not mean used,” commutation/interchange reasoning, and off-manifold cautions are established. Failed bounded probes do not prove absent information; simulator intervention labels do not establish causal identifiability of neural coordinates. |
| [Objective Mismatch](https://proceedings.mlr.press/v120/lambert20a.html), [Control Theory](https://arxiv.org/abs/2607.10362), [RC-aux](https://arxiv.org/abs/2605.07278), [Temporal-Distance JEPA](https://arxiv.org/abs/2607.25337) | Prediction objectives need not optimize control; planner-reachable cost fidelity, finite-budget reachability, and temporal progress motivate distinct model/cost objectives. | F4 cannot be advertised as the first evidence that low prediction error or Euclidean latent distance fails to rank useful actions. No new planning cost, control guarantee, or learned correction has been established here. |
| [WorldModelLens](https://arxiv.org/html/2606.09936v1), Sections 3–5 and 7; [interpretability perspective](https://arxiv.org/html/2610.25296v1), Sections 2–3 and 6–7 | Shared hook/cache/rollout interfaces, and an explicit agenda separating grounding, structure, causal verification and action. WorldModelLens itself limits its deep empirical demonstration to I-JEPA and labels broader studies/measurement details unfinished. | A tested release can be useful infrastructure without being the first world-model interpretability interface. The perspective motivates the work, but does not experimentally validate this study or establish that close empirical precedents do not exist. |

Joseph et al. and Zhang additionally occupy physical probing and probe-derived
steering, while Hong et al. occupy robustness-oriented activation steering and
feedback control. Terver et al. already study architecture, objective and planner
choices. These are relevant context, not claims that their exact systems were
reproduced or that they are drop-in PushT baselines.

## 6. Narrowly defensible contribution and interpretation

### Suggested contribution statement, conditional on the local reports

> We apply established measurement, intervention, and decision controls to two
> precisely identified frozen RGB PushT releases. On the tested generated pilot,
> the fitted physical readouts remain too coarse for the intended small-edit
> interpretation, while candidate-conditioned latent effects outperform zero
> effect and wrong assignments. A separate finite-bank comparison finds that
> selecting through actual encoded endpoints need not improve physical outcomes
> over selecting through model forecasts. We report these outcomes jointly,
> preserve the failed editor-qualification result, and release the diagnostic
> implementation and provenance rather than infer a general absence of physics.

The empirical basis is [MOLAB_RESULTS_20261101.md](MOLAB_RESULTS_20261101.md) and
[FOLLOWUP_RESULTS_20261102.md](FOLLOWUP_RESULTS_20261102.md), **read as project
reports, not independently rerun in this audit**. Until public packaging is
finished, change “release” to “provide” or a future-tense commitment as appropriate;
do not imply that local archives are already a public, anonymous artifact.

The candidate novelty is the **specific joint result and auditable implementation
on these releases/supports**, particularly the contrast between unqualified
physical measurement, useful latent action specificity, and the DINO endpoint/
forecast ranking reversal. Even that is an incremental empirical contribution,
not a verified priority claim or an acceptance prediction. A combination of
previously published controls does not automatically constitute a new method.

The original appearance-transport hypothesis would require independently
qualified readouts, clean editing and native counterfactual prediction, followed
by the prespecified appearance interaction and executed consequences. The current
reports do not establish that conjunction. It must not be reconstructed from a
readout-shift table or from a new title for the failed pilot.

### Name the three privileged controls precisely

1. **Native counterfactual initialization** supplies an actual counterfactual
   observation/history. It tests a privileged route through the model, not
   perfect physical forecasting or a deployable editor.
2. **Actual encoded-endpoint ranking** supplies future observations but still
   uses the model's potentially misaligned latent goal metric. It is not an
   upper bound on task success.
3. **Best executed candidate in the bank** uses true simulator task outcomes.
   It is a finite-bank outcome oracle, not the optimal policy over all actions.

For clarity, let $c_i$ be executed simulator cost, $o_i$ the actual endpoint,
$\hat z_i$ the forecast, $E$ the encoder, and $d$ the unchanged latent goal cost.
Then

$$
i_E=\arg\min_i d(E(o_i),E(g)),\qquad
i_P=\arg\min_i d(\hat z_i,E(g)),\qquad
R(i)=c_i-\min_j c_j.
$$

Consequently $R(i_P)-R(i_E)=c_{i_P}-c_{i_E}$ can have either sign. This is
clarifying notation, **not a new theorem**. Forecast bias could interact with
metric bias, but the observed sign alone does not identify that mechanism or
prove that inaccurate prediction is generally beneficial. D-JEPA already makes
predicted/realized latent costs and executed outcomes distinct quantities.

## 7. Proposed related-work prose with checked citation keys

The following is original draft prose for the manuscript owner to integrate;
the manuscript itself was not edited. Citation commands refer to
[paper/references.bib](paper/references.bib). Retain only passages needed by the
eventual paper scope and venue length limit.

### Frozen visual world models and release provenance

Action-conditioned latent world models support goal-directed planning without
requiring pixel reconstruction at inference. DINO-WM predicts frozen DINOv2 patch
features, whereas LeWorldModel learns a compact visual representation and
predictor end to end \citep{zhou2025dinowm,oquab2023dinov2,maes2026lewm}.
Architecture, objective and planner choices are themselves established objects
of empirical study \citep{terver2025jepawms}. Our experiments use the released
LeWM PushT model and the separate RGB/no-proprioception DINO-WM checkpoint trained
and released by the Point-LeWM authors; the latter is neither their point-cloud
model nor an original DINO-WM training run \citep{oberweger2026pointlewm}.
The environment lineage includes PushT and the stable-worldmodel platform
\citep{chi2023diffusionpolicy,maes2026stableworldmodel}. These releases are
checkpoint case studies rather than a controlled comparison of learning
paradigms.

### Measurement, physical grounding, and intervention fidelity

Probing establishes recoverability under a chosen readout, not necessarily
behavioral use \citep{elazar2021amnesic}. Physical probing and steering have
already been studied in video encoders and learned environment simulators
\citep{joseph2026interpretingphysics,zhang2026probingworldmodels}; LeWM itself
reports physical probes, and PSG-JEPA diagnoses limitations of state and
state-change recovery before adding training-time grounding objectives
\citep{maes2026lewm,yan2026psgjepa}. Most directly, the Intervention Gap audit
requires current-state capture and real-endpoint effect resolution before
interpreting native propagation, while simulator-fork analysis has separated
physical effects from observation and representation bottlenecks
\citep{vakalis2026interventiongap,subramanian2026counterfactual}. We adopt this
qualification logic rather than propose it. Our bounded negative readout results
therefore leave physical interpretation unqualified; they do not establish that
physical information is absent from the checkpoints.

### Internal editing and appearance-dependent semantics

Interchange interventions and intervention-supervised representation learning
provide established foundations for testing correspondences between neural and
high-level variables \citep{geiger2021causalabstractions,lippe2022citris}.
Recent world-model studies already use native counterfactual references and
low-rank patches to launch autonomous futures, or evaluate whether corrections
persist after an internal intervention ends
\citep{liu2026latentcarriers,yao2026rolloutfaith}. RolloutFaith also trains an
editor through frozen future transitions. GeoCo-SAVi calibrates geometry-command
semantics across appearances, and robustness steering uses activation geometry
and feedback control to alter world-action-model behavior
\citep{huang2026geocosavi,hong2026steering}. These precedents, together with
known risks from divergent neural interventions \citep{grant2025divergent},
preclude treating a probe change or persistent perturbation as a certified
physical edit. Our original appearance-transfer experiment remains unqualified
and is not presented as a successful extension of these editing results.

### Action effects and decision quality

Distinguishing action consequences is also an established learning objective:
Delta-JEPA decodes actions from latent differences, and Do-JEPA supervises matched
action effects using simulator forks and, where used, context twins
\citep{zhang2026deltajepa,resani2026dojepa}. Our action-effect analysis is instead
a frozen-checkpoint diagnostic. Its latent specificity does not certify
block-coordinate fidelity. More broadly, prediction/control objective mismatch
predates current JEPA models \citep{lambert2020objectivemismatch}; recent work
studies planner-reachable cost fidelity, finite-budget reachability, and temporal
planning costs \citep{you2026controltheory,li2026rcaux,bai2026tdjepa}. D-JEPA
directly relates candidate predictions, realized latent costs and executed
outcomes, then learns decision alignment \citep{liu2026djepa}. Our narrower
comparison asks how the unchanged metric selects from forecasts versus actual
encoded endpoints in the same finite bank. It does not introduce decision
alignment or establish closed-loop control gains.

### Relation to the interpretability agenda and tooling

The separation of representation grounding, dynamical structure, causal
verification and action is explicitly advocated in recent latent-world
interpretability work \citep{wang2026worldmodelinterpretability}. Shared
world-model instrumentation is likewise an existing goal of WorldModelLens
\citep{challagundla2026worldmodellens}. Our contribution is a bounded empirical
audit with preserved negative outcomes and explicit checkpoint/data provenance,
not the first interpretability interface or a realization of every stage of that
agenda.

## 8. Remaining blockers and handoff

### Scientific blockers, not resolved by literature work

1. **No confirmed physical-editor claim.** The original readout/editor gates did
   not establish qualification. Poor readout scores do not prove harmful physical
   edits, and readout appearance sensitivity does not prove editor-specific
   transport failure. A revived editor claim requires new qualified evidence.
2. **Exploratory reuse.** Follow-ups were designed after viewing pilot outcomes.
   Their fixed protocol is useful but is not untouched confirmation. A stronger
   headline needs a frozen, fresh-root replication, or must remain explicitly an
   exploratory case study. Pointwise intervals are not simultaneous confirmation.
3. **Weak decision-bank support.** The follow-up report states 127/256 correlated
   root–branch banks have no positive goal coverage, mean best available coverage
   is 11.28%, and neither forecast ranking clearly beats hold. Retain all roots,
   absolute coverage, ties and bank quality. Any redesigned bank changes the
   estimand and needs its own prospective protocol; do not filter the current
   bank into an apparent control-success result.
4. **Measurement and published-protocol comparability.** Bounded ridge/MLP failure,
   single-image hash collisions and off-canvas observations do not establish
   full-history information absence. Match the original probe protocols before
   making a reproduction claim. A new measurement method must be selected without
   tuning against the already-inspected pilot's desired outcome.
5. **Scope and numerical reproducibility.** One environment and two unmatched
   released checkpoints do not support architecture-wide inference. Recorded
   windows overlap the pretraining distribution. The documented per-record
   numerical drift is not experimentally localized to batching/backend behavior.
   Preserve these limitations; artifact integrity and passing software tests do
   not resolve them.

### Submission and manuscript blockers

- **Final venue instructions are missing.** Obtain the actual template, limits,
  required sections, deadline time/timezone and submission invitation when
  published. The present CFP is insufficient to certify format compliance.
- **Integrate the citations and distinctions.** The manuscript still needs a
  related-work section, consistent checkpoint labels, and these checked
  references. This audit intentionally made no changes to it.
- **Prepare the review artifact for the announced public process.** Check author
  names, repository ownership links, PDF metadata, absolute host paths and logs
  against the final anonymity rules. Do not assume a public repository is allowed
  merely because reviews are public. Preserve licenses/attributions if assets are
  redistributed; local verified archives are not automatically a public release.
- **Repeat the targeted search before submission.** In particular, recheck the
  close 2026 preprints and their later versions. The current search is sufficient
  to reject broad novelty claims, not to prove priority for the remaining narrow
  empirical conjunction. Human authors should inspect the cited methods and
  approve the final comparisons and any AI-assistance statement required by the
  eventual rules.

**Completed by this audit:** primary-source attribution and version checks;
29-entry verified-source bibliography; closest-precedent distinctions; narrowly
qualified contribution language; citation-keyed related-work prose; and a live
official-venue rules/missing-rules ledger. **Not completed:** fresh experiments,
public artifact release, manuscript integration, venue formatting, or submission.

**Local document checks:** all 29 citation keys in the proposed prose resolve to
29 unique bibliography records, and all nine local document links resolve. No
editor diagnostics were reported for either deliverable. This is citation/link
validation, not a venue-formatted LaTeX/BibTeX build or scientific replication.
