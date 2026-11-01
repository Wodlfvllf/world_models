# Molab diagnostics — 2026-11-01

## Status and scientific conclusion

**All six prerequisite diagnostics completed on the hosted GPU.** Both pretrained
world models remained frozen; only the actual-image physical readouts were fitted.
No editor refit, new pilot generation, or fresh confirmation was launched.

**The main hypothesis is still not established.** LeWM has useful latent prediction
against the tested controls on generated data. DINO has useful short-horizon
prediction there, but not a clear advantage over persistence at horizon 8.
Both models predict well against controls on the recorded pretraining distribution.
Physical-readout accuracy remains insufficient for confidently interpreting small
physical edits. Recorded success is not held-out generalization or proof of
physical counterfactual competence.

## Executed protocol

- Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB reported
  VRAM (96 GB class). No additional paid machines were provisioned.
- Environment: isolated Python 3.11.14, Torch 2.10.0+cu128, torchvision 0.25.0+cu128;
  a real CUDA kernel executed during bootstrap. Inference used float32/eval.
- Paired data: original, receipt-verified pilot restored unchanged; 128 scorer,
  64 validation, and 128 pilot roots. Editors/confirmation were not rerun.
- Readout: fit on original scorer images, select on original validation images,
  then evaluate the locked readout under original/color/background appearances.
  There are 1,536 sampled images per appearance, correlated within 128 pilot roots.
- Prediction: all 128 pilot roots × 3 appearances × 2 branches × 8 candidates ×
  horizons 1/4/8 = 18,432 metric records per model. Inference compared native
  rollout, persistence, and zero-future-action rollout with original prefix actions.
- Recorded: identical 128 metadata/seed-selected episode/start windows for both
  models, original appearance only; 384 horizon records per model. No selected
  window was clipped or dropped. Six selected windows contain controls outside
  the simulator's nominal unit box; all were retained.
- Uncertainty: pointwise 95% clustered bootstrap intervals; roots or episodes are
  equally weighted. Metric records are **not** independent replicates.
- Experimental source: paired run at `275cdadd8130178112c607649b4dcf471637871b`;
  recorded continuation at `864974efee26a5d09edf24d67f9e8aa3ed4f30b3`.
  The continuation did not change the completed paired source or outputs.

| Diagnostic | LeWM elapsed | DINO elapsed |
|---|---:|---:|
| Actual-image readout fit/select/evaluate | 120.1 s | 118.7 s |
| Generated paired prediction | 251.1 s | 309.5 s |
| Recorded prediction, including full H5 rehash | 44.8 s | 44.6 s |

These are command durations, not total session time. Provisioning, data transfer,
checksum verification, recovery and debugging required additional time. Disk/CPU
stages naturally leave the GPU idle; utilization is not a scientific outcome.

## Actual-image physical readouts

Original appearance, all diagnostic context/future samples. Distances are native
simulator coordinate units in a 512-unit arena, not resized-image pixel distances.

| Model | Selected grid / ridge alpha | Scaled loss | Mean pusher distance error | Mean block distance error | Angle MAE |
|---|---|---:|---:|---:|---:|
| LeWM | CLS / 100 | 35.411 | 120.855 | 26.932 | 17.017° |
| DINO | 4×4 / 10,000 | 36.868 | 99.765 | 35.157 | 39.480° |

Both beat their training-mean baseline: selected-minus-mean scaled loss is
−47.048 [−52.209, −42.009] for LeWM and −45.592 [−50.395, −40.723] for DINO.
That is useful improvement, **not** adequate physical measurement by itself.

Paired shifted-minus-original readout loss increases:

| Model | Color only | Background only |
|---|---|---|
| LeWM | +3.610 [1.261, 6.448] | +2.920 [1.059, 5.125] |
| DINO | +1.052 [0.586, 1.539] | +1.304 [0.440, 2.147] |

These measure **readout appearance sensitivity**, not transfer failure of a
qualified physical editor. They must not be presented as confirmation of the
main editor hypothesis.

### Post-hoc observability check — descriptive, not a new gate

The retained scatter arrays contain pusher labels well outside the image canvas:
the 15-unit-radius pusher circle is wholly outside the 0–512 canvas in 134/1,536
samples (8.72%). Its center is on-canvas in 1,396/1,536 samples (90.89%). This is a
geometric bound, **not** a semantic visibility/occlusion test.

Pusher mean error for center-on-canvas samples is still 89.837 units for LeWM and
65.363 for DINO; wholly-off-canvas errors are 441.464 and 454.662. Off-canvas
labels contribute substantially, but do not explain away all readout error.
No samples were removed from any primary result, no readout was refitted, and
these post-hoc frame summaries do not carry independently sampled confidence
intervals. Future protocols need to predeclare observability handling, not
silently discard difficult roots after seeing outcomes.

## Native prediction versus persistence

Difference = native latent MSE minus persistence MSE; **negative favors native**.
Raw latent MSE cannot rank LeWM against DINO. Compare controls within each model.

| Data / model | Horizon 1 | Horizon 4 | Horizon 8 |
|---|---|---|---|
| Generated original / LeWM | −0.0639 [−0.0809, −0.0485] | −0.4660 [−0.5201, −0.4151] | −0.3464 [−0.4018, −0.2934] |
| Generated original / DINO | −0.0847 [−0.0979, −0.0711] | −0.1352 [−0.1788, −0.0933] | −0.0107 [−0.0667, 0.0423] |
| Recorded original / LeWM | −0.1534 [−0.1749, −0.1329] | −1.1682 [−1.2520, −1.0845] | −1.6071 [−1.6953, −1.5134] |
| Recorded original / DINO | −0.2639 [−0.2940, −0.2355] | −0.6957 [−0.7526, −0.6394] | −0.8703 [−0.9309, −0.8084] |

At generated horizon 8, LeWM also beats persistence under both color-only and
background-only changes: −0.2239 [−0.2877, −0.1617] and −0.2548 [−0.3114, −0.2012].
DINO's corresponding intervals cross zero: +0.0096 [−0.0454, 0.0623] and
+0.0456 [−0.0095, 0.0981].

Both models beat the zero-future-action control at horizon 8 in all three
generated appearances and at every tested recorded horizon. Recorded horizon-8
native-minus-zero differences are −1.5237 [−1.6115, −1.4289] for LeWM and
−0.8179 [−0.8817, −0.7537] for DINO.

This pattern motivates checking generated-distribution coverage and long-horizon
behavior, especially for DINO; it does not uniquely identify a cause.

## Recovery and integrity

1. The hosted notebook was an old ZIP-upload version despite a GitHub push.
   Its live deployment cell was replaced with a pinned GitHub checkout.
2. The new Molab session had no previous environment/cache/pilot. The original
   929,322,621-byte backup was transferred in 14 pieces to respect the 100 MiB
   upload limit, reconstructed, and verified against its original SHA256:
   `b7176b64ab0207b46482c3df1e411cbf03a3803671e8f2cc770a3b3986e8d476`.
3. The real recorded archive uses an 8 MiB Zstandard window. The old 128 KiB
   decoder bound was corrected to a bounded 128 MiB; oversized-window rejection
   remains tested. The 46,300,921,856-byte raw H5 was authenticated and inspected.
4. Actual recorded actions have 18,685 finite terminal labels, no interior
   nonfinite controls, and 2,904 nonterminal controls outside the unit box. The
   explicit recorded schema keeps raw controls and omits terminal rows only from
   outgoing transitions. Generated-data bounds remain strict by default.
5. Full-file action means/stds match the released normalization (within numerical
   precision). Model normalization was neither recomputed for fitting nor changed.
   All 2,336,736 rows were scanned before selecting windows from 18,685 episodes.
6. Re-downloading sources after inference exposed generated bytecode as an
   untracked upstream-cache change. No source was reset and no cache-integrity
   check was bypassed. The recorded continuation used existing hash-verified
   models directly; future notebook subprocesses disable bytecode writes.
7. Final implementation tests: **891 passed**, including real-adapter parity on
   unclipped recorded controls. The old failed attempts/logs are retained; the
   successful continuation has a new directory and explicit source revision.

Original recorded compressed SHA256:
`7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`.
Extracted H5 SHA256:
`b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`.
The raw inspection filename retains “unvalidated” from the recovery stage;
subsequent full hash/schema/normalization validation is recorded separately in
the final archive. Filename alone is not the validation status.

## Saved results and next decision

The complete 6,523,540-byte ZIP has **120 inventoried files**: six full reports,
plots, readouts, per-root/window records, both code snapshots, episode selections,
logs and recorded-data validation. SHA256:
`29d08eb94972b4768d3ed639b77021c8c03dabbd0b8a764b5237184e545ac015`.
It was verified on Molab, downloaded locally, and restored with every inventory
hash/size checked. The notebook also provides its download link and result tabs.
The original pilot data and large model/H5 caches are **not** in this results ZIP;
the separately verified original pilot backup must be retained.

Local archive: [complete results ZIP](runs/molab-20261101-retrieved/molab-diagnostics-20261101-complete.zip).
Local receipt: [SHA256 receipt](runs/molab-20261101-retrieved/molab-diagnostics-20261101-complete.zip.sha256).
Extracted summary: [structured six-experiment summary](runs/molab-20261101-retrieved/complete/summary.json).
These generated files are deliberately not committed to Git.

A separate [post-run notebook copy](runs/molab-20261101-retrieved/completed_molab_diagnostics_20261101.py)
has lint-only private-import alias cleanup and an empty cell removed; it passes
`marimo check` and was not re-executed. The immutable archive retains its original
execution snapshot. The final hosted status check reported no GPU processes;
`EXECUTE=False` was set in the live notebook. Molab was not shut down. Subsequent
live cleanup edits could not be confirmed saved (the UI offered unsaved-change
recovery), so the verified local archive, not the live editor, is the durable record.

**Next scientific gate:** resolve physical measurement accuracy and observability
under a predeclared protocol before revisiting clean editor competence. A final
appearance-transfer confirmation still requires new root families and a frozen
protocol. Do not reuse these exploratory results as a confirmatory test.