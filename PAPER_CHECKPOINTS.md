# Paper evidence checkpoints

## v0 — complete ICML-style working paper, 2026-11-03

**Action Fidelity Without Physical Qualification: A Replicated Diagnostic Audit
of Frozen PushT World Models**

- **Read:** [compiled paper PDF](paper/checkpoints/v0/paper.pdf).
- **Edit:** [LaTeX entry point](paper/main.tex), [main text](paper/sections/body.tex)
  and [appendix](paper/sections/appendix.tex).
- **Restore:** [frozen manuscript source ZIP](paper/checkpoints/v0/source.zip),
  [checkpoint inventory](paper/checkpoints/v0/checkpoint.json),
  [build/audit receipt](paper/checkpoints/v0/build-receipt.json), and
  [manuscript test receipt](paper/checkpoints/v0/manuscript-tests.xml).
- **Extend:** [stable H01–H04 hypothesis roadmap](paper/HYPOTHESIS_ROADMAP.md).
- **Evidence:** [experiment tracker](EXPERIMENT_TRACKER.md) and
  [completed numerical/replication results](PAPER_STUDIES_RESULTS_20261103.md).

The PDF has **6 main-text pages and 12 total pages**: anonymous two-column
ICML-style main text, impact statement, references, and a detailed appendix.
It contains 21 resolved citation keys from the 29-entry checked bibliography,
source-generated tables, primary replication plots and all-appearance/horizon
supplementary figures. Every one of 206 newly linked displayed statistics was
checked against its authenticated report source. All MLP seeds and the
inconclusive DINO primary ranking result remain visible.

The official ICML 2026 style files are unmodified; only the manuscript's notice
explicitly states **working draft, not submitted or accepted**. This is not a
claim of ICML acceptance or compliance with a future ICML/ICWM call. The PDF uses
anonymous author metadata; the internal build/test receipts and source bundle
are not certified anonymous supplementary material for submission.

## Verification

LaTeX/BibTeX compiled with no unresolved citations/references or overfull boxes;
US-letter page dimensions, anonymous author metadata and absence of Type 3 fonts
were checked. **19 manuscript contract tests passed**, zero failures/errors/skips.
An independent scientific review identified two interface/reporting details,
which were corrected: DINO's normalized 224→196 encoder resize and the distinction
between fixed MLP training and reporting target scales.

The source ZIP was independently extracted and compiled without needing model
execution or the research datasets. Its PDF reproduced the exact extracted text
of **every one of the 12 pages**. Title/results/appendix pages were visually
inspected. The source snapshot is commit
`9c67ec9324853f94d37247f743e28740b24e8d22`.

PDF SHA256: `49de2f5a863edd94b257b8949d191894f0eb787297087714450701978395e0b4`.
Source ZIP SHA256: `4cebf63a4bad5ee1ec52ead0d32ad7cbb5f5c04941807de3c6c6ebc41294d0aa`.
The immutable checkpoint inventory records the PDF, source ZIP, build receipt and
test receipt checksums. These are evidence-integrity checks, not scientific
priority, author-approval or venue-eligibility certification.

## How the paper changes next

v0 is a complete narrative of current evidence, not a slot reserved for hoped-for
positive results. Its contribution is a checkpoint-specific empirical audit,
not a new method or universal theorem. A later v1+ may add:

1. **H01:** qualified full-history versus single-image physical readout.
2. **H02:** block-specific action-effect accuracy versus pusher-dominated effects.
3. **H03:** contact/reachable-bank or closed-loop decisions versus metric mismatch.
4. **H04:** further numerical portability only if a claim requires it.

Each needs a separate fixed protocol, competing alternatives, declared budget,
fresh evaluation, practical resolution gates and preserved unsuccessful outcomes
before it becomes a result. The roadmap maps each study to stable LaTeX labels.
Do not adapt sample size/seed to chase significance, erase v0 failures, or rename
exploratory findings as independent confirmation. No new research experiment was
launched during this manuscript task.