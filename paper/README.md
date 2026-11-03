# ICML-style evidence checkpoint

A complete working manuscript, not an empty template and not a submitted or
accepted conference paper. It uses the official **ICML 2026** review style as a
current reference. It does not certify the next ICML cycle or ICWM rules.
The official margins/fonts and style files are unchanged. The notice is
explicitly overridden in the manuscript to say **working draft, not submitted
or accepted**, rather than make a false under-review statement.

## Current document

Title: **Action Fidelity Without Physical Qualification: A Replicated Diagnostic
Audit of Frozen PushT World Models**.

The complete main text has seven sections: introduction, related work, design,
estimands/controls, results, discussion/limitations and conclusion. A separate
impact statement, bibliography and five-part appendix accompany it. Results
distinguish inspected discovery from the fixed fresh-root primary family.
The build mechanically checks the 8-page main-text working target; it does not
fill pages with unrun hypotheses or unsupported claims.

## Entry points

- [LaTeX entry point](main.tex), [main narrative](sections/body.tex), and
  [appendix](sections/appendix.tex).
- [Verified bibliography](references.bib): selected primary-source references,
  not a guarantee of scientific priority or correctness of external studies.
- [Hypothesis roadmap](HYPOTHESIS_ROADMAP.md): stable extension ideas, competing
  explanations and evidence gates. These are not new results or executable
  preregistrations. Do not add hypotheses to the Results section before running
  and verifying their studies.
- [Asset generator](generate_assets.py): reads nine authenticated reports and
  generates tables/figures and a statistic-level source manifest. Uses CPU only;
  no fitting, model execution, new bootstrap draws or new research runs.
- [Build script](build.sh) and [mechanical audit](audit_manuscript.py): latexmk,
  official style integrity, bibliography keys, reference resolution, source
  values, page size, main-text limit, overflow, metadata and obvious placeholders.

## Build and provenance

Run the build script with Bash from any directory. It uses project-local Python
for the audit and requires latexmk/pdflatex plus `bibtexparser==1.4.3` and
`PyMuPDF==1.26.6`. The existing workspace environment supplies them.
The script compiles existing generated assets rather than rerunning experiments.
To regenerate assets, run the asset generator with the project environment;
it refuses existing output unless `--overwrite` is explicitly supplied.
Full authenticated experiment archives under the workspace runs directory are
needed for generation/source audit, but not for a plain LaTeX compile of the
source bundle's included assets.

Official template downloaded from
https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip.
SHA256: `8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c`.
The upstream files retain their attribution/notices and are not rewritten.
The example document is retained as a usage reference, not our manuscript text.

The 2026 author instructions prescribe at most eight main-body pages, followed
by references/appendices, US letter, anonymous review, and an impact statement
for the main track. These are a **working formatting target**, not an applicable
future submission decision. No uploaded paper-checker result or submission is
claimed. Human authors must check the actual target call, anonymity/redistribution,
financial conflicts, author details, AI-assistance disclosure and reference
updates before submitting. The bibliography has primary-source lineage from
the prior literature audit; external results are not independently reproduced.

## Paper-first development rule

Checkpoint v0 records the evidence we actually have: failed measurement/editor
qualification, exploratory feedback, replicated latent action fidelity and
specificity, inconclusive independent ranking, and bounded numerical sensitivity.
New evidence may change the abstract and conclusions in later versions, but
does not erase v0 or relabel its failures as successes. Archive each PDF/source
checkpoint with hashes and link its claim changes to verified experiments.
Do not extend a prespecified seed/sample merely to obtain significance.