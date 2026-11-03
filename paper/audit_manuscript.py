"""Audit compiled working paper, citations, linked statistics and asset hashes.

No model execution, fitting, new experiments or bibliographic priority claims.
Does not certify venue acceptance, the next cycle's rules or author anonymity
of every future supplementary release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import bibtexparser
import pymupdf

STYLE_SHA = "8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pointer(value, path):
    for part in path.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def citations(tex_files, bib):
    database = bibtexparser.loads(bib.read_text())
    ids = [entry["ID"] for entry in database.entries]
    require(len(ids) == len(set(ids)), "Duplicate bibliography keys")
    cited = set()
    for path in tex_files:
        for match in re.finditer(r"\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", path.read_text()):
            cited.update(x.strip() for x in match.group(1).split(","))
    require(cited <= set(ids), f"Unresolved citations: {sorted(cited - set(ids))}")
    return sorted(cited), len(ids)


def audit(root):
    paper = root / "paper"
    pdf = paper / "build/main.pdf"
    log = (paper / "build/main.log").read_text(errors="replace")
    aux = (paper / "build/main.aux").read_text()
    require(pdf.is_file(), "No compiled PDF")
    require(not re.search(r"(?:^! |Undefined control sequence|LaTeX Error|Emergency stop|Fatal error)", log, re.M), "LaTeX compilation error")
    require(not re.search(r"(?:Citation .* undefined|Reference .* undefined|There were undefined|Rerun to get cross-references right)", log), "Unresolved cross-references/citations")
    overfull = re.findall(r"Overfull \\[hv]box.*", log)
    require(not overfull, "Overfull typesetting boxes: " + str(overfull))
    require(sha(paper / "vendor/icml2026.zip") == STYLE_SHA, "Official style bundle checksum changed")
    with zipfile.ZipFile(paper / "vendor/icml2026.zip") as z:
        for name in ("icml2026.sty", "icml2026.bst"):
            require((paper / "vendor" / name).read_bytes() == z.read(name), "Official style was edited")
    tex_files = [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]
    cited, bibliography_count = citations(tex_files, paper / "references.bib")
    main_end = re.search(r"\\newlabel\{maintextend\}\{\{[^}]*\}\{(\d+)\}", aux)
    require(main_end is not None, "No main-text end page marker")
    main_pages = int(main_end.group(1))
    require(main_pages <= 8, f"Main text exceeds ICML 2026 review working target: {main_pages}")
    doc = pymupdf.open(pdf)
    require(all(abs(page.rect.width - 612) < .1 and abs(page.rect.height - 792) < .1 for page in doc), "PDF must be US letter")
    text = "\n".join(page.get_text() for page in doc)
    require("Inconclusive" in text and "128" in text and "99%" in text, "Missing central result/uncertainty")
    require("not submitted or accepted" in text, "Do not imply conference submission")
    require(not any(s in text.lower() for s in ("/users/", "/marimo/", "shashank", "wodlfvllf", "sb.molab.run", "firstname", "todo", "xxx", "authorerr")), "Placeholder/error or identifying workspace material in PDF")
    require(doc.metadata.get("author", "") in ("", "Anonymous Authors"), "PDF author metadata must be anonymous in this working version")
    type3 = [font for page in doc for font in page.get_fonts(full=True) if "Type3" in str(font)]
    require(not type3, "Type 3 fonts found")
    assets = json.loads((paper / "generated/assets.sources.json").read_text())
    for name, digest in assets["output_sha256"].items():
        require(sha(paper / "generated" / name) == digest, "Generated asset changed: " + name)
    report_values = {}
    for key, receipt in assets["source_reports"].items():
        archive = root / receipt["archive"]
        require(sha(archive) == receipt["archive_sha256"], "Asset source archive changed")
        with zipfile.ZipFile(archive) as z:
            raw = z.read(receipt["member"])
        require(hashlib.sha256(raw).hexdigest() == receipt["sha256"], "Asset report changed")
        report_values[key] = json.loads(raw)
    for stat in assets["displayed_statistics"]:
        actual = pointer(report_values[stat["source"]], stat["pointer"])
        if "value" in stat:
            require(actual == stat["value"], "Displayed scalar differs from source")
        else:
            require(actual["estimate"] == stat["estimate"], "Displayed estimate differs")
            for ci in ("ci95", "ci99"):
                if ci in stat:
                    require(actual[ci] == stat[ci], "Displayed interval differs")
    input_files = tex_files + [paper / "references.bib", paper / "build.sh", paper / "generate_assets.py",
                             paper / "audit_manuscript.py", paper / "HYPOTHESIS_ROADMAP.md",
                             paper / "README.md", paper / "seal_checkpoint.py", paper / "requirements.txt",
                             root / "tests/test_paper_manuscript.py",
                             paper / "vendor/icml2026.sty", paper / "vendor/icml2026.bst",
                             paper / "generated/assets.sources.json"]
    receipt = {"schema": "wm_audit.paper_build.v1", "utc": datetime.now(timezone.utc).isoformat(),
               "status": "compiled and mechanically audited working draft; not submission certification",
               "main_text_pages": main_pages, "total_pages": len(doc), "pdf_bytes": pdf.stat().st_size,
               "pdf_sha256": sha(pdf), "citation_keys": cited, "bibliography_entries": bibliography_count,
               "style_source": "https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip",
               "style_archive_sha256": STYLE_SHA, "official_style_unmodified": True,
               "working_notice_override": True, "unresolved_references": 0, "overfull_boxes": overfull,
               "letter_pages": True, "pdf_author_anonymous": True, "type3_fonts": 0,
               "source_linked_statistics_checked": len(assets["displayed_statistics"]),
               "source_reports_checked": len(report_values), "no_new_experiments": True,
               "source_sha256": {p.relative_to(root).as_posix(): sha(p) for p in input_files},
               "git_commit_at_build": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()}
    (paper / "build/build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Paper verified: {main_pages} main pages / {len(doc)} total; {len(cited)} citations; {len(assets['displayed_statistics'])} source-linked statistics")
    doc.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    audit(args.root.resolve())