"""Seal an exclusive paper evidence checkpoint after build/audit/tests pass."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal(version):
    if not re.fullmatch(r"v[0-9]+(?:-[a-z0-9-]+)?", version):
        raise ValueError("Use a stable version ID, such as v0")
    paper = ROOT / "paper"
    output = paper / "checkpoints" / version
    if output.exists():
        raise FileExistsError("Keep previous paper checkpoints immutable")
    receipt_path = paper / "build/build-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    pdf = paper / "build/main.pdf"
    if receipt["pdf_sha256"] != sha(pdf) or receipt["main_text_pages"] > 8:
        raise ValueError("Require the current audited PDF")
    for name, digest in receipt["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Manuscript source changed after audit: " + name)
    tests = paper / "build/manuscript-tests.xml"
    suites = list(ET.parse(tests).getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")):
        raise ValueError("Require successful current manuscript contract tests")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(pdf, output / "paper.pdf")
    shutil.copyfile(receipt_path, output / "build-receipt.json")
    shutil.copyfile(tests, output / "manuscript-tests.xml")
    sources = [p for p in paper.rglob("*") if p.is_file()
               and p.relative_to(paper).parts[0] not in ("build", "checkpoints", "__pycache__")
               and p.suffix in (".tex", ".bib", ".sty", ".bst", ".py", ".sh", ".md", ".txt", ".pdf", ".json", ".png", ".zip")
               and p.name != "example_paper.tex"]
    inventory = {}
    bundle = output / "source.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(sources):
            name = path.relative_to(ROOT).as_posix()
            if path.is_symlink():
                raise ValueError("No source symlinks")
            z.write(path, name)
            inventory[name] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
        z.writestr("SOURCE_INVENTORY.json", json.dumps(inventory, indent=2))
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:
            raise ValueError("Source ZIP failed CRC validation")
        for name, item in inventory.items():
            if hashlib.sha256(z.read(name)).hexdigest() != item["sha256"]:
                raise ValueError("Source ZIP member checksum failed")
    checkpoint = {"schema": "wm_audit.paper_checkpoint.v1", "version": version,
                  "utc": datetime.now(timezone.utc).isoformat(),
                  "status": "Complete working ICML-style paper; not submitted, accepted, or venue-certified",
                  "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                  "main_pages": receipt["main_text_pages"], "total_pages": receipt["total_pages"],
                  "sources_in_bundle": len(inventory),
                  "files": {p.name: {"sha256": sha(p), "size_bytes": p.stat().st_size}
                            for p in sorted(output.iterdir()) if p.is_file()},
                  "claim_boundary": "four primary latent-action contrasts supported; DINO ranking inconclusive; physical editor unqualified",
                  "future_hypotheses": "H01-H04 are roadmap items, not completed experiments"}
    (output / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    print("SEALED", version, checkpoint["main_pages"], "main pages; PDF SHA256", checkpoint["files"]["paper.pdf"]["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    args = parser.parse_args()
    seal(args.version)