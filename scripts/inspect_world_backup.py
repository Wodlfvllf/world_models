"""Verify/restore a backup to a NEW directory and visualize real stored roots.

No checkpoint loading, generation, model execution or code execution from ZIP.
Run with the project's Python environment (NumPy/Pillow required for visualization).
"""
import argparse
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def restore(archive_path, destination):
    archive_path, destination = Path(archive_path), Path(destination)
    receipt = Path(str(archive_path) + ".sha256").read_text().split()[0]
    with archive_path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != receipt:
        raise ValueError("Backup SHA256 differs from receipt")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Recovery destination must be NEW")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP members")
        manifest = json.loads(archive.read("BACKUP.json"))
        if manifest["schema"] != "wm_audit.backup.v1":
            raise ValueError("Unsupported backup schema")
        if set(names) != {"BACKUP.json", *manifest["files"]}:
            raise ValueError("Backup inventory mismatch")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or ":" in info.filename or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("Unsafe ZIP member")
        destination.mkdir(parents=True)
        for name, metadata in manifest["files"].items():
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != metadata["sha256"] or target.stat().st_size != metadata["size_bytes"]:
                raise ValueError("Restored artifact hash/size mismatch; do not use partial recovery")
        (destination / "BACKUP.json").write_text(json.dumps(manifest, indent=2))
        (destination / "RESTORE-VERIFIED.json").write_text(json.dumps({
            "archive_sha256": actual, "files_verified": len(manifest["files"]),
            "completed_roots": manifest["completed_roots"],
        }, indent=2))
    return manifest


def visualize(run, output, limit=3, candidate=1):
    import numpy as np
    from PIL import Image, ImageDraw
    from wm_audit.data import load_root

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted((run / "data/pilot").glob("*.npz"))[:limit]
    if not paths:
        raise ValueError("No held-out pilot roots found")
    summaries, sections = [], []
    for path in paths:
        root = load_root(path)
        if not 0 <= candidate < len(root.actions):
            raise ValueError("Candidate index outside stored bank")
        appearances = ("Original", "Color only", "Background only")
        horizon = root.future.shape[3]
        selected = sorted({0, min(3, horizon - 1), horizon - 1})
        columns = [("Anchor t=0", "context", 0), ("Rollout entry t=10", "context", 2)]
        columns += [(f"Future h={h+1}, t={15+5*h}", "future", h) for h in selected]
        columns += [("Goal", "goal", 0)]
        width, height, left, top = 224, 248, 155, 60
        sheet = Image.new("RGB", (left + width * len(columns), top + height * 6), "#eef2f6")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 12), f"{root.metadata['root_id']} | candidate {candidate} | real simulator images, NOT model predictions", fill="black")
        for col, (label, _, _) in enumerate(columns):
            draw.text((left + col * width + 4, 38), label, fill="black")
        for appearance in range(3):
            for branch in range(2):
                row = appearance * 2 + branch
                draw.text((8, top + row * height + 80), appearances[appearance] + "\n" + ("Factual" if branch == 0 else "Counterfactual"), fill="black")
                for col, (_, kind, step) in enumerate(columns):
                    pixels = (root.context[appearance, branch, step] if kind == "context" else
                              root.future[appearance, branch, candidate, step] if kind == "future" else root.goal_pixels[appearance])
                    sheet.paste(Image.fromarray(pixels), (left + col * width, top + row * height))
        basename = path.stem
        sheet.save(output / (basename + ".png"))
        frames = []
        for t in range(3 + horizon):
            frame = Image.new("RGB", (3 * 224, 2 * 250 + 30), "#eef2f6")
            label = ImageDraw.Draw(frame)
            label.text((8, 8), f"Candidate {candidate} | primitive step {5*t} | top: factual; bottom: counterfactual", fill="black")
            for a in range(3):
                for b in range(2):
                    pixels = root.context[a, b, t] if t < 3 else root.future[a, b, candidate, t - 3]
                    frame.paste(Image.fromarray(pixels), (a * 224, 48 + b * 250))
                    label.text((a * 224 + 5, 32 + b * 250), appearances[a], fill="black")
            frames.append(frame)
        frames[0].save(output / (basename + ".gif"), save_all=True, append_images=frames[1:], duration=350, loop=0)
        # All candidate endpoints; never select examples based on good/bad outcomes.
        bank = Image.new("RGB", (224 * len(root.actions), 500), "#eef2f6")
        bank_draw = ImageDraw.Draw(bank)
        for b in range(2):
            for k in range(len(root.actions)):
                bank.paste(Image.fromarray(root.future[0, b, k, -1]), (k * 224, b * 250 + 24))
                bank_draw.text((k * 224 + 4, b * 250 + 5), f"{'F' if b == 0 else 'CF'} k={k} cost={root.costs[b,k]:.4f}", fill="black")
        bank.save(output / (basename + "-candidates.png"))
        summary = {"root_id": basename, "visualized_candidate": candidate, "intervention": root.metadata["intervention"],
                   "anchor_request": root.request.tolist(), "costs": root.costs.tolist(),
                   "candidate_cost_spread": np.ptp(root.costs, axis=1).tolist(),
                   "context_shape": list(root.context.shape), "future_shape": list(root.future.shape),
                   "actions_shape": list(root.actions.shape), "recipes": root.metadata["recipes"]}
        summaries.append(summary)
        sections.append(f'<section><h2>{html.escape(basename)}</h2><img src="{basename}.gif"><p>First three roots by filename; no outcome-based selection.</p><img src="{basename}.png"><h3>Every candidate endpoint and executed cost</h3><img src="{basename}-candidates.png"><pre>{html.escape(json.dumps(summary, indent=2))}</pre></section>')
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    (output / "index.html").write_text('''<!doctype html><meta charset="utf-8"><title>World-model paired data inspection</title>
<style>body{font:16px system-ui;background:#101827;color:#e5edf7;max-width:1600px;margin:40px auto;padding:20px}section{background:#1b283b;padding:24px;margin:24px 0;border-radius:16px}img{max-width:100%;height:auto}pre{white-space:pre-wrap}p{line-height:1.6}</style>
<h1>How the paired PushT data was generated</h1>
<p>Fresh simulator roots, not modified downloaded video. The physical intervention is applied before the observation history. Factual and counterfactual branches execute identical controls. The three appearances rerender each branch without changing its physics. These are stored simulator ground truths, not decoded model forecasts.</p>
<p>Context frames are primitive steps 0, 5, 10. Each model action groups five XY controls. Each control lasts 0.1 simulator seconds. The time-series candidate is fixed before inspecting outcomes (default 1); candidate 0 is a no-op control. All candidates appear below. Goal images are reference targets, not future observations.</p>''' + "".join(sections))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("restore")
    p.add_argument("archive", type=Path)
    p.add_argument("destination", type=Path)
    p = sub.add_parser("visualize")
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "restore":
        print(json.dumps(restore(args.archive, args.destination)["completed_roots"], indent=2))
    else:
        print(json.dumps(visualize(args.run, args.output), indent=2))