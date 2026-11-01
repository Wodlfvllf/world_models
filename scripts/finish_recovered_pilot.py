"""Finish the backed-up LeWM-complete / DINO-not-started pilot without regeneration.

Default prints a validated plan; --execute runs it. Existing data and completed
LeWM fitting/evaluations are preserved. DINO fits use a NEW output directory.
This is a specific continuation path, not interrupted-fit checkpoint resume.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from wm_audit.provenance import digest, read_config, select_primary
from wm_audit.progress import progress, stage


def plan(source, config_path, output, cache, device):
    source, config_path, output, cache = map(lambda p: Path(p).resolve(), (source, config_path, output, cache))
    config = read_config(config_path)
    if output.exists():
        raise FileExistsError("Choose a NEW DINO continuation output directory")
    if (source / "dinowm/training.json").exists():
        raise ValueError("DINO is already fitted in source; inspect its completed stages instead of refitting")
    data = source / "data"
    manifest = json.loads((data / "manifest-development.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Config differs from generated data")
    for split in ("editor", "scorer", "validation", "pilot"):
        paths = sorted((data / split).glob("*.npz"))
        actual = {p.relative_to(data).as_posix(): digest(p)
              for p in progress(paths, desc=f"Verify recovered {split} hashes", unit="file")}
        if len(paths) != config["counts"][split] or actual != manifest["files"][split]:
            raise ValueError(f"Incomplete or changed {split} data")
    training = json.loads((source / "lewm/training.json").read_text())
    validation = json.loads((source / "lewm/validation.json").read_text())
    pilot = json.loads((source / "lewm/pilot.json").read_text())
    if any(p["config"] != config for p in (training, validation, pilot)):
        raise ValueError("Completed LeWM artifacts use a different config")
    primary = select_primary(config, validation["records"])[0]
    if pilot["primary_method"] != primary:
        raise ValueError("LeWM pilot differs from clean validation selection")
    commands = []
    prefix = [sys.executable, "-u", "-m", "wm_audit.cli", "--cache", str(cache), "--device", device, "--trust-checkpoint"]
    common = ["--config", str(config_path), "--data", str(data)]
    if not (source / "lewm/report-pilot.json").exists():
        commands.append(prefix + ["analyze", *common, "--run", str(source / "lewm"), "--split", "pilot"])
    else:
        report = json.loads((source / "lewm/report-pilot.json").read_text())
        if report["evaluation_sha256"] != digest(source / "lewm/pilot.json") or report["config"] != config:
            raise ValueError("Existing LeWM report differs from saved evaluation/config")
    run = output / "dinowm"
    root = sorted((data / "pilot").glob("*.npz"))[0]
    commands.append(prefix + ["infer", "--model", "dinowm", "--root", str(root), "--output", str(run / "inference.json")])
    commands.append(prefix + ["fit", "--model", "dinowm", *common, "--run", str(run)])
    for split in ("validation", "pilot"):
        commands.append(prefix + ["evaluate", "--model", "dinowm", *common, "--run", str(run), "--split", split])
    commands.append(prefix + ["analyze", *common, "--run", str(run), "--split", "pilot"])
    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--execute", action="store_true", help="Execute; otherwise only verify and print commands")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Authorize the pinned DINO release pickle")
    args = parser.parse_args()
    commands = plan(args.source, args.config, args.output, args.cache, args.device)
    print("PLAN (commands below have not run yet):", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.execute:
        if not args.trust_checkpoint:
            parser.error("--execute requires explicit --trust-checkpoint for DINO loading")
        if args.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                parser.error("CUDA unavailable on this host; execute on your GPU VM")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "continuation.json").write_text(json.dumps({
            "source": str(args.source.resolve()), "config": str(args.config.resolve()),
            "commands": commands, "note": "Reuses data and LeWM results; DINO fit starts fresh",
        }, indent=2))
        for index, command in enumerate(commands, 1):
            with stage(f"continuation stage {index}/{len(commands)}: {shlex.join(command)}"):
                subprocess.run(command, check=True)
        print("DONE: DINO report at", args.output / "dinowm/report-pilot.json")