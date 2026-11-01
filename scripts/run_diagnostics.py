"""Run exploratory prerequisites on existing data; never regenerate or refit editors.

Use the project interpreter. Each diagnostic publishes into a fresh directory.
Combined stdout/stderr is streamed to the terminal and console.log. Not a resume
tool: use --stages with a NEW output to run additional diagnostics later.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def commands(args):
    stages = args.stages or ["readout", "prediction-paired"] + (["prediction-recorded"] if args.h5 else [])
    if len(set(stages)) != len(stages):
        raise ValueError("Stages must be unique")
    if "prediction-recorded" in stages and args.h5 is None:
        raise ValueError("Recorded prediction requires --h5")
    if "dinowm" in args.models and not args.trust_checkpoint:
        raise ValueError("DINO requires --trust-checkpoint for the pinned released pickle")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Models must be unique")
    result = []
    prefix = [sys.executable, "-u", "-m", "wm_audit.cli", "--cache", str(args.cache), "--device", args.device]
    if args.trust_checkpoint:
        prefix.append("--trust-checkpoint")
    for model in args.models:
        for stage in stages:
            command = "diagnose-readout" if stage == "readout" else "diagnose-prediction"
            argv = prefix + [command, "--config", str(args.config), "--model", model,
                             "--output", str(args.output / model / stage)]
            if stage == "prediction-recorded":
                argv += ["--h5", str(args.h5), "--episodes", str(args.episodes)]
            else:
                argv += ["--data", str(args.data)]
            result.append(argv)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Existing paired data directory, including scorer/validation/pilot")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, required=True, help="New run directory under runs/ for backup inclusion")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--models", nargs="+", choices=("lewm", "dinowm"), default=["lewm", "dinowm"])
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--h5", type=Path, help="Optional recorded PushT H5; never downloaded implicitly")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--stages", nargs="+", choices=("readout", "prediction-paired", "prediction-recorded"))
    args = parser.parse_args(argv)
    plan = commands(args)
    if not args.data.is_dir() or not args.config.is_file() or (args.h5 is not None and not args.h5.is_file()):
        parser.error("Input data/config/H5 path does not exist")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnostics-plan.json").write_text(json.dumps({"commands": plan, "note": "Exploratory; no world-model training, no editor refit, no generation"}, indent=2))
    with (args.output / "console.log").open("x") as log:
        for index, command in enumerate(plan, 1):
            message = f"\n>>> DIAGNOSTIC {index}/{len(plan)}: {shlex.join(command)}\n"
            print(message, flush=True)
            log.write(message)
            log.flush()
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            if code:
                raise SystemExit(f"Diagnostic exited {code}; preserve this output. No automatic resume or overwrite.")
    print(f"DONE: inspect {args.output}/<model>/<diagnostic>/index.html and report.json", flush=True)
    if args.h5 is None:
        print("Recorded-data diagnostic was NOT run: supply --h5 for that separate check.", flush=True)


if __name__ == "__main__":
    main()