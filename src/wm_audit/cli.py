"""Portable commands; run ``wm-audit --help`` after bootstrap."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .provenance import (create_lock, digest, read_config, runtime, select_primary,
                         verify_lock, write_json)
from .progress import note, progress, stage


def parser():
    p = argparse.ArgumentParser(description="Frozen LeWM / RGB DINO-WM PushT physical intervention audit")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--threads", type=int, default=4, help="CPU intra-op threads (avoid oversubscription)")
    p.add_argument("--trust-checkpoint", action="store_true", help="Authorize ONLY the hash-pinned released DINO pickle")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("download", help="Pinned models, code, and optional original HDF5 dataset")
    q.add_argument("--models", nargs="+", choices=("lewm", "dinowm"), default=["lewm", "dinowm"])
    q.add_argument("--dataset", action="store_true", help="Opt into ~13.14GB archive / ~46GB extracted HDF5")
    q = sub.add_parser("preflight", help="Execute device kernels and both real native rollouts")
    q.add_argument("--models", nargs="+", choices=("lewm", "dinowm"), default=["lewm", "dinowm"])
    q.add_argument("--output", type=Path, default=Path("runs/preflight.json"))
    q = sub.add_parser("dataset-info")
    q.add_argument("path", type=Path)
    q = sub.add_parser("generate", help="Generate paired physical roots; confirmation requires a lock")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--split", choices=("development", "confirmation"), default="development")
    q.add_argument("--run", type=Path, help="Locked model run required for confirmation")
    q = sub.add_parser("qualify", help="Fresh replay and physics-invariant renderer qualification")
    q.add_argument("root", type=Path)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("infer", help="Native inference on one stored paired root")
    q.add_argument("--model", choices=("lewm", "dinowm"), required=True)
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    for command in ("diagnose-readout", "diagnose-prediction"):
        q = sub.add_parser(command, help="Exploratory prerequisite diagnostic; no world-model training")
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--model", choices=("lewm", "dinowm"), required=True)
        q.add_argument("--output", type=Path, required=True, help="New diagnostic directory; never overwrite a pilot")
        if command == "diagnose-readout":
            q.add_argument("--data", type=Path, required=True)
            q.add_argument("--grids", type=int, nargs="+", default=[2, 4])
            q.add_argument("--alphas", type=float, nargs="+", default=[.01, 1., 100., 10000.])
        else:
            source = q.add_mutually_exclusive_group(required=True)
            source.add_argument("--data", type=Path)
            source.add_argument("--h5", type=Path)
            q.add_argument("--episodes", type=int, default=128, help="Number of recorded episodes; one window each")
    for command in ("fit", "evaluate", "lock", "analyze"):
        q = sub.add_parser(command)
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--data", type=Path, required=True)
        q.add_argument("--run", type=Path, required=True)
        if command in ("fit", "evaluate"):
            q.add_argument("--model", choices=("lewm", "dinowm"), required=True)
        if command in ("evaluate", "analyze"):
            q.add_argument("--split", choices=("validation", "pilot", "confirmation"), default="pilot")
        if command == "lock":
            q.add_argument("--acknowledge-draft", action="store_true", help="Confirm scientific review of scales/margins/protocol")
    return p


def _model(args):
    from .models import load_model
    with stage(f"load {args.model} on {args.device}"):
        model = load_model(args.model, args.cache, device=args.device, trust_checkpoint=args.trust_checkpoint)
    note(f"Model ready on {args.device}; pretrained weights frozen")
    return model


def _sim(args):
    from .simulator import PushTSimulator
    return PushTSimulator(args.cache / "sources" / "point" / "stable-worldmodel")


def _primary(args, config):
    if args.split == "confirmation":
        return verify_lock(config, args.run, args.data)["primary_method"]
    validation = args.run / "validation.json"
    if args.split != "validation":
        return select_primary(config, json.loads(validation.read_text())["records"])[0]
    return "ridge" if "ridge" in config["editors"] else config["editors"][0]


def preflight(args):
    from .data import generate_root, GenerationConfig, qualify_root
    from .models import load_model
    report = {"runtime": runtime(), "models": {}, "research_claim": "none; software qualification only"}
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install the Linux cu128 environment and check NVIDIA driver")
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 10 and (torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8)):
            raise RuntimeError("Blackwell requires this project's CUDA 12.8+ build")
        report["cuda_arch_list"] = torch.cuda.get_arch_list()
    x = torch.ones(8, 8, device=args.device)
    if (x @ x).sum().item() != 512:
        raise RuntimeError("Device kernel self-test failed")
    sim = _sim(args)
    root = generate_root(split="pilot", split_seed=987654321, root_index=0,
                         config=GenerationConfig(horizon=2, candidates=2), simulator=sim)
    report["simulator"] = qualify_root(root, sim)
    for name in progress(args.models, desc="Preflight models", unit="model"):
        begin = time.perf_counter()
        with stage(f"preflight load {name} on {args.device}"):
            model = load_model(name, args.cache, device=args.device, trust_checkpoint=args.trust_checkpoint)
        with torch.no_grad():
            history = model.encode(root.context[0])
            future = model.rollout(history, torch.tensor(np.repeat(root.actions[:1], 2, axis=0), device=args.device))
            goal = model.encode(root.goal_pixels[0][None, None])[:, 0]
            cost = model.goal_cost(future[:, -1], goal.expand_as(future[:, -1]))
        if not torch.isfinite(future).all() or not torch.isfinite(cost).all():
            raise RuntimeError(f"{name} returned nonfinite native inference")
        report["models"][name] = {"metadata": model.metadata, "future_shape": list(future.shape),
                                  "cost": cost.cpu().tolist(), "seconds": time.perf_counter() - begin}
        del model
    write_json(args.output, report)
    return {"passed": True, "output": str(args.output), "models": list(report["models"])}


def execute(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    args.cache = args.cache.expanduser().resolve()
    if args.command == "download":
        from .artifacts import prepare
        return prepare(args.cache, models=tuple(args.models), dataset=args.dataset)
    if args.command == "dataset-info":
        from .offline_data import dataset_info
        return dataset_info(args.path)
    if args.command == "preflight":
        if args.output.exists():
            raise FileExistsError(args.output)
        return preflight(args)
    if args.command == "qualify":
        from .data import load_root, qualify_root
        value = qualify_root(load_root(args.root), _sim(args))
        write_json(args.output, value)
        return {"passed": True, "output": str(args.output)}
    if args.command == "infer":
        from .data import load_root
        root = load_root(args.root)
        model = _model(args)
        with torch.no_grad():
            note("Inference: encoding original-appearance histories and goal")
            histories = model.encode(root.context[0])
            goals = model.encode(root.goal_pixels[0][None, None])[:, 0]
            output = []
            for branch in range(2):
                costs = []
                for action in progress(root.actions, desc=f"Inference branch {branch} on {args.device}", unit="candidate"):
                    future = model.rollout(histories[branch:branch + 1], torch.tensor(action[None], device=args.device))
                    costs.append(model.goal_cost(future[:, -1], goals).item())
                output.append({"branch": branch, "latent_goal_costs": costs,
                               "selected_candidate": int(np.argmin(costs)), "actual_costs": root.costs[branch].tolist()})
        value = {"model": model.metadata, "root_id": root.metadata["root_id"], "branches": output,
                 "runtime": runtime(), "root_sha256": digest(args.root)}
        write_json(args.output, value)
        return value
    config = read_config(args.config)
    if args.command in ("diagnose-readout", "diagnose-prediction"):
        from .diagnostics import readout_diagnostic, prediction_diagnostic
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        model = _model(args)
        if args.command == "diagnose-readout":
            return readout_diagnostic(config, args.data, args.output, model, grids=args.grids, alphas=args.alphas)
        return prediction_diagnostic(config, args.output, model, data=args.data, h5=args.h5, episodes=args.episodes)
    if args.command == "generate":
        from .data import DEFAULT_SPLIT_SEEDS, GenerationConfig, generate_splits
        counts = dict(config["counts"])
        if args.split == "development":
            counts["confirmation"] = 0
        else:
            if args.run is None:
                raise ValueError("Confirmation generation requires --run with a protocol lock")
            verify_lock(config, args.run, args.data)
            counts = {key: value if key == "confirmation" else 0 for key, value in counts.items()}
        result = generate_splits(args.data, counts=counts, split_seeds=DEFAULT_SPLIT_SEEDS,
                                 config=GenerationConfig(horizon=config["horizon"], candidates=config["candidates"]),
                                 simulator=_sim(args))
        manifest = {"config": config, "runtime": runtime(), "files":
                    {key: {str(p.relative_to(args.data.resolve())): digest(p) for p in paths} for key, paths in result.items()}}
        write_json(args.data / f"manifest-{args.split}.json", manifest)
        return {"counts": {key: len(paths) for key, paths in result.items()}, "data": str(args.data)}
    if args.command == "fit":
        from .pipeline import fit
        return fit(config, args.data, args.run, _model(args), seed=config["seed"])
    if args.command == "lock":
        if not args.acknowledge_draft:
            raise ValueError("Review the protocol and explicit margins, then pass --acknowledge-draft")
        return create_lock(config, args.run, args.data)
    primary = _primary(args, config)
    if args.command == "evaluate":
        from .pipeline import evaluate
        output = args.run / f"{args.split}.json"
        if output.exists():
            raise FileExistsError(output)
        if args.split == "confirmation":
            manifest = json.loads((args.data / "manifest-confirmation.json").read_text())
            expected = manifest["files"]["confirmation"]
            actual = {str(p.relative_to(args.data)): digest(p) for p in sorted((args.data / "confirmation").glob("*.npz"))}
            if expected != actual or len(actual) != config["counts"]["confirmation"]:
                raise ValueError("Confirmation roster differs from generated manifest")
        rows = evaluate(config, args.data, args.run, _model(args), split=args.split, primary_method=primary)
        write_json(output, {"config": config, "primary_method": primary, "runtime": runtime(), "records": rows})
        return {"output": str(output), "records": len(rows), "primary_method": primary}
    if args.command == "analyze":
        from .statistics import analyze_records
        source = args.run / f"{args.split}.json"
        payload = json.loads(source.read_text())
        if payload["config"] != config or payload["primary_method"] != primary:
            raise ValueError("Analysis config/primary differs from saved evaluation")
        report = analyze_records(payload["records"], resamples=config["bootstrap_resamples"], seed=config["seed"],
                                 primary_method=primary, native_margin=config["native_margin"],
                                 readout_margin=config["readout_margin"], regret_margin=config["regret_margin"])
        report["evaluation_sha256"] = digest(source)
        report["config"] = config
        output = args.run / f"report-{args.split}.json"
        write_json(output, report)
        return {"output": str(output), "conclusions": report["conclusions"]}
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        label = " ".join(str(value) for value in (
            args.command, getattr(args, "model", ""), getattr(args, "split", "")) if value)
        with stage(label):
            note(f"Requested model device: {args.device}; simulation, file validation and some algebraic fits use CPU")
            result = execute(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, PermissionError) as error:
        raise SystemExit(f"wm-audit: {error}") from error
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()