"""Configuration, exclusive outputs, and local (not externally registered) locks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from .data import SPLITS


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(content)


def read_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    required = {"name", "counts", "horizon", "candidates", "horizons", "scales", "editors",
                "rank", "steps", "batch_size", "seed", "bootstrap_resamples", "native_margin",
                "readout_margin", "regret_margin"}
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("name must be a nonempty string")
    if not isinstance(config["counts"], dict) or set(config["counts"]) != set(SPLITS):
        raise ValueError("All five split counts must be explicit")
    for key in ("horizon", "candidates", "rank", "batch_size", "bootstrap_resamples"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for value in [config["seed"], config["steps"], *config["counts"].values()]:
        if type(value) is not int or value < 0:
            raise ValueError("Seeds, steps and counts must be nonnegative integers")
    h = config["horizons"]
    if (not isinstance(h, list) or not h
            or any(type(v) is not int or not 1 <= v <= config["horizon"] for v in h)
            or len(set(h)) != len(h)):
        raise ValueError("Horizons must be distinct 1-indexed forecast steps")
    if (not isinstance(config["scales"], list) or len(config["scales"]) != 6
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in config["scales"])):
        raise ValueError("Six fixed positive physical scales are required")
    if (not isinstance(config["editors"], list) or not config["editors"]
            or any(not isinstance(v, str) for v in config["editors"])
            or len(set(config["editors"])) != len(config["editors"])
            or not set(config["editors"]) <= {"inverse", "ridge", "mlp", "rollout"}):
        raise ValueError("Select distinct implemented editor families")
    for key in ("native_margin", "readout_margin", "regret_margin"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    return config


def runtime():
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__, "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def code_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def fitted_hashes(run_dir):
    run_dir = Path(run_dir)
    paths = [run_dir / "training.json", run_dir / "scorer.pt", *sorted((run_dir / "editors").glob("*.pt"))]
    if len(paths) < 3:
        raise ValueError("No fitted editors; run fit first")
    return {str(p.relative_to(run_dir)): digest(p) for p in paths}


def select_primary(config, records):
    scores = {}
    roster = None
    for method in config["editors"]:
        rows = [r for r in records if r["method"] == method and r["appearance"] == "original"]
        if not rows or any(r["split"] != "validation" for r in rows):
            raise ValueError(f"Missing clean validation for {method}")
        ids = [r["root_id"] for r in rows]
        if (len(ids) != config["counts"]["validation"] or len(set(ids)) != len(ids)
                or (roster is not None and set(ids) != roster)):
            raise ValueError(f"Incomplete or inconsistent clean validation roster for {method}")
        roster = set(ids)
        for row in rows:
            for key in ("history_fidelity", "specificity_loss"):
                value = row[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid clean validation metric: {key}")
        scores[method] = float(np.mean([r["history_fidelity"] + r["specificity_loss"] for r in rows]))
        if not math.isfinite(scores[method]):
            raise ValueError("Clean validation score overflowed")
    return min(scores, key=lambda m: (scores[m], m)), scores


def create_lock(config, run_dir, data_dir):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    if list((data_dir / "confirmation").glob("*.npz")):
        raise ValueError("Confirmation data already exists. Lock before generating it.")
    validation = run_dir / "validation.json"
    result = json.loads(validation.read_text())
    if result["config"] != config or json.loads((run_dir / "training.json").read_text())["config"] != config:
        raise ValueError("Config must match fitting and validation")
    selected, scores = select_primary(config, result["records"])
    if {p.stem for p in (run_dir / "editors").glob("*.pt")} != set(config["editors"]):
        raise ValueError("Fitted editor roster differs from configuration")
    if config["counts"]["confirmation"] < 2:
        raise ValueError("No confirmation roster configured")
    payload = {"schema": 1, "config": config, "config_hash": json_digest(config),
               "code": code_hashes(), "fitted": fitted_hashes(run_dir),
               "validation_hash": digest(validation), "primary_method": selected,
               "selection_scores": scores,
               "development_data": {str(p.relative_to(data_dir)): digest(p) for p in sorted(data_dir.glob("*/*.npz"))},
               "selection_rule": "min mean(clean history_fidelity + specificity_loss); lexical ties",
               "runtime": runtime(), "status": "LOCAL COMMITMENT; NOT EXTERNAL REGISTRATION"}
    write_json(run_dir / "protocol-lock.json", payload)
    return payload


def verify_lock(config, run_dir, data_dir):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    lock = json.loads((run_dir / "protocol-lock.json").read_text())
    if lock["validation_hash"] != digest(run_dir / "validation.json"):
        raise ValueError("Validation data changed after lock")
    selected, scores = select_primary(config, json.loads((run_dir / "validation.json").read_text())["records"])
    if (lock["schema"] != 1 or lock["config"] != config
            or lock["primary_method"] != selected or lock["selection_scores"] != scores
            or lock["config_hash"] != json_digest(config) or lock["code"] != code_hashes()
            or lock["fitted"] != fitted_hashes(run_dir)
            or lock["validation_hash"] != digest(run_dir / "validation.json")):
        raise ValueError("Protocol/configuration/code/fitted artifacts changed after lock")
    current = {str(p.relative_to(data_dir)): digest(p) for p in sorted(data_dir.glob("*/*.npz"))
               if p.parent.name != "confirmation"}
    if current != lock["development_data"]:
        raise ValueError("Development data changed after lock")
    return lock