"""Independently seeded paired PushT roots and a strict compressed NPZ schema.

Schema ``wm_audit.pusht.paired.v1`` (C=3, B=2, S=224; H defaults to 8):

=================== ======================= ================================
key                 shape                   dtype / meaning
=================== ======================= ================================
context             [C,B,3,S,S,3]           uint8 RGB at primitive 0,5,10
future              [C,B,K,H,S,S,3]         uint8 RGB at 15,20,...,10+5H
states_context      [B,3,6]                 float64 physical features
states_future       [B,K,H,6]               float64 physical features
actions             [K,2+H,10]              float32 native controls, NOT z-scored
goal_pixels         [C,S,S,3]               uint8, same goal for both branches
costs               [B,K]                   float64, 1 - final goal coverage
request             [6]                     float64, CF - factual at anchor
coverage_context    [B,3]                   float64 geometric coverage
coverage_future     [B,K,H]                 float64 geometric coverage
metadata            []                      Unicode JSON, no pickle/objects
=================== ======================= ================================

Branch 0 is factual, branch 1 counterfactual. State fields are pusher XY,
native block body-origin XY, sin(theta), cos(theta). The last two request
entries are sin(theta_cf)-sin(theta_f) and cos(theta_cf)-cos(theta_f), NOT
sin/cos of the requested angle. Context is not duplicated over candidates;
physics labels/actions are not duplicated over appearances. Only explicit
``dense_pixels()``/``dense_states()`` materialize the equivalent dense schema.

All K candidates have IDENTICAL first two action chunks. By default these
are ten zero controls (a resting prefix). All remaining controls are fixed
before outcomes, and every candidate is run to the fixed horizon even after
native success. Bounds/contact rejection occurs ONLY at the initial anchor,
before the prefix. Ties, misses, future wall contact and low-effect outcomes
are kept. A numerical/replay failure raises; it never triggers resampling.

Each root gets its own 128-bit seed, with separate anchor/request/goal/action/
appearance/environment streams. Split seeds must be distinct. Changing the
count/order of roots or another root's rejection count cannot change a root.
Palettes are independent of requests and physics. The three conditions are
original, object-color-only, background-only; this is NOT the four-condition
factorial suggested in the research plan.

Metadata stores complete initial construction recipes, source hashes, package
versions, exact time/action conventions, rejection counts and SHA256 hashes of
EVERY primitive-step physical diagnostic/reward/contact/termination trace.
Those diagnostics are checks, not incomplete simulator snapshots. Replays
always construct fresh environments and replay the entire action history.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .simulator import (
    ACTION_DIM, CONTEXT_FRAMES, FRAME_SKIP, IMAGE_SIZE, ORIGINAL, POINT_REVISION,
    SOURCE_HASHES, STATE_FIELDS, Anchor, Appearance, EpisodeRecipe, Intervention,
    InvalidAnchor, PushTSimulator, QualificationError, anchor_request, array_digest,
    checked_seed, pack_actions, validate_actions,
)


SCHEMA_VERSION = "wm_audit.pusht.paired.v1"
SPLITS = ("editor", "scorer", "validation", "pilot", "confirmation")
DEFAULT_SPLIT_SEEDS = dict(zip(SPLITS, (10201, 20323, 30449, 40577, 50707)))
DEFAULT_SPLIT_COUNTS = dict(zip(SPLITS, (2000, 600, 256, 128, 400)))
STREAMS = ("anchor", "intervention", "goal", "actions", "appearance", "environment")
ARRAY_KEYS = ("context", "future", "states_context", "states_future", "actions",
              "goal_pixels", "costs", "request", "coverage_context", "coverage_future")
METADATA_KEYS = {
    "schema", "root_id", "split", "split_seed", "root_index", "root_seed", "stream_seeds",
    "config", "frame_indices", "primitive_control_seconds", "physics_substep_seconds",
    "action_encoding", "action_scale", "state_fields", "block_position", "request_encoding",
    "recipes", "intervention", "appearances", "source", "runtime", "geometry", "sampling",
    "actions_source", "audit",
}


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _derive_seed(seed: int, *labels: Any) -> int:
    encoded = json.dumps([checked_seed(seed), *labels], separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")


def derive_root_seed(split_seed: int, split: str, root_index: int) -> int:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}.")
    return _derive_seed(split_seed, SCHEMA_VERSION, split, _integer(root_index, "root_index"))


def _root_id(split: str, index: int, seed: int) -> str:
    return f"{split}-{index:06d}-{seed:032x}"


@dataclass(frozen=True)
class GenerationConfig:
    horizon: int = 8
    candidates: int = 8
    anchor_clearance: float = 2.0
    max_anchor_attempts: int = 256

    def __post_init__(self) -> None:
        for name in ("horizon", "candidates", "max_anchor_attempts"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if not math.isfinite(float(self.anchor_clearance)) or self.anchor_clearance < 0:
            raise ValueError("anchor_clearance must be finite and nonnegative.")
        object.__setattr__(self, "anchor_clearance", float(self.anchor_clearance))


@dataclass
class PairedRoot:
    context: np.ndarray
    future: np.ndarray
    states_context: np.ndarray
    states_future: np.ndarray
    actions: np.ndarray
    goal_pixels: np.ndarray
    costs: np.ndarray
    request: np.ndarray
    coverage_context: np.ndarray
    coverage_future: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        validate_root(self)

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: getattr(self, key) for key in ARRAY_KEYS}

    def dense_pixels(self) -> np.ndarray:
        """Explicitly allocate [C,B,K,3+H,224,224,3]; normally unnecessary."""
        self.validate()
        k = self.actions.shape[0]
        prefix = np.broadcast_to(self.context[:, :, None], (3, 2, k, 3, IMAGE_SIZE, IMAGE_SIZE, 3))
        return np.concatenate((prefix, self.future), axis=3)

    def dense_states(self) -> np.ndarray:
        """Explicitly allocate [B,K,3+H,6]."""
        self.validate()
        prefix = np.broadcast_to(self.states_context[:, None], (2, self.actions.shape[0], 3, 6))
        return np.concatenate((prefix, self.states_future), axis=2)


def sample_appearances(seed: int) -> tuple[Appearance, ...]:
    rng = np.random.default_rng(checked_seed(seed))
    agent = np.asarray(ORIGINAL.agent_color) + rng.integers(-18, 19, size=3)
    block = np.asarray(ORIGINAL.block_color) + rng.integers(-18, 19, size=3)
    if np.array_equal(agent, ORIGINAL.agent_color) and np.array_equal(block, ORIGINAL.block_color):
        block[0] += 1
    background = rng.integers(228, 246, size=3)
    return (ORIGINAL, Appearance("color_only", tuple(agent), tuple(block)),
            Appearance("background_only", background_color=tuple(background)))


def sample_intervention(seed: int) -> Intervention:
    """Uniform prespecified 13-cell grid: sham, eight translations, four rotations."""
    grid = [Intervention()]
    for axis in (0, 1):
        for distance in (-32.0, -16.0, 16.0, 32.0):
            delta = [0.0, 0.0]
            delta[axis] = distance
            grid.append(Intervention(block_translation=tuple(delta)))
    grid.extend(Intervention(block_rotation=math.radians(degrees)) for degrees in (-20, -10, 10, 20))
    return grid[int(np.random.default_rng(checked_seed(seed)).integers(len(grid)))]


def _sample_goal(seed: int) -> Anchor:
    rng = np.random.default_rng(seed)
    # This support is geometrically valid without outcome-based goal rejection.
    return Anchor(tuple(rng.uniform((64, 420), (96, 450))),
                  tuple(rng.uniform((200, 180), (300, 260))), float(rng.uniform(-math.pi, math.pi)))


def make_candidate_bank(anchor: Anchor, goal: Anchor, seed: int,
                        config: GenerationConfig = GenerationConfig(),
                        prefix: np.ndarray | None = None,
                        local_cog: Sequence[float] = (0.0, 45.0)) -> np.ndarray:
    """Prespecified open-loop bank, constructed without executing/evaluating it.

    Candidate 0 holds; 1 approaches the factual block; 2 moves away; the rest
    are smooth random controls biased toward approach/goal directions. These
    are intentions, not outcome/contact strata. No candidate is success-filtered.
    ``prefix``, if supplied, is exactly ten primitive native controls [10,2].
    """
    rng = np.random.default_rng(checked_seed(seed))
    count = FRAME_SKIP * config.horizon
    if prefix is None:
        prefix = np.zeros((10, 2), dtype=np.float32)
    if np.asarray(prefix).shape != (10, 2):
        raise ValueError("prefix must contain ten primitive 2D controls.")
    prefix_chunks = pack_actions(prefix)

    def center(pose: Anchor) -> np.ndarray:
        x, y = local_cog
        c, s = math.cos(pose.angle), math.sin(pose.angle)
        return np.asarray(pose.block_xy) + (c * x - s * y, s * x + c * y)

    direction = center(anchor) - anchor.pusher_xy
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    goal_direction = center(goal) - center(anchor)
    goal_direction = goal_direction / max(float(np.linalg.norm(goal_direction)), 1e-12)
    bank = np.zeros((config.candidates, 2 + config.horizon, ACTION_DIM), dtype=np.float32)
    bank[:, :2] = prefix_chunks
    ramp = np.minimum(1.0, np.arange(1, count + 1) / 5.0)[:, None]
    for candidate in range(1, config.candidates):
        if candidate == 1:
            future = 0.65 * ramp * direction
        elif candidate == 2:
            future = -0.25 * ramp * direction
        else:
            smooth = rng.uniform(-0.5, 0.5, size=2)
            future = np.empty((count, 2), dtype=np.float32)
            for step in range(count):
                smooth = 0.85 * smooth + 0.15 * rng.uniform(-1, 1, size=2)
                bias = (1 - step / count) * direction + (step / count) * goal_direction
                future[step] = np.clip(0.45 * smooth + 0.30 * bias, -0.8, 0.8)
        bank[candidate, 2:] = pack_actions(future)
    return bank


def _validate_bank(actions: np.ndarray, config: GenerationConfig) -> None:
    if not isinstance(actions, np.ndarray) or actions.dtype != np.float32 or actions.shape != (config.candidates, 2 + config.horizon, ACTION_DIM):
        raise ValueError("actions must be float32 [K,2+H,10].")
    for candidate in actions:
        validate_actions(candidate)
    if not np.array_equal(actions[:, :2], np.broadcast_to(actions[0, :2], actions[:, :2].shape)):
        raise ValueError("Every candidate must share the exact same two context-action chunks.")


def _prefix_digest(trajectory: Any) -> str:
    return array_digest(trajectory.states[:3], trajectory.coverage[:3], trajectory.physics_trace[:11],
                        trajectory.rewards[:10], trajectory.contacts[:10], trajectory.contact_points[:10],
                        trajectory.terminated[:10], trajectory.truncated[:10])


def _execute(simulator: PushTSimulator, recipes: Sequence[EpisodeRecipe],
             actions: np.ndarray, appearances: Sequence[Appearance]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    k, transitions, _ = actions.shape
    horizon = transitions - 2
    arrays = {
        "context": np.empty((3, 2, 3, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "future": np.empty((3, 2, k, horizon, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "states_context": np.empty((2, 3, 6), dtype=np.float64),
        "states_future": np.empty((2, k, horizon, 6), dtype=np.float64),
        "actions": actions.copy(),
        "goal_pixels": np.empty((3, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "costs": np.empty((2, k), dtype=np.float64),
        "coverage_context": np.empty((2, 3), dtype=np.float64),
        "coverage_future": np.empty((2, k, horizon), dtype=np.float64),
    }
    audit: dict[str, Any] = {
        "diagnostic_sha256": [[""] * k for _ in range(2)], "prefix_sha256": ["", ""],
        "native_contact_points": [[0] * k for _ in range(2)],
        "native_terminated_any": [[False] * k for _ in range(2)],
        "actions_sha256": array_digest(actions),
    }
    for branch, recipe in enumerate(recipes):
        for candidate, controls in enumerate(actions):
            trajectory = simulator.rollout(recipe, controls, appearances)
            prefix_hash = _prefix_digest(trajectory)
            if candidate == 0:
                arrays["context"][:, branch] = trajectory.pixels[:, :3]
                arrays["states_context"][branch] = trajectory.states[:3]
                arrays["coverage_context"][branch] = trajectory.coverage[:3]
                audit["prefix_sha256"][branch] = prefix_hash
            elif (prefix_hash != audit["prefix_sha256"][branch]
                  or not np.array_equal(arrays["context"][:, branch], trajectory.pixels[:, :3])):
                raise QualificationError("Candidate prefix failed fresh-replay equality; do not resample.")
            if branch == 0 and candidate == 0:
                arrays["goal_pixels"][:] = trajectory.goal_pixels
            elif not np.array_equal(arrays["goal_pixels"], trajectory.goal_pixels):
                raise QualificationError("Goal rendering changed between branches/candidates.")
            arrays["future"][:, branch, candidate] = trajectory.pixels[:, 3:]
            arrays["states_future"][branch, candidate] = trajectory.states[3:]
            arrays["coverage_future"][branch, candidate] = trajectory.coverage[3:]
            arrays["costs"][branch, candidate] = trajectory.cost
            audit["diagnostic_sha256"][branch][candidate] = trajectory.diagnostic_digest
            audit["native_contact_points"][branch][candidate] = int(trajectory.contact_points.sum())
            audit["native_terminated_any"][branch][candidate] = bool(trajectory.terminated.any())
    arrays["request"] = arrays["states_context"][1, 0] - arrays["states_context"][0, 0]
    return arrays, audit


def generate_root(*, split_seed: int, root_index: int = 0, split: str = "pilot",
                  config: GenerationConfig = GenerationConfig(), simulator: PushTSimulator | None = None,
                  intervention: Intervention | None = None, anchor: Anchor | None = None,
                  goal: Anchor | None = None, actions: np.ndarray | None = None) -> PairedRoot:
    """Generate one independent root; optional explicit inputs are frozen pre-outcome.

    Invalid explicit anchors raise immediately. Only randomly proposed initial
    anchors are rejection-sampled. Simulation failures are never caught by that
    loop. A shared environment seed is used for both physical branches and K
    fresh reconstructions; no evolving RNG/environment is shared across roots.
    """
    root_seed = derive_root_seed(split_seed, split, root_index)
    seeds = {name: _derive_seed(root_seed, name) for name in STREAMS}
    simulator = simulator or PushTSimulator()
    intervention = intervention if intervention is not None else sample_intervention(seeds["intervention"])
    goal = goal if goal is not None else _sample_goal(seeds["goal"])
    simulator.check_anchor(goal, clearance=0).require()
    rng = np.random.default_rng(seeds["anchor"])
    rejections: Counter[str] = Counter()
    for attempt in range(1, config.max_anchor_attempts + 1):
        factual = anchor if anchor is not None else Anchor(
            tuple(rng.uniform(50, 450, size=2)), tuple(rng.uniform(100, 400, size=2)),
            float(rng.uniform(-math.pi, math.pi)),
        )
        counterfactual = intervention.apply(factual, simulator.block_local_cog)
        checks = [simulator.check_anchor(pose, config.anchor_clearance) for pose in (factual, counterfactual)]
        if all(check.valid for check in checks):
            break
        if anchor is not None:
            raise InvalidAnchor("Explicit factual/counterfactual anchor rejected: " + repr([c.reasons for c in checks]))
        for branch, check in zip(("factual", "counterfactual"), checks):
            rejections.update(f"{branch}.{reason}" for reason in check.reasons)
    else:
        raise InvalidAnchor(f"No valid initial anchor in {config.max_anchor_attempts} attempts: {dict(rejections)}")
    appearances = sample_appearances(seeds["appearance"])
    actions_source = "fixed_bank_v1" if actions is None else "supplied_before_rollout"
    if actions is None:
        actions = make_candidate_bank(factual, goal, seeds["actions"], config, local_cog=simulator.block_local_cog)
    _validate_bank(actions, config)
    recipes = [EpisodeRecipe(seeds["environment"], pose, goal) for pose in (factual, counterfactual)]
    # Everything above is fixed before the first candidate is executed.
    arrays, audit = _execute(simulator, recipes, actions, appearances)
    metadata = {
        "schema": SCHEMA_VERSION,
        "root_id": _root_id(split, root_index, root_seed),
        "split": split, "split_seed": checked_seed(split_seed), "root_index": int(root_index),
        "root_seed": root_seed, "stream_seeds": seeds, "config": asdict(config),
        "frame_indices": list(range(0, 5 * (3 + config.horizon), 5)),
        "primitive_control_seconds": 0.1, "physics_substep_seconds": 0.01,
        "action_encoding": "native_relative_xy_time_major_5", "action_scale": 100.0,
        "state_fields": list(STATE_FIELDS), "block_position": "native_body_origin",
        "request_encoding": "counterfactual_minus_factual_anchor_features",
        "recipes": [recipe.to_dict() for recipe in recipes], "intervention": asdict(intervention),
        "appearances": [asdict(appearance) for appearance in appearances],
        "source": deepcopy(simulator.source), "runtime": dict(simulator.runtime),
        "geometry": deepcopy(simulator.geometry),
        "sampling": {"attempts": attempt, "rejected_anchors": attempt - 1,
                     "rejection_reasons": dict(sorted(rejections.items())),
                     "eligibility": "anchor_only", "anchor_source": "supplied" if anchor is not None else "sampled",
                     "rng": "numpy.PCG64", "future_filtering": False},
        "actions_source": actions_source, "audit": audit,
    }
    # Canonical JSON types make an in-memory root and a loaded root identical.
    metadata = json.loads(json.dumps(metadata, allow_nan=False, sort_keys=True))
    result = PairedRoot(**arrays, metadata=metadata)
    validate_root(result)
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_root(root: PairedRoot) -> None:
    """Strict structural/semantic validation; use qualify_root for physical replay.

    Future positions are NOT range/contact filtered: the unmodified native
    relative controller can push the kinematic pusher outside the arena.
    """
    try:
        _validate_root_fields(root)
    except (AttributeError, KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed paired-root arrays or nested metadata.") from error


def _validate_root_fields(root: PairedRoot) -> None:
    meta = root.metadata
    if not isinstance(meta, dict) or set(meta) != METADATA_KEYS or meta.get("schema") != SCHEMA_VERSION:
        raise ValueError("Unexpected metadata keys or schema version.")
    nested_keys = {
        "config": {"horizon", "candidates", "anchor_clearance", "max_anchor_attempts"},
        "source": {"root", "point_revision", "sha256"},
        "geometry": {"block_local_cog", "block_vertices", "pusher_radius", "declared_block_scale",
                     "effective_tee_scale", "arena_inner_bounds"},
        "intervention": {"pusher_translation", "block_translation", "block_rotation", "rotation_pivot"},
        "sampling": {"attempts", "rejected_anchors", "rejection_reasons", "eligibility", "anchor_source", "rng", "future_filtering"},
    }
    for name, keys in nested_keys.items():
        if not isinstance(meta[name], dict) or set(meta[name]) != keys:
            raise ValueError(f"Invalid {name} metadata fields.")
    if not isinstance(meta["source"]["root"], str) or not meta["source"]["root"]:
        raise ValueError("Source provenance must identify the vendored root directory.")
    if (not isinstance(meta["runtime"], dict) or not meta["runtime"]
            or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in meta["runtime"].items())):
        raise ValueError("Runtime provenance must contain named version strings.")
    try:
        json.dumps(meta, allow_nan=False)
        config = GenerationConfig(**meta["config"])
        seed = derive_root_seed(meta["split_seed"], meta["split"], meta["root_index"])
        recipes = [EpisodeRecipe.from_dict(value) for value in meta["recipes"]]
        intervention = Intervention(**meta["intervention"])
        appearances = tuple(Appearance(**value) for value in meta["appearances"])
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError("Malformed root metadata.") from error
    if meta["root_seed"] != seed or meta["root_id"] != _root_id(meta["split"], meta["root_index"], seed):
        raise ValueError("Root identity/seed does not match the independent seed derivation.")
    seeds = {name: _derive_seed(seed, name) for name in STREAMS}
    if meta["stream_seeds"] != seeds:
        raise ValueError("Stream seeds do not match the root seed.")
    if appearances != sample_appearances(seeds["appearance"]):
        raise ValueError("Appearance roster is not the independent, three-condition seeded roster.")
    if len(recipes) != 2 or any(recipe.seed != seeds["environment"] for recipe in recipes) or recipes[0].goal != recipes[1].goal:
        raise ValueError("Branches must have the same environment seed and goal.")
    if meta["source"].get("sha256") != SOURCE_HASHES or meta["source"].get("point_revision") != POINT_REVISION:
        raise ValueError("Wrong simulator source provenance.")
    geometry = meta["geometry"]
    if (geometry.get("block_local_cog") != [0.0, 45.0] or geometry.get("pusher_radius") != 15.0
            or geometry.get("effective_tee_scale") != 30.0 or geometry.get("declared_block_scale") != 40.0
            or geometry.get("arena_inner_bounds") != [7.0, 504.0]):
        raise ValueError("Wrong Point PushT geometry contract.")
    vertices = np.asarray(geometry["block_vertices"], dtype=np.float64)
    expected_polygons = {
        frozenset(((-60.0, 30.0), (60.0, 30.0), (60.0, 0.0), (-60.0, 0.0))),
        frozenset(((-15.0, 30.0), (-15.0, 120.0), (15.0, 120.0), (15.0, 30.0))),
    }
    if (vertices.shape != (2, 4, 2) or not np.isfinite(vertices).all()
            or {frozenset(map(tuple, polygon)) for polygon in vertices} != expected_polygons):
        raise ValueError("Stored geometry is not the native scale-30 T.")
    expected_cf = intervention.apply(recipes[0].anchor, geometry["block_local_cog"])
    if not np.allclose(expected_cf.features(), recipes[1].anchor.features(), rtol=0, atol=1e-10):
        raise ValueError("Counterfactual initial condition does not implement the declared intervention.")
    expected_conventions = {
        "frame_indices": list(range(0, FRAME_SKIP * (CONTEXT_FRAMES + config.horizon), FRAME_SKIP)),
        "primitive_control_seconds": 0.1, "physics_substep_seconds": 0.01,
        "action_encoding": "native_relative_xy_time_major_5", "action_scale": 100.0,
        "state_fields": list(STATE_FIELDS), "block_position": "native_body_origin",
        "request_encoding": "counterfactual_minus_factual_anchor_features",
    }
    if any(meta[key] != value for key, value in expected_conventions.items()):
        raise ValueError("Time, action, feature or request conventions do not match the schema.")
    k, h, s = config.candidates, config.horizon, IMAGE_SIZE
    specs = {
        "context": ((3, 2, 3, s, s, 3), np.uint8),
        "future": ((3, 2, k, h, s, s, 3), np.uint8),
        "states_context": ((2, 3, 6), np.float64), "states_future": ((2, k, h, 6), np.float64),
        "actions": ((k, 2 + h, 10), np.float32), "goal_pixels": ((3, s, s, 3), np.uint8),
        "costs": ((2, k), np.float64), "request": ((6,), np.float64),
        "coverage_context": ((2, 3), np.float64), "coverage_future": ((2, k, h), np.float64),
    }
    for name, (shape, dtype) in specs.items():
        value = getattr(root, name)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}.")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values.")
    _validate_bank(root.actions, config)
    for states in (root.states_context, root.states_future):
        if not np.allclose(np.sum(states[..., 4:6] ** 2, axis=-1), 1.0, rtol=0, atol=1e-10):
            raise ValueError("State orientation must be unit sin/cos, not angles or deltas.")
    expected_anchor = np.stack([recipe.anchor.features() for recipe in recipes])
    if not np.allclose(root.states_context[:, 0], expected_anchor, rtol=0, atol=1e-10):
        raise ValueError("Frame zero does not match the physical intervention anchors.")
    for expected in (anchor_request(recipes[0].anchor, recipes[1].anchor),
                     root.states_context[1, 0] - root.states_context[0, 0]):
        if not np.allclose(root.request, expected, rtol=0, atol=1e-10):
            raise ValueError("request is not the proper physical feature delta at the initial anchor.")
    for value in (root.coverage_context, root.coverage_future, root.costs):
        if np.any((value < 0) | (value > 1)):
            raise ValueError("Coverage/cost must be in [0,1].")
    if not np.allclose(root.costs, 1.0 - root.coverage_future[:, :, -1], rtol=0, atol=1e-12):
        raise ValueError("costs must equal 1 - final native geometric goal coverage.")
    sampling = meta["sampling"]
    attempts = _integer(sampling["attempts"], "attempts", 1)
    rejected = _integer(sampling["rejected_anchors"], "rejected_anchors")
    if (attempts > config.max_anchor_attempts or rejected != attempts - 1
            or sampling["eligibility"] != "anchor_only" or sampling["future_filtering"] is not False
            or sampling["rng"] != "numpy.PCG64" or sampling["anchor_source"] not in ("supplied", "sampled")
            or (sampling["anchor_source"] == "supplied" and attempts != 1)):
        raise ValueError("Invalid rejection/eligibility metadata.")
    reasons = sampling["rejection_reasons"]
    allowed_reasons = {f"{b}.{r}" for b in ("factual", "counterfactual")
                       for r in ("pusher_wall", "block_wall", "pusher_block")}
    if not isinstance(reasons, dict) or not set(reasons) <= allowed_reasons:
        raise ValueError("Only initial geometry can be a rejection reason.")
    counts = [_integer(value, "rejection reason count", 1) for value in reasons.values()]
    if sum(counts) < attempts - 1 or any(value > attempts - 1 for value in counts):
        raise ValueError("Inconsistent anchor rejection counts.")
    if meta["actions_source"] not in ("fixed_bank_v1", "supplied_before_rollout"):
        raise ValueError("Unknown candidate construction rule.")
    if meta["actions_source"] == "fixed_bank_v1":
        expected_bank = make_candidate_bank(recipes[0].anchor, recipes[0].goal, seeds["actions"], config,
                                            local_cog=geometry["block_local_cog"])
        if not np.array_equal(root.actions, expected_bank):
            raise ValueError("Action bank does not reproduce the declared pre-outcome generation rule.")
    audit = meta["audit"]
    if set(audit) != {"diagnostic_sha256", "prefix_sha256", "native_contact_points", "native_terminated_any", "actions_sha256"}:
        raise ValueError("Invalid primitive audit fields.")
    digests = np.asarray(audit["diagnostic_sha256"])
    if digests.shape != (2, k) or not all(_is_sha256(x) for x in digests.flat):
        raise ValueError("Missing primitive diagnostic hashes.")
    if len(audit["prefix_sha256"]) != 2 or not all(_is_sha256(x) for x in audit["prefix_sha256"]):
        raise ValueError("Missing prefix diagnostic hashes.")
    contact_counts = np.asarray(audit["native_contact_points"])
    terminal_flags = np.asarray(audit["native_terminated_any"])
    if contact_counts.shape != (2, k) or contact_counts.dtype.kind not in "iu" or np.any(contact_counts < 0):
        raise ValueError("Invalid native contact diagnostics.")
    if terminal_flags.shape != (2, k) or terminal_flags.dtype != np.bool_:
        raise ValueError("Invalid native termination diagnostics.")
    if audit["actions_sha256"] != array_digest(root.actions):
        raise ValueError("Action bank does not match its provenance hash.")


def save_root(path: str | Path, root: PairedRoot, *, overwrite: bool = False) -> Path:
    """Validate, compress and atomically publish one NPZ (exclusive by default)."""
    validate_root(root)
    path = Path(path).expanduser().resolve()
    if path.suffix != ".npz":
        raise ValueError("A root archive must have the .npz suffix.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(root.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".paired-", suffix=".npz", delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **root.arrays(), metadata=np.array(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomic, and cannot overwrite a concurrent writer
        return path
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def load_root(path: str | Path) -> PairedRoot:
    """Read with allow_pickle=False and validate the full fixed schema."""
    with np.load(Path(path), allow_pickle=False) as archive:
        expected = {*ARRAY_KEYS, "metadata"}
        if set(archive.files) != expected or len(archive.files) != len(expected):
            raise ValueError("Unexpected/missing/duplicate NPZ keys; dense and deduplicated schemas cannot be mixed.")
        encoded = archive["metadata"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("metadata must be scalar Unicode JSON, never pickled objects.")
        metadata = json.loads(encoded.item(), object_pairs_hook=_json_object, parse_constant=_reject_json_constant)
        root = PairedRoot(**{name: archive[name] for name in ARRAY_KEYS}, metadata=metadata)
    validate_root(root)
    return root


def replay_root(root: PairedRoot, simulator: PushTSimulator | None = None) -> PairedRoot:
    """Reconstruct all branches/candidates from recipes, never from stored states."""
    validate_root(root)
    simulator = simulator or PushTSimulator(root.metadata["source"]["root"])
    recipes = [EpisodeRecipe.from_dict(value) for value in root.metadata["recipes"]]
    appearances = [Appearance(**value) for value in root.metadata["appearances"]]
    arrays, audit = _execute(simulator, recipes, root.actions, appearances)
    metadata = deepcopy(root.metadata)
    metadata["audit"] = audit
    metadata["runtime"] = dict(simulator.runtime)
    result = PairedRoot(**arrays, metadata=metadata)
    validate_root(result)
    return result


def qualify_root(root: PairedRoot, simulator: PushTSimulator | None = None,
                 *, check_appearance: bool = True) -> dict[str, Any]:
    """Strict E01 qualification of a stored root, including every candidate.

    Recompute all stored pixels, features, coverage, costs and primitive hashes.
    By default also run independent clean-only/repeat/multi-appearance physics
    comparisons for every branch/candidate. No outcomes are used for filtering.
    Returns a JSON-serializable assertion report; raises on any mismatch.
    """
    validate_root(root)
    simulator = simulator or PushTSimulator(root.metadata["source"]["root"])
    replayed = replay_root(root, simulator)
    for name in ARRAY_KEYS:
        if not np.array_equal(getattr(root, name), getattr(replayed, name)):
            raise QualificationError(f"Stored {name} failed fresh reconstruction/replay.")
    if root.metadata["audit"] != replayed.metadata["audit"]:
        raise QualificationError("Stored primitive physics/contact/reward/termination hashes failed replay.")
    reports = []
    if check_appearance:
        appearances = [Appearance(**value) for value in root.metadata["appearances"]]
        for value in root.metadata["recipes"]:
            recipe = EpisodeRecipe.from_dict(value)
            for actions in root.actions:
                reports.append(simulator.qualify_replay_and_appearance(recipe, actions, appearances))
    return {"passed": True, "root_id": root.metadata["root_id"],
            "stored_arrays_exact": True, "primitive_diagnostics_exact": True,
            "appearance_checked": check_appearance, "appearance_reports": reports,
            "runtime_matches_generation": root.metadata["runtime"] == simulator.runtime,
            "runtime": dict(simulator.runtime)}


def validate_split_disjointness(roots: Iterable[PairedRoot | str | Path]) -> None:
    """Reject duplicate root families/seeds and reused split seed rosters."""
    identities: set[str] = set()
    root_seeds: set[int] = set()
    environment_seeds: set[int] = set()
    split_seeds: dict[str, int] = {}
    for value in roots:
        root = value if isinstance(value, PairedRoot) else load_root(value)
        validate_root(root)
        meta = root.metadata
        identity, seed, env_seed = meta["root_id"], meta["root_seed"], meta["stream_seeds"]["environment"]
        if identity in identities or seed in root_seeds or env_seed in environment_seeds:
            raise ValueError("Duplicate source root/episode seed; siblings must not cross splits or be counted twice.")
        identities.add(identity)
        root_seeds.add(seed)
        environment_seeds.add(env_seed)
        split, split_seed = meta["split"], meta["split_seed"]
        if split in split_seeds and split_seeds[split] != split_seed:
            raise ValueError("A split contains inconsistent seed rosters.")
        if any(other != split and used == split_seed for other, used in split_seeds.items()):
            raise ValueError("Split seeds must be distinct.")
        split_seeds[split] = split_seed


def generate_splits(output_dir: str | Path, *, counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
                    split_seeds: Mapping[str, int] = DEFAULT_SPLIT_SEEDS,
                    config: GenerationConfig = GenerationConfig(), simulator: PushTSimulator | None = None,
                    qualify: bool = False) -> dict[str, list[Path]]:
    """Write separate editor/scorer/validation/pilot/confirmation root archives.

    Supply all five counts (zero is allowed) and five distinct seeds. Defaults
    are the planned 3,384 roots, NOT a smoke run. Files are never overwritten.
    Optional qualification aborts on failure rather than replacing that root.
    """
    if set(counts) != set(SPLITS) or set(split_seeds) != set(SPLITS):
        raise ValueError("Provide counts and seeds for all five named splits.")
    counts = {split: _integer(counts[split], f"{split} count") for split in SPLITS}
    seeds = {split: checked_seed(split_seeds[split]) for split in SPLITS}
    if len(set(seeds.values())) != len(SPLITS):
        raise ValueError("Editor/scorer/validation/pilot/confirmation need separate split seeds.")
    output = Path(output_dir).expanduser().resolve()
    roster = {split: [output / split / f"{_root_id(split, index, derive_root_seed(seeds[split], split, index))}.npz"
                      for index in range(counts[split])] for split in SPLITS}
    planned_seeds = [derive_root_seed(seeds[split], split, index) for split in SPLITS for index in range(counts[split])]
    if len(planned_seeds) != len(set(planned_seeds)):
        raise ValueError("Unexpected root-seed collision.")
    if any(path.exists() for paths in roster.values() for path in paths):
        raise FileExistsError("A planned root already exists; choose a fresh output/seed roster.")
    if not planned_seeds:
        return roster
    simulator = simulator or PushTSimulator()
    for split in SPLITS:
        for index, path in enumerate(roster[split]):
            root = generate_root(split_seed=seeds[split], root_index=index, split=split,
                                 config=config, simulator=simulator)
            if qualify:
                qualify_root(root, simulator)
            save_root(path, root)
    return roster