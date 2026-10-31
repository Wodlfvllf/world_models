"""Exact Point-LeWM vendored, RGB/Pymunk PushT adapter (not a surrogate).

Only the four inspected upstream modules are loaded, in a private namespace.
Their bytes are hash checked; neither the installed ``stable_worldmodel`` nor
its heavyweight package initializer is used. Set WM_AUDIT_POINT_ROOT to the
directory containing ``stable_worldmodel/`` (the default is the inspected
``/tmp/wcm-point/stable-worldmodel`` checkout).

Runtime imports: numpy, gymnasium, loguru, pygame, pymunk >= 7, shapely, cv2
(install ONE of opencv-python/opencv-python-headless). No torch/Lance/MuJoCo.

One primitive control is the native relative action in [-1, 1]^2: the PD
target is current pusher position + 100 * action, NOT an absolute target or a
dataset-z-scored action. It runs ten 0.01 s physics substeps. Five distinct
controls, flattened in time-major XY order, form one 10D model action.

An Anchor is a resting initial condition, never a moving-state snapshot.
Every rollout constructs a new native environment, performs its native reset,
and replays ALL controls, including the ten context controls. The native reset
has two settling substeps (goal then initial pose); both poses are checked
contact-free BEFORE reset. Frame zero is after that reset, before any control.
The seven native observation numbers are never used to restore a trajectory.

Features are pusher XY, block *body-origin* XY, sin(theta), cos(theta). Native
body-origin and center of gravity differ. Rotations default to the physical
center of gravity; their induced body-origin displacement is included in the
6D request. A request is CF_features - factual_features, NOT sin/cos(dtheta).
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Sequence

import numpy as np


FRAME_SKIP = 5
CONTEXT_FRAMES = 3
IMAGE_SIZE = 224
STATE_DIM = 6
ACTION_DIM = 10
POINT_REVISION = "1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424"
SOURCE_HASHES = {
    "utils.py": "9979483129df100cf4728dd45d1b6d0bb02d1fadec637d7f7065fe7894a3c276",
    "spaces.py": "73079b2a66dace087abc7c417691e3cc339887866915553e21d0936a362229eb",
    "envs/utils.py": "2e5d98a5c13227cba6f5574bf72b6eab149ee5b3783df0481b8d240f34047f56",
    "envs/pusht/env.py": "92d2b684e3ba494703f9f8012116b29b4a93ac13cd556f3c6388c28da9a0214f",
}
DEFAULT_POINT_ROOT = Path("/tmp/wcm-point/stable-worldmodel")
STATE_FIELDS = ("pusher_x", "pusher_y", "block_x", "block_y", "sin_theta", "cos_theta")


class SourceMismatchError(RuntimeError):
    """The supplied source is not the inspected Point vendored implementation."""


class SimulatorDependencyError(ImportError):
    """A dependency of the real simulator is missing or incompatible."""


class InvalidAnchor(ValueError):
    """An initial pose, not a future outcome, violates geometric eligibility."""


class QualificationError(RuntimeError):
    """Replay, rendering invariance, or the native initialization contract failed."""


def checked_seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Seeds must be nonnegative integers, not booleans.")
    if not 0 <= int(value) < 2**128:
        raise ValueError("Seeds must be in [0, 2**128).")
    return int(value)


def _finite_vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numbers.")
    return tuple(float(x) for x in array)


@dataclass(frozen=True)
class Anchor:
    pusher_xy: tuple[float, float]
    block_xy: tuple[float, float]
    angle: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pusher_xy", _finite_vector(self.pusher_xy, 2, "pusher_xy"))
        object.__setattr__(self, "block_xy", _finite_vector(self.block_xy, 2, "block_xy"))
        if not math.isfinite(float(self.angle)):
            raise ValueError("angle must be finite radians.")
        object.__setattr__(self, "angle", math.remainder(float(self.angle), 2 * math.pi))

    def native_state(self) -> np.ndarray:
        """Native reset argument, with zero pusher velocity; NOT a snapshot."""
        return np.array((*self.pusher_xy, *self.block_xy, self.angle, 0.0, 0.0), dtype=np.float64)

    def features(self) -> np.ndarray:
        return physical_features(self.native_state())


def physical_features(native_state: Sequence[float]) -> np.ndarray:
    state = np.asarray(native_state, dtype=np.float64)
    if state.shape not in ((5,), (7,)) or not np.isfinite(state).all():
        raise ValueError("Expected a finite native pose (5) or observation (7).")
    return np.array((*state[:4], np.sin(state[4]), np.cos(state[4])), dtype=np.float64)


@dataclass(frozen=True)
class Intervention:
    pusher_translation: tuple[float, float] = (0.0, 0.0)
    block_translation: tuple[float, float] = (0.0, 0.0)
    block_rotation: float = 0.0
    rotation_pivot: str = "center_of_gravity"

    def __post_init__(self) -> None:
        for field in ("pusher_translation", "block_translation"):
            object.__setattr__(self, field, _finite_vector(getattr(self, field), 2, field))
        if not math.isfinite(float(self.block_rotation)):
            raise ValueError("block_rotation must be finite radians.")
        object.__setattr__(self, "block_rotation", float(self.block_rotation))
        if self.rotation_pivot not in ("center_of_gravity", "origin"):
            raise ValueError("rotation_pivot must be center_of_gravity or origin.")

    def apply(self, anchor: Anchor, local_cog: Sequence[float]) -> Anchor:
        cog = np.asarray(_finite_vector(local_cog, 2, "local_cog"))
        angle = math.remainder(anchor.angle + self.block_rotation, 2 * math.pi)
        block = np.asarray(anchor.block_xy) + self.block_translation
        if self.rotation_pivot == "center_of_gravity" and self.block_rotation != 0:
            def rotate(theta: float) -> np.ndarray:
                c, s = math.cos(theta), math.sin(theta)
                return np.array((c * cog[0] - s * cog[1], s * cog[0] + c * cog[1]))

            block = block + rotate(anchor.angle) - rotate(angle)
        return Anchor(tuple(np.asarray(anchor.pusher_xy) + self.pusher_translation), tuple(block), angle)


def anchor_request(factual: Anchor, counterfactual: Anchor) -> np.ndarray:
    """A zero edit is six zeros; circular orientation is differenced correctly."""
    return counterfactual.features() - factual.features()


@dataclass(frozen=True)
class Appearance:
    name: str = "original"
    agent_color: tuple[int, int, int] = (65, 105, 225)
    block_color: tuple[int, int, int] = (119, 136, 153)
    background_color: tuple[int, int, int] = (255, 255, 255)
    goal_color: tuple[int, int, int] = (144, 238, 144)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("An appearance needs a nonempty name.")
        for field in ("agent_color", "block_color", "background_color", "goal_color"):
            color = np.asarray(getattr(self, field))
            if color.shape != (3,) or color.dtype.kind not in "iu" or np.any((color < 0) | (color > 255)):
                raise ValueError(f"{field} must be three integer RGB values in [0, 255].")
            object.__setattr__(self, field, tuple(int(x) for x in color))


ORIGINAL = Appearance()


@dataclass(frozen=True)
class EpisodeRecipe:
    seed: int
    anchor: Anchor
    goal: Anchor

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", checked_seed(self.seed))
        if not isinstance(self.anchor, Anchor) or not isinstance(self.goal, Anchor):
            raise ValueError("anchor and goal must be Anchor instances.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeRecipe:
        if set(value) != {"seed", "anchor", "goal"}:
            raise ValueError("Invalid episode recipe keys.")
        return cls(value["seed"], Anchor(**value["anchor"]), Anchor(**value["goal"]))


@dataclass(frozen=True)
class AnchorValidity:
    reasons: tuple[str, ...]
    pusher_wall_clearance: float
    block_wall_clearance: float
    pusher_block_clearance: float

    @property
    def valid(self) -> bool:
        return not self.reasons

    def require(self) -> None:
        if self.reasons:
            raise InvalidAnchor("Invalid anchor: " + ", ".join(self.reasons))


def validate_actions(actions: np.ndarray) -> np.ndarray:
    """Validate grouped native actions, without clipping or normalizing them."""
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] < 1:
        raise ValueError("Actions must have shape [number_of_transitions, 10].")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all() or np.any(np.abs(array) > 1):
        raise ValueError("Native relative actions must be finite and in [-1, 1].")
    return np.ascontiguousarray(array, dtype=np.float32)


def pack_actions(primitives: np.ndarray) -> np.ndarray:
    array = np.asarray(primitives)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0 or array.shape[0] % FRAME_SKIP:
        raise ValueError("Primitive actions must have shape [5 * transitions, 2].")
    return validate_actions(array.reshape(-1, ACTION_DIM))


def unpack_actions(actions: np.ndarray) -> np.ndarray:
    return validate_actions(actions).reshape(-1, 2)


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(json.dumps([array.dtype.str, array.shape], separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class Trajectory:
    pixels: np.ndarray                 # [appearance, sampled_time, 224, 224, 3]
    states: np.ndarray                 # [sampled_time, 6], float64
    goal_pixels: np.ndarray            # [appearance, 224, 224, 3]
    coverage: np.ndarray               # [sampled_time], union intersection / goal area
    actions: np.ndarray                # [sampled_time - 1, 10], float32
    physics_trace: np.ndarray          # [primitive_time + 1, 18], diagnostics, NOT restorable
    rewards: np.ndarray                # [primitive_time], native negative state distance
    contacts: np.ndarray               # [primitive_time], native ceil(points / substeps)
    contact_points: np.ndarray         # [primitive_time], raw native callback count
    terminated: np.ndarray             # [primitive_time], native pose-based success
    truncated: np.ndarray              # [primitive_time]

    @property
    def cost(self) -> float:
        return float(1.0 - self.coverage[-1])

    def diagnostic_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.states, self.physics_trace, self.coverage, self.rewards,
                self.contacts, self.contact_points, self.terminated, self.truncated)

    @property
    def diagnostic_digest(self) -> str:
        return array_digest(*self.diagnostic_arrays())


_LOAD_LOCK = threading.RLock()
_CLASS_CACHE: dict[str, type] = {}


def inspect_point_source(source_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(source_root or os.environ.get("WM_AUDIT_POINT_ROOT", DEFAULT_POINT_ROOT)).expanduser().resolve()
    package = root / "stable_worldmodel"
    for relative, expected in SOURCE_HASHES.items():
        path = package / relative
        if not path.is_file():
            raise SourceMismatchError(f"Missing Point vendored source: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SourceMismatchError(f"Not the inspected Point source: {path}; refusing a silent simulator substitution.")
    return {"root": str(root), "point_revision": POINT_REVISION, "sha256": dict(SOURCE_HASHES)}


def load_point_pusht(source_root: str | Path | None = None) -> type:
    """Load unchanged, checked upstream source without importing its package facade.

    The private modules' import resolver redirects ONLY their two absolute
    ``stable_worldmodel`` imports to the same private package. It does not
    monkey-patch sys.path, builtins, or an installed stable_worldmodel package.
    """
    source = inspect_point_source(source_root)
    root = Path(source["root"]) / "stable_worldmodel"
    key = str(root)
    with _LOAD_LOCK:
        if key in _CLASS_CACHE:
            return _CLASS_CACHE[key]
        namespace = "_wm_audit_point_" + hashlib.sha256(key.encode()).hexdigest()[:16]
        modules: dict[str, ModuleType] = {}
        for suffix, directory in (("", root), (".envs", root / "envs"), (".envs.pusht", root / "envs/pusht")):
            module = ModuleType(namespace + suffix)
            module.__path__ = [str(directory)]
            module.__package__ = module.__name__
            sys.modules[module.__name__] = module
            modules[suffix] = module
        package = modules[""]
        package.envs = modules[".envs"]
        package.envs.pusht = modules[".envs.pusht"]

        def private_import(name: str, globals: Any = None, locals: Any = None,
                           fromlist: Any = (), level: int = 0) -> Any:
            if level == 0 and name == "stable_worldmodel":
                return package
            return builtins.__import__(name, globals, locals, fromlist, level)

        try:
            for relative in SOURCE_HASHES:
                name = namespace + "." + relative.removesuffix(".py").replace("/", ".")
                path = root / relative
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None:
                    raise SourceMismatchError(f"Cannot load {path}")
                module = importlib.util.module_from_spec(spec)
                module.__dict__["__builtins__"] = {**vars(builtins), "__import__": private_import}
                sys.modules[name] = module
                # Compile the actual checked bytes; no vendored source or pyc is written.
                contents = path.read_bytes()
                if hashlib.sha256(contents).hexdigest() != SOURCE_HASHES[relative]:
                    raise SourceMismatchError(f"Source changed during import: {path}")
                exec(compile(contents, str(path), "exec"), module.__dict__)
                parent, _, child = name.rpartition(".")
                setattr(sys.modules[parent], child, module)
            cls = sys.modules[namespace + ".envs.pusht.env"].PushT
            import pymunk

            if not hasattr(pymunk.Space, "on_collision"):
                raise SimulatorDependencyError("The inspected Point environment requires pymunk >= 7 (Space.on_collision).")
            _CLASS_CACHE[key] = cls
            return cls
        except Exception as error:
            for name in tuple(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    del sys.modules[name]
            if isinstance(error, ImportError) and not isinstance(error, SimulatorDependencyError):
                raise SimulatorDependencyError(
                    "Real PushT needs numpy, gymnasium, loguru, pygame, pymunk>=7,<8, "
                    "shapely and opencv-python-headless (or opencv-python). No mock fallback is available."
                ) from error
            raise


def _body_diagnostics(env: Any) -> np.ndarray:
    """Readout of velocities/forces as well as pose; never accepted for restore."""
    values = []
    for body in (env.agent, env.block):
        values.extend((*body.position, body.angle, *body.velocity, body.angular_velocity,
                       *body.force, body.torque))
    return np.asarray(values, dtype=np.float64)


def _block_polygon(env: Any, body: Any = None) -> Any:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    body = env.block if body is None else body
    # Union, not MultiPolygon: the two native T polygons share an edge.
    return unary_union([Polygon([tuple(body.local_to_world(v)) for v in shape.get_vertices()])
                        for shape in env.block.shapes])


def goal_coverage(env: Any) -> float:
    """Geometric T coverage. This is intentionally NOT the upstream reward."""
    goal_body = env._get_goal_pose_body(np.asarray(env.goal_pose, dtype=np.float64))
    target = _block_polygon(env, goal_body)
    current = _block_polygon(env)
    if not target.is_valid or not current.is_valid or target.area <= 0:
        raise QualificationError("Invalid native block/goal geometry.")
    value = float(current.intersection(target).area / target.area)
    if not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
        raise QualificationError("Nonfinite or invalid native goal coverage.")
    return min(1.0, max(0.0, value))  # only roundoff, never a failure penalty


class PushTSimulator:
    """Fresh-reconstruction simulator. Instances hold no evolving episode state."""

    def __init__(self, source_root: str | Path | None = None) -> None:
        self.source = inspect_point_source(source_root)
        self.env_class = load_point_pusht(self.source["root"])
        import cv2
        import pygame
        import pymunk
        import shapely

        self.runtime = {"python": platform.python_version(), "platform": platform.platform()}
        self.runtime.update({"chipmunk": str(pymunk.chipmunk_version), "opencv": cv2.__version__,
                             "sdl": ".".join(map(str, pygame.get_sdl_version())),
                             "geos": str(shapely.geos_version_string)})
        for package in ("numpy", "gymnasium", "loguru", "pygame", "pymunk", "shapely",
                        "opencv-python", "opencv-python-headless"):
            try:
                self.runtime[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        env = self._new_env()
        try:
            env._setup()
            self.block_local_cog = tuple(float(x) for x in env.block.center_of_gravity)
            self.geometry = {
                "block_local_cog": list(self.block_local_cog),
                "block_vertices": sorted([[list(v) for v in s.get_vertices()] for s in env.block.shapes]),
                "pusher_radius": float(next(iter(env.agent.shapes)).radius),
                "declared_block_scale": float(env.variation_space["block"]["scale"].value),
                "effective_tee_scale": 30.0,
                "arena_inner_bounds": [7.0, 504.0],
            }
        finally:
            env.close()

    def _new_env(self) -> Any:
        import pymunk.pygame_util

        if pymunk.pygame_util.positive_y_is_up:
            raise QualificationError("Global Pymunk render coordinates were changed; native PushT requires screen-down Y.")
        return self.env_class(resolution=IMAGE_SIZE, relative=True, with_target=True,
                              render_action=False, render_mode="rgb_array")

    def check_anchor(self, anchor: Anchor, clearance: float = 1.0) -> AnchorValidity:
        """Check native geometry BEFORE any physics; no future-state rejection."""
        from shapely.geometry import Point

        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("clearance must be finite and nonnegative.")
        env = self._new_env()
        try:
            env._setup()  # geometry-only throwaway space; no reset or physics step
            env.agent.position = anchor.pusher_xy
            env.block.angle = anchor.angle
            env.block.position = anchor.block_xy  # native setter's angle-before-position order
            polygon = _block_polygon(env)
            x, y = anchor.pusher_xy
            radius = float(next(iter(env.agent.shapes)).radius)
            left, bottom, right, top = polygon.bounds
            pusher_wall = min(x - radius - 7, y - radius - 7, 504 - x - radius, 504 - y - radius)
            block_wall = min(left - 7, bottom - 7, 504 - right, 504 - top)
            separation = float(polygon.distance(Point(x, y)) - radius)
            metrics = (pusher_wall, block_wall, separation)
            names = ("pusher_wall", "block_wall", "pusher_block")
            reasons = tuple(name for name, value in zip(names, metrics)
                            if value <= 0 or value < clearance)
            return AnchorValidity(reasons, *map(float, metrics))
        finally:
            env.close()

    @staticmethod
    def reset_options(recipe: EpisodeRecipe) -> dict[str, Any]:
        """Native reset recipe, including a goal marker aligned with goal_state."""
        return {
            "variation": (),
            "state": recipe.anchor.native_state(),
            "goal_state": recipe.goal.native_state(),
            "variation_values": {
                "goal.position": np.asarray(recipe.goal.block_xy, dtype=np.float64),
                "goal.angle": recipe.goal.angle,
            },
        }

    def _reset(self, recipe: EpisodeRecipe) -> Any:
        env = self._new_env()
        try:
            env.reset(seed=recipe.seed, options=self.reset_options(recipe))
            if not np.allclose(physical_features(env._get_obs()), recipe.anchor.features(), rtol=0, atol=1e-10):
                raise QualificationError("Native reset changed the requested contact-free anchor.")
            for body in (env.agent, env.block):
                if not np.allclose((*body.velocity, body.angular_velocity, *body.force, body.torque), 0, rtol=0, atol=1e-12):
                    raise QualificationError("Native reset is not at rest; refusing an incomplete-state repair.")
            if env.n_contact_points:
                raise QualificationError("Native reset produced anchor/goal contacts.")
            return env
        except Exception:
            env.close()
            raise

    @staticmethod
    def render_appearances(env: Any, appearances: Sequence[Appearance]) -> np.ndarray:
        """Use native color fields ONLY; no reset, pose setter, or physics step."""
        if not appearances or not all(isinstance(a, Appearance) for a in appearances):
            raise ValueError("At least one Appearance is required.")
        fields = (("agent", "agent_color"), ("block", "block_color"),
                  ("background", "background_color"), ("goal", "goal_color"))
        old = {body: np.array(env.variation_space[body]["color"].value, copy=True) for body, _ in fields}
        before, contacts = _body_diagnostics(env), env.n_contact_points
        images = []
        try:
            for appearance in appearances:
                for body, field in fields:
                    env.variation_space[body]["color"].set_value(np.array(getattr(appearance, field), dtype=np.uint8))
                image = env.render()
                if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
                    raise QualificationError("Native renderer violated the RGB uint8 224x224 contract.")
                images.append(image.copy())
        finally:
            for body, _ in fields:
                env.variation_space[body]["color"].set_value(old[body])
            env._set_body_color(env.agent, old["agent"].tolist())
            env._set_body_color(env.block, old["block"].tolist())
        if not np.array_equal(before, _body_diagnostics(env)) or contacts != env.n_contact_points:
            raise QualificationError("Appearance rendering changed physical diagnostics or contacts.")
        return np.stack(images)

    def rollout(self, recipe: EpisodeRecipe, actions: np.ndarray,
                appearances: Sequence[Appearance] = (ORIGINAL,)) -> Trajectory:
        """Run every primitive, even after native success; never restore a prefix pose."""
        actions = validate_actions(actions)
        appearances = tuple(appearances)
        if (not appearances or not all(isinstance(a, Appearance) for a in appearances)
            or len({a.name for a in appearances}) != len(appearances)):
            raise ValueError("Appearances must be nonempty and uniquely named.")
        self.check_anchor(recipe.anchor, clearance=0).require()
        self.check_anchor(recipe.goal, clearance=0).require()
        primitives = unpack_actions(actions)
        count, sampled = len(primitives), len(actions) + 1
        pixels = np.empty((len(appearances), sampled, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        states = np.empty((sampled, STATE_DIM), dtype=np.float64)
        coverage = np.empty(sampled, dtype=np.float64)
        trace = np.empty((count + 1, 18), dtype=np.float64)
        rewards = np.empty(count, dtype=np.float64)
        contacts = np.empty(count, dtype=np.int32)
        contact_points = np.empty(count, dtype=np.int32)
        terminated, truncated = np.empty(count, dtype=np.bool_), np.empty(count, dtype=np.bool_)
        goal_env = self._reset(EpisodeRecipe(recipe.seed, recipe.goal, recipe.goal))
        try:
            goal_pixels = self.render_appearances(goal_env, appearances)
        finally:
            goal_env.close()
        env = self._reset(recipe)
        try:
            def observe(index: int) -> None:
                states[index] = physical_features(env._get_obs())
                coverage[index] = goal_coverage(env)
                pixels[:, index] = self.render_appearances(env, appearances)

            observe(0)
            trace[0] = _body_diagnostics(env)
            for index, action in enumerate(primitives):
                _, rewards[index], terminated[index], truncated[index], info = env.step(action.copy())
                contacts[index] = info["n_contacts"]
                contact_points[index] = env.n_contact_points
                trace[index + 1] = _body_diagnostics(env)
                if (index + 1) % FRAME_SKIP == 0:
                    observe((index + 1) // FRAME_SKIP)
        finally:
            env.close()
        if not all(np.isfinite(a).all() for a in (states, coverage, trace, rewards)):
            raise QualificationError("Nonfinite native trajectory; stop/report, never resample this root.")
        return Trajectory(pixels, states, goal_pixels, coverage, actions.copy(), trace,
                          rewards, contacts, contact_points, terminated, truncated)

    def qualify_replay_and_appearance(self, recipe: EpisodeRecipe, actions: np.ndarray,
                                      appearances: Sequence[Appearance], atol: float = 0.0) -> dict[str, Any]:
        """E01: independent fresh replays, including contacts/rewards/termination.

        Compare a clean-render-only execution to another clean execution AND to
        an execution with all nuisance renders inserted between physics steps.
        Diagnostic arrays include both bodies' velocities, angular velocities,
        forces and torques at EVERY primitive step. Failure is fatal, not an
        eligibility filter. Exact equality is the default on a pinned runtime.
        """
        appearances = tuple(appearances)
        if not appearances or appearances[0] != ORIGINAL:
            raise ValueError("Qualification requires original appearance first.")
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and nonnegative.")
        reference = self.rollout(recipe, actions, (ORIGINAL,))
        repeat = self.rollout(recipe, actions, (ORIGINAL,))
        rendered = self.rollout(recipe, actions, appearances)
        maximum_error = 0.0
        for other in (repeat, rendered):
            for expected, actual in zip(reference.diagnostic_arrays(), other.diagnostic_arrays()):
                if expected.dtype.kind in "biu":
                    equal = np.array_equal(expected, actual)
                else:
                    maximum_error = max(maximum_error, float(np.max(np.abs(expected - actual))))
                    equal = np.allclose(expected, actual, rtol=0, atol=atol)
                if not equal:
                    raise QualificationError("Fresh replay changed physical state, contacts, reward, coverage or termination.")
            if not np.array_equal(reference.pixels[0], other.pixels[0]) or not np.array_equal(reference.goal_pixels[0], other.goal_pixels[0]):
                raise QualificationError("Clean pixels are not exactly reproducible.")
        changed = [int(np.count_nonzero(rendered.pixels[c] != rendered.pixels[0])) for c in range(1, len(appearances))]
        goal_changed = [int(np.count_nonzero(rendered.goal_pixels[c] != rendered.goal_pixels[0])) for c in range(1, len(appearances))]
        if any(value == 0 for value in changed + goal_changed):
            raise QualificationError("A declared nuisance does not change both trajectory and goal pixels.")
        return {"passed": True, "atol": atol, "max_diagnostic_error": maximum_error,
                "primitive_controls": len(reference.rewards), "physics_substeps_per_control": 10,
                "native_contact_points": int(reference.contact_points.sum()),
                "native_terminated_any": bool(reference.terminated.any()),
                "changed_pixel_channels": changed, "changed_goal_pixel_channels": goal_changed,
                "diagnostic_sha256": reference.diagnostic_digest,
                "source": self.source, "runtime": self.runtime}