"""Original scene-camera PushT HDF5 data, not the newer sensor-view Lance data.

Public API: ``dataset_info(path)``, ``compute_action_stats(path)``, and
``H5Windows(path, horizon=8, *, episodes=None, stride=1,
normalize_actions=False, stats=None, context=3, frameskip=5)``.

A window is a dict with uint8 ``pixels[T,224,224,3]``, ``context``/``future``
views, and RAW native ``actions[T-1,10]``, where T=3+horizon. Five successive
XY controls connect each pair of sampled frames. No action after the last
frame is needed: that frame may be the episode's terminal observation.
``normalize_actions=True`` ADDS ``normalized_actions``; it never replaces
the raw ``actions`` consumed by the model adapters. Supply released full-file
``stats`` when working with a subset; otherwise stats describe this whole H5,
not the selected episodes. Full episodes are available via ``.episode(id)``
with primitive ``actions[L-1,2]`` and no terminal NaN in the returned controls.

Only episode metadata and chunked action statistics are read during indexing.
Pixels use contiguous window reads, not whole-file loads or slow H5 strides.
h5py/hdf5plugin are imported lazily; each worker opens its own read-only handle.
This loader validates the native schema, not camera provenance: use artifacts'
pinned original archive to establish provenance. Logged state is NOT a complete
simulator snapshot and must not be used to restore moving counterfactuals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

import numpy as np


CONTEXT = 3
FRAMESKIP = 5
IMAGE_SHAPE = (224, 224, 3)
_CHUNK_ROWS = 65536


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, not a boolean.")
    return int(value)


def _h5py():
    try:
        import h5py
        import hdf5plugin  # noqa: F401 -- registers the original Blosc/LZ4 pixel filter
    except ImportError as exc:
        raise ImportError("Reading original PushT H5 requires h5py and hdf5plugin; run bootstrap first.") from exc
    return h5py


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _column(handle, name: str):
    h5py = _h5py()
    if name not in handle:
        raise ValueError(f"Missing native H5 column: {name}")
    if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"H5 column {name} must not be a soft/external link.")
    value = handle[name]
    if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
        raise ValueError(f"H5 column {name} must be an ordinary in-file dataset.")
    return value


def _layout(handle) -> tuple[int, np.ndarray, np.ndarray]:
    pixels, actions = _column(handle, "pixels"), _column(handle, "action")
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != IMAGE_SHAPE or pixels.dtype != np.uint8:
        raise ValueError("pixels must be original uint8 [N,224,224,3] RGB; no automatic resizing.")
    rows = int(pixels.shape[0])
    if rows < 1 or actions.shape != (rows, 2) or actions.dtype.kind != "f":
        raise ValueError("action must be floating [N,2] with N > 0 and terminal NaNs.")
    offsets, lengths = _column(handle, "ep_offset"), _column(handle, "ep_len")
    if (offsets.ndim != 1 or lengths.ndim != 1 or offsets.shape != lengths.shape
            or not len(lengths) or offsets.dtype.kind not in "iu" or lengths.dtype.kind not in "iu"):
        raise ValueError("ep_offset and ep_len must be nonempty matching integer vectors.")
    # Check before int64 conversion to prevent unsigned overflow or float truncation.
    offsets, lengths = offsets[:], lengths[:]
    end = 0
    for offset, length in zip(offsets, lengths):
        if int(offset) != end or not 1 <= int(length) <= rows:
            raise ValueError("Episodes must be positive-length, contiguous, non-overlapping, and start at row 0.")
        end += int(length)
        if end > rows:
            raise ValueError("Episode boundaries extend beyond the H5 rows.")
    if end != rows:
        raise ValueError("Episode boundaries must cover every H5 row exactly once.")
    if "state" in handle:
        state = _column(handle, "state")
        if state.ndim != 2 or state.shape[0] != rows or state.shape[1] < 1 or state.dtype.kind not in "fiu":
            raise ValueError("Optional logged state must be a real numeric [N,D] array.")
    return rows, offsets.astype(np.int64), lengths.astype(np.int64)


def _scan_actions(handle, offsets: np.ndarray, lengths: np.ndarray, chunk_rows: int) -> dict[str, Any]:
    column = handle["action"]
    terminals = offsets + lengths - 1
    count = 0
    mean, m2 = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    minimum, maximum = np.full(2, np.inf), np.full(2, -np.inf)
    for lo in range(0, len(column), chunk_rows):
        hi = min(lo + chunk_rows, len(column))
        block = np.asarray(column[lo:hi], dtype=np.float64)
        terminal_mask = np.zeros(hi - lo, dtype=bool)
        left, right = np.searchsorted(terminals, (lo, hi))
        terminal_mask[terminals[left:right] - lo] = True
        if not np.isnan(block[terminal_mask]).all():
            raise ValueError("Every episode's terminal action must be exactly two NaNs (no outgoing control).")
        values = block[~terminal_mask]
        if not np.isfinite(values).all():
            raise ValueError("Nonterminal actions must be finite: interior NaNs/Inf cannot be dropped or zero-filled.")
        if np.any(np.abs(values) > 1):
            raise ValueError("Expected raw native relative XY actions in [-1,1], not z-scored/absolute actions.")
        size = len(values)
        if not size:
            continue
        # Merge float64 block moments (population variance, ddof=0), bounded memory.
        block_mean = values.mean(axis=0)
        delta = block_mean - mean
        total = count + size
        m2 += np.square(values - block_mean).sum(axis=0) + delta * delta * (count * size / total)
        mean += delta * (size / total)
        count = total
        minimum, maximum = np.minimum(minimum, values.min(0)), np.maximum(maximum, values.max(0))
    if not count:
        raise ValueError("H5 contains no finite nonterminal actions for population statistics.")
    population_std = np.sqrt(np.maximum(m2 / count, 0))
    # Match the released Point H5 statistics' constant-column handling.
    scale = np.where(population_std < 1e-8, 1.0, population_std)
    return {
        "action_key": "action", "action_mean": mean.tolist(), "action_std": scale.tolist(),
        "action_population_std": population_std.tolist(), "action_min": minimum.tolist(),
        "action_max": maximum.tolist(), "count": count, "ddof": 0,
        "terminal_rows_excluded": len(terminals),
    }


def dataset_info(path: str | Path, *, chunk_rows: int = _CHUNK_ROWS) -> dict[str, Any]:
    """JSON-serializable native schema, episode boundaries, and full-file stats.

    Scan actions, NOT pixels. Reject any malformed episode/terminal action;
    never silently shorten an episode, remove interior NaNs, or cross boundaries.
    """
    chunk_rows = _integer(chunk_rows, "chunk_rows", 1)
    path = Path(path).expanduser().resolve(strict=True)
    before = _fingerprint(path)
    with _h5py().File(path, "r") as handle:
        rows, offsets, lengths = _layout(handle)
        stats = _scan_actions(handle, offsets, lengths, chunk_rows)
        stats["source"] = str(path)
        result = {
            "schema": "wm_audit.pusht.h5.v1", "path": str(path), "size_bytes": before[2],
            "rows": rows, "episodes": len(lengths), "image_shape": list(IMAGE_SHAPE),
            "episode_offsets": offsets.tolist(), "episode_lengths": lengths.tolist(),
            "action_stats": stats, "has_state": "state" in handle,
            "camera_provenance": "not inferred from schema; verify the original LeWM archive",
        }
    if _fingerprint(path) != before:
        raise RuntimeError("H5 changed while its metadata/actions were being validated.")
    return result


def compute_action_stats(path: str | Path, *, chunk_rows: int = _CHUNK_ROWS) -> dict[str, Any]:
    """Finite full-file population statistics; only validated terminal NaNs are excluded."""
    return dataset_info(path, chunk_rows=chunk_rows)["action_stats"]


class H5Windows(Sequence):
    """Deterministic, episode-contained C=3/frameskip=5 model windows.

    ``episodes`` contains original episode IDs, without duplicates. ``stride``
    spaces window starts in primitive steps (not model steps). No global RNG,
    shuffling, outcome filtering, pixel normalization, or torch import is used.
    Indexing is O(log episodes) and stores no million-entry list of windows.
    """

    def __init__(self, path: str | Path, horizon: int = 8, *, episodes=None, stride: int = 1,
                 normalize_actions: bool = False, stats: Mapping[str, Any] | None = None,
                 context: int = CONTEXT, frameskip: int = FRAMESKIP):
        self.horizon = _integer(horizon, "horizon", 1)
        self.context = _integer(context, "context", 1)
        self.frameskip = _integer(frameskip, "frameskip", 1)
        self.stride = _integer(stride, "stride", 1)
        if self.context != CONTEXT or self.frameskip != FRAMESKIP:
            raise ValueError("Released models require context=3 and frameskip=5.")
        if type(normalize_actions) is not bool:
            raise ValueError("normalize_actions must be a boolean.")
        self.normalize_actions = normalize_actions
        self.path = Path(path).expanduser().resolve(strict=True)
        self.info = dataset_info(self.path)
        self._identity = _fingerprint(self.path)
        self.ep_offset = np.asarray(self.info["episode_offsets"], dtype=np.int64)
        self.ep_len = np.asarray(self.info["episode_lengths"], dtype=np.int64)
        if episodes is None:
            selected = list(range(len(self.ep_len)))
        else:
            # Inspect the original elements: np.asarray([True, 2]) loses the bool.
            selected = [_integer(value, "episode") for value in episodes]
        if len(set(selected)) != len(selected) or any(value >= len(self.ep_len) for value in selected):
            raise ValueError("episodes must contain unique, in-range episode IDs.")
        self.episodes = np.asarray(selected, dtype=np.int64)
        self._selected = frozenset(selected)
        self.num_frames = self.context + self.horizon
        self.span = (self.num_frames - 1) * self.frameskip
        counts = [max(0, (int(self.ep_len[e]) - 1 - self.span) // self.stride + 1) for e in selected]
        self._cumulative = np.cumsum(counts, dtype=np.int64)
        self.stats = dict(self.info["action_stats"] if stats is None else stats)
        self.action_mean = np.asarray(self.stats.get("action_mean"), dtype=np.float32)
        self.action_std = np.asarray(self.stats.get("action_std"), dtype=np.float32)
        if (self.stats.get("action_key") != "action" or self.action_mean.shape != (2,)
                or self.action_std.shape != (2,) or not np.isfinite(self.action_mean).all()
                or not np.isfinite(self.action_std).all() or np.any(self.action_std <= 0)):
            raise ValueError("stats must contain action_key='action', two finite means, and two positive stds.")
        self._h5 = None
        self._pid = None

    def __len__(self) -> int:
        return int(self._cumulative[-1]) if len(self._cumulative) else 0

    def _file(self):
        if _fingerprint(self.path) != self._identity:
            self.close()
            raise RuntimeError("H5 changed after indexing; construct a new validated H5Windows instance.")
        if self._h5 is None or self._pid != os.getpid():
            self.close()
            self._h5 = _h5py().File(self.path, "r", rdcc_nbytes=64 << 20, rdcc_nslots=1009)
            self._pid = os.getpid()
        return self._h5

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("Window index must be an integer, not a boolean/slice.")
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Window index out of range.")
        position = int(np.searchsorted(self._cumulative, index, side="right"))
        previous = int(self._cumulative[position - 1]) if position else 0
        return self.window(int(self.episodes[position]), (index - previous) * self.stride)

    def _episode_id(self, episode: int) -> int:
        episode = _integer(episode, "episode")
        if episode not in self._selected:
            raise IndexError("Episode is not in this loader's selected episodes.")
        return episode

    def _add_normalized(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.normalize_actions:
            actions = result["actions"]
            normalized = (actions.reshape(-1, 2) - self.action_mean) / self.action_std
            if not np.isfinite(normalized).all():
                raise ValueError("Action normalization produced nonfinite values.")
            result["normalized_actions"] = np.ascontiguousarray(normalized.reshape(actions.shape))
        return result

    def window(self, episode: int, start: int = 0) -> dict[str, Any]:
        """Read an explicitly located window; ``start`` is episode-local primitive time."""
        episode, start = self._episode_id(episode), _integer(start, "start")
        if start + self.span >= int(self.ep_len[episode]):
            raise IndexError("Window would cross the episode boundary or consume a terminal action.")
        handle = self._file()
        lo = int(self.ep_offset[episode]) + start
        hi = lo + self.span
        pixels = np.ascontiguousarray(handle["pixels"][lo:hi + 1][::self.frameskip])
        actions = np.asarray(handle["action"][lo:hi], dtype=np.float32)
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("Window contains invalid native actions; data changed after indexing.")
        result = {
            "pixels": pixels, "context": pixels[:self.context], "future": pixels[self.context:],
            "actions": np.ascontiguousarray(actions.reshape(self.num_frames - 1, 10)),
            "episode": episode, "start": start,
            "frame_indices": start + np.arange(self.num_frames, dtype=np.int64) * self.frameskip,
            "row_indices": lo + np.arange(self.num_frames, dtype=np.int64) * self.frameskip,
        }
        if "state" in handle:
            state = np.ascontiguousarray(handle["state"][lo:hi + 1][::self.frameskip])
            if not np.isfinite(state).all():
                raise ValueError("Window contains nonfinite logged states.")
            result["state"] = state
        return self._add_normalized(result)

    def episode(self, episode: int) -> dict[str, Any]:
        """Load one whole episode with L frames and L-1 primitive RAW XY actions."""
        episode = self._episode_id(episode)
        handle = self._file()
        lo, length = int(self.ep_offset[episode]), int(self.ep_len[episode])
        result = {
            "episode": episode, "pixels": handle["pixels"][lo:lo + length],
            "actions": np.asarray(handle["action"][lo:lo + length - 1], dtype=np.float32),
            "frame_indices": np.arange(length, dtype=np.int64),
            "row_indices": lo + np.arange(length, dtype=np.int64),
        }
        if "state" in handle:
            result["state"] = handle["state"][lo:lo + length]
            if not np.isfinite(result["state"]).all():
                raise ValueError("Episode contains nonfinite logged states.")
        return self._add_normalized(result)

    def close(self) -> None:
        handle = getattr(self, "_h5", None)
        if handle is not None:
            handle.close()
        self._h5, self._pid = None, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        result = self.__dict__.copy()
        result.update(_h5=None, _pid=None)
        return result

    def __del__(self):
        self.close()