"""Contract tests plus real native PushT integration tests; no mock physics.

Integration fixtures skip only when the vendored checkout or a required import
is absent. A source hash mismatch, incompatible runtime, replay discrepancy or
physics failure FAILS the test, rather than silently switching simulators.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

from wm_audit.data import (
    ARRAY_KEYS, DEFAULT_SPLIT_SEEDS, SCHEMA_VERSION, SPLITS, GenerationConfig,
    derive_root_seed, generate_root, generate_splits, load_root, make_candidate_bank,
    qualify_root, replay_root, sample_appearances, sample_intervention, save_root,
    validate_root, validate_split_disjointness,
)
from wm_audit.simulator import (
    DEFAULT_POINT_ROOT, ORIGINAL, SOURCE_HASHES, Anchor, Appearance, EpisodeRecipe,
    Intervention, InvalidAnchor, PushTSimulator, QualificationError, SourceMismatchError,
    anchor_request, checked_seed, goal_coverage, inspect_point_source, load_point_pusht,
    pack_actions, physical_features, unpack_actions, validate_actions,
)


FACTUAL = Anchor((256.0, 125.0), (256.0, 220.0), 0.0)
GOAL = Anchor((64.0, 448.0), (256.0, 256.0), math.pi / 4)


@pytest.fixture(scope="module")
def simulator():
    for dependency in ("gymnasium", "loguru", "pygame", "pymunk", "shapely", "cv2"):
        pytest.importorskip(dependency, reason=f"Real PushT dependency missing: {dependency}")
    source = Path(os.environ.get("WM_AUDIT_POINT_ROOT", DEFAULT_POINT_ROOT))
    if not (source / "stable_worldmodel/envs/pusht/env.py").is_file():
        pytest.skip("Set WM_AUDIT_POINT_ROOT to the inspected Point vendored checkout.")
    return PushTSimulator(source)


@pytest.fixture(scope="module")
def paired(simulator):
    return generate_root(
        split_seed=71237, root_index=4, split="pilot", simulator=simulator,
        config=GenerationConfig(horizon=2, candidates=3), anchor=FACTUAL, goal=GOAL,
        intervention=Intervention(block_translation=(16.0, 0.0)),
    )


@pytest.fixture(scope="module")
def small_root(simulator):
    return generate_root(
        split_seed=81971, simulator=simulator, config=GenerationConfig(horizon=1, candidates=1),
        anchor=FACTUAL, goal=GOAL, intervention=Intervention(block_rotation=math.radians(10)),
    )


def test_sincos_request_is_anchor_feature_difference():
    factual = Anchor((100, 100), (256, 220), math.radians(179))
    edited = Intervention(block_rotation=math.radians(4), rotation_pivot="origin").apply(factual, (0, 45))
    request = anchor_request(factual, edited)
    np.testing.assert_array_equal(request[:4], np.zeros(4))
    np.testing.assert_allclose(request[4:], edited.features()[4:] - factual.features()[4:], rtol=0, atol=1e-14)
    assert not np.allclose(request[4:], [math.sin(math.radians(4)), math.cos(math.radians(4))])
    np.testing.assert_array_equal(anchor_request(factual, factual), np.zeros(6))
    np.testing.assert_allclose(np.sum(edited.features()[4:] ** 2), 1.0, atol=1e-14)


def test_rotation_preserves_physical_center_not_native_origin():
    factual = Anchor((80, 80), (256, 220), 0)
    edited = Intervention(block_rotation=math.pi / 2).apply(factual, (0, 45))
    np.testing.assert_allclose(edited.block_xy, (301, 265), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(edited.block_xy) + (-45, 0), np.asarray(factual.block_xy) + (0, 45))
    np.testing.assert_array_equal(edited.pusher_xy, factual.pusher_xy)
    translated = Intervention(pusher_translation=(2, -3), block_translation=(-16, 0)).apply(factual, (0, 45))
    np.testing.assert_array_equal(anchor_request(factual, translated), [2, -3, -16, 0, 0, 0])


@pytest.mark.parametrize("value", [-1, 2**128, True, 1.5, "42"])
def test_seed_validation(value):
    with pytest.raises(ValueError):
        checked_seed(value)


@pytest.mark.parametrize("value", [[0] * 6, [0, 0, 0, 0, np.nan], [0, 0, 0, 0, np.inf]])
def test_physical_feature_validation(value):
    with pytest.raises(ValueError):
        physical_features(value)


def test_action_chunks_contain_five_ordered_controls_not_repeats():
    primitives = np.linspace(-0.9, 0.9, 30, dtype=np.float32).reshape(15, 2)
    chunks = pack_actions(primitives)
    assert chunks.shape == (3, 10)
    np.testing.assert_array_equal(chunks[0], primitives[:5].reshape(-1))
    np.testing.assert_array_equal(unpack_actions(chunks), primitives)
    assert not np.array_equal(primitives[0], primitives[1])


@pytest.mark.parametrize("actions", [np.zeros((2, 2)), np.zeros((0, 10)),
                                     np.full((1, 10), 1.01), np.full((1, 10), np.nan),
                                     np.full((1, 10), np.inf), np.zeros((1, 10), dtype=bool)])
def test_invalid_action_chunks_fail_without_clipping(actions):
    with pytest.raises(ValueError):
        validate_actions(actions)


def test_candidate_bank_is_deterministic_with_shared_prefix():
    config = GenerationConfig()
    assert config.horizon == 8
    prefix = np.linspace(-0.1, 0.1, 20, dtype=np.float32).reshape(10, 2)
    first = make_candidate_bank(FACTUAL, GOAL, 17, config, prefix)
    second = make_candidate_bank(FACTUAL, GOAL, 17, config, prefix)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 10, 10)
    np.testing.assert_array_equal(first[:, :2], np.broadcast_to(pack_actions(prefix), (8, 2, 10)))
    np.testing.assert_array_equal(first[0, 2:], np.zeros((8, 10)))
    assert np.unique(first[1, 2].reshape(5, 2), axis=0).shape[0] == 5
    assert np.all(np.abs(first) <= 1)


def test_seed_and_palette_streams_are_order_independent():
    roster = [(split, index) for split in SPLITS for index in range(7)]
    seeds = {(split, index): derive_root_seed(DEFAULT_SPLIT_SEEDS[split], split, index) for split, index in roster}
    reverse = {(split, index): derive_root_seed(DEFAULT_SPLIT_SEEDS[split], split, index) for split, index in reversed(roster)}
    assert seeds == reverse
    assert len(set(seeds.values())) == len(seeds)
    palette = sample_appearances(519)
    assert palette == sample_appearances(519)
    assert len(palette) == 3 and palette[0] == ORIGINAL
    assert palette[1].background_color == ORIGINAL.background_color
    assert palette[2].agent_color == ORIGINAL.agent_color
    assert palette[2].block_color == ORIGINAL.block_color
    assert palette[2].background_color != ORIGINAL.background_color
    assert sample_intervention(91) == sample_intervention(91)
    with pytest.raises(ValueError):
        Appearance("bad", block_color=(1.5, 10, 20))


def test_missing_or_different_sources_fail_closed(tmp_path):
    with pytest.raises(SourceMismatchError):
        inspect_point_source(tmp_path)
    other = Path("/tmp/wcm-stable")
    if (other / "stable_worldmodel/envs/pusht/env.py").is_file():
        with pytest.raises(SourceMismatchError):
            inspect_point_source(other)


def test_loader_uses_actual_point_source_without_heavy_package_imports(simulator):
    tracked = ("stable_worldmodel", "torch", "torchvision", "lancedb", "mujoco")
    before = {name: sys.modules.get(name) for name in tracked}
    native_class = load_point_pusht(simulator.source["root"])
    assert native_class is simulator.env_class
    assert Path(inspect.getfile(native_class)).resolve() == Path(simulator.source["root"]) / "stable_worldmodel/envs/pusht/env.py"
    assert simulator.source["sha256"] == SOURCE_HASHES
    assert before == {name: sys.modules.get(name) for name in tracked}
    assert simulator.geometry["declared_block_scale"] == 40
    assert simulator.geometry["effective_tee_scale"] == 30
    assert simulator.block_local_cog == (0, 45)
    vertices = np.concatenate(simulator.geometry["block_vertices"])
    np.testing.assert_array_equal(vertices.min(axis=0), (-60, 0))
    np.testing.assert_array_equal(vertices.max(axis=0), (60, 120))


def test_anchor_contact_and_bounds_checks_happen_without_physics(simulator, monkeypatch):
    import pymunk

    def forbidden(*args, **kwargs):
        raise AssertionError("Eligibility must not advance any physics.")

    with monkeypatch.context() as patch:
        patch.setattr(pymunk.Space, "step", forbidden)
        assert simulator.check_anchor(FACTUAL).valid
        overlap = simulator.check_anchor(Anchor((256, 230), (256, 220), 0))
        wall = simulator.check_anchor(Anchor((80, 80), (490, 490), 0))
        assert "pusher_block" in overlap.reasons
        assert "block_wall" in wall.reasons
        with pytest.raises(InvalidAnchor):
            simulator.rollout(EpisodeRecipe(7, Anchor((256, 230), (256, 220), 0), GOAL), np.zeros((3, 10), dtype=np.float32))


def test_real_native_action_pixel_reward_and_contact_parity(simulator):
    recipe = EpisodeRecipe(19237, FACTUAL, GOAL)
    actions = make_candidate_bank(FACTUAL, GOAL, 719, GenerationConfig(horizon=3, candidates=2))[1]
    trajectory = simulator.rollout(recipe, actions)
    native = simulator.env_class(resolution=224, relative=True, with_target=True, render_action=False, render_mode="rgb_array")
    try:
        observation, info = native.reset(seed=recipe.seed, options={
            "variation": (), "state": FACTUAL.native_state(), "goal_state": GOAL.native_state(),
            "variation_values": {"goal.position": np.asarray(GOAL.block_xy), "goal.angle": GOAL.angle},
        })
        np.testing.assert_array_equal(trajectory.goal_pixels[0], info["goal"])
        np.testing.assert_array_equal(trajectory.pixels[0, 0], native.render())
        np.testing.assert_array_equal(trajectory.states[0], physical_features(observation["state"]))
        for index, action in enumerate(unpack_actions(actions)):
            observation, reward, terminated, truncated, info = native.step(action.copy())
            assert trajectory.rewards[index] == reward
            assert trajectory.terminated[index] == terminated
            assert trajectory.truncated[index] == truncated
            assert trajectory.contacts[index] == info["n_contacts"]
            assert trajectory.contact_points[index] == native.n_contact_points
            if (index + 1) % 5 == 0:
                frame = (index + 1) // 5
                np.testing.assert_array_equal(trajectory.pixels[0, frame], native.render())
                np.testing.assert_array_equal(trajectory.states[frame], physical_features(observation["state"]))
                assert trajectory.coverage[frame] == goal_coverage(native)
    finally:
        native.close()
    assert trajectory.contact_points.sum() > 0, "Qualification must exercise real contact, not only resting scenes."
    assert np.any(trajectory.physics_trace[:, 12:14] != 0), "Block velocities omitted by native observations must be audited."
    assert len(trajectory.rewards) == 25
    assert trajectory.cost == 1 - trajectory.coverage[-1]


def test_fresh_contact_replay_and_appearance_invariance(simulator):
    actions = make_candidate_bank(FACTUAL, GOAL, 117, GenerationConfig(horizon=3, candidates=2))[1]
    recipe = EpisodeRecipe(732, FACTUAL, GOAL)
    appearances = sample_appearances(917)
    report = simulator.qualify_replay_and_appearance(recipe, actions, appearances)
    assert report["passed"] and report["max_diagnostic_error"] == 0
    assert report["native_contact_points"] > 0
    assert all(value > 0 for value in report["changed_pixel_channels"])
    assert all(value > 0 for value in report["changed_goal_pixel_channels"])
    normal = simulator.rollout(recipe, actions, appearances)
    reverse = simulator.rollout(recipe, actions, tuple(reversed(appearances)))
    assert normal.diagnostic_digest == reverse.diagnostic_digest
    np.testing.assert_array_equal(normal.pixels, reverse.pixels[::-1])
    np.testing.assert_array_equal(normal.goal_pixels, reverse.goal_pixels[::-1])


def test_initial_intervention_precedes_all_three_context_frames(paired, small_root):
    np.testing.assert_allclose(paired.request, [0, 0, 16, 0, 0, 0], rtol=0, atol=1e-12)
    for branch in range(2):
        np.testing.assert_array_equal(paired.states_context[branch], np.broadcast_to(paired.states_context[branch, 0], (3, 6)))
    np.testing.assert_array_equal(paired.states_context[0, :, :2], paired.states_context[1, :, :2])
    np.testing.assert_allclose(paired.states_context[1, :, 2] - paired.states_context[0, :, 2], 16, atol=1e-12)
    # A rotation is physically applied at frame 0, not pasted into frame 10.
    for frame in range(3):
        np.testing.assert_allclose(small_root.states_context[1, frame] - small_root.states_context[0, frame], small_root.request, rtol=0, atol=1e-10)
    assert np.linalg.norm(small_root.request[4:]) > 0


def test_default_horizon_timing_sham_and_native_success_continuation(simulator):
    root = generate_root(split_seed=8101, simulator=simulator,
                         config=GenerationConfig(candidates=1), anchor=GOAL, goal=GOAL,
                         intervention=Intervention())
    assert root.metadata["frame_indices"] == list(range(0, 51, 5))
    assert root.actions.shape == (1, 10, 10)
    assert root.future.shape[3] == 8
    np.testing.assert_array_equal(root.request, np.zeros(6))
    np.testing.assert_array_equal(root.context[:, 0], root.context[:, 1])
    np.testing.assert_array_equal(root.future[:, 0], root.future[:, 1])
    np.testing.assert_allclose(root.costs, 0, atol=1e-14)
    assert root.metadata["audit"]["native_terminated_any"] == [[True], [True]]
    recipe = EpisodeRecipe.from_dict(root.metadata["recipes"][0])
    trajectory = simulator.rollout(recipe, root.actions[0])
    assert trajectory.terminated.all() and len(trajectory.terminated) == 50
    assert trajectory.physics_trace.shape == (51, 18)
    np.testing.assert_allclose(trajectory.coverage, 1, atol=1e-14)


def test_future_out_of_bounds_is_retained_not_rejected(simulator):
    config = GenerationConfig(horizon=3, candidates=1)
    controls = np.zeros((1, 5, 10), dtype=np.float32)
    controls[0, 2:, 0::2] = -1
    root = generate_root(split_seed=319, simulator=simulator, config=config,
                         anchor=Anchor((100, 100), (300, 280), 0), goal=GOAL,
                         intervention=Intervention(), actions=controls)
    assert root.states_future[0, 0, -1, 0] < 7
    assert root.metadata["sampling"]["attempts"] == 1
    assert root.metadata["sampling"]["future_filtering"] is False
    validate_root(root)


def test_ambient_renderer_coordinate_mutation_fails_closed(simulator, monkeypatch):
    import pymunk.pygame_util

    with monkeypatch.context() as patch:
        patch.setattr(pymunk.pygame_util, "positive_y_is_up", True)
        with pytest.raises(QualificationError, match="coordinates"):
            simulator.check_anchor(FACTUAL)


def test_schema_and_optional_dense_views(paired):
    validate_root(paired)
    assert paired.metadata["schema"] == SCHEMA_VERSION
    assert paired.context.shape == (3, 2, 3, 224, 224, 3)
    assert paired.future.shape == (3, 2, 3, 2, 224, 224, 3)
    assert paired.states_context.shape == (2, 3, 6)
    assert paired.states_future.shape == (2, 3, 2, 6)
    assert paired.actions.shape == (3, 4, 10)
    assert paired.goal_pixels.shape == (3, 224, 224, 3)
    assert paired.costs.shape == (2, 3)
    assert paired.request.shape == (6,)
    pixels, states = paired.dense_pixels(), paired.dense_states()
    assert pixels.shape == (3, 2, 3, 5, 224, 224, 3)
    assert states.shape == (2, 3, 5, 6)
    for candidate in range(3):
        np.testing.assert_array_equal(pixels[:, :, candidate, :3], paired.context)
        np.testing.assert_array_equal(states[:, candidate, :3], paired.states_context)
    np.testing.assert_array_equal(paired.costs, 1 - paired.coverage_future[:, :, -1])
    assert np.asarray(paired.metadata["audit"]["native_contact_points"]).max() > 0


@pytest.mark.parametrize("corruption", ["prefix", "nan_action", "action_bounds", "orientation", "request",
                                       "cost", "state_dtype", "pixel_shape", "time", "appearance",
                                       "source", "future_filter", "rejection_count", "extra_metadata"])
def test_strict_validation_rejects_malformed_roots(paired, corruption):
    root = deepcopy(paired)
    if corruption == "prefix":
        root.actions[1, 0, 0] = 0.1
    elif corruption == "nan_action":
        root.actions[1, -1, 0] = np.nan
    elif corruption == "action_bounds":
        root.actions[1, -1, 0] = 100
    elif corruption == "orientation":
        root.states_future[0, 0, 0, 4:] = 0
    elif corruption == "request":
        root.request[-1] += 1
    elif corruption == "cost":
        root.costs[0, 0] = 0.5 if root.costs[0, 0] != 0.5 else 0
    elif corruption == "state_dtype":
        root.states_context = root.states_context.astype(np.float32)
    elif corruption == "pixel_shape":
        root.context = root.context[:2]
    elif corruption == "time":
        root.metadata["frame_indices"][1] = 4
    elif corruption == "appearance":
        root.metadata["appearances"][1]["background_color"] = [0, 0, 0]
    elif corruption == "source":
        root.metadata["source"]["sha256"]["envs/pusht/env.py"] = "0" * 64
    elif corruption == "future_filter":
        root.metadata["sampling"]["future_filtering"] = True
    elif corruption == "rejection_count":
        root.metadata["sampling"]["rejected_anchors"] = 999
    else:
        root.metadata["surprise"] = True
    with pytest.raises(ValueError):
        validate_root(root)


@pytest.mark.parametrize("field", ["source", "runtime", "geometry", "sampling", "audit", "config",
                                   "recipes", "intervention", "appearances"])
def test_malformed_nested_metadata_raises_value_error(small_root, field):
    root = deepcopy(small_root)
    root.metadata[field] = None
    with pytest.raises(ValueError):
        validate_root(root)


def test_forged_geometry_and_missing_provenance_fields_fail(small_root):
    root = deepcopy(small_root)
    root.metadata["geometry"]["block_vertices"][0][0][0] += 1
    with pytest.raises(ValueError, match="geometry"):
        validate_root(root)
    root = deepcopy(small_root)
    del root.metadata["sampling"]["rng"]
    with pytest.raises(ValueError, match="sampling"):
        validate_root(root)


def test_npz_roundtrip_is_compressed_strict_and_pickle_free(paired, tmp_path):
    path = save_root(tmp_path / "root.npz", paired)
    with zipfile.ZipFile(path) as archive:
        assert all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist())
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {*ARRAY_KEYS, "metadata"}
        assert archive["metadata"].dtype.kind == "U" and archive["metadata"].shape == ()
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    loaded = load_root(path)
    assert loaded.metadata == paired.metadata
    for name in ARRAY_KEYS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(paired, name))
    with pytest.raises(FileExistsError):
        save_root(path, paired)
    assert not list(tmp_path.glob(".paired-*"))
    save_root(path, paired, overwrite=True)


@pytest.mark.parametrize("encoding", ["pickle", "duplicate", "nonfinite", "extra_array"])
def test_npz_reader_rejects_ambiguous_metadata(small_root, tmp_path, encoding):
    path = tmp_path / "bad.npz"
    arrays = small_root.arrays()
    encoded = json.dumps(small_root.metadata)
    if encoding == "pickle":
        metadata = np.array(small_root.metadata, dtype=object)
    elif encoding == "duplicate":
        metadata = np.array('{"schema":"duplicate",' + encoded[1:])
    elif encoding == "nonfinite":
        metadata = np.array(encoded.replace('"primitive_control_seconds": 0.1', '"primitive_control_seconds": NaN'))
    else:
        metadata = np.array(encoded)
        arrays["pixels"] = np.zeros(1, dtype=np.uint8)
    np.savez_compressed(path, **arrays, metadata=metadata)
    with pytest.raises(ValueError):
        load_root(path)


def test_stored_root_replay_and_appearance_qualification(small_root, simulator):
    report = qualify_root(small_root, simulator)
    assert report["passed"] and report["appearance_checked"]
    assert report["stored_arrays_exact"] and report["primitive_diagnostics_exact"]
    assert len(report["appearance_reports"]) == 2
    replayed = replay_root(small_root, simulator)
    assert replayed.metadata == small_root.metadata
    corrupted = deepcopy(small_root)
    corrupted.future[0, 0, 0, 0, 0, 0, 0] ^= np.uint8(1)
    validate_root(corrupted)  # shape validation is not claimed to prove replay
    with pytest.raises(QualificationError):
        qualify_root(corrupted, simulator, check_appearance=False)
    corrupted = deepcopy(small_root)
    corrupted.metadata["audit"]["diagnostic_sha256"][0][0] = "0" * 64
    with pytest.raises(QualificationError):
        qualify_root(corrupted, simulator, check_appearance=False)


def test_root_reconstruction_is_independent_of_generation_order(simulator):
    kwargs = dict(split_seed=12763, split="editor", config=GenerationConfig(horizon=1, candidates=1), simulator=simulator)
    first = generate_root(root_index=9, **kwargs)
    generate_root(root_index=2, **kwargs)
    repeated = generate_root(root_index=9, **kwargs)
    assert first.metadata == repeated.metadata
    for name in ARRAY_KEYS:
        np.testing.assert_array_equal(getattr(first, name), getattr(repeated, name))
    seed_before = first.metadata["stream_seeds"]["appearance"]
    different_request = generate_root(root_index=9, intervention=Intervention(), **kwargs)
    assert seed_before == different_request.metadata["stream_seeds"]["appearance"]
    assert first.metadata["appearances"] == different_request.metadata["appearances"]


def test_separate_split_rosters_and_duplicate_family_rejection(simulator, tmp_path):
    roster = generate_splits(tmp_path, counts={split: 1 for split in SPLITS},
                             config=GenerationConfig(horizon=1, candidates=1), simulator=simulator)
    paths = [path for paths in roster.values() for path in paths]
    roots = [load_root(path) for path in paths]
    validate_split_disjointness(roots)
    assert len({root.metadata["root_seed"] for root in roots}) == 5
    assert len({root.metadata["split_seed"] for root in roots}) == 5
    assert {root.metadata["split"] for root in roots} == set(SPLITS)
    assert all(path.parent.name == root.metadata["split"] for path, root in zip(paths, roots))
    with pytest.raises(ValueError):
        validate_split_disjointness([*roots, roots[0]])
    with pytest.raises(FileExistsError):
        generate_splits(tmp_path, counts={split: 1 for split in SPLITS},
                        config=GenerationConfig(horizon=1, candidates=1), simulator=simulator)


def test_split_seed_reuse_rejected_before_loading_simulator(tmp_path):
    counts = {split: 0 for split in SPLITS}
    with pytest.raises(ValueError, match="separate split seeds"):
        generate_splits(tmp_path, counts=counts, split_seeds={split: 42 for split in SPLITS})
    assert generate_splits(tmp_path, counts=counts) == {split: [] for split in SPLITS}
    with pytest.raises(ValueError):
        generate_splits(tmp_path, counts={"pilot": 1})


def test_invalid_explicit_anchor_is_not_resampled(simulator):
    with pytest.raises(InvalidAnchor, match="Explicit"):
        generate_root(split_seed=7193, simulator=simulator,
                      config=GenerationConfig(horizon=1, candidates=1),
                      anchor=Anchor((256, 230), (256, 220), 0), goal=GOAL,
                      intervention=Intervention())