"""Intermediate loading stages: the ancestors of a target plan.

``plans/progressive_intermediate_level_design_v2.md``, sections 3 and 6.
The expected ancestors are stored in ``intermediate_bricks_cases.json``
(reviewed with the design; never regenerated), not recomputed here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cellier.render._backstop import intermediate_bricks
from cellier.render.block_cache import BlockCache3D, compute_block_cache_parameters_3d
from cellier.render.block_cache._image_residency import ImageResidency3D, unpack_keys
from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.render.scheduling import DEAD, ChunkClass, Tier
from cellier.render.visuals._chunked import (
    desired_bricks,
    log_intermediate_cap_once,
    target_room,
)
from cellier.visuals import MultiscaleImageSingleAppearance, ProgressiveLoadingConfig
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from tests import _plane_fixtures as fx
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading
from tests.render.test_backstop import _Planner

_DOCUMENT = json.loads(
    (Path(__file__).parent / "intermediate_bricks_cases.json").read_text()
)
_CASES = _DOCUMENT["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_the_ancestors_are_the_stored_ones(case) -> None:
    pyramid = _DOCUMENT["pyramids"][case["pyramid"]]
    ndim = len(pyramid["grid"])
    backstop_level = case["backstop_level"] or len(pyramid["spans"])

    got = intermediate_bricks(
        np.asarray(case["target"], dtype=np.int64).reshape(-1, 1 + ndim),
        case["offset"],
        backstop_level,
        pyramid["grid"],
        pyramid["spans"],
        pyramid["counts"],
    )

    assert got.dtype == np.int64
    assert got.shape == (len(case["expected"]), 1 + ndim)
    assert got.tolist() == case["expected"]


# -- the planner tail -------------------------------------------------------------

BLOCK = 4
SHAPES = [(32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)]
SCALES = np.array([[1.0] * 3, [2.0] * 3, [4.0] * 3, [8.0] * 3])
BACKSTOP = np.array([[4, 0, 0, 0]])


def _residency(n_bricks: int) -> ImageResidency3D:
    """An atlas of about *n_bricks* slots over a four-level 8^3-brick pyramid."""
    params = compute_block_cache_parameters_3d(
        block_size=BLOCK,
        gpu_budget_bytes=n_bricks * (BLOCK + 2) ** 3 * 4,
        overlap=1,
    )
    lut = LutIndirectionManager3D(
        BlockLayout3D(volume_shape=SHAPES[0], block_size=BLOCK),
        n_levels=4,
        level_scale_vecs_data=list(SCALES),
        level_shapes=SHAPES,
    )
    return ImageResidency3D(BlockCache3D(params), lut, BLOCK, SCALES, np.zeros((4, 3)))


def _box(level: int, side: int) -> np.ndarray:
    """Rows of the ``side^3`` bricks at the origin of *level*, first axis slowest."""
    g = np.stack(np.unravel_index(np.arange(side**3), (side,) * 3), -1)
    return np.column_stack([np.full(len(g), level), g])


def _loading(residency, offsets, stage_slots: float | None = None):
    """A config whose stage cap is *stage_slots* slots (the default if None)."""
    if stage_slots is None:
        return ProgressiveLoadingConfig(intermediate_offsets=offsets)
    return ProgressiveLoadingConfig(
        intermediate_offsets=offsets,
        intermediate_max_slot_fraction=(stage_slots + 0.5) / residency.n_slots,
    )


def _plan(residency, target, loading, backstop=BACKSTOP):
    return desired_bricks(
        _Planner(),
        residency,
        target,
        backstop_arr=backstop,
        backstop_cap=5,
        loading=loading,
    )


def _levels(ds) -> list[int]:
    return unpack_keys(ds.keys)[0].tolist()


def test_the_blocks_are_backstop_then_stages_coarse_to_fine_then_target() -> None:
    residency = _residency(400)
    target = _box(1, 4)[::-1]  # the far corner first
    ds = _plan(residency, target, _loading(residency, (2, 1), stage_slots=40))

    assert ds.cls.tolist() == [ChunkClass.BACKSTOP] + [2] * 1 + [1] * 8 + [0] * 64
    assert _levels(ds) == [4] + [3] * 1 + [2] * 8 + [1] * 64
    # Each stage is in the target's order: the first brick's ancestor first.
    _, _, grids = unpack_keys(ds.keys)
    assert grids[2].tolist() == [1, 1, 1]
    assert ds.n_truncated_intermediate == ds.n_truncated_target == 0


def test_no_offset_is_the_plan_of_before() -> None:
    residency = _residency(400)
    target = _box(1, 4)
    without = desired_bricks(
        _Planner(), residency, target, backstop_arr=BACKSTOP, backstop_cap=5
    )
    for loading in (None, ProgressiveLoadingConfig()):
        ds = _plan(residency, target, loading)
        assert ds.keys.tolist() == without.keys.tolist()
        assert ds.cls.tolist() == without.cls.tolist()


def test_one_cap_holds_for_all_the_stages_coarsest_first() -> None:
    residency = _residency(400)
    target = _box(1, 8)  # every brick: 8 ancestors at offset 2, 64 at offset 1
    ds = _plan(residency, target, _loading(residency, (2, 1), stage_slots=20))

    assert int((ds.cls == 2).sum()) == 8  # the coarser stage is kept whole
    assert int((ds.cls == 1).sum()) == 12  # the finer one gets the rest
    assert ds.n_truncated_intermediate == 64 - 12
    # Truncated to its nearest: the first twelve ancestors in target order.
    _, _, grids = unpack_keys(ds.keys[ds.cls == 1])
    assert grids.tolist() == _box(2, 4)[:12, 1:].tolist()


def test_a_key_planned_in_two_blocks_keeps_the_higher_class() -> None:
    residency = _residency(400)
    # A mixed-level plan: the level-2 brick is also the level-1 bricks' ancestor.
    target = np.concatenate([_box(1, 2), [[2, 0, 0, 0]], [[2, 3, 3, 3]]])
    ds = _plan(residency, target, _loading(residency, (1,)))

    assert len(np.unique(ds.keys)) == len(ds.keys)
    # The stage: [2, 0, 0, 0] for the level-1 bricks, then the two level-2
    # bricks' own ancestors.  [2, 0, 0, 0] is read once, as a stage brick;
    # [2, 3, 3, 3] is nobody's ancestor and stays a target brick.
    assert ds.cls.tolist() == [ChunkClass.BACKSTOP, 1, 1, 1] + [0] * 9
    assert _levels(ds) == [4, 2, 3, 3] + [1] * 8 + [2]


def test_the_target_gets_the_room_actually_left() -> None:
    residency = _residency(60)
    budget = residency.n_slots - 1
    target = _box(1, 8)
    ds = _plan(residency, target, _loading(residency, (2,), stage_slots=10))

    # The stage is the ancestors of the whole target, cut bricks included.
    assert int((ds.cls == 2).sum()) == 8
    assert len(ds.keys) == budget
    n_target = int((ds.cls == ChunkClass.TARGET).sum())
    assert n_target == budget - 1 - 8  # not budget less the caps
    assert ds.n_truncated_target == 512 - n_target


def test_backstop_only_plans_no_stage() -> None:
    residency = _residency(400)
    ds = _plan(residency, None, _loading(residency, (2, 1)))
    assert ds.cls.tolist() == [ChunkClass.BACKSTOP]


def test_no_stage_at_the_backstop_level_even_without_a_backstop_array() -> None:
    residency = _residency(400)
    ds = _plan(residency, _box(2, 2), _loading(residency, (2,)), backstop=None)
    assert (ds.cls == ChunkClass.TARGET).all()
    # A backstop level set below the coarsest moves the limit with it.
    loading = ProgressiveLoadingConfig(intermediate_offsets=(1,), backstop_level=3)
    ds = _plan(residency, _box(2, 2), loading, backstop=None)
    assert (ds.cls == ChunkClass.TARGET).all()


def test_target_room_subtracts_a_two_dimensional_estimate() -> None:
    residency = _residency(400)
    budget = residency.n_slots - 1
    room = budget - 1  # one backstop brick

    def asked(**loading) -> int:
        return target_room(residency, BACKSTOP, ProgressiveLoadingConfig(**loading))

    assert asked() == room
    assert asked(intermediate_offsets=(2,)) == room - room // 16
    assert asked(intermediate_offsets=(3, 2)) == room - room // 64 - room // 16
    # Limited by the stages' cap.
    cap = int(0.1 * residency.n_slots)
    assert room // 4 > cap
    assert asked(intermediate_offsets=(1,)) == room - cap


def test_the_stage_cap_is_logged_once_per_visual(caplog) -> None:
    residency = _residency(400)
    ds = _plan(residency, _box(1, 8), _loading(residency, (1,), stage_slots=5))
    owner = SimpleNamespace(visual_model_id="v")
    with caplog.at_level(logging.INFO, logger="cellier.render.cache"):
        log_intermediate_cap_once(owner, [ds])
        log_intermediate_cap_once(owner, [ds])
    lines = [r.getMessage() for r in caplog.records if "intermediate_capped" in r.msg]
    assert lines == [
        "intermediate_capped  visual=v  kept=5  dropped=59  (the intermediate "
        "stages outgrew intermediate_max_slot_fraction; use a larger offset or "
        "fewer stages, or raise the fraction)"
    ]


# -- through a controller -------------------------------------------------------------


def _wanted_classes(controller, gfx, dim) -> dict[int, list[int]]:
    """``class -> levels`` of the bricks the latest plan wants."""
    slot = gfx.slots[0]
    residency = slot.residency_3d() if dim == "3d" else slot.residency_2d()
    registry = controller._render_manager.scheduler.core.registry(residency.cache_id)
    wanted = (registry.tier == Tier.VISIBLE) & (registry.state != DEAD)
    levels = unpack_keys(registry.key[wanted])[0]
    return {
        int(cls): sorted(set(levels[registry.cls[wanted] == cls].tolist()))
        for cls in np.unique(registry.cls[wanted])
    }


@pytest.mark.parametrize("dim", ["3d", "2d"])
async def test_a_stage_is_planned_and_a_removed_offset_stops_being_wanted(
    controller, pyramid_root, dim
):
    """Levels 1 to 3 of 128^3 in 16-voxel bricks: target 1, stage 2, backstop 3."""
    store, _voxel_size = fx.open_pyramid(pyramid_root(fx.ISO), fx.ISO)
    scene = controller.add_scene(
        displayed_axes=("z", "y", "x") if dim == "3d" else ("y", "x"), name="scene"
    )
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(
            block_size=16,
            loading=ProgressiveLoadingConfig(intermediate_offsets=(1,)),
            **SMALL_BUDGETS,
        ),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    assert _wanted_classes(controller, gfx, dim) == {
        int(ChunkClass.TARGET): [1],
        1: [2],
        int(ChunkClass.BACKSTOP): [3],
    }
    n_stage = gfx.slots[0]._last_plan_stats["n_intermediate"]
    assert n_stage > 0
    progress = controller.loading_progress(visual.id)
    assert progress.intermediate == ((1, n_stage, n_stage),)
    assert progress.needed_target == gfx.slots[0]._last_plan_stats["n_target"]
    assert progress.complete and progress.stage is None

    # The runtime switch: a loading-only change replans by itself.
    controller.set_loading_config(visual.id, intermediate_offsets=())
    await drain_loading(controller)

    assert _wanted_classes(controller, gfx, dim) == {
        int(ChunkClass.TARGET): [1],
        int(ChunkClass.BACKSTOP): [3],
    }
    assert gfx.slots[0]._last_plan_stats["n_intermediate"] == 0
    assert controller.loading_progress(visual.id).intermediate == ()
