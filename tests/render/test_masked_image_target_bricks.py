"""``target_bricks="surface"``: load only the bricks near the surface (8.3).

Masked image design v2, Phase 9.  The read counts are checked against design
3.1's coarse pass, frozen in ``masked_surface_counts.json`` by
``scripts/masked_image/phase9_surface_counts.py`` (never re-frozen to make a
test pass).  The scenes are the probe's: a 128 cubed EDT ellipsoid in 16-voxel
bricks over a four-level pyramid, so the backstop is the 8x level and a coarse
cell is half a brick, as 3.1's 16x pass over 32-voxel bricks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from cellier.render.block_cache._surface_map import (
    MAX_SLICES,
    SurfaceMap,
    crossing_cells,
    mask_bricks_near_surface,
    value_bricks_near_surface,
)
from cellier.visuals import MaskedImageAppearance, MaskedImageSingleAppearance
from tests.render._affine import render
from tests.render._masked import (
    gfx_visual,
    load,
    masked_config,
    new_viewer,
    pyramid_store,
    record_targets,
    spiked_pair,
    store,
)

EXPECTED = json.loads(
    (Path(__file__).parent / "masked_surface_counts.json").read_text()
)
SIDE = 128
N_LEVELS = 4
#: (spike, iso, thickness) of each frozen case.
CASES = {
    "ellipsoid_iso1_t10": (False, 1.0, 10.0),
    "ellipsoid_iso10_t10": (False, 10.0, 10.0),
    "spike_iso1_t10": (True, 1.0, 10.0),
}
BUDGET = 48 * 1024**2


# ---------------------------------------------------------------------------
# The map and the filter, without a viewer
# ---------------------------------------------------------------------------


def _map(shape=(8, 8, 8), block=4, overlap=1) -> SurfaceMap:
    surface_map = SurfaceMap([(16, 16, 16), shape], block, overlap)
    surface_map.set_level(2)
    return surface_map


def _block(value, block=4, overlap=1) -> np.ndarray:
    """A padded block whose inner voxels are *value* and halo is -1."""
    side = block + 2 * overlap
    data = np.full((side, side, side), -1.0, np.float32)
    data[overlap:-overlap, overlap:-overlap, overlap:-overlap] = value
    return data


def test_the_map_copies_bricks_of_its_level_only():
    surface_map = _map()
    surface_map.record("t0", 1, (0, 0, 0), _block(5.0))
    assert surface_map.crossing("t0", 0.0) is None
    inner = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    surface_map.record("t0", 2, (1, 0, 1), _block(inner))

    values, known = surface_map._slices["t0"]
    assert np.array_equal(values[4:8, 0:4, 4:8], inner)
    assert known.sum() == 64
    assert known[4:8, 0:4, 4:8].all()


def test_a_brick_past_the_level_end_is_cut_to_it():
    surface_map = SurfaceMap([(6, 6, 6)], 4, 1)
    surface_map.set_level(1)
    surface_map.record("t0", 1, (1, 1, 1), _block(2.0))
    _, known = surface_map._slices["t0"]
    assert known.sum() == 8
    assert known[4:6, 4:6, 4:6].all()


def test_ready_once_every_cell_is_known_or_the_backstop_settles():
    surface_map = _map(shape=(4, 4, 8))
    surface_map.record("t0", 2, (0, 0, 0), _block(1.0))
    assert not surface_map.ready("t0")
    surface_map.record("t0", 2, (0, 0, 1), _block(1.0))
    assert surface_map.ready("t0")

    assert not surface_map.ready("t1")
    surface_map.settle("t1")
    assert surface_map.ready("t1")
    # Settled with nothing written: there is nothing to filter by.
    assert surface_map.crossing("t1", 0.0) is None


def test_a_new_level_forgets_every_slice():
    surface_map = _map()
    surface_map.record("t0", 2, (0, 0, 0), _block(1.0))
    surface_map.settle("t0")
    surface_map.set_level(2)
    assert surface_map.ready("t0")
    surface_map.set_level(1)
    assert not surface_map.ready("t0")
    assert surface_map._slices == {}


def test_the_map_keeps_its_newest_slices():
    surface_map = _map()
    for t in range(MAX_SLICES + 2):
        surface_map.record(t, 2, (0, 0, 0), _block(1.0))
        surface_map.settle(t)
    assert list(surface_map._slices) == list(range(2, MAX_SLICES + 2))
    assert not surface_map.ready(0)
    assert surface_map.ready(MAX_SLICES + 1)


def test_a_cell_crosses_when_its_neighbourhood_straddles_the_threshold():
    values = np.zeros((6, 6, 6), np.float32)
    values[:, :, 3:] = 10.0
    known = np.ones(values.shape, bool)
    crossing = crossing_cells(values, known, 5.0)
    # The step is between x = 2 and 3: those columns and no other.
    assert crossing[:, :, 2:4].all()
    assert not crossing[:, :, :2].any() and not crossing[:, :, 4:].any()
    # A threshold no cell reaches crosses nowhere.
    assert not crossing_cells(values, known, 20.0).any()


def test_an_unknown_cell_crosses_and_so_do_its_neighbours():
    values = np.zeros((6, 6, 6), np.float32)
    known = np.ones(values.shape, bool)
    known[3, 3, 3] = False
    crossing = crossing_cells(values, known, 5.0)
    assert crossing[2:5, 2:5, 2:5].all()
    assert crossing.sum() == 27


def _coarse(shape, cells) -> np.ndarray:
    crossing = np.zeros(shape, bool)
    for cell in cells:
        crossing[cell] = True
    return crossing


#: A two-level pyramid, level 1 the 2x mean: coarse voxel c sits at level-0
#: 2c + 0.5.
LEVELS = {
    "level_shapes": [(16, 16, 16), (8, 8, 8)],
    "scales": [np.ones(3), np.full(3, 2.0)],
    "translations": [np.zeros(3), np.full(3, 0.5)],
}


def test_a_brick_keeps_only_when_it_covers_a_crossing_cell():
    # 4-voxel level-0 bricks over 2-voxel coarse cells: brick g covers cells
    # 2g and 2g + 1 exactly, and no more.
    arr = np.array([[1, 0, 0, g] for g in range(4)], np.int64)
    crossing = _coarse((8, 8, 8), [(0, 0, 4)])
    kept = mask_bricks_near_surface(
        arr, crossing, block_size=4, coarse_level=2, **LEVELS
    )
    assert kept.tolist() == [[1, 0, 0, 2]]
    crossing = _coarse((8, 8, 8), [(0, 0, 3)])
    kept = mask_bricks_near_surface(
        arr, crossing, block_size=4, coarse_level=2, **LEVELS
    )
    assert kept.tolist() == [[1, 0, 0, 1]]


def test_a_coarse_brick_covers_its_own_extent():
    # A level-2 brick of 4 coarse voxels (8 level-0) spans cells 0 to 3.
    arr = np.array([[2, 0, 0, 0], [2, 0, 0, 1]], np.int64)
    crossing = _coarse((8, 8, 8), [(1, 1, 3)])
    kept = mask_bricks_near_surface(
        arr, crossing, block_size=4, coarse_level=2, **LEVELS
    )
    assert kept.tolist() == [[2, 0, 0, 0]]


def _value_kept(arr, crossing, value_to_mask, reach) -> list:
    return value_bricks_near_surface(
        arr,
        crossing,
        block_size=4,
        value_to_mask=value_to_mask,
        mask_scale=LEVELS["scales"][1],
        mask_translation=LEVELS["translations"][1],
        reach_voxels=np.full(3, float(reach)),
        **LEVELS,
    ).tolist()


def test_a_value_brick_keeps_within_reach_of_a_crossing_cell():
    arr = np.array([[1, 0, 0, g] for g in range(4)], np.int64)
    crossing = _coarse((8, 8, 8), [(0, 0, 4)])  # level-0 x 8 to 9
    identity = np.eye(4)
    assert _value_kept(arr, crossing, identity, 0) == [[1, 0, 0, 2]]
    # 2 voxels reach the bricks either side, 0.5 voxels short of a third.
    assert _value_kept(arr, crossing, identity, 2) == [
        [1, 0, 0, 1],
        [1, 0, 0, 2],
    ]
    assert len(_value_kept(arr, crossing, identity, 6)) == 4


def test_a_value_brick_maps_through_value_to_mask():
    # The value at twice the mask's resolution: value voxel v sits at mask
    # (v + 0.5) / 2 - 0.5, so value brick g spans mask x 2g - 0.5 to 2g + 1.5.
    matrix = np.eye(4)
    matrix[:3, :3] *= 0.5
    matrix[:3, 3] = -0.25
    arr = np.array([[1, 0, 0, g] for g in range(8)], np.int64)
    crossing = _coarse((8, 8, 8), [(0, 0, 2)])  # mask x 4 to 5
    assert _value_kept(arr, crossing, matrix, 0) == [[1, 0, 0, 2]]


# ---------------------------------------------------------------------------
# In the viewer
# ---------------------------------------------------------------------------


def _viewer(tmp_path, *, spike=False, target="surface", dim="3d", **single):
    """A loaded viewer of the probe's scene; ``(viewer, model, plans, stores)``."""
    mask, value = spiked_pair(SIDE, spike)
    viewer = new_viewer(dim)
    stores = (
        pyramid_store(tmp_path, "mask", mask, N_LEVELS),
        pyramid_store(tmp_path, "value", value, N_LEVELS),
    )
    model = viewer.controller.add_masked_image(
        *stores,
        viewer.scene.id,
        appearance=MaskedImageAppearance(
            target_bricks=target,
            force_level=1,
            value_force_level=1,
            frustum_cull=False,
        ),
        single=MaskedImageSingleAppearance(**single),
        render_config=masked_config(
            gpu_budget_bytes=BUDGET, value_gpu_budget_bytes=BUDGET
        ),
    )
    plans = record_targets(gfx_visual(viewer, model.id))
    load(viewer, size=(256, 256), slice_z=SIDE / 2 if dim == "2d" else None)
    return viewer, model, plans, stores


def _grids(rows) -> set[tuple[int, ...]]:
    return {tuple(int(v) for v in row[1:]) for row in rows if row[0] == 1}


def _truth(case: str, side: str) -> set[tuple[int, ...]]:
    return {tuple(g) for g in EXPECTED[case][f"truth_{side}"]}


@pytest.mark.parametrize("case", list(CASES))
def test_read_counts_match_the_coarse_pass_of_3_1(tmp_path, case):
    """Each side keeps 3.1's bricks; on the ellipsoid it misses none needed."""
    spike, iso, thickness = CASES[case]
    viewer, _model, plans, _ = _viewer(
        tmp_path, spike=spike, iso_threshold=iso, surface_thickness=thickness
    )
    try:
        mask, value = _grids(plans["mask"][-1]), _grids(plans["value"][-1])
        expected = EXPECTED[case]
        assert len(mask) == expected["plan_b_mask"]
        assert len(value) == expected["plan_b_value"]
        assert len(mask) < expected["total"]
        if not spike:
            assert _truth(case, "mask") <= mask
            assert _truth(case, "value") <= value
    finally:
        viewer.controller.close()


def test_a_thin_spike_is_never_loaded_under_surface(tmp_path):
    """Design 3.1: the spike's coarse mean is below the threshold."""
    spike_bricks = {tuple(g) for g in EXPECTED["spike_iso1_t10"]["truth_spike"]}
    missed = {}
    for target in ("visible", "surface"):
        viewer, _model, plans, _ = _viewer(
            tmp_path / target,
            spike=True,
            target=target,
            iso_threshold=1.0,
            surface_thickness=10.0,
        )
        try:
            missed[target] = len(spike_bricks - _grids(plans["mask"][-1]))
        finally:
            viewer.controller.close()
    assert missed["visible"] == 0
    assert missed["surface"] == EXPECTED["spike_iso1_t10"]["plan_b_spike_missed"]
    assert missed["surface"] > 0


def test_visible_keeps_every_brick_and_never_waits(tmp_path):
    viewer, model, plans, _ = _viewer(tmp_path, target="visible", iso_threshold=1.0)
    try:
        total = EXPECTED["ellipsoid_iso1_t10"]["total"]
        assert plans["mask"] and plans["value"]
        assert all(plan is not None and len(plan) == total for plan in plans["mask"])
        assert all(plan is not None and len(plan) == total for plan in plans["value"])
        coordinator = viewer.controller._render_manager._slice_coordinator
        assert not coordinator.visual_progress(model.id).target_deferred
    finally:
        viewer.controller.close()


def test_the_target_waits_for_the_coarse_mask_then_plans(tmp_path):
    """The first full plan holds the target back; the backstop re-plans it."""
    from tests.render._masked import delay

    mask, value = spiked_pair(SIDE, False)
    viewer = new_viewer("3d")
    stores = (
        pyramid_store(tmp_path, "mask", mask, N_LEVELS),
        pyramid_store(tmp_path, "value", value, N_LEVELS),
    )
    delay(stores[0], 0.02)
    try:
        model = viewer.controller.add_masked_image(
            *stores,
            viewer.scene.id,
            appearance=MaskedImageAppearance(target_bricks="surface", force_level=1),
            single=MaskedImageSingleAppearance(iso_threshold=1.0),
            render_config=masked_config(
                gpu_budget_bytes=BUDGET, value_gpu_budget_bytes=BUDGET
            ),
        )
        gfx = gfx_visual(viewer, model.id)
        plans = record_targets(gfx)
        coordinator = viewer.controller._render_manager._slice_coordinator
        deferred_at_settle = []
        settle = gfx.settle_surface

        def watched():
            progress = coordinator.visual_progress(model.id)
            deferred_at_settle.append(progress.target_deferred)
            return settle()

        gfx.settle_surface = watched
        load(viewer, size=(256, 256))

        assert plans["mask"][0] is None and plans["value"][0] is None
        assert deferred_at_settle[0] is True
        assert plans["mask"][-1] is not None and len(plans["mask"][-1]) > 0
        assert not gfx.target_deferred
        assert not coordinator.visual_progress(model.id).target_deferred
        assert coordinator.visual_progress(model.id).complete
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("kind", ["memory", "lazy"])
def test_surface_draws_what_visible_draws(tmp_path, kind):
    """The filter keeps every brick the picture needs (MI-D3: both stores)."""
    side = 64
    mask, value = spiked_pair(side, False)
    counts = {}
    for target in ("visible", "surface"):
        root = tmp_path / target
        viewer = new_viewer("3d")
        try:
            model = viewer.controller.add_masked_image(
                store(kind, root, "mask", mask),
                store(kind, root, "value", value),
                viewer.scene.id,
                appearance=MaskedImageAppearance(target_bricks=target),
                single=MaskedImageSingleAppearance(
                    iso_threshold=4.0, surface_thickness=4.0, clim=(0.0, 4.0)
                ),
                render_config=masked_config(),
            )
            plans = record_targets(gfx_visual(viewer, model.id))
            load(viewer, size=(256, 256))
            frame = render(viewer, 256)[0][..., :3].astype(np.int64)
            counts[target] = (frame, plans)
        finally:
            viewer.controller.close()
    visible, surface = counts["visible"][0], counts["surface"][0]
    drawn = int((visible.max(axis=-1) > 8).sum())
    assert drawn > 2000
    # A brick the surface needs and lacks costs hundreds of pixels.  What
    # differs is the silhouette's edge, by a few codes: a ray's step past a
    # kept brick reads the coarse backstop where "visible" has the target
    # level (the LOD difference).  Measured: 0 in memory, 8 of 3,825 lazy.
    differ = np.abs(visible - surface).max(axis=-1) > 8
    assert differ.sum() <= 0.005 * drawn, differ.sum()
    # It kept fewer bricks than "visible" did.
    kept = counts["surface"][1]["mask"][-1]
    assert len(kept) < len(counts["visible"][1]["mask"][-1])


def _backstop_reads(data_store) -> list:
    """Record each read of *data_store* at its coarsest level."""
    original = data_store.get_data
    reads = []

    async def counted(request):
        if int(getattr(request, "scale_index", -1)) == N_LEVELS - 1:
            reads.append(request)
        return await original(request)

    object.__setattr__(data_store, "get_data", counted)
    return reads


def test_a_threshold_change_re_plans_without_reading_the_coarse_level(tmp_path):
    viewer, model, plans, stores = _viewer(tmp_path, iso_threshold=1.0)
    try:
        reads = _backstop_reads(stores[0])
        n_plans = len(plans["mask"])
        viewer.controller.update_single_appearance_field(
            model.id, "iso_threshold", 10.0
        )
        load(viewer, size=(256, 256))
        after = plans["mask"][n_plans:]
        assert after and all(plan is not None for plan in after)
        assert len(_grids(after[-1])) == EXPECTED["ellipsoid_iso10_t10"]["plan_b_mask"]
        assert reads == []
    finally:
        viewer.controller.close()


def test_a_thickness_change_re_plans_the_value(tmp_path):
    viewer, model, plans, _ = _viewer(
        tmp_path, iso_threshold=1.0, surface_thickness=10.0
    )
    try:
        wide = len(_grids(plans["value"][-1]))
        mask = len(_grids(plans["mask"][-1]))
        viewer.controller.update_single_appearance_field(
            model.id, "surface_thickness", 0.0
        )
        load(viewer, size=(256, 256))
        assert len(_grids(plans["value"][-1])) < wide
        assert len(_grids(plans["mask"][-1])) == mask
    finally:
        viewer.controller.close()


def test_a_drag_re_plans_at_most_the_capped_rate(tmp_path, monkeypatch):
    """The first change re-plans at once, the rest fold into one trailing one."""
    viewer, model, _plans, _ = _viewer(tmp_path, iso_threshold=1.0)
    controller = viewer.controller
    calls = []
    monkeypatch.setattr(controller, "reslice_visual", calls.append)
    # After a quiet interval: the load's own re-plan counts toward the cap.
    controller._surface_replan_at.clear()

    async def drag():
        for iso in (2.0, 3.0, 4.0, 5.0, 6.0):
            controller.update_single_appearance_field(model.id, "iso_threshold", iso)
        assert len(calls) == 1
        await asyncio.gather(*controller._deferred_reslice_tasks())

    try:
        asyncio.run(drag())
        assert calls == [model.id, model.id]
    finally:
        controller.close()


def test_a_2d_view_is_unaffected(tmp_path):
    """Design 4.1: the 2D slice node plans no filter and never waits."""
    viewer, model, plans, _ = _viewer(tmp_path, dim="2d", iso_threshold=1.0)
    try:
        assert plans["mask"] == [] and plans["value"] == []
        gfx = gfx_visual(viewer, model.id)
        assert not gfx.target_deferred
        calls = []
        viewer.controller.reslice_visual = calls.append
        viewer.controller.update_single_appearance_field(model.id, "iso_threshold", 4.0)
        assert calls == []
    finally:
        viewer.controller.close()


def test_switching_target_bricks_re_plans(tmp_path):
    viewer, model, plans, _ = _viewer(tmp_path, target="visible", iso_threshold=1.0)
    try:
        total = EXPECTED["ellipsoid_iso1_t10"]["total"]
        viewer.controller.update_appearance_field(model.id, "target_bricks", "surface")
        load(viewer, size=(256, 256))
        assert len(plans["mask"][-1]) == EXPECTED["ellipsoid_iso1_t10"]["plan_b_mask"]
        viewer.controller.update_appearance_field(model.id, "target_bricks", "visible")
        load(viewer, size=(256, 256))
        assert len(plans["mask"][-1]) == total
    finally:
        viewer.controller.close()


def test_value_force_level_re_plans(tmp_path):
    viewer, model, plans, _ = _viewer(tmp_path, target="visible")
    try:
        assert set(plans["value"][-1][:, 0].tolist()) == {1}
        viewer.controller.update_appearance_field(model.id, "value_force_level", 2)
        load(viewer, size=(256, 256))
        assert set(plans["value"][-1][:, 0].tolist()) == {2}
        assert set(plans["mask"][-1][:, 0].tolist()) == {1}
    finally:
        viewer.controller.close()


def test_a_mask_store_change_forgets_the_coarse_mask_and_waits_again(tmp_path):
    viewer, _model, plans, stores = _viewer(tmp_path, iso_threshold=1.0)
    try:
        n_plans = len(plans["mask"])
        stores[0].notify_changed("contents")
        load(viewer, size=(256, 256))
        after = plans["mask"][n_plans:]
        assert after[0] is None
        assert after[-1] is not None
        assert len(_grids(after[-1])) == EXPECTED["ellipsoid_iso1_t10"]["plan_b_mask"]
    finally:
        viewer.controller.close()


def test_close_drops_the_value_sides_filter(tmp_path):
    viewer, model, _plans, _ = _viewer(tmp_path, target="visible")
    value = gfx_visual(viewer, model.id).value_side
    viewer.controller.close()
    assert value.target_filter is None


def test_a_time_step_waits_for_its_own_coarse_mask(tmp_path):
    """Each time point plans from its own coarse level (design 8.3).

    Time 1 is time 0 moved 16 voxels (one brick) along +x, so its kept mask
    bricks are time 0's moved one brick.
    """
    from tests.render._masked import lazy_store

    mask, value = spiked_pair(SIDE, False)
    shift = 16
    masks = np.stack([mask, np.roll(mask, shift, axis=2)])
    values = np.stack([value, np.roll(value, shift, axis=2)])
    viewer = new_viewer("3d", time=True)
    try:
        model = viewer.controller.add_masked_image(
            lazy_store(tmp_path, "mask", masks),
            lazy_store(tmp_path, "value", values),
            viewer.scene.id,
            appearance=MaskedImageAppearance(
                target_bricks="surface", force_level=1, frustum_cull=False
            ),
            single=MaskedImageSingleAppearance(iso_threshold=1.0),
            render_config=masked_config(
                gpu_budget_bytes=BUDGET, value_gpu_budget_bytes=BUDGET
            ),
        )
        plans = record_targets(gfx_visual(viewer, model.id))
        load(viewer, size=(256, 256))
        first = _grids(plans["mask"][-1])
        n_plans = len(plans["mask"])

        async def step() -> None:
            from cellier.convenience.capture import _load_data

            viewer.set_slice_positions({0: 1.0})
            await _load_data(viewer)

        asyncio.run(step())
        after = plans["mask"][n_plans:]
        assert after[0] is None
        moved = _grids(after[-1])
        assert moved == {(z, y, x + 1) for z, y, x in first}
    finally:
        viewer.controller.close()
