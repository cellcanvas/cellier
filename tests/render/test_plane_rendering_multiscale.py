"""Multiscale plane rendering: the right voxel, of the right level.

Plane rendering design v3, sections 5 and 6, and 14.8 (Phase 7).  Each
pixel test draws a multiscale image or labels visual in ``"plane"`` render
mode and compares the frame with the CPU reference of
``tests/render/_planes.py``.  The planner's own tests (culling rows, the
level rule, the cache) are in ``test_plane_planning.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.scheduling import Tier
from tests.render import _planes as h
from tests.render._plane_rig import (
    ANISO,
    BLOCK,
    ISO,
    POSES,
    SPECS,
    UNALIGNED,
    Rig,
    assert_matches,
    centre_of,
    compare_level,
    level_values,
    level_voxel,
    make_rig,
)
from tests.render.conftest import drain_loading

# -- the right voxel -----------------------------------------------------------


#: The pyramid every pose is drawn on: the one that never downsamples z,
#: where a plane along an axis and the anisotropy meet.
EVERY_POSE = "aniso"


def _level_cases() -> list:
    """Every pyramid with the oblique plane; one pyramid with every pose.

    The pyramids differ in how a level's voxels map to level 0, which is
    what this test is about, so each is drawn at every level, through the
    oblique plane: the pose that crosses every axis.  The two poses along an
    axis are drawn on `EVERY_POSE` only.  The full cross was 36 cases.
    """
    cases = [(level, "oblique", name) for name in SPECS for level in (0, 1, 2)]
    cases += [
        (level, pose, EVERY_POSE)
        for pose in POSES
        if pose != "oblique"
        for level in (0, 1, 2)
    ]
    return [
        pytest.param(level, pose, name, id=f"{level}-{pose}-{name}")
        for level, pose, name in cases
    ]


@pytest.mark.parametrize(("level", "pose", "spec_name"), _level_cases())
async def test_an_image_plane_shows_the_voxel_of_the_level_drawn(
    controller, tmp_path, spec_name, pose, level
):
    """Every pixel of the plane shows the forced level's voxel: the right
    voxel, and never the backstop's, so every brick the plane is drawn from
    was planned.  On an integer pyramid, one that never downsamples z, one
    with non-integer level ratios and one whose levels are not aligned with
    the lookup table's cells."""
    spec = SPECS[spec_name]
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES[pose])],
        appearance={"force_level": level + 1},
    )
    shot = await rig.settle()
    reference = rig.trace(shot)
    counts = compare_level(rig, shot, reference, level, level_values(spec, level))
    assert_matches(counts, least=1500)


@pytest.mark.parametrize("pose", list(POSES))
@pytest.mark.parametrize("level", [0, 1, 2])
async def test_a_label_plane_shows_the_label_of_the_level_drawn(
    controller, tmp_path, pose, level
):
    """Labels (14.8): label, background and level right at three levels."""
    spec = ANISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES[pose])],
        labels=True,
        appearance={"force_level": level + 1},
    )
    shot = await rig.settle()
    labels = level_values(spec, level)
    voxel, clear = level_voxel(spec, level, rig.trace(shot).data_position)
    # Background is decided by the level's own voxel grid.
    is_background = labels[voxel[..., 0], voxel[..., 1], voxel[..., 2]] == 0
    drawn_where_label = rig.trace(shot)
    sure = drawn_where_label.sure & clear
    hit = drawn_where_label.hit
    expected = h.srgb_encode(
        h.grey_of(labels[voxel[..., 0], voxel[..., 1], voxel[..., 2]])
    )
    red = shot.frame[..., 0].astype(float)
    label_pixels = sure & hit & ~is_background
    assert label_pixels.sum() > 1200
    assert shot.drawn[label_pixels].all()
    assert (np.abs(red - expected)[label_pixels] <= h.LINEAR_COLOUR_TOL).all()
    # Background voxels of the plane, and everything off it, draw nothing.
    assert (sure & hit & is_background).sum() > 100
    assert not shot.drawn[sure & hit & is_background].any()
    assert not shot.drawn[sure & ~hit].any()


async def test_a_multiscale_label_shows_through_another_plane_s_background(
    controller, tmp_path
):
    spec = ISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [
            h.plane_zyx(world, centre + np.array([0.0, 0.0, 9.0]), (0.3, 0.0, 1.0)),
            h.plane_zyx(world, centre, (1.0, 0.1, 0.0)),
        ],
        labels=True,
        appearance={"force_level": 1},
    )
    camera = {"view_dir": (-1.0, -0.3, -0.6)}
    shot = await rig.settle(**camera)
    labels = level_values(spec, 0)
    reference = rig.trace(shot, background=labels == 0)
    assert_matches(
        h.compare_greys(shot, reference, labels, linear_colours=True), least=1500
    )
    nearest = rig.trace(shot)
    seen_through = reference.sure & reference.hit & (reference.plane != nearest.plane)
    assert seen_through.sum() > 100
    assert shot.drawn[seen_through].all()


# -- extents and clipping ------------------------------------------------------


async def test_extents_and_clipping_cut_a_multiscale_plane(controller, tmp_path):
    spec = ANISO
    clipping = [((0.0, 0.0, 60.0), (0.0, 0.2, -1.0))]
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [
            h.plane_zyx(
                world,
                centre,
                (0.6, -0.5, 1.0),
                extent_0=(-20.0, 30.0),
                extent_1=(None, 25.0),
            )
        ],
        clipping=clipping,
        appearance={"force_level": 1},
    )
    shot = await rig.settle()
    reference = rig.trace(shot, clipping=clipping)
    assert_matches(compare_level(rig, shot, reference, 0, level_values(spec, 0)), 800)
    whole = h.trace(
        shot,
        [
            rig.planes[0].model_copy(
                update={"extent_0": (None, None), "extent_1": (None, None)}
            )
        ],
        spec.shape0,
        scale_zyx=spec.voxel_size,
    )
    assert (whole.hit & ~reference.hit).sum() > 500


# -- the level drawn is the level planned ---------------------------------------


def _expected_levels(rig: Rig, reference: h.Reference):
    """Per pixel: the level the plan names at the hit, and whether it is sure.

    The LUT names a brick per base cell (a level-0 brick), so the level at
    a position is that of the finest planned brick covering its base cell,
    or the backstop (the coarsest level) where none does.
    """
    from cellier.render._level_mapping import base_cell_range

    spec = rig.spec
    shape = np.asarray(spec.shape0)
    cells = np.ceil(shape / BLOCK).astype(int)
    level_of = np.full(tuple(cells), spec.n_levels, dtype=np.int64)
    for level, gz, gy, gx in rig.planned():
        factor = np.asarray(spec.factors[level - 1], dtype=np.float64)
        shift = (factor - 1.0) / 2.0
        low = (np.array([gz, gy, gx]) * BLOCK - 0.5) * factor + shift
        high = ((np.array([gz, gy, gx]) + 1) * BLOCK - 0.5) * factor + shift
        start, stop = base_cell_range(low + 1e-6, high - 1e-6, BLOCK)
        start, stop = np.maximum(start, 0), np.minimum(stop, cells)
        block = level_of[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]]
        np.minimum(block, level, out=block)
    index = reference.data_position[..., ::-1] + 0.5  # (z, y, x), [0, N)
    cell = np.clip(np.floor(index / BLOCK), 0, cells - 1).astype(int)
    edge = np.abs(index / BLOCK - np.round(index / BLOCK)).min(axis=-1) * BLOCK
    expected = level_of[cell[..., 0], cell[..., 1], cell[..., 2]]
    return expected, reference.sure & (edge > 0.2)


@pytest.mark.parametrize("spec_name", ["aniso", "nonpow2"])
@pytest.mark.parametrize("pose", ["xy", "oblique"])
async def test_each_pixel_draws_the_level_the_plan_names(
    controller, tmp_path, spec_name, pose
):
    """C1: every level holds one grey, so a pixel names the level it drew."""
    spec = SPECS[spec_name]
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES[pose])],
        fill=lambda level: np.full(spec.level_shape(level), level, dtype=np.int64),
        # Coarse enough, and seen at enough of a slant, that the plane's
        # near and far parts take different levels at this frame size.
        appearance={"settled_lod_bias": 4.0},
    )
    shot = await rig.settle(view_dir=(-1.0, -0.25, -0.5))
    reference = rig.trace(shot)
    expected, sure = _expected_levels(rig, reference)
    sure &= reference.hit
    assert sure.sum() > 3000
    assert shot.drawn[sure].all()
    # Level k (0-based) is drawn as grey (k + 1) / 10; the plan counts from 1.
    drawn_level = np.round(shot.frame[..., 0] / 255.0 * (h.N_VALUES + 1))
    wrong = sure & (drawn_level != expected)
    assert wrong.sum() == 0, (int(wrong.sum()), int(sure.sum()))
    # The rule used more than one level, so the test is not vacuous.
    assert len(np.unique(expected[sure])) > 1, np.unique(expected[sure])


# -- the held plan, with a real plane -------------------------------------------


def _record_plans(monkeypatch) -> list:
    from cellier.render.scene_manager import SceneManager

    plans: list = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        plans.extend(
            (visual_id, cfg.plan_mode.name, cfg.slicing_enabled)
            for visual_id, cfg in visual_configs.items()
        )
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return plans


@pytest.mark.parametrize("labels", [False, True])
async def test_a_dragged_plane_draws_held_bricks_or_the_backstop_never_nothing(
    controller, tmp_path, monkeypatch, labels
):
    """C2 with a real plane: in a scope the extent grows, the plane moves out
    of its bricks and a second plane is added.  Nothing is planned or read,
    and every pixel of the planes still draws: from a brick the last plan
    held, or from the backstop."""
    controller._render_manager.config.scheduler.dims_settle_s = 10.0
    spec = ISO
    # Every level one value (level k is k + 1), so a pixel names its level
    # and no label plane has background holes.
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [
            h.plane_zyx(
                world,
                centre,
                POSES["oblique"],
                extent_0=(-12.0, 12.0),
                extent_1=(-12.0, 12.0),
            )
        ],
        labels=labels,
        fill=lambda level: np.full(spec.level_shape(level), level + 1, dtype=np.int64),
    )
    await rig.settle()
    start = rig.planes[0]
    scheduler = controller._render_manager.scheduler
    plans = _record_plans(monkeypatch)
    world = h.world_of(controller, rig.scene)

    def drawn_levels(shot):
        reference = rig.trace(shot)
        sure = reference.sure & reference.hit
        assert sure.sum() > 500
        # Never nothing.
        assert shot.drawn[sure].all()
        grey = shot.frame[..., 0].astype(float)
        if labels:
            levels = [h.srgb_encode(h.grey_of(np.array(k + 1))) for k in range(3)]
        else:
            levels = [h.grey_of(np.array(k + 1)) for k in range(3)]
        index = np.abs(grey[..., None] - np.array(levels)).argmin(axis=-1)
        return index[sure]

    with controller.plane_interaction(rig.visual.id):
        grown = start.model_copy(
            update={"extent_0": (None, None), "extent_1": (None, None)}
        )
        rig.set_planes([grown])
        levels = drawn_levels(rig.shoot())
        # The part the last plan held is still fine; the rest is the backstop.
        assert (levels == 0).any() and (levels == 2).any()

        moved = grown.model_copy(
            update={
                "origin": tuple(np.array(grown.origin) + 20.0 * np.array(grown.normal))
            }
        )
        second = h.plane_zyx(world, centre_of(spec), POSES["xy"])
        rig.set_planes([moved, second])
        levels = drawn_levels(rig.shoot())
        assert (levels == 2).any()

        assert plans == []
        assert scheduler.core.idle()

    # The release plans the target for the planes as they are now.
    assert [(mode, sliced) for _vid, mode, sliced in plans] == [("FULL", True)]
    await drain_loading(controller)
    levels = drawn_levels(rig.shoot())
    assert (levels == 0).all()


# -- orthographic ---------------------------------------------------------------


async def test_an_orthographic_view_draws_a_plane_at_one_level(controller, tmp_path):
    spec = ANISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["oblique"])],
        fill=lambda level: np.full(spec.level_shape(level), level, dtype=np.int64),
        appearance={"settled_lod_bias": 4.0},
        fov=0.0,
    )
    shot = await rig.settle(view_dir=(-1.0, -0.25, -0.5))
    planned = rig.planned()
    assert len(planned)
    assert len(np.unique(planned[:, 0])) == 1
    reference = rig.trace(shot)
    sure = reference.sure & reference.hit
    assert sure.sum() > 3000 and shot.drawn[sure].all()
    drawn_level = np.round(shot.frame[..., 0] / 255.0 * (h.N_VALUES + 1))
    assert set(np.unique(drawn_level[sure])) == {float(planned[0, 0])}
    assert rig.gfx.slots[0]._last_plane_plan["level"] == planned[0, 0]


# -- force_level, and nothing to draw ---------------------------------------------


async def test_plane_mode_with_no_plane_plans_and_loads_nothing(
    controller, tmp_path, monkeypatch
):
    """P17: no backstop either; a plane that appears is planned at once."""
    spec = ISO
    rig = await make_rig(controller, tmp_path, spec, lambda world, centre: [])
    plans = _record_plans(monkeypatch)
    shot = await rig.settle()
    assert not shot.drawn.any()
    assert plans and not any(sliced for _vid, _mode, sliced in plans)
    registry = controller._render_manager.scheduler.core.registry(
        rig.gfx.slots[0].residency_3d().cache_id
    )
    assert not (registry.tier == Tier.VISIBLE).any()

    world = h.world_of(controller, rig.scene)
    rig.set_planes([h.plane_zyx(world, centre_of(spec), POSES["oblique"])])
    await drain_loading(controller)
    shot = rig.shoot()
    counts = compare_level(rig, shot, rig.trace(shot), 0, level_values(spec, 0))
    assert_matches(counts)


async def test_switching_a_multiscale_visual_between_plane_and_a_volume_mode(
    controller, tmp_path
):
    spec = ISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["xy"])],
        appearance={"force_level": 1},
    )
    shot = await rig.settle()
    plane_pixels = int(shot.drawn.sum())
    plane_bricks = len(rig.planned())
    assert_matches(compare_level(rig, shot, rig.trace(shot), 0, level_values(spec, 0)))

    controller.update_single_appearance_field(rig.visual.id, "render_mode", "mip")
    await drain_loading(controller)
    volume = rig.shoot()
    assert volume.drawn.sum() > 1.5 * plane_pixels
    # The volume reads every brick of the level, the plane only its own.
    assert len(rig.planned()) > 2 * plane_bricks

    controller.update_single_appearance_field(rig.visual.id, "render_mode", "plane")
    await drain_loading(controller)
    shot = rig.shoot()
    assert len(rig.planned()) == plane_bricks
    assert_matches(compare_level(rig, shot, rig.trace(shot), 0, level_values(spec, 0)))


# -- pick ------------------------------------------------------------------------


@pytest.mark.parametrize("labels", [False, True])
async def test_a_pick_on_a_multiscale_plane_names_the_voxel_at_the_hit(
    controller, tmp_path, labels
):
    spec = ANISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["oblique"])],
        labels=labels,
        appearance={"force_level": 1},
    )
    shot = await rig.settle()
    values = level_values(spec, 0)
    reference = rig.trace(shot, background=(values == 0) if labels else None)
    rows, cols = np.nonzero(reference.sure & reference.hit)
    chosen = np.linspace(0, len(rows) - 1, 150).astype(int)
    wrong, worst = [], 0.0
    for row, col in zip(rows[chosen], cols[chosen], strict=True):
        voxel = h.pick_voxel(shot, rig.gfx, int(row), int(col))
        expected = tuple(int(v) for v in reference.voxel[row, col])
        if voxel != expected:
            wrong.append((int(row), int(col), voxel, expected))
        position = h.pick_coordinate(shot, rig.gfx, int(row), int(col))
        worst = max(worst, np.abs(position - reference.data_position[row, col]).max())
    assert not wrong, wrong[:5]
    assert worst < 0.125 + max(spec.shape0) / 16383, worst


async def test_a_plane_just_past_a_cell_face_is_drawn_from_the_brick_that_owns_it(
    controller, tmp_path
):
    """The cull keeps the brick the lookup table names, not the brick whose
    own box holds the plane.

    With levels placed at no translation a level-1 brick's box ends half a
    level-0 voxel short of the cells it owns.  A plane in that half voxel is
    inside the next brick's box, but the shader draws it from the brick
    that owns the cell (reading the next brick's first voxels from its
    padding).  Were the owner not planned, the plane would fall to the
    backstop.
    """
    spec = UNALIGNED
    # Level-1 bricks along z: boxes [-1, 31), [31, 63); cells [-0.5, 31.5), ...
    z = 31.25
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [
            h.plane_zyx(world, (z, centre[1], centre[2]), POSES["xy"])
        ],
        appearance={"force_level": 2},
    )
    shot = await rig.settle()
    planned = rig.planned()
    assert len(planned) and set(planned[:, 1]) == {0}
    counts = compare_level(rig, shot, rig.trace(shot), 1, level_values(spec, 1))
    assert_matches(counts)


async def test_the_part_of_a_plane_the_cache_has_no_room_for_draws_the_backstop(
    controller, tmp_path
):
    """Truncation as shipped (D-P10): the near part of the plane is fine,
    the part dropped from the plan draws the backstop, and nothing is blank."""
    spec = ISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["oblique"])],
        fill=lambda level: np.full(spec.level_shape(level), level, dtype=np.int64),
        appearance={"force_level": 1},
        # Room for the backstop and about half of the plane's bricks.
        budget=2 * 1024**2,
    )
    shot = await rig.settle()
    progress = controller._render_manager.scheduler.progress(
        rig.gfx.slots[0].residency_3d().cache_id
    )
    assert progress.truncated_target > 0 and progress.truncated_backstop == 0
    reference = rig.trace(shot)
    sure = reference.sure & reference.hit
    assert sure.sum() > 3000 and shot.drawn[sure].all()
    drawn_level = np.round(shot.frame[..., 0] / 255.0 * (h.N_VALUES + 1))
    levels, counts = np.unique(drawn_level[sure], return_counts=True)
    # Level 1 where a brick was kept, the coarsest (3) where it was dropped.
    assert set(levels) == {1.0, 3.0}, dict(zip(levels, counts, strict=True))
    expected, clear = _expected_levels(rig, reference)
    assert not (sure & clear & (drawn_level != expected)).any()


async def test_switching_multiscale_labels_between_plane_and_a_volume_mode(
    controller, tmp_path
):
    spec = ISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["xy"])],
        labels=True,
        appearance={"force_level": 1},
    )
    labels = level_values(spec, 0)

    def check(shot) -> None:
        reference = rig.trace(shot, background=labels == 0)
        counts = h.compare_greys(shot, reference, labels, linear_colours=True)
        assert_matches(counts, least=1500)

    shot = await rig.settle()
    check(shot)
    plane_pixels = int(shot.drawn.sum())
    plane_bricks = len(rig.planned())

    controller.update_appearance_field(rig.visual.id, "render_mode", "flat_categorical")
    await drain_loading(controller)
    assert rig.shoot().drawn.sum() > 1.5 * plane_pixels
    assert len(rig.planned()) > 2 * plane_bricks

    controller.update_appearance_field(rig.visual.id, "render_mode", "plane")
    await drain_loading(controller)
    assert len(rig.planned()) == plane_bricks
    check(rig.shoot())


async def test_a_2d_view_plans_its_slice_and_ignores_the_mode(controller, tmp_path):
    """Design 4.4: in a 2D view a visual in plane mode shows the normal slice."""
    spec = ISO
    rig = await make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["oblique"])],
    )
    await rig.settle()
    assert len(rig.gfx.drawn_render_planes) == 1
    controller.set_displayed_axes(rig.scene.id, (1, 2))
    await drain_loading(controller)
    assert len(rig.gfx.drawn_render_planes) == 0
    residency = rig.gfx.slots[0].residency_2d()
    registry = controller._render_manager.scheduler.core.registry(residency.cache_id)
    # Tiles of the slice are wanted, as for any 2D image.
    assert (registry.tier == Tier.VISIBLE).any()
    controller.set_displayed_axes(rig.scene.id, (0, 1, 2))
    await drain_loading(controller)
    assert len(rig.gfx.drawn_render_planes) == 1
    assert len(rig.planned())
