"""Planning for the ``"plane"`` render mode: culling rows, levels, the cache.

Plane rendering design v3, section 6, and 14.8 (Phase 7).  Counts and
levels on small pyramids; no pixel is read here (those tests are in
``test_plane_rendering_multiscale.py``).  ``tests/_plane_fixtures.ANISO`` at
a 200 x 125 view is the stand-in for the measured store: its z is never
downsampled, so the 3D level rule picks levels several screen pixels per
voxel coarse on a plane, which is the fault the plane's own rule fixes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._frustum import bricks_in_frustum_arr, frustum_planes_from_corners
from cellier.render._level_of_detail import (
    cull_mask,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.render._plane_planning import (
    PLANE_REACH_VOXELS,
    build_plane_planning,
    owned_cell_boxes,
    plane_cull_masks,
    plane_level_orthographic,
    plane_row_sets,
    plane_thresholds,
    plane_voxel_sizes,
)
from cellier.render._render_planes import ReducedPlanes
from cellier.render.scheduling import ChunkClass
from tests import _plane_fixtures as fx
from tests.render import _planes as h
from tests.render._plane_rig import POSES, make_rig, target_rows

VIEW = (200, 125)
BLOCK = 16


def _frame(normal, spin_degrees: float = 0.0):
    """Two unit in-plane axes for *normal*, turned by *spin_degrees*."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    axis_0 = reference - (reference @ normal) * normal
    axis_0 /= np.linalg.norm(axis_0)
    axis_1 = np.cross(normal, axis_0)
    c, s = np.cos(np.radians(spin_degrees)), np.sin(np.radians(spin_degrees))
    return c * axis_0 + s * axis_1, -s * axis_0 + c * axis_1


def _reduced(*frames, origin=(0.0, 0.0, 0.0), extents=None) -> ReducedPlanes:
    n = len(frames)
    extent = np.tile([-1e30, 1e30, -1e30, 1e30], (n, 1))
    bounded = np.zeros((n, 4), dtype=bool)
    if extents is not None:
        for index, sides in enumerate(extents):
            for side, value in enumerate(sides):
                if value is not None:
                    extent[index, side] = value
                    bounded[index, side] = True
    return ReducedPlanes(
        origin=np.tile(np.asarray(origin, dtype=np.float64), (n, 1)),
        axis_0=np.array([f[0] for f in frames]),
        axis_1=np.array([f[1] for f in frames]),
        extent=extent,
        bounded=bounded,
    )


def _largest_voxel_by_search(axis_0, axis_1, voxel_world) -> float:
    """The longest a voxel is along any in-plane direction, by trying them."""
    angles = np.linspace(0.0, np.pi, 20001)
    directions = np.cos(angles)[:, None] * axis_0 + np.sin(angles)[:, None] * axis_1
    return float((1.0 / np.linalg.norm(directions / voxel_world, axis=1)).max())


# -- the measure (6.3) -----------------------------------------------------------

#: Level voxels of a pyramid that never downsamples z, in level-0 voxels (x, y, z).
SCALES = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [4.0, 4.0, 1.0]])
#: ``data -> world``: a voxel is 6.55 x 6.55 x 5 world units (x, y, z).
LINEAR = np.diag([6.55, 6.55, 5.0])


@pytest.mark.parametrize("normal", [(0, 0, 1), (0, 1, 0), (1.0, -0.5, 0.6)])
@pytest.mark.parametrize("spin", [0.0, 22.5, 45.0, 90.0])
def test_a_level_s_voxel_is_its_largest_size_in_the_plane(normal, spin):
    frame = _frame(normal, spin)
    sizes = plane_voxel_sizes(_reduced(frame), LINEAR, SCALES)
    for level, scale in enumerate(SCALES):
        searched = _largest_voxel_by_search(*frame, np.diag(LINEAR) * scale)
        assert sizes[level] == pytest.approx(searched, rel=1e-6)
    # The same for every spin of the frame about the normal.
    unspun = plane_voxel_sizes(_reduced(_frame(normal)), LINEAR, SCALES)
    assert np.allclose(sizes, unspun, rtol=1e-12)


def test_the_coarsest_in_plane_direction_counts_not_a_mean():
    """An xz plane: z is never downsampled, x is.  x sets the size."""
    sizes = plane_voxel_sizes(_reduced(_frame((0, 1, 0))), LINEAR, SCALES)
    assert np.allclose(sizes, 6.55 * SCALES[:, 0])


def test_several_planes_take_the_smallest_voxel_of_each_level():
    xy, xz = _frame((0, 0, 1)), _frame((0, 1, 0))
    oblique = _frame((1.0, -0.5, 0.6))
    each = [plane_voxel_sizes(_reduced(f), LINEAR, SCALES) for f in (xy, xz, oblique)]
    together = plane_voxel_sizes(_reduced(xy, xz, oblique), LINEAR, SCALES)
    assert np.allclose(together, np.min(each, axis=0))


def test_thresholds_are_the_2d_rule_s_transition_at_a_distance():
    sizes = np.array([1.0, 2.0, 4.0])
    fov, height = np.radians(60.0), 500.0
    thresholds = plane_thresholds(sizes, fov, height, bias=1.0)
    focal = (height / 2) / np.tan(fov / 2)
    # At a threshold a pixel covers the geometric mean of the two voxels.
    assert np.allclose(np.array(thresholds) / focal, [np.sqrt(2.0), np.sqrt(8.0)])
    coarser = plane_thresholds(sizes, fov, height, bias=2.0)
    assert np.allclose(np.array(coarser) * 2.0, thresholds)


def test_the_orthographic_level_is_the_one_whose_voxel_matches_the_pixel():
    sizes = np.array([1.0, 2.0, 4.0, 8.0])
    for pixel, level in ((0.5, 1), (1.4, 1), (1.5, 2), (2.8, 2), (2.9, 3), (50.0, 4)):
        assert plane_level_orthographic(sizes, pixel * 100.0, 100.0, 1.0) == level
    assert plane_level_orthographic(sizes, 100.0, 100.0, 3.0) == 3
    assert plane_level_orthographic(sizes, 0.0, 100.0, 1.0) is None


# -- culling rows (6.2) ----------------------------------------------------------


def test_plane_rows_keep_the_bricks_the_plane_crosses():
    """Under an anisotropic, shifted transform: rows in data space agree with
    the plane in world space, corner by corner."""
    rng = np.random.default_rng(3)
    linear = np.diag([2.0, 0.5, 3.0])
    offset = np.array([10.0, -4.0, 7.0])
    frame = _frame((0.4, 1.0, -0.7), 20.0)
    origin = np.array([40.0, 5.0, 60.0])
    reduced = _reduced(frame, origin=origin, extents=[(-15.0, 20.0, None, 9.0)])
    rows = plane_row_sets(reduced, linear, offset)[0]
    # Two for the slab, three for the bounded sides.
    assert rows.shape == (5, 4)

    normal = np.cross(*frame)
    half = np.array([4.0, 4.0, 4.0])
    reach = PLANE_REACH_VOXELS * float(np.abs(normal) @ np.abs(linear).sum(axis=1))
    agree = 0
    for centre in rng.uniform(-10, 60, size=(4000, 3)):
        corners = centre + half * (np.indices((2, 2, 2)).reshape(3, -1).T * 2 - 1)
        world = corners @ linear.T + offset
        rel = world - origin
        tests = [
            (rel @ normal + reach >= 0).any(),
            (-(rel @ normal) + reach >= 0).any(),
            (rel @ frame[0] >= -15.0).any(),
            (rel @ frame[0] <= 20.0).any(),
            (rel @ frame[1] <= 9.0).any(),
        ]
        by_rows = all((corners @ row[:3] + row[3] >= 0).any() for row in rows)
        assert by_rows == all(tests)
        agree += by_rows
    assert 100 < agree < 3900


def test_no_plane_keeps_no_brick():
    planning = build_plane_planning(
        _reduced(), lambda points: np.asarray(points), np.ones((2, 3))
    )
    grid = {
        "arr": np.zeros((5, 4), dtype=np.int32),
        "centres": np.zeros((5, 3)),
        "half_extents": np.ones(3),
        "centre_abs_max": np.zeros(3),
    }
    assert not cull_mask(grid, None, plane_row_sets=planning.row_sets).any()


# -- the plan, through the production path ----------------------------------------


def _focal(request) -> float:
    return (request.screen_size_px[1] / 2.0) / np.tan(request.fov_y_rad / 2.0)


def _pixels_per_voxel(rig, rows, request, plane, sizes_of=None) -> np.ndarray:
    """Per brick: screen pixels across one voxel of its level, in *plane*.

    Measured along the in-plane direction the level's voxel is longest in,
    found by search, at the brick centre's distance from the camera.
    """
    spec = rig.spec
    axis_0 = np.asarray(plane.in_plane_axis_0)[::-1]
    axis_1 = np.asarray(plane.in_plane_axis_1)[::-1]
    voxel = np.asarray(spec.voxel_size)[::-1]
    size = np.array(
        [
            _largest_voxel_by_search(axis_0, axis_1, voxel * np.asarray(f)[::-1])
            for f in spec.factors
        ]
    )
    centres = rig.brick_centres_world(rows)
    distance = np.linalg.norm(centres - np.asarray(request.camera_pos)[:3], axis=1)
    return size[rows[:, 0] - 1] * _focal(request) / distance


async def _plane_rig(controller, pyramid_root, spec, normals, **kwargs):
    """A rig over the session's shared pyramid of *spec*.

    These tests plan and never look at a voxel's value, so they need no
    pyramid of their own: writing one each was most of this file's time on
    the Windows runners.
    """
    return await make_rig(
        controller,
        None,
        spec,
        lambda world, centre: [h.plane_zyx(world, centre, n) for n in normals],
        size=VIEW,
        root=pyramid_root(spec, labels=kwargs.get("labels", False)),
        **kwargs,
    )


@pytest.mark.parametrize("spec", [fx.ANISO, fx.ISO], ids=["aniso", "iso"])
@pytest.mark.parametrize("pose", list(POSES))
async def test_the_plane_rule_draws_about_a_pixel_per_voxel(
    controller, pyramid_root, spec, pose
):
    """Level rule (14.8): 0.7 to 1.6 screen pixels per voxel at bias 1.0."""
    rig = await _plane_rig(controller, pyramid_root, spec, [POSES[pose]])
    (desired,), request = rig.plan()
    rows = target_rows(desired)
    assert len(rows) > 4
    assert desired.n_truncated_target == 0
    ratio = _pixels_per_voxel(rig, rows, request, rig.planes[0])
    # The finest level has nothing finer to hand over to, so a near brick of
    # it may be drawn larger; the bound below is the rule's.
    between = rows[:, 0] > 1
    assert np.median(ratio) > 0.7
    if between.any():
        assert np.percentile(ratio[between], 95) < 1.6
        assert np.percentile(ratio[between], 5) > 0.7


@pytest.mark.parametrize("pose", list(POSES))
async def test_the_3d_rule_is_several_pixels_per_voxel_coarse_on_a_plane(
    controller, pyramid_root, pose
):
    """Why 6.3 exists: on a pyramid that never downsamples z the shipping
    thresholds, applied to the same bricks, pick levels far too coarse."""
    rig = await _plane_rig(controller, pyramid_root, fx.ANISO, [POSES[pose]])
    (desired,), request = rig.plan()
    plane_rows = target_rows(desired)
    by_plane_rule = np.median(
        _pixels_per_voxel(rig, plane_rows, request, rig.planes[0])
    )

    slot = rig.planner
    geo = slot._volume_geometry
    planning = build_plane_planning(
        rig.gfx.drawn_render_planes, slot._to_level0_displayed, geo._scale_arr_shader
    )
    keep = plane_cull_masks(geo, planning, None)
    camera = slot._to_level0_displayed(np.asarray(request.camera_pos).reshape(1, -1))
    shipping = [
        geo._level_scale_factors[k - 1] * _focal(request)
        for k in range(1, geo.n_levels)
    ]
    rows = select_levels_from_cache(
        geo._level_grids,
        geo.n_levels,
        camera.ravel(),
        thresholds=shipping,
        base_layout=geo.base_layout,
        keep=keep,
    )
    by_3d_rule = np.median(_pixels_per_voxel(rig, rows, request, rig.planes[0]))
    assert by_3d_rule > 2.0
    assert by_3d_rule > 1.8 * by_plane_rule


@pytest.mark.parametrize("pose", ["xz", "oblique"])
async def test_the_plan_does_not_depend_on_the_frame_s_spin(
    controller, pyramid_root, pose
):
    rig = await _plane_rig(controller, pyramid_root, fx.ANISO, [POSES[pose]])
    start = rig.planes[0]
    plans = []
    for spin in (0.0, 22.5, 45.0, 90.0):
        a0, a1 = np.array(start.in_plane_axis_0), np.array(start.in_plane_axis_1)
        c, s = np.cos(np.radians(spin)), np.sin(np.radians(spin))
        rig.set_planes(
            [
                start.model_copy(
                    update={
                        "in_plane_axis_0": tuple(c * a0 + s * a1),
                        "in_plane_axis_1": tuple(-s * a0 + c * a1),
                    }
                )
            ]
        )
        (desired,), _request = rig.plan()
        plans.append(target_rows(desired))
    assert len(plans[0]) > 4
    for other in plans[1:]:
        assert np.array_equal(other, plans[0])


async def test_no_plane_is_drawn_coarser_than_asked_when_there_are_several(
    controller, pyramid_root
):
    """One set of thresholds serves the visual: the finest any plane asks."""
    normals = [POSES["xy"], POSES["xz"]]
    rig = await _plane_rig(controller, pyramid_root, fx.ANISO, normals)
    (desired,), request = rig.plan()
    rows = target_rows(desired)
    for plane in rig.planes:
        ratio = _pixels_per_voxel(rig, rows, request, plane)
        assert np.percentile(ratio, 5) > 0.7


CULL_CASES = {
    "extents": {
        "normals": [POSES["oblique"]],
        "extents": [((-300.0, 200.0), (None, 150.0))],
    },
    "clipping": {
        "normals": [POSES["oblique"]],
        "clipping": [((32, 128, 128), (0.3, 0.5, 1.0)), ((0, 100, 0), (0, 1, 0))],
    },
    "three_planes": {"normals": [POSES["xy"], POSES["xz"], (0.0, 0.0, 1.0)]},
    "three_bounded_and_clipped": {
        "normals": [POSES["xy"], POSES["xz"], POSES["oblique"]],
        "extents": [((-400.0, 300.0), (None, None))] * 3,
        "clipping": [((0, 0, 128), (0, 0, 1))],
    },
}


@pytest.mark.parametrize("case", list(CULL_CASES))
@pytest.mark.parametrize("bias", [1.0, 0.5])
async def test_the_cull_first_plane_plan_equals_ranking_every_brick_then_culling(
    controller, pyramid_root, case, bias
):
    """Cull equivalence (14.8): the same rows in the same order."""
    options = CULL_CASES[case]
    extents = options.get("extents")

    def planes(world, centre):
        out = []
        for index, normal in enumerate(options["normals"]):
            extra = {}
            if extents is not None:
                extra = {"extent_0": extents[index][0], "extent_1": extents[index][1]}
            out.append(h.plane_zyx(world, centre, normal, **extra))
        return out

    rig = await make_rig(
        controller,
        None,
        fx.ANISO,
        planes,
        size=VIEW,
        root=pyramid_root(fx.ANISO),
        clipping=options.get("clipping", ()),
        appearance={"settled_lod_bias": bias},
    )
    (desired,), request = rig.plan(view_dir=(-0.6, -0.3, -1.0))
    assert desired.n_truncated_target == 0
    planned = target_rows(desired)
    backstop = np.asarray(desired.keys)[
        np.asarray(desired.cls).ravel() == int(ChunkClass.BACKSTOP)
    ]

    slot = rig.planner
    geo = slot._volume_geometry
    planning = build_plane_planning(
        rig.gfx.drawn_render_planes, slot._to_level0_displayed, geo._scale_arr_shader
    )
    camera = slot._to_level0_displayed(
        np.asarray(request.camera_pos).reshape(1, -1)
    ).ravel()
    # Rank every brick of every level, sort them all, then cull.
    everything = select_levels_from_cache(
        geo._level_grids,
        geo.n_levels,
        camera,
        thresholds=plane_thresholds(
            planning.voxel_size, request.fov_y_rad, request.screen_size_px[1], bias
        ),
        base_layout=geo.base_layout,
        metric=planning.metric,
    )
    everything = sort_arr_by_distance(
        everything,
        camera,
        geo.block_size,
        scale_vecs_shader=geo._scale_arr_shader,
        translation_vecs_shader=geo._translation_arr_shader,
        metric=planning.metric,
    )
    clip_rows = slot._clip_rows()
    on_a_plane = np.zeros(len(everything), dtype=bool)
    for level, owned in enumerate(owned_cell_boxes(geo)):
        mask = owned["owns"] & cull_mask(owned, None, plane_row_sets=planning.row_sets)
        index = {tuple(row) for row in owned["arr"][mask]}
        of_level = everything[:, 0] == level + 1
        on_a_plane |= of_level & np.array([tuple(r) in index for r in everything])
    brute = everything[on_a_plane]
    rows = frustum_planes_from_corners(
        slot._to_level0_displayed(request.frustum_corners)
    )
    if clip_rows is not None:
        rows = np.concatenate([rows, clip_rows])
    brute, _ = bricks_in_frustum_arr(
        brute,
        geo.block_size,
        rows,
        level_scale_arr_shader=geo._scale_arr_shader,
        level_translation_arr_shader=geo._translation_arr_shader,
    )
    # The desired set leaves out target rows that are backstop bricks too.
    from cellier.render.visuals._chunked import desired_bricks

    expected = desired_bricks(
        slot,
        slot.residency_3d(),
        brute,
        backstop_arr=None,
    )
    expected_rows = target_rows(expected)
    is_backstop = np.isin(np.asarray(expected.keys), backstop)
    assert len(planned) > 6
    assert np.array_equal(planned, expected_rows[~is_backstop])
    # And it ranked fewer bricks than there are: those the planes cross.
    assert slot._last_plane_plan["n_ranked"] < sum(
        len(grid["arr"]) for grid in geo._level_grids
    )


async def test_force_level_replaces_the_plane_rule(controller, pyramid_root):
    rig = await _plane_rig(
        controller,
        pyramid_root,
        fx.ANISO,
        [POSES["oblique"]],
        appearance={"force_level": 3},
    )
    (desired,), _request = rig.plan()
    rows = target_rows(desired)
    assert len(rows) and set(rows[:, 0]) == {3}
    assert rig.planner._last_plane_plan["level"] == 3


# -- the cache (6.4) ---------------------------------------------------------------


@pytest.mark.parametrize("normals", [[POSES["oblique"]], list(POSES.values())])
async def test_one_channel_fits_the_cache(controller, pyramid_root, normals):
    rig = await _plane_rig(controller, pyramid_root, fx.ANISO, normals)
    (desired,), _request = rig.plan()
    assert len(target_rows(desired)) > 4
    assert desired.n_truncated_target == 0


async def test_a_plan_over_the_budget_is_truncated_nearest_first(
    controller, pyramid_root
):
    """D-P10: a plan over the budget is truncated; the part of the plane
    farthest from the camera, in world units, is left to the backstop."""
    normals = [POSES["oblique"]]
    full = await _plane_rig(controller, pyramid_root, fx.ANISO, normals)
    (wanted,), request = full.plan()
    wanted_rows = target_rows(wanted)
    small = await _plane_rig(
        controller, pyramid_root, fx.ANISO, normals, budget=2 * 1024**2
    )
    (desired,), _request = small.plan()
    kept = target_rows(desired)
    assert desired.n_truncated_target > 0
    assert len(kept) + desired.n_truncated_target == len(wanted_rows)
    # What is kept is the head of the full plan: the bricks nearest the camera.
    assert np.array_equal(kept, wanted_rows[: len(kept)])
    # Nearest in world units, although the data's voxels are not cubes.
    centres = full.brick_centres_world(wanted_rows)
    camera = np.asarray(request.camera_pos)[:3]
    distance = np.linalg.norm(centres - camera, axis=1)
    assert np.all(np.diff(distance) >= -1e-6)
    assert distance[: len(kept)].max() <= distance[len(kept) :].min() + 1e-6
    # In level-0 voxels the same rows are not in order: the two differ here.
    voxel = np.asarray(fx.ANISO.voxel_size)[::-1]
    in_voxels = np.linalg.norm(centres / voxel - camera / voxel, axis=1)
    assert np.any(np.diff(in_voxels) < -1e-6)


async def test_foreshortening_plans_at_most_half_again_as_many_bricks(
    controller, pyramid_root
):
    """B8: a plane seen at a slant is planned finer than the screen shows,
    never coarser, and within 1.6 times the face-on count.

    Summed over camera distances spanning two octaves: on a pyramid this
    small one distance is a single level band face-on, so a tilt that brings
    the near half into the next band is a fourfold step there and no step
    at the next distance.
    """
    rig = await _plane_rig(controller, pyramid_root, fx.ANISO, [POSES["xy"]])
    distances = [2.0 ** (step / 4.0) for step in range(-4, 6)]
    totals = {}
    for tilt in (0.0, 35.0, 60.0, 75.0, 85.0):
        a = np.radians(tilt)
        total = 0
        for factor in distances:
            (desired,), _request = rig.plan(
                view_dir=(-np.sin(a), 0.0, -np.cos(a)), distance_factor=factor
            )
            assert desired.n_truncated_target == 0
            total += len(target_rows(desired))
        totals[tilt] = total
    assert totals[0.0] > 200
    for total in totals.values():
        assert total <= 1.6 * totals[0.0], totals
        # Finer, never coarser.
        assert total >= totals[0.0], totals


# -- orthographic (6.3, D-P47, D-P48) ------------------------------------------------


@pytest.mark.parametrize("pose", list(POSES))
async def test_an_orthographic_plane_takes_one_level_in_the_2d_rule_s_band(
    controller, pyramid_root, pose
):
    rig = await _plane_rig(controller, pyramid_root, fx.ANISO, [POSES[pose]], fov=0.0)
    (desired,), request = rig.plan()
    rows = target_rows(desired)
    assert len(rows) > 2
    assert len(set(rows[:, 0])) == 1
    level = int(rows[0, 0])
    plan = rig.planner._last_plane_plan
    assert plan["level"] == level and plan["levels_stepped"] == 0
    # Screen pixels per voxel of the level, by the plane's own measure.
    pixel = request.world_extent[1] / request.screen_size_px[1]
    planning = build_plane_planning(
        rig.gfx.drawn_render_planes,
        rig.planner._to_level0_displayed,
        rig.planner._volume_geometry._scale_arr_shader,
    )
    ratio = planning.voxel_size[level - 1] / pixel
    n_levels = rig.spec.n_levels
    if 1 < level < n_levels:
        assert 1 / np.sqrt(2) - 1e-9 <= ratio <= np.sqrt(2) + 1e-9
    elif level == 1:
        assert ratio >= 1 / np.sqrt(2) - 1e-9


async def test_an_orthographic_plan_that_does_not_fit_takes_a_coarser_level(
    controller, pyramid_root
):
    """P19 (D-P48): step to the next coarser level until the plan fits."""
    normals = [POSES["oblique"]]
    kwargs = {"fov": 0.0, "appearance": {"settled_lod_bias": 0.25}}
    roomy = await _plane_rig(controller, pyramid_root, fx.ANISO, normals, **kwargs)
    (wanted,), _request = roomy.plan()
    rule_level = int(target_rows(wanted)[0, 0])
    assert roomy.planner._last_plane_plan["levels_stepped"] == 0
    tight = await _plane_rig(
        controller, pyramid_root, fx.ANISO, normals, budget=2 * 1024**2, **kwargs
    )
    (desired,), _request = tight.plan()
    rows = target_rows(desired)
    plan = tight.planner._last_plane_plan
    # The rule's level does not fit this atlas ...
    from cellier.render.visuals._chunked import target_room

    assert len(target_rows(wanted)) > target_room(
        tight.planner.residency_3d(), None, None
    )
    # ... so a coarser one is drawn, whole.
    assert plan["levels_stepped"] >= 1
    assert plan["level"] == rule_level + plan["levels_stepped"]
    assert len(rows) and set(rows[:, 0]) == {plan["level"]}
    assert desired.n_truncated_target == 0


async def test_a_perspective_plan_over_the_budget_is_not_stepped(
    controller, pyramid_root
):
    rig = await _plane_rig(
        controller,
        pyramid_root,
        fx.ANISO,
        [POSES["oblique"]],
        budget=2 * 1024**2,
        appearance={"settled_lod_bias": 0.25},
    )
    (desired,), _request = rig.plan()
    assert desired.n_truncated_target > 0
    assert rig.planner._last_plane_plan["levels_stepped"] == 0


# -- labels, composite ---------------------------------------------------------------


async def test_labels_plan_the_same_bricks_as_an_image(controller, pyramid_root):
    normals = [POSES["oblique"]]
    image = await _plane_rig(controller, pyramid_root, fx.ANISO, normals)
    labels = await _plane_rig(controller, pyramid_root, fx.ANISO, normals, labels=True)
    (of_image,), _ = image.plan()
    (of_labels,), _ = labels.plan()
    assert len(target_rows(of_image)) > 4
    assert np.array_equal(target_rows(of_image), target_rows(of_labels))


async def test_a_composite_splits_the_cache_and_every_channel_plans_the_same_bricks(
    controller, pyramid_root
):
    """B4: the budget is split between the drawn channels, so a plan that
    fits one channel is truncated with two; each channel gets the same
    bricks, and the same number is dropped from each."""
    from cellier.scene import spatial_axes
    from cellier.transform import AffineTransform
    from cellier.visuals import (
        MultiscaleImageAppearance,
        MultiscaleImageChannelAppearance,
        MultiscaleImageRenderConfig,
        RenderPlane,
    )
    from tests._gpu_budget import BUDGET_2D

    # Room for the plan with one channel, not with the budget halved.
    budget = 6 * 1024**2
    plans = {}
    for n_channels in (1, 2):
        spec = fx.channel_spec(n_channels)
        store, voxel = fx.open_pyramid(pyramid_root(spec), spec, name=f"c{n_channels}")
        scene = controller.add_scene(
            coordinate_system=spatial_axes("c", "z", "y", "x"),
            dim="3d",
            name=f"channels-{n_channels}",
        )
        controller._ensure_data_coordinate_systems(scene.id, store)
        data = store.data_coordinate_systems[0]
        world = controller._model.scenes[scene.id].dims.world_coordinate_system
        names = ("c", "z", "y", "x")
        transform = AffineTransform.from_axis_map(
            data,
            world,
            {data.axis_by_name(n).id: world.axis_by_name(n).id for n in names},
            scale={
                data.axis_by_name(n).id: v for n, v in zip(names, voxel, strict=True)
            },
        )
        centre = ((np.array(fx.ANISO.shape0) - 1) / 2 + 0.3) * np.array(
            fx.ANISO.voxel_size
        )
        plane = RenderPlane.from_point_normal(
            world, centre, POSES["oblique"], axes=("z", "y", "x")
        )
        visual = controller.add_image_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(settled_lod_bias=0.5),
            render_config=MultiscaleImageRenderConfig(
                block_size=BLOCK, gpu_budget_bytes=budget, gpu_budget_bytes_2d=BUDGET_2D
            ),
            channel_axis=0,
            composite=True,
            channels={
                index: MultiscaleImageChannelAppearance(render_mode="plane")
                for index in range(n_channels)
            },
            transform=transform,
            render_planes=(plane,),
        )
        controller.add_canvas(scene_id=scene.id, canvas_size=VIEW)
        canvas_id = controller.get_canvas_ids(scene.id)[0]
        view = controller._render_manager._canvases[canvas_id]
        view._canvas.get_logical_size = lambda: (float(VIEW[0]), float(VIEW[1]))
        h.shoot(controller, scene, size=VIEW)
        selection = controller._selections_for_scene(scene.id)[canvas_id]
        request = view.capture_reslicing_request(scene.dims.to_state(), selection, None)
        gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
        desired = gfx.plan(request, controller._render_config_for(scene.id, visual))
        assert len(desired) == n_channels
        rooms = [slot.residency_3d().n_slots for slot in gfx.slots[:n_channels]]
        plans[n_channels] = (desired, rooms, gfx.slots[0]._last_plane_plan)

    (alone,), (room_alone,), plan_alone = plans[1]
    both, rooms, plan_both = plans[2]
    wanted = plan_alone["n_selected"]
    # The plan is the same; the room for it is halved.
    assert plan_both["n_selected"] == wanted
    assert alone.n_truncated_target == 0
    assert rooms[0] == rooms[1] and rooms[0] < room_alone
    kept = [len(target_rows(d)) for d in both]
    dropped = [d.n_truncated_target for d in both]
    assert dropped[0] == dropped[1] > 0
    assert kept[0] == kept[1]
    assert np.array_equal(target_rows(both[0]), target_rows(both[1]))
    assert kept[0] + dropped[0] == len(target_rows(alone)) + alone.n_truncated_target
