"""The cull-first 3D brick planner (plane rendering design v3 6.1, Phase 1).

A clipped visual drops the bricks its clipping planes cut off before it ranks
and sorts them.  The plan must be the rows of the old order (rank everything,
cull last), row for row; the golden replay
(``test_plane_golden_replay.py``) checks that on the production path, and
these check the pieces.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cellier.render import _level_of_detail as lod
from cellier.render._frustum import bricks_in_frustum_arr
from cellier.render._level_of_detail import (
    build_level_grids,
    cull_mask,
    cull_masks,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.render.lut_indirection import BlockLayout3D
from cellier.render.visuals import _image, _label_multiscale
from cellier.visuals import (
    ClippingPlane,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests._plane_fixtures import ANISO, open_pyramid

BLOCK_SIZE = 8
SHAPES = [(40, 96, 96), (40, 48, 48), (40, 24, 24), (40, 12, 12)]


@pytest.fixture(scope="module")
def geometry():
    """An anisotropic four-level pyramid: z is never downsampled."""
    layout = BlockLayout3D(volume_shape=SHAPES[0], block_size=BLOCK_SIZE)
    # shader order (x, y, z)
    scales = np.array([[2.0**k, 2.0**k, 1.0] for k in range(4)])
    translations = np.array(
        [[(2.0**k - 1) / 2, (2.0**k - 1) / 2, 0.0] for k in range(4)]
    )
    grids = build_level_grids(
        layout, 4, list(scales), list(translations), level_shapes=SHAPES
    )
    return layout, grids, scales, translations


def _random_rows(rng, n):
    """*n* half-spaces through random points of the volume, pygfx order."""
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points = rng.uniform([0, 0, 0], [96, 96, 40], size=(n, 3))
    return np.column_stack([normals, -(normals * points).sum(axis=1)])


def _plan(geometry, camera, clip_rows, *, thresholds, force_level, cull_first):
    layout, grids, scales, translations = geometry
    keep = cull_masks(grids, clip_rows) if cull_first else None
    if force_level is not None:
        arr = select_levels_arr_forced(layout, force_level, grids, keep=keep)
    else:
        arr = select_levels_from_cache(
            grids, 4, camera, thresholds=thresholds, base_layout=layout, keep=keep
        )
    ranked = len(arr)
    arr = sort_arr_by_distance(
        arr,
        camera,
        BLOCK_SIZE,
        scale_vecs_shader=scales,
        translation_vecs_shader=translations,
    )
    arr, _ = bricks_in_frustum_arr(
        arr,
        BLOCK_SIZE,
        clip_rows,
        level_scale_arr_shader=scales,
        level_translation_arr_shader=translations,
    )
    return arr, ranked


@pytest.mark.parametrize("force_level", [None, 1, 3, 9])
@pytest.mark.parametrize("thresholds", [[60.0, 120.0, 240.0], None, []])
def test_cull_first_plan_equals_cull_last_plan(geometry, force_level, thresholds):
    rng = np.random.default_rng(7)
    for _ in range(40):
        clip_rows = _random_rows(rng, int(rng.integers(1, 4)))
        camera = rng.uniform([-100, -100, -100], [200, 200, 140])
        kwargs = {"thresholds": thresholds, "force_level": force_level}
        last, ranked_last = _plan(
            geometry, camera, clip_rows, cull_first=False, **kwargs
        )
        first, ranked_first = _plan(
            geometry, camera, clip_rows, cull_first=True, **kwargs
        )
        np.testing.assert_array_equal(first, last)
        assert ranked_first <= ranked_last


def test_axis_aligned_rows_on_brick_faces_keep_what_the_corner_test_keeps(geometry):
    """A plane lying exactly on brick faces is the rounding-sensitive case."""
    camera = np.array([150.0, 40.0, 80.0])
    for axis in range(3):
        for sign in (1.0, -1.0):
            for face in (-0.5, 7.5, 15.5, 31.5, 39.5):
                row = np.zeros((1, 4))
                row[0, axis] = sign
                row[0, 3] = -sign * face
                kwargs = {"thresholds": [60.0, 120.0, 240.0], "force_level": None}
                last, _ = _plan(geometry, camera, row, cull_first=False, **kwargs)
                first, _ = _plan(geometry, camera, row, cull_first=True, **kwargs)
                np.testing.assert_array_equal(first, last)


def test_cull_mask_is_none_with_nothing_to_cull_by(geometry):
    _, grids, _, _ = geometry
    assert cull_mask(grids[0]) is None
    assert cull_masks(grids) is None


def test_cull_mask_plane_row_sets_are_a_union(geometry):
    """A brick is kept when it reaches every row of at least one set."""
    _, grids, _, _ = geometry
    grid = grids[0]

    def slab(axis, at):
        low, high = np.zeros(4), np.zeros(4)
        low[axis], low[3] = 1.0, -(at - 1.0)
        high[axis], high[3] = -1.0, at + 1.0
        return np.stack([low, high])

    slab_x, slab_y = slab(0, 20.0), slab(1, 60.0)
    on_x = cull_mask(grid, plane_row_sets=[slab_x])
    on_y = cull_mask(grid, plane_row_sets=[slab_y])
    both = cull_mask(grid, plane_row_sets=[slab_x, slab_y])
    np.testing.assert_array_equal(both, on_x | on_y)
    assert 0 < on_x.sum() < len(on_x)
    # A slab is the intersection of its two rows, as clipping rows are.
    np.testing.assert_array_equal(on_x, cull_mask(grid, clip_rows=slab_x))
    # Clipping rows intersect with the union of the planes.
    clip = np.array([[0.0, 0.0, 1.0, -20.0]])
    np.testing.assert_array_equal(
        cull_mask(grid, clip, [slab_x, slab_y]), cull_mask(grid, clip) & both
    )


# --------------------------------------------------------------------------- #
# The production planners                                                      #
# --------------------------------------------------------------------------- #


def _clip(store, point, normal):
    return ClippingPlane.from_point_normal(
        store.data_coordinate_systems[0], point, normal, axes=("z", "y", "x")
    )


def _build(controller, pyramid_root, *, labels, clipped):
    store, _ = open_pyramid(pyramid_root(ANISO, labels=labels), ANISO)
    scene = controller.add_scene(dim="3d", name="s")
    controller._ensure_data_coordinate_systems(scene.id, store)
    planes = (_clip(store, (32, 128, 128), (-0.6, 0.5, 1.0)),) if clipped else ()
    # A fine bias, so the fine levels (most of the bricks) are ranked.
    if labels:
        visual = controller.add_labels_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(),
            render_config=MultiscaleLabelRenderConfig(block_size=16, **SMALL_BUDGETS),
            clipping_planes=planes,
        )
    else:
        visual = controller.add_image_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(),
            render_config=MultiscaleImageRenderConfig(block_size=16, **SMALL_BUDGETS),
            single=MultiscaleImageSingleAppearance(render_mode="mip"),
            clipping_planes=planes,
        )
    controller.add_canvas(scene_id=scene.id, canvas_size=(400, 300))
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller._render_manager._canvases[canvas_id]
    view._canvas.get_logical_size = lambda: (400.0, 300.0)
    controller.fit_camera(scene.id)
    return scene, visual


def _plan_counting(controller, scene, visual, monkeypatch, module):
    """Plan once; return ``(bricks ranked, bricks the cull kept, all bricks)``."""
    ranked = []
    original = lod.sort_arr_by_distance

    def spy(arr, *args, **kwargs):
        ranked.append(len(arr))
        return original(arr, *args, **kwargs)

    monkeypatch.setattr(module, "sort_arr_by_distance", spy)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller._render_manager._canvases[canvas_id]
    selection = controller._selections_for_scene(scene.id)[canvas_id]
    request = view.capture_reslicing_request(scene.dims.to_state(), selection, None)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    config = controller._render_config_for(scene.id, visual)
    # Every brick of the finest level, so the count does not hang on the camera.
    gfx.plan(request, replace(config, force_level=1))
    planner = gfx if module is _label_multiscale else gfx._slots[0]
    geo = planner._volume_geometry
    masks = cull_masks(geo._level_grids, planner._clip_rows())
    total = len(geo._level_grids[0]["arr"])
    kept = total if masks is None else int(masks[0].sum())
    monkeypatch.undo()
    return ranked[-1], kept, total


@pytest.mark.parametrize("labels", [False, True], ids=["image", "labels"])
def test_a_clipped_visual_ranks_only_the_bricks_the_cull_kept(
    controller, pyramid_root, monkeypatch, labels
):
    module = _label_multiscale if labels else _image
    scene, visual = _build(controller, pyramid_root, labels=labels, clipped=True)
    ranked, kept, total = _plan_counting(controller, scene, visual, monkeypatch, module)
    assert ranked <= kept
    assert 0 < kept < total


@pytest.mark.parametrize("labels", [False, True], ids=["image", "labels"])
def test_an_unclipped_visual_ranks_every_brick(
    controller, pyramid_root, monkeypatch, labels
):
    module = _label_multiscale if labels else _image
    scene, visual = _build(controller, pyramid_root, labels=labels, clipped=False)
    ranked, kept, total = _plan_counting(controller, scene, visual, monkeypatch, module)
    assert ranked == kept == total
