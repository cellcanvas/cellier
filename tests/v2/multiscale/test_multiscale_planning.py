"""The multiscale families plan from the region, and agree with what they did.

Driven through a live controller, because the region only reaches a visual
that the controller has placed: it is built from the scene's dims and the
canvas's rendered coordinate system, and it is pulled back through the store's
per-level transforms.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.controller import CellierController
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.render.visuals._slicing import round_world_to_voxel
from cellier.scene.dims import spatial_axes
from cellier.visuals import MultiscaleImageRenderConfig, MultiscaleImageSingleAppearance
from cellier.visuals._image import MultiscaleImageAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests._planning import planned_requests_3d
from tests._v2 import pyramid_levels


@pytest.fixture
def anisotropic_zarr(tmp_path):
    """A 4-D ``tzyx`` 3-level pyramid whose ``z`` is **not** downsampled.

    The class of pyramid this repo has already had to fix once: a single
    ``2 ** (level - 1)`` factor applied to every axis gets it wrong.  Four
    dimensions so there is a collapsed axis for the region to decide.
    """
    shapes = [(4, 8, 16, 16), (4, 8, 8, 8), (4, 8, 4, 4)]
    for name, shape in zip(("s0", "s1", "s2"), shapes):
        store = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(tmp_path / name)},
                "metadata": {
                    "shape": list(shape),
                    "data_type": "float32",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [4, 4, 4, 4]},
                    },
                },
                "create": True,
                "delete_existing": True,
            }
        ).result()
        store[...].write(np.zeros(shape, dtype=np.float32)).result()
    return tmp_path


def _store(path):
    """A store whose level transforms leave ``z`` alone and halve ``y``/``x``."""
    return MultiscaleZarrDataStore(
        zarr_path=str(path),
        scale_names=["s0", "s1", "s2"],
        **pyramid_levels(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 4.0, 4.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.5, 1.5]],
        ),
    )


def _viewer(path, world=None):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=world or [("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
    )
    controller.add_canvas(scene.id)
    store = _store(path)
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
    )
    return controller, scene, visual


def _plan(
    controller, scene, visual, *, with_region: bool = True, level: int | None = 0
):
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    selection = (
        controller._selections_for_scene(scene.id)[canvas_id] if with_region else None
    )
    requests = planned_requests_3d(
        gfx,
        camera_pos_world=np.array([8.0, 8.0, 40.0]),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=200.0,
        dims_state=scene.dims.to_state(),
        force_level=level,
        selection=selection,
    )
    return sorted((r.scale_index, r.axis_selections) for r in requests)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("slice_position", [0.0, 1.0, 3.0])
async def test_the_collapsed_axis_is_the_rounded_plane_at_every_level(
    anisotropic_zarr, level, slice_position
):
    """``t`` is not downsampled by this pyramid, so the plane the region
    selects is the same integer at every level -- and it is the world position
    rounded half-up, because the scene's transform is the identity.

    This was originally an equality against planning from ``dims_state``.
    Phase 8 deleted that path; the numbers it agreed on are asserted directly.
    """
    controller, scene, visual = _viewer(anisotropic_zarr)
    controller.update_slice_indices(scene.id, {0: slice_position})
    planned = _plan(controller, scene, visual, level=level)
    assert planned, f"level {level} planned nothing"
    expected = round_world_to_voxel(slice_position, 8)
    for _scale_index, selections in planned:
        assert selections[0] == expected


async def test_planning_without_a_region_is_now_an_error(anisotropic_zarr):
    """R8.3.  The multiscale fallback -- ``_build_world_to_level_transforms``
    plus ``_build_axis_selections_multiscale`` -- is gone with v1."""
    controller, scene, visual = _viewer(anisotropic_zarr)
    with pytest.raises(RuntimeError, match="no pulled-back region"):
        _plan(controller, scene, visual, with_region=False)


async def test_the_collapsed_axis_comes_from_the_region(anisotropic_zarr):
    """The store has no ``t`` axis metadata, so its axes are the world's
    trailing three plus a leading one -- and the region decides where the
    leading one sits, per level."""
    controller, scene, visual = _viewer(anisotropic_zarr)
    controller.update_slice_indices(scene.id, {0: 2.0})
    for level in (0, 1, 2):
        planned = _plan(controller, scene, visual, with_region=True, level=level)
        assert planned, f"level {level} planned nothing"
        for _scale_index, selections in planned:
            assert selections[0] == 2


async def test_a_level_transform_that_leaves_an_axis_alone_is_honoured(
    anisotropic_zarr,
):
    """The point of the anisotropic fixture: ``z`` is not downsampled, so its
    window is not divided by a power of two at the coarser levels.  The
    per-level answer comes from the per-level transform."""
    controller, scene, visual = _viewer(anisotropic_zarr)
    controller.update_slice_indices(scene.id, {0: 0.0})
    windows = {}
    for level in (0, 1, 2):
        planned = _plan(controller, scene, visual, with_region=True, level=level)
        # The z window spans the whole (undownsampled) axis at every level.
        windows[level] = {selections[1] for _s, selections in planned}
    assert windows[0] == windows[1] == windows[2]


async def test_the_visual_carries_its_stores_level_systems(anisotropic_zarr):
    """The pull-back needs a coordinate system per level, and a v2 transform
    between them.  The store still holds its level transforms as v1 matrices,
    which its brick grid and shader uniforms also read; the v2 form is derived
    against the store's stored per-level systems, so the ids are stable."""
    controller, _scene, visual = _viewer(anisotropic_zarr)
    spaces = controller.render_spaces(visual.id)
    assert len(spaces.data_levels) == 3
    assert len(spaces.level_transforms) == 3
    assert len({system.id for system in spaces.data_levels}) == 3
    for level, transform in enumerate(spaces.level_transforms):
        assert transform.input_coordinate_system == spaces.data_levels[level].id
        assert transform.output_coordinate_system == spaces.data.id
    np.testing.assert_allclose(spaces.level_transforms[0].matrix, np.eye(5))
    # level 2 -> level 0: t and z untouched, y and x by four.
    np.testing.assert_allclose(
        np.diag(spaces.level_transforms[2].matrix), [1.0, 1.0, 4.0, 4.0, 1.0]
    )


async def test_a_transposed_display_order_fetches_identical_bricks(anisotropic_zarr):
    """Design 3.14: a transpose costs no refetch.

    It changes neither which axes are displayed nor their extents, so every
    ``ChunkRequest`` is identical and the GPU cache stays valid.  Only the
    rendered system, the embedding and the node matrices are rebuilt.

    Before this, the brick grid was built in *display* order while
    ``axis_selections`` was assembled ascending, so a transposed tuple rotated
    the grid against its own uploaded array.
    """
    controller, scene, visual = _viewer(anisotropic_zarr)
    controller.update_slice_indices(scene.id, {0: 1.0})
    straight = _plan(controller, scene, visual, with_region=True, level=1)

    controller.update_displayed_axes(scene.id, (3, 2, 1))
    transposed = _plan(controller, scene, visual, with_region=True, level=1)

    assert straight == transposed


async def test_the_brick_grid_is_built_in_fetch_order(anisotropic_zarr):
    """The grid's axes must match the array's, which numpy gives ascending."""
    from cellier.render.visuals._image import _fetch_order

    assert _fetch_order((3, 2, 1)) == (1, 2, 3)
    controller, scene, visual = _viewer(anisotropic_zarr)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    controller.update_displayed_axes(scene.id, (3, 2, 1))
    # z is not downsampled, y and x halve: an ascending grid over (z, y, x)
    # keeps z's extent constant across levels.
    shapes = gfx._volume_geometry.level_shapes
    assert [s[0] for s in shapes] == [8, 8, 8]
    assert [s[1] for s in shapes] == [16, 8, 4]


@pytest.mark.parametrize(
    ("slice_position", "expected_plane"),
    [(1.0, 1), (1.4, 1), (1.5, 2), (1.79, 2), (2.0, 2)],
)
async def test_the_pick_names_the_plane_the_plan_fetched(
    anisotropic_zarr, slice_position, expected_plane
):
    """``pick_collapsed_indices`` rounds the way the fetch did, half-up.

    The multiscale families never populated ``_collapsed_indices`` -- it is
    declared, read by the node matrix, and never written -- so the pick path
    derives its answer from the level-0 box instead, through the same
    ``axis_selections_from_box`` the fetch uses.  Half-integer positions are
    the interesting ones: flooring the raw position would name plane 1 for
    every row here, and the slicer draws plane 2 for the last three.
    """
    controller, scene, visual = _viewer(anisotropic_zarr)
    controller.update_slice_indices(scene.id, {0: slice_position})
    _plan(controller, scene, visual, with_region=True, level=0)

    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    assert gfx.pick_collapsed_indices() == {0: expected_plane}


async def test_an_unplanned_multiscale_visual_reports_nothing(anisotropic_zarr):
    """No plan, no answer -- the controller falls back to the dims state.

    Returning a plane it has not drawn would be worse than declining: the
    fallback at least rounds the same way, and says where it came from.
    """
    controller, scene, visual = _viewer(anisotropic_zarr)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    assert gfx.pick_collapsed_indices() is None
