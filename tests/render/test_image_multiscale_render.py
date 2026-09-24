"""Render + commit tests for ``GFXMultiscaleImageVisual`` (Phase 3).

Exercises the multiscale image commit path -- tile/brick upload, LUT rebuild,
and ``build_material`` -- that the planning-only tests never reach, then reads
back pixels through the offscreen harness.  Shared fixtures come from
``conftest.py`` (``controller``, ``render_scene``, ``reslice``,
``multiscale_image_store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

from cellier.events._events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.scene import spatial_axes
from cellier.transform import (
    Axis,
    AxisCoordinates,
    ByDimensionTransform,
    NonUniformAxisTransform,
)
from cellier.visuals import MultiscaleImageSingleAppearance
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from tests._gpu_budget import SMALL_BUDGETS

if TYPE_CHECKING:
    from cellier.render.visuals._image import GFXMultiscaleImageVisual


def _gfx_visual(controller, scene_id, visual_id) -> GFXMultiscaleImageVisual:
    return controller._render_manager._scenes[scene_id].get_visual(visual_id)


def _add(controller, scene_id, store, appearance, block_size=8, **kwargs):
    return controller.add_image_multiscale(
        data=store,
        scene_id=scene_id,
        appearance=appearance,
        render_config=MultiscaleImageRenderConfig(
            **SMALL_BUDGETS, block_size=block_size
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_2d_builds_2d_material(controller, multiscale_image_store):
    """A 2D scene builds ``node_2d`` + a 2D image material and tile cache."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_2d is not None
    assert gfx.material_2d is not None
    assert gfx.material_3d is None
    assert gfx._block_cache_2d is not None


def test_construction_3d_builds_3d_material(controller, multiscale_image_store):
    """A 3D scene builds ``node_3d`` + a volume material and brick cache."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_3d is not None
    assert gfx.material_3d is not None
    assert gfx._block_cache_3d is not None


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------


async def test_render_2d_commits_tiles_and_draws(
    controller, render_scene, reslice, multiscale_image_store
):
    """The 2D pyramid commits tiles (LUT + material build) and draws pixels."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_2d.tile_manager.tilemap) > 0

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_3d_mip_commits_bricks_and_draws(
    controller, render_scene, reslice, multiscale_image_store
):
    """The 3D pyramid commits bricks and MIP-renders the bright interior."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(force_level=1),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_3d.tile_manager.tilemap) > 0

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# A non-affine (ByDimensionTransform) data -> world transform
# ---------------------------------------------------------------------------


def _nonuniform_z_transform(controller, scene_id, store) -> ByDimensionTransform:
    """A ``data -> world`` transform that routes ``z`` through a lookup table.

    Deliberately a non-physical axis choice (real irregular sampling belongs
    on time) -- the point is only that *some* axis is a
    ``NonUniformAxisTransform`` leaf, which makes the whole transform a
    ``ByDimensionTransform`` with no ``.linear`` / ``.translation`` and no
    ``.inverse()``.  ``y`` and ``x`` stay plain scale=1, so the store's
    (16, 16, 16) / (8, 8, 8) pyramid is otherwise unchanged.
    """
    controller._ensure_data_coordinate_systems(scene_id, store)
    data = store.data_coordinate_systems[0]
    world = controller._model.scenes[scene_id].dims.world_coordinate_system
    z_leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=tuple(float(i) for i in range(16))),
        input_coordinate_system=data.axis_by_name("z").id,
        output_coordinate_system=world.axis_by_name("z").id,
    )
    return ByDimensionTransform.from_axis_map(
        data,
        world,
        axis_map={
            data.axis_by_name(name).id: world.axis_by_name(name).id
            for name in ("z", "y", "x")
        },
        scale={data.axis_by_name(name).id: 1.0 for name in ("y", "x")},
        axis_transforms={data.axis_by_name("z").id: z_leaf},
        name="to_world",
    )


async def test_reslice_2d_with_a_nonuniform_axis_transform(
    controller, render_scene, reslice, multiscale_image_store
):
    """A 2D reslice does not need ``.linear`` on the ``data -> world`` transform.

    Regression test: the 2D planner's ``voxel_width`` computation
    (``_displayed_submatrix``) read ``transform.linear`` directly, which does
    not exist on a ``ByDimensionTransform`` -- the type every visual sharing
    the non-uniform time axis carries, whether or not it broadcasts.
    """
    scene = controller.add_scene(dim="2d", name="scene")
    transform = _nonuniform_z_transform(controller, scene.id, multiscale_image_store)
    visual = controller.add_image_multiscale(
        data=multiscale_image_store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        transform=transform,
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_2d.tile_manager.tilemap) > 0


def _tzyx_store_with_a_nonuniform_t(controller, scene_id, tmp_path):
    """A ``tzyx`` store whose ``t`` (collapsed in 3D) is a non-uniform leaf.

    ``z``/``y``/``x`` stay plain affine and displayed, so 3D display is
    exactly the ``multiscale_image_store`` shape; ``t`` is the axis the
    real light-sheet viewer collapses in 3D too, which is what makes this
    the representative case rather than ``_nonuniform_z_transform``'s
    (which puts the leaf on an axis 3D mode would display, and correctly
    raises ``NonAffineTransformError`` there -- see the design doc, "The GPU
    still gets a matrix").
    """
    import tensorstore as ts

    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore

    for name, shape in (("s0", (4, 16, 16, 16)), ("s1", (4, 8, 8, 8))):
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(tmp_path / name)},
            "metadata": {
                "shape": list(shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [max(1, s // 2) for s in shape]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        arr = np.ones(shape, dtype=np.float32)
        ts.open(spec).result()[...].write(arr).result()
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name="tzyx_store",
    )
    # The store is constructed without data_coordinate_systems.  The world
    # already has (t, z, y, x), so _ensure_data_coordinate_systems gives
    # every level exactly those trailing axes, names and types included.
    controller._ensure_data_coordinate_systems(scene_id, store)
    data = store.data_coordinate_systems[0]
    world = controller._model.scenes[scene_id].dims.world_coordinate_system
    t_leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=(0.0, 1.0, 2.5, 4.0)),
        input_coordinate_system=data.axis_by_name("t").id,
        output_coordinate_system=world.axis_by_name("t").id,
    )
    transform = ByDimensionTransform.from_axis_map(
        data,
        world,
        axis_map={
            data.axis_by_name(name).id: world.axis_by_name(name).id
            for name in ("t", "z", "y", "x")
        },
        scale={data.axis_by_name(name).id: 1.0 for name in ("z", "y", "x")},
        axis_transforms={data.axis_by_name("t").id: t_leaf},
        name="to_world",
    )
    return store, transform


async def test_reslice_3d_with_a_nonuniform_axis_transform(
    controller, render_scene, reslice, tmp_path
):
    """A 3D reslice does not need ``.inverse()`` on the ``data -> world`` transform.

    Regression test: the camera/frustum pull-back (``_to_level0_displayed``,
    formerly via ``_rendered_to_level0``) composed
    ``rendered_to_world.then(transform.inverse())``.
    ``NonUniformAxisTransform.inverse()`` is structurally ``None`` (the map
    is invertible; its inverse is not itself a transform of that class), so
    every ``ByDimensionTransform`` carrying one raised before the camera
    pull-back went through ``imap_coordinates`` in two steps instead.  ``t``
    is collapsed (not displayed) in 3D, matching the real light-sheet viewer.
    """
    scene = controller.add_scene(
        dim="3d",
        name="scene",
        coordinate_system=(
            Axis(name="t", axis_type="time"),
            *spatial_axes("z", "y", "x"),
        ),
    )
    store, transform = _tzyx_store_with_a_nonuniform_t(controller, scene.id, tmp_path)
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        transform=transform,
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_3d.tile_manager.tilemap) > 0


async def test_slider_positions_on_the_same_frame_share_brick_keys(
    controller, render_scene, reslice, tmp_path
):
    """Two slider positions that fetch the same frame reuse each other's bricks.

    The block cache key is the collapsed-axis selection the fetch uses, not
    the continuous pulled-back position.  ``t = 1.2`` and ``t = 0.8`` both
    fetch frame 1 of the ``(0, 1, 2.5, 4)`` lookup table; with a continuous
    key the second reslice missed every brick and fetched them all again.
    """
    scene = controller.add_scene(
        dim="3d",
        name="scene",
        coordinate_system=(
            Axis(name="t", axis_type="time"),
            *spatial_axes("z", "y", "x"),
        ),
    )
    store, transform = _tzyx_store_with_a_nonuniform_t(controller, scene.id, tmp_path)
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        transform=transform,
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = _gfx_visual(controller, scene.id, visual.id)

    controller.update_slice_indices(scene.id, {0: 1.2})
    await reslice(controller, scene.id)
    assert gfx.slots[0]._current_slice_coord_3d == ((0, 1),)
    assert gfx._last_plan_stats["total_required"] > 0

    controller.update_slice_indices(scene.id, {0: 0.8})
    await reslice(controller, scene.id)
    assert gfx.slots[0]._current_slice_coord_3d == ((0, 1),)
    # The chunk scheduler's view of the pass: every wanted brick already had a
    # record, so nothing new was queued.
    scheduler = controller._render_manager.scheduler
    desired, new = scheduler.core.pass_stats(gfx.slots[0].residency_3d().cache_id)
    stats = gfx._last_plan_stats
    assert desired == stats["n_backstop"] + stats["n_target"]
    assert new == 0


# ---------------------------------------------------------------------------
# Appearance updates (GPU-only handlers) + visibility
# ---------------------------------------------------------------------------


def _appearance_event(visual_id, field, value):
    return AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=visual_id,
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


async def test_colormap_and_clim_updates_apply_to_materials(
    controller, reslice, multiscale_image_store
):
    """``color_map``, ``clim`` and ``opacity`` changes push onto the 2D material.

    Driven through the controller: the render visual re-reads the model, so
    the change has to land there first.
    """
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)

    controller.update_single_appearance_field(visual.id, "color_map", "magma")
    assert gfx.material_2d.map is not None

    controller.update_single_appearance_field(visual.id, "clim", (10.0, 200.0))
    assert tuple(gfx.material_2d.clim) == (10.0, 200.0)

    controller.update_single_appearance_field(visual.id, "opacity", 0.5)
    assert gfx.material_2d.opacity == 0.5


async def test_render_mode_and_iso_threshold_update_3d_material(
    controller, reslice, multiscale_image_store
):
    """``render_mode`` / ``iso_threshold`` push onto the live volume material."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(force_level=1),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.2
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)

    controller.update_single_appearance_field(visual.id, "iso_threshold", 0.7)
    assert gfx.material_3d.threshold == pytest.approx(0.7)

    controller.update_single_appearance_field(visual.id, "render_mode", "mip")
    assert gfx.material_3d.render_mode == "mip"


async def test_attenuation_update_3d_material(
    controller, reslice, multiscale_image_store
):
    """An ``attenuation`` change reads and writes the brick material's uniform.

    Covers the ``attenuation`` getter/setter on ``MultiscaleVolumeBrickMaterial``
    (``_multiscale_volume_brick.py``), used by ``"attenuated_mip"`` mode and
    unreached by the other 3D tests.
    """
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(attenuation=1.0, force_level=1),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="attenuated_mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    controller.update_appearance_field(visual.id, "attenuation", 2.5)
    assert gfx.material_3d.attenuation == 2.5


async def test_visibility_toggle_hides_multiscale_image(
    controller, render_scene, reslice, multiscale_image_store
):
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=visual.id, visible=False
        )
    )
    hidden = render_scene(controller, scene.id)
    assert np.count_nonzero(hidden[..., 3]) == 0

    gfx.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=visual.id, visible=True
        )
    )
    shown = render_scene(controller, scene.id)
    assert np.count_nonzero(shown[..., 3]) > 0


def test_volume_material_is_brick_material(controller, multiscale_image_store):
    """The 3D inner node uses the multiscale brick volume material."""
    from cellier.render.visuals._image import MultiscaleVolumeBrickMaterial

    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx_visual = _gfx_visual(controller, scene.id, visual.id)
    assert isinstance(gfx_visual.material_3d, MultiscaleVolumeBrickMaterial)
    # A real colormap map is attached.
    assert gfx_visual.material_3d.map is not None
