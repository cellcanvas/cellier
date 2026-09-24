"""Render + commit tests for ``GFXMultiscaleLabelVisual`` (Phase 3).

Drives a real multiscale label pyramid through the controller: reslice commits
bricks/tiles to the GPU (the ``on_data_ready`` / material-build path that the
planning-only tests never reach) and the offscreen harness reads back pixels.
Uses the shared fixtures in ``conftest.py`` (``render_scene``, ``reslice``,
``multiscale_labels_store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.events import DimsUpdateEvent
from cellier.events._events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.scene import spatial_axes
from cellier.transform import Axis, ByDimensionTransform
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS

if TYPE_CHECKING:
    from cellier.render.visuals._label_multiscale import GFXMultiscaleLabelVisual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gfx_visual(controller, scene_id, visual_id) -> GFXMultiscaleLabelVisual:
    return controller._render_manager._scenes[scene_id].get_visual(visual_id)


def _opaque_colors(frame: np.ndarray) -> np.ndarray:
    opaque = frame[..., 3] > 0
    return np.unique(frame[opaque][:, :3], axis=0)


# ---------------------------------------------------------------------------
# Construction / node hierarchy
# ---------------------------------------------------------------------------


def test_construction_2d_scene_builds_only_2d_node(controller, multiscale_labels_store):
    """A 2D scene builds ``node_2d`` (Group) and defers ``node_3d``."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_2d is not None
    assert gfx._inner_node_2d is not None
    assert gfx.node_3d is None
    # 2D render owns a 2D block cache, not a 3D one.
    assert gfx._block_cache_2d is not None
    assert gfx._block_cache_3d is None


def test_construction_3d_scene_builds_3d_node(controller, multiscale_labels_store):
    """A 3D scene builds ``node_3d`` and its 3D brick cache."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_3d is not None
    assert gfx._inner_node_3d is not None
    assert gfx._block_cache_3d is not None


def test_construction_with_a_broadcast_transform_does_not_misindex_the_store(
    controller, multiscale_labels_store
):
    """A ``zyx`` store broadcast over a leading world ``t`` axis builds cleanly.

    Regression test: ``GFXMultiscaleLabelVisual.from_cellier_model`` used to
    index the store's own (data-space) ``level_shapes`` with **world** axis
    indices directly (``select_axes(level_shapes[k], displayed_axes)``),
    which only worked by coincidence when the store and the world had equal
    rank.  Here the store is ``zyx`` but the world is ``tzyx``, so the
    default 3D ``displayed_axes`` is ``(1, 2, 3)`` -- out of range for a
    3-element store shape -- and this raised ``IndexError`` before the data
    axes were translated via the transform's own
    ``axis_correspondence`` (see ``_world_axes_to_data_axes`` in
    ``render/visuals/_image.py``).
    """
    store = MultiscaleZarrDataStore(
        zarr_path=multiscale_labels_store.zarr_path,
        scale_names=multiscale_labels_store.scale_names,
        level_scales=multiscale_labels_store.level_scales,
        level_translations=multiscale_labels_store.level_translations,
        name="broadcast_labels_store",
    )
    scene = controller.add_scene(
        dim="3d",
        name="scene",
        coordinate_system=(
            Axis(name="t", axis_type="time"),
            *spatial_axes("z", "y", "x"),
        ),
    )
    world = scene.dims.world_coordinate_system
    # The store is constructed without data_coordinate_systems, so
    # _ensure_data_coordinate_systems -- what add_labels_multiscale calls
    # internally -- takes the world's trailing axes (z, y, x) for every
    # level, which is what a real call through the controller would also do.
    controller._ensure_data_coordinate_systems(scene.id, store)
    store_cs = store.data_coordinate_systems[0]
    transform = ByDimensionTransform.from_axis_map(
        store_cs,
        world,
        axis_map={
            store_cs.axis_by_name(name).id: world.axis_by_name(name).id
            for name in ("z", "y", "x")
        },
        broadcast_output_axes=[world.axis_by_name("t").id],
        name="to_world",
    )

    visual = controller.add_labels_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        transform=transform,
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_3d is not None
    # The level-0 shape is the store's own zyx shape, not a wrongly indexed slice
    # of it -- confirming the translation, not just the absence of a crash.
    assert gfx._volume_geometry.level_shapes[0] == (16, 16, 16)


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------


async def test_render_2d_shows_labels(
    controller, render_scene, reslice, multiscale_labels_store
):
    """The 2D tile pyramid commits and renders the two labelled blocks."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)

    assert np.count_nonzero(frame[..., 3]) > 0
    # Two distinct labels under the random colormap -> at least two colours.
    assert len(_opaque_colors(frame)) >= 2

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_2d.tile_manager.tilemap) > 0


async def test_render_3d_commits_bricks_and_draws(
    controller, render_scene, reslice, multiscale_labels_store
):
    """The 3D brick pyramid commits at least one brick and draws opaque pixels."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=1),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_3d.tile_manager.tilemap) > 0

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# Direct colormap mode (binds the direct-LUT textures in get_bindings)
# ---------------------------------------------------------------------------


async def test_render_2d_direct_mode_binds_lut(
    controller, render_scene, reslice, multiscale_labels_store
):
    """2D multiscale labels in direct mode bind the direct-LUT textures.

    Covers the ``label_keys_texture is not None`` branch of
    ``LabelBlockShader.get_bindings`` (``_label_multiscale.py``), unreached by
    the random-mode tests.  The store's labels are 3 and 7.
    """
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            colormap_mode="direct",
            color_dict={
                3: (1.0, 0.0, 0.0, 1.0),
                7: (0.0, 0.0, 1.0, 1.0),
            },
        ),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_3d_direct_mode_binds_lut(
    controller, render_scene, reslice, multiscale_labels_store
):
    """3D multiscale labels in direct mode bind the direct-LUT textures.

    Covers the direct-mode branch of ``LabelVolumeBrickShader.get_bindings``
    (``_label_multiscale.py``), unreached by the random-mode 3D tests.
    """
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            colormap_mode="direct",
            color_dict={
                3: (1.0, 0.0, 0.0, 1.0),
                7: (0.0, 0.0, 1.0, 1.0),
            },
            force_level=1,
        ),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# Appearance + visibility updates (drive the render-layer handlers directly)
# ---------------------------------------------------------------------------


async def test_salt_change_reseeds_colormap_buffer(
    controller, reslice, multiscale_labels_store
):
    """Changing ``salt`` writes the new seed into the label params buffer."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual.id,
            field_name="salt",
            new_value=42,
            requires_reslice=False,
        )
    )

    assert gfx._salt == 42
    assert int(gfx._label_params_buffer.data["salt"]) == 42


async def test_render_mode_change_updates_3d_material(
    controller, reslice, multiscale_labels_store
):
    """Switching ``render_mode`` pushes onto the live 3D material."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=1),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual.id,
            field_name="render_mode",
            new_value="flat_categorical",
            requires_reslice=False,
        )
    )
    assert gfx.material_3d.render_mode == "flat_categorical"


async def test_opacity_change_applies_to_material(
    controller, reslice, multiscale_labels_store
):
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual.id,
            field_name="opacity",
            new_value=0.25,
            requires_reslice=False,
        )
    )
    assert gfx.material_2d.opacity == 0.25


async def test_visibility_toggle_hides_render(
    controller, render_scene, reslice, multiscale_labels_store
):
    """Hiding the visual empties the rendered frame; re-showing restores it."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
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


# ---------------------------------------------------------------------------
# The per-label outline selection survives a material rebuild
# ---------------------------------------------------------------------------


def _selection_in_texture(material) -> dict[int, int]:
    """Read a material's selection texture back as ``{label: slot}``.

    Reads the *texture*, not ``n_outline_entries``: the entry count lives on
    a ``label_params_buffer`` shared across a visual's materials, so it
    survives a rebuild even when the per-material texture does not.  Only
    the texture says whether the selection is really still there.
    """
    n = int(material.label_params_buffer.data["n_outline_entries"])
    data = material.outline_selection_texture.data
    return {int(data[0, i, 0]): int(data[0, i, 1]) for i in range(n)}


def _selections(gfx) -> list[dict[int, int]]:
    return [
        _selection_in_texture(material)
        for material in (gfx.material_3d, gfx.material_2d)
        if material is not None
    ]


def test_the_label_selection_survives_a_geometry_rebuild(
    controller, multiscale_labels_store
):
    """A rebuild replaces the materials, and with them their selection texture.

    ``rebuild_geometry`` constructs a fresh ``LabelVolumeBrickMaterial``
    whenever the displayed level shapes change, so a selection written
    straight to the GPU and stored nowhere was silently lost -- the visual
    kept its outline but every selected label went back to unselected.  The
    visual records the selection now and puts it back.
    """
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = _gfx_visual(controller, scene.id, visual.id)

    selection = {1: 1, 2: 2}
    controller.set_label_selection(visual.id, selection)
    assert _selection_in_texture(gfx.material_3d) == selection

    material_before = gfx.material_3d
    gfx._rebuild_3d_resources()

    assert gfx.material_3d is not material_before, "the rebuild replaced nothing"
    assert _selection_in_texture(gfx.material_3d) == selection, "selection was lost"


def test_the_selection_is_recorded_on_the_model(controller, multiscale_labels_store):
    """``set_label_selection`` writes the visual, which is what persists it."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    controller.add_canvas(scene_id=scene.id)

    controller.set_label_selection(visual.id, {3: 1})

    assert visual.outline_selected_labels == {3: 1}


# ---------------------------------------------------------------------------
# A visual that starts in 2D can switch to 3D
# ---------------------------------------------------------------------------


def _toggle(controller, scene_id, displayed_axes, slice_indices) -> None:
    """Switch dims the way the Qt and anywidget 2D/3D toggles do."""
    controller._on_dims_update(
        DimsUpdateEvent(
            source_id=controller._id,
            scene_id=scene_id,
            displayed_axes=displayed_axes,
            slice_indices=slice_indices,
        )
    )


async def test_a_2d_start_switches_to_3d_and_back(
    controller, render_scene, reslice, multiscale_labels_store
):
    """Entering 3D builds the volume the 2D construction skipped.

    Regression test: a visual constructed with 2D displayed axes had no 3D
    geometry, and ``build_node`` never built one, so the toggle swapped the
    2D node out for ``None`` and the 3D planner raised ``AttributeError`` on
    the missing geometry.  Starting in 3D was unaffected.
    """
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx = _gfx_visual(controller, scene.id, visual.id)
    active = controller._render_manager._scenes[scene.id]._active_nodes

    _toggle(controller, scene.id, (0, 1, 2), {})
    await reslice(controller, scene.id)

    assert gfx.node_3d is not None
    assert active[visual.id] is gfx.node_3d
    assert np.count_nonzero(render_scene(controller, scene.id)[..., 3]) > 0

    _toggle(controller, scene.id, (1, 2), {0: 7.5})
    await reslice(controller, scene.id)

    assert active[visual.id] is gfx.node_2d
    assert np.count_nonzero(render_scene(controller, scene.id)[..., 3]) > 0


async def test_the_3d_node_built_on_switch_takes_the_current_state(
    controller, reslice, multiscale_labels_store
):
    """The late 3D material carries the appearance and selection set in 2D."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(opacity=0.5),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx = _gfx_visual(controller, scene.id, visual.id)
    selection = {3: 1}
    controller.set_label_selection(visual.id, selection)

    _toggle(controller, scene.id, (0, 1, 2), {})

    assert gfx.material_3d.opacity == 0.5
    assert _selection_in_texture(gfx.material_3d) == selection
