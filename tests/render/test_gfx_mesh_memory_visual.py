"""Tests for GFXMeshMemoryVisual."""

import asyncio
from uuid import uuid4

import numpy as np
import pygfx as gfx
import pytest

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.mesh._mesh_requests import MeshData, MeshSliceRequest
from cellier.events._events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    PickWriteChangedEvent,
    TransformChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
from cellier.transform import AffineTransform
from cellier.visuals import (
    MeshFlatAppearance,
    MeshPhongAppearance,
    MeshVisual,
)


def _appearance_event(field, value):
    return AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=uuid4(),
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


def _aabb_event(field, value):
    return AABBChangedEvent(
        source_id=uuid4(), visual_id=uuid4(), field_name=field, new_value=value
    )


def _dims_state(displayed=(0, 1, 2), sliced=None):
    return DimsState(
        axis_labels=tuple(str(i) for i in range(3)),
        selection=AxisAlignedSelectionState(
            displayed_axes=displayed, slice_indices=sliced or {}
        ),
    )


def _store():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return MeshMemoryStore(positions=pos, indices=idx)


def _visual(store, appearance=None):
    if appearance is None:
        appearance = MeshFlatAppearance()
    model = MeshVisual(name="test", data_store_id=str(store.id), appearance=appearance)
    return GFXMeshMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.positions.shape[1]),
    )


# ── Construction ──────────────────────────────────────────────────────────────


def test_single_node_for_both_modes():
    v = _visual(_store())
    assert v.node_2d is v.node
    assert v.node_3d is v.node


def test_pick_write_enabled_by_default():
    v = _visual(_store())
    assert v._material_2d.pick_write is True
    assert v._material_3d.pick_write is True


def test_pick_write_follows_model_flag():
    store = _store()
    model = MeshVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=MeshFlatAppearance(),
        pick_write=False,
    )
    v = GFXMeshMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.positions.shape[1]),
    )
    assert v._material_2d.pick_write is False
    assert v._material_3d.pick_write is False


def test_node_2d_is_node_3d():
    v = _visual(_store())
    assert v.node_2d is v.node_3d


# ── get_node_for_dims ─────────────────────────────────────────────────────────


def test_get_node_for_dims_always_returns_same_node():
    v = _visual(_store())
    assert v.get_node_for_dims((0, 1, 2)) is v.node
    assert v.get_node_for_dims((1, 2)) is v.node


def test_get_node_for_dims_updates_last_displayed_axes():
    v = _visual(_store())
    assert v._last_displayed_axes is None
    v.get_node_for_dims((1, 2))
    assert v._last_displayed_axes == (1, 2)


def test_get_node_for_dims_no_matrix_update_on_repeat():
    """Second call with same axes must not call _update_node_matrix."""
    v = _visual(_store())
    v.get_node_for_dims((1, 2))
    original_matrix = v.node.local.matrix.copy()
    v.get_node_for_dims((1, 2))  # same axes — should be a no-op
    np.testing.assert_array_equal(v.node.local.matrix, original_matrix)


# ── on_data_ready / on_data_ready_2d ─────────────────────────────────────────


def _make_batch(store, displayed=(0, 1, 2), sliced=None):
    if sliced is None:
        sliced = {}
    sid = uuid4()
    req = MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=displayed,
        slice_indices=sliced,
    )
    data = asyncio.run(store.get_data(req))
    return [(req, data)]


def test_on_data_ready_applies_3d_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready(_make_batch(s, displayed=(0, 1, 2)))
    assert v.node.material is v._material_3d


def test_on_data_ready_2d_applies_2d_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        assert v.node.material is v._material_2d


def test_empty_slab_applies_empty_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 1000}))
    assert v.node.material is v._empty_material


def test_phong_appearance_builds_phong_material():
    v = _visual(_store(), appearance=MeshPhongAppearance())
    assert isinstance(v._material_3d, gfx.MeshPhongMaterial)
    assert isinstance(v._material_2d, gfx.MeshBasicMaterial)


def test_flat_appearance_builds_basic_material():
    v = _visual(_store(), appearance=MeshFlatAppearance())
    assert isinstance(v._material_3d, gfx.MeshBasicMaterial)
    assert isinstance(v._material_2d, gfx.MeshBasicMaterial)


def test_transparent_3d_material_uses_blend():
    v = _visual(_store(), appearance=MeshFlatAppearance(opacity=0.8))
    assert v._material_3d.alpha_mode == "blend"
    assert v._material_3d.depth_test is True
    assert v._material_3d.depth_write is True  # model default; not auto-managed


def test_transparent_3d_material_depth_write_explicit():
    v = _visual(_store(), appearance=MeshFlatAppearance(opacity=0.8, depth_write=False))
    assert v._material_3d.alpha_mode == "blend"
    assert v._material_3d.depth_write is False


def test_opaque_3d_material_uses_solid():
    v = _visual(_store(), appearance=MeshFlatAppearance(opacity=1.0))
    assert v._material_3d.alpha_mode == "solid"
    assert v._material_3d.depth_test is True
    assert v._material_3d.depth_write is True


def test_opacity_event_updates_alpha_mode():
    v = _visual(_store(), appearance=MeshFlatAppearance(opacity=1.0))

    ev_t = AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=v.visual_model_id,
        field_name="opacity",
        new_value=0.8,
        requires_reslice=False,
    )
    v.on_appearance_changed(ev_t)
    assert v._material_3d.alpha_mode == "blend"
    assert v._material_3d.depth_write is True  # unchanged; not auto-managed by opacity

    ev_o = AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=v.visual_model_id,
        field_name="opacity",
        new_value=1.0,
        requires_reslice=False,
    )
    v.on_appearance_changed(ev_o)
    assert v._material_3d.alpha_mode == "solid"
    assert v._material_3d.depth_write is True


# ── Construction errors / material variants / trivia ───────────────────────────


def test_invalid_render_modes_raises():
    store = _store()
    model = MeshVisual(
        name="t", data_store_id=str(store.id), appearance=MeshFlatAppearance()
    )
    with pytest.raises(ValueError, match="render_modes"):
        GFXMeshMemoryVisual(
            visual_model=model,
            render_modes={"4d"},
            transform=AffineTransform.identity(ndim=3),
        )


def test_non_blend_transparency_sets_alpha_mode_directly():
    """A non-``blend`` transparency mode is written straight onto the material.

    Covers the ``transparency_mode != "blend"`` branch of ``_build_material_3d``.
    """
    v = _visual(_store(), appearance=MeshFlatAppearance(transparency_mode="add"))
    assert v._material_3d.alpha_mode == "add"


def test_n_levels_is_one():
    assert _visual(_store()).n_levels == 1


def test_empty_batch_is_noop():
    v = _visual(_store())
    v.on_data_ready([])
    v.on_data_ready_2d([])
    assert v.node.material is v._empty_material


def test_face_index_for_pick_maps_through_original_indices():
    v = _visual(_store())
    assert v.face_index_for_pick(2) == 2  # identity pass-through
    v._original_face_indices = np.array([5, 9, 11], dtype=np.int64)
    assert v.face_index_for_pick(1) == 9
    assert v.face_index_for_pick(999) == 999


# ── GFXVisual protocol stubs + slice-request planning ──────────────────────────


def test_protocol_node_accessors():
    v = _visual(_store())
    assert v.has_node("3d") is True
    assert v.get_node("3d") is v.node
    assert v.build_node("3d", None, (0, 1, 2), None, None) is v.node
    assert v.rebuild_node_geometry("3d", (0, 1, 2), None, None) is v.node
    v.on_stacked_axes_changed((0,))


def test_build_slice_request_updates_matrix_on_axis_change():
    v = _visual(_store())
    reqs = v.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=0.0,
        screen_height_px=100.0,
        dims_state=_dims_state(displayed=(0, 1, 2)),
    )
    assert len(reqs) == 1
    assert v._last_displayed_axes == (0, 1, 2)


def test_build_slice_request_2d_updates_matrix_on_axis_change():
    v = _visual(_store())
    reqs = v.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=_dims_state(displayed=(1, 2), sliced={0: 0}),
    )
    assert len(reqs) == 1
    assert v._last_displayed_axes == (1, 2)


def _mesh_data(color_mode="vertex", colors="default"):
    if colors == "default":
        colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
    return MeshData(
        request_id=uuid4(),
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int32),
        normals=np.tile([0.0, 0.0, 1.0], (3, 1)).astype(np.float32),
        colors=colors,
        color_mode=color_mode,
        is_empty=False,
    )


def test_commit_uploads_colors_without_touching_color_mode():
    """The store's colours are uploaded; the declared mode is untouched.

    Inverted at the D20 extension to mesh.  This previously asserted the
    opposite -- that incoming ``"face"`` data overrode the appearance's
    declaration on both live materials.
    """
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    v.on_data_ready([(None, _mesh_data(color_mode="vertex"))])

    assert v.node.geometry.colors is not None
    assert v._color_mode == "vertex"
    assert v._material_3d.color_mode == "vertex"
    assert v._material_2d.color_mode == "vertex"


def test_declared_uniform_survives_store_colors():
    """Store carries colours, appearance says uniform -> stays uniform (D20)."""
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="uniform"))
    v.on_data_ready([(None, _mesh_data(color_mode="vertex"))])

    assert v._color_mode == "uniform"
    assert v._material_3d.color_mode == "uniform"
    assert v._material_2d.color_mode == "uniform"


def test_declared_colored_without_colors_raises():
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    with pytest.raises(ValueError, match="carries no colors"):
        v.on_data_ready([(None, _mesh_data(colors=None))])


def test_declared_layout_contradicting_the_store_raises():
    """Declaring "vertex" against per-face colours is refused, not corrected.

    pygfx renders a mismatched buffer without complaint -- verified -- so
    an unchecked contradiction is a wrong picture with no signal.
    """
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    with pytest.raises(ValueError, match="colors_layout is 'face'"):
        v.on_data_ready([(None, _mesh_data(color_mode="face"))])


def test_empty_slice_does_not_raise_on_declared_colors():
    """An empty slice carries no colours by construction; not a bug."""
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    data = MeshData(
        request_id=uuid4(),
        positions=np.zeros((3, 3), dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int32),
        normals=np.zeros((3, 3), dtype=np.float32),
        colors=None,
        color_mode="vertex",
        is_empty=True,
    )
    v.on_data_ready([(None, data)])
    assert v._is_empty


# ── Event handlers — appearance ────────────────────────────────────────────────


def test_on_appearance_changed_shared_fields():
    v = _visual(_store(), appearance=MeshFlatAppearance())
    v.on_appearance_changed(_appearance_event("color", (1.0, 0.0, 0.0, 1.0)))
    np.testing.assert_allclose(v._material_3d.color, (1.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(v._material_2d.color, (1.0, 0.0, 0.0, 1.0))

    v.on_appearance_changed(_appearance_event("color_mode", "vertex"))
    assert v._material_3d.color_mode == "vertex"
    assert v._color_mode == "vertex"

    v.on_appearance_changed(_appearance_event("side", "front"))
    assert v._material_3d.side == "front"

    v.on_appearance_changed(_appearance_event("depth_test", False))
    assert v._material_3d.depth_test is False
    assert v._material_2d.depth_test is False

    v.on_appearance_changed(_appearance_event("depth_write", False))
    assert v._material_3d.depth_write is False
    assert v._material_2d.depth_write is False

    v.on_appearance_changed(_appearance_event("depth_compare", "<="))
    assert v._material_3d.depth_compare == "<="
    assert v._material_2d.depth_compare == "<="

    v.on_appearance_changed(_appearance_event("render_order", 5))
    assert v.node.render_order == 5


def test_on_appearance_changed_transparency_non_blend_and_blend():
    v = _visual(_store(), appearance=MeshFlatAppearance(opacity=0.8))
    v.on_appearance_changed(_appearance_event("transparency_mode", "add"))
    assert v._material_3d.alpha_mode == "add"
    assert v._material_2d.alpha_mode == "add"

    # Switching back to blend re-derives the alpha mode from opacity.
    v.on_appearance_changed(_appearance_event("transparency_mode", "blend"))
    assert v._material_3d.alpha_mode == "blend"  # opacity 0.8 -> blend


def test_on_appearance_changed_flat_only_fields():
    v = _visual(_store(), appearance=MeshFlatAppearance())
    v.on_appearance_changed(_appearance_event("wireframe", True))
    assert v._material_3d.wireframe is True
    v.on_appearance_changed(_appearance_event("wireframe_thickness", 3.0))
    assert v._material_3d.wireframe_thickness == pytest.approx(3.0)


def test_on_appearance_changed_phong_only_fields():
    v = _visual(_store(), appearance=MeshPhongAppearance())
    v.on_appearance_changed(_appearance_event("shininess", 50.0))
    assert v._material_3d.shininess == pytest.approx(50.0)
    v.on_appearance_changed(_appearance_event("flat_shading", True))
    assert v._material_3d.flat_shading is True


def test_on_visibility_changed_toggles_node():
    v = _visual(_store())
    v.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=uuid4(), visible=False
        )
    )
    assert v.node.visible is False


def test_on_pick_write_changed_applies_to_both_materials():
    v = _visual(_store())
    v.on_pick_write_changed(
        PickWriteChangedEvent(source_id=uuid4(), visual_id=uuid4(), pick_write=False)
    )
    assert v._material_3d.pick_write is False
    assert v._material_2d.pick_write is False


def test_on_transform_changed_updates_matrix():
    v = _visual(_store())
    v.get_node_for_dims((0, 1, 2))
    new_tf = AffineTransform.identity(ndim=3)
    v.on_transform_changed(
        TransformChangedEvent(
            source_id=uuid4(), scene_id=uuid4(), visual_id=uuid4(), transform=new_tf
        )
    )
    assert v._transform is new_tf


def test_on_aabb_changed_stores_params_without_line():
    v = _visual(_store())
    v.on_aabb_changed(_aabb_event("enabled", True))
    v.on_aabb_changed(_aabb_event("color", "red"))
    v.on_aabb_changed(_aabb_event("line_width", 3.0))
    assert v._aabb_enabled is True
    assert v._aabb_color == "red"
    assert v._aabb_line_width == 3.0


def test_on_aabb_changed_applies_to_line_when_present():
    v = _visual(_store())
    v._aabb_line = gfx.Line(
        gfx.Geometry(positions=np.zeros((2, 3), dtype=np.float32)),
        gfx.LineSegmentMaterial(),
    )
    v.on_aabb_changed(_aabb_event("enabled", False))
    assert v._aabb_line.visible is False
    v.on_aabb_changed(_aabb_event("color", (0.0, 1.0, 0.0, 1.0)))
    np.testing.assert_allclose(v._aabb_line.material.color, (0.0, 1.0, 0.0, 1.0))
    v.on_aabb_changed(_aabb_event("line_width", 4.0))
    assert v._aabb_line.material.thickness == pytest.approx(4.0)


def test_tick_is_noop():
    _visual(_store()).tick()


# ── Rendered output (controller-driven) ────────────────────────────────────────


async def test_render_2d_draws_pixels(controller, render_scene, reslice):
    pos = np.array([[0, 2, 2], [0, 2, 28], [0, 20, 2], [0, 20, 28]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    store = MeshMemoryStore(positions=pos, indices=idx)
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_mesh(
        data=store,
        scene_id=scene.id,
        appearance=MeshFlatAppearance(color=(1.0, 0.0, 0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_3d_draws_pixels(controller, render_scene, reslice):
    pos = np.array([[2, 2, 2], [2, 2, 28], [2, 20, 2], [10, 20, 28]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    store = MeshMemoryStore(positions=pos, indices=idx)
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_mesh(
        data=store,
        scene_id=scene.id,
        appearance=MeshFlatAppearance(color=(0.0, 1.0, 0.0, 1.0), side="both"),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0
