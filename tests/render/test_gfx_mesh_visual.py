"""Tests for GFXMeshVisual: nodes, materials, upload, bounds, events.

Loading through the scheduler is in ``test_mesh_loading.py``; the display
rule itself in ``test_level_residency.py``.
"""

import asyncio
from uuid import uuid4

import numpy as np
import pygfx as gfx
import pytest

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.mesh._mesh_requests import MeshData
from cellier.events._events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    MeshPickInfo,
    PickWriteChangedEvent,
    TransformChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.render.scheduling import ChunkClass, PlanMode, is_chunked_visual
from cellier.render.visuals._mesh import FINE_LEVEL, GFXMeshVisual
from cellier.visuals import (
    MeshFlatAppearance,
    MeshPhongAppearance,
    MeshVisual,
)
from tests._planning import planned_requests_2d, planned_requests_3d
from tests._v2 import Context, identity

KEY_3D = ((0, 1, 2), None, None)
KEY_2D = ((1, 2), None, None)


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


def _dims_state(displayed=(0, 1, 2)):
    return DimsState(
        axis_labels=tuple(str(i) for i in range(3)),
        selection=AxisAlignedSelectionState(
            displayed_axes=displayed,
        ),
    )


def _store():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return MeshMemoryStore(positions=pos, indices=idx)


def _visual(store, appearance=None, transform=None, **kwargs):
    if appearance is None:
        appearance = MeshFlatAppearance()
    model = MeshVisual(name="test", data_store_id=str(store.id), appearance=appearance)
    return GFXMeshVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=transform
        if transform is not None
        else identity(store.positions.shape[1]),
        axis_extents=store.axis_extents,
        **kwargs,
    )


def _placed(store, displayed=(0, 1, 2), slice_indices=None, appearance=None):
    """A visual with its coordinate systems, as the controller places it."""
    ctx = Context(store.ndim, displayed_axes=displayed, slice_indices=slice_indices)
    v = _visual(store, appearance=appearance, transform=ctx.transform)
    ctx.place(v)
    v.get_node_for_dims(displayed)
    return v, ctx


def _read_3d(store, v, ctx):
    request = planned_requests_3d(
        v,
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=0.0,
        screen_height_px=100.0,
        dims_state=_dims_state((0, 1, 2)),
        selection=ctx.selection,
    )[0]
    return request, asyncio.run(store.get_data(request))


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


# ── Construction ──────────────────────────────────────────────────────────────


def test_it_loads_through_the_scheduler():
    v = _visual(_store())
    assert is_chunked_visual(v)
    assert list(v.residencies()) == [v._residency.cache_id]


def test_one_node_per_dimensionality():
    v = _visual(_store())
    assert isinstance(v.node_2d, gfx.Group)
    assert isinstance(v.node_3d, gfx.Group)
    assert v.node_2d is not v.node_3d
    fine = v._levels[FINE_LEVEL]
    assert fine.mesh_3d.parent is v.node_3d
    assert fine.group_2d.parent is v.node_2d
    assert fine.fill.parent is fine.group_2d


def test_a_mode_left_out_has_no_node():
    store = _store()
    model = MeshVisual(
        name="t", data_store_id=str(store.id), appearance=MeshFlatAppearance()
    )
    v = GFXMeshVisual(visual_model=model, render_modes={"3d"}, transform=identity(3))
    assert v.node_2d is None
    assert v.has_node("3d") and not v.has_node("2d")
    assert v.get_node_for_dims((1, 2)) is None


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
    v = GFXMeshVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=identity(store.positions.shape[1]),
    )
    assert v._material_2d.pick_write is False
    assert v._material_3d.pick_write is False


def test_a_hidden_model_starts_with_hidden_nodes():
    store = _store()
    model = MeshVisual(
        name="t",
        data_store_id=str(store.id),
        appearance=MeshFlatAppearance(visible=False),
    )
    v = GFXMeshVisual(
        visual_model=model, render_modes={"2d", "3d"}, transform=identity(3)
    )
    assert v.node_2d.visible is False
    assert v.node_3d.visible is False


def test_invalid_render_modes_raises():
    store = _store()
    model = MeshVisual(
        name="t", data_store_id=str(store.id), appearance=MeshFlatAppearance()
    )
    with pytest.raises(ValueError, match="render_modes"):
        GFXMeshVisual(
            visual_model=model,
            render_modes={"4d"},
            transform=identity(3),
        )


def test_n_levels_is_one():
    assert _visual(_store()).n_levels == 1


# ── get_node_for_dims ─────────────────────────────────────────────────────────


def test_get_node_for_dims_returns_the_node_of_the_dimensionality():
    v = _visual(_store())
    assert v.get_node_for_dims((0, 1, 2)) is v.node_3d
    assert v.get_node_for_dims((1, 2)) is v.node_2d


def test_get_node_for_dims_updates_last_displayed_axes():
    v = _visual(_store())
    assert v._last_displayed_axes is None
    v.get_node_for_dims((1, 2))
    assert v._last_displayed_axes == (1, 2)


def test_protocol_node_accessors():
    v = _visual(_store())
    assert v.has_node("3d") is True
    assert v.get_node("3d") is v.node_3d
    assert v.get_node("2d") is v.node_2d
    assert v.build_node("3d", None, (0, 1, 2), None, None) is v.node_3d
    assert v.rebuild_node_geometry("2d", (1, 2), None, None) is v.node_2d


# ── Materials ─────────────────────────────────────────────────────────────────


def test_phong_appearance_builds_phong_material():
    v = _visual(_store(), appearance=MeshPhongAppearance())
    assert isinstance(v._material_3d, gfx.MeshPhongMaterial)
    assert isinstance(v._material_2d, gfx.MeshBasicMaterial)


def test_flat_appearance_builds_basic_material():
    v = _visual(_store(), appearance=MeshFlatAppearance())
    assert isinstance(v._material_3d, gfx.MeshBasicMaterial)
    assert isinstance(v._material_2d, gfx.MeshBasicMaterial)


def test_the_children_use_the_shared_materials():
    v = _visual(_store())
    fine = v._levels[FINE_LEVEL]
    assert fine.mesh_3d.material is v._material_3d
    assert fine.fill.material is v._material_2d


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


def test_non_blend_transparency_sets_alpha_mode_directly():
    v = _visual(_store(), appearance=MeshFlatAppearance(transparency_mode="add"))
    assert v._material_3d.alpha_mode == "add"


def test_opacity_event_updates_alpha_mode():
    v = _visual(_store(), appearance=MeshFlatAppearance(opacity=1.0))
    v.on_appearance_changed(_appearance_event("opacity", 0.8))
    assert v._material_3d.alpha_mode == "blend"
    assert v._material_3d.depth_write is True  # unchanged; not auto-managed
    v.on_appearance_changed(_appearance_event("opacity", 1.0))
    assert v._material_3d.alpha_mode == "solid"


# ── Planning (L1, L4) ─────────────────────────────────────────────────────────


def test_a_plan_is_one_target_key_and_updates_the_matrix():
    store = _store()
    ctx = Context(3, displayed_axes=(0, 1, 2))
    v = _visual(store, transform=ctx.transform)
    ctx.place(v)
    requests = planned_requests_3d(
        v,
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=0.0,
        screen_height_px=100.0,
        dims_state=_dims_state((0, 1, 2)),
        selection=ctx.selection,
    )
    assert len(requests) == 1
    assert requests[0].scale_index == 0
    assert requests[0].output_axes == (2, 1, 0)
    assert v._last_displayed_axes == (0, 1, 2)


def test_a_2d_plan_emits_the_retained_axes_reversed():
    store = _store()
    ctx = Context(3, displayed_axes=(1, 2), slice_indices={0: 0.0})
    v = _visual(store, transform=ctx.transform)
    ctx.place(v)
    requests = planned_requests_2d(
        v,
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=_dims_state((1, 2)),
        selection=ctx.selection,
    )
    assert len(requests) == 1
    assert requests[0].retained_axes == (1, 2)
    assert requests[0].output_axes == (2, 1)
    assert v._last_displayed_axes == (1, 2)


def test_the_request_key_holds_what_a_read_depends_on():
    """L1: same view, same key; another slice position, another key."""
    store = _store()
    keys = []
    for position in (0.0, 0.0, 1.0):
        ctx = Context(3, displayed_axes=(1, 2), slice_indices={0: position})
        v = _visual(store, transform=ctx.transform)
        ctx.place(v)
        request = planned_requests_2d(
            v,
            camera_pos_world=np.zeros(3),
            viewport_width_px=100.0,
            world_width=10.0,
            view_min_world=None,
            view_max_world=None,
            dims_state=_dims_state((1, 2)),
            selection=ctx.selection,
        )[0]
        keys.append(GFXMeshVisual.request_key(request))
    assert keys[0] == keys[1]
    assert hash(keys[0]) == hash(keys[1])
    assert keys[0] != keys[2]


def test_a_single_level_plans_its_level_in_either_mode():
    from cellier.render._requests import ReslicingRequest

    store = _store()
    v, ctx = _placed(store)
    request = ReslicingRequest(
        camera_type="perspective",
        camera_pos=np.zeros(3),
        frustum_corners=None,
        fov_y_rad=1.0,
        screen_size_px=(100.0, 100.0),
        world_extent=(0.0, 0.0),
        dims_state=_dims_state((0, 1, 2)),
        selection=ctx.selection,
        request_id=uuid4(),
        scene_id=uuid4(),
        canvas_id=uuid4(),
        target_visual_ids=None,
    )
    for mode in (PlanMode.FULL, PlanMode.BACKSTOP_ONLY):
        (desired,) = v.plan(request, None, mode)
        assert desired.cache_id == v._residency.cache_id
        assert desired.cls.tolist() == [int(ChunkClass.TARGET)]


# ── Upload (L5) ───────────────────────────────────────────────────────────────


def test_the_upload_copies_no_array():
    """The commit on the UI thread is ``gfx.Geometry`` and bookkeeping."""
    store = _store()
    v, ctx = _placed(store)
    request, data = _read_3d(store, v, ctx)
    v._upload_level(FINE_LEVEL, GFXMeshVisual.request_key(request), data)
    geometry = v._levels[FINE_LEVEL].mesh_3d.geometry
    for name in ("positions", "indices", "normals"):
        uploaded = getattr(geometry, name).data
        assert np.shares_memory(uploaded, getattr(data, name)), name


def test_the_read_hands_pygfx_its_bounds():
    store = _store()
    v, ctx = _placed(store)
    request, data = _read_3d(store, v, ctx)
    v._upload_level(FINE_LEVEL, GFXMeshVisual.request_key(request), data)
    mesh = v._levels[FINE_LEVEL].mesh_3d
    np.testing.assert_allclose(mesh.get_geometry_bounding_box(), data.bounds)
    np.testing.assert_allclose(data.bounds, [[0, 0, 0], [1, 1, 1]])


def test_a_3d_result_goes_to_the_3d_child():
    v = _visual(_store())
    v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(colors=None))
    fine = v._levels[FINE_LEVEL]
    assert fine.holds == "3d"
    assert fine.mesh_3d.geometry.indices.data.shape == (1, 3)


def test_a_2d_result_goes_to_the_fill():
    v = _visual(_store())
    v._upload_level(FINE_LEVEL, KEY_2D, _mesh_data(colors=None))
    fine = v._levels[FINE_LEVEL]
    assert fine.holds == "2d"
    assert fine.fill.geometry.indices.data.shape == (1, 3)


def test_an_empty_result_draws_nothing():
    v = _visual(_store())
    empty = MeshData(
        request_id=uuid4(),
        positions=np.zeros((3, 3), dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int32),
        normals=None,
        colors=None,
        is_empty=True,
    )
    v._upload_level(FINE_LEVEL, KEY_2D, empty)
    fine = v._levels[FINE_LEVEL]
    fine.show(True)
    assert fine.is_empty
    assert fine.group_2d.visible is False
    assert fine.mesh_3d.visible is False


def test_a_release_goes_back_to_the_placeholder():
    v = _visual(_store())
    v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(colors=None))
    v._release_level(FINE_LEVEL)
    fine = v._levels[FINE_LEVEL]
    assert fine.is_empty and fine.holds is None
    assert fine.original_face_indices is None
    assert fine.mesh_3d.geometry.positions.data.shape == (3, 3)


def test_upload_keeps_the_declared_color_mode():
    """The store's colours are uploaded; the declared mode is untouched (D20)."""
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(color_mode="vertex"))
    assert v._levels[FINE_LEVEL].mesh_3d.geometry.colors is not None
    assert v._color_mode == "vertex"
    assert v._material_3d.color_mode == "vertex"
    assert v._material_2d.color_mode == "vertex"


def test_declared_uniform_survives_store_colors():
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="uniform"))
    v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(color_mode="vertex"))
    assert v._material_3d.color_mode == "uniform"
    assert v._material_2d.color_mode == "uniform"


def test_declared_colored_without_colors_raises():
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    with pytest.raises(ValueError, match="carries no colors"):
        v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(colors=None))


def test_declared_layout_contradicting_the_store_raises():
    """Declaring "vertex" against per-face colours is refused, not corrected."""
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    with pytest.raises(ValueError, match="colors_layout is 'face'"):
        v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(color_mode="face"))


def test_empty_slice_does_not_raise_on_declared_colors():
    """An empty slice carries no colours by construction; not a bug."""
    v = _visual(_store(), appearance=MeshFlatAppearance(color_mode="vertex"))
    data = MeshData(
        request_id=uuid4(),
        positions=np.zeros((3, 3), dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int32),
        normals=None,
        colors=None,
        color_mode="vertex",
        is_empty=True,
    )
    v._upload_level(FINE_LEVEL, KEY_3D, data)
    assert v._levels[FINE_LEVEL].is_empty


# ── Picking ───────────────────────────────────────────────────────────────────


def test_face_index_for_pick_maps_through_original_indices():
    v = _visual(_store())
    assert v.face_index_for_pick(2) == 2  # identity pass-through
    v._levels[FINE_LEVEL].original_face_indices = np.array([5, 9, 11], dtype=np.int64)
    assert v.face_index_for_pick(1) == 9
    assert v.face_index_for_pick(999) == 999


def test_decode_pick_reads_the_child_that_was_hit():
    v = _visual(_store())
    fine = v._levels[FINE_LEVEL]
    fine.original_face_indices = np.array([5, 9, 11], dtype=np.int64)
    assert v.decode_pick(fine.mesh_3d, {"face_index": 2}) == MeshPickInfo(
        face_index=11, part="face", level=0
    )
    assert v.decode_pick(fine.mesh_3d, {}) is None
    assert v.decode_pick(gfx.Group(), {"face_index": 0}) is None


def test_decode_pick_tells_the_parts_of_a_section_apart():
    """Plan 5.10: an in-plane face, a cap triangle, and the outline."""
    v = _visual(_store())
    fine = v._levels[FINE_LEVEL]
    fine.fill_face_ids = np.array([-1, -1, 7], dtype=np.int64)
    fine.outline_face_ids = np.array([3, 4], dtype=np.int64)

    assert v.decode_pick(fine.fill, {"face_index": 2}) == MeshPickInfo(
        face_index=7, part="face", level=0
    )
    assert v.decode_pick(fine.fill, {"face_index": 0}) == MeshPickInfo(
        face_index=None, part="fill", level=0
    )
    # Two vertices per segment: vertices 2 and 3 are segment 1.
    for vertex in (2, 3):
        assert v.decode_pick(fine.outline, {"vertex_index": vertex}) == MeshPickInfo(
            face_index=4, part="outline", level=0
        )
    assert v.decode_pick(fine.outline, {}) is None


def test_whole_faces_drawn_flat_pick_as_faces():
    """A 2D view with nothing to cut: the fill's rows are the kept faces."""
    v = _visual(_store())
    fine = v._levels[FINE_LEVEL]
    fine.fill_face_ids = np.array([5, 9], dtype=np.int64)
    assert v.decode_pick(fine.fill, {"face_index": 1}).face_index == 9
    fine.fill_face_ids = None  # every face, in order
    assert v.decode_pick(fine.fill, {"face_index": 1}).face_index == 1


# ── Bounds (M5) ───────────────────────────────────────────────────────────────


def test_the_box_is_the_store_extent_before_any_data():
    store = MeshMemoryStore(
        positions=np.array([[1, 2, 3], [5, 2, 3], [1, 7, 3], [1, 2, 9]], np.float32),
        indices=np.array([[0, 1, 2], [0, 1, 3]], np.int32),
    )
    v, _ = _placed(store)
    corners = np.asarray(v._aabb_line.geometry.positions.data)
    # The node's frame is (x, y, z); the data is (z, y, x).
    np.testing.assert_allclose(corners.min(axis=0), [3, 2, 1])
    np.testing.assert_allclose(corners.max(axis=0), [9, 7, 5])
    assert v._aabb_line.visible is False  # not enabled


def test_the_placeholder_sits_at_the_extent_corner():
    store = MeshMemoryStore(
        positions=np.array([[1, 2, 3], [5, 2, 3], [1, 7, 3], [1, 2, 9]], np.float32),
        indices=np.array([[0, 1, 2], [0, 1, 3]], np.int32),
    )
    v, _ = _placed(store)
    placeholder = v._levels[FINE_LEVEL].mesh_3d.geometry.positions.data
    np.testing.assert_allclose(placeholder, np.tile([3, 2, 1], (3, 1)))
    box = v.node_3d.get_bounding_box()
    np.testing.assert_allclose(box, [[3, 2, 1], [9, 7, 5]])


def test_a_new_extent_resizes_the_box():
    store = _store()
    v, _ = _placed(store)
    v.on_aabb_changed(_aabb_event("enabled", True))
    v.set_axis_extents(((0.0, 4.0), (0.0, 5.0), (0.0, 6.0)))
    corners = np.asarray(v._aabb_line.geometry.positions.data)
    np.testing.assert_allclose(corners.max(axis=0), [6, 5, 4])
    assert v._aabb_line.visible is True


def test_an_empty_store_has_no_box():
    store = _store()
    v, _ = _placed(store)
    v.set_axis_extents(None)
    v.on_aabb_changed(_aabb_event("enabled", True))
    assert v._aabb_line.visible is False


# ── Event handlers ────────────────────────────────────────────────────────────


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
    fine = v._levels[FINE_LEVEL]
    assert fine.mesh_3d.render_order == 5
    assert fine.fill.render_order == 5


def test_on_appearance_changed_transparency_non_blend_and_blend():
    v = _visual(_store(), appearance=MeshFlatAppearance(opacity=0.8))
    v.on_appearance_changed(_appearance_event("transparency_mode", "add"))
    assert v._material_3d.alpha_mode == "add"
    assert v._material_2d.alpha_mode == "add"
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


def test_visibility_is_on_the_nodes_not_the_level_children():
    v = _visual(_store())
    v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(colors=None))
    fine = v._levels[FINE_LEVEL]
    fine.show(True)
    v.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=uuid4(), visible=False
        )
    )
    assert v.node_3d.visible is False
    assert v.node_2d.visible is False
    assert fine.mesh_3d.visible is True


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
    new_tf = identity(3)
    v.on_transform_changed(
        TransformChangedEvent(
            source_id=uuid4(), scene_id=uuid4(), visual_id=uuid4(), transform=new_tf
        )
    )
    assert v._transform is new_tf


def test_on_aabb_changed_reaches_both_lines():
    v = _visual(_store())
    v.on_aabb_changed(_aabb_event("enabled", True))
    v.on_aabb_changed(_aabb_event("color", (0.0, 1.0, 0.0, 1.0)))
    v.on_aabb_changed(_aabb_event("line_width", 4.0))
    assert v._aabb_enabled is True
    for line in (v._aabb_line_2d, v._aabb_line_3d):
        np.testing.assert_allclose(line.material.color, (0.0, 1.0, 0.0, 1.0))
        assert line.material.thickness == pytest.approx(4.0)


def test_prepare_draw_reports_a_change_once():
    v = _visual(_store())
    canvas = uuid4()
    assert v.prepare_draw(canvas, False, False) is False  # nothing to draw
    v._upload_level(FINE_LEVEL, KEY_3D, _mesh_data(colors=None))
    v._residency.planned = {FINE_LEVEL: 1}
    v._residency.held = {FINE_LEVEL: (1, 1)}
    assert v.prepare_draw(canvas, False, False) is True
    assert v._levels[FINE_LEVEL].mesh_3d.visible is True
    assert v.prepare_draw(canvas, False, False) is False
    # Another canvas has its own record.
    assert v.prepare_draw(uuid4(), False, False) is True


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
