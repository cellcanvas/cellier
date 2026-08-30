# tests/v2/render/test_gfx_lines_memory_visual.py
"""Tests for GFXLinesMemoryVisual."""

import asyncio
from uuid import uuid4

import numpy as np
import pygfx as gfx
import pytest

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.lines._lines_requests import LinesData, LinesSliceRequest
from cellier.events._events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    PickWriteChangedEvent,
    TransformChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
from cellier.transform import AffineTransform
from cellier.visuals import LinesMemoryAppearance, LinesVisual


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
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # segment 0
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 3.0],  # segment 1
        ],
        dtype=np.float32,
    )
    return LinesMemoryStore(positions=positions)


def _visual(store, appearance=None):
    if appearance is None:
        appearance = LinesMemoryAppearance()
    model = LinesVisual(name="test", data_store_id=str(store.id), appearance=appearance)
    return GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )


def _make_batch(store, displayed=(0, 1, 2), sliced=None):
    if sliced is None:
        sliced = {}
    sid = uuid4()
    req = LinesSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=displayed,
        slice_indices=sliced,
    )
    data = asyncio.run(store.get_data(req))
    return [(req, data)]


# ── Construction ──────────────────────────────────────────────────────────────


def test_single_node_for_both_modes():
    v = _visual(_store())
    assert v.node_2d is v.node
    assert v.node_3d is v.node


def test_pick_write_enabled_by_default():
    v = _visual(_store())
    assert v._material.pick_write is True


def test_pick_write_follows_model_flag():
    store = _store()
    model = LinesVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=LinesMemoryAppearance(),
        pick_write=False,
    )
    v = GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    assert v._material.pick_write is False


def test_initial_material_is_empty():
    v = _visual(_store())
    assert v.node.material is v._empty_material


def test_n_levels_is_one():
    v = _visual(_store())
    assert v.n_levels == 1


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
    v = _visual(_store())
    v.get_node_for_dims((1, 2))
    original_matrix = v.node.local.matrix.copy()
    v.get_node_for_dims((1, 2))
    np.testing.assert_array_equal(v.node.local.matrix, original_matrix)


# ── on_data_ready ─────────────────────────────────────────────────────────────


def test_on_data_ready_applies_real_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready(_make_batch(s, displayed=(0, 1, 2)))
    assert v.node.material is v._material


def test_empty_slab_applies_empty_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 1000}))
    assert v.node.material is v._empty_material


# ── Coordinate reorder ────────────────────────────────────────────────────────


def test_3d_positions_reordered_zyx_to_xyz():
    """After on_data_ready, GPU positions must be in (x, y, z) order."""
    # Segment: (z=1, y=2, x=3) → (z=4, y=5, x=6).
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    store = LinesMemoryStore(positions=positions)
    model = LinesVisual(name="t", data_store_id=str(store.id))
    v = GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready(_make_batch(store, displayed=(0, 1, 2)))
    gpu_pos = v.node.geometry.positions.data
    # After reversal: first vertex should be (x=3, y=2, z=1).
    np.testing.assert_allclose(gpu_pos[0], [3.0, 2.0, 1.0])
    # Second vertex should be (x=6, y=5, z=4).
    np.testing.assert_allclose(gpu_pos[1], [6.0, 5.0, 4.0])


def test_2d_positions_padded_with_zero():
    """After on_data_ready_2d, GPU positions must be (row, col, 0)."""
    # Both vertices at z=0 so they pass the default thickness=0.5 filter.
    positions = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=np.float32)
    store = LinesMemoryStore(positions=positions)
    model = LinesVisual(name="t", data_store_id=str(store.id))
    v = GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"2d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready_2d(_make_batch(store, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        gpu_pos = v.node.geometry.positions.data
        np.testing.assert_allclose(gpu_pos[:, 2], 0.0)


# ── Construction errors / trivia ───────────────────────────────────────────────


def test_invalid_render_modes_raises():
    store = _store()
    model = LinesVisual(name="t", data_store_id=str(store.id))
    with pytest.raises(ValueError, match="render_modes"):
        GFXLinesMemoryVisual(
            visual_model=model,
            render_modes={"4d"},
            transform=AffineTransform.identity(ndim=store.ndim),
        )


def test_empty_batch_is_noop():
    v = _visual(_store())
    v.on_data_ready([])
    v.on_data_ready_2d([])
    assert v.node.material is v._empty_material


def test_edge_index_for_vertex_maps_through_original_indices():
    v = _visual(_store())
    # No filtering -> vertex//2 passes through.
    assert v.edge_index_for_vertex(4) == 2
    v._original_edge_indices = np.array([5, 9, 11], dtype=np.int64)
    assert v.edge_index_for_vertex(2) == 9  # rendered edge 1 -> original 9
    assert v.edge_index_for_vertex(999) == 499  # out-of-range -> vertex//2


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


def _colored_lines_data():
    return LinesData(
        request_id=uuid4(),
        positions=np.array(
            [[0, 1, 2], [3, 4, 5], [0, 1, 6], [3, 4, 7]], dtype=np.float32
        ),
        colors=np.array(
            [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]],
            dtype=np.float32,
        ),
        color_mode="vertex",
        is_empty=False,
    )


def test_commit_uploads_colors_without_touching_color_mode():
    """A store's colours are uploaded; the declared color_mode is untouched.

    Inverted at the D20 migration.  This test previously asserted the
    opposite -- that incoming vertex-coloured data overrode a declared
    ``"uniform"`` on the live material.
    """
    v = _visual(_store(), appearance=LinesMemoryAppearance(color_mode="vertex"))
    v.on_data_ready([(None, _colored_lines_data())])
    assert v.node.geometry.colors is not None
    assert v._material.color_mode == "vertex"


def test_declared_uniform_survives_store_colors():
    """Store carries colours, appearance says uniform -> stays uniform (D20)."""
    v = _visual(_store(), appearance=LinesMemoryAppearance(color_mode="uniform"))
    v.on_data_ready([(None, _colored_lines_data())])
    assert v._material.color_mode == "uniform"
    assert v._color_mode == "uniform"


def test_declared_vertex_without_colors_raises():
    """The mismatch is a misconfiguration, not a silent fallback (D20)."""
    v = _visual(_store(), appearance=LinesMemoryAppearance(color_mode="vertex"))
    data = LinesData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        colors=None,
        color_mode="uniform",
        is_empty=False,
    )
    with pytest.raises(ValueError, match="color_mode='vertex'"):
        v.on_data_ready([(None, data)])


def test_thickness_space_appearance_change_reaches_the_material():
    """thickness_space is applied live, not silently dropped."""
    v = _visual(_store())
    assert v._material.thickness_space == "screen"
    v.on_appearance_changed(_appearance_event("thickness_space", "world"))
    assert v._material.thickness_space == "world"


def test_lines_alpha_buffer_is_ones():
    """The migrated visual uploads an all-ones alpha buffer."""
    v = _visual(_store())
    data = LinesData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        colors=None,
        is_empty=False,
    )
    v.on_data_ready([(None, data)])
    alphas = v.node.geometry.alphas.data
    assert alphas.shape == (2,)
    assert np.all(alphas == 1.0)


# ── Event handlers ─────────────────────────────────────────────────────────────


def test_on_appearance_changed_updates_material_fields():
    v = _visual(_store())
    v.on_appearance_changed(_appearance_event("color", (1.0, 0.0, 0.0, 1.0)))
    np.testing.assert_allclose(v._material.color, (1.0, 0.0, 0.0, 1.0))

    v.on_appearance_changed(_appearance_event("color_mode", "vertex"))
    assert v._material.color_mode == "vertex"
    assert v._color_mode == "vertex"

    v.on_appearance_changed(_appearance_event("opacity", 0.3))
    assert v._material.opacity == pytest.approx(0.3)

    v.on_appearance_changed(_appearance_event("thickness", 6.0))
    assert v._material.thickness == pytest.approx(6.0)

    v.on_appearance_changed(_appearance_event("depth_test", False))
    assert v._material.depth_test is False

    v.on_appearance_changed(_appearance_event("depth_write", False))
    assert v._material.depth_write is False

    v.on_appearance_changed(_appearance_event("depth_compare", "<="))
    assert v._material.depth_compare == "<="

    v.on_appearance_changed(_appearance_event("transparency_mode", "add"))
    assert v._material.alpha_mode == "add"

    v.on_appearance_changed(_appearance_event("render_order", 5))
    assert v.node.render_order == 5


def test_on_appearance_changed_unknown_field_is_noop():
    v = _visual(_store())
    before = v._material.opacity
    v.on_appearance_changed(_appearance_event("thickness_space", "world"))
    assert v._material.opacity == before


def test_on_visibility_changed_toggles_node():
    v = _visual(_store())
    v.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=uuid4(), visible=False
        )
    )
    assert v.node.visible is False


def test_on_pick_write_changed():
    v = _visual(_store())
    v.on_pick_write_changed(
        PickWriteChangedEvent(source_id=uuid4(), visual_id=uuid4(), pick_write=False)
    )
    assert v._material.pick_write is False


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


def test_tick_and_cancel_are_noops():
    v = _visual(_store())
    v.tick()
    v.cancel_pending()
    v.cancel_pending_2d()


# ── Rendered output (controller-driven) ────────────────────────────────────────


async def test_render_2d_draws_pixels(controller, render_scene, reslice):
    positions = np.array(
        [[0, 2, 2], [0, 20, 28], [0, 4, 26], [0, 22, 4]], dtype=np.float32
    )
    store = LinesMemoryStore(positions=positions)
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_lines(
        data=store,
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(thickness=5.0, color=(1.0, 0.0, 0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_3d_draws_pixels(controller, render_scene, reslice):
    positions = np.array(
        [[2, 2, 2], [10, 20, 28], [4, 4, 26], [8, 22, 4]], dtype=np.float32
    )
    store = LinesMemoryStore(positions=positions)
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_lines(
        data=store,
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(thickness=5.0, color=(0.0, 1.0, 0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0
