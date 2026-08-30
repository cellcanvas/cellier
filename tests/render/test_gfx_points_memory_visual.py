# tests/v2/render/test_gfx_points_memory_visual.py
"""Tests for GFXPointsMemoryVisual."""

import asyncio
from uuid import uuid4

import numpy as np
import pygfx as gfx
import pytest

from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.data.points._points_requests import PointsSliceRequest
from cellier.events._events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    PickWriteChangedEvent,
    TransformChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.render.visuals._points_memory import GFXPointsMemoryVisual
from cellier.transform import AffineTransform
from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual


def _appearance_event(field, value):
    return AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=uuid4(),
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


def _store():
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    return PointsMemoryStore(positions=positions)


def _visual(store, appearance=None):
    if appearance is None:
        appearance = PointsMarkerAppearance()
    model = PointsVisual(
        name="test", data_store_id=str(store.id), appearance=appearance
    )
    return GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )


def _make_batch(store, displayed=(0, 1, 2), sliced=None):
    if sliced is None:
        sliced = {}
    sid = uuid4()
    req = PointsSliceRequest(
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
    model = PointsVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=PointsMarkerAppearance(),
        pick_write=False,
    )
    v = GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    assert v._material.pick_write is False


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
    v.get_node_for_dims((1, 2))  # same axes — must be a no-op
    np.testing.assert_array_equal(v.node.local.matrix, original_matrix)


# ── on_data_ready ─────────────────────────────────────────────────────────────


def test_on_data_ready_applies_real_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready(_make_batch(s, displayed=(0, 1, 2)))
    assert v.node.material is v._material


def test_on_data_ready_2d_applies_real_material():
    s = _store()
    v = _visual(s)
    # All points are at various z positions; use thick slab to guarantee some survive.
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        assert v.node.material is v._material


def test_empty_slab_applies_empty_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 1000}))
    assert v.node.material is v._empty_material


# ── Coordinate reorder ────────────────────────────────────────────────────────


def test_3d_positions_reordered_zyx_to_xyz():
    """After on_data_ready, GPU positions must be in (x, y, z) order."""
    # Single point at known (z=1, y=2, x=3).
    positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)
    model = PointsVisual(name="t", data_store_id=str(store.id))
    v = GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready(_make_batch(store, displayed=(0, 1, 2)))
    gpu_pos = v.node.geometry.positions.data
    # After reversal: column 0 should be x=3, column 1 y=2, column 2 z=1.
    np.testing.assert_allclose(gpu_pos[0], [3.0, 2.0, 1.0])


def test_2d_positions_padded_with_zero():
    """After on_data_ready_2d, GPU positions must be (row, col, 0)."""
    # All points at z=0 so they pass the default thickness=0.5 filter.
    positions = np.array([[0.0, 2.0, 3.0], [0.0, 4.0, 5.0]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)
    model = PointsVisual(name="t", data_store_id=str(store.id))
    v = GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"2d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready_2d(_make_batch(store, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        gpu_pos = v.node.geometry.positions.data
        # Third column must be zero (the padding column).
        np.testing.assert_allclose(gpu_pos[:, 2], 0.0)


# ── Construction errors / trivia ───────────────────────────────────────────────


def test_invalid_render_modes_raises():
    store = _store()
    model = PointsVisual(name="t", data_store_id=str(store.id))
    with pytest.raises(ValueError, match="render_modes"):
        GFXPointsMemoryVisual(
            visual_model=model,
            render_modes={"4d"},
            transform=AffineTransform.identity(ndim=store.ndim),
        )
    with pytest.raises(ValueError, match="render_modes"):
        GFXPointsMemoryVisual(
            visual_model=model,
            render_modes=set(),
            transform=AffineTransform.identity(ndim=store.ndim),
        )


def test_n_levels_is_one():
    assert _visual(_store()).n_levels == 1


def test_empty_batch_is_noop():
    v = _visual(_store())
    v.on_data_ready([])
    v.on_data_ready_2d([])
    assert v.node.material is v._empty_material


def test_point_index_for_vertex_maps_through_original_indices():
    v = _visual(_store())
    v.on_data_ready(_make_batch(_store(), displayed=(0, 1, 2)))
    # With no filtering the map is identity (None) -> pass-through.
    assert v.point_index_for_vertex(2) == 2
    # A filtered view stores the surviving-index map; it remaps.
    v._original_indices = np.array([5, 9, 11], dtype=np.int64)
    assert v.point_index_for_vertex(1) == 9
    # Out-of-range passes through unchanged.
    assert v.point_index_for_vertex(99) == 99


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

    v.on_appearance_changed(_appearance_event("size", 12.0))
    assert v._material.size == pytest.approx(12.0)

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
    v.on_appearance_changed(_appearance_event("size_space", "world"))
    assert v._material.opacity == before


def test_on_visibility_changed_toggles_node():
    v = _visual(_store())
    v.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=uuid4(), visible=False
        )
    )
    assert v.node.visible is False
    v.on_visibility_changed(
        VisualVisibilityChangedEvent(source_id=uuid4(), visual_id=uuid4(), visible=True)
    )
    assert v.node.visible is True


def test_on_pick_write_changed():
    v = _visual(_store())
    v.on_pick_write_changed(
        PickWriteChangedEvent(source_id=uuid4(), visual_id=uuid4(), pick_write=False)
    )
    assert v._material.pick_write is False


def test_on_transform_changed_updates_matrix():
    v = _visual(_store())
    v.get_node_for_dims((0, 1, 2))  # establishes _last_displayed_axes
    new_tf = AffineTransform.identity(ndim=3)
    v.on_transform_changed(
        TransformChangedEvent(
            source_id=uuid4(), scene_id=uuid4(), visual_id=uuid4(), transform=new_tf
        )
    )
    assert v._transform is new_tf


def _aabb_event(field, value):
    return AABBChangedEvent(
        source_id=uuid4(), visual_id=uuid4(), field_name=field, new_value=value
    )


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


# ── GFXVisual protocol stubs + slice-request planning ──────────────────────────


def _dims_state(displayed=(0, 1, 2), sliced=None):
    from cellier._state import AxisAlignedSelectionState, DimsState

    return DimsState(
        axis_labels=tuple(str(i) for i in range(3)),
        selection=AxisAlignedSelectionState(
            displayed_axes=displayed, slice_indices=sliced or {}
        ),
    )


def test_protocol_node_accessors():
    v = _visual(_store())
    assert v.has_node("3d") is True
    assert v.get_node("3d") is v.node
    assert v.build_node("3d", None, (0, 1, 2), None, None) is v.node
    assert v.rebuild_node_geometry("3d", (0, 1, 2), None, None) is v.node
    v.on_stacked_axes_changed((0,))  # no-op


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


def test_commit_uploads_colors_and_sizes_without_touching_color_mode():
    """A store's colours are uploaded; the declared color_mode is untouched.

    Inverted at the D20 migration.  This test previously asserted the
    opposite -- that incoming vertex-coloured data overrode a declared
    ``"uniform"`` on the live material.  ``color_mode`` is now a caller
    declaration of where RGB comes from, never inferred from the data.
    """
    from cellier.data.points._points_requests import PointsData

    v = _visual(_store(), appearance=PointsMarkerAppearance(color_mode="vertex"))
    data = PointsData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        colors=np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32),
        sizes=np.array([4.0, 8.0], dtype=np.float32),
        color_mode="vertex",
        is_empty=False,
    )
    v.on_data_ready([(None, data)])
    assert "colors" in v.node.geometry.__dict__ or v.node.geometry.colors is not None
    assert v.node.geometry.sizes is not None
    # color_mode is the declaration, so it is left exactly as configured.
    assert v._material.color_mode == "vertex"
    assert v._color_mode == "vertex"


def test_declared_uniform_survives_store_colors():
    """Store carries colours, appearance says uniform -> stays uniform (D20)."""
    from cellier.data.points._points_requests import PointsData

    v = _visual(_store(), appearance=PointsMarkerAppearance(color_mode="uniform"))
    data = PointsData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        colors=np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32),
        sizes=None,
        color_mode="vertex",
        is_empty=False,
    )
    v.on_data_ready([(None, data)])
    assert v._material.color_mode == "uniform"
    assert v._color_mode == "uniform"


def test_declared_vertex_without_colors_raises():
    """The mismatch is a misconfiguration, not a silent fallback (D20)."""
    from cellier.data.points._points_requests import PointsData

    v = _visual(_store(), appearance=PointsMarkerAppearance(color_mode="vertex"))
    data = PointsData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2]], dtype=np.float32),
        colors=None,
        sizes=None,
        color_mode="uniform",
        is_empty=False,
    )
    with pytest.raises(ValueError, match="color_mode='vertex'"):
        v.on_data_ready([(None, data)])


def test_declared_vertex_size_mode_reaches_the_material():
    """size_mode is a declaration, applied at construction (not inferred).

    Before this was fixed the material's ``size_mode`` was never written at
    all, so a store's ``sizes`` buffer was uploaded and then silently
    ignored at render time.
    """
    v = _visual(_store(), appearance=PointsMarkerAppearance(size_mode="vertex"))
    assert v._material.size_mode == "vertex"
    assert v._size_mode == "vertex"


def test_declared_uniform_survives_store_sizes():
    """Store carries sizes, appearance says uniform -> stays uniform."""
    from cellier.data.points._points_requests import PointsData

    v = _visual(_store(), appearance=PointsMarkerAppearance(size_mode="uniform"))
    data = PointsData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        colors=None,
        sizes=np.array([4.0, 20.0], dtype=np.float32),
        size_mode="vertex",
        is_empty=False,
    )
    v.on_data_ready([(None, data)])
    assert v._material.size_mode == "uniform"
    assert v._size_mode == "uniform"


def test_declared_vertex_without_sizes_raises():
    """The mismatch is a misconfiguration, not a silent fallback."""
    from cellier.data.points._points_requests import PointsData

    v = _visual(_store(), appearance=PointsMarkerAppearance(size_mode="vertex"))
    data = PointsData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2]], dtype=np.float32),
        colors=None,
        sizes=None,
        is_empty=False,
    )
    with pytest.raises(ValueError, match="size_mode='vertex'"):
        v.on_data_ready([(None, data)])


def test_empty_slice_does_not_raise_on_declared_vertex_size():
    """An empty slice carries no sizes by construction; that is not a bug."""
    from cellier.data.points._points_requests import PointsData

    v = _visual(_store(), appearance=PointsMarkerAppearance(size_mode="vertex"))
    data = PointsData(
        request_id=uuid4(),
        positions=np.zeros((1, 3), dtype=np.float32),
        colors=None,
        sizes=None,
        is_empty=True,
    )
    v.on_data_ready([(None, data)])
    assert v._is_empty


def test_size_mode_appearance_change_reaches_the_material():
    v = _visual(_store())
    assert v._material.size_mode == "uniform"
    v.on_appearance_changed(_appearance_event("size_mode", "vertex"))
    assert v._material.size_mode == "vertex"
    assert v._size_mode == "vertex"


def test_size_space_appearance_change_reaches_the_material():
    """size_space is applied live, not silently dropped.

    It was previously unhandled, with a docstring claiming the pygfx field
    was constructor-only.  That is false for the pinned pygfx, so changing
    ``appearance.size_space`` did nothing and the code said that was fine.
    """
    v = _visual(_store())
    assert v._material.size_space == "screen"
    v.on_appearance_changed(_appearance_event("size_space", "world"))
    assert v._material.size_space == "world"


def test_points_alpha_buffer_is_ones():
    """The migrated visual uploads an all-ones alpha buffer."""
    from cellier.data.points._points_requests import PointsData

    v = _visual(_store())
    data = PointsData(
        request_id=uuid4(),
        positions=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        colors=None,
        sizes=None,
        is_empty=False,
    )
    v.on_data_ready([(None, data)])
    alphas = v.node.geometry.alphas.data
    assert alphas.shape == (2,)
    assert np.all(alphas == 1.0)


# ── Rendered output (controller-driven) ────────────────────────────────────────


async def test_render_2d_and_3d_draw_pixels(controller, render_scene, reslice):
    positions = np.array([[0, 4, 4], [0, 8, 12], [0, 12, 20]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)

    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(size=20.0, color=(1.0, 0.0, 0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_vertex_size_mode_renders_different_marker_sizes(
    controller, render_scene, reslice
):
    """Per-point sizes must actually change the drawn marker size.

    The regression this whole change exists for.  Before it, the store's
    ``sizes`` buffer was uploaded but ``material.size_mode`` was never
    written, so every marker rendered at the uniform size and the buffer
    was inert -- invisible to any test that only checked the geometry.
    Counting lit pixels per marker is what catches that; asserting the
    buffer was uploaded does not.
    """
    # Two points far apart on the displayed plane so their markers cannot
    # overlap, with a 5x size difference between them.
    positions = np.array([[0, 8, 8], [0, 8, 40]], dtype=np.float32)
    sizes = np.array([4.0, 20.0], dtype=np.float32)
    store = PointsMemoryStore(positions=positions, sizes=sizes)

    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(
            size=4.0, color=(1.0, 0.0, 0.0, 1.0), size_mode="vertex"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id, size=(256, 256))

    # Split the frame down the middle: one marker per half.
    lit = frame[..., 3] > 0
    small = int(lit[:, : lit.shape[1] // 2].sum())
    large = int(lit[:, lit.shape[1] // 2 :].sum())

    assert small > 0 and large > 0, f"both markers should draw: {small}, {large}"
    assert large > small * 2, (
        f"the 20.0-size marker should cover far more pixels than the 4.0-size "
        f"one, got {large} vs {small} -- size_mode is not reaching the material"
    )


async def test_uniform_size_mode_renders_equal_marker_sizes(
    controller, render_scene, reslice
):
    """The control for the test above: declared uniform ignores the store.

    Same store, same sizes buffer, only the declaration differs -- so the
    two markers must come out the same size.  Without this, the test above
    would still pass if markers happened to differ for some other reason.
    """
    positions = np.array([[0, 8, 8], [0, 8, 40]], dtype=np.float32)
    sizes = np.array([4.0, 20.0], dtype=np.float32)
    store = PointsMemoryStore(positions=positions, sizes=sizes)

    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(
            size=10.0, color=(1.0, 0.0, 0.0, 1.0), size_mode="uniform"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id, size=(256, 256))

    lit = frame[..., 3] > 0
    left = int(lit[:, : lit.shape[1] // 2].sum())
    right = int(lit[:, lit.shape[1] // 2 :].sum())

    assert left > 0 and right > 0
    assert abs(left - right) <= max(2, left * 0.1), (
        f"declared uniform must ignore the store's sizes, got {left} vs {right}"
    )


async def test_size_space_change_alters_the_rendered_marker(
    controller, render_scene, reslice
):
    """Flipping size_space on a live visual must change the drawn size.

    Asserting the material property alone is not enough: pygfx could accept
    the assignment without rebuilding the pipeline, in which case the
    picture would never update.  This renders twice on one live visual and
    compares lit pixels.
    """
    positions = np.array([[0, 16, 16], [0, 16, 16]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)

    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(
            size=10.0, color=(1.0, 0.0, 0.0, 1.0), size_space="screen"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    def _red_px() -> int:
        frame = render_scene(controller, scene.id, size=(256, 256))
        return int((frame[..., 0] > 100).sum())

    screen_px = _red_px()
    visual.appearance.size_space = "world"
    world_px = _red_px()

    assert screen_px > 0, "the marker should draw in screen space"
    assert world_px != screen_px, (
        f"flipping size_space must change the rendered marker, got "
        f"{screen_px} px both ways -- the change is not reaching the material"
    )


async def test_render_3d_draws_pixels(controller, render_scene, reslice):
    positions = np.array([[2, 4, 4], [6, 8, 12], [10, 12, 20]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)

    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(size=20.0, color=(0.0, 1.0, 0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0
