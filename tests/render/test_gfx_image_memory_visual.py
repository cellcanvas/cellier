"""Tests for GFXImageMemoryVisual."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.image._image_requests import ChunkRequest
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import ImageVisual, InMemoryImageAppearance
from tests._v2 import Context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(shape=(10, 20, 30)) -> ImageMemoryStore:
    return ImageMemoryStore(data=np.zeros(shape, dtype=np.float32))


def _make_appearance() -> InMemoryImageAppearance:
    return InMemoryImageAppearance()


def _make_visual_model(store: ImageMemoryStore) -> ImageVisual:
    return ImageVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=_make_appearance(),
    )


def _make_dims_state_2d() -> DimsState:
    """3D store, 2D scene: display (Y, X), slice Z at 5."""
    return DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
        ),
    )


def _make_dims_state_3d() -> DimsState:
    """3D store, 3D scene: display all axes."""
    return DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(0, 1, 2),
        ),
    )


# ---------------------------------------------------------------------------
# Tests: planning
# ---------------------------------------------------------------------------


@patch("cellier.render.visuals._image_memory.gfx")
def test_build_slice_request_2d_returns_one_request(mock_gfx):
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    ctx = Context(3, displayed_axes=(1, 2), slice_indices={0: 5.0})
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"2d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = _make_dims_state_2d()
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
        selection=ctx.selection,
    )

    assert len(requests) == 1
    req = requests[0]
    assert isinstance(req, ChunkRequest)
    assert req.scale_index == 0
    # Axis 0 (z) is sliced at index 5
    assert req.axis_selections[0] == 5
    # Axes 1 and 2 (y, x) are displayed — full extent
    assert req.axis_selections[1] == (0, 20)
    assert req.axis_selections[2] == (0, 30)
    # Both UUIDs are freshly generated
    assert req.chunk_request_id is not None
    assert req.slice_request_id is not None


@patch("cellier.render.visuals._image_memory.gfx")
def test_build_slice_request_3d_returns_one_request(mock_gfx):
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    ctx = Context(3, displayed_axes=(0, 1, 2))
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"3d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = _make_dims_state_3d()
    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=600.0,
        dims_state=dims,
        selection=ctx.selection,
    )

    assert len(requests) == 1
    req = requests[0]
    assert req.axis_selections == ((0, 10), (0, 20), (0, 30))


@patch("cellier.render.visuals._image_memory.gfx")
def test_build_slice_request_3d_handles_none_dims_state(mock_gfx):
    """With dims_state=None all axes are treated as displayed."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(5, 6, 7))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=600.0,
        dims_state=None,
    )
    assert requests[0].axis_selections == ((0, 5), (0, 6), (0, 7))


@patch("cellier.render.visuals._image_memory.gfx")
def test_5d_dims_state_2d_scene(mock_gfx):
    """5D store, 2D scene: display (Y, X), slice T=0, C=1, Z=5."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(2, 3, 10, 20, 30))
    model = _make_visual_model(store)
    ctx = Context(5, displayed_axes=(3, 4), slice_indices={0: 0.0, 1: 1.0, 2: 5.0})
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"2d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = DimsState(
        axis_labels=("t", "c", "z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(3, 4),
        ),
    )
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
        selection=ctx.selection,
    )
    req = requests[0]
    assert req.axis_selections == (0, 1, 5, (0, 20), (0, 30))


@patch("cellier.render.visuals._image_memory.gfx")
def test_scaled_transform_halves_slice_index_2d(mock_gfx):
    """With scale=(2,2,2), world z=10 should map to data z=5."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    ctx = Context(3, (2.0, 2.0, 2.0), displayed_axes=(1, 2), slice_indices={0: 10.0})
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"2d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            # world z=10
        ),
    )
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
        selection=ctx.selection,
    )
    req = requests[0]
    # world z=10, scale=2 → data z=5
    assert req.axis_selections[0] == 5
    assert req.axis_selections[1] == (0, 20)
    assert req.axis_selections[2] == (0, 30)


@patch("cellier.render.visuals._image_memory.gfx")
def test_non_spatial_axis_not_transformed_3d(mock_gfx):
    """4D store, 3D scene: non-spatial axis (t) is NOT transformed."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(8, 10, 20, 30))
    model = _make_visual_model(store)
    # The transform is full rank now: ``t`` is scale 1, which is what
    # "outside the 3D transform" used to mean when a 3-D transform was
    # expanded to 4 by prepending identity axes.
    ctx = Context(
        4,
        (1.0, 2.0, 2.0, 2.0),
        displayed_axes=(1, 2, 3),
        slice_indices={0: 6.0},
        labels=("t", "z", "y", "x"),
    )
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"3d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = DimsState(
        axis_labels=("t", "z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2, 3),
            # t=6, non-spatial
        ),
    )
    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=600.0,
        dims_state=dims,
        selection=ctx.selection,
    )
    req = requests[0]
    # t is non-spatial (outside 3D transform) → stays at 6
    assert req.axis_selections[0] == 6
    assert req.axis_selections[1] == (0, 10)
    assert req.axis_selections[2] == (0, 20)
    assert req.axis_selections[3] == (0, 30)


@patch("cellier.render.visuals._image_memory.gfx")
def test_scaled_transform_on_spatial_slice_in_4d(mock_gfx):
    """4D store, 2D scene: spatial z-axis IS transformed by scale."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(8, 10, 20, 30))
    model = _make_visual_model(store)
    ctx = Context(
        4,
        (1.0, 2.0, 2.0, 2.0),
        displayed_axes=(2, 3),
        slice_indices={0: 3.0, 1: 8.0},
        labels=("t", "z", "y", "x"),
    )
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"2d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = DimsState(
        axis_labels=("t", "z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(2, 3),
            # t=3 (non-spatial), z=8 (spatial)
        ),
    )
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
        selection=ctx.selection,
    )
    req = requests[0]
    # t=3 non-spatial → stays at 3
    assert req.axis_selections[0] == 3
    # z=8 spatial, scale=2 → data z=4
    assert req.axis_selections[1] == 4
    assert req.axis_selections[2] == (0, 20)
    assert req.axis_selections[3] == (0, 30)


@patch("cellier.render.visuals._image_memory.gfx")
def test_identity_transform_preserves_slice_index(mock_gfx):
    """Identity transform should not alter slice indices."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    ctx = Context(3, displayed_axes=(1, 2), slice_indices={0: 7.0})
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"2d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
        ),
    )
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
        selection=ctx.selection,
    )
    assert requests[0].axis_selections[0] == 7


@patch("cellier.render.visuals._image_memory.gfx")
def test_slice_index_outside_store_bounds_draws_nothing(mock_gfx):
    """A transformed slice index outside the data plans nothing (design 3.2)."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    # scale=0.5 means world z=5 -> data z=10, which is out of bounds (max 9)
    ctx = Context(3, (0.5, 0.5, 0.5), displayed_axes=(1, 2), slice_indices={0: 5.0})
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"2d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            # world z=5, data z=10 -> past the last plane
        ),
    )
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
        selection=ctx.selection,
    )
    assert requests == []
    assert visual._slice_empty is True


# ---------------------------------------------------------------------------
# Tests: commit (on_data_ready / on_data_ready_2d)
# ---------------------------------------------------------------------------


@patch("cellier.render.visuals._image_memory.gfx")
def test_on_data_ready_2d_transposes_correctly(mock_gfx):
    """Data (H=20, W=30) must be uploaded as (H=20, W=30, 1) — no transpose."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_modes={"2d"})

    data = np.random.rand(20, 30).astype(np.float32)
    req = ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=(5, (0, 20), (0, 30)),
    )

    captured_arrays = []
    mock_gfx.Texture.side_effect = lambda arr, **kw: captured_arrays.append(arr)
    mock_gfx.Geometry.return_value = MagicMock()

    visual.on_data_ready_2d([(req, data)])

    assert len(captured_arrays) == 1
    arr = captured_arrays[0]
    assert arr.shape == (20, 30, 1)
    np.testing.assert_array_equal(arr[:, :, 0], data)


@patch("cellier.render.visuals._image_memory.gfx")
def test_on_data_ready_3d_no_transpose(mock_gfx):
    """Data (D=10, H=20, W=30) is uploaded as-is -- no transpose.

    pygfx maps a numpy texture's last axis to texture-x, so (D, H, W) =
    (z, y, x) already lands as local (x=W, y=H, z=D). Transposing here would
    double-reverse the axes and swap data-x with data-z in the rendered volume.
    """
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    data = np.random.rand(10, 20, 30).astype(np.float32)
    req = ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=((0, 10), (0, 20), (0, 30)),
    )

    captured_arrays = []
    mock_gfx.Texture.side_effect = lambda arr, **kw: captured_arrays.append(arr)
    mock_gfx.Geometry.return_value = MagicMock()

    visual.on_data_ready([(req, data)])

    assert len(captured_arrays) == 1
    arr = captured_arrays[0]
    assert arr.shape == (10, 20, 30)
    np.testing.assert_array_equal(arr, data)


@patch("cellier.render.visuals._image_memory.gfx")
def test_on_data_ready_noop_on_empty_batch(mock_gfx):
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store()
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    # Should not raise
    visual.on_data_ready([])
    visual.on_data_ready_2d([])


# ---------------------------------------------------------------------------
# Tests: transform
# ---------------------------------------------------------------------------


@patch("cellier.render.visuals._image_memory.gfx")
def test_node_matrix_set_lazily_on_slice_request(mock_gfx):
    """Node matrix is set on first build_slice_request, not construction."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store()
    model = _make_visual_model(store)
    ctx = Context(3, (4.0, 2.0, 3.0), displayed_axes=(0, 1, 2))
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"3d"}, transform=ctx.transform
    )
    ctx.place(visual)

    # Before any slice request, _last_displayed_axes is None.
    assert visual._last_displayed_axes is None

    # Trigger a slice request with displayed axes (0, 1, 2).
    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(0, 1, 2),
        ),
    )
    visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=600.0,
        dims_state=dims,
        selection=ctx.selection,
    )
    assert visual._last_displayed_axes == (0, 1, 2)
    # visual -> data -> world -> rendered composes to the same diagonal;
    # _pygfx_matrix reverses cellier order (z, y, x) to pygfx (x, y, z),
    # so scale (4, 2, 3) in data order becomes (3, 2, 4).
    expected = np.diag([3.0, 2.0, 4.0, 1.0]).astype(np.float32)
    np.testing.assert_array_equal(visual.node_3d.local.matrix, expected)


@patch("cellier.render.visuals._image_memory.gfx")
def test_on_transform_changed_updates_node_after_initial_slice(mock_gfx):
    """Transform change updates node matrix if displayed axes are known."""
    from cellier.events._events import TransformChangedEvent
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store()
    model = _make_visual_model(store)
    ctx = Context(3, displayed_axes=(1, 2), slice_indices={0: 5.0})
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"2d"}, transform=ctx.transform
    )
    ctx.place(visual)

    # First, trigger a slice to establish displayed axes.
    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
        ),
    )
    visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
        selection=ctx.selection,
    )

    # Now change the transform, against the same coordinate systems: a v2
    # transform names its endpoints, so a replacement has to keep them.
    new_ctx = Context(
        3,
        (3.0, 5.0, 7.0),
        displayed_axes=(1, 2),
        slice_indices={0: 5.0},
        data=ctx.data,
        world=ctx.world,
    )
    new_t = new_ctx.transform
    event = TransformChangedEvent(
        source_id=uuid4(),
        scene_id=uuid4(),
        visual_id=model.id,
        transform=new_t,
    )
    visual.on_transform_changed(event)

    assert visual._transform is new_t
    # The 2D node retains data axes (1, 2) -> diag(5, 7) in cellier order
    # (y, x), which _pygfx_matrix reverses to pygfx (x, y) -> diag(7, 5).
    expected_2d = np.diag([7.0, 5.0, 1.0, 1.0]).astype(np.float32)
    np.testing.assert_array_equal(visual.node_2d.local.matrix, expected_2d)


# ---------------------------------------------------------------------------
# Tests: 3D render_mode / iso_threshold (real pygfx, no mock)
# ---------------------------------------------------------------------------


def _make_iso_model(store, render_mode="iso", iso_threshold=0.5):
    return ImageVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            render_mode=render_mode,
            iso_threshold=iso_threshold,
        ),
    )


def test_appearance_render_mode_defaults():
    """Default render_mode is mip; iso_threshold defaults to 0.5."""
    from cellier.visuals import InMemoryImageSingleAppearance

    appearance = InMemoryImageSingleAppearance(color_map="viridis")
    assert appearance.render_mode == "mip"
    assert appearance.iso_threshold == 0.5


def test_3d_material_default_is_mip():
    import pygfx as gfx

    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(4, 5, 6))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    assert isinstance(visual._inner_node_3d.material, gfx.VolumeMipMaterial)


def test_3d_material_iso_applies_threshold():
    import pygfx as gfx

    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(4, 5, 6))
    model = _make_iso_model(store, render_mode="iso", iso_threshold=0.3)
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    material = visual._inner_node_3d.material
    assert isinstance(material, gfx.VolumeIsoMaterial)
    assert material.threshold == 0.3


def test_3d_material_minip():
    import pygfx as gfx

    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(4, 5, 6))
    model = _make_iso_model(store, render_mode="minip")
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    assert isinstance(visual._inner_node_3d.material, gfx.VolumeMinipMaterial)


def test_single_render_mode_change_swaps_material():
    """Changing render_mode swaps the 3D material and preserves settings."""
    import pygfx as gfx

    from cellier.events._events import SingleAppearanceChangedEvent
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(4, 5, 6))
    model = _make_visual_model(store)  # defaults to mip
    model.single.opacity = 0.5
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    assert isinstance(visual._inner_node_3d.material, gfx.VolumeMipMaterial)

    model.single.render_mode = "iso"
    visual.on_single_appearance_changed(
        SingleAppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=model.id,
            field_name="render_mode",
            new_value="iso",
        )
    )

    new_material = visual._inner_node_3d.material
    assert isinstance(new_material, gfx.VolumeIsoMaterial)
    # Dynamic settings carried over from the previous material.
    assert new_material.opacity == 0.5
    assert tuple(new_material.clim) == (0.0, 1.0)


def test_single_iso_threshold_change_updates_material():
    from cellier.events._events import SingleAppearanceChangedEvent
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(4, 5, 6))
    model = _make_iso_model(store, render_mode="iso", iso_threshold=0.5)
    visual = GFXImageMemoryVisual(model, store, render_modes={"3d"})

    model.single.iso_threshold = 0.8
    visual.on_single_appearance_changed(
        SingleAppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=model.id,
            field_name="iso_threshold",
            new_value=0.8,
        )
    )

    assert visual._inner_node_3d.material.threshold == 0.8


@patch("cellier.render.visuals._image_memory.gfx")
def test_identity_transform_is_noop_3d(mock_gfx):
    """Identity transform should produce the same result as no transform."""
    from cellier.render.visuals import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    ctx = Context(3, displayed_axes=(0, 1, 2))
    visual = GFXImageMemoryVisual(
        model, store, render_modes={"3d"}, transform=ctx.transform
    )
    ctx.place(visual)

    dims = _make_dims_state_3d()
    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=600.0,
        dims_state=dims,
        selection=ctx.selection,
    )
    assert len(requests) == 1
    assert requests[0].axis_selections == ((0, 10), (0, 20), (0, 30))
