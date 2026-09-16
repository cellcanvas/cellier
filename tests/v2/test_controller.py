"""Unit tests for CellierController.

All tests are headless — no Qt event loop, no GPU.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier._state import CameraState
from cellier.controller import CellierController
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events._events import CameraChangedEvent, TransformChangedEvent
from cellier.render._config import CameraConfig, RenderManagerConfig
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.transform import WorldCoordinateSystem  # noqa: TC001
from cellier.viewer_model import ViewerModel
from cellier.visuals import (
    LinesMemoryAppearance,
    MeshFlatAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageVisual,
)
from cellier.visuals._points_memory import PointsMarkerAppearance
from tests._v2 import bound, identity, pyramid_levels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cs() -> WorldCoordinateSystem:
    return world_coordinate_system(spatial_axes("z", "y", "x"), name="world")


def _make_appearance(**kwargs) -> MultiscaleImageAppearance:
    defaults = {"color_map": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return MultiscaleImageAppearance(**defaults)


def _make_store(small_zarr_store, **kwargs) -> MultiscaleZarrDataStore:
    """Build a 2-level test store with standard power-of-2 transforms."""
    defaults = {
        "zarr_path": str(small_zarr_store),
        "scale_names": ["s0", "s1"],
        **pyramid_levels(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
    }
    defaults.update(kwargs)
    return MultiscaleZarrDataStore(**defaults)


def _make_mesh_store() -> MeshMemoryStore:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    return MeshMemoryStore(positions=positions, indices=indices)


def _make_points_store() -> PointsMemoryStore:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    return PointsMemoryStore(positions=positions)


def _make_lines_store() -> LinesMemoryStore:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    return LinesMemoryStore(positions=positions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_add_scene_registers_in_model():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    assert scene.id in controller._model.scenes
    assert controller._model.scenes[scene.id].name == "main"


def test_add_data_store_registers_in_model(small_zarr_store):
    controller = CellierController()
    store = _make_store(small_zarr_store)
    result = controller.add_data_store(store)
    assert result is store
    assert store.id in controller._model.data.stores


def test_add_image_populates_scene_and_render_layer(small_zarr_store):
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    appearance = _make_appearance(lod_bias=1.0, force_level=None, frustum_cull=True)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=appearance, name="vol"
    )

    assert isinstance(visual, MultiscaleImageVisual)
    # Model layer
    assert visual in controller._model.scenes[scene.id].visuals
    assert store.id in controller._model.data.stores
    # Render layer
    assert visual.id in controller._render_manager._visual_to_scene


def test_add_image_auto_registers_data_store(small_zarr_store):
    """add_image should register the data store even without add_data_store."""
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )
    assert store.id in controller._model.data.stores


def test_add_mesh_defaults_transform_to_identity():
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    store = _make_mesh_store()

    visual = controller.add_mesh(
        data=store,
        scene_id=scene.id,
        appearance=MeshFlatAppearance(),
        name="mesh",
    )

    expected = identity(store.positions.shape[1])
    np.testing.assert_array_equal(visual.transform.matrix, expected.matrix)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    np.testing.assert_array_equal(gfx_visual._transform.matrix, expected.matrix)


def test_add_mesh_uses_explicit_transform():
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    store = _make_mesh_store()
    transform = bound(controller, scene.id, store, (1.5, 2.0, 3.0), (4.0, 5.0, 6.0))

    visual = controller.add_mesh(
        data=store,
        scene_id=scene.id,
        appearance=MeshFlatAppearance(),
        name="mesh",
        transform=transform,
    )

    np.testing.assert_array_equal(visual.transform.matrix, transform.matrix)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    np.testing.assert_array_equal(gfx_visual._transform.matrix, transform.matrix)


def test_add_points_defaults_transform_to_identity():
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    store = _make_points_store()

    visual = controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(),
        name="points",
    )

    expected = identity(store.ndim)
    np.testing.assert_array_equal(visual.transform.matrix, expected.matrix)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    np.testing.assert_array_equal(gfx_visual._transform.matrix, expected.matrix)


def test_add_points_uses_explicit_transform():
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    store = _make_points_store()
    transform = bound(controller, scene.id, store, (1.5, 2.0, 3.0), (4.0, 5.0, 6.0))

    visual = controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(),
        name="points",
        transform=transform,
    )

    np.testing.assert_array_equal(visual.transform.matrix, transform.matrix)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    np.testing.assert_array_equal(gfx_visual._transform.matrix, transform.matrix)


def test_add_lines_defaults_transform_to_identity():
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    store = _make_lines_store()

    visual = controller.add_lines(
        data=store,
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(),
        name="lines",
    )

    expected = identity(store.ndim)
    np.testing.assert_array_equal(visual.transform.matrix, expected.matrix)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    np.testing.assert_array_equal(gfx_visual._transform.matrix, expected.matrix)


def test_add_lines_uses_explicit_transform():
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    store = _make_lines_store()
    transform = bound(controller, scene.id, store, (1.5, 2.0, 3.0), (4.0, 5.0, 6.0))

    visual = controller.add_lines(
        data=store,
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(),
        name="lines",
        transform=transform,
    )

    np.testing.assert_array_equal(visual.transform.matrix, transform.matrix)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    np.testing.assert_array_equal(gfx_visual._transform.matrix, transform.matrix)


def test_get_scene_by_name():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="my_scene")
    found = controller.get_scene_by_name("my_scene")
    assert found.id == scene.id
    with pytest.raises(KeyError):
        controller.get_scene_by_name("nonexistent")


def test_reslice_scene_reads_appearance_fields(small_zarr_store):
    """reslice_scene must translate MultiscaleImageAppearance fields into VisualRenderConfig."""  # noqa: E501
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    appearance = _make_appearance(lod_bias=2.5, force_level=1, frustum_cull=False)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=appearance, name="vol"
    )

    captured = {}

    def capturing_reslice(s_id, dims_state, visual_configs=None, **_):
        captured.update(visual_configs or {})

    controller._render_manager.reslice_scene = capturing_reslice

    controller.reslice_scene(scene.id)

    assert visual.id in captured
    cfg = captured[visual.id]
    assert cfg.lod_bias == 2.5
    assert cfg.force_level == 1
    assert cfg.frustum_cull is False


def test_to_file_roundtrip(tmp_path, small_zarr_store):
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    path = tmp_path / "session.json"
    controller.to_file(path)

    restored = ViewerModel.from_file(path)
    assert len(restored.scenes) == 1
    assert len(next(iter(restored.scenes.values())).visuals) == 1


# ---------------------------------------------------------------------------
# Event bus tests (headless — no Qt, no GPU)
# ---------------------------------------------------------------------------


def test_dims_bridge_emits_event():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    fired = []
    controller.on_dims_changed(scene.id, fired.append, owner_id=uuid4())

    controller.get_scene(scene.id).dims.selection.slice_indices = {0: 5}
    assert len(fired) == 1
    # The snapshot stopped carrying the positions in Phase 8 (D5); they stay
    # on the selection the sliders write to.
    assert controller.get_scene(scene.id).dims.selection.slice_indices == {0: 5}


def test_dims_bridge_displayed_axes_flag_true():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    events = []
    controller._outgoing_events.subscribe(
        __import__("cellier.events", fromlist=["DimsChangedEvent"]).DimsChangedEvent,
        events.append,
        entity_id=scene.id,
    )

    controller.get_scene(scene.id).dims.selection.displayed_axes = (0, 1)
    assert len(events) == 1
    assert events[0].displayed_axes_changed is True


def test_dims_bridge_displayed_axes_flag_false():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    events = []
    controller._outgoing_events.subscribe(
        __import__("cellier.events", fromlist=["DimsChangedEvent"]).DimsChangedEvent,
        events.append,
        entity_id=scene.id,
    )

    controller.get_scene(scene.id).dims.selection.slice_indices = {
        0: 3.0,
        1: 0.0,
        2: 0.0,
    }
    assert len(events) == 1
    assert events[0].displayed_axes_changed is False


def test_dims_update_event_merges_positions_and_sets_displayed_axes():
    """No ordering dance: every axis keeps a position while displayed (D36).

    The update names only the axes that moved, and the others keep theirs.
    """
    from cellier.events import DimsUpdateEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    assert scene.dims.selection.slice_indices == {0: 0.0, 1: 0.0, 2: 0.0}
    controller.update_slice_indices(scene.id, {2: 7.0})

    controller._incoming_events.emit(
        DimsUpdateEvent(
            source_id=uuid4(),
            scene_id=scene.id,
            slice_indices={0: 4.0},
            displayed_axes=(0, 1, 2),
        )
    )

    assert scene.dims.selection.displayed_axes == (0, 1, 2)
    assert scene.dims.selection.slice_indices == {0: 4.0, 1: 0.0, 2: 7.0}


def test_switching_displayed_axes_keeps_every_position():
    """3D -> 2D -> 3D -> 2D reuses each axis's stored position."""
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    controller.update_slice_indices(scene.id, {0: 5.0, 1: 3.0})

    for displayed in [(0, 1, 2), (1, 2), (0, 1, 2), (1, 2)]:
        controller.set_displayed_axes(scene.id, displayed)
        assert scene.dims.selection.slice_indices == {0: 5.0, 1: 3.0, 2: 0.0}


def test_appearance_bridge_shared_field(small_zarr_store):
    from cellier.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    events = []
    controller._outgoing_events.subscribe(
        AppearanceChangedEvent, events.append, entity_id=visual.id
    )

    visual.appearance.interpolation = "linear"
    assert len(events) == 1
    assert events[0].field_name == "interpolation"
    assert events[0].requires_reslice is False


def test_appearance_bridge_lod_bias(small_zarr_store):
    from cellier.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    events = []
    controller._outgoing_events.subscribe(
        AppearanceChangedEvent, events.append, entity_id=visual.id
    )
    reslice_calls = []
    controller._render_manager.reslice_visual = lambda vid, dims, cfg, **_: (
        reslice_calls.append(vid)
    )

    visual.appearance.lod_bias = 2.0
    assert len(events) == 1
    assert events[0].requires_reslice is True
    assert len(reslice_calls) == 1
    assert reslice_calls[0] == visual.id


def test_appearance_bridge_force_level(small_zarr_store):
    from cellier.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    events = []
    controller._outgoing_events.subscribe(
        AppearanceChangedEvent, events.append, entity_id=visual.id
    )
    reslice_calls = []
    controller._render_manager.reslice_visual = lambda vid, dims, cfg, **_: (
        reslice_calls.append(vid)
    )

    visual.appearance.force_level = 1
    assert len(events) == 1
    assert events[0].requires_reslice is True
    assert len(reslice_calls) == 1
    assert reslice_calls[0] == visual.id


def test_appearance_bridge_frustum_cull(small_zarr_store):
    from cellier.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    events = []
    controller._outgoing_events.subscribe(
        AppearanceChangedEvent, events.append, entity_id=visual.id
    )
    reslice_calls = []
    controller._render_manager.reslice_visual = lambda vid, dims, cfg, **_: (
        reslice_calls.append(vid)
    )

    visual.appearance.frustum_cull = False

    assert events[0].requires_reslice is True
    assert len(reslice_calls) == 1
    assert reslice_calls[0] == visual.id


def test_appearance_bridge_interpolation_does_not_reslice(small_zarr_store):
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )
    reslice_calls = []
    controller._render_manager.reslice_visual = lambda vid, dims, cfg, **_: (
        reslice_calls.append(vid)
    )

    visual.appearance.interpolation = "linear"

    assert len(reslice_calls) == 0


def test_appearance_bridge_visible(small_zarr_store):
    from cellier.events import AppearanceChangedEvent, VisualVisibilityChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    appearance_events = []
    visibility_events = []
    controller._outgoing_events.subscribe(
        AppearanceChangedEvent, appearance_events.append, entity_id=visual.id
    )
    controller._outgoing_events.subscribe(
        VisualVisibilityChangedEvent, visibility_events.append, entity_id=visual.id
    )

    visual.appearance.visible = False
    assert len(visibility_events) == 1
    assert visibility_events[0].visible is False
    assert len(appearance_events) == 0


def test_scene_added_event_emitted():
    from cellier.events import SceneAddedEvent

    controller = CellierController()
    events = []
    controller._outgoing_events.subscribe(SceneAddedEvent, events.append)

    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    assert len(events) == 1
    assert events[0].scene_id == scene.id


def test_visual_added_event_emitted(small_zarr_store):
    from cellier.events import VisualAddedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)

    events = []
    controller._outgoing_events.subscribe(VisualAddedEvent, events.append)

    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    assert len(events) == 1
    assert events[0].visual_id == visual.id
    assert events[0].scene_id == scene.id


def test_on_dims_changed_callback():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    fired = []
    controller.on_dims_changed(scene.id, fired.append, owner_id=uuid4())

    controller.get_scene(scene.id).dims.selection.slice_indices = {0: 7}
    assert len(fired) == 1
    assert controller.get_scene(scene.id).dims.selection.slice_indices == {0: 7}


def test_unsubscribe_all_cleans_up(small_zarr_store):
    from cellier.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    fired = []
    controller._outgoing_events.subscribe(
        AppearanceChangedEvent,
        fired.append,
        entity_id=visual.id,
        owner_id=visual.id,
    )
    controller._outgoing_events.unsubscribe_all(visual.id)

    visual.appearance.interpolation = "linear"
    assert fired == []


# ---------------------------------------------------------------------------
# Camera settle tests (async — no Qt, no GPU)
# ---------------------------------------------------------------------------


def _make_camera_state() -> CameraState:
    """Minimal perspective CameraState for event payloads."""
    return CameraState(
        camera_type="perspective",
        position=(0.0, 0.0, 100.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fov=70.0,
        zoom=1.0,
        extent=(0.0, 0.0),
        depth_range=(1.0, 8000.0),
    )


def _make_camera_event(canvas_id, scene_id) -> CameraChangedEvent:
    return CameraChangedEvent(
        source_id=canvas_id,
        scene_id=scene_id,
        camera_state=_make_camera_state(),
    )


def _make_settle_controller(small_zarr_store, threshold_s=0.05):
    """Build a minimal controller wired for settle tests.

    Returns (controller, scene, visual, canvas_id, reslice_calls).
    """
    controller = CellierController(
        render_config=RenderManagerConfig(
            camera=CameraConfig(settle_threshold_s=threshold_s)
        )
    )
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=_make_appearance(lod_bias=1.0, force_level=None, frustum_cull=True),
        name="vol",
    )

    reslice_calls = []

    def _capturing_reslice(
        scene_id, dims_state, visual_configs=None, target_visual_ids=None, **_
    ):
        reslice_calls.append(
            {
                "scene_id": scene_id,
                "target_visual_ids": target_visual_ids,
            }
        )

    controller._render_manager.reslice_scene = _capturing_reslice
    # Bypass camera model writeback — no canvas model wired in headless tests.
    controller._update_camera_model = lambda scene_id, canvas_id, camera_state: None

    canvas_id = uuid4()
    return controller, scene, visual, canvas_id, reslice_calls


async def test_settle_fires_reslice_after_threshold(small_zarr_store):
    controller, scene, _visual, canvas_id, reslice_calls = _make_settle_controller(
        small_zarr_store, threshold_s=0.05
    )

    event = _make_camera_event(canvas_id, scene.id)
    controller._on_camera_changed(event)

    # Before threshold: no reslice yet.
    await asyncio.sleep(0.01)
    assert len(reslice_calls) == 0

    # After threshold: exactly one reslice.
    await asyncio.sleep(0.1)
    assert len(reslice_calls) == 1


async def test_settle_cancellation_on_rapid_events(small_zarr_store):
    controller, scene, _visual, canvas_id, reslice_calls = _make_settle_controller(
        small_zarr_store, threshold_s=0.05
    )

    event = _make_camera_event(canvas_id, scene.id)
    controller._on_camera_changed(event)  # starts first settle task
    controller._on_camera_changed(event)  # cancels it; starts second

    await asyncio.sleep(0.15)
    assert len(reslice_calls) == 1


async def test_settle_disabled_flag_suppresses_reslice(small_zarr_store):
    controller, scene, _visual, canvas_id, reslice_calls = _make_settle_controller(
        small_zarr_store, threshold_s=0.05
    )

    # Disable camera reslicing.
    controller.camera_reslice_enabled = False

    event = _make_camera_event(canvas_id, scene.id)
    controller._on_camera_changed(event)

    # Wait well past the threshold.
    await asyncio.sleep(0.15)
    assert len(reslice_calls) == 0

    # Re-enable and confirm reslicing resumes.
    controller.camera_reslice_enabled = True
    controller._on_camera_changed(event)
    await asyncio.sleep(0.15)
    assert len(reslice_calls) == 1


async def test_settle_target_visual_ids_excludes_non_reslice_visuals(small_zarr_store):
    from cellier.visuals._base_visual import BaseAppearance, BaseVisual

    class _NonResliceVisual(BaseVisual):
        appearance: BaseAppearance = BaseAppearance()

    _NonResliceVisual.model_rebuild()

    controller, scene, visual, canvas_id, reslice_calls = _make_settle_controller(
        small_zarr_store, threshold_s=0.05
    )

    # Append a non-camera-reslice visual directly to the scene model.
    non_reslice = _NonResliceVisual(
        name="static", data_store_id="00000000-0000-0000-0000-000000000000"
    )
    scene.visuals.append(non_reslice)

    event = _make_camera_event(canvas_id, scene.id)
    controller._on_camera_changed(event)
    await asyncio.sleep(0.15)

    assert len(reslice_calls) == 1
    ids = reslice_calls[0]["target_visual_ids"]
    assert visual.id in ids
    assert non_reslice.id not in ids


def test_on_camera_changed_updates_orthographic_camera_model():
    """OrthographicCamera.width/height are written back by _on_camera_changed."""
    from cellier.scene import Canvas
    from cellier.scene.cameras import OrthographicCamera, PanZoomCameraController
    from cellier.scene.dims import AxisAlignedSelection, DimsManager
    from cellier.scene.scene import Scene

    cs = _make_cs()
    dims = DimsManager(
        world_coordinate_system=cs,
        selection=AxisAlignedSelection(
            displayed_axes=(1, 2),
            slice_indices={0: 0, 1: 0, 2: 0},
        ),
    )
    ortho_camera = OrthographicCamera(
        width=100.0,
        height=100.0,
        controller=PanZoomCameraController(enabled=True),
    )
    canvas_model = Canvas(cameras={"2d": ortho_camera})
    scene = Scene(
        name="test_2d",
        dims=dims,
        render_modes={"2d"},
        visuals=[],
        canvases={canvas_model.id: canvas_model},
    )

    controller = CellierController()
    controller._model.scenes[scene.id] = scene
    controller.camera_reslice_enabled = False

    ortho_state = CameraState(
        camera_type="orthographic",
        position=(5.0, 10.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fov=0.0,
        zoom=1.5,
        extent=(320.0, 240.0),
        depth_range=(-500.0, 500.0),
    )
    event = CameraChangedEvent(
        source_id=canvas_model.id,
        scene_id=scene.id,
        camera_state=ortho_state,
    )

    controller._on_camera_changed(event)

    updated = canvas_model.cameras["2d"]
    assert isinstance(updated, OrthographicCamera)
    assert updated.width == pytest.approx(320.0)
    assert updated.height == pytest.approx(240.0)
    assert updated.zoom == pytest.approx(1.5)
    assert updated.near_clipping_plane == pytest.approx(-500.0)
    assert updated.far_clipping_plane == pytest.approx(500.0)
    assert updated.position == pytest.approx(np.array([5.0, 10.0, 0.0]))


# ---------------------------------------------------------------------------
# Transform wiring tests
# ---------------------------------------------------------------------------


def test_wire_transform_emits_event(small_zarr_store):
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    fired = []
    controller._outgoing_events.subscribe(
        TransformChangedEvent, fired.append, entity_id=visual.id
    )

    new_t = bound(controller, scene.id, store, (2.0, 2.0, 2.0))
    visual.transform = new_t

    assert len(fired) == 1
    assert fired[0].visual_id == visual.id
    assert fired[0].scene_id == scene.id
    np.testing.assert_array_equal(fired[0].transform.matrix, new_t.matrix)


def test_transform_change_triggers_reslice(small_zarr_store):
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    reslice_calls = []

    def _capture(
        scene_id, dims_state, visual_configs=None, target_visual_ids=None, **_
    ):
        reslice_calls.append(scene_id)

    controller._render_manager.reslice_scene = _capture

    visual.transform = bound(
        controller, scene.id, store, (1.0, 1.0, 1.0), (10.0, 0.0, 0.0)
    )

    assert len(reslice_calls) == 1
    assert reslice_calls[0] == scene.id


# ---------------------------------------------------------------------------
# remove_* tests
# ---------------------------------------------------------------------------


def _make_scene_with_visual(controller, small_zarr_store):
    """Add a scene, data store, and image visual; return (scene, visual, store)."""
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    store = _make_store(small_zarr_store)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )
    return scene, visual, store


# -- remove_visual -----------------------------------------------------------


def test_remove_visual_cleans_model(small_zarr_store):
    controller = CellierController()
    scene, visual, _ = _make_scene_with_visual(controller, small_zarr_store)

    controller.remove_visual(visual.id)

    assert visual not in controller._model.scenes[scene.id].visuals
    assert visual.id not in controller._visual_to_scene
    assert visual.id not in controller._visual_psygnal_handlers


def test_remove_visual_cleans_render_layer(small_zarr_store):
    controller = CellierController()
    scene, visual, _ = _make_scene_with_visual(controller, small_zarr_store)

    controller.remove_visual(visual.id)

    scene_manager = controller._render_manager._scenes[scene.id]
    assert visual.id not in scene_manager._visuals
    assert visual.id not in controller._render_manager._visual_to_scene


def test_remove_visual_disconnects_psygnal_bridge(small_zarr_store):
    """After removal, mutating appearance must not fire any bus event."""
    from cellier.events import AppearanceChangedEvent

    controller = CellierController()
    _scene, visual, _ = _make_scene_with_visual(controller, small_zarr_store)

    fired = []
    controller._outgoing_events.subscribe(
        AppearanceChangedEvent, fired.append, entity_id=visual.id
    )

    controller.remove_visual(visual.id)
    visual.appearance.interpolation = "linear"

    assert fired == []


def test_remove_visual_emits_event(small_zarr_store):
    from cellier.events import VisualRemovedEvent

    controller = CellierController()
    scene, visual, _ = _make_scene_with_visual(controller, small_zarr_store)

    events = []
    controller._outgoing_events.subscribe(VisualRemovedEvent, events.append)

    controller.remove_visual(visual.id)

    assert len(events) == 1
    assert events[0].visual_id == visual.id
    assert events[0].scene_id == scene.id


# -- remove_scene ------------------------------------------------------------


def test_remove_scene_cleans_model(small_zarr_store):
    controller = CellierController()
    scene, _visual, _ = _make_scene_with_visual(controller, small_zarr_store)

    controller.remove_scene(scene.id)

    assert scene.id not in controller._model.scenes
    assert scene.id not in controller._dims_cache
    assert scene.id not in controller._scene_render_modes
    assert scene.id not in controller._scene_to_canvases


def test_remove_scene_cascades_to_visuals(small_zarr_store):
    """Removing a scene must also tear down its child visual."""
    controller = CellierController()
    scene, visual, _ = _make_scene_with_visual(controller, small_zarr_store)

    controller.remove_scene(scene.id)

    assert visual.id not in controller._visual_to_scene
    assert visual.id not in controller._render_manager._visual_to_scene


async def test_remove_scene_cancels_settle_task(small_zarr_store):
    controller, scene, _visual, canvas_id, _ = _make_settle_controller(
        small_zarr_store, threshold_s=10.0
    )

    # Register the fake canvas so remove_scene can find it in _scene_to_canvases.
    # This mirrors what add_canvas_model does in real usage.
    controller._scene_to_canvases[scene.id].append(canvas_id)
    controller._canvas_to_scene[canvas_id] = scene.id

    event = _make_camera_event(canvas_id, scene.id)
    controller._on_camera_changed(event)
    task = controller._settle_tasks.get(canvas_id)
    assert task is not None and not task.done()

    controller.remove_scene(scene.id)
    assert canvas_id not in controller._settle_tasks

    # Yield to the event loop so the CancelledError is processed.
    await asyncio.sleep(0)
    assert task.cancelled()


def test_remove_scene_emits_event(small_zarr_store):
    from cellier.events import SceneRemovedEvent

    controller = CellierController()
    scene, _visual, _ = _make_scene_with_visual(controller, small_zarr_store)

    events = []
    controller._outgoing_events.subscribe(SceneRemovedEvent, events.append)

    controller.remove_scene(scene.id)

    assert len(events) == 1
    assert events[0].scene_id == scene.id


# -- remove_data_store -------------------------------------------------------


def test_remove_data_store_happy_path(small_zarr_store):
    controller = CellierController()
    _scene, visual, store = _make_scene_with_visual(controller, small_zarr_store)

    controller.remove_visual(visual.id)
    controller.remove_data_store(store.id)

    assert store.id not in controller._model.data.stores


def test_remove_data_store_raises_when_referenced(small_zarr_store):
    controller = CellierController()
    _scene, visual, store = _make_scene_with_visual(controller, small_zarr_store)

    with pytest.raises(ValueError) as exc_info:
        controller.remove_data_store(store.id)

    msg = str(exc_info.value)
    assert visual.name in msg
    assert str(visual.id) in msg


# ---------------------------------------------------------------------------
# Hidden image visuals skip slicing
# ---------------------------------------------------------------------------


def _capture_scene_configs(controller) -> dict:
    captured: dict = {}

    def capturing_reslice(s_id, dims_state, visual_configs=None, **_):
        captured.update(visual_configs or {})

    controller._render_manager.reslice_scene = capturing_reslice
    return captured


def _capture_visual_reslices(controller) -> list:
    calls: list = []

    def capturing_reslice(visual_id, dims_state, visual_config=None, **_):
        calls.append((visual_id, visual_config))

    controller._render_manager.reslice_visual = capturing_reslice
    return calls


def test_hidden_image_visual_disables_slicing(small_zarr_store):
    """A hidden multiscale image is planned with slicing_enabled=False."""
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    visual = controller.add_image_multiscale(
        data=_make_store(small_zarr_store),
        scene_id=scene.id,
        appearance=_make_appearance(visible=False),
        name="vol",
    )
    captured = _capture_scene_configs(controller)

    controller.reslice_scene(scene.id)
    assert captured[visual.id].slicing_enabled is False

    visual.appearance.visible = True
    controller.reslice_scene(scene.id)
    assert captured[visual.id].slicing_enabled is True


def test_showing_image_visual_reslices_it(small_zarr_store):
    """Showing a hidden image reslices it; hiding one does not."""
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    visual = controller.add_image_multiscale(
        data=_make_store(small_zarr_store),
        scene_id=scene.id,
        appearance=_make_appearance(),
        name="vol",
    )
    calls = _capture_visual_reslices(controller)

    visual.appearance.visible = False
    assert calls == []

    visual.appearance.visible = True
    assert len(calls) == 1
    visual_id, cfg = calls[0]
    assert visual_id == visual.id
    assert cfg.slicing_enabled is True


def test_hidden_non_image_visual_keeps_slicing():
    """The hidden-visual gate only applies to image visuals."""
    controller = CellierController()
    scene = controller.add_scene(dim="3d", coordinate_system=_make_cs(), name="main")
    visual = controller.add_lines(
        data=_make_lines_store(),
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(visible=False),
        name="lines",
    )
    captured = _capture_scene_configs(controller)

    controller.reslice_scene(scene.id)
    assert captured[visual.id].slicing_enabled is True
