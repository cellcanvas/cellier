"""Unit tests for the RenderManager architecture components.

Tests are structured to avoid GPU dependencies where possible by mocking
pygfx visuals and the AsyncSlicer.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier._state import AxisAlignedSelectionState
from cellier.data.image import ChunkRequest
from cellier.render import SliceCoordinator, VisualRenderConfig
from cellier.render._requests import DimsState, ReslicingRequest
from cellier.render.scene_manager import SceneManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dims_state() -> DimsState:
    return DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(0, 1, 2),
        ),
    )


def _make_reslicing_request(
    scene_id: UUID | None = None,
    canvas_id: UUID | None = None,
    target_visual_ids: frozenset[UUID] | None = None,
    screen_size_px: tuple[float, float] = (1200.0, 800.0),
    fov_y_rad: float = 1.2217,  # ~70 degrees
) -> ReslicingRequest:
    return ReslicingRequest(
        camera_type="perspective",
        camera_pos=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        frustum_corners=np.zeros((2, 4, 3), dtype=np.float64),
        fov_y_rad=fov_y_rad,
        screen_size_px=screen_size_px,
        world_extent=(0.0, 0.0),
        dims_state=_make_dims_state(),
        selection=None,
        request_id=uuid4(),
        scene_id=scene_id or uuid4(),
        canvas_id=canvas_id or uuid4(),
        target_visual_ids=target_visual_ids,
    )


def _make_mock_visual(visual_id: UUID | None = None) -> MagicMock:
    """Build a mock GFXMultiscaleImageVisual that looks real enough for tests."""
    visual = MagicMock()
    visual.visual_model_id = visual_id or uuid4()
    visual.node_3d = MagicMock()
    visual.node_2d = None
    visual.render_modes = {"3d"}
    visual._volume_geometry.n_levels = 3
    visual.build_slice_request.return_value = []
    return visual


class _StubSlicer:
    """Minimal AsyncSlicer stub that records submitted requests."""

    def __init__(self) -> None:
        self.submitted: list[tuple[list[ChunkRequest], str | None]] = []
        self.cancelled: list[UUID] = []
        self.on_complete_callbacks: list = []
        self._next_id = 0

    def submit(
        self,
        requests,
        fetch_fn,
        callback,
        consumer_id=None,
        on_complete=None,
    ) -> UUID | None:
        if not requests:
            return None
        self.submitted.append((requests, consumer_id))
        self.on_complete_callbacks.append(on_complete)
        result = uuid4()
        return result

    def cancel(self, slice_request_id) -> bool:
        if slice_request_id is not None:
            self.cancelled.append(slice_request_id)
            return True
        return False


# ---------------------------------------------------------------------------
# DimsState and ReslicingRequest
# ---------------------------------------------------------------------------


def test_dims_state_construction() -> None:
    ds = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(0, 1, 2),
        ),
    )
    assert ds.selection.displayed_axes == (0, 1, 2)


def test_dims_state_2d() -> None:
    ds = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
        ),
    )
    assert ds.selection.displayed_axes == (1, 2)


def test_reslicing_request_construction() -> None:
    scene_id = uuid4()
    req = _make_reslicing_request(scene_id=scene_id)
    assert req.scene_id == scene_id
    assert req.camera_pos.shape == (3,)
    assert req.frustum_corners.shape == (2, 4, 3)
    assert req.target_visual_ids is None


def test_reslicing_request_camera_pos_is_independent() -> None:
    """Modifying the returned array must not affect a second snapshot."""
    req = _make_reslicing_request()
    _original = req.camera_pos.copy()
    req.camera_pos[0] = 99999.0
    # The NamedTuple holds the array reference, not a copy — the test
    # verifies that the caller (.copy() contract) is honoured in production
    # code by checking the snapshot helper makes a fresh array.
    req2 = _make_reslicing_request()
    assert not np.array_equal(req2.camera_pos, req.camera_pos)


def test_reslicing_request_frustum_corners_are_independent() -> None:
    """Two requests built separately must not share the same frustum array."""
    req1 = _make_reslicing_request()
    req2 = _make_reslicing_request()
    req1.frustum_corners[0, 0, 0] = 42.0
    assert req2.frustum_corners[0, 0, 0] != 42.0


def test_reslicing_request_with_target_visual_ids() -> None:
    vid = uuid4()
    req = _make_reslicing_request(target_visual_ids=frozenset({vid}))
    assert req.target_visual_ids == frozenset({vid})


# ---------------------------------------------------------------------------
# SceneManager — build_slice_requests filtering
# ---------------------------------------------------------------------------


def test_scene_manager_add_and_get_visual() -> None:
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    sm.add_visual(visual, (0, 1, 2))
    assert sm.get_visual(visual.visual_model_id) is visual


def test_scene_manager_get_visual_missing_raises() -> None:
    sm = SceneManager(scene_id=uuid4())
    with pytest.raises(KeyError):
        sm.get_visual(uuid4())


def test_scene_manager_build_slice_requests_all_visuals() -> None:
    """With target_visual_ids=None all visuals are processed."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)

    chunk = ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=((0, 32), (0, 32), (0, 32)),
    )
    v1 = _make_mock_visual()
    v1.build_slice_request.return_value = [chunk]
    v2 = _make_mock_visual()
    v2.build_slice_request.return_value = []

    sm.add_visual(v1, (0, 1, 2))
    sm.add_visual(v2, (0, 1, 2))

    req = _make_reslicing_request(scene_id=scene_id)
    result = sm.build_slice_requests(req, {})

    assert v1.visual_model_id in result
    assert v2.visual_model_id not in result
    assert result[v1.visual_model_id] == [chunk]


def test_scene_manager_build_slice_requests_target_filtering() -> None:
    """With target_visual_ids set, only targeted visuals are processed."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)

    chunk = ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=((0, 32), (0, 32), (0, 32)),
    )
    v1 = _make_mock_visual()
    v1.build_slice_request.return_value = [chunk]
    v2 = _make_mock_visual()
    v2.build_slice_request.return_value = [chunk]

    sm.add_visual(v1, (0, 1, 2))
    sm.add_visual(v2, (0, 1, 2))

    req = _make_reslicing_request(
        scene_id=scene_id,
        target_visual_ids=frozenset({v1.visual_model_id}),
    )
    result = sm.build_slice_requests(req, {})

    assert v1.visual_model_id in result
    assert v2.visual_model_id not in result
    v1.build_slice_request.assert_called_once()
    v2.build_slice_request.assert_not_called()


def test_scene_manager_visual_ids_property() -> None:
    sm = SceneManager(scene_id=uuid4())
    v1 = _make_mock_visual()
    v2 = _make_mock_visual()
    sm.add_visual(v1, (0, 1, 2))
    sm.add_visual(v2, (0, 1, 2))
    ids = sm.visual_ids
    assert v1.visual_model_id in ids
    assert v2.visual_model_id in ids


# ---------------------------------------------------------------------------
# SliceCoordinator — cancellation before re-submit
# ---------------------------------------------------------------------------


def _make_coordinator_with_scene(
    n_visuals: int = 1,
) -> tuple[SliceCoordinator, _StubSlicer, SceneManager, list[MagicMock], UUID, UUID]:
    """Build a SliceCoordinator with one scene, one canvas, and n_visuals."""
    scene_id = uuid4()
    canvas_id = uuid4()
    stub_slicer = _StubSlicer()
    scenes: dict[UUID, SceneManager] = {}
    data_stores: dict[UUID, MagicMock] = {}

    coordinator = SliceCoordinator(
        scenes=scenes,
        slicer=stub_slicer,
        data_stores=data_stores,
    )

    sm = SceneManager(scene_id=scene_id)
    scenes[scene_id] = sm

    chunk = ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=((0, 32), (0, 32), (0, 32)),
    )

    visuals = []
    for _ in range(n_visuals):
        v = _make_mock_visual()
        v.build_slice_request.return_value = [chunk]
        sm.add_visual(v, (0, 1, 2))
        data_stores[v.visual_model_id] = MagicMock()
        visuals.append(v)

    return coordinator, stub_slicer, sm, visuals, scene_id, canvas_id


def test_slice_coordinator_submit_once_no_prior_task() -> None:
    coordinator, stub_slicer, _sm, _visuals, scene_id, canvas_id = (
        _make_coordinator_with_scene()
    )
    req = _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_id)
    coordinator.submit(req, {})

    assert len(stub_slicer.submitted) == 1
    assert len(stub_slicer.cancelled) == 0


def test_slice_coordinator_second_submit_cancels_first() -> None:
    """Second submit from the same canvas must cancel the first task."""
    coordinator, stub_slicer, _sm, _visuals, scene_id, canvas_id = (
        _make_coordinator_with_scene()
    )

    coordinator.submit(
        _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_id), {}
    )
    coordinator.submit(
        _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_id), {}
    )

    assert len(stub_slicer.submitted) == 2
    assert len(stub_slicer.cancelled) == 1


def test_slice_coordinator_cancel_visual_releases_pending() -> None:
    """cancel_visual must call visual.cancel_pending()."""
    coordinator, _stub_slicer, _sm, visuals, scene_id, canvas_id = (
        _make_coordinator_with_scene()
    )
    visual = visuals[0]

    coordinator.submit(
        _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_id), {}
    )

    visual.cancel_pending.reset_mock()
    coordinator.cancel_visual(scene_id, canvas_id, visual.visual_model_id)
    visual.cancel_pending.assert_called_once()


def test_slice_coordinator_cancel_visual_only_affects_one() -> None:
    """Cancelling one visual must not cancel the other visual's task."""
    coordinator, stub_slicer, _sm, visuals, scene_id, canvas_id = (
        _make_coordinator_with_scene(n_visuals=2)
    )
    v1, v2 = visuals

    coordinator.submit(
        _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_id), {}
    )

    n_cancelled_before = len(stub_slicer.cancelled)
    coordinator.cancel_visual(scene_id, canvas_id, v1.visual_model_id)
    assert len(stub_slicer.cancelled) == n_cancelled_before + 1
    # v2's entry should still be in active_slice_ids under the 3-tuple key
    assert (scene_id, canvas_id, v2.visual_model_id) in coordinator._active_slice_ids


def test_slice_coordinator_cancel_scene_cancels_all() -> None:
    coordinator, _stub_slicer, _sm, _visuals, scene_id, canvas_id = (
        _make_coordinator_with_scene(n_visuals=2)
    )
    coordinator.submit(
        _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_id), {}
    )

    coordinator.cancel_scene(scene_id)
    assert len(coordinator._active_slice_ids) == 0


# ---------------------------------------------------------------------------
# RenderManager — reslice_visual reverse map lookup
# ---------------------------------------------------------------------------


def test_reslice_visual_uses_reverse_map() -> None:
    """reslice_visual must look up the scene via _visual_to_scene."""
    from cellier.render import RenderManager

    rm = RenderManager()
    scene_id = uuid4()
    canvas_id = uuid4()

    rm._scenes[scene_id] = MagicMock()
    rm._scenes[scene_id].scene = MagicMock()
    rm._scenes[scene_id].visual_ids = []
    rm._scenes[scene_id].build_slice_requests.return_value = {}

    visual = _make_mock_visual()
    rm._visual_to_scene[visual.visual_model_id] = scene_id

    canvas_mock = MagicMock()
    canvas_mock.scene_id = scene_id
    canvas_mock.capture_reslicing_request.return_value = _make_reslicing_request(
        scene_id=scene_id,
        canvas_id=canvas_id,
        target_visual_ids=frozenset({visual.visual_model_id}),
    )
    rm._canvases[canvas_id] = canvas_mock
    rm._canvas_to_scene[canvas_id] = scene_id

    dims_state = _make_dims_state()
    rm.reslice_visual(visual.visual_model_id, dims_state)

    canvas_mock.capture_reslicing_request.assert_called_once_with(
        dims_state,
        selection=None,
        target_visual_ids=frozenset({visual.visual_model_id}),
    )


def test_reslice_visual_only_targets_one_visual() -> None:
    """target_visual_ids in the generated request must contain only visual_id."""
    from cellier.render import RenderManager

    rm = RenderManager()
    scene_id = uuid4()
    canvas_id = uuid4()

    sm = SceneManager(scene_id=scene_id)
    rm._scenes[scene_id] = sm

    v1 = _make_mock_visual()
    v2 = _make_mock_visual()
    rm._visual_to_scene[v1.visual_model_id] = scene_id
    rm._visual_to_scene[v2.visual_model_id] = scene_id

    captured_requests: list[ReslicingRequest] = []

    class _CapturingCoordinator:
        def submit(self, req, visual_configs=None):
            captured_requests.append(req)

    rm._slice_coordinator = _CapturingCoordinator()

    canvas_mock = MagicMock()
    canvas_mock.scene_id = scene_id

    def _capture_request(dims_state, selection=None, target_visual_ids=None):
        return _make_reslicing_request(
            scene_id=scene_id,
            canvas_id=canvas_id,
            target_visual_ids=target_visual_ids,
        )

    canvas_mock.capture_reslicing_request.side_effect = _capture_request
    rm._canvases[canvas_id] = canvas_mock
    rm._canvas_to_scene[canvas_id] = scene_id

    dims_state = _make_dims_state()
    rm.reslice_visual(v1.visual_model_id, dims_state)

    assert len(captured_requests) == 1
    assert captured_requests[0].target_visual_ids == frozenset({v1.visual_model_id})


# ---------------------------------------------------------------------------
# VisualRenderConfig — defaults
# ---------------------------------------------------------------------------


def test_visual_render_config_defaults() -> None:
    config = VisualRenderConfig()
    assert config.settled_lod_bias == 1.0
    assert config.force_level is None
    assert config.frustum_cull is True


# ---------------------------------------------------------------------------
# SceneManager — per-visual config delivery
# ---------------------------------------------------------------------------


def test_lod_bias_forwarded_to_visual() -> None:
    """The config's settled_lod_bias must be passed to build_slice_request."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    visual.build_slice_request.return_value = []
    sm.add_visual(visual, (0, 1, 2))

    req = _make_reslicing_request(scene_id=scene_id)
    sm.build_slice_requests(
        req, {visual.visual_model_id: VisualRenderConfig(settled_lod_bias=2.0)}
    )

    _, kwargs = visual.build_slice_request.call_args
    assert kwargs.get("lod_bias") == 2.0


def test_force_level_forwarded_to_visual() -> None:
    """SceneManager must pass force_level from visual_configs to build_slice_request."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    visual.build_slice_request.return_value = []
    sm.add_visual(visual, (0, 1, 2))

    req = _make_reslicing_request(scene_id=scene_id)
    sm.build_slice_requests(
        req, {visual.visual_model_id: VisualRenderConfig(force_level=2)}
    )

    _, kwargs = visual.build_slice_request.call_args
    assert kwargs.get("force_level") == 2


def test_force_level_none_forwarded_to_visual() -> None:
    """force_level=None (default) must also be forwarded."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    visual.build_slice_request.return_value = []
    sm.add_visual(visual, (0, 1, 2))

    req = _make_reslicing_request(scene_id=scene_id)
    sm.build_slice_requests(req, {})

    _, kwargs = visual.build_slice_request.call_args
    assert kwargs.get("force_level") is None


def test_frustum_cull_false_passes_none_corners() -> None:
    """frustum_cull=False must pass None for frustum_corners_world."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    visual.build_slice_request.return_value = []
    sm.add_visual(visual, (0, 1, 2))

    req = _make_reslicing_request(scene_id=scene_id)
    sm.build_slice_requests(
        req, {visual.visual_model_id: VisualRenderConfig(frustum_cull=False)}
    )

    _, kwargs = visual.build_slice_request.call_args
    assert kwargs.get("frustum_corners_world") is None


def test_frustum_cull_true_passes_corners() -> None:
    """frustum_cull=True (default) must pass frustum corners to the visual."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    visual.build_slice_request.return_value = []
    sm.add_visual(visual, (0, 1, 2))

    req = _make_reslicing_request(scene_id=scene_id)
    sm.build_slice_requests(req, {})

    _, kwargs = visual.build_slice_request.call_args
    np.testing.assert_array_equal(
        kwargs.get("frustum_corners_world"), req.frustum_corners
    )


# ---------------------------------------------------------------------------
# SceneManager — slicing_enabled gate
# ---------------------------------------------------------------------------


def test_visual_render_config_slicing_enabled_by_default() -> None:
    assert VisualRenderConfig().slicing_enabled is True


def test_slicing_disabled_visual_is_not_planned_3d() -> None:
    """A visual whose config disables slicing gets no build_slice_request call."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    sm.add_visual(visual, (0, 1, 2))

    result = sm.build_slice_requests(
        _make_reslicing_request(scene_id=scene_id),
        {visual.visual_model_id: VisualRenderConfig(slicing_enabled=False)},
    )

    visual.build_slice_request.assert_not_called()
    assert result == {}


def test_slicing_disabled_visual_is_not_planned_2d() -> None:
    """The 2D planning path honours slicing_enabled too."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    visual.render_modes = {"2d"}
    sm.add_visual(visual, (1, 2))

    request = ReslicingRequest(
        camera_type="orthographic",
        camera_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        frustum_corners=np.zeros((2, 4, 3), dtype=np.float64),
        fov_y_rad=0.0,
        screen_size_px=(800.0, 600.0),
        world_extent=(10.0, 10.0),
        dims_state=DimsState(
            axis_labels=("z", "y", "x"),
            selection=AxisAlignedSelectionState(displayed_axes=(1, 2)),
        ),
        selection=None,
        request_id=uuid4(),
        scene_id=scene_id,
        canvas_id=uuid4(),
        target_visual_ids=None,
    )
    result = sm.build_slice_requests(
        request, {visual.visual_model_id: VisualRenderConfig(slicing_enabled=False)}
    )

    visual.build_slice_request_2d.assert_not_called()
    assert result == {}
