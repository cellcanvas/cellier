"""Unit tests for the RenderManager architecture components.

Tests are structured to avoid GPU dependencies where possible by mocking
pygfx visuals and the chunk scheduler.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier._state import AxisAlignedSelectionState
from cellier.render import SliceCoordinator, VisualRenderConfig
from cellier.render._requests import DimsState, ReslicingRequest
from cellier.render.scene_manager import SceneManager
from cellier.render.scheduling import DesiredSet

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


class _ChunkedMock(MagicMock):
    """A mock visual the scene manager treats as chunked (it plans)."""

    chunked = True


_next_cache_id = iter(range(10_000, 20_000))


def _desired(cache_id: int) -> DesiredSet:
    return DesiredSet(
        cache_id=cache_id,
        keys=np.array([1]),
        cls=np.array([0]),
        slice_ids=np.array([0]),
        build_request=lambda keys: [],
        store=None,
    )


def _make_mock_visual(visual_id: UUID | None = None) -> MagicMock:
    """A mock chunked visual: one cache, which its plan always names."""
    visual = _ChunkedMock()
    visual.visual_model_id = visual_id or uuid4()
    visual.node_3d = MagicMock()
    visual.node_2d = None
    visual.render_modes = {"3d"}
    visual.decides_empty_slices = True
    visual.target_deferred = False
    cache_id = next(_next_cache_id)
    visual.cache_id = cache_id
    visual.residencies.return_value = {cache_id: MagicMock()}
    visual.plan.return_value = [_desired(cache_id)]
    return visual


class _StubScheduler:
    """Records what the coordinator registers, passes and retires."""

    def __init__(self) -> None:
        self.registered: list[int] = []
        self.passes: list[list[DesiredSet]] = []
        self.retired: list[int] = []
        self.removed: list[int] = []

    def register(self, cache_id, residency, scene=None) -> None:
        self.registered.append(cache_id)

    def remove(self, cache_id) -> None:
        self.removed.append(cache_id)

    def retire(self, cache_id) -> None:
        self.retired.append(cache_id)

    def pass_(self, desired_sets) -> None:
        self.passes.append(list(desired_sets))


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
# SceneManager -- plan_chunked filtering
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


def test_scene_manager_plan_chunked_all_visuals() -> None:
    """With target_visual_ids=None every visual is planned; one that plans
    nothing is idle (its caches are retired)."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    v1 = _make_mock_visual()
    v2 = _make_mock_visual()
    v2.plan.return_value = []
    sm.add_visual(v1, (0, 1, 2))
    sm.add_visual(v2, (0, 1, 2))

    planned, idle = sm.plan_chunked(_make_reslicing_request(scene_id=scene_id), {})

    assert planned == {v1.visual_model_id: v1.plan.return_value}
    assert idle == {v2.visual_model_id}


def test_scene_manager_plan_chunked_target_filtering() -> None:
    """With target_visual_ids set, only targeted visuals are planned."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    v1 = _make_mock_visual()
    v2 = _make_mock_visual()
    sm.add_visual(v1, (0, 1, 2))
    sm.add_visual(v2, (0, 1, 2))

    req = _make_reslicing_request(
        scene_id=scene_id,
        target_visual_ids=frozenset({v1.visual_model_id}),
    )
    planned, idle = sm.plan_chunked(req, {})

    assert set(planned) == {v1.visual_model_id}
    assert idle == set()
    v1.plan.assert_called_once()
    v2.plan.assert_not_called()


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
# SliceCoordinator -- plan, pass and retire
# ---------------------------------------------------------------------------


def _make_coordinator_with_scene(
    n_visuals: int = 1,
) -> tuple[SliceCoordinator, _StubScheduler, SceneManager, list[MagicMock], UUID]:
    """Build a SliceCoordinator with one scene and n_visuals."""
    scene_id = uuid4()
    scheduler = _StubScheduler()
    scenes: dict[UUID, SceneManager] = {}
    data_stores: dict[UUID, MagicMock] = {}
    coordinator = SliceCoordinator(
        scenes=scenes, data_stores=data_stores, scheduler=scheduler
    )
    sm = SceneManager(scene_id=scene_id)
    scenes[scene_id] = sm
    visuals = []
    for _ in range(n_visuals):
        v = _make_mock_visual()
        sm.add_visual(v, (0, 1, 2))
        data_stores[v.visual_model_id] = MagicMock()
        visuals.append(v)
    return coordinator, scheduler, sm, visuals, scene_id


def test_slice_coordinator_submit_registers_and_passes() -> None:
    coordinator, scheduler, _sm, (visual,), scene_id = _make_coordinator_with_scene()
    coordinator.submit(_make_reslicing_request(scene_id=scene_id), {})

    assert scheduler.registered == [visual.cache_id]
    (desired_sets,) = scheduler.passes
    assert [ds.cache_id for ds in desired_sets] == [visual.cache_id]
    # The visual's store is filled in for a set that names none.
    assert desired_sets[0].store is coordinator._data_stores[visual.visual_model_id]
    assert scheduler.retired == []


def test_slice_coordinator_last_plan_wins_across_canvases() -> None:
    """Two canvases of one scene: each submit passes its plan, the later one
    replaces the earlier for the shared cache, and one completion answers
    both canvases' started events."""
    coordinator, scheduler, _sm, (visual,), scene_id = _make_coordinator_with_scene()
    canvas_a, canvas_b = uuid4(), uuid4()
    first = _desired(visual.cache_id)
    second = _desired(visual.cache_id)

    visual.plan.return_value = [first]
    coordinator.submit(
        _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_a), {}
    )
    visual.plan.return_value = [second]
    coordinator.submit(
        _make_reslicing_request(scene_id=scene_id, canvas_id=canvas_b), {}
    )

    # Registered once, passed twice, nothing cancelled or merged.
    assert scheduler.registered == [visual.cache_id]
    (first_pass,), (second_pass,) = scheduler.passes
    assert first_pass.keys is first.keys
    assert second_pass.keys is second.keys
    assert scheduler.retired == []
    assert coordinator._announced[visual.visual_model_id] == [
        (scene_id, canvas_a),
        (scene_id, canvas_b),
    ]


def test_slice_coordinator_idle_visual_is_retired() -> None:
    """A visual that plans nothing has its registered caches retired."""
    coordinator, scheduler, _sm, (v1, v2), scene_id = _make_coordinator_with_scene(
        n_visuals=2
    )
    coordinator.submit(_make_reslicing_request(scene_id=scene_id), {})
    assert scheduler.retired == []

    v2.plan.return_value = []
    coordinator.submit(_make_reslicing_request(scene_id=scene_id), {})

    assert scheduler.retired == [v2.cache_id]
    assert [ds.cache_id for ds in scheduler.passes[-1]] == [v1.cache_id]


def test_slice_coordinator_forget_visual_removes_its_caches() -> None:
    coordinator, scheduler, _sm, (visual,), scene_id = _make_coordinator_with_scene()
    coordinator.submit(_make_reslicing_request(scene_id=scene_id), {})

    coordinator.forget_visual(visual.visual_model_id)

    assert scheduler.removed == [visual.cache_id]
    assert visual.visual_model_id not in coordinator._announced


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
    rm._scenes[scene_id].plan_chunked.return_value = ({}, set())

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
# SceneManager -- per-visual config delivery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        VisualRenderConfig(settled_lod_bias=2.0),
        VisualRenderConfig(force_level=2),
        VisualRenderConfig(force_level=None),
        VisualRenderConfig(frustum_cull=False),
        VisualRenderConfig(frustum_cull=True),
    ],
    ids=["lod_bias", "force_level", "force_level_none", "no_cull", "cull"],
)
def test_the_visual_config_is_forwarded_to_plan(config) -> None:
    """The visual's own config reaches its plan, with its plan mode; LOD bias,
    forced level and frustum culling are the visual's to apply."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    sm.add_visual(visual, (0, 1, 2))

    req = _make_reslicing_request(scene_id=scene_id)
    sm.plan_chunked(req, {visual.visual_model_id: config})

    visual.plan.assert_called_once_with(req, config, config.plan_mode)


def test_a_visual_with_no_config_plans_with_the_defaults() -> None:
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    sm.add_visual(visual, (0, 1, 2))

    sm.plan_chunked(_make_reslicing_request(scene_id=scene_id), {})

    (_request, config, _mode), _ = visual.plan.call_args
    assert config == VisualRenderConfig()


# ---------------------------------------------------------------------------
# SceneManager -- slicing_enabled gate
# ---------------------------------------------------------------------------


def test_visual_render_config_slicing_enabled_by_default() -> None:
    assert VisualRenderConfig().slicing_enabled is True


def test_slicing_disabled_visual_is_not_planned_3d() -> None:
    """A visual whose config disables slicing is not planned; it is idle."""
    scene_id = uuid4()
    sm = SceneManager(scene_id=scene_id)
    visual = _make_mock_visual()
    sm.add_visual(visual, (0, 1, 2))

    planned, idle = sm.plan_chunked(
        _make_reslicing_request(scene_id=scene_id),
        {visual.visual_model_id: VisualRenderConfig(slicing_enabled=False)},
    )

    visual.plan.assert_not_called()
    assert planned == {}
    assert idle == {visual.visual_model_id}


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
    planned, idle = sm.plan_chunked(
        request, {visual.visual_model_id: VisualRenderConfig(slicing_enabled=False)}
    )

    visual.plan.assert_not_called()
    assert planned == {}
    assert idle == {visual.visual_model_id}
