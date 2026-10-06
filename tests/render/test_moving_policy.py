"""The moving policy: what a multiscale visual plans while it moves.

Plane rendering design v3, section 7.  A visual moves while its planes are
dragged (a ``plane_interaction`` scope) or its scene's dims are scrubbed.
With ``coarsest_while_moving_3d`` on (the 3D default) a plane drag plans
nothing and keeps the last plan, a dims tick plans the backstop, and the
target is planned once every motion has stopped.

The plans are recorded where every reslice path meets
(``SceneManager.plan_chunked``); what a plan wants is read from the
scheduler's registry.  Counts only, no clocks beyond the settle time.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cellier.render.scheduling import ChunkClass, Tier
from cellier.scene import spatial_axes
from cellier.visuals import (
    ClippingPlane,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests import _plane_fixtures as fx
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

EAGER = {"coarsest_while_moving_3d": False}
SETTLE_S = 0.03


def _record_plans(monkeypatch) -> list[tuple]:
    """Record ``(visual_id, plan mode name)`` for every visual a pass plans."""
    from cellier.render.scene_manager import SceneManager

    plans: list[tuple] = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        for visual_id, cfg in visual_configs.items():
            targets = request.target_visual_ids
            if targets is None or visual_id in targets:
                plans.append((visual_id, cfg.plan_mode.name))
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return plans


def _modes(plans: list[tuple], visual_id) -> list[str]:
    return [mode for vid, mode in plans if vid == visual_id]


class Rig:
    """A 3D scene with one multiscale visual of a small pyramid, one canvas."""

    def __init__(self, controller, tmp_path, *, kind="image", spec=fx.ANISO, **moving):
        self.controller = controller
        controller._render_manager.config.scheduler.dims_settle_s = SETTLE_S
        labels = kind == "labels"
        fx.write_pyramid(tmp_path, spec, labels=labels)
        self.store, _voxel = fx.open_pyramid(tmp_path, spec)
        self.system = self.store.data_coordinate_systems[0]
        if "t" in spec.axes:
            self.scene = controller.add_scene(
                coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
                dim="3d",
                name="scene",
            )
        else:
            self.scene = controller.add_scene(dim="3d", name="scene")
        self.depth = spec.shape0[-3]
        if labels:
            self.visual = controller.add_labels_multiscale(
                data=self.store,
                scene_id=self.scene.id,
                appearance=MultiscaleLabelsAppearance(force_level=1, **moving),
                render_config=MultiscaleLabelRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                clipping_planes=(self.plane(4.0),),
            )
        else:
            self.visual = controller.add_image_multiscale(
                data=self.store,
                scene_id=self.scene.id,
                appearance=MultiscaleImageAppearance(force_level=1, **moving),
                render_config=MultiscaleImageRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                single=MultiscaleImageSingleAppearance(
                    color_map="viridis", clim=(0.0, 65535.0), render_mode="mip"
                ),
                clipping_planes=(self.plane(4.0),),
            )
        controller.add_canvas(scene_id=self.scene.id, canvas_size=(200, 125))
        self.canvas_id = controller.get_canvas_ids(self.scene.id)[-1]
        self.item = self.visual.clipping_planes[0]
        self.events: list[tuple] = []
        controller.on_plane_interaction(
            self.visual.id,
            lambda event: self.events.append((event.phase, event.reason)),
            owner_id=controller._id,
        )

    def plane(self, z: float) -> ClippingPlane:
        """Keep data ``z >= z``."""
        return ClippingPlane.from_point_normal(
            self.system, (z, 0, 0), (1, 0, 0), axes=("z", "y", "x")
        )

    def move(self, z: float, **kwargs) -> None:
        self.controller.set_clipping_plane(
            self.visual.id, self.item.id, self.plane(z).plane, **kwargs
        )

    async def loaded(self) -> None:
        self.controller.fit_camera(self.scene.id)
        self.controller.reslice_all()
        await drain_loading(self.controller)

    @property
    def cache_id(self) -> int:
        gfx = self.controller._render_manager._scenes[self.scene.id].get_visual(
            self.visual.id
        )
        residency = (
            gfx.slots[0].residency_3d() if hasattr(gfx, "slots") else gfx.residency_3d()
        )
        return residency.cache_id

    def registry(self):
        return self.controller._render_manager.scheduler.core.registry(self.cache_id)

    def wanted(self) -> dict[int, frozenset[int]]:
        """Per chunk class, the keys the latest plan wants."""
        reg = self.registry()
        visible = reg.tier == Tier.VISIBLE
        return {
            int(cls): frozenset(int(k) for k in reg.key[visible & (reg.cls == cls)])
            for cls in (ChunkClass.BACKSTOP, ChunkClass.TARGET)
        }


@pytest.fixture
def make_rig(controller, tmp_path):
    def _make(**kwargs) -> Rig:
        return Rig(controller, tmp_path, **kwargs)

    return _make


# -- a plane drag holds the plan ------------------------------------------------


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_a_drag_plans_nothing_and_its_release_plans_once(
    make_rig, monkeypatch, kind
):
    rig = make_rig(kind=kind)
    await rig.loaded()
    before = rig.wanted()
    assert before[int(ChunkClass.TARGET)]
    plans = _record_plans(monkeypatch)

    with rig.controller.plane_interaction(rig.visual.id):
        for z in (8.0, 20.0, 36.0):
            rig.move(z)
        assert plans == []
        # The last plan is still the desired set: nothing was dropped.
        assert rig.wanted() == before
        assert rig.controller.plane_interaction_state(rig.visual.id) == "active"

    # The release planned before the block returned.
    assert _modes(plans, rig.visual.id) == ["FULL"]
    assert rig.events == [("start", None), ("end", "release")]
    await drain_loading(rig.controller)
    after = rig.wanted()
    assert after[int(ChunkClass.TARGET)] != before[int(ChunkClass.TARGET)]
    assert not rig.controller._plane_plan_owed


async def test_a_held_drag_always_has_the_backstop_to_draw(make_rig, monkeypatch):
    """What a drag reveals is drawn from the backstop: it is never clipped.

    The plane starts far in (little kept) and is dragged out, so most of what
    the drag shows was not in the held plan.
    """
    rig = make_rig()
    rig.move(48.0)
    await rig.loaded()
    backstop = rig.wanted()[int(ChunkClass.BACKSTOP)]
    assert backstop
    with rig.controller.plane_interaction(rig.visual.id):
        rig.move(2.0)
        assert rig.wanted()[int(ChunkClass.BACKSTOP)] == backstop
        progress = rig.controller._render_manager.scheduler.progress(rig.cache_id)
        assert progress.resident_backstop == progress.needed_backstop > 0
    await drain_loading(rig.controller)


async def test_stillness_ends_a_drag_and_plans_and_the_next_move_holds_again(
    make_rig, monkeypatch
):
    rig = make_rig()
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    source = uuid4()
    rig.controller.begin_plane_interaction(rig.visual.id, source_id=source)
    rig.move(8.0)
    assert plans == []
    await asyncio.sleep(SETTLE_S * 4)  # held still, the scope still open
    assert rig.events == [("start", None), ("end", "settle")]
    assert _modes(plans, rig.visual.id) == ["FULL"]

    rig.move(20.0)  # a new motion
    assert rig.events[-1] == ("start", None)
    assert _modes(plans, rig.visual.id) == ["FULL"]
    rig.controller.end_plane_interaction(rig.visual.id, source_id=source)
    assert rig.events[-1] == ("end", "release")
    assert _modes(plans, rig.visual.id) == ["FULL", "FULL"]
    await drain_loading(rig.controller)


async def test_a_change_outside_a_scope_is_a_jump_and_plans_at_once(
    make_rig, monkeypatch
):
    rig = make_rig()
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    rig.move(20.0)
    assert _modes(plans, rig.visual.id) == ["FULL"]
    assert rig.events == []
    await drain_loading(rig.controller)


async def test_eager_plans_every_tick_and_nothing_at_the_end(make_rig, monkeypatch):
    rig = make_rig(**EAGER)
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    with rig.controller.plane_interaction(rig.visual.id):
        for z in (8.0, 20.0, 36.0):
            rig.move(z)
        assert _modes(plans, rig.visual.id) == ["FULL"] * 3
    assert _modes(plans, rig.visual.id) == ["FULL"] * 3
    assert rig.events == [("start", None), ("end", "release")]
    await drain_loading(rig.controller)


async def test_a_scope_closed_by_an_exception_ends_the_drag_and_plans(
    make_rig, monkeypatch
):
    rig = make_rig()
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    with pytest.raises(RuntimeError):
        with rig.controller.plane_interaction(rig.visual.id):
            rig.move(20.0)
            raise RuntimeError("the script failed mid-drag")
    assert rig.events == [("start", None), ("end", "release")]
    assert _modes(plans, rig.visual.id) == ["FULL"]
    assert rig.controller.plane_interaction_state(rig.visual.id) == "idle"
    await drain_loading(rig.controller)


def test_with_no_event_loop_each_change_plans(make_rig, monkeypatch):
    """No loop, no timer: each change settles at once, and so plans."""
    rig = make_rig()
    plans = _record_plans(monkeypatch)
    with rig.controller.plane_interaction(rig.visual.id):
        rig.move(8.0)
        rig.move(20.0)
    assert _modes(plans, rig.visual.id) == ["FULL"] * 2
    assert not rig.controller._plane_plan_owed


async def test_the_setting_is_read_at_each_tick(make_rig, monkeypatch):
    """Assigning the setting plans nothing; a drag turned eager stays planned."""
    rig = make_rig()
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    with rig.controller.plane_interaction(rig.visual.id):
        rig.move(8.0)  # held
        rig.visual.appearance.coarsest_while_moving_3d = False
        assert plans == []
        rig.move(20.0)  # eager now
        assert _modes(plans, rig.visual.id) == ["FULL"]
    # The held tick was still owed its plan at the end.
    assert _modes(plans, rig.visual.id) == ["FULL", "FULL"]
    await drain_loading(rig.controller)


# -- jumps end a drag -------------------------------------------------------------


@pytest.mark.parametrize("change", ["visible", "render_mode"])
async def test_a_visibility_or_mode_change_ends_a_drag(make_rig, monkeypatch, change):
    rig = make_rig()
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    source = uuid4()
    rig.controller.begin_plane_interaction(rig.visual.id, source_id=source)
    rig.move(20.0)
    assert plans == []
    if change == "visible":
        rig.visual.appearance.visible = False
        rig.visual.appearance.visible = True
    else:
        rig.visual.single.render_mode = "iso"
    assert rig.events == [("start", None), ("end", "cancel")]
    assert rig.controller.plane_interaction_state(rig.visual.id) == "idle"
    # The plan the drag held back was made by the jump.
    assert "FULL" in _modes(plans, rig.visual.id)
    assert not rig.controller._plane_plan_owed
    n_plans = len(plans)
    rig.controller.end_plane_interaction(rig.visual.id, source_id=source)
    assert len(plans) == n_plans  # nothing more at the scope's end
    await drain_loading(rig.controller)


async def test_a_displayed_axes_change_ends_a_drag(make_rig, monkeypatch):
    rig = make_rig()
    await rig.loaded()
    source = uuid4()
    rig.controller.begin_plane_interaction(rig.visual.id, source_id=source)
    rig.move(20.0)
    plans = _record_plans(monkeypatch)
    rig.controller.set_displayed_axes(rig.scene.id, (1, 2))
    assert rig.events == [("start", None), ("end", "cancel")]
    assert not rig.controller._plane_plan_owed
    await drain_loading(rig.controller)
    assert set(_modes(plans, rig.visual.id)) <= {"FULL"}
    rig.controller.end_plane_interaction(rig.visual.id, source_id=source)
    assert rig.events == [("start", None), ("end", "cancel")]


async def test_removing_a_visual_mid_drag_forgets_what_it_was_owed(make_rig):
    rig = make_rig()
    await rig.loaded()
    rig.controller.begin_plane_interaction(rig.visual.id, source_id=uuid4())
    rig.move(20.0)
    assert rig.visual.id in rig.controller._plane_plan_owed
    rig.controller.remove_visual(rig.visual.id)
    assert not rig.controller._plane_plan_owed
    assert rig.controller._plane_interaction_driver.tasks() == []


# -- C3: a camera end during a drag ----------------------------------------------


def _moved(state, step: float):
    return state._replace(position=tuple(p + step for p in state.position))


async def test_a_camera_end_during_a_drag_leaves_the_target_to_the_drag(
    make_rig, monkeypatch
):
    rig = make_rig()
    controller = rig.controller
    await rig.loaded()
    controller.camera_reslice_enabled = True  # the fixture turns it off
    plans = _record_plans(monkeypatch)
    state = controller.get_camera_state(rig.canvas_id)

    with controller.plane_interaction(rig.visual.id):
        rig.move(8.0)
        with controller.camera_interaction(rig.canvas_id):
            for step in (1.0, 2.0, 3.0):
                controller.set_camera_state(rig.canvas_id, _moved(state, step))
        # The orbit was released: its end planned nothing of this visual.
        assert controller.camera_interaction_state(rig.canvas_id) == "idle"
        assert plans == []
        rig.move(20.0)
        assert plans == []
    assert _modes(plans, rig.visual.id) == ["FULL"]
    await drain_loading(controller)


async def test_a_drag_that_ends_while_the_camera_moves_is_planned_by_the_camera_end(
    make_rig, monkeypatch
):
    rig = make_rig()
    controller = rig.controller
    await rig.loaded()
    controller.camera_reslice_enabled = True  # the fixture turns it off
    plans = _record_plans(monkeypatch)
    state = controller.get_camera_state(rig.canvas_id)

    with controller.camera_interaction(rig.canvas_id):
        controller.set_camera_state(rig.canvas_id, _moved(state, 1.0))
        with controller.plane_interaction(rig.visual.id):
            rig.move(8.0)
        # The drag ended first: no target from a camera still moving.
        assert rig.events == [("start", None), ("end", "release")]
        assert plans == []
    assert _modes(plans, rig.visual.id) == ["FULL"]
    await drain_loading(controller)


async def test_with_camera_reslicing_off_a_drag_end_plans_whatever_the_camera_does(
    make_rig, monkeypatch
):
    rig = make_rig()
    controller = rig.controller
    await rig.loaded()
    assert controller.camera_reslice_enabled is False
    plans = _record_plans(monkeypatch)
    state = controller.get_camera_state(rig.canvas_id)
    with controller.camera_interaction(rig.canvas_id):
        controller.set_camera_state(rig.canvas_id, _moved(state, 1.0))
        with controller.plane_interaction(rig.visual.id):
            rig.move(8.0)
        # No camera end will plan it, so the drag's end does.
        assert _modes(plans, rig.visual.id) == ["FULL"]
    await drain_loading(controller)


# -- C4: a plane drag and a dims scrub together ---------------------------------


async def test_a_drag_and_a_scrub_together_plan_the_target_once_both_have_stopped(
    make_rig, monkeypatch
):
    rig = make_rig(spec=fx.TZYX)
    controller = rig.controller
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    scrub = uuid4()

    with controller.plane_interaction(rig.visual.id):
        rig.move(8.0)
        assert plans == []
        controller.begin_dims_interaction(rig.scene.id, source_id=scrub)
        controller.update_slice_indices(rig.scene.id, {0: 3.0})
        # The new slice's backstop: the held plan belongs to another slice.
        assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY"]
        rig.move(12.0)
        assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY"]
        # The scrub ends first: the target waits for the drag.
        controller.end_dims_interaction(rig.scene.id, source_id=scrub)
        assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY"]
    assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY", "FULL"]
    await drain_loading(controller)


async def test_a_drag_that_ends_during_a_scrub_is_planned_by_the_scrub_end(
    make_rig, monkeypatch
):
    rig = make_rig(spec=fx.TZYX)
    controller = rig.controller
    await rig.loaded()
    plans = _record_plans(monkeypatch)

    with controller.dims_interaction(rig.scene.id):
        controller.update_slice_indices(rig.scene.id, {0: 3.0})
        with controller.plane_interaction(rig.visual.id):
            rig.move(8.0)
        # The drag ended first: no target while the dims still move.
        assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY"]
    assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY", "FULL"]
    await drain_loading(controller)


# -- the 3D dims scrub ------------------------------------------------------------


async def test_a_3d_scrub_plans_the_backstop_per_tick_by_default(make_rig, monkeypatch):
    rig = make_rig(spec=fx.TZYX)
    controller = rig.controller
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    with controller.dims_interaction(rig.scene.id):
        for t in (1.0, 2.0, 3.0):
            controller.update_slice_indices(rig.scene.id, {0: t})
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert rig.wanted()[int(ChunkClass.TARGET)] == frozenset()
        assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY"] * 3
    assert _modes(plans, rig.visual.id) == ["BACKSTOP_ONLY"] * 3 + ["FULL"]
    await drain_loading(controller)
    assert rig.wanted()[int(ChunkClass.TARGET)]


async def test_an_eager_3d_scrub_plans_in_full_per_tick(make_rig, monkeypatch):
    rig = make_rig(spec=fx.TZYX, **EAGER)
    controller = rig.controller
    await rig.loaded()
    plans = _record_plans(monkeypatch)
    with controller.dims_interaction(rig.scene.id):
        for t in (1.0, 2.0):
            controller.update_slice_indices(rig.scene.id, {0: t})
    assert _modes(plans, rig.visual.id) == ["FULL"] * 2
    await drain_loading(controller)


# -- other visuals are untouched --------------------------------------------------


async def test_a_2d_view_plans_a_clipping_change_on_every_tick(
    controller, tmp_path, monkeypatch
):
    fx.write_pyramid(tmp_path, fx.ISO)
    store, _ = fx.open_pyramid(tmp_path, fx.ISO)
    scene = controller.add_scene(dim="2d", name="scene")
    system = store.data_coordinate_systems[0]

    def plane(x: float) -> ClippingPlane:
        return ClippingPlane.from_point_normal(
            system, (0, 0, x), (0, 0, 1), axes=("z", "y", "x")
        )

    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(coarsest_while_moving_2d=True),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        clipping_planes=(plane(8.0),),
    )
    controller.add_canvas(scene_id=scene.id)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    affects = controller._render_manager.clipping_planes_affect_request(visual.id)
    reslices: list = []
    original = controller.reslice_visual
    monkeypatch.setattr(
        controller,
        "reslice_visual",
        lambda vid: (reslices.append(vid), original(vid))[1],
    )
    with controller.plane_interaction(visual.id):
        controller.set_clipping_planes(visual.id, (plane(20.0),))
        # Not held: a 2D view's only motion is a dims scrub.
        assert reslices == ([visual.id] if affects else [])
    assert not controller._plane_plan_owed
    await drain_loading(controller)
