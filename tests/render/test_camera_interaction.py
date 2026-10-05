"""Camera motion: the per-canvas interaction tracker (design 4.5, 4.6).

Headless: an offscreen canvas, synthetic input through
``canvas.submit_event``, and a fake clock in pygfx's controllers so a drag's
damped tail takes the same frames every run.  A frame is drawn only when one
was requested, as an on-demand canvas does.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

import pygfx.controllers._base as pygfx_controller_base
import pytest

from cellier.controller import CellierController
from cellier.render import CameraConfig, RenderManagerConfig
from cellier.visuals import MultiscaleImageSingleAppearance, ProgressiveLoadingConfig
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

if TYPE_CHECKING:
    from cellier.events import CameraChangedEvent

DT = 1 / 60
#: Long enough that no test's motion settles unless it means to.
NO_SETTLE = 30.0


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Rig:
    """One controller, one scene with a multiscale image, offscreen canvases."""

    def __init__(
        self,
        store,
        clock: _Clock,
        *,
        dim: str = "3d",
        settle_s: float = NO_SETTLE,
        loading: ProgressiveLoadingConfig | None = None,
        n_canvases: int = 1,
        render_modes: set[str] | None = None,
        with_image: bool = True,
    ) -> None:
        self.clock = clock
        self.controller = CellierController(
            gui="offscreen",
            render_config=RenderManagerConfig(
                camera=CameraConfig(settle_threshold_s=settle_s)
            ),
        )
        kwargs = {} if render_modes is None else {"render_modes": render_modes}
        self.scene = self.controller.add_scene(dim=dim, name="scene", **kwargs)
        self.store = store
        self.visual = self.add_image(loading) if with_image else None
        self.canvas_ids = []
        self.views = []
        for _ in range(n_canvases):
            self.controller.add_canvas(self.scene.id, canvas_size=(200, 150))
            canvas_id = self.controller.get_canvas_ids(self.scene.id)[-1]
            self.canvas_ids.append(canvas_id)
            self.views.append(self.controller.get_canvas_view(canvas_id))
        self.canvas_id = self.canvas_ids[0]
        self.view = self.views[0]

        self.requested = [False] * n_canvases
        self.frames = [0] * n_canvases
        for index, view in enumerate(self.views):
            self._track_requests(index, view)

        # What the tests assert on.
        self.reslices: list[dict] = []
        self.events: list[tuple] = []
        self.camera_events: list[CameraChangedEvent] = []
        render_manager = self.controller._render_manager
        original = render_manager.reslice_scene

        def reslice_scene(scene_id, dims_state, visual_configs=None, **kwargs):
            self.reslices.append(
                {
                    "targets": kwargs.get("target_visual_ids"),
                    "modes": {
                        vid: cfg.plan_mode.name
                        for vid, cfg in (visual_configs or {}).items()
                    },
                    "in_draw": self.controller._drawing(),
                    "task": _current_task(),
                    "camera": self.view.capture_camera_state(),
                }
            )
            return original(scene_id, dims_state, visual_configs, **kwargs)

        render_manager.reslice_scene = reslice_scene
        self.controller.on_camera_interaction(
            self.scene.id,
            lambda event: self.events.append(
                (self.canvas_ids.index(event.canvas_id), event.phase, event.reason)
            ),
            owner_id=self.controller._id,
        )
        self.controller.on_camera_changed(
            self.scene.id, self.camera_events.append, owner_id=self.controller._id
        )

    def add_image(self, loading: ProgressiveLoadingConfig | None = None):
        return self.controller.add_image_multiscale(
            data=self.store,
            scene_id=self.scene.id,
            appearance=MultiscaleImageAppearance(force_level=1),
            render_config=MultiscaleImageRenderConfig(
                **SMALL_BUDGETS,
                block_size=8,
                loading=loading or ProgressiveLoadingConfig(),
            ),
            single=MultiscaleImageSingleAppearance(
                color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
            ),
        )

    def _track_requests(self, index: int, view) -> None:
        canvas = view.widget
        original = canvas.request_draw

        def request_draw(draw_function=None):
            self.requested[index] = True
            return original(draw_function)

        canvas.request_draw = request_draw

    # -- input and frames -------------------------------------------------------

    def submit(self, event_type: str, *, canvas: int = 0, **fields) -> None:
        event = {
            "event_type": event_type,
            "x": 100.0,
            "y": 75.0,
            "button": 0,
            "buttons": (),
            "modifiers": (),
            "ntouches": 0,
            "touches": {},
        }
        event.update(fields)
        self.views[canvas].widget.submit_event(event)

    def step(self, *, draw: bool = True) -> list[bool]:
        """One vsync: advance the clock, deliver input, draw what asked to."""
        self.clock.now += DT
        drawn = []
        for index, view in enumerate(self.views):
            view.widget._process_events()
            if draw and self.requested[index]:
                self.requested[index] = False
                view.widget.draw()
                self.frames[index] += 1
                drawn.append(True)
            else:
                drawn.append(False)
        return drawn

    async def run(self, n: int, **kwargs) -> None:
        for _ in range(n):
            self.step(**kwargs)
            await asyncio.sleep(0)

    async def run_until_idle(self, limit: int = 400) -> None:
        idle = 0
        for _ in range(limit):
            if any(self.step()):
                idle = 0
            else:
                idle += 1
                if idle >= 5:
                    break
            await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def settle_first_frames(self) -> None:
        """Fit, load and draw until quiet, then forget what that recorded."""
        self.controller.fit_camera(self.scene.id)
        self.controller.reslice_all()
        await drain_loading(self.controller)
        for index in range(len(self.views)):
            self.requested[index] = True
        await self.run_until_idle()
        await drain_loading(self.controller)
        await self.run_until_idle()
        self.clear()

    def clear(self) -> None:
        del self.reslices[:]
        del self.events[:]
        del self.camera_events[:]

    # -- gestures ---------------------------------------------------------------

    async def press(self, *, canvas: int = 0, button: int = 1) -> None:
        self.submit("pointer_down", canvas=canvas, button=button, buttons=(button,))
        await self.run(1)

    async def move(self, moves: int = 5, *, canvas: int = 0, start: float = 100.0):
        x = start
        for _ in range(moves):
            x += 6.0
            self.submit("pointer_move", canvas=canvas, x=x, buttons=(1,))
            await self.run(1)
        return x

    def press_and_move_now(
        self, moves: int = 3, *, canvas: int = 0, tail: bool = False
    ) -> float:
        """Press and move without yielding to the event loop.

        For tests with a short real-time settle threshold.  The stillness
        timer is a task on the loop, so it cannot fire between two of these
        frames however long they take; with ``await``-ed frames a slow
        machine settles the motion mid-gesture.  With *tail* the damped
        motion is drawn to its end too.  The timer runs from the last
        camera change once the caller awaits.
        """
        self.submit("pointer_down", canvas=canvas, button=1, buttons=(1,))
        self.step()
        x = 100.0
        for _ in range(moves):
            x += 6.0
            self.submit("pointer_move", canvas=canvas, x=x, buttons=(1,))
            self.step()
        idle = 0
        for _ in range(400 if tail else 0):
            idle = 0 if any(self.step()) else idle + 1
            if idle >= 5:
                break
        return x

    async def release(self, x: float, *, canvas: int = 0) -> None:
        self.submit("pointer_up", canvas=canvas, x=x, button=1, buttons=())
        await self.run_until_idle()

    async def drag(self, moves: int = 5, *, canvas: int = 0) -> None:
        await self.press(canvas=canvas)
        x = await self.move(moves, canvas=canvas)
        await self.release(x, canvas=canvas)

    def camera_reslices(self) -> list[dict]:
        """Reslices aimed at the camera-sensitive visuals only."""
        return [r for r in self.reslices if r["targets"] is not None]


def _current_task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(pygfx_controller_base, "perf_counter", clock)
    return clock


@pytest.fixture
def make_rig(multiscale_image_store, clock):
    rigs: list[Rig] = []

    def _make(**kwargs) -> Rig:
        rig = Rig(multiscale_image_store, clock, **kwargs)
        rigs.append(rig)
        return rig

    yield _make
    for rig in rigs:
        rig.controller.close()


# -- a drag ---------------------------------------------------------------------


@pytest.mark.parametrize("dim", ["3d", "2d"])
async def test_a_drag_starts_in_the_first_moved_frame_and_ends_on_release(
    make_rig, dim
):
    rig = make_rig(dim=dim)
    await rig.settle_first_frames()
    controller = rig.controller
    before = rig.view.capture_camera_state()

    await rig.press()
    assert rig.events == []  # a press moves nothing yet
    assert controller.camera_interaction_state(rig.canvas_id) == "idle"

    rig.submit("pointer_move", x=106.0, buttons=(1,))
    await rig.run(1)
    # The frame that drew the moved camera is the one that reported it.
    assert rig.view.capture_camera_state() != before
    assert rig.events == [(0, "start", None)]
    assert len(rig.camera_events) == 1 and rig.camera_events[0].interactive
    assert rig.view.camera_moving is True
    assert controller.camera_interaction_state(rig.canvas_id) == "active"
    assert rig.reslices == []  # nothing is resliced while moving

    x = await rig.move(4, start=106.0)
    assert rig.reslices == []
    await rig.release(x)

    assert rig.events == [(0, "start", None), (0, "end", "release")]
    assert rig.view.camera_moving is False
    assert controller.camera_interaction_state(rig.canvas_id) == "idle"
    # One reslice, of the camera-sensitive visual, for the final camera, and
    # planned from the loop: never inside the draw that saw the release.
    (reslice,) = rig.reslices
    assert reslice["targets"] == frozenset({rig.visual.id})
    assert reslice["modes"][rig.visual.id] == "FULL"
    assert reslice["in_draw"] is False
    assert reslice["camera"] == rig.view.capture_camera_state()
    assert not controller._deferred_reslice_tasks()
    await drain_loading(controller)


async def test_the_motion_ends_one_frame_after_the_camera_stops(make_rig):
    """The scope covers the damped tail, not just the button (design C3)."""
    rig = make_rig()
    await rig.settle_first_frames()
    await rig.press()
    x = await rig.move(5)
    rig.submit("pointer_up", x=x, button=1, buttons=())

    states = []
    ended_at = None
    for frame in range(400):
        rig.step()
        await asyncio.sleep(0)
        states.append(rig.view.capture_camera_state())
        if rig.events and rig.events[-1][1] == "end":
            ended_at = frame
            break
    assert ended_at is not None
    changes = [i for i in range(1, len(states)) if states[i] != states[i - 1]]
    # The camera kept moving after the button went up (the tail) ...
    assert changes and changes[-1] > 1
    # ... and the end came in the first frame with no change.
    assert ended_at == changes[-1] + 1
    assert rig.events[-1] == (0, "end", "release")
    await rig.run_until_idle()


async def test_accumulation_resets_in_the_first_moved_frame(make_rig, monkeypatch):
    rig = make_rig()
    await rig.settle_first_frames()
    resets: list[int] = []
    accum = rig.view._accum_pass
    original = accum.reset
    monkeypatch.setattr(
        accum, "reset", lambda: (resets.append(rig.frames[0]), original())[1]
    )
    await rig.press()
    assert resets == []
    frame = rig.frames[0]
    rig.submit("pointer_move", x=106.0, buttons=(1,))
    await rig.run(1)
    # Reset while drawing the frame that shows the new camera (the counter
    # is bumped after the draw returns).
    assert resets == [frame]
    await rig.release(106.0)


async def test_a_click_that_does_not_move_the_camera_emits_nothing(make_rig):
    rig = make_rig()
    await rig.settle_first_frames()
    await rig.press()
    assert rig.view._driving is True  # the controller holds an action
    await rig.release(100.0)
    assert rig.view._driving is False
    assert rig.events == []
    assert rig.camera_events == []
    assert rig.reslices == []
    assert not rig.controller._deferred_reslice_tasks()


async def test_a_hover_draws_nothing(make_rig):
    rig = make_rig()
    await rig.settle_first_frames()
    frames = rig.frames[0]
    for x in (103.0, 106.0, 109.0):
        rig.submit("pointer_move", x=x, buttons=())
        await rig.run(1)
    assert rig.frames[0] == frames
    assert rig.events == []


async def test_a_wheel_zoom_is_a_motion_that_ends_on_release(make_rig):
    rig = make_rig()
    await rig.settle_first_frames()
    for _ in range(3):
        rig.submit("wheel", dx=0.0, dy=100.0)
        await rig.run(1)
    assert rig.events == [(0, "start", None)]
    await rig.run_until_idle()
    assert rig.events == [(0, "start", None), (0, "end", "release")]
    assert len(rig.reslices) == 1
    await drain_loading(rig.controller)


# -- holding still ---------------------------------------------------------------


async def test_a_drag_held_still_settles_and_moving_again_starts_again(make_rig):
    rig = make_rig(settle_s=0.05)
    await rig.settle_first_frames()
    # Held still, the button down: the damped motion finishes, then stillness.
    x = rig.press_and_move_now(3, tail=True)
    assert rig.events == [(0, "start", None)]

    for _ in range(400):
        await rig.run(1)
        if rig.events[-1][1] == "end":
            break
        await asyncio.sleep(0.005)
    assert rig.events == [(0, "start", None), (0, "end", "settle")]
    assert rig.view._driving is True  # the scope is still open
    assert rig.view.camera_moving is False
    await asyncio.sleep(0)
    assert len(rig.reslices) == 1

    x = await rig.move(2, start=x)
    assert rig.events[-1] == (0, "start", None)
    assert rig.view.camera_moving is True
    rig.controller._render_manager.config.camera.settle_threshold_s = NO_SETTLE
    await rig.move(1, start=x)  # re-arm with the long settle time
    await rig.release(x + 6.0)
    assert rig.events[-1] == (0, "end", "release")
    assert len(rig.reslices) == 2
    await drain_loading(rig.controller)


async def test_a_settle_end_on_a_canvas_that_is_not_drawing_requests_a_draw(make_rig):
    rig = make_rig(settle_s=0.05)
    await rig.settle_first_frames()
    rig.press_and_move_now(3)
    # The canvas stops drawing (a hidden output): requests go unanswered.
    for _ in range(3):
        rig.step(draw=False)
    rig.requested[0] = False
    # Wait on the stillness timer itself: a count of short sleeps is not a
    # length of time on a loop with a coarse clock (Windows, Python 3.12).
    timers = rig.controller._camera_driver.tasks()
    assert len(timers) == 1
    await asyncio.wait_for(asyncio.gather(*timers), timeout=30.0)
    assert rig.events == [(0, "start", None), (0, "end", "settle")]
    # Without the request a visual with no camera reslice would keep its
    # moving picture until the next input.
    assert rig.requested[0] is True
    assert rig.view.camera_moving is False
    # Nothing saw ``driving`` fall, so the scope is still open (bounded: the
    # timer ended the motion).
    assert rig.controller._camera_driver.scope_open(rig.canvas_id)
    await drain_loading(rig.controller)


async def test_a_pending_camera_end_is_a_deferred_reslice_task(make_rig):
    rig = make_rig(settle_s=0.05)
    await rig.settle_first_frames()
    rig.press_and_move_now(3)
    (task,) = rig.controller._deferred_reslice_tasks()
    assert not task.done()
    # A drain waits for it instead of calling the scene idle.
    await drain_loading(rig.controller)
    assert rig.events[-1] == (0, "end", "settle")
    assert len(rig.reslices) == 1


# -- programmatic moves -----------------------------------------------------------


async def test_fit_camera_is_a_jump(make_rig):
    rig = make_rig()
    controller = rig.controller
    await rig.settle_first_frames()
    state = controller.get_camera_state(rig.canvas_id)
    moved = state._replace(position=tuple(p + 5.0 for p in state.position))
    controller.set_camera_state(rig.canvas_id, moved)
    rig.clear()

    controller.fit_camera(rig.scene.id)
    # At once, in this call: no motion, no timer, nothing queued.
    (reslice,) = rig.reslices
    assert reslice["targets"] == frozenset({rig.visual.id})
    assert reslice["camera"] == rig.view.capture_camera_state()
    assert rig.events == []
    assert controller.camera_interaction_state(rig.canvas_id) == "idle"
    assert not controller._deferred_reslice_tasks()
    (event,) = rig.camera_events
    assert event.interactive is False
    assert event.camera_state == rig.view.capture_camera_state()

    # The next frame sees no difference: the jump is not also a motion.
    rig.requested[0] = True
    await rig.run_until_idle()
    assert rig.events == []
    assert len(rig.reslices) == 1
    await drain_loading(controller)


async def test_fit_camera_on_a_fitted_canvas_does_nothing(make_rig):
    rig = make_rig()
    await rig.settle_first_frames()
    rig.controller.fit_camera(rig.scene.id)
    assert rig.reslices == []
    assert rig.events == []
    assert rig.camera_events == []


async def test_look_at_visual_and_the_depth_range_are_jumps(make_rig):
    rig = make_rig()
    controller = rig.controller
    await rig.settle_first_frames()
    controller.look_at_visual(rig.visual.id, rig.canvas_id, view_direction=(0, 0, -1))
    assert len(rig.reslices) == 1
    controller.set_camera_depth_range(rig.canvas_id, (2.0, 500.0))
    assert len(rig.reslices) == 2
    assert rig.events == []
    assert [event.interactive for event in rig.camera_events] == [False, False]
    canvas_model = rig.scene.canvases[rig.canvas_id]
    assert canvas_model.cameras["3d"].near_clipping_plane == 2.0
    await drain_loading(controller)


async def test_a_jump_ends_a_motion(make_rig):
    rig = make_rig()
    controller = rig.controller
    await rig.settle_first_frames()
    state = controller.get_camera_state(rig.canvas_id)

    controller.set_camera_state(rig.canvas_id, state, interactive=True)  # unchanged
    assert rig.events == []
    assert rig.camera_events == []
    moved = state._replace(position=tuple(p + 5.0 for p in state.position))
    controller.set_camera_state(rig.canvas_id, moved, interactive=True)
    assert rig.events == [(0, "start", None)]
    assert rig.view.camera_moving is True
    assert rig.reslices == []

    controller.fit_camera(rig.scene.id)  # a script moves the camera: a jump
    assert rig.events == [(0, "start", None), (0, "end", "jump")]
    assert len(rig.reslices) == 1  # the jump's own, not one more for the end
    assert rig.view.camera_moving is False
    assert not controller._camera_driver.tasks()
    assert not controller._deferred_reslice_tasks()
    await drain_loading(controller)


async def test_inside_a_scope_a_programmatic_move_is_a_motion_tick(make_rig):
    rig = make_rig()
    controller = rig.controller
    await rig.settle_first_frames()
    state = controller.get_camera_state(rig.canvas_id)

    with controller.camera_interaction(rig.canvas_id):
        for step in range(1, 4):
            pose = state._replace(position=tuple(p + step for p in state.position))
            controller.set_camera_state(rig.canvas_id, pose)
        assert rig.events == [(0, "start", None)]
        assert all(event.interactive for event in rig.camera_events)
        assert rig.reslices == []  # a fly-through reslices once, at its end
        controller.fit_camera(rig.scene.id)  # in scope: a tick, not a jump
        assert rig.events == [(0, "start", None)]
        assert rig.reslices == []

    assert rig.events == [(0, "start", None), (0, "end", "release")]
    assert len(rig.reslices) == 1
    assert rig.reslices[0]["in_draw"] is False
    assert not controller._deferred_reslice_tasks()
    await drain_loading(controller)


async def test_a_scope_on_an_unknown_canvas_is_refused(make_rig):
    rig = make_rig()
    with pytest.raises(KeyError):
        rig.controller.begin_camera_interaction(uuid4(), source_id=uuid4())


async def test_a_camera_state_of_the_wrong_kind_is_refused(make_rig):
    rig = make_rig(dim="2d")
    await rig.settle_first_frames()
    perspective = rig.controller.get_camera_state(rig.canvas_id)._replace(
        camera_type="perspective"
    )
    with pytest.raises(ValueError, match="orthographic"):
        rig.controller.set_camera_state(rig.canvas_id, perspective)


async def test_set_camera_state_round_trips(make_rig):
    rig = make_rig(dim="2d")
    controller = rig.controller
    await rig.settle_first_frames()
    state = controller.get_camera_state(rig.canvas_id)
    target = state._replace(
        position=(3.0, 4.0, state.position[2]), extent=(20.0, 10.0), zoom=2.0
    )
    controller.set_camera_state(rig.canvas_id, target)
    assert controller.get_camera_state(rig.canvas_id) == target
    camera_model = rig.scene.canvases[rig.canvas_id].cameras["2d"]
    assert (camera_model.width, camera_model.height) == (20.0, 10.0)
    await drain_loading(controller)


async def test_with_camera_reslicing_off_there_are_events_and_no_reslice(make_rig):
    rig = make_rig()
    controller = rig.controller
    await rig.settle_first_frames()
    controller.camera_reslice_enabled = False
    await rig.drag()
    assert rig.events == [(0, "start", None), (0, "end", "release")]
    state = controller.get_camera_state(rig.canvas_id)
    controller.set_camera_state(rig.canvas_id, state._replace(position=(1.0, 2.0, 3.0)))
    assert len(rig.camera_events) > 1
    assert rig.reslices == []
    assert not controller._deferred_reslice_tasks()


async def test_on_ready_fires_once_when_a_jump_follows_the_reslice(make_rig):
    rig = make_rig()
    controller = rig.controller
    await rig.settle_first_frames()
    state = controller.get_camera_state(rig.canvas_id)
    fired: list[int] = []
    controller.reslice_scene(rig.scene.id, on_ready=lambda: fired.append(1))
    controller.set_camera_state(
        rig.canvas_id, state._replace(position=tuple(p + 5 for p in state.position))
    )
    await drain_loading(controller)
    await rig.run_until_idle()
    await drain_loading(controller)
    assert fired == [1]


# -- paint mode -----------------------------------------------------------------


async def test_a_disabled_controller_opens_no_scope_and_ticks_nothing(make_rig):
    """Paint mode disables the camera controller for the session."""
    rig = make_rig()
    await rig.settle_first_frames()
    rig.view.set_controller_enabled(False)
    frames = rig.frames[0]
    await rig.press()
    x = await rig.move(3)
    await rig.release(x)
    assert rig.events == []
    assert rig.camera_events == []
    assert rig.view._driving is False
    assert rig.frames[0] == frames  # and its input requests no draw


async def test_disabling_the_controller_mid_drag_ends_the_motion(make_rig):
    rig = make_rig()
    await rig.settle_first_frames()
    await rig.press()
    await rig.move(3)
    rig.view.set_controller_enabled(False)
    rig.requested[0] = True
    await rig.run_until_idle()
    assert rig.events == [(0, "start", None), (0, "end", "release")]
    assert rig.view._driving is False
    await drain_loading(rig.controller)


# -- the 2D/3D switch -----------------------------------------------------------


async def test_the_2d_3d_switch_cancels_and_its_reslice_sees_the_fitted_view(
    make_rig,
):
    rig = make_rig(render_modes={"2d", "3d"})
    controller = rig.controller
    await rig.settle_first_frames()
    await rig.press()
    await rig.move(3)
    assert rig.events == [(0, "start", None)]
    rig.clear()

    controller.set_displayed_axes(rig.scene.id, (1, 2))
    assert rig.events == [(0, "end", "cancel")]
    assert rig.view.camera_moving is False
    # The first visit fitted the 2D camera, and was reported, not as motion.
    (event,) = rig.camera_events
    assert event.interactive is False
    assert event.camera_state.camera_type == "orthographic"
    # One reslice: the displayed-axes change's own, of the whole scene, made
    # after the fit.  No camera reslice follows.
    (reslice,) = rig.reslices
    assert reslice["targets"] is None
    assert reslice["camera"] == rig.view.capture_camera_state()
    assert not controller._deferred_reslice_tasks()

    rig.requested[0] = True
    await rig.run_until_idle()
    await drain_loading(controller)
    await rig.run_until_idle()
    assert rig.camera_reslices() == []
    assert rig.events == [(0, "end", "cancel")]


async def test_the_deferred_fit_is_a_jump_queued_on_the_loop(make_rig):
    """A scene empty at the toggle is fitted when it first has something to fit.

    That happens on the way to a draw (a visual added, a commit round), where
    planning must not run: the fit is a jump whose reslice is queued.
    """
    rig = make_rig(render_modes={"2d", "3d"}, with_image=False)
    controller = rig.controller
    controller.set_displayed_axes(rig.scene.id, (1, 2))
    assert rig.canvas_id in controller._canvases_awaiting_fit
    before = rig.view.capture_camera_state()
    rig.clear()

    rig.visual = rig.add_image()
    assert rig.canvas_id not in controller._canvases_awaiting_fit
    assert rig.view.capture_camera_state() != before
    # Reported as a jump, not seen as motion on the next frame ...
    (event,) = rig.camera_events
    assert event.interactive is False
    assert rig.events == []
    assert not controller._camera_driver.tasks()
    # ... and its reslice is queued: one task, nothing planned yet.
    assert rig.reslices == []
    (task,) = controller._deferred_reslice_tasks()

    await drain_loading(controller)
    rig.requested[0] = True
    await rig.run_until_idle()
    await drain_loading(controller)
    (reslice,) = rig.camera_reslices()  # once
    assert reslice["task"] is task
    assert reslice["in_draw"] is False
    assert reslice["camera"] == rig.view.capture_camera_state()
    assert rig.events == []


# -- several canvases -----------------------------------------------------------


async def test_moving_one_canvas_leaves_the_other_idle(make_rig, monkeypatch):
    rig = make_rig(n_canvases=2)
    controller = rig.controller
    await rig.settle_first_frames()
    submitted: list = []
    coordinator = controller._render_manager._slice_coordinator
    original = coordinator.submit

    def submit(request, visual_configs):
        submitted.append(request.canvas_id)
        return original(request, visual_configs)

    monkeypatch.setattr(coordinator, "submit", submit)

    await rig.press(canvas=1)
    x = await rig.move(3, canvas=1)
    assert rig.events == [(1, "start", None)]
    assert controller.camera_interaction_state(rig.canvas_ids[0]) == "idle"
    assert controller.camera_interaction_state(rig.canvas_ids[1]) == "active"
    assert rig.views[0].camera_moving is False
    assert rig.views[1].camera_moving is True
    await rig.release(x, canvas=1)

    assert rig.events == [(1, "start", None), (1, "end", "release")]
    # The end reslices the scene: one request per canvas, each from its own
    # camera (residency is per visual, so one canvas alone would be wrong).
    assert len(rig.reslices) == 1
    assert sorted(submitted, key=str) == sorted(rig.canvas_ids, key=str)
    await drain_loading(controller)


# -- camera and dims together -----------------------------------------------------

DRAG = ProgressiveLoadingConfig(dims_drag="backstop")


@pytest.mark.parametrize("how", ["motion_end", "jump"])
async def test_a_camera_reslice_during_a_scrub_plans_coarse(make_rig, how):
    rig = make_rig(dim="2d", loading=DRAG)
    controller = rig.controller
    await rig.settle_first_frames()
    visual_id = rig.visual.id
    # The drag below draws frames in real time.  The dims stillness timer
    # is real time too, and must not end the scrub under it: a scrub that
    # settles mid-drag hands the visual to the camera's end, which plans it
    # in full (seen on a slow CI runner).
    controller._render_manager.config.scheduler.dims_settle_s = NO_SETTLE

    with controller.dims_interaction(rig.scene.id):
        controller.update_slice_indices(rig.scene.id, {0: 6.0})
        rig.clear()
        if how == "motion_end":
            await rig.drag(3)
        else:
            state = controller.get_camera_state(rig.canvas_id)
            controller.set_camera_state(
                rig.canvas_id, state._replace(zoom=state.zoom * 2)
            )
        (reslice,) = rig.camera_reslices()
        assert reslice["modes"][visual_id] == "BACKSTOP_ONLY"
        assert visual_id in controller._dims_scrub_pending[rig.scene.id]
        rig.clear()

    # The scrub's end plans it in full, once.
    (reslice,) = rig.reslices
    assert reslice["modes"][visual_id] == "FULL"
    assert reslice["targets"] == frozenset({visual_id})
    await drain_loading(controller)


async def test_a_scrub_ending_during_camera_motion_leaves_the_target_to_the_camera(
    make_rig,
):
    rig = make_rig(dim="2d", loading=DRAG)
    controller = rig.controller
    await rig.settle_first_frames()
    visual_id = rig.visual.id

    await rig.press()
    x = await rig.move(3)
    with controller.dims_interaction(rig.scene.id):  # a player, during the drag
        controller.update_slice_indices(rig.scene.id, {0: 6.0})
        assert rig.reslices[-1]["modes"][visual_id] == "BACKSTOP_ONLY"
        rig.clear()
    # The scrub ended while the camera moves: no full plan from a camera
    # that is still moving.  The visual keeps its backstop.
    assert rig.reslices == []
    assert controller._camera_handoff == {rig.scene.id: {visual_id}}
    controller.reslice_scene(rig.scene.id)  # any reslice meanwhile stays coarse
    assert rig.reslices[-1]["modes"][visual_id] == "BACKSTOP_ONLY"
    rig.clear()

    await rig.release(x)
    # The camera end plans it in full, once.
    (reslice,) = rig.reslices
    assert reslice["modes"][visual_id] == "FULL"
    assert controller._camera_handoff == {}
    await drain_loading(controller)


async def test_a_scrub_ending_with_no_camera_moving_plans_the_target(make_rig):
    rig = make_rig(dim="2d", loading=DRAG)
    controller = rig.controller
    await rig.settle_first_frames()
    with controller.dims_interaction(rig.scene.id):
        controller.update_slice_indices(rig.scene.id, {0: 6.0})
        rig.clear()
    (reslice,) = rig.reslices
    assert reslice["modes"][rig.visual.id] == "FULL"
    assert controller._camera_handoff == {}
    await drain_loading(controller)


async def test_a_handed_over_visual_is_planned_even_with_camera_reslicing_off(
    make_rig,
):
    rig = make_rig(dim="2d", loading=DRAG)
    controller = rig.controller
    await rig.settle_first_frames()
    controller.camera_reslice_enabled = False
    await rig.press()
    x = await rig.move(3)
    with controller.dims_interaction(rig.scene.id):
        controller.update_slice_indices(rig.scene.id, {0: 6.0})
    rig.clear()
    await rig.release(x)
    (reslice,) = rig.reslices  # it is owed its full plan
    assert reslice["targets"] == frozenset({rig.visual.id})
    assert reslice["modes"][rig.visual.id] == "FULL"
    await drain_loading(controller)


# -- teardown -------------------------------------------------------------------


async def test_removing_a_canvas_or_a_scene_drops_the_motion(make_rig):
    rig = make_rig(n_canvases=2, settle_s=5.0)
    controller = rig.controller
    await rig.settle_first_frames()
    for canvas_id in rig.canvas_ids:
        state = controller.get_camera_state(canvas_id)
        controller.set_camera_state(
            canvas_id, state._replace(position=(1.0, 2.0, 3.0)), interactive=True
        )
    first, second = controller._camera_driver.tasks()
    rig.clear()

    controller.remove_canvas(rig.canvas_ids[1])
    await asyncio.sleep(0)
    assert second.cancelled()
    assert controller._camera_driver.tasks() == [first]
    controller.remove_scene(rig.scene.id)
    await asyncio.sleep(0)
    assert first.cancelled()
    assert rig.events == []  # dropped: no end event, no reslice
    assert rig.reslices == []
    assert not controller._deferred_reslice_tasks()


# -- no event loop --------------------------------------------------------------


def test_with_no_event_loop_a_camera_end_reslices_after_the_render(
    multiscale_image_store, clock, monkeypatch
):
    rig = Rig(multiscale_image_store, clock)
    try:
        controller = rig.controller
        rig.requested[0] = True
        for _ in range(3):
            rig.step()
        rig.clear()
        renders: list[int] = []
        renderer = rig.view._renderer
        original = renderer.render
        monkeypatch.setattr(
            renderer,
            "render",
            lambda *args, **kwargs: (renders.append(1), original(*args, **kwargs))[1],
        )
        at_reslice: list[int] = []
        render_manager = controller._render_manager
        recorded = render_manager.reslice_scene
        render_manager.reslice_scene = lambda *args, **kwargs: (
            at_reslice.append(len(renders)),
            recorded(*args, **kwargs),
        )[1]

        rig.submit("pointer_down", button=1, buttons=(1,))
        rig.step()
        renders_before = len(renders)
        rig.submit("pointer_move", x=106.0, buttons=(1,))
        rig.step()  # one draw: it sees the camera move

        # No timer can run, so the tick settled at once ...
        assert rig.events == [(0, "start", None), (0, "end", "settle")]
        # ... and its reslice ran in that draw, after the render.
        assert at_reslice == [renders_before + 1]
        assert len(rig.reslices) == 1
        assert controller._camera_reslices_after_draw == []
    finally:
        rig.controller.close()
