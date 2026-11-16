"""Startup is observable: state, per-scene readiness, progress, and stalling.

Before these, a viewer could report exactly one outcome -- success -- so one
that never finished starting looked identical to one still working.  These pin
the four things that make the difference visible: the state machine advances
in order, it never goes backwards, an unfinished startup says *where* it
stopped, and the aggregate never hides a stuck scene behind finished ones.
"""

from __future__ import annotations

import asyncio

import pytest

from cellier.convenience._startup import StartupState, StartupTracker
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageSingleAppearance


def _tracker(*keys: str) -> StartupTracker:
    return StartupTracker(list(keys))


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def test_a_fresh_tracker_is_idle():
    assert _tracker("scene").state is StartupState.IDLE


def test_the_state_advances_through_the_startup_stages():
    tracker = _tracker("scene")
    for state in (
        StartupState.WAITING_FOR_CANVAS,
        StartupState.WAITING_FOR_FRAME,
        StartupState.LOADING,
        StartupState.READY,
    ):
        tracker.advance("scene", state)
        assert tracker.state is state


def test_the_state_never_goes_backwards():
    """A duplicate signal must not undo progress.

    Both liveness signals can fire more than once -- a second Qt ``Show``, a
    second message from the browser -- and a canvas can render many frames.
    Re-reporting an earlier stage would walk a loaded viewer back to
    "waiting".
    """
    tracker = _tracker("scene")
    tracker.advance("scene", StartupState.READY)
    tracker.advance("scene", StartupState.WAITING_FOR_CANVAS)
    assert tracker.state is StartupState.READY


def test_the_aggregate_is_the_least_advanced_scene():
    """Three finished panels must not hide the one that is stuck."""
    tracker = _tracker("xy", "xz", "yz", "vol")
    for key in ("xy", "xz", "yz"):
        tracker.advance(key, StartupState.READY)
    tracker.advance("vol", StartupState.WAITING_FOR_FRAME)

    assert tracker.state is StartupState.WAITING_FOR_FRAME
    assert tracker.scene_states["xy"] is StartupState.READY
    assert tracker.scene_states["vol"] is StartupState.WAITING_FOR_FRAME


def test_all_scenes_ready_makes_the_viewer_ready():
    tracker = _tracker("a", "b")
    tracker.advance("a", StartupState.READY)
    assert tracker.state is not StartupState.READY
    tracker.advance("b", StartupState.READY)
    assert tracker.state is StartupState.READY


# ---------------------------------------------------------------------------
# Stalling -- the signal that did not exist
# ---------------------------------------------------------------------------


def test_stalling_reports_each_unfinished_scene_and_where_it_stopped():
    tracker = _tracker("xy", "vol")
    tracker.advance("xy", StartupState.READY)
    tracker.advance("vol", StartupState.WAITING_FOR_CANVAS)
    reports: list[dict] = []
    tracker.on_stalled(reports.append)

    tracker.mark_stalled()

    assert tracker.state is StartupState.STALLED
    # Names the scene and the stage, which is what makes it actionable.
    assert reports == [{"vol": StartupState.WAITING_FOR_CANVAS}]
    assert tracker.stalled_at == {"vol": StartupState.WAITING_FOR_CANVAS}
    # The finished scene is still reported as finished.
    assert tracker.scene_states["xy"] is StartupState.READY


def test_a_finished_startup_never_stalls():
    tracker = _tracker("scene")
    tracker.advance("scene", StartupState.READY)
    reports: list[dict] = []
    tracker.on_stalled(reports.append)

    tracker.mark_stalled()

    assert reports == []
    assert tracker.state is StartupState.READY


def test_progress_after_a_stall_clears_it():
    """A slow scene that eventually arrives is not stalled any more."""
    tracker = _tracker("scene")
    tracker.advance("scene", StartupState.LOADING)
    tracker.mark_stalled()
    assert tracker.state is StartupState.STALLED

    tracker.advance("scene", StartupState.READY)

    assert tracker.state is StartupState.READY
    assert tracker.stalled_at == {}


# ---------------------------------------------------------------------------
# Per-scene readiness and progress
# ---------------------------------------------------------------------------


def test_on_scene_ready_fires_for_that_scene_only():
    tracker = _tracker("xy", "vol")
    fired: list[str] = []
    tracker.on_scene_ready("vol", lambda: fired.append("vol"))

    tracker.advance("xy", StartupState.READY)
    assert fired == []

    tracker.advance("vol", StartupState.READY)
    assert fired == ["vol"]


def test_on_scene_ready_fires_immediately_when_already_ready():
    """Registering after the fact must not silently never fire."""
    tracker = _tracker("scene")
    tracker.advance("scene", StartupState.READY)
    fired: list[int] = []

    tracker.on_scene_ready("scene", lambda: fired.append(1))

    assert fired == [1]


def test_progress_counts_scenes_as_they_become_ready():
    tracker = _tracker("a", "b", "c")
    seen: list[tuple[int, int]] = []
    tracker.on_progress(lambda done, total: seen.append((done, total)))

    tracker.advance("a", StartupState.READY)
    tracker.advance("b", StartupState.READY)
    tracker.advance("c", StartupState.READY)

    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert tracker.progress == (3, 3)


def test_describe_names_the_stage_and_the_scenes():
    tracker = _tracker("xy", "vol")
    tracker.advance("xy", StartupState.READY)
    tracker.advance("vol", StartupState.WAITING_FOR_CANVAS)

    report = tracker.describe()

    assert "waiting_for_canvas" in report
    assert "1/2 scenes ready" in report
    assert "vol=waiting_for_canvas" in report


# ---------------------------------------------------------------------------
# The viewer surface
# ---------------------------------------------------------------------------


def test_an_unstarted_viewer_reports_idle_rather_than_raising():
    """Reading the state must work before ``display``/``launch`` has run."""
    pytest.importorskip("qtpy")
    import numpy as np

    from cellier.convenience import Viewer
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.visuals import InMemoryImageAppearance

    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="qt"
    )
    viewer.add_image(
        ImageMemoryStore(data=np.random.rand(4, 4, 4).astype(np.float32)),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis"),
    )

    assert viewer.startup_state is StartupState.IDLE
    assert viewer.scene_startup_states == {}
    assert "idle" in viewer.startup_report()


def test_the_startup_hooks_refuse_before_the_viewer_has_started():
    """A hook registered too early would silently never fire."""
    pytest.importorskip("qtpy")
    from cellier.convenience import Viewer

    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="qt"
    )

    for register in (
        lambda: viewer.on_scene_ready("scene", lambda: None),
        lambda: viewer.on_startup_progress(lambda done, total: None),
        lambda: viewer.on_startup_stalled(lambda stalled: None),
    ):
        with pytest.raises(RuntimeError, match="started viewer"):
            register()


async def test_a_viewer_that_never_connects_reports_stalled(qtbot):
    """End to end: the timer arms, fires, and names what it was waiting for.

    The tracker unit tests drive ``mark_stalled`` directly; this pins the part
    that only exists at runtime -- that ``_init_view`` actually arms a timer,
    and that an anywidget canvas with no browser behind it is what trips it.
    """
    pytest.importorskip("qtpy")
    import numpy as np

    from cellier.convenience import Viewer, axis_values_from_viewer
    from cellier.convenience._launch import _init_view
    from cellier.convenience.gui import build_canvas_widget
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.visuals import InMemoryImageAppearance

    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="anywidget"
    )
    viewer.add_image(
        ImageMemoryStore(data=np.random.rand(4, 4, 4).astype(np.float32)),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis"),
    )
    build_canvas_widget(viewer, axis_values_from_viewer(viewer))

    stalled: list[dict] = []
    _init_view(viewer, fit="none", stall_timeout=0.2)
    viewer.on_startup_stalled(stalled.append)

    await asyncio.sleep(0.5)

    assert viewer.startup_state is StartupState.STALLED
    # Names the stage, not just the failure: nothing ever mounted the canvas.
    assert stalled == [{"scene": StartupState.WAITING_FOR_CANVAS}]


def test_a_stall_timeout_of_none_arms_no_timer():
    """Opting out must be possible; a long-running load is not a fault."""
    from types import SimpleNamespace
    from uuid import uuid4

    from cellier.convenience._launch import _init_view

    controller = SimpleNamespace(
        _id=uuid4(),
        camera_reslice_enabled=True,
        get_canvas_ids=lambda scene_id: [],
        on_scene_ready=lambda scene_id, callback: callback(),
        on_scene_removed=lambda scene_id, callback, owner_id: SimpleNamespace(
            unsubscribe=lambda: None
        ),
    )
    viewer = SimpleNamespace(
        controller=controller, scene=SimpleNamespace(id=uuid4()), _ready_callbacks=[]
    )

    tracker = _init_view(viewer, fit="none", stall_timeout=None)

    assert tracker.state is StartupState.READY


# ---------------------------------------------------------------------------
# Cancelling a startup hook
# ---------------------------------------------------------------------------


def test_a_scene_ready_handle_cancels_the_callback():
    tracker = _tracker("scene")
    fired: list[int] = []
    handle = tracker.on_scene_ready("scene", lambda: fired.append(1))

    handle.unsubscribe()
    tracker.advance("scene", StartupState.READY)

    assert fired == []
    assert not handle.active


def test_a_scene_ready_handle_goes_inactive_when_it_fires():
    tracker = _tracker("scene")
    handle = tracker.on_scene_ready("scene", lambda: None)
    assert handle.active

    tracker.advance("scene", StartupState.READY)

    assert not handle.active


def test_a_scene_ready_handle_is_inactive_when_the_scene_was_ready():
    tracker = _tracker("scene")
    tracker.advance("scene", StartupState.READY)

    assert not tracker.on_scene_ready("scene", lambda: None).active


def test_progress_and_stall_handles_stay_active_until_cancelled():
    tracker = _tracker("a", "b")
    seen: list = []
    progress = tracker.on_progress(lambda done, total: seen.append(done))
    stalled = tracker.on_stalled(seen.append)

    tracker.advance("a", StartupState.READY)
    assert progress.active  # fires more than once, so it stays
    progress.unsubscribe()
    stalled.unsubscribe()
    tracker.mark_stalled()
    tracker.advance("b", StartupState.READY)

    assert seen == [1]
    assert not progress.active
    assert not stalled.active


# ---------------------------------------------------------------------------
# A scene removed during startup
# ---------------------------------------------------------------------------


def test_forgetting_a_scene_leaves_the_others_to_decide_the_state():
    tracker = _tracker("a", "b")
    seen: list = []
    tracker.on_progress(lambda done, total: seen.append((done, total)))
    fired: list[int] = []
    handle = tracker.on_scene_ready("a", lambda: fired.append(1))
    tracker.advance("b", StartupState.READY)
    tracker.mark_stalled()

    tracker.forget("a")

    assert tracker.state is StartupState.READY
    assert tracker.scene_states == {"b": StartupState.READY}
    assert tracker.stalled_at == {}
    assert seen == [(1, 2), (1, 1)]
    # The removed scene's callback is dropped unfired, and says so.
    assert fired == []
    assert not handle.active


def _two_scene_viewer(controller):
    """A stand-in viewer with two scenes: "a" never starts, "b" is ready.

    "a" has a canvas that is never shown, so it waits for its canvas for
    good; "b" has none, so it has nothing to wait for and is ready at once.
    """
    from types import SimpleNamespace

    import numpy as np

    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.visuals import InMemoryImageAppearance

    scenes = {}
    for key in ("a", "b"):
        scene = controller.add_scene(displayed_axes=("z", "y", "x"), name=key)
        controller.add_image(
            data=ImageMemoryStore(data=np.random.rand(8, 8, 8).astype(np.float32)),
            scene_id=scene.id,
            appearance=InMemoryImageAppearance(),
            single=InMemoryImageSingleAppearance(color_map="viridis"),
        )
        scenes[key] = scene
    controller.add_canvas(scene_id=scenes["a"].id)
    return SimpleNamespace(controller=controller, scenes=scenes, _ready_callbacks=[])


def _start():
    from cellier.controller import CellierController
    from cellier.convenience._launch import _init_view
    from cellier.convenience._startup import on_viewer_ready

    controller = CellierController()
    viewer = _two_scene_viewer(controller)
    fired: list[int] = []
    on_viewer_ready(viewer, lambda: fired.append(1))
    tracker = _init_view(viewer, fit="none", stall_timeout=None)
    assert tracker.scene_states == {
        "a": StartupState.WAITING_FOR_CANVAS,
        "b": StartupState.READY,
    }
    assert not controller.camera_reslice_enabled
    return controller, viewer, tracker, fired


def test_a_scene_removed_mid_startup_counts_as_done(qtbot):
    """The remaining scenes finish startup; reslicing comes back on.

    Removing a scene cancels its readiness callback, so before the fix the
    launcher waited for it for good: no ``on_ready``, the state stuck short
    of ``READY``, and camera reslicing left off for every scene.
    """
    controller, viewer, tracker, fired = _start()

    controller.remove_scene(viewer.scenes["a"].id)

    assert fired == [1]
    assert tracker.state is StartupState.READY
    assert tracker.scene_states == {"b": StartupState.READY}
    assert controller.camera_reslice_enabled
    controller.close()


def test_removing_a_ready_scene_does_not_count_it_twice(qtbot):
    """A scene already counted as ready must not also count as removed."""
    controller, viewer, tracker, fired = _start()

    controller.remove_scene(viewer.scenes["b"].id)

    assert fired == []
    assert tracker.state is StartupState.WAITING_FOR_CANVAS
    assert not controller.camera_reslice_enabled

    controller.remove_scene(viewer.scenes["a"].id)

    assert fired == [1]
    assert tracker.state is StartupState.READY
    assert controller.camera_reslice_enabled
    controller.close()
