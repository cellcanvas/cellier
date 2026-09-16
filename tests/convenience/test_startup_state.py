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

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="qt")
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

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="qt")

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

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="anywidget")
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
        reslice_scene=lambda scene_id, on_ready=None: on_ready and on_ready(),
    )
    viewer = SimpleNamespace(
        controller=controller, scene=SimpleNamespace(id=uuid4()), _ready_callbacks=[]
    )

    tracker = _init_view(viewer, fit="none", stall_timeout=None)

    assert tracker.state is StartupState.READY
