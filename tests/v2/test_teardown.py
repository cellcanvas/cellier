"""Teardown contracts: closing must actually release what was acquired.

Each of these pins one thing that was leaking before.  They are cheap unit
tests rather than whole-suite measurements, so a regression names itself
instead of surfacing as an unrelated test failing on a resource warning
whenever the garbage collector happens to run (which is how these were found).
"""

from __future__ import annotations

import asyncio
import warnings

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events._events import AppearanceChangedEvent
from cellier.scene.dims import CoordinateSystem

_POS = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)


def _controller_with_visual():
    controller = CellierController()
    scene = controller.add_scene(
        dim="3d",
        coordinate_system=CoordinateSystem(name="world", axis_labels=("z", "y", "x")),
        name="s",
    )
    visual = controller.add_points(
        PointsMemoryStore(positions=_POS), scene.id, None, "p"
    )
    return controller, scene, visual


def test_close_disconnects_the_psygnal_bridges():
    """A closed controller must stop reacting to the models it was watching.

    psygnal holds callbacks **strongly**, and the bridge handlers close over
    the controller, so leaving them connected keeps the whole controller
    reachable from the model -- and still handling its events.
    """
    controller, _scene, visual = _controller_with_visual()
    received: list = []
    controller._outgoing_events.subscribe(AppearanceChangedEvent, received.append)

    visual.appearance.opacity = 0.5
    assert received, "sanity: the bridge was connected to begin with"

    controller.close()
    received.clear()
    visual.appearance.opacity = 0.25

    assert received == []
    assert controller._visual_psygnal_handlers == {}
    assert controller._scene_psygnal_handlers == {}


def test_close_clears_the_event_buses():
    """They hold strong references to every subscribed handler."""
    controller, _scene, _visual = _controller_with_visual()
    assert controller._outgoing_events._subs, "sanity: something was subscribed"

    controller.close()

    assert controller._outgoing_events._subs == {}
    assert controller._outgoing_events._handle_index == {}
    assert controller._incoming_events._subs == {}


async def test_close_cancels_pending_camera_settle_tasks(qtbot):
    """A closed controller must not leave the camera-settle debounce running.

    Every other teardown path cancels these -- the ``camera_reslice_enabled``
    setter, ``remove_scene``, ``remove_canvas`` -- and ``close`` was the one
    that did not, so closing mid-gesture left a task holding the scene it was
    about to reslice.
    """
    controller, scene, _visual = _controller_with_visual()
    controller.add_canvas(scene_id=scene.id)
    canvas_id = controller._scene_to_canvases[scene.id][0]

    # Exactly what _on_camera_changed schedules once the camera moves.
    task = asyncio.create_task(controller._settle_after(canvas_id, scene.id))
    controller._settle_tasks[canvas_id] = task
    await asyncio.sleep(0)
    assert not task.done(), "sanity: the settle task is still pending"

    controller.close()
    await asyncio.sleep(0)

    assert controller._settle_tasks == {}
    assert task.cancelled()


async def test_close_cancels_slice_tasks_the_coordinator_no_longer_tracks(qtbot):
    """A superseded non-cancellable reslice must still be stopped by ``close``.

    ``SliceCoordinator.submit`` deliberately lets a non-cancellable visual
    (mesh, lines, points) run to completion rather than cancelling it, then
    overwrites its entry in ``_active_slice_ids``.  The predecessor keeps
    running under an id nothing upstream can name, so walking the scenes
    cannot reach it -- only going straight to the slicer can.
    """
    controller, scene, _visual = _controller_with_visual()
    controller.add_canvas(scene_id=scene.id)
    slicer = controller._render_manager._slicer
    coordinator = controller._render_manager._slice_coordinator

    # No await between them, so nothing completes and each supersedes the last.
    for _ in range(3):
        controller.reslice_all()

    tracked = set(coordinator._active_slice_ids.values())
    untracked = [sid for sid in slicer._tasks if sid not in tracked]
    assert untracked, "sanity: resubmission left an untracked task behind"

    controller.close()

    assert slicer._tasks == {}
    assert coordinator._active_slice_ids == {}


async def test_a_completed_slice_stops_being_tracked(qtbot):
    """Finishing a reslice must clear its ``_active_slice_ids`` entry.

    Only cancellation used to remove one, so a scene that simply loaded left
    its key behind forever and every later ``cancel_scene`` walked a dead
    request.
    """
    controller, scene, _visual = _controller_with_visual()
    controller.add_canvas(scene_id=scene.id)
    coordinator = controller._render_manager._slice_coordinator
    slicer = controller._render_manager._slicer

    controller.reslice_all()
    assert coordinator._active_slice_ids, "sanity: the request was tracked"

    await asyncio.gather(*list(slicer._tasks.values()))

    assert coordinator._active_slice_ids == {}
    assert slicer._tasks == {}


async def test_a_superseded_slice_completing_does_not_untrack_its_successor(qtbot):
    """The late finisher must drop only its own entry, never the newer one.

    A non-cancellable visual (points here) is left running when a newer
    reslice supersedes it, so the *older* task finishes after the newer one
    was recorded under the same key.  Pruning unconditionally would untrack
    work that is still in flight, putting back the orphan that
    ``cancel_all`` exists to sweep up.

    Both tasks share one event loop, so awaiting either runs both to
    completion; the invariant is checked at the moment each completion fires
    rather than afterwards.
    """
    controller, scene, _visual = _controller_with_visual()
    controller.add_canvas(scene_id=scene.id)
    coordinator = controller._render_manager._slice_coordinator
    slicer = controller._render_manager._slicer

    # Snapshot the tracking table at each completion — the pruning happens
    # immediately before this call, so it captures the effect of one finisher.
    snapshots: list[dict] = []
    emit = coordinator._emit_reslice_completed

    def _spy(*args, **kwargs):
        snapshots.append(dict(coordinator._active_slice_ids))
        return emit(*args, **kwargs)

    coordinator._emit_reslice_completed = _spy

    controller.reslice_all()
    first = dict(coordinator._active_slice_ids)
    controller.reslice_all()
    second = dict(coordinator._active_slice_ids)
    assert first and second and first != second, (
        "sanity: the second reslice superseded the first under the same key"
    )

    await asyncio.gather(*list(slicer._tasks.values()))

    assert len(snapshots) == 2, "sanity: both tasks ran to completion"
    # The superseded task finished first and must have left the successor's
    # entry alone.
    assert snapshots[0] == second
    # Once the successor finishes too, nothing is tracked.
    assert coordinator._active_slice_ids == {}


def test_close_is_idempotent():
    """Teardown runs from more than one place, so it must tolerate repeats."""
    controller, _scene, _visual = _controller_with_visual()
    controller.close()
    controller.close()


def test_the_scene_dims_bridge_is_recorded_so_it_can_be_disconnected():
    """It used to be connected without being recorded, so nothing could undo it."""
    controller, scene, _visual = _controller_with_visual()

    assert scene.id in controller._scene_psygnal_handlers

    controller.remove_scene(scene.id)
    assert scene.id not in controller._scene_psygnal_handlers


def test_an_anywidget_control_leaves_the_global_registry_when_closed():
    """``ipywidgets`` registers every widget process-globally at construction.

    Cellier's controls override ``close`` to emit ``closed`` for the bus; if
    that override does not also chain to ``Widget.close`` the widget stays in
    that table for the lifetime of the process, holding its traits and
    everything it subscribed to.
    """
    pytest.importorskip("ipywidgets")
    from ipywidgets import Widget

    from cellier.gui.anywidget.visuals import AnywidgetVisibleToggle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Widget.widgets is deprecated
        widget = AnywidgetVisibleToggle(_controller_with_visual()[2].id)
        model_id = widget.model_id
        assert model_id in Widget.widgets, "sanity: it registered itself"

        widget.close()
        assert model_id not in Widget.widgets

    # Note that a widget also registers its ``Layout``/``Style`` children, and
    # closing the parent does not close those.  The autouse teardown in
    # ``tests/conftest.py`` tracks every ``Widget`` for that reason; this test
    # is about the control's own override chaining to ``Widget.close``.


def test_closing_a_control_still_notifies_the_controller():
    """Chaining to ``Widget.close`` must not drop the ``closed`` signal."""
    pytest.importorskip("ipywidgets")
    from cellier.gui.anywidget.visuals import AnywidgetVisibleToggle

    controller, _scene, visual = _controller_with_visual()
    widget = AnywidgetVisibleToggle(visual.id)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widget.close()

    # The controller unsubscribed the widget's handlers on ``closed``.
    assert all(
        sub.owner_id != widget._id
        for subs in controller._outgoing_events._subs.values()
        for sub in subs
    )


# ---------------------------------------------------------------------------
# The application teardown path, end to end
# ---------------------------------------------------------------------------
#
# The unit tests above pin one contract each.  These pin the thing a user
# actually does: build a viewer, display it, close the handle.  Every widget
# created along the way must leave the global registry -- not only the controls
# but the containers the host composed them into, and the ``Layout`` widget
# ipywidgets attaches to each of them.


def _widget_registry() -> set[str]:
    from ipywidgets import Widget

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Widget.widgets is deprecated
        return set(Widget.widgets)


def _mesh_store():
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore

    return MeshMemoryStore(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32),
        indices=np.array([[0, 1, 2], [0, 1, 3]], np.int32),
    )


_RANGES = {0: (0.0, 2.0), 1: (0.0, 2.0), 2: (0.0, 2.0)}


def test_closing_a_displayed_viewer_releases_every_widget():
    """The whole point: a notebook can build and close viewers repeatedly."""
    pytest.importorskip("ipywidgets")
    from cellier.convenience import (
        AppearanceControls,
        Layout,
        MeshControlsConfig,
        Viewer,
        display,
    )
    from cellier.convenience.gui import build_canvas_widget
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    before = _widget_registry()

    viewer = Viewer(("z", "y", "x"), dim="3d", gui="anywidget")
    viewer.add_mesh(
        _mesh_store(),
        appearance=MeshFlatAppearance(),
        controls=MeshControlsConfig(appearance=True),
    )
    handle = display(
        viewer,
        Layout(
            center=build_canvas_widget(viewer, _RANGES, canvas_size=(120, 120)),
            left_dock=AppearanceControls(),
        ),
        fit="none",
        host="jupyter",
    )

    assert _widget_registry() - before, "sanity: display() built some widgets"

    handle.close()

    assert _widget_registry() - before == set()


def test_closing_an_ortho_viewer_releases_every_widget():
    """Four panels and a dock -- the widest composition the layout produces."""
    pytest.importorskip("ipywidgets")
    from cellier.convenience import (
        AppearanceControls,
        Layout,
        MeshControlsConfig,
        OrthoViewer,
        display,
    )
    from cellier.convenience.gui import build_ortho_grid_widget
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    before = _widget_registry()

    ortho = OrthoViewer(("z", "y", "x"), gui="anywidget")
    ortho.add_mesh(
        _mesh_store(),
        appearance=MeshFlatAppearance(),
        controls=MeshControlsConfig(appearance=True),
    )
    handle = display(
        ortho,
        Layout(
            center=build_ortho_grid_widget(ortho, _RANGES, canvas_size=(80, 80)),
            left_dock=AppearanceControls(),
        ),
        fit="none",
        host="jupyter",
    )

    handle.close()

    assert _widget_registry() - before == set()


def test_the_host_releases_the_wrapper_it_rendered():
    """``JupyterHost.present`` returns ``None``, so only the host can free it.

    It wraps the composed root in an outer box and displays it as a side
    effect; the caller never sees that wrapper, so without
    ``close_presented`` nothing could ever release it.
    """
    pytest.importorskip("ipywidgets")
    from cellier.convenience._hosts import JupyterHost
    from cellier.gui.anywidget import AnywidgetBox

    before = _widget_registry()
    host = JupyterHost()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert host.present(AnywidgetBox(children=[])) is None
        assert _widget_registry() - before

        host.close_presented()

    assert _widget_registry() - before == set()


def test_a_container_closes_the_subtree_it_composed():
    pytest.importorskip("ipywidgets")
    from cellier.gui.anywidget import AnywidgetBox

    before = _widget_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inner = AnywidgetBox(children=[])
        outer = AnywidgetBox(children=[inner])
        assert _widget_registry() - before

        outer.close()

    assert _widget_registry() - before == set()


# ---------------------------------------------------------------------------
# The Qt side of the same contract
# ---------------------------------------------------------------------------
#
# Qt has no process-global widget table, so the ipywidgets problem does not
# arise -- but the *asymmetry* did: the anywidget path hands back a
# ``DisplayHandle`` whose ``close()`` unsubscribes the controls it built, while
# the Qt path handed back a bare window and nothing else, so its controls
# stayed subscribed (and kept receiving events) for as long as the controller
# lived.  Design section 7.7 recorded this as a known gap.


def _qt_viewer_with_window():
    from cellier.convenience import (
        AppearanceControls,
        Layout,
        MeshControlsConfig,
        Viewer,
    )
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout._qt_renderer import render_qt
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    viewer = Viewer(("z", "y", "x"), dim="3d", gui="qt")
    viewer.add_mesh(
        _mesh_store(),
        appearance=MeshFlatAppearance(),
        controls=MeshControlsConfig(appearance=True),
    )
    canvas = build_canvas_widget(viewer, _RANGES)
    window = render_qt(Layout(center=canvas, left_dock=AppearanceControls()), viewer)
    return viewer, window, canvas


def _subscription_count(controller) -> int:
    return sum(len(subs) for subs in controller._outgoing_events._subs.values())


def _subscribed_owners(controller) -> set:
    return {
        getattr(subscription, "owner_id", None)
        for subscriptions in controller._outgoing_events._subs.values()
        for subscription in subscriptions
    }


def _dock_widget_ids(window) -> set:
    """Ids of the dock widgets the renderer built, read off its own list.

    Circular by nature -- it asks the renderer what it made -- so it can only
    check that everything the renderer *tracked* is released, never that it
    tracked the right things.  That second half is the center's, and it is
    asserted against the canvas directly below.
    """
    return {
        widget._id for widget in window._cellier_closeables if hasattr(widget, "_id")
    }


def test_closing_a_qt_window_unsubscribes_the_controls_it_built(qtbot):
    """The Qt counterpart of ``_RenderView.close()``.

    Asserted per widget rather than as "fewer subscriptions than before".
    The weaker form was satisfied by tearing down the docks alone, which is
    exactly what the renderer did: it never passed its ``closeables`` list
    into the center, so the canvas's dims control stayed subscribed for the
    life of the controller.

    What is deliberately *not* asserted is that the buses end up empty. The
    render manager and the render visuals keep their subscriptions, which is
    the point of ``test_closing_a_qt_window_leaves_the_controller_usable``
    below -- a window is a view, and the viewer may outlive it.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")

    viewer, window, canvas = _qt_viewer_with_window()
    # Captured before closing: teardown clears the list it walks.  The dims
    # control is named off the *canvas*, not off that list -- the bug was the
    # renderer never adding the center to it, which a list-derived expectation
    # could never have caught.
    widget_ids = _dock_widget_ids(window) | {canvas.dims_control._id}
    assert widget_ids, "sanity: the renderer built some widgets"
    assert widget_ids <= _subscribed_owners(viewer.controller), (
        "sanity: every widget it built subscribed something"
    )

    window.close()

    assert widget_ids & _subscribed_owners(viewer.controller) == set()


def test_closing_a_qt_window_leaves_the_controller_usable(qtbot):
    """A window is a view; the viewer may outlive it.

    Teardown deliberately stops at the widgets the renderer built -- releasing
    the canvases and the controller is ``CellierController.close()``, on this
    toolkit exactly as on the other.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")

    viewer, window, _canvas = _qt_viewer_with_window()
    visual = viewer.scene.visuals[0]

    window.close()

    # The model still works, and the render layer is still wired to it.
    viewer.controller.update_appearance_field(visual.id, "opacity", 0.5)
    assert visual.appearance.opacity == pytest.approx(0.5)


def test_the_qt_window_teardown_is_idempotent(qtbot):
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")

    _viewer, window, _canvas = _qt_viewer_with_window()
    window.close()
    window.close()
