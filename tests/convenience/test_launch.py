"""Tests for ``cellier.convenience._launch`` orchestration.

``_init_view`` is pure controller-orchestration, so it is exercised with a stub
controller/viewer and no Qt event loop. ``run`` dispatch and ``DisplayHandle``
teardown are likewise driven with stubs.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cellier.convenience import _launch
from cellier.convenience._launch import DisplayHandle, _init_view, run


class StubController:
    """Records the calls ``_init_view`` makes; fires first-frame callbacks on demand."""

    def __init__(self, canvas_ids_map=None):
        self._id = uuid4()
        self.camera_reslice_enabled = True
        self._canvas_ids = canvas_ids_map or {}
        self.fit_calls: list = []
        self.reslice_calls: list = []
        self.cancel_calls: list = []
        self._first_frame_cbs: list = []
        self._connected_cbs: list = []

    def get_canvas_ids(self, scene_id):
        return self._canvas_ids.get(scene_id, [])

    def fit_camera(self, scene_id):
        self.fit_calls.append(scene_id)

    def reslice_scene(self, scene_id, on_ready=None):
        self.reslice_calls.append(scene_id)
        if on_ready is not None:
            on_ready()

    def on_canvas_first_frame(self, canvas_id, cb, owner_id=None):
        self._first_frame_cbs.append(cb)

    def on_canvas_connected(self, canvas_id, cb, owner_id=None):
        self._connected_cbs.append(cb)

    def fire_connected(self):
        """Simulate the front end reporting the canvas live."""
        for cb in list(self._connected_cbs):
            cb()

    def fire_first_frames(self):
        """Simulate a rendered frame.

        Fires the connection first: a frame implies a live canvas, and
        startup now waits for the two in order.
        """
        self.fire_connected()
        for cb in list(self._first_frame_cbs):
            cb()

    def cancel_pending_slices(self, scene_id):
        self.cancel_calls.append(scene_id)


def _single_scene_viewer(controller, gui="qt"):
    scene = SimpleNamespace(id=uuid4())
    viewer = SimpleNamespace(
        controller=controller, scene=scene, gui=gui, _ready_callbacks=[]
    )
    return viewer, scene


# ---------------------------------------------------------------------------
# _init_view -- no-canvas branch
# ---------------------------------------------------------------------------


def test_no_canvas_reslices_directly_and_restores_reslice():
    controller = StubController()  # no canvas registered for the scene
    viewer, scene = _single_scene_viewer(controller)
    called = []

    _init_view(viewer, fit="ready", on_ready=lambda: called.append(1))

    assert controller.reslice_calls == [scene.id]
    assert controller.fit_calls == []  # no canvas -> never fits
    assert called == [1]
    assert controller.camera_reslice_enabled is True  # restored


def test_viewer_ready_callbacks_and_on_ready_both_fire():
    controller = StubController()
    viewer, _scene = _single_scene_viewer(controller)
    order = []
    viewer._ready_callbacks.append(lambda: order.append("viewer"))

    _init_view(viewer, on_ready=lambda: order.append("arg"))

    assert order == ["viewer", "arg"]


# ---------------------------------------------------------------------------
# _init_view -- canvas branch fit-policy matrix
# ---------------------------------------------------------------------------


def test_reslice_suppressed_until_ready_then_restored():
    canvas_id = uuid4()
    scene_id = uuid4()
    controller = StubController({scene_id: [canvas_id]})
    scene = SimpleNamespace(id=scene_id)
    viewer = SimpleNamespace(
        controller=controller, scene=scene, gui="qt", _ready_callbacks=[]
    )

    _init_view(viewer, fit="ready")

    # First frame has not fired yet: reslicing is suppressed, no fit happened.
    assert controller.camera_reslice_enabled is False
    assert controller.fit_calls == []

    controller.fire_first_frames()

    # fit="ready": fit on first frame + re-fit on ready == two fits.
    assert controller.fit_calls == [scene_id, scene_id]
    assert controller.camera_reslice_enabled is True  # restored after ready


def test_fit_immediate_fits_once():
    canvas_id = uuid4()
    scene_id = uuid4()
    controller = StubController({scene_id: [canvas_id]})
    scene = SimpleNamespace(id=scene_id)
    viewer = SimpleNamespace(
        controller=controller, scene=scene, gui="qt", _ready_callbacks=[]
    )

    _init_view(viewer, fit="immediate")
    controller.fire_first_frames()

    assert controller.fit_calls == [scene_id]  # no re-fit on ready


def test_fit_none_never_fits():
    canvas_id = uuid4()
    scene_id = uuid4()
    controller = StubController({scene_id: [canvas_id]})
    scene = SimpleNamespace(id=scene_id)
    viewer = SimpleNamespace(
        controller=controller, scene=scene, gui="qt", _ready_callbacks=[]
    )

    _init_view(viewer, fit="none")
    controller.fire_first_frames()

    assert controller.fit_calls == []


def test_empty_scene_list_fires_startup_immediately():
    controller = StubController()
    viewer = SimpleNamespace(controller=controller, scenes={}, _ready_callbacks=[])
    called = []

    _init_view(viewer, on_ready=lambda: called.append(1))

    assert called == [1]
    assert controller.camera_reslice_enabled is True


# ---------------------------------------------------------------------------
# run() dispatch
# ---------------------------------------------------------------------------


def test_run_anywidget_calls_display(monkeypatch):
    calls = {}

    def fake_display(viewer, layout, *, fit, on_ready):
        calls["display"] = (viewer, layout, fit, on_ready)
        return "handle"

    monkeypatch.setattr(_launch, "display", fake_display)
    viewer = SimpleNamespace(gui="anywidget")
    layout = object()

    result = run(viewer, layout, fit="immediate")

    assert result == "handle"
    assert calls["display"][0] is viewer
    assert calls["display"][2] == "immediate"


def test_run_qt_calls_launch_and_returns_none(monkeypatch):
    calls = {}

    def fake_launch(viewer, layout, *, fit, on_ready):
        calls["launch"] = (viewer, layout, fit)

    monkeypatch.setattr(_launch, "launch", fake_launch)
    viewer = SimpleNamespace(gui="qt")
    layout = object()

    result = run(viewer, layout)

    assert result is None
    assert calls["launch"][0] is viewer


def test_run_unknown_gui_raises():
    viewer = SimpleNamespace(gui="bogus")
    with pytest.raises(ValueError, match=r"Unknown viewer\.gui"):
        run(viewer, object())


# ---------------------------------------------------------------------------
# DisplayHandle
# ---------------------------------------------------------------------------


def test_display_handle_close_tears_down_once():
    controller = StubController()
    scene = SimpleNamespace(id=uuid4())
    viewer = SimpleNamespace(controller=controller, scene=scene)
    view = SimpleNamespace(close_calls=0)
    view.close = lambda: setattr(view, "close_calls", view.close_calls + 1)
    sidecar = SimpleNamespace(close_calls=0)
    sidecar.close = lambda: setattr(sidecar, "close_calls", sidecar.close_calls + 1)

    handle = DisplayHandle(viewer, view, sidecar=sidecar)
    handle.close()

    assert view.close_calls == 1
    assert controller.cancel_calls == [scene.id]
    assert sidecar.close_calls == 1

    # Idempotent: a second close is a no-op.
    handle.close()
    assert view.close_calls == 1
    assert sidecar.close_calls == 1


def test_display_handle_close_multi_scene_cancels_each():
    controller = StubController()
    scenes = {"a": SimpleNamespace(id=uuid4()), "b": SimpleNamespace(id=uuid4())}
    viewer = SimpleNamespace(controller=controller, scenes=scenes)
    view = SimpleNamespace(close=lambda: None)

    DisplayHandle(viewer, view).close()

    assert set(controller.cancel_calls) == {s.id for s in scenes.values()}


def test_display_handle_repr_is_inert():
    handle = DisplayHandle(SimpleNamespace(), SimpleNamespace(close=lambda: None))
    assert handle._repr_mimebundle_() == {}
    assert repr(handle) == ""
