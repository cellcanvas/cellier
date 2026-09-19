"""Test fixtures for Cellier."""

import gc
import sys
import weakref

import numpy as np
import pytest
import tensorstore as ts

# Diagnostic bookkeeping for ``pytest_terminal_summary`` (see below).
# Each entry is (nodeid of the previous test, live renderers when the next test
# starts).  Counting at the next test's setup, not at teardown, lets the earlier
# test's own fixtures release their references first.
_LEAK_LOG: list[tuple[str, int]] = []
_ALL_RENDERERS: list[weakref.ref] = []
_LAST_NODEID: list[str] = []


def _track_instances(monkeypatch, cls) -> list[weakref.ref]:
    """Return a list that collects a weakref to every *cls* built from now on.

    Tests build these ad hoc rather than through a shared fixture, so
    instances are tracked at construction instead of via a fixture handle.
    """
    created: list[weakref.ref] = []
    original_init = cls.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(weakref.ref(self))

    monkeypatch.setattr(cls, "__init__", _tracking_init)
    return created


@pytest.fixture(autouse=True)
def _close_cellier_objects(monkeypatch, request):
    """Close every ``CellierController`` and ``CanvasView`` a test creates.

    Both own resources Python refcounting does not reclaim, and neither is
    released by dropping the object:

    * A render canvas is a parentless widget owned by the GUI backend, so a
      test that simply drops its controller leaks the canvas, its
      ``WgpuRenderer``, and the whole graph they reach.  Left alone the leak is
      cumulative: every canvas any earlier test built stays live and keeps
      drawing (they are created ``update_mode="continuous"``), so a full run
      ends holding ~100 of them.  That both starves the later tests
      (``tests/render`` slows to a crawl on CI) and leaves torn-down widgets
      for the Qt event loop to trip over (the Windows access violation).
    * An ``ipywidgets.Widget`` -- which every anywidget control is -- registers
      itself in a **process-global** table at construction, and only
      ``Widget.close()`` removes it.  Unclosed, a full run ends holding several
      hundred, each keeping its traits and everything it subscribed to alive.
    * A ``CellierController`` owns an ``AsyncSlicer`` holding live
      ``asyncio.Task`` objects.  Dropped rather than closed, those tasks are
      finalised by the garbage collector at an arbitrary later moment, and
      pytest's ``unraisableexception`` plugin reports the resulting
      ``ResourceWarning`` / "coroutine was never awaited" against **whatever
      test happens to be running then** -- which is how an unrelated test
      starts failing because a different module grew.

    Controllers are closed first: ``CellierController.close`` cancels pending
    slices and closes the canvases it owns, so the canvas pass afterwards only
    has to catch canvases built without a controller.  Widgets go last, because
    a cellier widget's ``close`` emits ``closed`` for the controller to act on.
    All three ``close`` methods are safe to call twice.
    """
    import pygfx as gfx
    from ipywidgets import Widget

    from cellier.controller import CellierController
    from cellier.render.canvas_view import CanvasView

    controllers = _track_instances(monkeypatch, CellierController)
    canvas_views = _track_instances(monkeypatch, CanvasView)
    # Tracked on the ipywidgets base, so every anywidget control is covered
    # without naming them one by one.
    widgets = _track_instances(monkeypatch, Widget)

    # Renderers have no close(); they are only counted, to see which tests
    # leave GPU-owning objects alive after teardown.
    renderers = _track_instances(monkeypatch, gfx.WgpuRenderer)

    if _LAST_NODEID:
        gc.collect()
        _LEAK_LOG.append((_LAST_NODEID[0], _live_renderer_count()))
    _LAST_NODEID[:] = [request.node.nodeid]

    yield

    for refs in (controllers, canvas_views, widgets):
        for ref in refs:
            obj = ref()
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:
                # Teardown must not turn a passing test into an error, and a
                # test that deliberately half-builds a controller is allowed.
                pass

    # Closing only breaks the reference chains.  Qt deletes closed widgets when
    # the event loop next runs, and the cycles left behind (draw callbacks,
    # event filters, pygfx objects) free their wgpu buffers and textures only
    # when the cyclic collector runs.  Do both now, or the GPU resources of
    # every earlier test stay allocated until some arbitrary later moment.
    _drain_qt_events()
    del controllers, canvas_views, widgets
    gc.collect()
    _drain_qt_events()

    _ALL_RENDERERS.extend(renderers)


def _drain_qt_events() -> None:
    """Process pending Qt events, if a Qt application exists."""
    widgets_module = sys.modules.get("PySide6.QtWidgets")
    if widgets_module is None:
        return
    app = widgets_module.QApplication.instance()
    if app is not None:
        app.processEvents()


def _live_renderer_count() -> int:
    return sum(1 for ref in _ALL_RENDERERS if ref() is not None)


def pytest_terminal_summary(terminalreporter) -> None:
    """Report where live ``WgpuRenderer`` objects accumulate (diagnostic).

    Temporary: used to find what exhausts the GPU device on Windows CI.  Lists
    each test after which the live renderer count rose.
    """
    if not _LEAK_LOG:
        return
    terminalreporter.section("wgpu renderer survivors (diagnostic)")
    terminalreporter.write_line(
        f"live renderers: max {max(n for _, n in _LEAK_LOG)}, "
        f"final {_live_renderer_count()}"
    )
    previous = 0
    rises = []
    for nodeid, live in _LEAK_LOG:
        if live > previous:
            rises.append((nodeid, live - previous, live))
        previous = live
    for nodeid, rise, live in rises[:40]:
        terminalreporter.write_line(f"  +{rise} (live {live})  {nodeid}")


@pytest.fixture(scope="session")
def offscreen_gpu() -> None:
    """Skip the test unless a usable wgpu offscreen adapter exists.

    Anything that captures a frame needs a real adapter.  ``tests/render`` gets
    this probe for free inside ``offscreen_renderer``; tests elsewhere (the
    convenience-layer capture tests, for one) need it on its own, and a shared
    session-scoped probe means a machine without an adapter skips cleanly
    instead of erroring once per test.
    """
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas as OffscreenRenderCanvas

    try:
        gfx.WgpuRenderer(OffscreenRenderCanvas(size=(16, 16), pixel_ratio=1))
    except Exception as exc:  # pragma: no cover - env-dependent skip path
        pytest.skip(f"no usable wgpu offscreen adapter: {exc}")


@pytest.fixture
def small_zarr_store(tmp_path):
    """A minimal 2-level multiscale zarr v3 store on disk (zeros, float32).

    Shared across the ``gui`` and ``convenience`` suites. ``tests/render`` keeps
    its own local copy (with extra render-specific data fixtures alongside it).
    """
    for name, shape in [("s0", (8, 8, 8)), ("s1", (4, 4, 4))]:
        level_path = tmp_path / name
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(level_path)},
            "metadata": {
                "shape": list(shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [4, 4, 4]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(np.zeros(shape, dtype=np.float32)).result()

    return tmp_path
