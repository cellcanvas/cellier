"""Test fixtures for Cellier."""

import sys
import weakref

import numpy as np
import pytest
import tensorstore as ts


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


def pytest_report_header(config):
    """Say which coverage core this run measures with, if it measures.

    ``SysMonitor`` is cheap; ``CTracer`` and ``PyTracer`` pay a call on
    every line run and make a CI job half again as slow
    (``scripts/ci_profiling/ci_profiling_investigation.md``).
    """
    try:
        import coverage
    except ImportError:
        return None
    cov = coverage.Coverage.current()
    if cov is None:
        return None
    return f"coverage core: {dict(cov.sys_info()).get('core')}"


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Give rendercanvas its asyncgen hooks back after an async test.

    rendercanvas picks its sleep by reading ``sys.get_asyncgen_hooks()``, and
    its Qt loop installs hooks of its own when it starts.  asyncio saves the
    hooks when a loop starts running and puts the saved ones back when it
    stops.  So when the Qt loop starts *inside* an async test (the test pumps
    Qt, ``canvas.force_draw()`` for one), asyncio wipes the hooks rendercanvas
    just installed as the test returns, and rendercanvas does not notice.  Its
    sleep is then a no-op, the canvas scheduler never yields, and the next
    ``processEvents`` -- pytest-qt calls one right after the test body --
    draws frames forever.

    This runs between the test body and that ``processEvents``.  An app is not
    exposed: ``QtAsyncio`` leaves the hooks alone, and its loop lasts as long
    as the app does.
    """
    try:
        return (yield)
    finally:
        _restore_rendercanvas_asyncgen_hooks()


def _restore_rendercanvas_asyncgen_hooks() -> None:
    qt_backend = sys.modules.get("rendercanvas.qt")
    if qt_backend is None:
        return
    loop = qt_backend.loop
    # Name-mangled private state: there is no public way to ask whether the
    # loop believes its hooks are installed.
    believes_installed = getattr(loop, "_BaseLoop__hook_data", None) is not None
    if believes_installed and sys.get_asyncgen_hooks().firstiter is None:
        sys.set_asyncgen_hooks(
            firstiter=loop._asyncgen_firstiter_hook,
            finalizer=loop._asyncgen_finalizer_hook,
        )


@pytest.fixture(autouse=True)
def _close_cellier_objects(monkeypatch):
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
    from ipywidgets import Widget

    from cellier.controller import CellierController
    from cellier.render.canvas_view import CanvasView

    controllers = _track_instances(monkeypatch, CellierController)
    canvas_views = _track_instances(monkeypatch, CanvasView)
    # Tracked on the ipywidgets base, so every anywidget control is covered
    # without naming them one by one.
    widgets = _track_instances(monkeypatch, Widget)

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

    # Qt deletes a closed widget only when the event loop next runs, so drain
    # it now rather than leave every earlier test's widgets pending.
    _drain_qt_events()


def _drain_qt_events() -> None:
    """Process pending Qt events, if a Qt application exists."""
    widgets_module = sys.modules.get("PySide6.QtWidgets")
    if widgets_module is None:
        return
    app = widgets_module.QApplication.instance()
    if app is not None:
        # ``deleteLater`` (what closing through ``qtbot`` does) is acted on
        # only by a running event loop; ``processEvents`` alone leaves the
        # widget alive.  Deliver those deletions too.
        core_module = sys.modules["PySide6.QtCore"]
        app.sendPostedEvents(None, core_module.QEvent.Type.DeferredDelete)
        app.processEvents()


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


@pytest.fixture(scope="session")
def pyramid_root(tmp_path_factory):
    """Return ``root_of(spec, labels=False)``: where a pyramid is on disk.

    Each ``(spec, labels)`` pyramid of ``tests/_plane_fixtures`` is written once
    per session and shared: a pyramid is thousands of chunk files, and writing
    one per test dominated the plane tests on the Windows runners.  The stores
    are for reading only; a test that writes to its store must write its own
    with ``write_pyramid``.
    """
    from tests._plane_fixtures import write_pyramid

    written = {}

    def root_of(spec, *, labels: bool = False):
        key = (spec, labels)
        if key not in written:
            root = tmp_path_factory.mktemp("pyramid")
            write_pyramid(root, spec, labels=labels)
            written[key] = root
        return written[key]

    return root_of


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
