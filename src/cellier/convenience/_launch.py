"""Blocking and non-blocking launchers for a cellier Viewer.

Qt imports are lazy so that importing this module does not require a Qt
installation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Literal

from cellier.convenience._startup import StartupState, StartupTracker
from cellier.events._bus import fire_registered

if TYPE_CHECKING:
    from sidecar import Sidecar

    from cellier.convenience._ortho_viewer import OrthoViewer
    from cellier.convenience._sidecar import SidecarOptions
    from cellier.convenience._viewer import Viewer
    from cellier.convenience.layout._spec import Layout
    from cellier.events import SubscriptionHandle

    ViewerLike = Viewer | OrthoViewer

FitMode = Literal["ready", "immediate", "none"]

#: Seconds before startup is reported stalled.  Long enough that slow
#: remote data is not maligned, short enough to notice within one look.
_DEFAULT_STALL_TIMEOUT = 30.0


def launch(
    viewer: ViewerLike,
    layout_or_window: object,
    *,
    fit: FitMode | None = None,
    on_ready: Callable[[], None] | None = None,
    handle_sigint: bool = True,
) -> None:
    """Show *layout_or_window* and block until it is closed.

    Creates a ``QApplication`` if one does not already exist, then starts the
    Qt + asyncio event loop via ``PySide6.QtAsyncio``.  Intended for scripts
    and the CLI.  For interactive / Jupyter use, call `show` instead.

    The initial camera fit and data load are deferred to the canvas's first
    rendered frame (see `CellierController.on_canvas_first_frame`), so
    they run only once the canvas has its final size and the event loop is
    live — no timer required.

    Parameters
    ----------
    viewer : Viewer or OrthoViewer
        The viewer whose controller is used for ``fit_camera`` /
        ``on_scene_ready``.
    layout_or_window : Layout or QMainWindow or QWidget
        Either a `Layout` spec (rendered to
        a ``QMainWindow`` by the Qt renderer) or a pre-built top-level Qt
        window.
    fit : "ready", "immediate", "none" or None
        Camera-fit policy applied at startup.  ``"ready"`` fits the camera
        on the first frame and again once the startup data has loaded;
        ``"immediate"`` fits on the first frame only; ``"none"`` never
        fits.  Both fit to the box measured from the stores (see
        ``CellierController.fit_camera``), so they differ only when that
        box changes while the data loads.
        ``None`` (the default) is ``"ready"``, except for a scene whose
        cameras were all placed from their models (a viewer loaded with
        ``from_file``), which keeps its view.
    on_ready : Callable[[], None] or None
        Optional zero-argument callback fired once, after every scene's startup
        data has committed to the GPU.  Fires in addition to any callbacks
        registered via ``viewer.on_ready``.
    handle_sigint : bool
        Pass ``True`` (default) to let QtAsyncio install a ``SIGINT`` handler
        so that Ctrl-C closes the window cleanly.
    """
    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([sys.argv[0]])

    window = _resolve_qt_window(layout_or_window, viewer)
    window.show()
    # _init_view only subscribes to the first-frame event and requests a draw;
    # the actual fit + reslice run inside that callback, which fires once
    # QtAsyncio.run() below starts the loop and the canvas paints.  No timer is
    # needed because the work is gated on a real render event, not a delay.
    _init_view(viewer, fit=fit, on_ready=on_ready)
    QtAsyncio.run(handle_sigint=handle_sigint)


def show(
    viewer: ViewerLike,
    layout_or_window: object,
    *,
    fit: FitMode | None = None,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Show a viewer window without blocking.

    Requires a Qt event loop that is already running (e.g. inside IPython /
    Jupyter after ``%gui qt``).  The initial camera fit and data load are
    deferred to the canvas's first rendered frame.

    Parameters
    ----------
    viewer : Viewer or OrthoViewer
        The viewer whose controller is used for ``fit_camera`` /
        ``on_scene_ready``.
    layout_or_window : Layout or QMainWindow or QWidget
        Either a `Layout` spec (rendered to
        a ``QMainWindow`` by the Qt renderer) or a pre-built top-level Qt
        window.
    fit : "ready", "immediate", "none" or None
        Camera-fit policy applied at startup.  ``"ready"`` fits the camera
        on the first frame and again once the startup data has loaded;
        ``"immediate"`` fits on the first frame only; ``"none"`` never
        fits.  Both fit to the box measured from the stores (see
        ``CellierController.fit_camera``), so they differ only when that
        box changes while the data loads.
        ``None`` (the default) is ``"ready"``, except for a scene whose
        cameras were all placed from their models (a viewer loaded with
        ``from_file``), which keeps its view.
    on_ready : Callable[[], None] or None
        Optional zero-argument callback fired once, after every scene's startup
        data has committed to the GPU.

    Raises
    ------
    RuntimeError
        If no ``QApplication`` instance is running.
    """
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        raise RuntimeError(
            "No Qt event loop is running. "
            "Use launch() for scripts, or run inside IPython/Jupyter with %gui qt."
        )
    window = _resolve_qt_window(layout_or_window, viewer)
    window.show()
    _init_view(viewer, fit=fit, on_ready=on_ready)


class DisplayHandle:
    """Inert teardown handle returned by `display`.

    Rendering is performed imperatively by the host inside ``present()``; this
    handle is the cell's return value but has no representation of its own (its
    ``_repr_mimebundle_`` is empty), so the viewer is rendered exactly once.

    Call `close` to unsubscribe the control panel(s) from the bus,
    cancel any pending slices, and close the sidecar panel (if any) -- e.g.
    before re-running a cell so the prior panel and in-flight reads do not
    leak.
    """

    def __init__(
        self,
        viewer: ViewerLike,
        view: object,
        sidecar: Sidecar | None = None,
        host: object | None = None,
    ) -> None:
        self._viewer = viewer
        self._view = view
        self._sidecar = sidecar
        self._host = host
        self._closed = False

    def close(self) -> None:
        """Tear down the controls and close the sidecar."""
        if self._closed:
            return
        self._closed = True
        self._view.close()
        # An imperative host renders a wrapper of its own that the caller never
        # sees, so only the host can release it.
        close_presented = getattr(self._host, "close_presented", None)
        if close_presented is not None:
            close_presented()
        if self._sidecar is not None:
            self._sidecar.close()

    def _repr_mimebundle_(self, **kwargs):
        # Inert: the host already rendered the viewer imperatively in present(),
        # so the handle itself must not produce a second copy in the cell.
        return {}

    def __repr__(self) -> str:
        # IPython's plain-text formatter falls back to repr() even when the mime
        # bundle is empty; blank it so no stray ``Out[]`` text appears under the
        # viewer.
        return ""


def display(
    viewer: ViewerLike,
    layout: Layout,
    *,
    fit: FitMode | None = None,
    on_ready: Callable[[], None] | None = None,
    host: str | None = None,
    sidecar: bool | SidecarOptions | None = None,
    stall_timeout: float | None = _DEFAULT_STALL_TIMEOUT,
) -> object:
    """Compose and present an anywidget viewer non-blockingly.

    The notebook counterpart of `launch`.  Resolves the anywidget host
    (Jupyter or marimo), renders the *layout* spec for that host, presents
    the result, and arms first-frame startup.

    Parameters
    ----------
    viewer : Viewer or OrthoViewer
        The viewer whose controller drives fit / reslice.
    layout : Layout
        Declarative layout spec -- center canvas(es) plus optional dock
        controls.  Build with `Layout` or
        its preset (``Layout.single``).
    fit : "ready", "immediate", "none" or None
        Camera-fit policy applied at startup.  ``"ready"`` fits the camera
        on the first frame and again once the startup data has loaded;
        ``"immediate"`` fits on the first frame only; ``"none"`` never
        fits.  Both fit to the box measured from the stores (see
        ``CellierController.fit_camera``), so they differ only when that
        box changes while the data loads.
        ``None`` (the default) is ``"ready"``, except for a scene whose
        cameras were all placed from their models (a viewer loaded with
        ``from_file``), which keeps its view.
    on_ready : Callable[[], None] or None
        Optional zero-argument callback fired once every scene's startup data
        has committed to the GPU.
    host : "jupyter", "marimo", or None
        Explicit host override; auto-detected when ``None``.
    stall_timeout : float or None
        Seconds before startup is reported stalled through
        ``viewer.on_startup_stalled``.  ``None`` disables the check.
    sidecar : True, SidecarOptions, or None
        Present the viewer in a ``jupyterlab-sidecar`` tab instead of below
        the cell.  ``True`` uses `SidecarOptions`
        defaults.
        Requires the optional ``sidecar`` package and the Jupyter host (not
        marimo, which already places cell output in its own tab).

    Returns
    -------
    object
        For imperative hosts (Jupyter) an inert `DisplayHandle` whose
        `DisplayHandle.close` tears down the controls, cancels pending
        slices, and closes the sidecar panel (if any).  For return-value hosts
        (marimo) the host-native renderable (so the cell renders it), with a
        best-effort ``close`` attached.
    """
    from cellier.convenience._hosts import JupyterHost, resolve_host
    from cellier.convenience.layout._anywidget_renderer import render_anywidget

    resolved_host = resolve_host(host)

    sidecar_instance = None
    if sidecar:
        if not isinstance(resolved_host, JupyterHost):
            raise RuntimeError(
                "sidecar=... requires the Jupyter host; marimo already "
                "places cell output in its own re-arrangeable tab."
            )
        from cellier.convenience._sidecar import resolve_sidecar

        sidecar_instance = resolve_sidecar(sidecar)

    render_view = render_anywidget(layout, viewer, resolved_host)
    if sidecar_instance is not None:
        with sidecar_instance:
            cell_value = resolved_host.present(render_view.root)
    else:
        cell_value = resolved_host.present(render_view.root)

    _init_view(viewer, fit=fit, on_ready=on_ready, stall_timeout=stall_timeout)

    handle = DisplayHandle(
        viewer, render_view, sidecar=sidecar_instance, host=resolved_host
    )
    if cell_value is None:
        return handle
    try:
        cell_value.close = handle.close  # type: ignore[attr-defined]
    except Exception:
        pass
    return cell_value


def run(
    viewer: ViewerLike,
    layout: Layout,
    *,
    fit: FitMode | None = None,
    on_ready: Callable[[], None] | None = None,
) -> object:
    """Show a viewer, dispatching to the right host based on ``viewer.gui``.

    The portable entry point: replaces separate `display` / `launch`
    calls so notebook and script code can be identical up to the ``gui=``
    argument on `Viewer`.

    * ``gui="anywidget"`` -- calls `display` (non-blocking; returns a
      `DisplayHandle` for Jupyter or the renderable for marimo).
    * ``gui="qt"`` -- calls `launch` (blocking; returns ``None`` after
      the window is closed).

    Parameters
    ----------
    viewer : Viewer or OrthoViewer
        The viewer whose ``gui`` attribute selects the dispatch target.
    layout : Layout
        Declarative layout spec.
    fit : "ready", "immediate", "none" or None
        Camera-fit policy applied at startup.  ``None`` (the default) is
        ``"ready"``, except for a scene whose cameras were all placed from
        their models (a viewer loaded with ``from_file``), which keeps its
        view.
    on_ready : Callable[[], None] or None
        Optional zero-argument callback fired once every scene's startup data
        has committed to the GPU.

    Returns
    -------
    object
        `DisplayHandle` (or marimo renderable) for anywidget; ``None``
        for Qt (after the window closes).

    Raises
    ------
    ValueError
        If ``viewer.gui`` is ``"offscreen"``, which has no window to show.
    """
    from cellier.convenience._backend import backend_for

    # Resolving the backend is what refuses a gui with no widgets -- including
    # "offscreen" -- so the message lives in one place rather than once per
    # entry point.
    backend = backend_for(viewer.gui, lacks="no window to show", what="viewer.gui")
    if backend.name == "anywidget":
        return display(viewer, layout, fit=fit, on_ready=on_ready)
    launch(viewer, layout, fit=fit, on_ready=on_ready)
    return None


def _resolve_qt_window(layout_or_window: object, viewer: object) -> object:
    """Return a QMainWindow: render *layout_or_window* if it is a Layout."""
    from cellier.convenience.layout._spec import Layout

    if isinstance(layout_or_window, Layout):
        from cellier.convenience.layout._qt_renderer import render_qt

        return render_qt(layout_or_window, viewer)
    return layout_or_window


def _init_view(
    viewer: ViewerLike,
    *,
    fit: FitMode | None = None,
    on_ready: Callable[[], None] | None = None,
    stall_timeout: float | None = _DEFAULT_STALL_TIMEOUT,
) -> StartupTracker:
    """Arm startup (fit + reslice) for every scene, and track its progress.

    Supports both the single-scene `Viewer` (which exposes ``scene``)
    and the multi-panel `OrthoViewer` (which exposes ``scenes``).

    Startup runs in stages, each observable through the returned tracker and
    through ``viewer.startup_state``: the canvas must reach the front end,
    render a frame, load its data, and commit it.  Splitting them is what lets
    a viewer that never finishes say *where* it stopped instead of just
    staying blank.

    The fit policy controls how the camera is framed:

    * ``"ready"`` -- fit on the first frame, then **re-fit** once the
      scene's data has committed to the GPU.  Both fits use the box measured
      from the stores, so the re-fit moves the camera only when that box
      changed while the data loaded.
    * ``"immediate"`` — fit on the first frame only; no re-fit after load.
    * ``"none"`` — never fit; just trigger the initial data load.
    * ``None`` (default) — ``"none"`` for a scene whose canvases all show a
      camera placed from its model (a viewer restored with ``from_file``,
      or a camera model's pose assigned before startup), ``"ready"`` for
      every other scene.

    Any callbacks registered on the viewer via ``viewer.on_ready`` plus the
    *on_ready* argument fire once, after the *last* scene becomes ready.  A
    ``viewer.on_ready`` callback fires on the first startup only; it is
    dropped once it has run.  A scene removed (``remove_scene``) before it
    is ready counts as done, so the remaining scenes can still finish
    startup; removing the last one finishes it at once.

    Camera-driven reslicing is suppressed for the duration of the startup load
    and restored once every scene is ready.  Without this, the initial
    ``fit_camera`` (a camera jump, which reslices at once) would add a second
    reslice to the startup load.  The ``fit="ready"`` re-fit is the one
    exception: reslicing is on for it, so the re-fitted view loads.

    Parameters
    ----------
    viewer : Viewer or OrthoViewer
        The viewer to start.
    fit : "ready", "immediate", "none" or None
        Camera-fit policy; ``None`` picks it per scene, as above.
    on_ready : Callable[[], None] or None
        Fired once, after every scene is ready.
    stall_timeout : float or None
        Seconds to wait before declaring startup stalled.  ``None`` disables
        the check.  Requires a running asyncio loop; without one the timer is
        simply not armed, which is why the state is also readable directly.

    Returns
    -------
    StartupTracker
        Also attached to the viewer as ``_startup``.
    """
    controller = viewer.controller
    scenes = getattr(viewer, "scenes", None)
    scene_items = (
        list(scenes.items()) if scenes is not None else [("scene", viewer.scene)]
    )

    tracker = StartupTracker([key for key, _ in scene_items])
    viewer._startup = tracker

    # Read when startup finishes, not now: ``viewer.on_ready`` may add to it
    # or cancel from it until then.
    ready_callbacks = getattr(viewer, "_ready_callbacks", None)

    # Suppress camera reslices during startup (the initial ``fit_camera`` is a
    # jump, which reslices) so the initial load is a single generation;
    # restore the prior setting once ready.
    prev_reslice_enabled = controller.camera_reslice_enabled
    controller.camera_reslice_enabled = False

    aggregate = {"remaining": len(scene_items), "fired": False}

    removal_handles: list[SubscriptionHandle] = []

    def _finish_startup() -> None:
        aggregate["fired"] = True
        for handle in removal_handles:
            handle.unsubscribe()
        controller.camera_reslice_enabled = prev_reslice_enabled
        if ready_callbacks is not None:
            fire_registered(ready_callbacks, once=True)
        if on_ready is not None:
            on_ready()

    def _scene_ready(key: str) -> None:
        tracker.advance(key, StartupState.READY)
        aggregate["remaining"] -= 1
        if aggregate["remaining"] <= 0 and not aggregate["fired"]:
            _finish_startup()

    def _scene_removed(key: str) -> None:
        # Removing a scene cancels its readiness callback, so a scene removed
        # before it was ready would hold startup open for good: no
        # ``on_ready``, and camera reslicing left off for every other scene.
        # Count it as done instead.  (Only armed until startup finishes.)
        counted = tracker.scene_states.get(key) is StartupState.READY
        tracker.forget(key)
        if not counted:
            aggregate["remaining"] -= 1
            if aggregate["remaining"] <= 0 and not aggregate["fired"]:
                _finish_startup()

    if not scene_items:
        _finish_startup()
        return tracker

    for key, scene in scene_items:
        removal_handles.append(
            controller.on_scene_removed(
                scene.id,
                lambda _event, k=key: _scene_removed(k),
                owner_id=controller._id,
            )
        )

    for key, scene in scene_items:
        canvas_ids = controller.get_canvas_ids(scene.id)
        if not canvas_ids:
            # No canvas attached yet; still load data so the model is populated.
            tracker.advance(key, StartupState.LOADING)
            controller.on_scene_ready(scene.id, lambda k=key: _scene_ready(k))
            continue

        tracker.set_canvas(key, canvas_ids[0])
        tracker.advance(key, StartupState.WAITING_FOR_CANVAS)

        def _start(s=scene, k=key) -> None:
            tracker.advance(k, StartupState.LOADING)
            scene_fit = fit
            if scene_fit is None:
                placed = all(
                    controller._canvas_camera_placed(cid)
                    for cid in controller.get_canvas_ids(s.id)
                )
                scene_fit = "none" if placed else "ready"
            if scene_fit != "none":
                controller.fit_camera(s.id)

            def _ready(s=s, k=k, scene_fit=scene_fit) -> None:
                if scene_fit == "ready":
                    # The re-fit is a camera jump, and a jump reslices at
                    # once or not at all: with camera reslicing still off
                    # the re-fitted view would never load.  So it is on for
                    # the re-fit, and off again until every scene is ready.
                    controller.camera_reslice_enabled = prev_reslice_enabled
                    try:
                        controller.fit_camera(s.id)
                    finally:
                        controller.camera_reslice_enabled = False
                _scene_ready(k)

            controller.on_scene_ready(s.id, _ready)

        # ``start=_start`` is bound now, not looked up later: ``_connected``
        # runs asynchronously, and without the default it would close over the
        # loop variable and every scene would run the *last* scene's starter.
        def _connected(cid=canvas_ids[0], k=key, start=_start) -> None:
            tracker.advance(k, StartupState.WAITING_FOR_FRAME)
            controller.on_canvas_first_frame(cid, start, owner_id=controller._id)

        # Two steps, not one: the canvas has to be live before a frame can be
        # asked of it, and on the anywidget backend that can be a long wait --
        # or never.  Splitting them is what lets a stalled viewer say which of
        # the two it is stuck on.
        controller.on_canvas_connected(
            canvas_ids[0], _connected, owner_id=controller._id
        )

    _arm_stall_timer(tracker, stall_timeout)
    return tracker


def _arm_stall_timer(tracker: StartupTracker, timeout: float | None) -> None:
    """Mark *tracker* stalled if startup has not finished within *timeout*.

    Best-effort: it needs a running asyncio loop, and there is not always one
    (a bare script before ``QtAsyncio.run``, a synchronous test).  Without a
    loop the timer is skipped rather than raising -- the state stays readable
    either way, which is the point of it being a property rather than only a
    callback.
    """
    if not timeout:
        return
    import asyncio

    async def _watch() -> None:
        await asyncio.sleep(timeout)
        tracker.mark_stalled()

    try:
        asyncio.get_running_loop().create_task(_watch())
    except RuntimeError:
        return
