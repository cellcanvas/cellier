"""CanvasView — owns one rendered canvas with camera and controller."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier._state import CameraState
from cellier.events._events import (
    CameraChangedEvent,
    CanvasConnectedEvent,
    CanvasSizeChangedEvent,
    FrameRenderedEvent,
    PlaneGizmoMovedEvent,
    _CameraControllerEvent,
)
from cellier.logging import _CAMERA_LOGGER
from cellier.render._cellier_blender import (
    NORMAL_TARGET,
    OUTLINE_ID_TARGET,
    ensure_extra_targets,
    install_cellier_blender,
)
from cellier.render._outline import OutlinePass
from cellier.render._pick_buffer import enable_pick_texture_binding
from cellier.render._requests import DimsState, ReslicingRequest
from cellier.render._ssao import SSAOPass
from cellier.render._temporal_accumulation import TemporalAccumulationPass
from cellier.render.visuals._plane_gizmo import GFXPlaneGizmo, drop_stale_gizmo_capture

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget

    from cellier.events._bus import EventBus
    from cellier.render._config import AmbientOcclusionConfig
    from cellier.render.visuals._canvas_overlay import GFXCanvasOverlay
    from cellier.transform import RegionSelection


def _no_draw() -> None:
    """Draw callback installed on a closed canvas; holds no reference to a view."""


def _detach_renderer_events(canvas: object, renderer: gfx.WgpuRenderer) -> None:
    """Remove the renderer's ``convert_event`` handlers from *canvas*.

    ``WgpuRenderer.disable_events`` cannot do this: rendercanvas removes a
    handler by identity (``cb is not callback``), and ``self.convert_event``
    builds a fresh bound-method object on every access, so it never matches
    the one ``enable_events`` registered.  Instead find the registered objects
    and hand *those* back to ``remove_event_handler``.
    """
    # A QRenderWidget forwards its events to an inner widget.
    emitter = getattr(getattr(canvas, "_subwidget", canvas), "_events", None)
    handlers = getattr(emitter, "_event_handlers", None)
    if handlers is None:
        return
    for event_type, entries in list(handlers.items()):
        for _order, callback in list(entries):
            if getattr(callback, "__self__", None) is renderer:
                canvas.remove_event_handler(callback, event_type)


class CanvasView:
    """Owns one rendered canvas: widget, renderer, camera, and controller.

    Responsible for rendering one scene from one camera viewpoint.
    ``CanvasView`` does not hold a direct reference to the scene graph;
    instead it receives a ``get_scene_fn`` callable that is invoked each
    frame so ownership of the scene stays with ``SceneManager``.

    Camera change detection is implemented by comparing a cached
    ``CameraState`` snapshot each frame in ``_draw_frame``.  The
    ``_applying_model_state`` flag suppresses detection during
    programmatic camera updates to prevent feedback loops.

    The view drives its pygfx camera controllers itself
    (``auto_update=False``): each frame it ticks the active controller and
    applies the state it returns, **before** the comparison.  A moved camera
    is therefore detected in the frame that draws it, and whether the
    controller is still driving the camera (a drag held, its damped tail, a
    wheel or key animation) is known through public pygfx API.  Changes of
    that answer are reported as ``_CameraControllerEvent``.

    Attributes
    ----------
    camera_moving : bool
        Whether this canvas's camera is in motion, set by the controller's
        camera tracker through ``RenderManager.set_camera_moving``.  Read by
        per-frame draw choices (a visual that draws coarse during motion).

    Parameters
    ----------
    canvas_id : UUID
        Unique identifier for this canvas.
    scene_id : UUID
        ID of the scene this canvas renders.
    get_scene_fn : Callable[[UUID], gfx.Scene]
        Called each frame to retrieve the current scene.  Provided by
        ``RenderManager`` at construction time.
    dim : str
        Scene dimensionality: ``"2d"`` or ``"3d"``.  Controls which
        camera type and interaction controller are used.
    parent : QWidget or None
        Parent widget for the underlying ``QRenderWidget``.
    fov : float
        Vertical field of view in degrees (3D perspective only).
    depth_range : tuple[float, float]
        Near and far clip distances ``(near, far)``.
    outline_enabled : bool
        When ``True``, install a blender carrying the ``outline_id`` render
        target so label outlines have a per-pixel label key.  Costs 4 bytes
        per pixel.  Deciding here is the cheap path, not the only one:
        :meth:`ensure_render_targets` adds it later for the price of one
        recompile frame.
    ambient_occlusion_enabled : bool
        When ``True``, install a blender carrying the ``normal`` render
        target so cellier's volume shaders can hand the ambient occlusion
        pass a real surface normal instead of one reconstructed from
        depth.  Costs 8 bytes per pixel, and is addable later by the same
        route as *outline_enabled*.
    """

    def __init__(
        self,
        canvas_id: UUID,
        scene_id: UUID,
        get_scene_fn: Callable[[UUID], gfx.Scene],
        dim: str = "3d",
        parent: QWidget | None = None,
        fov: float = 70.0,
        depth_range: tuple[float, float] = (1.0, 8000.0),
        event_bus: EventBus | None = None,
        gui: str = "qt",
        size: tuple[int, int] | None = None,
        outline_enabled: bool = False,
        ambient_occlusion_enabled: bool = False,
    ) -> None:
        self._canvas_id = canvas_id
        self._scene_id = scene_id
        self._get_scene_fn = get_scene_fn
        self._applying_model_state: bool = False
        self._id: UUID = uuid4()
        self._dim = dim
        self._event_bus: EventBus | None = event_bus
        self._camera_dirty: bool = False
        # Set by ``invalidate_accumulation``; consumed at the top of
        # ``_draw_frame``.  A flag rather than a direct ``reset()`` so the
        # discard is guaranteed to land *before* the next render rather
        # than racing a frame the backend has already queued.
        self._accum_dirty: bool = False
        self._tick_visuals_fn: Callable[[], None] | None = None
        self._closed: bool = False
        # Whether the active camera controller had a running action in the
        # last frame drawn; see ``_draw_frame``.
        self._driving: bool = False
        self.camera_moving: bool = False
        # A capture canvas (a screenshot): it draws the finest level whatever
        # the scene's interaction state.
        self.is_capture: bool = False
        # True for the length of ``_draw_frame``: planning must not run
        # inside a draw, so the controller queues reslices it detects there.
        self._drawing: bool = False
        self._resize_filter: object | None = None

        self._fov = fov
        self._depth_range = depth_range
        self._gui = gui
        # Set by ``_on_canvas_connected`` on the first sign of a live
        # front end; see ``CanvasConnectedEvent``.
        self._connected = False
        self._size = size

        self._canvas = self._create_canvas(parent, gui=gui, size=size)
        self._renderer = gfx.WgpuRenderer(self._canvas)

        # The pick texture ships without TEXTURE_BINDING, and its usage can
        # only be raised before the first draw -- so the grant happens here
        # unconditionally, even though outlines default to off.  Deferring it
        # until outlines are switched on would be too late.  A False result
        # leaves the outline pass installed but permanently a passthrough;
        # RenderManager warns if outlines are then requested.
        # The extra render targets are installed here for the canvases that
        # opted in, so a viewer configured up front pays no recompile.  A
        # canvas that did not opt in can still gain them later through
        # ``ensure_render_targets``, which costs one recompile frame.
        # Canvases that never enable either feature keep the stock blender
        # and pay nothing: ``outline_id`` costs 4 bytes per pixel, and
        # ``normal`` 8.
        #
        # This runs *before* the pick grant: installing replaces the whole
        # blender, so granting first would throw the grant away.
        # ``install_cellier_blender`` also copies usage bits across, so the
        # order is belt-and-braces rather than load-bearing.
        extra_targets: list[str] = []
        if outline_enabled:
            extra_targets.append(OUTLINE_ID_TARGET)
        if ambient_occlusion_enabled:
            extra_targets.append(NORMAL_TARGET)
        installed = (
            install_cellier_blender(self._renderer, extra_targets)
            if extra_targets
            else False
        )
        self._outline_id_available: bool = installed and outline_enabled
        self._normal_target_available: bool = installed and ambient_occlusion_enabled

        self._outline_available: bool = enable_pick_texture_binding(self._renderer)
        self._visual_lut_sync_fn: Callable[[], None] | None = None

        self._wire_resize_event(gui)

        # Both camera/controller pairs are created upfront so toggling only
        # requires enabling/disabling — no construction or destruction.
        self._camera_3d = gfx.PerspectiveCamera(fov, 16 / 9, depth_range=depth_range)
        # ``auto_update=False``: the controllers still turn input into
        # actions, but ``_draw_frame`` ticks them and applies the camera
        # state, and ``_on_controller_input`` requests the draws.
        self._controller_3d = gfx.OrbitController(
            camera=self._camera_3d, register_events=self._renderer, auto_update=False
        )
        self._camera_2d = gfx.OrthographicCamera(maintain_aspect=True)
        self._controller_2d = gfx.PanZoomController(
            camera=self._camera_2d, register_events=self._renderer, auto_update=False
        )
        self._renderer.add_event_handler(
            self._on_controller_input,
            "pointer_down",
            "pointer_up",
            "pointer_move",
            "wheel",
            "key_down",
            "key_up",
        )

        # Activate the initial dim; disable the other controller.
        if dim == "2d":
            self._camera = self._camera_2d
            self._controller = self._controller_2d
            self._controller_3d.enabled = False
        else:
            self._camera = self._camera_3d
            self._controller = self._controller_3d
            self._controller_2d.enabled = False

        # Track which dims have been fitted to the scene.
        self._fitted: set[str] = set()

        self._accum_pass = TemporalAccumulationPass(alpha=0.2)
        if dim == "2d":
            self._accum_pass.enabled = False

        # Ambient occlusion runs first of all: it darkens the *fill*, and
        # both passes after it must see that fill already darkened.  Ahead
        # of the outline because an outline is a UI annotation rather than
        # a lit surface, and several render tests assert its palette
        # colours by exact match.  Ahead of accumulation because the EMA
        # there averages away the per-frame kernel rotation for free,
        # which is what lets the sample count sit at 16 instead of 64.
        #
        # The pass is 3D only: a 2D cellier scene is an image plane at
        # near-constant depth under an orthographic camera, where the
        # reconstructed normal is constant and the occlusion comes out
        # uniform.  ``_ssao_requested`` keeps the configured state apart
        # from that restriction, so a 2D -> 3D toggle restores whatever
        # the config asked for rather than switching the pass on.
        self._ssao_requested: bool = False
        self._ssao_pass = SSAOPass(self._renderer, lambda: self._camera)
        self._ssao_pass.enabled = False

        # Outlines composite *before* accumulation: the volume raymarcher
        # jitters per frame, so silhouette pixels shift sub-pixel between
        # frames and the EMA turns that into a free antialiased edge rather
        # than a flicker.  DDAA stays last so it antialiases the outline.
        # The pass starts disabled and pygfx's flush() skips disabled passes
        # entirely, so this costs nothing until outlines are switched on.
        self._outline_pass = OutlinePass(self._renderer)
        self._renderer.effect_passes = (
            self._ssao_pass,
            self._outline_pass,
            self._accum_pass,
            *self._renderer.effect_passes,
        )

        self._last_camera_state: CameraState = self.capture_camera_state()
        self._overlays: list[GFXCanvasOverlay] = []
        # Plane gizmos (gizmo design v2, 5.1), and the ones whose proxy moved
        # since the last frame.  The handlers are kept as attributes because
        # the canvas removes a handler by identity.
        self._plane_gizmos: dict[UUID, GFXPlaneGizmo] = {}
        self._gizmo_moves_pending: set[UUID] = set()
        self._gizmo_before_draw_handler = self._report_gizmo_moves
        self._gizmo_press_handler = self._on_canvas_press
        # Ahead of the renderer's own handler: while another canvas holds a
        # stale gizmo capture, pygfx drops the press before any handler of
        # the renderer runs.
        self._canvas.add_event_handler(
            self._gizmo_press_handler, "pointer_down", order=-10
        )
        # A hold on drawing (see ``hold_draws``): how long, its test, and
        # the deadline set by the first frame skipped.
        self._hold_seconds: float = 0.0
        self._hold_waiting: Callable[[], bool] | None = None
        self._hold_deadline: float | None = None
        self._canvas.request_draw(self._draw_frame)

    def _create_canvas(
        self,
        parent: QWidget | None,
        *,
        gui: str = "qt",
        size: tuple[int, int] | None = None,
    ) -> object:
        """Create the render canvas widget.

        This is the single seam where the GUI backend is selected.  The
        ``rendercanvas`` backend imports are deferred to here so that importing
        ``CanvasView`` (and therefore the ``cellier`` package) does not pull in
        a Qt toolkit or anywidget; the chosen backend is only required once a
        canvas is actually constructed.

        Parameters
        ----------
        parent : QWidget or None
            Parent widget for the underlying ``QRenderWidget``.  Ignored for
            the anywidget backend (notebook canvases are not laid out by a
            Qt parent).
        gui : str
            Which GUI toolkit to target: ``"qt"``, ``"anywidget"`` or
            ``"offscreen"``.
        size : tuple[int, int] or None
            Initial CSS pixel size for the anywidget canvas, or the exact
            framebuffer size for the offscreen canvas.  Defaults to
            ``(600, 600)`` when ``None``.  Ignored for the Qt backend, which
            is sized by its parent layout.

        Returns
        -------
        object
            The render canvas widget (a ``QRenderWidget`` for ``"qt"``, an
            ``AnywidgetRenderCanvas`` for ``"anywidget"``, or an
            ``OffscreenRenderCanvas`` for ``"offscreen"``).

        Raises
        ------
        ValueError
            If *gui* is not ``"qt"``, ``"anywidget"`` or ``"offscreen"``.
        """
        if gui == "offscreen":
            # No window, no scheduler, no event loop: ``canvas.draw()`` runs
            # the draw callback synchronously and returns the presented
            # frame.  That is what makes a capture reproducible -- see
            # ``cellier.render._capture``.
            #
            # ``pixel_ratio=1`` so logical size == physical size == the shape
            # of the returned array.  Any other value would silently make the
            # output depend on a display the canvas does not have.
            from rendercanvas.offscreen import RenderCanvas as OffscreenRenderCanvas

            return OffscreenRenderCanvas(size=size or (600, 600), pixel_ratio=1)
        if gui == "qt":
            from rendercanvas.qt import QRenderWidget

            return QRenderWidget(parent=parent, update_mode="continuous")
        elif gui == "anywidget":
            # rendercanvas's anywidget backend (the older `jupyter` backend is
            # deprecated).  Anywidget canvases are not laid out by a parent;
            # `parent` is ignored.  `size` still sets the initial height and
            # the physical resolution rendercanvas starts with, but width is
            # made responsive below so the canvas fills whatever flex column
            # cellier's layout gives it (see AnywidgetBox's `min_width`, which is
            # set from this same `size` at the convenience layer).
            from rendercanvas.anywidget import RenderCanvas

            class _CellierAnywidgetCanvas(RenderCanvas):
                """Notifies ``_cellier_on_resize`` on every real browser resize.

                rendercanvas's ``_css_width`` traitlet is write-only from
                Python (nothing on the JS side writes a new value back into
                it when the *browser* resizes the canvas), so observing it
                (the previous approach) misses organic resizes entirely.  The
                "resize" custom message, handled here, fires on every actual
                ``ResizeObserver`` event and carries the real physical size,
                from which rendercanvas itself derives ``_size_info``
                (including ``logical_size``, already corrected for device
                pixel ratio) -- reused here instead of re-parsing a CSS
                string.
                """

                _cellier_closing = False

                def _rfb_handle_msg(self, widget, content, buffers) -> None:
                    super()._rfb_handle_msg(widget, content, buffers)
                    # Any inbound message means the browser has mounted this
                    # canvas and can draw -- which is the one thing Python
                    # cannot otherwise know, and which it may wait forever for.
                    cb = getattr(self, "_cellier_on_connected", None)
                    if cb is not None:
                        cb()
                    if content.get("type") == "resize":
                        cb = getattr(self, "_cellier_on_resize", None)
                        if cb is not None:
                            w, h = self._size_info["logical_size"]
                            cb(round(w), round(h))

                def _rc_close(self) -> None:
                    # rendercanvas's anywidget backend closes re-entrantly:
                    # _rc_close dispatches a synthetic "close" message whose
                    # handler calls close() again, and it only sets _is_closed
                    # *after* that dispatch -- while close() never checks it
                    # anyway.  Left alone this recurses until the stack blows.
                    if self._cellier_closing:
                        return
                    self._cellier_closing = True
                    super()._rc_close()
                    # The canvas is an ipywidgets widget, so it owns a
                    # ``Layout`` widget registered in the same process-global
                    # table and not released with its owner -- see
                    # ``cellier.gui.anywidget._teardown``.
                    from cellier.gui.anywidget._teardown import close_aux_widgets

                    close_aux_widgets(self)

            canvas = _CellierAnywidgetCanvas(
                size=size or (600, 600), update_mode="continuous"
            )
            canvas.set_css_width("100%")
            return canvas
        raise ValueError(
            f"Unknown gui {gui!r}. Expected 'qt', 'anywidget' or 'offscreen'."
        )

    def _wire_resize_event(self, gui: str) -> None:
        """Hook the backend resize notification to emit CanvasSizeChangedEvent.

        Qt: installs a resizeEvent override on the QRenderWidget via an
        event filter on a QObject proxy so we don't need to subclass the
        widget.  anywidget: the canvas is a ``_CellierAnywidgetCanvas``
        (see ``_create_canvas``), which calls ``_cellier_on_resize`` on every
        real browser resize.  offscreen: nothing to hook -- the canvas has no
        window and never resizes organically, so every size change is a
        deliberate ``set_logical_size`` by the caller who already knows about
        it.
        """
        if gui == "offscreen":
            return
        if gui == "qt":
            from PySide6.QtCore import QEvent, QObject

            canvas = self._canvas

            class _ResizeFilter(QObject):
                def __init__(self_f, parent_view: CanvasView) -> None:
                    super().__init__()
                    self_f._view = parent_view

                def eventFilter(self_f, obj, event) -> bool:
                    if event.type() == QEvent.Type.Resize:
                        s = event.size()
                        self_f._view._on_canvas_resize(s.width(), s.height())
                    elif event.type() == QEvent.Type.Show:
                        self_f._view._on_canvas_connected()
                    return False

            self._resize_filter = _ResizeFilter(self)
            canvas.installEventFilter(self._resize_filter)

        elif gui == "anywidget":
            self._canvas._cellier_on_resize = self._on_canvas_resize
            self._canvas._cellier_on_connected = self._on_canvas_connected

    def _on_canvas_connected(self) -> None:
        """Emit ``CanvasConnectedEvent`` once, on the first sign of a front end.

        Called from each backend's own liveness signal: a Qt ``Show`` event,
        or the first message the anywidget canvas receives from the browser.
        Both can fire repeatedly, so this reports only the first.
        """
        if self._connected:
            return
        self._connected = True
        if self._event_bus is not None:
            self._event_bus.emit(
                CanvasConnectedEvent(
                    source_id=self._canvas_id,
                    canvas_id=self._canvas_id,
                    gui=self._gui,
                )
            )

    @property
    def connected(self) -> bool:
        """Whether this canvas's front end has reported itself live."""
        return self._connected

    def _on_canvas_resize(self, width: int, height: int) -> None:
        """Emit CanvasSizeChangedEvent; called by both backend resize hooks."""
        if self._event_bus is not None:
            self._event_bus.emit(
                CanvasSizeChangedEvent(
                    source_id=self._id,
                    canvas_id=self._canvas_id,
                    width=width,
                    height=height,
                )
            )

    @property
    def canvas_id(self) -> UUID:
        """Unique identifier for this canvas."""
        return self._canvas_id

    @property
    def scene_id(self) -> UUID:
        """ID of the scene this canvas renders."""
        return self._scene_id

    @property
    def widget(self) -> object:
        """The render canvas element to embed in the application layout.

        A ``QRenderWidget`` for the Qt backend or an ``AnywidgetRenderCanvas``
        for the anywidget backend.
        """
        return self._canvas

    def close(self) -> None:
        """Close the canvas, stopping its draw loop and releasing the GPU.

        Dropping the last Python reference to a ``CanvasView`` is *not* enough
        to reclaim it.  The canvas is a parentless (top-level) render widget,
        so the backend owns it and keeps it alive; through its draw callback
        and event filter it in turn pins this view, the ``WgpuRenderer``, and
        the whole object graph they reach.  Closing the canvas is what breaks
        that chain, after which normal refcounting reclaims everything.

        Safe to call more than once, and safe when the GUI backend has already
        destroyed the canvas itself (e.g. the user closed the window).
        """
        if self._closed:
            return
        self._closed = True
        self._overlays.clear()
        for gizmo_id in list(self._plane_gizmos):
            self.remove_plane_gizmo(gizmo_id)
        self._canvas.remove_event_handler(self._gizmo_press_handler, "pointer_down")

        # Break the canvas -> view/renderer references before closing it.  The
        # canvas holds this view's draw callback and, through its event
        # emitter, the renderer's ``convert_event`` handler.  Closing a Qt
        # canvas leaves its Python wrapper behind, and cycles through a shiboken
        # wrapper are invisible to the garbage collector, so without this the
        # whole renderer (and its GPU resources) is never freed.
        _detach_renderer_events(self._canvas, self._renderer)
        self._canvas.request_draw(_no_draw)

        try:
            if self._resize_filter is not None:
                self._canvas.removeEventFilter(self._resize_filter)
            self._canvas.close()
        except RuntimeError:
            # Qt already deleted the underlying C++ widget; nothing to release.
            pass
        finally:
            self._resize_filter = None

    def capture_reslicing_request(
        self,
        dims_state: DimsState,
        selection: RegionSelection | None = None,
        target_visual_ids: frozenset[UUID] | None = None,
    ) -> ReslicingRequest:
        """Snapshot the current camera state into a ReslicingRequest.

        All array fields are copied.  Screen size is read from the canvas
        at call time and baked into the returned request.

        Parameters
        ----------
        dims_state : DimsState
            Current dimension display state.
        selection : RegionSelection or None
            The region this canvas is showing, built by the controller from
            the scene's dims and this canvas's rendered system.
        target_visual_ids : frozenset[UUID] or None
            ``None`` reslices all visuals in the scene.

        Returns
        -------
        ReslicingRequest
            Fully populated snapshot with independent array copies.
        """
        screen_w, screen_h = self._canvas.get_logical_size()

        if self._dim == "2d":
            return self._capture_orthographic(
                dims_state, selection, target_visual_ids, screen_w, screen_h
            )
        return self._capture_perspective(
            dims_state, selection, target_visual_ids, screen_w, screen_h
        )

    def _capture_perspective(
        self,
        dims_state: DimsState,
        selection: RegionSelection | None,
        target_visual_ids: frozenset[UUID] | None,
        screen_w: float,
        screen_h: float,
    ) -> ReslicingRequest:
        """Build a ReslicingRequest for a perspective camera."""
        frustum = np.asarray(self._camera.frustum, dtype=np.float64)
        return ReslicingRequest(
            camera_type="perspective",
            camera_pos=np.array(self._camera.world.position, dtype=np.float64),
            frustum_corners=frustum.copy(),
            fov_y_rad=float(np.radians(self._camera.fov)),
            screen_size_px=(float(screen_w), float(screen_h)),
            world_extent=(0.0, 0.0),
            dims_state=dims_state,
            selection=selection,
            request_id=uuid4(),
            scene_id=self._scene_id,
            canvas_id=self._canvas_id,
            target_visual_ids=target_visual_ids,
        )

    def _capture_orthographic(
        self,
        dims_state: DimsState,
        selection: RegionSelection | None,
        target_visual_ids: frozenset[UUID] | None,
        screen_w: float,
        screen_h: float,
    ) -> ReslicingRequest:
        """Build a ReslicingRequest for an orthographic camera.

        Computes the actual visible world extent accounting for the
        canvas aspect ratio.  The OrthographicCamera exposes ``width``
        and ``height`` which define the *minimum* visible extent; the
        actual extent is expanded in one dimension to match the canvas
        aspect ratio. This is because maintain_aspect is set to True.
        (see pygfx OrthographicCamera docstring)
        """
        cam = self._camera
        vw = screen_w if screen_w > 0 else 800.0
        vh = screen_h if screen_h > 0 else 600.0
        canvas_aspect = vw / vh

        cam_w = cam.width if cam.width > 0 else 1.0
        cam_h = cam.height if cam.height > 0 else 1.0
        cam_aspect = cam_w / cam_h

        if canvas_aspect >= cam_aspect:
            world_height = cam_h
            world_width = cam_h * canvas_aspect
        else:
            world_width = cam_w
            world_height = cam_w / canvas_aspect

        return ReslicingRequest(
            camera_type="orthographic",
            camera_pos=np.array(cam.world.position, dtype=np.float64),
            frustum_corners=np.zeros((2, 4, 3), dtype=np.float64),
            fov_y_rad=0.0,
            screen_size_px=(float(vw), float(vh)),
            world_extent=(float(world_width), float(world_height)),
            dims_state=dims_state,
            selection=selection,
            request_id=uuid4(),
            scene_id=self._scene_id,
            canvas_id=self._canvas_id,
            target_visual_ids=target_visual_ids,
        )

    def set_depth_range(self, depth_range: tuple[float, float]) -> None:
        """Set the active camera near/far clip distances.

        Parameters
        ----------
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances in world units.
        """
        self._camera.depth_range = depth_range

    def set_depth_range_for_dim(
        self, dim: str, depth_range: tuple[float, float]
    ) -> None:
        """Set the near/far clip distances on the 2D or 3D camera.

        Unlike :meth:`set_depth_range`, this targets a specific camera
        regardless of which is currently active.  Both the 2D orthographic
        and 3D perspective cameras are created up front (see ``__init__``),
        so the reserve camera must have its depth range set independently —
        otherwise it keeps the active camera's range, which for a 2D->3D
        toggle leaves the perspective camera with an invalid (e.g. negative)
        near plane and renders nothing.

        Parameters
        ----------
        dim : str
            ``"2d"`` or ``"3d"``.
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances in world units.
        """
        camera = self._camera_2d if dim == "2d" else self._camera_3d
        camera.depth_range = depth_range

    def show_object(self, scene: gfx.Scene) -> bool:
        """Fit the camera to the scene bounding box, if there is one.

        A scene with nothing in it has no bounding sphere, and pygfx raises
        rather than guessing.  That happens for real: switching a scene's
        displayed axes rebuilds its visuals' geometry and fits the camera
        *before* the reslice that fills them has committed, so for one moment
        the scene is empty.  Refusing to fit -- and, crucially, not marking
        the dim as fitted -- lets the caller try again once data arrives.

        Parameters
        ----------
        scene : gfx.Scene
            The scene to fit the camera to.

        Returns
        -------
        bool
            ``True`` if the camera was fitted.  ``False`` if the scene had no
            bounds yet, in which case nothing was changed.
        """
        if scene.get_world_bounding_sphere() is None:
            return False
        if self._dim == "2d":
            self._camera.show_object(scene, view_dir=(0, 0, -1), up=(0, 1, 0))
        else:
            self._camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
        self._fitted.add(self._dim)
        return True

    @property
    def camera(self) -> gfx.Camera:
        """The active pygfx camera for this canvas."""
        return self._camera

    def add_overlay(self, overlay: GFXCanvasOverlay) -> None:
        """Attach a screen-space overlay to this canvas.

        The overlay is rendered as an additional post-pass on top of the
        main scene each frame.  Multiple overlays are rendered in insertion
        order.

        Parameters
        ----------
        overlay : GFXCanvasOverlay
            The render-layer overlay to attach.
        """
        self._overlays.append(overlay)

    def remove_overlay(self, overlay: GFXCanvasOverlay) -> None:
        """Detach a screen-space overlay from this canvas.

        Parameters
        ----------
        overlay : GFXCanvasOverlay
            The render-layer overlay to detach.  An overlay that is not
            attached is ignored.
        """
        if overlay in self._overlays:
            self._overlays.remove(overlay)

    # -- plane gizmos (gizmo design v2, 5.1) ----------------------------------

    def add_plane_gizmo(
        self, gizmo_id: UUID, point, normal, *, screen_size: float = 100.0
    ) -> GFXPlaneGizmo:
        """Add a plane gizmo to this canvas, placed at a pose.

        The gizmo is drawn in 3D only, after the scene and under the canvas
        overlays.  Its drags are reported as ``PlaneGizmoMovedEvent``.

        Parameters
        ----------
        gizmo_id : UUID
            Names the gizmo on its events.
        point, normal : array-like
            Where to place it, in rendered ``(x, y, z)``: a point on the
            plane and the plane's normal.
        screen_size : float
            The gizmo's size on screen, in logical pixels.

        Returns
        -------
        GFXPlaneGizmo

        Raises
        ------
        ValueError
            If the canvas already has a gizmo with that id.
        """
        if gizmo_id in self._plane_gizmos:
            raise ValueError(f"Canvas {self._canvas_id} already has gizmo {gizmo_id}.")
        gizmo = GFXPlaneGizmo(
            gizmo_id,
            self._renderer,
            self._camera_3d,
            on_grab=self._on_gizmo_grab,
            on_move=self._on_gizmo_move,
            on_release=self._on_gizmo_release,
            screen_size=screen_size,
        )
        gizmo.set_pose(point, normal)
        if not self._plane_gizmos:
            # Ahead of the scheduler's commit hook on the same event, so a
            # frame plans, then commits, then draws.
            self._canvas.add_event_handler(
                self._gizmo_before_draw_handler, "before_draw", order=-10
            )
        self._plane_gizmos[gizmo_id] = gizmo
        self.request_draw()
        return gizmo

    def remove_plane_gizmo(self, gizmo_id: UUID) -> None:
        """Remove a plane gizmo; a drag in progress is ended and reported.

        A gizmo this canvas does not have is ignored.
        """
        gizmo = self._plane_gizmos.get(gizmo_id)
        if gizmo is None:
            return
        gizmo.cancel_drag()
        del self._plane_gizmos[gizmo_id]
        self._gizmo_moves_pending.discard(gizmo_id)
        gizmo.close()
        if not self._plane_gizmos:
            self._canvas.remove_event_handler(
                self._gizmo_before_draw_handler, "before_draw"
            )
        if not self._closed:
            self.request_draw()

    def get_plane_gizmo(self, gizmo_id: UUID) -> GFXPlaneGizmo:
        """The plane gizmo with that id.

        Raises
        ------
        KeyError
            If this canvas has no such gizmo.
        """
        return self._plane_gizmos[gizmo_id]

    @property
    def dim(self) -> str:
        """``"2d"`` or ``"3d"``: which camera this canvas draws with."""
        return self._dim

    def orbit_point(self) -> tuple[float, float, float] | None:
        """The point the 3D camera orbits about, in rendered ``(x, y, z)``.

        ``None`` in 2D.  With no custom target, pygfx's
        ``OrbitController.target`` is the offset from the camera, so the
        camera's position is added (gizmo design v2, V5).
        """
        if self._dim != "3d":
            return None
        controller = self._controller_3d
        target = np.asarray(controller.target, dtype=np.float64)
        if controller._custom_target is None:
            target = target + np.asarray(
                self._camera_3d.world.position, dtype=np.float64
            )
        return float(target[0]), float(target[1]), float(target[2])

    def plane_gizmo_object_ids(self) -> set[int]:
        """The pygfx ids of every element of this canvas's plane gizmos."""
        ids: set[int] = set()
        for gizmo in self._plane_gizmos.values():
            ids |= gizmo.object_ids()
        return ids

    def _emit_gizmo(self, gizmo: GFXPlaneGizmo, phase: str) -> None:
        if self._event_bus is None:
            return
        point, normal = gizmo.pose()
        self._event_bus.emit(
            PlaneGizmoMovedEvent(
                source_id=self._canvas_id,
                canvas_id=self._canvas_id,
                gizmo_id=gizmo.gizmo_id,
                point=point,
                normal=normal,
                phase=phase,
                handle_kind=gizmo.handle_kind,
                handle_axis=gizmo.handle_axis,
            )
        )

    def _gizmo_changed(self) -> None:
        """The gizmo's pixels changed: draw it now, with no stale blend."""
        if self._closed:
            return
        # The user is dragging a handle and should see it follow, whatever
        # is loading: as for camera input.
        self.release_hold()
        self.request_draw()

    def _on_gizmo_grab(self, gizmo: GFXPlaneGizmo) -> None:
        self._emit_gizmo(gizmo, "start")
        self._gizmo_changed()

    def _on_gizmo_move(self, gizmo: GFXPlaneGizmo) -> None:
        # Several pointer moves arrive per frame and each report costs a
        # plan: the pose is reported once, ahead of the next draw.
        self._gizmo_moves_pending.add(gizmo.gizmo_id)
        self._gizmo_changed()

    def _on_gizmo_release(self, gizmo: GFXPlaneGizmo) -> None:
        # The last pose of a drag is never dropped, and does not wait for a
        # frame.
        if gizmo.gizmo_id in self._gizmo_moves_pending:
            self._gizmo_moves_pending.discard(gizmo.gizmo_id)
            self._emit_gizmo(gizmo, "move")
        self._emit_gizmo(gizmo, "end")
        self._gizmo_changed()

    def _report_gizmo_moves(self, _event=None) -> None:
        """Report each gizmo that moved since the last frame, once.

        A handler of the canvas's ``before_draw`` event: ahead of the draw
        callback, so the plan a report causes runs outside the draw and the
        accumulation reset it asks for lands in the frame about to be drawn.
        """
        if not self._gizmo_moves_pending:
            return
        pending, self._gizmo_moves_pending = self._gizmo_moves_pending, set()
        for gizmo_id in pending:
            gizmo = self._plane_gizmos.get(gizmo_id)
            if gizmo is not None:
                self._emit_gizmo(gizmo, "move")

    def _on_canvas_press(self, event) -> None:
        """A press in this canvas: drop a gizmo capture whose release was lost."""
        drop_stale_gizmo_capture(event)

    def invalidate_accumulation(self) -> None:
        """Discard the temporal accumulation history before the next frame.

        Call whenever the image that *should* be drawn changes, so the
        next frame is not an average with a picture that no longer
        applies.  Cheap and idempotent: it sets a flag that
        ``_draw_frame`` consumes, and the pass itself only zeroes a
        counter.

        Every content change needs this, not just the conspicuous ones.
        Hiding a visual is merely the case where the stale average is
        obvious; a colormap, clim, opacity, transform or data commit
        leaves the same residue and reads as sluggishness instead.
        """
        self._accum_dirty = True

    def request_draw(self) -> None:
        """Request a redraw of the canvas.

        Cellier calls this for content changes, so it also invalidates
        the accumulation history.  Idle and continuous redraws come from
        the backend's own scheduler and never reach here, which is what
        lets them keep accumulating.
        """
        self.invalidate_accumulation()
        self._canvas.request_draw(self._draw_frame)

    def request_frame(self) -> None:
        """Ask the canvas for a frame without discarding accumulation.

        For a frame whose content did not change: the next step of a camera
        controller's animation, or the still picture after a camera motion
        ended.  :meth:`request_draw` is the one for content changes.
        """
        if not self._closed:
            self._canvas.request_draw()

    def _on_controller_input(self, event) -> None:
        """Request a frame for input the camera controller will act on.

        With ``auto_update=False`` pygfx no longer requests draws for input.
        A hover (a move with no button held) drives nothing, so it draws
        nothing.  Only matters on an on-demand canvas; the Qt and anywidget
        canvases draw continuously.
        """
        if not self._controller.enabled:
            return
        if event.type == "pointer_move" and not event.buttons:
            return
        # The camera is about to move: draw it now, whatever is loading.
        self.release_hold()
        self.request_frame()

    def accept_camera_state(self) -> bool:
        """Take the camera's current state as the one already reported.

        Call after moving the pygfx camera programmatically.  The next draw
        then sees no difference, so the move is not mistaken for camera
        motion; the caller reports it instead.  The accumulation history is
        discarded, because the camera really did move.

        Returns
        -------
        bool
            ``True`` if the camera differs from the last reported state.
            ``False`` if nothing moved, in which case nothing is changed.
        """
        current = self.capture_camera_state()
        if current == self._last_camera_state:
            return False
        self._last_camera_state = current
        self.invalidate_accumulation()
        return True

    @property
    def last_camera_state(self) -> CameraState:
        """The camera state most recently reported or accepted."""
        return self._last_camera_state

    def set_camera_state(self, state: CameraState) -> None:
        """Apply a ``CameraState`` to the active pygfx camera.

        Only the fields of the active camera's kind are applied: a
        perspective camera takes ``fov``, an orthographic one ``extent``.
        ``up`` is not applied: it is the up direction the rotation already
        gives.  The caller reports the move; see :meth:`accept_camera_state`.

        Parameters
        ----------
        state : CameraState
            The state to apply.

        Raises
        ------
        ValueError
            If *state* is for the other kind of camera than the active one.
        """
        expected = "orthographic" if self._dim == "2d" else "perspective"
        if state.camera_type != expected:
            raise ValueError(
                f"Cannot apply a {state.camera_type!r} camera state to a "
                f"canvas showing {self._dim}: its camera is {expected}."
            )
        camera = self._camera
        if self._dim == "2d":
            camera.width, camera.height = state.extent
        else:
            camera.fov = state.fov
        camera.world.position = state.position
        camera.world.rotation = state.rotation
        camera.zoom = state.zoom
        camera.depth_range = state.depth_range

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire the EventBus after construction."""
        self._event_bus = event_bus

    def set_controller_enabled(self, enabled: bool) -> None:
        """Enable or disable the active camera controller for this canvas.

        ``self._controller`` already points to the currently active
        controller (``_controller_2d`` or ``_controller_3d`` depending on
        the canvas dim), so this correctly targets whichever type is in use.

        Parameters
        ----------
        enabled : bool
            False disables the controller (paint mode active).
            True restores normal camera interaction.
        """
        self._controller.enabled = enabled

    def switch_dim(self, new_dim: str) -> bool:
        """Switch the canvas between ``"2d"`` and ``"3d"`` rendering modes.

        Disables the current controller and enables the one for ``new_dim``.
        Camera pose is preserved across toggles.

        Parameters
        ----------
        new_dim : str
            ``"2d"`` or ``"3d"``.

        Returns
        -------
        bool
            ``True`` if this is the first time ``new_dim`` has been activated
            on this canvas (caller should call ``show_object`` to fit the
            camera).  ``False`` if the camera pose was already set by a
            previous visit.
        """
        if new_dim == self._dim:
            return False
        self._controller.enabled = False
        if new_dim == "2d":
            # Gizmos are not drawn in 2D, so a handle cannot be released.
            for gizmo in self._plane_gizmos.values():
                gizmo.cancel_drag()
            self._camera = self._camera_2d
            self._controller = self._controller_2d
            self._accum_pass.enabled = False
        else:
            self._camera = self._camera_3d
            self._controller = self._controller_3d
            self._accum_pass.enabled = True
        self._controller.enabled = True
        self._dim = new_dim
        self._apply_ssao_enabled()
        # Caching the state of the camera we just swapped *to* means the diff
        # in ``_draw_frame`` sees no change, even though the whole view did.
        # And on the way back to 3D the pass has been skipped for the entire
        # 2D excursion, so its history still holds the pre-excursion image.
        self._last_camera_state = self.capture_camera_state()
        self.invalidate_accumulation()
        first_visit = new_dim not in self._fitted
        return first_visit

    def apply_ambient_occlusion_config(self, config: AmbientOcclusionConfig) -> None:
        """Push an ``AmbientOcclusionConfig`` onto this canvas's occlusion pass.

        Parameters
        ----------
        config : AmbientOcclusionConfig
            The configuration to apply.  Its ``enabled`` flag is recorded
            as the *requested* state; the pass itself stays off while the
            canvas is in 2D.
        """
        self._ssao_pass.apply_config(config)
        self.set_ssao_enabled(config.enabled)

    def set_ssao_enabled(self, enabled: bool) -> None:
        """Request ambient occlusion on this canvas.

        Honoured only in 3D.  The request is remembered either way, so a
        canvas switched to 2D and back returns to the requested state.

        Parameters
        ----------
        enabled : bool
            Whether the caller wants the pass to run.
        """
        self._ssao_requested = bool(enabled)
        self._apply_ssao_enabled()

    def _apply_ssao_enabled(self) -> None:
        self._ssao_pass.enabled = self._ssao_requested and self._dim != "2d"

    def ensure_render_targets(
        self, *, outline: bool = False, ssao: bool = False
    ) -> None:
        """Add the render targets a feature needs, after construction.

        The targets are chosen at construction from the render config, which
        is right for a viewer that was configured up front and wrong for one
        where a user ticks the box later: a feature switched on afterwards
        would run without its target and quietly degrade -- ambient
        occlusion to normals reconstructed from depth, outlines to
        whole-object silhouettes with no per-label boundaries.  This adds
        the missing target instead, at the cost of one recompile frame.

        Safe to call repeatedly; it does nothing when the targets are
        already present, which is the common case.  **Not safe to call from
        inside a draw callback** -- see :func:`ensure_extra_targets`.

        Parameters
        ----------
        outline : bool
            Ensure the ``outline_id`` target, for per-label outlines.
        ssao : bool
            Ensure the ``normal`` target, for occlusion on raymarched
            isosurfaces.
        """
        wanted: list[str] = []
        if outline and not self._outline_id_available:
            wanted.append(OUTLINE_ID_TARGET)
        if ssao and not self._normal_target_available:
            wanted.append(NORMAL_TARGET)
        if not wanted:
            return

        if not ensure_extra_targets(self._renderer, wanted):
            # pygfx is not the expected shape.  The features still run,
            # against the fallbacks they were designed with.
            return
        if OUTLINE_ID_TARGET in wanted:
            self._outline_id_available = True
        if NORMAL_TARGET in wanted:
            self._normal_target_available = True
        # The new blender's textures do not exist yet, so the accumulated
        # history is an average with frames drawn against the old one.
        # ``request_draw`` invalidates it for us.
        self.request_draw()

    def set_scene_extent(self, diagonal: float) -> None:
        """Forward the scene bounding box diagonal to the occlusion pass.

        Parameters
        ----------
        diagonal : float
            Length of the scene bounding box diagonal, in scene units.
        """
        self._ssao_pass.set_scene_extent(diagonal)

    def capture_camera_state(self) -> CameraState:
        """Snapshot the current pygfx camera into a CameraState NamedTuple."""
        cam = self._camera
        pos = cam.world.position
        rot = cam.world.rotation  # quaternion (x, y, z, w)
        dr = cam.depth_range
        depth_range = (float(dr[0]), float(dr[1])) if dr is not None else (0.0, 0.0)

        if self._dim == "2d":
            return CameraState(
                camera_type="orthographic",
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                rotation=(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
                up=(0.0, 1.0, 0.0),
                fov=0.0,
                zoom=float(cam.zoom),
                extent=(float(cam.width), float(cam.height)),
                depth_range=depth_range,
            )
        else:
            up = cam.world.up
            return CameraState(
                camera_type="perspective",
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                rotation=(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
                up=(float(up[0]), float(up[1]), float(up[2])),
                fov=float(cam.fov),
                zoom=float(cam.zoom),
                extent=(0.0, 0.0),
                depth_range=depth_range,
            )

    def hold_draws(self, seconds: float, waiting: Callable[[], bool]) -> None:
        """Let the next frames be skipped while *waiting* is true.

        A frame that comes while *waiting* returns ``True`` is not rendered,
        so the canvas keeps showing what it last drew.  The first frame
        skipped starts the clock: frames are skipped for at most *seconds*
        from it, then one is drawn whatever *waiting* says.  The hold ends
        with the first frame drawn, or on camera input.  Asking again while
        frames are being skipped does not restart the clock, so no frame is
        delayed by more than *seconds*.

        Parameters
        ----------
        seconds : float
            The longest a frame is delayed; 0 or less does nothing.
        waiting : Callable[[], bool]
            Whether there is still something to wait for.
        """
        if seconds <= 0.0:
            return
        self._hold_seconds = seconds
        self._hold_waiting = waiting

    def release_hold(self) -> None:
        """End a hold: the next frame is drawn."""
        self._hold_seconds = 0.0
        self._hold_waiting = None
        self._hold_deadline = None

    def _holding(self) -> bool:
        if self._hold_waiting is None:
            return False
        now = time.perf_counter()
        if self._hold_deadline is None:
            self._hold_deadline = now + self._hold_seconds
        if now < self._hold_deadline and self._hold_waiting():
            return True
        self.release_hold()
        return False

    def _draw_frame(self) -> None:
        # A draw already queued with the backend can still arrive after close();
        # rendering it would touch a released surface.
        if self._closed:
            return
        if self._holding():
            # Nothing is rendered, so the canvas keeps its last picture.
            # An on-demand canvas needs asking for the frame that ends this.
            self._canvas.request_draw()
            return
        self._drawing = True
        try:
            self._draw_frame_inner()
        finally:
            self._drawing = False

    def _draw_frame_inner(self) -> None:
        # Content changed since the last frame: the history is of a picture
        # that no longer applies.  Ahead of the render, so the stale blend
        # never lands even once.
        if self._accum_dirty:
            self._accum_dirty = False
            self._accum_pass.reset()

        # Drive the camera controller.  ``tick`` returns the camera state
        # while the controller has a running action and ``None`` otherwise;
        # with ``auto_update=False`` it does not apply the state, so that is
        # done here.  Ahead of the comparison below, so the change is seen in
        # the frame that draws it.  Each driven frame asks for the next, as
        # pygfx does when it drives.
        controller_state = self._controller.tick() if self._controller.enabled else None
        driving = controller_state is not None
        if driving:
            self._camera.set_state(controller_state)
            self.request_frame()

        # Detect camera changes by comparing against the cached state.
        current_state = self.capture_camera_state()
        if current_state != self._last_camera_state and not self._applying_model_state:
            self._camera_dirty = True
            self._last_camera_state = current_state
            self._accum_pass.reset()
            _CAMERA_LOGGER.debug(
                "camera_changed  canvas=%s  scene=%s",
                self._canvas_id,
                self._scene_id,
            )

        if self._camera_dirty and self._event_bus is not None:
            self._camera_dirty = False
            _CAMERA_LOGGER.debug(
                "emit_camera_event  canvas=%s  scene=%s",
                self._canvas_id,
                self._scene_id,
            )
            self._event_bus.emit(
                CameraChangedEvent(
                    source_id=self._canvas_id,
                    scene_id=self._scene_id,
                    camera_state=current_state,
                    interactive=True,
                )
            )

        # After the change above, so the last camera change of a motion
        # re-arms the stillness timer before the controller's scope closes.
        if driving != self._driving:
            self._driving = driving
            if self._event_bus is not None:
                self._event_bus.emit(
                    _CameraControllerEvent(
                        source_id=self._canvas_id,
                        canvas_id=self._canvas_id,
                        scene_id=self._scene_id,
                        driving=driving,
                    )
                )

        if self._tick_visuals_fn is not None:
            self._tick_visuals_fn()

        # Re-sync the shared visual LUT from the authoritative per-visual
        # map.  Both the outline pass and the ambient occlusion pass read
        # that one table.
        # World objects are rebuilt on 2D/3D switches, multiscale brick
        # updates and channel changes, and every rebuild gives out a fresh
        # global_id -- so a write-once LUT would silently lose its entries.
        if self._visual_lut_sync_fn is not None:
            self._visual_lut_sync_fn()

        scene = self._get_scene_fn(self._scene_id)

        # Plane gizmos are drawn in 3D only, after the scene and before the
        # overlays: with no depth test a gizmo drawn last would paint over
        # them.
        gizmos = (
            [gizmo for gizmo in self._plane_gizmos.values() if gizmo.visible]
            if self._dim == "3d"
            else []
        )
        t_frame = time.perf_counter()
        if gizmos or self._overlays:
            canvas_width, canvas_height = self._canvas.get_logical_size()
            # First pass: main scene. flush=False keeps the colour and depth
            # buffers open for the subsequent passes.
            self._renderer.render(scene, self._camera, flush=False)
            remaining = len(gizmos) + len(self._overlays)
            for gizmo in gizmos:
                remaining -= 1
                self._renderer.render(gizmo.scene, self._camera, flush=remaining == 0)
            for overlay in self._overlays:
                overlay.on_frame(canvas_width, canvas_height)
                remaining -= 1
                self._renderer.render(
                    overlay.overlay_scene,
                    overlay.overlay_camera,
                    flush=remaining == 0,
                )
        else:
            # Fast path — no extra passes; single render call as before.
            self._renderer.render(scene, self._camera)
        frame_time_ms = (time.perf_counter() - t_frame) * 1000.0

        if self._event_bus is not None:
            self._event_bus.emit(
                FrameRenderedEvent(
                    source_id=self._canvas_id,
                    canvas_id=self._canvas_id,
                    frame_time_ms=frame_time_ms,
                )
            )
