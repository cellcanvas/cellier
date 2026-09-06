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

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget

    from cellier.events._bus import EventBus
    from cellier.render._config import AmbientOcclusionConfig
    from cellier.render.visuals._canvas_overlay import GFXCanvasOverlay


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
        self._controller_3d = gfx.OrbitController(
            camera=self._camera_3d, register_events=self._renderer
        )
        self._camera_2d = gfx.OrthographicCamera(maintain_aspect=True)
        self._controller_2d = gfx.PanZoomController(
            camera=self._camera_2d, register_events=self._renderer
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
        target_visual_ids: frozenset[UUID] | None = None,
    ) -> ReslicingRequest:
        """Snapshot the current camera state into a ReslicingRequest.

        All array fields are copied.  Screen size is read from the canvas
        at call time and baked into the returned request.

        Parameters
        ----------
        dims_state : DimsState
            Current dimension display state.
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
                dims_state, target_visual_ids, screen_w, screen_h
            )
        return self._capture_perspective(
            dims_state, target_visual_ids, screen_w, screen_h
        )

    def _capture_perspective(
        self,
        dims_state: DimsState,
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
            request_id=uuid4(),
            scene_id=self._scene_id,
            canvas_id=self._canvas_id,
            target_visual_ids=target_visual_ids,
        )

    def _capture_orthographic(
        self,
        dims_state: DimsState,
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

    def apply_camera_state(self, request: ReslicingRequest) -> None:
        """Apply a camera snapshot from the model layer (programmatic move).

        The ``_applying_model_state`` guard prevents the resulting camera
        setter calls from firing ``_on_controller_event``, which would
        otherwise cause a feedback loop: model change -> apply to pygfx ->
        controller event -> model change -> ...

        Parameters
        ----------
        request : ReslicingRequest
            Camera snapshot to apply.
        """
        self._applying_model_state = True
        try:
            self._camera.world.position = tuple(request.camera_pos)
        finally:
            self._applying_model_state = False
        # Caching the new state suppresses the diff in ``_draw_frame``, which
        # is what stops the feedback loop -- but the camera really did move,
        # so the history has to be discarded here instead.
        self._last_camera_state = self.capture_camera_state()
        self.invalidate_accumulation()

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

    def _draw_frame(self) -> None:
        # A draw already queued with the backend can still arrive after close();
        # rendering it would touch a released surface.
        if self._closed:
            return

        # Content changed since the last frame: the history is of a picture
        # that no longer applies.  Ahead of the render, so the stale blend
        # never lands even once.
        if self._accum_dirty:
            self._accum_dirty = False
            self._accum_pass.reset()

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

        t_frame = time.perf_counter()
        if self._overlays:
            canvas_width, canvas_height = self._canvas.get_logical_size()
            # First pass: main scene. flush=False keeps the colour and depth
            # buffers open for the subsequent overlay passes.
            self._renderer.render(scene, self._camera, flush=False)
            for index, overlay in enumerate(self._overlays):
                overlay.on_frame(canvas_width, canvas_height)
                is_last = index == len(self._overlays) - 1
                self._renderer.render(
                    overlay.overlay_scene,
                    overlay.overlay_camera,
                    flush=is_last,
                )
        else:
            # Fast path — no overlays; single render call as before.
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
