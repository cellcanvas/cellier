"""Single-scene convenience viewer wrapping CellierController."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar
from uuid import UUID

from cellier.controller import CellierController
from cellier.convenience._controls_registry import ControlsRegistryMixin
from cellier.convenience._render_settings import RenderSettingsMixin
from cellier.convenience._startup import StartupState
from cellier.events import CanvasAddedEvent
from cellier.render._capture import write_png
from cellier.scene.dims import WorldAxesLike, world_coordinate_system
from cellier.visuals._canvas_overlay import CanvasOverlay
from cellier.visuals._scene_overlay import SceneOverlay

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from PySide6.QtWidgets import QWidget

    from cellier.convenience.gui._controls_config import (
        GraphControlsConfig,
        InMemoryImageControlsConfig,
        LabelsControlsConfig,
        LinesControlsConfig,
        MeshControlsConfig,
        MultiscaleImageControlsConfig,
        MultiscaleLabelsControlsConfig,
        PointsControlsConfig,
    )
    from cellier.data._base_data_store import BaseDataStore
    from cellier.data.graph._graph_memory_store import GraphMemoryStore
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.data.label._label_memory_store import LabelMemoryStore
    from cellier.data.lines._lines_memory_store import LinesMemoryStore
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.data.points._points_memory_store import PointsMemoryStore
    from cellier.render._config import RenderManagerConfig
    from cellier.scene._background import BackgroundAppearance
    from cellier.scene.scene import Scene
    from cellier.transform import BaseTransform
    from cellier.visuals._base_visual import VisualOutline
    from cellier.visuals._graph_memory import (
        GraphAppearance,
        GraphVisual,
        TrailConfig,
    )
    from cellier.visuals._image import (
        MultiscaleImageAppearance,
        MultiscaleImageChannelAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageSingleAppearance,
        MultiscaleImageVisual,
    )
    from cellier.visuals._image_memory import (
        ImageVisual,
        InMemoryImageAppearance,
        InMemoryImageChannelAppearance,
        InMemoryImageSingleAppearance,
    )
    from cellier.visuals._label_memory import (
        BaseLabelsAppearance,
        LabelMemoryVisual,
        OutlineMode,
    )
    from cellier.visuals._labels import (
        MultiscaleLabelRenderConfig,
        MultiscaleLabelsAppearance,
        MultiscaleLabelVisual,
    )
    from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
    from cellier.visuals._mesh_memory import MeshAppearance, MeshVisual
    from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual

_T = TypeVar("_T", bound="BaseDataStore")


class Viewer(ControlsRegistryMixin, RenderSettingsMixin):
    """Single-scene viewer wrapping a CellierController.

    Creates a controller and a single scene pre-wired and ready to receive
    data and visuals. No Qt objects are constructed here; call
    :meth:`add_canvas` when you are ready to attach a render surface.

    Parameters
    ----------
    axes : WorldAxesLike
        The world axes in order: a ``WorldCoordinateSystem``, or a sequence of
        ``Axis`` objects and/or ``(name, axis_type)`` pairs.  Their number
        determines the dimensionality of the scene.  Axis types are stated,
        never inferred from the name -- ``spatial_axes("z", "y", "x")`` is the
        shorthand for an all-spatial world, and a mixed one spells the rest
        out::

            Viewer([("t", "time"), *spatial_axes("z", "y", "x")])
    dim : "2d" or "3d"
        Initial display dimensionality. Default ``"2d"``.
    render_modes : set[str] or None
        Which rendering modes the scene (and its visuals) should support.
        Defaults to ``{"2d", "3d"}`` when ``None``.
    render_config : RenderManagerConfig or None
        Render pipeline configuration passed through to the controller.
        Uses controller defaults when ``None``.
    gui : "qt", "anywidget", or "offscreen"
        Which GUI toolkit the canvas should target. ``"qt"`` (default) renders
        into a Qt widget; ``"anywidget"`` renders into a notebook canvas for
        Jupyter / marimo; ``"offscreen"`` renders with no window at all, for
        headless capture via :meth:`screenshot`. Fixed at construction.
        ``"offscreen"`` viewers have no embeddable widget, so the layout
        builders (``build_canvas_widget``, ``launch``, ``show``, ``display``)
        reject them.
    """

    def __init__(
        self,
        axes: WorldAxesLike,
        *,
        dim: Literal["2d", "3d"] = "2d",
        render_modes: set[str] | None = None,
        render_config: RenderManagerConfig | None = None,
        gui: Literal["qt", "anywidget", "offscreen"] = "qt",
    ) -> None:
        resolved_render_modes = (
            render_modes if render_modes is not None else {"2d", "3d"}
        )
        self._controller = CellierController(render_config=render_config, gui=gui)
        self._scene = self._controller.add_scene(
            name="main",
            dim=dim,
            coordinate_system=world_coordinate_system(axes),
            render_modes=resolved_render_modes,
        )
        # Callbacks fired once the scene's startup data is on the GPU; consumed
        # by the launcher (see convenience._launch._init_view).
        self._ready_callbacks: list[Callable[[], None]] = []
        # Controls configs recorded by the add_* methods, read by the layout
        # docks; kept current as visuals are removed.
        self._init_controls_registry()
        self._init_overlays()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def controller(self) -> CellierController:
        """The underlying CellierController."""
        return self._controller

    @property
    def gui(self) -> str:
        """The GUI toolkit this viewer renders into (``"qt"`` or ``"anywidget"``)."""
        return self._controller._gui

    @property
    def scene(self) -> Scene:
        """The single scene managed by this viewer."""
        return self._scene

    @property
    def background(self) -> BackgroundAppearance:
        """Appearance of the background drawn behind the scene's visuals.

        Mutate its fields to update the canvas at runtime::

            viewer.background.mode = "uniform"
            viewer.background.color = (0.0, 0.0, 0.0, 1.0)

        Assigning a whole new ``BackgroundAppearance`` works too.
        """
        return self._scene.background

    @background.setter
    def background(self, value: BackgroundAppearance) -> None:
        self._scene.background = value

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _init_overlays(self) -> None:
        """Start holding canvas overlays requested before a canvas exists.

        A convenience viewer builds its canvas late -- in ``add_canvas`` or
        the layout builders -- so a canvas overlay added first is kept here
        and attached to the first canvas when ``CanvasAddedEvent`` announces
        it.  The subscription is weak so a dropped viewer is not kept alive
        by a controller that outlives it.
        """
        self._pending_canvas_overlays: list[CanvasOverlay] = []
        self._controller._outgoing_events.subscribe(
            CanvasAddedEvent,
            self._attach_pending_canvas_overlays,
            entity_id=self._scene.id,
            weak=True,
        )

    @property
    def overlays(self) -> tuple[CanvasOverlay | SceneOverlay, ...]:
        """Every overlay on this viewer, scene overlays first.

        Scene overlays in add order, then the canvas overlays of each canvas
        in creation order, then any canvas overlay still waiting for a canvas.
        """
        canvas_overlays = [
            overlay
            for canvas_id in self.canvases
            for overlay in self._scene.canvases[canvas_id].overlays
        ]
        return (
            *self._scene.overlays,
            *canvas_overlays,
            *self._pending_canvas_overlays,
        )

    def add_scene_overlay(self, overlay: SceneOverlay) -> SceneOverlay:
        """Add a world-space overlay to this viewer's scene.

        A scene overlay is drawn in the scene's world by the scene camera and
        follows the scene's contents -- see
        :meth:`~cellier.controller.CellierController.add_scene_overlay`.
        Its fields are live afterwards::

            box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))
            box.appearance.color = (1.0, 0.0, 0.0, 1.0)
            box.visible = False

        Parameters
        ----------
        overlay : SceneOverlay
            The overlay model, e.g. a :class:`~cellier.visuals.SceneBoundingBox`.

        Returns
        -------
        SceneOverlay
            The same model.

        Raises
        ------
        TypeError
            If *overlay* is not a scene overlay.
        """
        if not isinstance(overlay, SceneOverlay):
            raise TypeError(
                f"add_scene_overlay takes a SceneOverlay, got "
                f"{type(overlay).__name__}.  Canvas overlays go through "
                "add_canvas_overlay."
            )
        self._controller.add_scene_overlay(self._scene.id, overlay)
        self._controls_changed.emit()
        return overlay

    def add_canvas_overlay(self, overlay: CanvasOverlay) -> CanvasOverlay:
        """Add a screen-space overlay to this viewer's canvas.

        Attached to the viewer's first canvas.  Before any canvas exists --
        the usual case, since ``launch`` / ``show`` / ``display`` build it --
        the overlay is held and attached as soon as the canvas is created.
        Its fields are live once attached.

        Parameters
        ----------
        overlay : CanvasOverlay
            The overlay model, e.g. a :class:`~cellier.visuals.CenteredAxes2D`.

        Returns
        -------
        CanvasOverlay
            The same model.

        Raises
        ------
        TypeError
            If *overlay* is not a canvas overlay.
        """
        if not isinstance(overlay, CanvasOverlay):
            raise TypeError(
                f"add_canvas_overlay takes a CanvasOverlay, got "
                f"{type(overlay).__name__}.  Scene overlays go through "
                "add_scene_overlay."
            )
        canvases = self.canvases
        if canvases:
            self._controller.add_canvas_overlay(canvases[0], overlay)
        else:
            self._pending_canvas_overlays.append(overlay)
        self._controls_changed.emit()
        return overlay

    def remove_overlay(self, overlay: CanvasOverlay | SceneOverlay | UUID) -> None:
        """Remove an overlay of either category.

        Parameters
        ----------
        overlay : CanvasOverlay, SceneOverlay or UUID
            The overlay model, or its id.

        Raises
        ------
        KeyError
            If the overlay is not on this viewer.
        """
        overlay_id = overlay if isinstance(overlay, UUID) else overlay.id
        pending = [o for o in self._pending_canvas_overlays if o.id == overlay_id]
        if pending:
            self._pending_canvas_overlays = [
                o for o in self._pending_canvas_overlays if o.id != overlay_id
            ]
        else:
            if not any(o.id == overlay_id for o in self.overlays):
                raise KeyError(f"No overlay with id={overlay_id!r} on this viewer.")
            self._controller.remove_overlay(overlay_id)
        self._controls_changed.emit()

    def _attach_pending_canvas_overlays(self, event: CanvasAddedEvent) -> None:
        """Attach the held canvas overlays to the viewer's first canvas."""
        if not self._pending_canvas_overlays:
            return
        pending, self._pending_canvas_overlays = self._pending_canvas_overlays, []
        for overlay in pending:
            self._controller.add_canvas_overlay(event.canvas_id, overlay)
        # The overlays were already listed while pending, but a dock built
        # against them before the canvas existed may want to rebuild.
        self._controls_changed.emit()

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def on_ready(self, callback: Callable[[], None]) -> None:
        """Register a callback fired once the scene's startup data is on the GPU.

        The callback runs after the initial reslice triggered by
        :func:`~cellier.convenience.launch` / :func:`~cellier.convenience.show`
        has committed all visuals (in-memory, multiscale, multichannel, and
        geometry) to the GPU.  Use it to hide a loading indicator, capture a
        screenshot, or enable controls once the first view is fully loaded.

        Must be called before ``launch``/``show``.  For ad-hoc readiness
        signals outside the convenience launchers, use
        :meth:`~cellier.controller.CellierController.on_scene_ready` directly.

        Parameters
        ----------
        callback : Callable[[], None]
            Zero-argument callback.
        """
        self._ready_callbacks.append(callback)

    @property
    def startup_state(self) -> StartupState:
        """How far this viewer has got through starting up.

        Readable at any moment, with no callback, no event loop and no front
        end -- which is the point.  A blank viewer used to be undiagnosable
        from Python; now it can say whether it is waiting for the canvas to
        reach the browser, waiting for a first frame, loading data, or done.

        ``StartupState.IDLE`` until ``display``/``launch``/``show`` runs.
        """
        tracker = getattr(self, "_startup", None)
        return StartupState.IDLE if tracker is None else tracker.state

    @property
    def scene_startup_states(self) -> dict[str, StartupState]:
        """Each scene's startup state, keyed as this viewer keys its scenes.

        An aggregate hides which panel is stuck; this does not.
        """
        tracker = getattr(self, "_startup", None)
        return {} if tracker is None else tracker.scene_states

    def startup_report(self) -> str:
        """A one-line summary of startup progress, for a cell or a log line."""
        tracker = getattr(self, "_startup", None)
        if tracker is None:
            return "idle (not started)"
        return tracker.describe()

    def on_scene_ready(self, key: str, callback: Callable[[], None]) -> None:
        """Fire *callback* when one scene's data is on the GPU.

        The per-scene counterpart of :meth:`on_ready`, which waits for *all*
        of them.  Fires immediately if that scene is already ready.

        Must be called after ``display``/``launch``, which is when the
        startup tracker exists.

        Parameters
        ----------
        key : str
            The scene key -- ``"scene"`` for a single-scene viewer, or the
            panel key (``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``) for an
            ``OrthoViewer``.
        callback : Callable[[], None]
            Zero-argument callback.
        """
        tracker = getattr(self, "_startup", None)
        if tracker is None:
            raise RuntimeError(
                "on_scene_ready requires a started viewer; call it after "
                "display()/launch()/show()."
            )
        tracker.on_scene_ready(key, callback)

    def on_startup_progress(self, callback: Callable[[int, int], None]) -> None:
        """Fire ``callback(scenes_ready, scenes_total)`` as scenes load.

        For a progress bar over a slow or remote dataset, where the gap
        between "shown" and "loaded" is long enough to need reporting.
        """
        tracker = getattr(self, "_startup", None)
        if tracker is None:
            raise RuntimeError(
                "on_startup_progress requires a started viewer; call it after "
                "display()/launch()/show()."
            )
        tracker.on_progress(callback)

    def on_startup_stalled(self, callback: Callable[[dict], None]) -> None:
        """Fire ``callback({scene key: state})`` if startup does not finish.

        The signal that was missing entirely: startup could only ever report
        success, so a viewer that never finished looked exactly like one still
        working.  The payload names each unfinished scene and the state it
        stopped in, so the report can say *what* it was waiting for.

        Stalling is not an error -- slow remote data legitimately takes a
        while -- it is the cue that something is worth looking at.  Tune the
        window with ``display(..., stall_timeout=)``.
        """
        tracker = getattr(self, "_startup", None)
        if tracker is None:
            raise RuntimeError(
                "on_startup_stalled requires a started viewer; call it after "
                "display()/launch()/show()."
            )
        tracker.on_stalled(callback)

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def on_pick(
        self,
        canvas_id: UUID,
        event_type: type,
        callback: Callable[[Any], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> Any:
        """Register a callback fired when a visual of one kind is picked.

        Mirrors :meth:`CellierController.on_pick`; see it for the events,
        their timing and gating.

        Parameters
        ----------
        canvas_id : UUID
            The canvas to watch, e.g. one of ``viewer.canvases``.
        event_type : type
            One of :data:`cellier.events.PICK_EVENT_TYPES`.
        callback : Callable
            Called with each pick event.
        owner_id : UUID or None
            Owner for bulk removal.  Defaults to *canvas_id*, so removing the
            canvas removes the subscription.
        weak : bool
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
            Pass it to :meth:`unsubscribe_pick`.
        """
        return self._controller.on_pick(
            canvas_id,
            event_type,
            callback,
            owner_id=canvas_id if owner_id is None else owner_id,
            weak=weak,
        )

    def unsubscribe_pick(self, handle: Any) -> None:
        """Remove a subscription created by :meth:`on_pick`."""
        self._controller.unsubscribe_pick(handle)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @property
    def canvases(self) -> tuple[UUID, ...]:
        """IDs of the canvases attached to this viewer's scene, in creation order.

        Empty until :meth:`add_canvas` (or a layout builder) has run.  A viewer
        with no canvas is still capturable -- see :meth:`screenshot`.
        """
        return tuple(self._controller.get_canvas_ids(self._scene.id))

    def screenshot(
        self,
        *,
        canvas: UUID | None = None,
        size: tuple[int, int] | None = None,
        scale: float = 1.0,
        frames: int | Literal["converged"] = 1,
        save: str | Path | None = None,
        **capture_kwargs,
    ) -> np.ndarray:
        """Capture a reproducible screenshot as an RGBA uint8 array.

        The frame is rendered offscreen, so two captures of the same viewer
        state produce byte-identical arrays and the result does not depend on
        a window being open, on the display's pixel ratio, or on which GUI
        toolkit the viewer targets.

        The capture shows the data **currently resident on the GPU** -- it does
        not reslice.  Capture from :meth:`on_ready` (or after ``launch`` /
        ``show`` / ``display`` have fired it) when a load may still be running.

        A viewer with **no canvas** still renders, fitting the camera to the
        scene, but it will have no data to show: slice requests are planned
        per canvas, so a viewer that never called :meth:`add_canvas` has never
        loaded anything.  Add a canvas and let the reslice finish first --
        ``cellier.convenience.capture`` does this for you.

        Parameters
        ----------
        canvas : UUID or None
            Which canvas's viewpoint to reproduce, from :attr:`canvases`.
            This selects a **viewpoint, not a surface**: the pixels always
            come from a fresh offscreen canvas.  With ``None`` (default) the
            viewer's single canvas is used, the scene is fitted if there is no
            canvas, and an ambiguous choice raises rather than guessing.
        size : tuple[int, int] or None
            ``(width, height)`` in pixels before *scale*.  Defaults to the
            selected canvas's physical size (so the on-screen framing is
            reproduced), or ``(600, 600)`` when there is no canvas.
        scale : float
            Multiplier applied to *size*.  ``scale=2`` doubles the resolution.
        frames : int or "converged"
            ``1`` (default) draws a single frame with temporal accumulation
            off.  ``"converged"`` draws the number of frames the accumulator
            needs to settle (44 at the default blend weight), which is what
            you want whenever ambient occlusion is enabled.  ``N`` draws
            exactly N accumulated frames.
        save : str, Path, or None
            When given, also write the frame to this path as a PNG.
        **capture_kwargs
            Forwarded to the capture helper (``max_frames``, ``residual``).

        Returns
        -------
        np.ndarray
            RGBA uint8 array of shape ``(height, width, 4)``.

        Raises
        ------
        ValueError
            If *canvas* is not one of this viewer's canvases, or if it is
            omitted while the viewer has more than one canvas.
        """
        frame = self._controller_screenshot(
            canvas=canvas, size=size, scale=scale, frames=frames, **capture_kwargs
        )
        if save is not None:
            write_png(save, frame)
        return frame

    def _controller_screenshot(
        self,
        *,
        canvas: UUID | None,
        size: tuple[int, int] | None,
        scale: float,
        frames: int | Literal["converged"],
        **capture_kwargs,
    ) -> np.ndarray:
        """Resolve which canvas to reproduce, then capture through it."""
        canvas_ids = self.canvases
        if canvas is not None:
            if canvas not in canvas_ids:
                raise ValueError(
                    f"canvas {canvas} is not one of this viewer's canvases "
                    f"{list(canvas_ids)}"
                )
            target = canvas
        elif len(canvas_ids) == 1:
            target = canvas_ids[0]
        elif not canvas_ids:
            # No canvas to copy a viewpoint from: fit the scene instead.
            return self._controller.screenshot_scene(
                self._scene.id,
                size=size,
                scale=scale,
                frames=frames,
                **capture_kwargs,
            )
        else:
            # Silently taking canvases[0] is how a screenshot of the wrong
            # view goes unnoticed.
            raise ValueError(
                f"this viewer has {len(canvas_ids)} canvases, so screenshot() "
                "cannot choose one for you. Pass canvas=<id> from "
                f"viewer.canvases: {list(canvas_ids)}"
            )
        return self._controller.screenshot(
            target, size=size, scale=scale, frames=frames, **capture_kwargs
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_file(self, path: str | Path) -> None:
        """Serialize the viewer model state to a JSON file.

        The file captures scenes, visuals, data stores, canvas camera state,
        and the render pipeline configuration. Pass the path to
        :meth:`from_file` to restore an equivalent ``Viewer``.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        self._controller.to_file(path)

    @classmethod
    def from_file(cls, path: str | Path) -> Viewer:
        """Restore a ``Viewer`` from a previously serialized file.

        The render pipeline configuration, scenes, visuals, data stores, and
        canvas camera state are all restored from the file. No extra arguments
        are required.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file written by :meth:`to_file`.

        Returns
        -------
        Viewer

        Raises
        ------
        ValueError
            If the file contains zero or more than one scene.
        """
        controller = CellierController.from_file(path)
        scenes = list(controller._model.scenes.values())
        if len(scenes) != 1:
            raise ValueError(
                f"Expected exactly one scene in the file, found {len(scenes)}. "
                "Use CellierController.from_file directly for multi-scene models."
            )
        return cls._from_existing(controller, scenes[0])

    @classmethod
    def _from_existing(cls, controller: CellierController, scene: Scene) -> Viewer:
        """Construct a Viewer from a pre-built controller and scene.

        Bypasses ``__init__``; used by :meth:`from_file`.
        """
        obj = object.__new__(cls)
        obj._controller = controller
        obj._scene = scene
        obj._ready_callbacks: list[Callable[[], None]] = []
        obj._init_controls_registry()
        obj._init_overlays()
        return obj

    # ------------------------------------------------------------------
    # Canvas
    # ------------------------------------------------------------------

    def add_canvas(
        self,
        *,
        render_modes: set[str] | None = None,
        initial_dim: str | None = None,
        fov: float = 70.0,
        depth_range_3d: tuple[float, float] = (1.0, 8000.0),
        depth_range_2d: tuple[float, float] = (-500.0, 500.0),
        canvas_size: tuple[int, int] | None = None,
    ) -> QWidget:
        """Create a canvas attached to this viewer's scene.

        Parameters
        ----------
        render_modes : set[str] or None
            Which camera modes to prepare. Defaults to the scene's own
            ``render_modes`` when ``None``.
        initial_dim : str or None
            Which mode is active first. Inferred from the scene's current
            ``displayed_axes`` when ``None``.
        fov : float
            Vertical field of view in degrees for the 3D camera. Default 70.
        depth_range_3d : tuple[float, float]
            ``(near, far)`` clip distances for the 3D camera.
        depth_range_2d : tuple[float, float]
            ``(near, far)`` clip distances for the 2D camera.
        canvas_size : tuple[int, int] or None
            Initial CSS pixel size for the anywidget canvas. Ignored for the
            Qt gui. Defaults to ``(600, 600)`` for the anywidget gui.

        Returns
        -------
        QWidget
            Embed with ``layout.addWidget(widget)``.
        """
        resolved_render_modes = (
            render_modes if render_modes is not None else set(self._scene.render_modes)
        )
        return self._controller.add_canvas(
            self._scene.id,
            render_modes=resolved_render_modes,
            initial_dim=initial_dim,
            fov=fov,
            depth_range_3d=depth_range_3d,
            depth_range_2d=depth_range_2d,
            canvas_size=canvas_size,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _resolve_data_store(self, data: _T | UUID) -> _T:
        if isinstance(data, UUID):
            return self._controller.get_data_store(data)  # type: ignore[return-value]
        return data

    # ------------------------------------------------------------------
    # Dims control
    # ------------------------------------------------------------------

    def set_displayed_dimensions(self, axis_names: tuple[str, ...]) -> None:
        """Set which axes are displayed by name.

        Switches the scene between 2D and 3D rendering by resolving
        *axis_names* to axis indices and calling the controller's dims API.
        Every axis keeps its slice position whether or not it is displayed,
        so an axis that goes from displayed back to sliced slices where it
        last was.

        Parameters
        ----------
        axis_names : tuple[str, ...]
            Axis labels to display, e.g. ``("y", "x")`` for 2D or
            ``("z", "y", "x")`` for 3D.  Must contain 2 or 3 names that
            are present in the scene's coordinate system.

        Raises
        ------
        ValueError
            If *axis_names* does not have 2 or 3 entries, or if any name is
            not in the scene's coordinate system.
        """
        if len(axis_names) not in (2, 3):
            raise ValueError(
                f"axis_names must have 2 or 3 entries, got {len(axis_names)}: "
                f"{axis_names!r}"
            )

        coord_labels = self._scene.dims.axis_labels
        label_to_index = {label: i for i, label in enumerate(coord_labels)}

        invalid = [n for n in axis_names if n not in label_to_index]
        if invalid:
            raise ValueError(
                f"Unknown axis names: {invalid}. Available: {list(coord_labels)}"
            )

        new_displayed = tuple(label_to_index[n] for n in axis_names)
        self._controller.cancel_pending_slices(self._scene.id)
        self._controller.set_displayed_axes(self._scene.id, new_displayed)
        self._controller.fit_camera(self._scene.id)

    # ------------------------------------------------------------------
    # Visual add methods
    # ------------------------------------------------------------------

    def add_image(
        self,
        data: ImageMemoryStore | UUID,
        appearance: InMemoryImageAppearance | None = None,
        name: str = "image",
        controls: InMemoryImageControlsConfig | None = None,
        *,
        single: InMemoryImageSingleAppearance | None = None,
        channel_axis: int | None = None,
        composite: bool = False,
        channels: dict[int, InMemoryImageChannelAppearance] | None = None,
        max_channels: int = 4,
        transform: BaseTransform | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> ImageVisual:
        """Add an in-memory image visual.

        One visual draws the image single-channel or composited; see
        :meth:`cellier.controller.CellierController.add_image`.

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : InMemoryImageAppearance or None
            Shared by both modes.  ``None`` uses the defaults.
        name : str
            Human-readable label. Default ``"image"``.
        controls : InMemoryImageControlsConfig or None
            Appearance panel configuration.  When ``None`` (default), no
            appearance panel is created for this visual.
        single : single appearance or None
            Single mode's appearance.  ``None`` uses the defaults.
        channel_axis : int or None
            The data axis a composite draws channels along; it must map to a
            world axis.  ``None`` (default) gives an image with no channels.
        composite : bool
            Start in composite mode.  Requires *channel_axis*.
        channels : dict[int, channel appearance] or None
            Composite mode's per-channel appearances.
        max_channels : int
            The most channels the visual may hold.  Default 4.
        transform : BaseTransform or None
            Data-to-world transform.  Identity when ``None``.
        outline : VisualOutline or None
            Screen-space outline assignment.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.
        pick_write : bool
            Whether the visual writes to the pick buffer.  Default ``True``.

        Returns
        -------
        ImageVisual
        """
        visual = self._controller.add_image(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            single=single,
            channel_axis=channel_axis,
            composite=composite,
            channels=channels,
            max_channels=max_channels,
            transform=transform,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        self._store_controls([visual.id], controls)
        return visual

    def add_labels(
        self,
        data: LabelMemoryStore | UUID,
        appearance: BaseLabelsAppearance | None = None,
        name: str = "labels",
        transform: BaseTransform | None = None,
        controls: LabelsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
        outline_selected_labels: dict[int, int] | None = None,
        outline_mode: OutlineMode = "per_label",
    ) -> LabelMemoryVisual:
        """Add an in-memory label visual.

        Parameters
        ----------
        data : LabelMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : BaseLabelsAppearance or None
            Appearance parameters. Defaults to ``InMemoryLabelsAppearance()``
            when ``None``.
        name : str
            Human-readable label. Default ``"labels"``.
        transform : BaseTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : LabelsControlsConfig or None
            Appearance controls configuration.  When ``None`` (default), no
            appearance controls are created.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.
        outline_selected_labels : dict[int, int] or None
            Maps a label value to the palette slot the selection layer draws
            it in.  ``None`` (default) selects no label, so an outlined
            labels visual shows boundaries only.
        outline_mode : {"per_label", "whole_object", "all_boundaries"}
            How the labels are outlined.  ``"per_label"`` (default) outlines
            the label values in ``outline_selected_labels``, each in its own
            slot's colour.  ``"whole_object"`` outlines the volume as one
            silhouette and ``"all_boundaries"`` every label's boundary, both
            in the colour of the ``outline`` slot.

        Returns
        -------
        LabelMemoryVisual
        """
        visual = self._controller.add_labels(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            transform,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
            outline_selected_labels=outline_selected_labels,
            outline_mode=outline_mode,
        )
        self._store_controls([visual.id], controls)
        return visual

    def add_mesh(
        self,
        data: MeshMemoryStore | UUID,
        appearance: MeshAppearance,
        name: str = "mesh",
        transform: BaseTransform | None = None,
        controls: MeshControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> MeshVisual:
        """Add a mesh visual.

        Parameters
        ----------
        data : MeshMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : MeshFlatAppearance, MeshPhongAppearance,
            Appearance parameters.
        name : str
            Human-readable label. Default ``"mesh"``.
        transform : BaseTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : MeshControlsConfig or None
            Appearance controls configuration.  When ``None`` (default), no
            appearance controls are created.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        MeshVisual
        """
        visual = self._controller.add_mesh(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            transform,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        self._store_controls([visual.id], controls)
        return visual

    def add_points(
        self,
        data: PointsMemoryStore | UUID,
        appearance: PointsMarkerAppearance | None = None,
        name: str = "points",
        transform: BaseTransform | None = None,
        controls: PointsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> PointsVisual:
        """Add a points visual.

        Parameters
        ----------
        data : PointsMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : PointsMarkerAppearance or None
            Appearance parameters. Defaults to ``PointsMarkerAppearance()``
            when ``None``.
        name : str
            Human-readable label. Default ``"points"``.
        transform : BaseTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : PointsControlsConfig or None
            Appearance controls configuration.  When ``None`` (default), no
            appearance controls are created.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        PointsVisual
        """
        visual = self._controller.add_points(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            transform,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        self._store_controls([visual.id], controls)
        return visual

    def add_graph(
        self,
        data: GraphMemoryStore | UUID,
        appearance: GraphAppearance | None = None,
        name: str = "graph",
        transform: BaseTransform | None = None,
        trail: dict[int, TrailConfig] | None = None,
        controls: GraphControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> GraphVisual:
        """Add a spatial-graph visual.

        Parameters
        ----------
        data : GraphMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : GraphAppearance or None
            Appearance parameters. Defaults to ``GraphAppearance()`` when
            ``None``.
        name : str
            Human-readable label. Default ``"graph"``.
        trail : dict[int, TrailConfig] or None
            Axis index -> window configuration. Extends the slab on that
            axis and optionally fades elements by distance from the current
            slice index. An out-of-range axis raises ``ValueError``.
        transform : BaseTransform or None
            Data-to-world transform. When ``None`` the store's own transform
            is used if it has one (a geff file's per-axis scale and offset),
            and identity otherwise.
        controls : GraphControlsConfig or None
            Appearance controls configuration.  When ``None`` (default), no
            appearance controls are created.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        GraphVisual
        """
        visual = self._controller.add_graph(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            transform,
            trail,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        self._store_controls([visual.id], controls)
        return visual

    def add_lines(
        self,
        data: LinesMemoryStore | UUID,
        appearance: LinesMemoryAppearance | None = None,
        name: str = "lines",
        transform: BaseTransform | None = None,
        controls: LinesControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> LinesVisual:
        """Add a lines visual.

        Parameters
        ----------
        data : LinesMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : LinesMemoryAppearance or None
            Appearance parameters. Defaults to ``LinesMemoryAppearance()``
            when ``None``.
        name : str
            Human-readable label. Default ``"lines"``.
        transform : BaseTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : LinesControlsConfig or None
            Appearance controls configuration.  When ``None`` (default), no
            appearance controls are created.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        LinesVisual
        """
        visual = self._controller.add_lines(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            transform,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        self._store_controls([visual.id], controls)
        return visual

    def add_image_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleImageAppearance | None = None,
        name: str = "image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: BaseTransform | None = None,
        controls: MultiscaleImageControlsConfig | None = None,
        *,
        single: MultiscaleImageSingleAppearance | None = None,
        channel_axis: int | None = None,
        composite: bool = False,
        channels: dict[int, MultiscaleImageChannelAppearance] | None = None,
        max_channels: int = 4,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale data store or UUID of an already-registered store.
        appearance : MultiscaleImageAppearance or None
            Shared by both modes, including the LOD settings.
        name : str
            Human-readable label. Default ``"image"``.
        render_config : MultiscaleImageRenderConfig or None
            GPU cache configuration.  Uses defaults when ``None``.
        transform : BaseTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : MultiscaleImageControlsConfig or None
            Appearance panel configuration.
        single : single appearance or None
            Single mode's appearance.  ``None`` uses the defaults.
        channel_axis : int or None
            The data axis a composite draws channels along; it must map to a
            world axis.  ``None`` (default) gives an image with no channels.
        composite : bool
            Start in composite mode.  Requires *channel_axis*.
        channels : dict[int, channel appearance] or None
            Composite mode's per-channel appearances.
        max_channels : int
            The most channels the visual may hold.  Default 4.
        outline : VisualOutline or None
            Screen-space outline assignment.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.
        pick_write : bool
            Whether the visual writes to the pick buffer.  Default ``True``.

        Returns
        -------
        MultiscaleImageVisual
        """
        visual = self._controller.add_image_multiscale(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            render_config,
            transform,
            single=single,
            channel_axis=channel_axis,
            composite=composite,
            channels=channels,
            max_channels=max_channels,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        self._store_controls([visual.id], controls)
        return visual

    def add_labels_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleLabelsAppearance,
        name: str = "labels",
        render_config: MultiscaleLabelRenderConfig | None = None,
        transform: BaseTransform | None = None,
        controls: MultiscaleLabelsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
        outline_selected_labels: dict[int, int] | None = None,
        outline_mode: OutlineMode = "per_label",
    ) -> MultiscaleLabelVisual:
        """Add a multiscale label visual.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale label store or UUID of an already-registered store.
        appearance : MultiscaleLabelsAppearance
            Visual appearance parameters.
        name : str
            Human-readable label. Default ``"labels"``.
        render_config : MultiscaleLabelRenderConfig or None
            LOD and rendering configuration. Uses
            defaults when ``None``.
        transform : BaseTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : MultiscaleLabelsControlsConfig or None
            Appearance controls configuration.  When ``None`` (default), no
            appearance controls are created.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.
        outline_selected_labels : dict[int, int] or None
            Maps a label value to the palette slot the selection layer draws
            it in.  ``None`` (default) selects no label, so an outlined
            labels visual shows boundaries only.
        outline_mode : {"per_label", "whole_object", "all_boundaries"}
            How the labels are outlined.  ``"per_label"`` (default) outlines
            the label values in ``outline_selected_labels``, each in its own
            slot's colour.  ``"whole_object"`` outlines the volume as one
            silhouette and ``"all_boundaries"`` every label's boundary, both
            in the colour of the ``outline`` slot.

        Returns
        -------
        MultiscaleLabelVisual
        """
        visual = self._controller.add_labels_multiscale(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            render_config,
            transform,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
            outline_selected_labels=outline_selected_labels,
            outline_mode=outline_mode,
        )
        self._store_controls([visual.id], controls)
        return visual
