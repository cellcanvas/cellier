"""Single-scene convenience viewer wrapping CellierController."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, TypeVar
from uuid import UUID

from cellier.controller import CellierController
from cellier.convenience._render_settings import RenderSettingsMixin
from cellier.convenience._startup import StartupState
from cellier.render._capture import write_png
from cellier.scene.dims import CoordinateSystem

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from PySide6.QtWidgets import QWidget

    from cellier.convenience.gui._controls_config import (
        BaseControlsConfig,
        ChannelControlsConfig,
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
    from cellier.transform import AffineTransform
    from cellier.visuals._base_visual import VisualOutline
    from cellier.visuals._channel_appearance import ChannelAppearance
    from cellier.visuals._graph_memory import (
        GraphAppearance,
        GraphVisual,
        TrailConfig,
    )
    from cellier.visuals._image import (
        MultichannelMultiscaleImageVisual,
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageVisual,
    )
    from cellier.visuals._image_memory import BaseImageAppearance, ImageVisual
    from cellier.visuals._image_memory_multichannel import MultichannelImageVisual
    from cellier.visuals._label_memory import BaseLabelsAppearance, LabelMemoryVisual
    from cellier.visuals._labels import (
        MultiscaleLabelRenderConfig,
        MultiscaleLabelsAppearance,
        MultiscaleLabelVisual,
    )
    from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
    from cellier.visuals._mesh_memory import MeshAppearance, MeshVisual
    from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual

_T = TypeVar("_T", bound="BaseDataStore")


class Viewer(RenderSettingsMixin):
    """Single-scene viewer wrapping a CellierController.

    Creates a controller and a single scene pre-wired and ready to receive
    data and visuals. No Qt objects are constructed here; call
    :meth:`add_canvas` when you are ready to attach a render surface.

    Parameters
    ----------
    axis_labels : tuple[str, ...]
        World-axis names in order, e.g. ``("z", "y", "x")``.
        The number of labels determines the dimensionality of the scene.
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
        axis_labels: tuple[str, ...],
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
            coordinate_system=CoordinateSystem(name="world", axis_labels=axis_labels),
            render_modes=resolved_render_modes,
        )
        # Saved world-space slice positions, keyed by axis index. Populated
        # by set_displayed_dimensions so axes restore their last position when
        # they cycle back from displayed to sliced.
        self._saved_slice_positions: dict[int, float] = {}
        # Callbacks fired once the scene's startup data is on the GPU; consumed
        # by the launcher (see convenience._launch._init_view).
        self._ready_callbacks: list[Callable[[], None]] = []
        # Controls configs keyed by visual id; set by add_image / add_image_multiscale.
        self._controls_configs: dict[UUID, BaseControlsConfig] = {}

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
        ``scripts/capture.py`` does this for you.

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
        obj._saved_slice_positions: dict[int, float] = {}
        obj._ready_callbacks: list[Callable[[], None]] = []
        obj._controls_configs: dict[UUID, BaseControlsConfig] = {}
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
        Slice positions for axes that transition from displayed to sliced are
        restored from the last known position (or default to 0 on first call).

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

        coord_labels = self._scene.dims.coordinate_system.axis_labels
        label_to_index = {label: i for i, label in enumerate(coord_labels)}

        invalid = [n for n in axis_names if n not in label_to_index]
        if invalid:
            raise ValueError(
                f"Unknown axis names: {invalid}. Available: {list(coord_labels)}"
            )

        new_displayed = tuple(label_to_index[n] for n in axis_names)
        new_displayed_set = set(new_displayed)

        selection = self._scene.dims.selection
        current_slices = dict(selection.slice_indices)
        stacked = set(selection.stacked_axes)
        ndim = len(coord_labels)

        # Save current slice positions before they potentially become displayed.
        for axis, value in current_slices.items():
            self._saved_slice_positions[axis] = float(value)

        # Build the new slice_indices: every axis that is neither displayed
        # nor stacked must appear in slice_indices.
        new_slices: dict[int, float] = {}
        for i in range(ndim):
            if i not in new_displayed_set and i not in stacked:
                new_slices[i] = self._saved_slice_positions.get(i, 0.0)

        self._controller.cancel_pending_slices(self._scene.id)
        current_displayed = set(selection.displayed_axes)
        adding_axes = new_displayed_set - current_displayed
        if adding_axes:
            # Expanding displayed axes: extend displayed first so the axis is
            # covered before it disappears from slice_indices.
            self._controller.set_displayed_axes(self._scene.id, new_displayed)
            self._controller.update_slice_indices(self._scene.id, new_slices)
        else:
            # Contracting displayed axes: add the new slice_indices entries
            # first so coverage is maintained before displayed shrinks.
            self._controller.update_slice_indices(self._scene.id, new_slices)
            self._controller.set_displayed_axes(self._scene.id, new_displayed)
        self._controller.fit_camera(self._scene.id)

    # ------------------------------------------------------------------
    # Visual add methods
    # ------------------------------------------------------------------

    def add_image(
        self,
        data: ImageMemoryStore | UUID,
        appearance: BaseImageAppearance,
        name: str = "image",
        controls: InMemoryImageControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> ImageVisual:
        """Add an in-memory image visual.

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : BaseImageAppearance
            Appearance parameters.
        name : str
            Human-readable label. Default ``"image"``.
        controls : InMemoryImageControlsConfig or None
            Appearance panel configuration. When ``None``
            (default), no appearance panel is created for this visual.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.

        Returns
        -------
        ImageVisual
        """
        visual = self._controller.add_image(
            self._resolve_data_store(data),
            self._scene.id,
            appearance,
            name,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_labels(
        self,
        data: LabelMemoryStore | UUID,
        appearance: BaseLabelsAppearance | None = None,
        name: str = "labels",
        transform: AffineTransform | None = None,
        controls: LabelsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        outline_selected_labels: dict[int, int] | None = None,
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
        transform : AffineTransform or None
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
        outline_selected_labels : dict[int, int] or None
            Maps a label value to the palette slot the selection layer draws
            it in.  ``None`` (default) selects no label, so an outlined
            labels visual shows boundaries only.

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
            outline_selected_labels=outline_selected_labels,
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_mesh(
        self,
        data: MeshMemoryStore | UUID,
        appearance: MeshAppearance,
        name: str = "mesh",
        transform: AffineTransform | None = None,
        controls: MeshControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
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
        transform : AffineTransform or None
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
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_points(
        self,
        data: PointsMemoryStore | UUID,
        appearance: PointsMarkerAppearance | None = None,
        name: str = "points",
        transform: AffineTransform | None = None,
        controls: PointsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
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
        transform : AffineTransform or None
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
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_graph(
        self,
        data: GraphMemoryStore | UUID,
        appearance: GraphAppearance | None = None,
        name: str = "graph",
        transform: AffineTransform | None = None,
        trail: dict[int, TrailConfig] | None = None,
        controls: GraphControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
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
        transform : AffineTransform or None
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
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_lines(
        self,
        data: LinesMemoryStore | UUID,
        appearance: LinesMemoryAppearance | None = None,
        name: str = "lines",
        transform: AffineTransform | None = None,
        controls: LinesControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
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
        transform : AffineTransform or None
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
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_image_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleImageAppearance,
        name: str = "image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: AffineTransform | None = None,
        controls: MultiscaleImageControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale data store or UUID of an already-registered store.
        appearance : MultiscaleImageAppearance
            Visual appearance parameters.
        name : str
            Human-readable label. Default ``"image"``.
        render_config : MultiscaleImageRenderConfig or None
            LOD and rendering configuration. Uses
            defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : MultiscaleImageControlsConfig or None
            Appearance panel configuration. When ``None`` (default), no
            appearance panel is created for this visual.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.

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
            outline=outline,
            ambient_occlusion=ambient_occlusion,
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_labels_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleLabelsAppearance,
        name: str = "labels",
        render_config: MultiscaleLabelRenderConfig | None = None,
        transform: AffineTransform | None = None,
        controls: MultiscaleLabelsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        outline_selected_labels: dict[int, int] | None = None,
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
        transform : AffineTransform or None
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
        outline_selected_labels : dict[int, int] or None
            Maps a label value to the palette slot the selection layer draws
            it in.  ``None`` (default) selects no label, so an outlined
            labels visual shows boundaries only.

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
            outline_selected_labels=outline_selected_labels,
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_multichannel_image(
        self,
        data: ImageMemoryStore | UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance],
        name: str = "multichannel_image",
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
        controls: ChannelControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> MultichannelImageVisual:
        """Add an in-memory multichannel image visual.

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or UUID of an already-registered store.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance]
            Per-channel appearance keyed by channel index.
        name : str
            Display name. Default ``"multichannel_image"``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.
        controls : ChannelControlsConfig or None
            Per-channel controls configuration. When ``None`` (default), no
            channel controls are created for this visual.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.

        Returns
        -------
        MultichannelImageVisual
        """
        visual = self._controller.add_multichannel_image(
            self._resolve_data_store(data),
            self._scene.id,
            channel_axis,
            channels,
            name,
            max_channels_2d,
            max_channels_3d,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual

    def add_multichannel_image_multiscale(
        self,
        data: BaseDataStore | UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance],
        name: str = "multichannel_image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: AffineTransform | None = None,
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
        controls: ChannelControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> MultichannelMultiscaleImageVisual:
        """Add a multiscale multichannel image visual.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale store or UUID of an already-registered store.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance]
            Per-channel appearance keyed by channel index.
        name : str
            Display name. Default ``"multichannel_image"``.
        render_config : MultiscaleImageRenderConfig or None
            LOD and rendering configuration. Uses
            defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.
        controls : ChannelControlsConfig or None
            Per-channel controls configuration. When ``None`` (default), no
            channel controls are created for this visual.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled; see
            ``outline_enabled``.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.

        Returns
        -------
        MultichannelMultiscaleImageVisual
        """
        visual = self._controller.add_multichannel_image_multiscale(
            self._resolve_data_store(data),
            self._scene.id,
            channel_axis,
            channels,
            name,
            render_config,
            transform,
            max_channels_2d,
            max_channels_3d,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
        )
        if controls is not None:
            self._controls_configs[visual.id] = controls
        return visual
