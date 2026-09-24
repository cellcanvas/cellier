"""Four-panel orthoviewer convenience class wrapping CellierController.

An :class:`OrthoViewer` manages a single controller with four pre-wired scenes
-- three orthogonal 2D slice panels (``xy``, ``xz``, ``yz``) and one 3D volume
panel (``vol``) -- all sharing the same world coordinate system.  Like
:class:`~cellier.convenience.Viewer` it launches empty; the ``add_*`` methods
register one data store and fan a visual out to every panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar
from uuid import UUID

from cellier.controller import CellierController
from cellier.convenience._controls_registry import ControlsRegistryMixin
from cellier.convenience._ortho_dims import OrthoDimsController
from cellier.convenience._render_settings import RenderSettingsMixin
from cellier.convenience._startup import StartupState
from cellier.render._capture import write_png
from cellier.scene._background import viewer_background
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    WorldAxesLike,
    world_coordinate_system,
)
from cellier.scene.scene import Scene

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from cellier.convenience.gui._controls_config import (
        BaseControlsConfig,
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
    from cellier.events import (
        BackstopCompleteEvent,
        LoadingProgress,
        ResliceProgressEvent,
        SubscriptionHandle,
    )
    from cellier.render._config import RenderManagerConfig
    from cellier.scene._background import BackgroundAppearance
    from cellier.transform import BaseTransform, WorldCoordinateSystem
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
    from cellier.visuals._label_memory import BaseLabelsAppearance, LabelMemoryVisual
    from cellier.visuals._labels import (
        MultiscaleLabelRenderConfig,
        MultiscaleLabelsAppearance,
        MultiscaleLabelVisual,
    )
    from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
    from cellier.visuals._loading import ProgressiveLoadingConfig
    from cellier.visuals._mesh_memory import MeshAppearance, MeshVisual
    from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual

_T = TypeVar("_T", bound="BaseDataStore")


def _callback_ref(callback: Callable, weak: bool) -> Callable[[], Callable | None]:
    """A zero-argument getter for *callback*, weak when asked.

    The group subscriptions wrap the caller's callback in a closure, so the
    bus's own weak reference would hold the closure (collected at once)
    rather than the callback; the weak reference is taken here instead.
    """
    if not weak:
        return lambda: callback
    import weakref

    if getattr(callback, "__name__", None) == "<lambda>":
        raise ValueError("Cannot create a weak subscription to a lambda.")
    if hasattr(callback, "__self__"):
        return weakref.WeakMethod(callback)
    return weakref.ref(callback)


# Panel keys in display order. ``vol`` is the 3D panel; the rest are 2D slices.
_PANEL_KEYS: tuple[str, ...] = ("xy", "xz", "yz", "vol")

# The appearance-controls groups a fanned-out add records: one control drives
# the three 2D panels, another the 3D panel.  Key -> (panel keys, dock label
# format).  A setting such as a render mode or iso threshold only means
# something in 3D, and the 2D and 3D views often want different contrast, so
# the two are never linked.
_CONTROLS_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "2d": (("xy", "xz", "yz"), "{name} (2D views)"),
    "3d": (("vol",), "{name} (3D view)"),
}


def _copy(model):
    """A copy of *model*, so each panel's visual owns its appearance.

    A new model built from the same field values.  Not ``model_copy``: a
    shallow copy of an ``EventedModel`` shares its signal group, so every
    panel would hear every other panel's edits, and a deep copy of a
    ``Colormap`` loses its registered name.  Field values such as a
    ``Colormap`` are immutable and shared safely.  A plain dict (validated
    later by the visual model) is copied too.
    """
    if model is None:
        return None
    if isinstance(model, dict):
        return dict(model)
    return type(model)(
        **{name: getattr(model, name) for name in type(model).model_fields}
    )


def _copy_channels(channels):
    if channels is None:
        return None
    return {index: _copy(appearance) for index, appearance in channels.items()}


def _resolve_spatial_axes(
    axis_labels: tuple[str, ...],
    spatial_axes: tuple[str, ...] | tuple[int, ...] | None,
) -> tuple[int, int, int]:
    """Resolve the three spatial axis indices in ``(z, y, x)`` order.

    Defaults to the last three axes when *spatial_axes* is ``None``.  Otherwise
    accepts a length-3 tuple of axis names or indices.
    """
    ndim = len(axis_labels)
    if ndim < 3:
        raise ValueError(
            f"OrthoViewer requires at least 3 axes, got {ndim}: {axis_labels!r}"
        )

    if spatial_axes is None:
        return (ndim - 3, ndim - 2, ndim - 1)

    if len(spatial_axes) != 3:
        raise ValueError(
            f"spatial_axes must have exactly 3 entries, got {len(spatial_axes)}: "
            f"{spatial_axes!r}"
        )

    label_to_index = {label: i for i, label in enumerate(axis_labels)}
    resolved: list[int] = []
    for entry in spatial_axes:
        if isinstance(entry, str):
            if entry not in label_to_index:
                raise ValueError(
                    f"Unknown spatial axis name {entry!r}. "
                    f"Available: {list(axis_labels)}"
                )
            resolved.append(label_to_index[entry])
        else:
            index = int(entry)
            if not 0 <= index < ndim:
                raise ValueError(
                    f"spatial axis index {index} out of range for {ndim} axes."
                )
            resolved.append(index)

    if len(set(resolved)) != 3:
        raise ValueError(f"spatial_axes must be distinct, got {spatial_axes!r}")
    return (resolved[0], resolved[1], resolved[2])


class OrthoViewer(ControlsRegistryMixin, RenderSettingsMixin):
    """Four-panel orthoviewer wrapping a single CellierController.

    Creates a controller and four pre-wired scenes that share one world
    coordinate system: three orthogonal 2D slice panels (``xy``, ``xz``,
    ``yz``) and one 3D volume panel (``vol``).  No Qt objects are constructed
    here; build canvases with
    :func:`cellier.convenience.gui.build_ortho_grid_widget` when ready.

    The three spatial axes (default: the last three axis labels, treated as
    ``z, y, x``) define the slice planes.  Any remaining axes are "extra" axes
    (e.g. channel or time).  By default every axis's slice position, thickness
    and slider override is mirrored across the four panels by an
    :class:`~cellier.convenience._ortho_dims.OrthoDimsController`, so the
    panels share one world point (design 3.6).

    Every panel starts with a uniform black background
    (:func:`~cellier.scene._background.viewer_background`); see
    :meth:`set_background`.

    Appearance controls come in two groups per added visual: one drives the
    three 2D panels together, the other the 3D panel.  An
    ``AppearanceControls()`` dock offers both, labelled ``"{name} (2D
    views)"`` and ``"{name} (3D view)"``.

    Parameters
    ----------
    axes : WorldAxesLike
        The world axes in order: a ``WorldCoordinateSystem``, or a sequence of
        ``Axis`` objects and/or ``(name, axis_type)`` pairs.  Their number
        sets the dimensionality; at least 3 are required.  Axis types are
        stated, never inferred -- a 4-D ortho viewer over a channel stack is
        ``OrthoViewer([("c", "channel"), *spatial_axes("z", "y", "x")])``.
    spatial_axes : tuple[str, ...], tuple[int, ...], or None
        The three axes (names or indices, in ``z, y, x`` order) that form the
        orthogonal planes.  Defaults to the last three axes when ``None``.
    link_axes : bool
        When ``True`` (default), every axis's slice position, thickness and
        slider override is mirrored across the four panels.
    render_config : RenderManagerConfig or None
        Render pipeline configuration passed through to the controller.
    gui : "qt", "anywidget", or "offscreen"
        Which GUI toolkit the canvases should target. ``"qt"`` (default)
        renders into Qt widgets; ``"anywidget"`` renders into notebook canvases
        for Jupyter / marimo; ``"offscreen"`` renders with no window at all,
        for headless capture via :meth:`screenshot`. Fixed at construction.
        ``"offscreen"`` orthoviewers have no embeddable widgets, so the grid
        builders reject them.
    """

    def __init__(
        self,
        axes: WorldAxesLike,
        *,
        spatial_axes: tuple[str, ...] | tuple[int, ...] | None = None,
        link_axes: bool = True,
        render_config: RenderManagerConfig | None = None,
        gui: Literal["qt", "anywidget", "offscreen"] = "qt",
    ) -> None:
        self._controller = CellierController(render_config=render_config, gui=gui)
        world = world_coordinate_system(axes)
        self._spatial_axes = _resolve_spatial_axes(world.axis_names(), spatial_axes)
        self._ndim = world.ndim
        self._extra_axes = {i for i in range(self._ndim) if i not in self._spatial_axes}
        self._scenes = self._build_scenes(world)
        self._dims_controller: OrthoDimsController | None = None
        # Per-visual controls configs, keyed by a representative visual id;
        # _visual_groups maps that id to the visuals one widget drives: the
        # three 2D panels, or the 3D panel (_CONTROLS_GROUPS).  Not
        # channel-specific: any fanned-out add_* records its groups here.
        self._init_controls_registry()
        # Callbacks fired once all panel scenes' startup data is on the GPU;
        # consumed by the launcher (see convenience._launch._init_view).
        self._ready_callbacks: list[Callable[[], None]] = []
        if link_axes:
            self._wire_axis_sync()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scenes(self, world: WorldCoordinateSystem) -> dict[str, Scene]:
        s0, s1, s2 = self._spatial_axes
        # displayed axes per panel; the remaining spatial axis is sliced.
        displayed_by_key: dict[str, tuple[int, ...]] = {
            "xy": (s1, s2),
            "xz": (s0, s2),
            "yz": (s0, s1),
            "vol": (s0, s1, s2),
        }
        scenes: dict[str, Scene] = {}
        for key in _PANEL_KEYS:
            displayed = displayed_by_key[key]
            render_modes = {"3d"} if key == "vol" else {"2d"}
            # Every axis gets a position, displayed ones included (D36).
            slice_indices = dict.fromkeys(range(self._ndim), 0.0)
            scene = Scene(
                name=key,
                dims=DimsManager(
                    world_coordinate_system=world,
                    selection=AxisAlignedSelection(
                        displayed_axes=displayed,
                        slice_indices=slice_indices,
                    ),
                ),
                render_modes=render_modes,
                lighting="none",
                background=viewer_background(),
            )
            scenes[key] = self._controller.add_scene_model(scene)
        return scenes

    def _wire_axis_sync(self) -> None:
        """Mirror dims across the panels, xy first so it wins a disagreement."""
        self._dims_controller = OrthoDimsController(
            self._controller, [self._scenes[key] for key in _PANEL_KEYS]
        )

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
    def scenes(self) -> dict[str, Scene]:
        """The four panel scenes keyed ``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``."""
        return self._scenes

    def set_background(self, background: BackgroundAppearance) -> None:
        """Apply one background appearance to all four panels.

        Each panel is its own ``Scene`` and so owns its own background; use
        ``viewer.scenes["xy"].background`` to change just one.  A copy is
        given to each panel so that later edits to one panel's background do
        not leak into the others.

        Parameters
        ----------
        background : BackgroundAppearance
            The background appearance to apply to every panel.
        """
        for scene in self._scenes.values():
            scene.background = background.model_copy(deep=True)

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
            The canvas to watch, e.g. one of a panel scene's canvases.
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

    def screenshot(
        self,
        *,
        panel: str | None = None,
        size: tuple[int, int] | None = None,
        scale: float = 1.0,
        frames: int | Literal["converged"] = 1,
        save: str | Path | None = None,
        **capture_kwargs,
    ) -> np.ndarray:
        """Capture one panel, or all four as a 2x2 grid, as RGBA uint8.

        Each panel is rendered offscreen, so the result is reproducible and
        does not need a window.

        The capture shows the data **currently resident on the GPU** -- it does
        not reslice.  Capture from :meth:`on_ready` when a load may still be
        running.  A panel with **no canvas** renders empty, because slice
        requests are planned per canvas: give every panel a canvas (the grid
        builder does, and so does ``cellier.convenience.capture``) and let the reslice
        finish before capturing.

        Parameters
        ----------
        panel : str or None
            Which panel to capture: ``"xy"``, ``"xz"``, ``"yz"`` or ``"vol"``.
            With ``None`` (default) all four are captured and composited into
            a 2x2 grid laid out the way ``build_ortho_grid_widget`` arranges
            them -- XY and XZ on the top row, YZ and the 3D volume below.
        size : tuple[int, int] or None
            ``(width, height)`` **per panel**, before *scale*, so a grid
            capture comes back at twice this in each direction.  Defaults to
            each panel canvas's physical size, or ``(600, 600)`` when a panel
            has no canvas.  A grid capture requires one size for all four, so
            it falls back to ``(600, 600)`` unless the panels agree.
        scale : float
            Multiplier applied to *size*.
        frames : int or "converged"
            ``1`` (default) draws a single frame with temporal accumulation
            off.  ``"converged"`` draws the number of frames the accumulator
            needs to settle.  This matters more here than elsewhere: the
            ``vol`` panel has accumulation enabled while the three slice
            panels do not, so a fixed ``frames=N`` is right for one panel and
            wrong for three.
        save : str, Path, or None
            When given, also write the frame to this path as a PNG.
        **capture_kwargs
            Forwarded to the capture helper.

        Returns
        -------
        np.ndarray
            RGBA uint8 array: ``(height, width, 4)`` for one panel, or
            ``(2 * height, 2 * width, 4)`` for the grid.

        Raises
        ------
        ValueError
            If *panel* is not one of the four panel keys.

        Notes
        -----
        The grid composite is **canvases only**.  The Qt grid carries ``XY`` /
        ``XZ`` / ``YZ`` / ``3D`` labels above the panels; those are chrome and
        do not appear here.  Use
        :func:`~cellier.convenience.screenshot_window` for a picture with the
        labels and docks in it.
        """
        if panel is not None:
            if panel not in _PANEL_KEYS:
                raise ValueError(
                    f"Unknown panel {panel!r}. Expected one of {list(_PANEL_KEYS)}."
                )
            frame = self._screenshot_panel(
                panel, size=size, scale=scale, frames=frames, **capture_kwargs
            )
        else:
            frame = self._screenshot_grid(
                size=size, scale=scale, frames=frames, **capture_kwargs
            )
        if save is not None:
            write_png(save, frame)
        return frame

    def _screenshot_panel(
        self,
        panel: str,
        *,
        size: tuple[int, int] | None,
        scale: float,
        frames: int | Literal["converged"],
        **capture_kwargs,
    ) -> np.ndarray:
        """Capture a single panel through its canvas, or by fitting its scene."""
        scene_id = self._scenes[panel].id
        canvas_ids = self._controller.get_canvas_ids(scene_id)
        if canvas_ids:
            return self._controller.screenshot(
                canvas_ids[0], size=size, scale=scale, frames=frames, **capture_kwargs
            )
        return self._controller.screenshot_scene(
            scene_id, size=size, scale=scale, frames=frames, **capture_kwargs
        )

    def _screenshot_grid(
        self,
        *,
        size: tuple[int, int] | None,
        scale: float,
        frames: int | Literal["converged"],
        **capture_kwargs,
    ) -> np.ndarray:
        """Capture all four panels and tile them into the on-screen arrangement."""
        import numpy as np

        panel_size = size if size is not None else self._common_panel_size()
        frames_by_panel = {
            key: self._screenshot_panel(
                key,
                size=panel_size,
                scale=scale,
                frames=frames,
                **capture_kwargs,
            )
            for key in _PANEL_KEYS
        }
        # The arrangement ``build_ortho_grid_widget`` uses, so the composite
        # reads like the window rather than like an arbitrary tiling.
        top = np.hstack([frames_by_panel["xy"], frames_by_panel["xz"]])
        bottom = np.hstack([frames_by_panel["yz"], frames_by_panel["vol"]])
        return np.vstack([top, bottom])

    def _common_panel_size(self) -> tuple[int, int]:
        """Return one ``(width, height)`` all four panels can be captured at.

        A grid needs equal tiles, and the four canvases need not be the same
        size (a user can resize one Qt dock).  When they disagree -- or when
        some panel has no canvas at all -- there is no honest "current" size,
        so the capture default is used rather than silently stretching one
        panel to match another.
        """
        sizes = set()
        for key in _PANEL_KEYS:
            canvas_ids = self._controller.get_canvas_ids(self._scenes[key].id)
            if not canvas_ids:
                return (600, 600)
            view = self._controller.get_canvas_view(canvas_ids[0])
            sizes.add(tuple(int(v) for v in view.widget.get_physical_size()))
        if len(sizes) != 1:
            return (600, 600)
        return sizes.pop()

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def on_ready(self, callback: Callable[[], None]) -> None:
        """Register a callback fired once *all* panels' startup data is on the GPU.

        The callback runs after the initial reslice triggered by
        :func:`~cellier.convenience.launch` / :func:`~cellier.convenience.show`
        has committed every visual across all four panel scenes.  Use it to
        hide a loading indicator, capture a screenshot, or enable controls once
        the first view is fully loaded.

        Must be called before ``launch``/``show``.  For per-scene readiness,
        use :meth:`~cellier.controller.CellierController.on_scene_ready`
        directly.

        Parameters
        ----------
        callback : Callable[[], None]
            Zero-argument callback.
        """
        self._ready_callbacks.append(callback)

    @property
    def spatial_axes(self) -> tuple[int, int, int]:
        """The three spatial axis indices in ``(z, y, x)`` order."""
        return self._spatial_axes

    @property
    def extra_axes(self) -> set[int]:
        """Indices of the non-spatial (extra) axes."""
        return set(self._extra_axes)

    @property
    def dims_controller(self) -> OrthoDimsController | None:
        """The controller mirroring dims across the panels, if linked."""
        return self._dims_controller

    @property
    def axis_sync_enabled(self) -> bool:
        """Whether dims edits are mirrored across the panels."""
        return self._dims_controller is not None and self._dims_controller.enabled

    @axis_sync_enabled.setter
    def axis_sync_enabled(self, value: bool) -> None:
        if self._dims_controller is None:
            if value:
                self._wire_axis_sync()
            return
        self._dims_controller.enabled = value

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_file(self, path: str | Path) -> None:
        """Serialize the orthoviewer model state to a JSON file.

        Captures the four scenes (dims, slice positions), visuals, data stores,
        canvas camera state, and the render pipeline configuration.  The live
        dims mirroring is *not* serialized -- the panels are saved in agreement
        -- and :meth:`from_file` re-establishes it.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        self._controller.to_file(path)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        link_axes: bool = True,
        render_config: RenderManagerConfig | None = None,
    ) -> OrthoViewer:
        """Restore an ``OrthoViewer`` from a previously serialized file.

        The four panels are rebound by scene name (``xy``, ``xz``, ``yz``,
        ``vol``) and the spatial axes are recovered from the ``vol`` panel's
        displayed axes -- no extra metadata is stored.  Dims mirroring is
        re-established when *link_axes* is ``True``.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file written by :meth:`to_file`.
        link_axes : bool
            Re-establish the cross-panel dims mirroring.  Default ``True``.
        render_config : RenderManagerConfig or None
            Override the serialized render pipeline configuration.

        Returns
        -------
        OrthoViewer

        Raises
        ------
        ValueError
            If the file does not contain exactly the four expected panels.
        """
        controller = CellierController.from_file(path, render_config=render_config)
        scenes_by_name = {
            scene.name: scene for scene in controller._model.scenes.values()
        }
        expected = set(_PANEL_KEYS)
        if set(scenes_by_name) != expected or len(controller._model.scenes) != len(
            _PANEL_KEYS
        ):
            raise ValueError(
                "Expected exactly the four orthoviewer panels "
                f"{sorted(expected)} in the file, found "
                f"{sorted(scenes_by_name)}. Use CellierController.from_file "
                "directly for other models."
            )
        scenes = {key: scenes_by_name[key] for key in _PANEL_KEYS}
        vol_displayed = tuple(scenes["vol"].dims.selection.displayed_axes)
        ndim = len(scenes["vol"].dims.axis_labels)

        obj = object.__new__(cls)
        obj._controller = controller
        obj._scenes = scenes
        obj._spatial_axes = vol_displayed  # type: ignore[assignment]
        obj._ndim = ndim
        obj._extra_axes = {i for i in range(ndim) if i not in vol_displayed}
        obj._dims_controller = None
        obj._init_controls_registry()
        if link_axes:
            obj._wire_axis_sync()
        return obj

    # ------------------------------------------------------------------
    # Dims control
    # ------------------------------------------------------------------

    def center_slices(self) -> None:
        """Move the three spatial slice positions to the middle of the data.

        Reads the world-space extent of the loaded visuals and sets every
        panel's position on each spatial axis to its midpoint -- once, through
        the dims controller when the panels are linked.  Extra-axis positions
        are left unchanged.  Call after adding data.

        Raises
        ------
        ValueError
            If no visuals with known shapes have been added yet.
        """
        from cellier.convenience._geometry import axis_values_from_ortho

        ranges = axis_values_from_ortho(self)
        # A slice position is a float world coordinate (D3), so the midpoint
        # is not rounded.  Every panel keeps a position for every axis (D36),
        # so the displayed ones are centred too.
        midpoints = {
            axis: (ranges[axis].min + ranges[axis].max) / 2.0
            for axis in self._spatial_axes
            if axis in ranges
        }
        if not midpoints:
            return
        if self.axis_sync_enabled:
            self._dims_controller.set_slice_positions(midpoints)
            return
        for scene in self._scenes.values():
            self._controller.update_slice_indices(scene.id, midpoints)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_data_store(self, data: _T | UUID) -> _T:
        if isinstance(data, UUID):
            return self._controller.get_data_store(data)  # type: ignore[return-value]
        return data

    def _fan_out(self, add_one) -> dict[str, object]:
        """Call *add_one(key, scene)* for every panel and collect the results."""
        return {key: add_one(key, scene) for key, scene in self._scenes.items()}

    # ------------------------------------------------------------------
    # Visual add methods (one data store, one visual per panel)
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
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> dict[str, ImageVisual]:
        """Add an in-memory image to every panel from a single data store.

        Each panel gets its own visual model with its own copy of the
        appearances.  :meth:`set_image_composite`,
        :meth:`update_image_single_field` and
        :meth:`update_image_channel_field` set all four panels at once.  The
        image controls instead drive the 2D panels and the 3D panel
        separately, so the two can differ.  A composited axis cannot be one
        of the spatial axes, since some panel always displays it (unified
        image design 3.4).

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : InMemoryImageAppearance or None
            Shared by both modes.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        controls : InMemoryImageControlsConfig or None
            Appearance controls configuration, used for two controls: one
            driving the three 2D panels, one the 3D panel.
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

        Returns
        -------
        dict[str, ImageVisual]
            The per-panel visuals keyed ``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``.
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_image(
                store,
                scene.id,
                _copy(appearance),
                f"{name}_{key}",
                single=_copy(single),
                channel_axis=channel_axis,
                composite=composite,
                channels=_copy_channels(channels),
                max_channels=max_channels,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

    def add_labels(
        self,
        data: LabelMemoryStore | UUID,
        appearance: BaseLabelsAppearance | None = None,
        name: str = "labels",
        transform: BaseTransform | None = None,
        controls: LabelsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        outline_selected_labels: dict[int, int] | None = None,
    ) -> dict[str, LabelMemoryVisual]:
        """Add an in-memory label image to every panel from one data store.

        Parameters
        ----------
        data : LabelMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : BaseLabelsAppearance or None
            Appearance parameters.
            Defaults to ``InMemoryLabelsAppearance()`` when ``None``.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : BaseTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.
        controls : LabelsControlsConfig or None
            Appearance controls configuration, used for two controls: one
            drives the three 2D panels' visuals in lock-step, the other the
            3D panel's.  When ``None`` (default), no appearance controls are
            created.

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
        dict[str, LabelMemoryVisual]
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_labels(
                store,
                scene.id,
                appearance,
                f"{name}_{key}",
                transform,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
                outline_selected_labels=outline_selected_labels,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

    def add_mesh(
        self,
        data: MeshMemoryStore | UUID,
        appearance: MeshAppearance,
        name: str = "mesh",
        transform: BaseTransform | None = None,
        controls: MeshControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> dict[str, MeshVisual]:
        """Add a mesh to every panel from a single data store.

        Parameters
        ----------
        data : MeshMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : MeshFlatAppearance, MeshPhongAppearance,
            Appearance parameters.  When passing a dict, include the
            ``appearance_type`` key (``"flat"`` or ``"phong"``).
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : BaseTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.
        controls : MeshControlsConfig or None
            Appearance controls configuration, used for two controls: one
            drives the three 2D panels' visuals in lock-step, the other the
            3D panel's.  When ``None`` (default), no appearance controls are
            created.

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
        dict[str, MeshVisual]
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_mesh(
                store,
                scene.id,
                appearance,
                f"{name}_{key}",
                transform,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

    def add_points(
        self,
        data: PointsMemoryStore | UUID,
        appearance: PointsMarkerAppearance | None = None,
        name: str = "points",
        transform: BaseTransform | None = None,
        controls: PointsControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> dict[str, PointsVisual]:
        """Add a points visual to every panel from a single data store.

        Parameters
        ----------
        data : PointsMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : PointsMarkerAppearance or None
            Appearance parameters.
            Defaults to ``PointsMarkerAppearance()`` when ``None``.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : BaseTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.
        controls : PointsControlsConfig or None
            Appearance controls configuration, used for two controls: one
            drives the three 2D panels' visuals in lock-step, the other the
            3D panel's.  When ``None`` (default), no appearance controls are
            created.

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
        dict[str, PointsVisual]
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_points(
                store,
                scene.id,
                appearance,
                f"{name}_{key}",
                transform,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

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
    ) -> dict[str, GraphVisual]:
        """Add a spatial-graph visual to every panel from a single data store.

        Parameters
        ----------
        data : GraphMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : GraphAppearance or None
            Appearance parameters. Defaults
            to ``GraphAppearance()`` when ``None``.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : BaseTransform or None
            Data-to-world transform. Falls back to the store's own transform,
            then to identity.
        trail : dict[int, TrailConfig] or None
            Axis index -> window configuration, applied to every panel.
        controls : GraphControlsConfig or None
            Appearance controls configuration, used for two controls: one
            drives the three 2D panels' visuals in lock-step, the other the
            3D panel's.  When ``None`` (default), no appearance controls are
            created.

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
        dict[str, GraphVisual]
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_graph(
                store,
                scene.id,
                appearance,
                f"{name}_{key}",
                transform,
                trail,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

    def add_lines(
        self,
        data: LinesMemoryStore | UUID,
        appearance: LinesMemoryAppearance | None = None,
        name: str = "lines",
        transform: BaseTransform | None = None,
        controls: LinesControlsConfig | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
    ) -> dict[str, LinesVisual]:
        """Add a lines visual to every panel from a single data store.

        Parameters
        ----------
        data : LinesMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : LinesMemoryAppearance or None
            Appearance parameters.
            Defaults to ``LinesMemoryAppearance()`` when ``None``.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : BaseTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.
        controls : LinesControlsConfig or None
            Appearance controls configuration, used for two controls: one
            drives the three 2D panels' visuals in lock-step, the other the
            3D panel's.  When ``None`` (default), no appearance controls are
            created.

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
        dict[str, LinesVisual]
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_lines(
                store,
                scene.id,
                appearance,
                f"{name}_{key}",
                transform,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

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
    ) -> dict[str, MultiscaleImageVisual]:
        """Add a multiscale image to every panel from a single data store.

        See :meth:`add_image` for the modes and the group methods.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale store or UUID of an already-registered store.
        appearance : MultiscaleImageAppearance or None
            Shared by both modes.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        render_config : MultiscaleImageRenderConfig or None
            GPU cache configuration.  Uses defaults when ``None``.
        transform : BaseTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.
        controls : MultiscaleImageControlsConfig or None
            Appearance controls configuration, used for two controls: one
            driving the three 2D panels, one the 3D panel.
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

        Returns
        -------
        dict[str, MultiscaleImageVisual]
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_image_multiscale(
                store,
                scene.id,
                _copy(appearance),
                f"{name}_{key}",
                render_config,
                transform,
                single=_copy(single),
                channel_axis=channel_axis,
                composite=composite,
                channels=_copy_channels(channels),
                max_channels=max_channels,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

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
        outline_selected_labels: dict[int, int] | None = None,
    ) -> dict[str, MultiscaleLabelVisual]:
        """Add a multiscale label image to every panel from one data store.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale label store or UUID of an already-registered store.
        appearance : MultiscaleLabelsAppearance
            Appearance parameters.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        render_config : MultiscaleLabelRenderConfig or None
            LOD and rendering configuration.  Uses defaults when ``None``.
        transform : BaseTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.
        controls : MultiscaleLabelsControlsConfig or None
            Appearance controls configuration, used for two controls: one
            drives the three 2D panels' visuals in lock-step, the other the
            3D panel's.  When ``None`` (default), no appearance controls are
            created.

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
        dict[str, MultiscaleLabelVisual]
        """
        store = self._resolve_data_store(data)
        visuals = self._fan_out(
            lambda key, scene: self._controller.add_labels_multiscale(
                store,
                scene.id,
                appearance,
                f"{name}_{key}",
                render_config,
                transform,
                outline=outline,
                ambient_occlusion=ambient_occlusion,
                outline_selected_labels=outline_selected_labels,
            )
        )
        self._record_controls(visuals, controls, name)
        return visuals

    # ------------------------------------------------------------------
    # Image group methods (mode and settings mirrored across the panels)
    #
    # These set all four panels at once and are for scripts.  The image
    # controls do not use them: each drives its own group (the 2D panels or
    # the 3D panel) through the bus.
    # ------------------------------------------------------------------

    def image_group(self, visual: object) -> list[UUID]:
        """The four panel siblings of an image added with ``add_image*``.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's visual (or its id), or the dict ``add_image*``
            returned.

        Returns
        -------
        list[UUID]
            The sibling visual ids, in panel order.

        Raises
        ------
        KeyError
            If *visual* is not a panel image of this viewer.
        """
        if isinstance(visual, dict):
            return [v.id for v in visual.values()]
        visual_id = getattr(visual, "id", visual)
        model = self._controller.get_visual_model(visual_id)
        base = str(model.name).rsplit("_", 1)[0]
        group = []
        for key in _PANEL_KEYS:
            for candidate in self._scenes[key].visuals:
                if candidate.name == f"{base}_{key}" and type(candidate) is type(model):
                    group.append(candidate.id)
        if visual_id not in group:
            raise KeyError(f"{visual_id} is not a panel image of this OrthoViewer.")
        return group

    # ------------------------------------------------------------------
    # Progressive loading (multiscale visuals), over a panel group
    # ------------------------------------------------------------------

    def loading_progress(self, visual: object) -> LoadingProgress | None:
        """How far a multiscale visual's four panels have loaded, summed.

        Mirrors :meth:`CellierController.loading_progress` over the panel
        group: counts are added and the completion flags hold only when they
        hold for every panel.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's multiscale visual; see :meth:`image_group`.

        Returns
        -------
        LoadingProgress or None
            ``None`` while no panel has been planned.
        """
        from cellier.gui._loading import sum_progress

        progress = [
            self._controller.loading_progress(vid) for vid in self.image_group(visual)
        ]
        return sum_progress(p for p in progress if p is not None)

    def on_reslice_progress(
        self,
        visual: object,
        callback: Callable[[ResliceProgressEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> list[SubscriptionHandle]:
        """Register a callback fired as a multiscale visual's panels load.

        Mirrors :meth:`CellierController.on_reslice_progress` over the panel
        group.  The callback gets each panel's event with ``progress``
        replaced by the group sum (:meth:`loading_progress`); ``visual_id``
        and ``scene_id`` name the panel whose change triggered it.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's multiscale visual; see :meth:`image_group`.
        callback : Callable
            Called with a ``ResliceProgressEvent``.
        owner_id : UUID or None
            Owner for ``controller.unsubscribe_owner``.  Defaults to each
            panel visual's id, so removing a panel's visual removes its
            subscription.
        weak : bool
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        list[SubscriptionHandle]
            One per panel.
        """
        group = self.image_group(visual)
        callback_ref = _callback_ref(callback, weak)

        def _on_progress(event: ResliceProgressEvent) -> None:
            target = callback_ref()
            if target is None:
                return
            total = self.loading_progress(group[0])
            target(event._replace(progress=total) if total is not None else event)

        return [
            self._controller.on_reslice_progress(
                vid, _on_progress, owner_id=vid if owner_id is None else owner_id
            )
            for vid in group
        ]

    def on_backstop_complete(
        self,
        visual: object,
        callback: Callable[[BackstopCompleteEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> list[SubscriptionHandle]:
        """Register a callback fired when every panel's backstop is in.

        Mirrors :meth:`CellierController.on_backstop_complete` over the
        panel group: fires on the panel event that completes the group, with
        that panel's event.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's multiscale visual; see :meth:`image_group`.
        callback : Callable
            Called with a ``BackstopCompleteEvent``.
        owner_id : UUID or None
            Owner for ``controller.unsubscribe_owner``.  Defaults to each
            panel visual's id.
        weak : bool
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        list[SubscriptionHandle]
            One per panel.
        """
        group = self.image_group(visual)
        callback_ref = _callback_ref(callback, weak)

        def _on_backstop(event: BackstopCompleteEvent) -> None:
            target = callback_ref()
            if target is None:
                return
            progress = [self._controller.loading_progress(vid) for vid in group]
            if all(p is not None and p.backstop_complete for p in progress):
                target(event)

        return [
            self._controller.on_backstop_complete(
                vid, _on_backstop, owner_id=vid if owner_id is None else owner_id
            )
            for vid in group
        ]

    def set_loading(self, visual: object, **fields: Any) -> ProgressiveLoadingConfig:
        """Change how every panel of a multiscale visual loads.

        Mirrors :meth:`CellierController.set_loading_config`, applied to each
        panel.  The merged config is validated before any panel changes, so
        an invalid combination raises and changes nothing.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's multiscale visual; see :meth:`image_group`.
        **fields :
            ``ProgressiveLoadingConfig`` fields, e.g. ``dims_drag="backstop"``.

        Returns
        -------
        ProgressiveLoadingConfig
            The first panel's config after the call.
        """
        group = self.image_group(visual)
        # The first write validates; the rest cannot fail differently, since
        # the panels are kept equal.
        result = self._controller.set_loading_config(group[0], **fields)
        for vid in group[1:]:
            self._controller.set_loading_config(vid, **fields)
        return result

    def set_image_composite(self, visual: object, composite: bool) -> None:
        """Switch every panel's image between single and composite mode.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's visual; see :meth:`image_group`.
        composite : bool
            ``True`` for composite mode.
        """
        self._controller.set_image_composite_group(self.image_group(visual), composite)

    def update_image_single_field(self, visual: object, field: str, value) -> None:
        """Set one single-mode field on every panel's image.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's visual; see :meth:`image_group`.
        field : str
            Field name on ``single``.
        value :
            New value.
        """
        self._controller.update_single_group_field(
            self.image_group(visual), field, value
        )

    def update_image_channel_field(
        self, visual: object, channel_index: int, field: str, value
    ) -> None:
        """Set one channel field on every panel's image.

        Parameters
        ----------
        visual : UUID, visual model, or dict
            Any panel's visual; see :meth:`image_group`.
        channel_index : int
            The channel.
        field : str
            Field name on the channel appearance.
        value :
            New value.
        """
        self._controller.update_channel_group_field(
            self.image_group(visual), channel_index, field, value
        )

    def _record_controls(
        self,
        visuals: dict[str, object],
        controls: BaseControlsConfig | None,
        name: str,
    ) -> None:
        """Record a controls config for a fanned-out add, as two groups.

        One group holds the three 2D panels' visuals and one the 3D panel's
        (:data:`_CONTROLS_GROUPS`), both with the same config, so one widget
        drives the 2D views together and another the 3D view.  Each is
        labelled for the dock's selector, e.g. ``"image (2D views)"``.

        The appearance docks resolve this record through
        ``appearance_targets``, which is what makes ``AppearanceControls()``
        work on an ``OrthoViewer`` at all (section 4.1).
        """
        for keys, label in _CONTROLS_GROUPS.values():
            self._store_controls(
                [visuals[key].id for key in keys],
                controls,
                label=label.format(name=name),
            )
