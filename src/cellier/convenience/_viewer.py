"""Single-scene convenience viewer wrapping CellierController."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, TypeVar
from uuid import UUID

from cellier.controller import CellierController
from cellier.scene.dims import CoordinateSystem

if TYPE_CHECKING:
    from pathlib import Path

    from PySide6.QtWidgets import QWidget

    from cellier.convenience._kwarg_dicts import (
        ChannelAppearanceKwargs,
        ChannelControlsKwargs,
        GraphAppearanceKwargs,
        InMemoryImageAppearanceKwargs,
        InMemoryImageControlsKwargs,
        InMemoryLabelsAppearanceKwargs,
        LinesMemoryAppearanceKwargs,
        MeshFlatAppearanceKwargs,
        MeshPhongAppearanceKwargs,
        MultiscaleImageAppearanceKwargs,
        MultiscaleImageControlsKwargs,
        MultiscaleImageRenderConfigKwargs,
        MultiscaleLabelRenderConfigKwargs,
        MultiscaleLabelsAppearanceKwargs,
        PointsMarkerAppearanceKwargs,
    )
    from cellier.convenience.gui._controls_config import (
        BaseControlsConfig,
        ChannelControlsConfig,
        InMemoryImageControlsConfig,
        MultiscaleImageControlsConfig,
    )
    from cellier.data._base_data_store import BaseDataStore
    from cellier.data.graph._graph_memory_store import GraphMemoryStore
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.data.label._label_memory_store import LabelMemoryStore
    from cellier.data.lines._lines_memory_store import LinesMemoryStore
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.data.points._points_memory_store import PointsMemoryStore
    from cellier.render._config import RenderManagerConfig
    from cellier.scene.scene import Scene
    from cellier.transform import AffineTransform
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


class Viewer:
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
    gui : "qt" or "anywidget"
        Which GUI toolkit the canvas should target. ``"qt"`` (default) renders
        into a Qt widget; ``"anywidget"`` renders into a notebook canvas for
        Jupyter / marimo. Fixed at construction.
    """

    def __init__(
        self,
        axis_labels: tuple[str, ...],
        *,
        dim: Literal["2d", "3d"] = "2d",
        render_modes: set[str] | None = None,
        render_config: RenderManagerConfig | None = None,
        gui: Literal["qt", "anywidget"] = "qt",
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
        appearance: BaseImageAppearance | InMemoryImageAppearanceKwargs,
        name: str = "image",
        controls: InMemoryImageControlsConfig
        | InMemoryImageControlsKwargs
        | None = None,
    ) -> ImageVisual:
        """Add an in-memory image visual.

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : BaseImageAppearance or dict
            Appearance parameters. Accepts an ``InMemoryImageAppearance``
            instance or a plain dict with the same keys (see
            ``InMemoryImageAppearanceKwargs`` for the full set of accepted
            keys and their types).
        name : str
            Human-readable label. Default ``"image"``.
        controls : InMemoryImageControlsConfig, dict, or None
            Appearance panel configuration. Accepts an
            ``InMemoryImageControlsConfig`` instance or a plain dict with the
            same keys (see ``InMemoryImageControlsKwargs``). When ``None``
            (default), no appearance panel is created for this visual.

        Returns
        -------
        ImageVisual
        """
        from cellier.convenience.gui._controls_config import (
            resolve_inmemory_image_controls,
        )
        from cellier.visuals._image_memory import InMemoryImageAppearance

        resolved_appearance = (
            InMemoryImageAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        visual = self._controller.add_image(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
        )
        resolved_controls = resolve_inmemory_image_controls(controls)
        if resolved_controls is not None:
            self._controls_configs[visual.id] = resolved_controls
        return visual

    def add_labels(
        self,
        data: LabelMemoryStore | UUID,
        appearance: BaseLabelsAppearance | InMemoryLabelsAppearanceKwargs | None = None,
        name: str = "labels",
        transform: AffineTransform | None = None,
    ) -> LabelMemoryVisual:
        """Add an in-memory label visual.

        Parameters
        ----------
        data : LabelMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : BaseLabelsAppearance, dict, or None
            Appearance parameters. Accepts an ``InMemoryLabelsAppearance``
            instance or a plain dict with the same keys (see
            ``InMemoryLabelsAppearanceKwargs`` for the full set of accepted
            keys and their types). Defaults to ``InMemoryLabelsAppearance()``
            when ``None``.
        name : str
            Human-readable label. Default ``"labels"``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.

        Returns
        -------
        LabelMemoryVisual
        """
        from cellier.visuals._label_memory import InMemoryLabelsAppearance

        resolved_appearance = (
            InMemoryLabelsAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._controller.add_labels(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
            transform,
        )

    def add_mesh(
        self,
        data: MeshMemoryStore | UUID,
        appearance: MeshAppearance
        | MeshFlatAppearanceKwargs
        | MeshPhongAppearanceKwargs,
        name: str = "mesh",
        transform: AffineTransform | None = None,
    ) -> MeshVisual:
        """Add a mesh visual.

        Parameters
        ----------
        data : MeshMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : MeshFlatAppearance, MeshPhongAppearance, or dict
            Appearance parameters. Accepts a ``MeshFlatAppearance`` or
            ``MeshPhongAppearance`` instance, or a plain dict with the same
            keys (see ``MeshFlatAppearanceKwargs`` and
            ``MeshPhongAppearanceKwargs``). When passing a dict, include the
            ``appearance_type`` key (``"flat"`` or ``"phong"``) so Pydantic
            can select the correct variant.
        name : str
            Human-readable label. Default ``"mesh"``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.

        Returns
        -------
        MeshVisual
        """
        from pydantic import TypeAdapter

        from cellier.visuals._mesh_memory import MeshAppearance as _MeshAppearance

        resolved_appearance = (
            TypeAdapter(_MeshAppearance).validate_python(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._controller.add_mesh(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
            transform,
        )

    def add_points(
        self,
        data: PointsMemoryStore | UUID,
        appearance: PointsMarkerAppearance | PointsMarkerAppearanceKwargs | None = None,
        name: str = "points",
        transform: AffineTransform | None = None,
    ) -> PointsVisual:
        """Add a points visual.

        Parameters
        ----------
        data : PointsMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : PointsMarkerAppearance, dict, or None
            Appearance parameters. Accepts a ``PointsMarkerAppearance``
            instance or a plain dict with the same keys (see
            ``PointsMarkerAppearanceKwargs`` for the full set of accepted
            keys and their types). Defaults to ``PointsMarkerAppearance()``
            when ``None``.
        name : str
            Human-readable label. Default ``"points"``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.

        Returns
        -------
        PointsVisual
        """
        from cellier.visuals._points_memory import (
            PointsMarkerAppearance as _PointsMarkerAppearance,
        )

        resolved_appearance = (
            _PointsMarkerAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._controller.add_points(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
            transform,
        )

    def add_graph(
        self,
        data: GraphMemoryStore | UUID,
        appearance: GraphAppearance | GraphAppearanceKwargs | None = None,
        name: str = "graph",
        transform: AffineTransform | None = None,
        trail: dict[int, TrailConfig] | None = None,
    ) -> GraphVisual:
        """Add a spatial-graph visual.

        Parameters
        ----------
        data : GraphMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : GraphAppearance, dict, or None
            Appearance parameters. Accepts a ``GraphAppearance`` instance or
            a plain dict with the same keys (see ``GraphAppearanceKwargs``
            for the full set of accepted keys and their types). Defaults to
            ``GraphAppearance()`` when ``None``.
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

        Returns
        -------
        GraphVisual
        """
        from cellier.visuals._graph_memory import (
            GraphAppearance as _GraphAppearance,
        )

        resolved_appearance = (
            _GraphAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._controller.add_graph(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
            transform,
            trail,
        )

    def add_lines(
        self,
        data: LinesMemoryStore | UUID,
        appearance: LinesMemoryAppearance | LinesMemoryAppearanceKwargs | None = None,
        name: str = "lines",
        transform: AffineTransform | None = None,
    ) -> LinesVisual:
        """Add a lines visual.

        Parameters
        ----------
        data : LinesMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : LinesMemoryAppearance, dict, or None
            Appearance parameters. Accepts a ``LinesMemoryAppearance``
            instance or a plain dict with the same keys (see
            ``LinesMemoryAppearanceKwargs`` for the full set of accepted
            keys and their types). Defaults to ``LinesMemoryAppearance()``
            when ``None``.
        name : str
            Human-readable label. Default ``"lines"``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.

        Returns
        -------
        LinesVisual
        """
        from cellier.visuals._lines_memory import (
            LinesMemoryAppearance as _LinesMemoryAppearance,
        )

        resolved_appearance = (
            _LinesMemoryAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._controller.add_lines(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
            transform,
        )

    def add_image_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleImageAppearance | MultiscaleImageAppearanceKwargs,
        name: str = "image",
        render_config: MultiscaleImageRenderConfig
        | MultiscaleImageRenderConfigKwargs
        | None = None,
        transform: AffineTransform | None = None,
        controls: MultiscaleImageControlsConfig
        | MultiscaleImageControlsKwargs
        | None = None,
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale data store or UUID of an already-registered store.
        appearance : MultiscaleImageAppearance or dict
            Visual appearance parameters. Accepts a
            ``MultiscaleImageAppearance`` instance or a plain dict with the
            same keys (see ``MultiscaleImageAppearanceKwargs`` for the full
            set of accepted keys and their types).
        name : str
            Human-readable label. Default ``"image"``.
        render_config : MultiscaleImageRenderConfig, dict, or None
            LOD and rendering configuration. Accepts a
            ``MultiscaleImageRenderConfig`` instance or a plain dict with the
            same keys (see ``MultiscaleImageRenderConfigKwargs``). Uses
            defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        controls : MultiscaleImageControlsConfig, dict, or None
            Appearance panel configuration. Accepts a
            ``MultiscaleImageControlsConfig`` instance or a plain dict with
            the same keys (see ``MultiscaleImageControlsKwargs``). When
            ``None`` (default), no appearance panel is created for this
            visual.

        Returns
        -------
        MultiscaleImageVisual
        """
        from cellier.convenience.gui._controls_config import (
            resolve_multiscale_image_controls,
        )
        from cellier.visuals._image import (
            MultiscaleImageAppearance as _MultiscaleImageAppearance,
        )
        from cellier.visuals._image import (
            MultiscaleImageRenderConfig as _MultiscaleImageRenderConfig,
        )

        resolved_appearance = (
            _MultiscaleImageAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        resolved_render_config = (
            _MultiscaleImageRenderConfig.model_validate(render_config)
            if isinstance(render_config, dict)
            else render_config
        )
        visual = self._controller.add_image_multiscale(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
            resolved_render_config,
            transform,
        )
        resolved_controls = resolve_multiscale_image_controls(controls)
        if resolved_controls is not None:
            self._controls_configs[visual.id] = resolved_controls
        return visual

    def add_labels_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleLabelsAppearance | MultiscaleLabelsAppearanceKwargs,
        name: str = "labels",
        render_config: MultiscaleLabelRenderConfig
        | MultiscaleLabelRenderConfigKwargs
        | None = None,
        transform: AffineTransform | None = None,
    ) -> MultiscaleLabelVisual:
        """Add a multiscale label visual.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale label store or UUID of an already-registered store.
        appearance : MultiscaleLabelsAppearance or dict
            Visual appearance parameters. Accepts a
            ``MultiscaleLabelsAppearance`` instance or a plain dict with the
            same keys (see ``MultiscaleLabelsAppearanceKwargs`` for the full
            set of accepted keys and their types).
        name : str
            Human-readable label. Default ``"labels"``.
        render_config : MultiscaleLabelRenderConfig, dict, or None
            LOD and rendering configuration. Accepts a
            ``MultiscaleLabelRenderConfig`` instance or a plain dict with the
            same keys (see ``MultiscaleLabelRenderConfigKwargs``). Uses
            defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.

        Returns
        -------
        MultiscaleLabelVisual
        """
        from cellier.visuals._labels import (
            MultiscaleLabelRenderConfig as _MultiscaleLabelRenderConfig,
        )
        from cellier.visuals._labels import (
            MultiscaleLabelsAppearance as _MultiscaleLabelsAppearance,
        )

        resolved_appearance = (
            _MultiscaleLabelsAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        resolved_render_config = (
            _MultiscaleLabelRenderConfig.model_validate(render_config)
            if isinstance(render_config, dict)
            else render_config
        )
        return self._controller.add_labels_multiscale(
            self._resolve_data_store(data),
            self._scene.id,
            resolved_appearance,
            name,
            resolved_render_config,
            transform,
        )

    def add_multichannel_image(
        self,
        data: ImageMemoryStore | UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance | ChannelAppearanceKwargs],
        name: str = "multichannel_image",
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
        controls: ChannelControlsConfig | ChannelControlsKwargs | None = None,
    ) -> MultichannelImageVisual:
        """Add an in-memory multichannel image visual.

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or UUID of an already-registered store.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance or dict]
            Per-channel appearance keyed by channel index. Each value may be
            a ``ChannelAppearance`` instance or a plain dict with the same
            keys (see ``ChannelAppearanceKwargs`` for the full set of
            accepted keys and their types).
        name : str
            Display name. Default ``"multichannel_image"``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.
        controls : ChannelControlsConfig, dict, or None
            Per-channel controls configuration. Accepts a
            ``ChannelControlsConfig`` instance or a plain dict with the same
            keys (see ``ChannelControlsKwargs``). When ``None`` (default), no
            channel controls are created for this visual.

        Returns
        -------
        MultichannelImageVisual
        """
        from cellier.convenience.gui._controls_config import resolve_channel_controls
        from cellier.visuals._channel_appearance import (
            ChannelAppearance as _ChannelAppearance,
        )

        resolved_channels = {
            k: (_ChannelAppearance.model_validate(v) if isinstance(v, dict) else v)
            for k, v in channels.items()
        }
        visual = self._controller.add_multichannel_image(
            self._resolve_data_store(data),
            self._scene.id,
            channel_axis,
            resolved_channels,
            name,
            max_channels_2d,
            max_channels_3d,
        )
        resolved_controls = resolve_channel_controls(controls)
        if resolved_controls is not None:
            self._controls_configs[visual.id] = resolved_controls
        return visual

    def add_multichannel_image_multiscale(
        self,
        data: BaseDataStore | UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance | ChannelAppearanceKwargs],
        name: str = "multichannel_image",
        render_config: MultiscaleImageRenderConfig
        | MultiscaleImageRenderConfigKwargs
        | None = None,
        transform: AffineTransform | None = None,
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
        controls: ChannelControlsConfig | ChannelControlsKwargs | None = None,
    ) -> MultichannelMultiscaleImageVisual:
        """Add a multiscale multichannel image visual.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale store or UUID of an already-registered store.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance or dict]
            Per-channel appearance keyed by channel index. Each value may be
            a ``ChannelAppearance`` instance or a plain dict with the same
            keys (see ``ChannelAppearanceKwargs`` for the full set of
            accepted keys and their types).
        name : str
            Display name. Default ``"multichannel_image"``.
        render_config : MultiscaleImageRenderConfig, dict, or None
            LOD and rendering configuration. Accepts a
            ``MultiscaleImageRenderConfig`` instance or a plain dict with the
            same keys (see ``MultiscaleImageRenderConfigKwargs``). Uses
            defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when ``None``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.
        controls : ChannelControlsConfig, dict, or None
            Per-channel controls configuration. Accepts a
            ``ChannelControlsConfig`` instance or a plain dict with the same
            keys (see ``ChannelControlsKwargs``). When ``None`` (default), no
            channel controls are created for this visual.

        Returns
        -------
        MultichannelMultiscaleImageVisual
        """
        from cellier.convenience.gui._controls_config import resolve_channel_controls
        from cellier.visuals._channel_appearance import (
            ChannelAppearance as _ChannelAppearance,
        )
        from cellier.visuals._image import (
            MultiscaleImageRenderConfig as _MultiscaleImageRenderConfig,
        )

        resolved_channels = {
            k: (_ChannelAppearance.model_validate(v) if isinstance(v, dict) else v)
            for k, v in channels.items()
        }
        resolved_render_config = (
            _MultiscaleImageRenderConfig.model_validate(render_config)
            if isinstance(render_config, dict)
            else render_config
        )
        visual = self._controller.add_multichannel_image_multiscale(
            self._resolve_data_store(data),
            self._scene.id,
            channel_axis,
            resolved_channels,
            name,
            resolved_render_config,
            transform,
            max_channels_2d,
            max_channels_3d,
        )
        resolved_controls = resolve_channel_controls(controls)
        if resolved_controls is not None:
            self._controls_configs[visual.id] = resolved_controls
        return visual
