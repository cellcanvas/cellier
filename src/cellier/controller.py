"""CellierController is the coordinator between the models and the rendered views."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal
from uuid import UUID, uuid4

import numpy as np

from cellier.events import (
    AABBChangedEvent,
    AABBUpdateEvent,
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    CameraChangedEvent,
    CanvasSizeChangedEvent,
    ChannelAppearanceChangedEvent,
    ChannelAppearanceUpdateEvent,
    DimsChangedEvent,
    DimsUpdateEvent,
    EventBus,
    FrameRenderedEvent,
    PickWriteChangedEvent,
    ResliceCompletedEvent,
    ResliceStartedEvent,
    SceneAddedEvent,
    SceneRemovedEvent,
    SubscriptionHandle,
    SubscriptionSpec,
    TransformChangedEvent,
    VisualAddedEvent,
    VisualRemovedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMouseMove3DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    CanvasMouseRelease3DEvent,
    _CanvasRawPointerEvent,
)
from cellier.logging import _CAMERA_LOGGER, _SOURCE_ID_LOGGER
from cellier.render._config import RenderManagerConfig
from cellier.render._scene_config import VisualRenderConfig
from cellier.render._visual_lut import (
    KIND_LABEL,
    KIND_WHOLE_OBJECT,
    PLACEMENT_OUTWARD,
)
from cellier.render.render_manager import RenderManager
from cellier.render.visuals._canvas_overlay import GFXCenteredAxes2D
from cellier.render.visuals._image import GFXMultiscaleImageVisual
from cellier.render.visuals._image_memory import GFXImageMemoryVisual
from cellier.render.visuals._image_memory_multichannel import (
    GFXMultichannelImageMemoryVisual,
)
from cellier.render.visuals._image_multiscale_multichannel import (
    GFXMultichannelMultiscaleImageVisual,
)
from cellier.render.visuals._label_memory import GFXLabelMemoryVisual
from cellier.render.visuals._label_multiscale import GFXMultiscaleLabelVisual
from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
from cellier.render.visuals._points_memory import GFXPointsMemoryVisual
from cellier.scene.cameras import (
    CameraType,
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
)
from cellier.scene.canvas import Canvas
from cellier.scene.dims import AxisAlignedSelection, CoordinateSystem, DimsManager
from cellier.scene.scene import Scene
from cellier.transform import AffineTransform
from cellier.viewer_model import DataManager, ViewerModel

if TYPE_CHECKING:
    from cellier.visuals._base_visual import BaseVisual
from cellier.visuals._canvas_overlay import CenteredAxes2D
from cellier.visuals._image import (
    MultichannelMultiscaleImageVisual,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageVisual,
)
from cellier.visuals._image_memory import (
    BaseImageAppearance,
    ImageVisual,
    MultichannelImageVisual,
)
from cellier.visuals._label_memory import BaseLabelsAppearance, LabelMemoryVisual
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    MultiscaleLabelVisual,
)
from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
from cellier.visuals._mesh_memory import (
    MeshAppearance,
    MeshPhongAppearance,
    MeshVisual,
)
from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual

if TYPE_CHECKING:
    import pathlib

    from psygnal import EmissionInfo
    from PySide6.QtWidgets import QWidget

    from cellier._state import CameraState, DimsState
    from cellier.data._base_data_store import BaseDataStore
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.data.label._label_memory_store import LabelMemoryStore
    from cellier.data.lines._lines_memory_store import LinesMemoryStore
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.data.points._points_memory_store import PointsMemoryStore
    from cellier.gui._protocol import WidgetView
    from cellier.render._config import RenderManagerConfig
    from cellier.render.canvas_view import CanvasView
    from cellier.visuals._canvas_overlay import CanvasOverlay
    from cellier.visuals._channel_appearance import ChannelAppearance
    from cellier.visuals._types import VisualType


# Appearance fields that require a reslice (not just a GPU material update).
_RESLICE_FIELDS: frozenset[str] = frozenset({"lod_bias", "force_level", "frustum_cull"})

# Context variable used by update_slice_indices / update_appearance_field to
# thread a caller-supplied source_id through the synchronous psygnal bridge.
# Default None means the bridge falls back to the controller's own ID.
_source_id_override: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "_source_id_override", default=None
)

# Parallel context variable for update_aabb_field / _make_aabb_handler.
_aabb_source_id_override: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "_aabb_source_id_override", default=None
)


class CellierController:
    """The main class for constructing and controlling a cellier visualization.

    Wraps a ViewerModel (model layer) and a RenderManager (render layer) and
    performs the synchronization between both.
    """

    def __init__(
        self,
        widget_parent: object | None = None,
        render_config: RenderManagerConfig | None = None,
        gui: Literal["qt", "anywidget"] = "qt",
    ) -> None:
        # gui selects the render canvas toolkit ("qt" or "anywidget"); threaded
        # to CanvasView via add_canvas.  widget_parent is Qt-only and ignored
        # for the anywidget gui (notebook canvases have no Qt parent).
        self._gui = gui
        self._widget_parent = widget_parent
        self._model = ViewerModel(data=DataManager())
        self._render_manager = RenderManager(config=render_config)
        # Keep model.render_config pointing at the same object so runtime
        # mutations (e.g. camera_settle_threshold_s) are captured on serialize.
        self._model.render_config = self._render_manager._config
        # Reverse map: visual_model_id → scene_id (model layer mirror of render layer)
        self._visual_to_scene: dict[UUID, UUID] = {}
        # Reverse map: canvas_id → scene_id
        self._canvas_to_scene: dict[UUID, UUID] = {}
        # Forward map: scene_id → list[canvas_id]
        self._scene_to_canvases: dict[UUID, list[UUID]] = {}
        # Event buses
        self._id: UUID = uuid4()
        self._outgoing_events: EventBus = EventBus()
        self._incoming_events: EventBus = EventBus()
        # Cache of last-known displayed_axes per scene for change detection
        self._dims_cache: dict[UUID, tuple[int, ...]] = {}
        # render_modes registered per scene (determines which nodes visuals build)
        self._scene_render_modes: dict[UUID, set[Literal["2d", "3d"]]] = {}
        # Camera settle
        self._settle_tasks: dict[UUID, asyncio.Task] = {}
        # Per-canvas count of active element-picking subscribers (the six
        # on_mouse_* registrations).  Drives RenderManager pick-detail gating
        # (Decision 4).  Keyed by canvas_id.
        self._pick_subscriber_counts: dict[UUID, int] = {}
        # Map a mouse SubscriptionHandle's id to its canvas so that
        # unsubscribe_mouse can decrement the right counter.
        self._mouse_handle_canvas: dict[int, UUID] = {}
        # Stored psygnal bridge handlers keyed by visual_id.
        # Each entry is a list of (signal, handler) pairs.  psygnal
        # disconnect() requires the exact handler object; closures are not
        # equality-comparable by value, so references must be retained.
        # Storing the signal alongside the handler avoids branching on visual
        # type during teardown.
        self._visual_psygnal_handlers: dict[UUID, list[tuple]] = {}
        # When True, transform-change handlers skip reslice_scene.  Managed
        # by the suppress_reslice context manager.  This is a flat boolean, so
        # nested suppress_reslice calls or concurrent async transform mutations
        # will interfere — use a depth counter if that ever becomes necessary.
        self._suppress_reslice: bool = False
        self._outgoing_events.subscribe(
            CameraChangedEvent,
            self._on_camera_changed,
            owner_id=self._id,
        )
        self._outgoing_events.subscribe(
            CanvasSizeChangedEvent,
            self._on_canvas_size_changed,
            owner_id=self._id,
        )
        # Must be called before subscribing _on_dims_changed_bus below so the
        # SliceCoordinator invalidates stale 2D caches before the controller
        # submits new slice requests.
        self._render_manager.connect_event_bus(self._outgoing_events)
        # Controller's own dims handler: reslice the affected scene.
        self._outgoing_events.subscribe(
            DimsChangedEvent,
            self._on_dims_changed_bus,
            owner_id=self._id,
        )
        # Subscribe to internal raw pointer events emitted by RenderManager.
        self._outgoing_events.subscribe(
            _CanvasRawPointerEvent,
            self._on_raw_pointer_event,
            owner_id=self._id,
        )
        # Incoming bus: GUI update events dispatched to update_* methods.
        self._incoming_events.subscribe(
            AppearanceUpdateEvent,
            self._on_appearance_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            DimsUpdateEvent,
            self._on_dims_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            AABBUpdateEvent,
            self._on_aabb_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            ChannelAppearanceUpdateEvent,
            self._on_channel_appearance_update,
            owner_id=self._id,
        )

    @property
    def incoming_events(self) -> EventBus:
        """Incoming event bus for GUI-driven model mutations.

        Emit ``AppearanceUpdateEvent``, ``DimsUpdateEvent``, or
        ``AABBUpdateEvent`` onto this bus to request model changes.
        The controller dispatches each event to the corresponding
        ``update_*`` method, preserving ``source_id`` end-to-end.
        """
        return self._incoming_events

    def set_widget_parent(self, parent: object) -> None:
        """Set the Qt parent for subsequently created canvas widgets.

        Only meaningful when ``self._gui == "qt"``; the anywidget gui ignores
        the parent (notebook canvases are not laid out by a Qt parent).
        """
        self._widget_parent = parent

    # ------------------------------------------------------------------
    # Construction class methods (stubs)
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: ViewerModel,
        widget_parent: QWidget | None = None,
        render_config: RenderManagerConfig | None = None,
    ) -> CellierController:
        """Construct a controller from a serialized ViewerModel.

        Iteratively adds all data stores, scenes, visuals, and canvases
        through the public API. The order is:

        1. Data stores  — registered before visuals reference them.
        2. Scenes       — registered with render_modes and lighting from model.
        3. Visuals      — added per scene; data stores must already be present.
        4. Canvases     — restored with camera state from model.

        Parameters
        ----------
        model : ViewerModel
            A ViewerModel loaded from disk or constructed programmatically.
        widget_parent : QWidget or None
            Qt parent for canvas widgets. Defaults to None.
        render_config : RenderManagerConfig or None
            Render pipeline configuration. Defaults to None (uses defaults).

        Returns
        -------
        CellierController
        """
        controller = cls(
            widget_parent=widget_parent,
            render_config=render_config
            if render_config is not None
            else model.render_config,
        )

        # 1. Register all data stores.
        for store in model.data.stores.values():
            controller.add_data_store(store)

        # 2. Register all scenes (render_modes and lighting come from the model).
        for scene in model.scenes.values():
            controller.add_scene_model(scene)

        # 3. Add visuals per scene. Data stores are already registered in step 1.
        # The deserialized model's scene.visuals already contains the models.
        # Clear them first so _add_* helpers can re-append with proper wiring.
        for scene in model.scenes.values():
            visual_models = list(scene.visuals)
            scene.visuals.clear()
            for visual_model in visual_models:
                controller.add_visual(scene.id, visual_model)

        # 4. Restore canvases with camera state from the model.
        for scene in model.scenes.values():
            for canvas_model in scene.canvases.values():
                controller.add_canvas_model(scene.id, canvas_model)

        return controller

    @classmethod
    def from_file(
        cls,
        path: str | pathlib.Path,
        widget_parent: QWidget | None = None,
        render_config: RenderManagerConfig | None = None,
    ) -> CellierController:
        """Deserialize a ViewerModel from disk and construct a controller.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file previously written by ``to_file``.
        widget_parent : QWidget or None
            Qt parent for canvas widgets.
        render_config : RenderManagerConfig or None
            Render pipeline configuration.

        Returns
        -------
        CellierController
        """
        model = ViewerModel.from_file(path)
        return cls.from_model(
            model,
            widget_parent=widget_parent,
            render_config=render_config,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_file(self, path: str | pathlib.Path) -> None:
        """Serialize the current model state to a JSON file."""
        self._model.to_file(path)

    def to_model(self) -> ViewerModel:
        """Return a copy of the current model state."""
        return self._model.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_scene_model(self, scene: Scene) -> Scene:
        """Register a pre-built Scene model with the controller.

        Used by ``from_model`` to restore scenes from a serialized
        ViewerModel, and called internally by ``add_scene``.

        Parameters
        ----------
        scene : Scene
            Pre-built scene model.

        Returns
        -------
        Scene
            The same object passed in.
        """
        self._model.scenes[scene.id] = scene
        self._scene_render_modes[scene.id] = scene.render_modes
        self._render_manager.add_scene(scene.id, lighting=scene.lighting)
        self._scene_to_canvases[scene.id] = []
        self._wire_dims_model(scene)
        self._outgoing_events.emit(
            SceneAddedEvent(source_id=self._id, scene_id=scene.id)
        )
        return scene

    def add_scene(
        self,
        *,
        name: str = "scene",
        dim: Literal["2d", "3d"] = "3d",
        coordinate_system: CoordinateSystem | None = None,
        render_modes: set[Literal["2d", "3d"]] | None = None,
        lighting: Literal["none", "default"] = "none",
    ) -> Scene:
        """Create a Scene from keyword arguments and register it.

        Parameters
        ----------
        name : str
            Human-readable scene name.
        dim : "2d" or "3d"
            Initial display dimensionality.  ``"3d"`` sets
            ``displayed_axes`` to the last three axes of the coordinate
            system; ``"2d"`` sets it to the last two.
        coordinate_system : CoordinateSystem or None
            World coordinate system.  Defaults to a 3-axis ``("z", "y", "x")``
            system when ``None``.
        render_modes : set or None
            Which rendering modes visuals should support.  Defaults to
            ``{"2d", "3d"}``.
        lighting : "none" or "default"
            Pass ``"default"`` to add ambient/directional lights (required for
            ``MeshPhongAppearance``).

        Returns
        -------
        Scene
            The newly created and registered Scene.
        """
        if coordinate_system is None:
            coordinate_system = CoordinateSystem(
                name="world", axis_labels=("z", "y", "x")
            )
        ndim = len(coordinate_system.axis_labels)
        n_displayed = 3 if dim == "3d" else 2
        if ndim < n_displayed:
            raise ValueError(
                f"coordinate_system has {ndim} axes but dim={dim!r} requires "
                f"at least {n_displayed}."
            )
        displayed_axes = tuple(range(ndim - n_displayed, ndim))
        slice_indices = {i: 0 for i in range(ndim) if i not in displayed_axes}
        dims = DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=displayed_axes,
                slice_indices=slice_indices,
            ),
        )
        scene = Scene(
            name=name,
            dims=dims,
            render_modes=render_modes if render_modes is not None else {"2d", "3d"},
            lighting=lighting,
        )
        return self.add_scene_model(scene)

    # ------------------------------------------------------------------
    # Data store management
    # ------------------------------------------------------------------

    def add_data_store(self, data_store: BaseDataStore) -> BaseDataStore:
        """Register a data store and return it.

        Parameters
        ----------
        data_store : BaseDataStore
            The store to register.

        Returns
        -------
        BaseDataStore
            The same object passed in.
        """
        self._model.data.stores[data_store.id] = data_store
        return data_store

    # ------------------------------------------------------------------
    # Visual management — public API
    # ------------------------------------------------------------------

    def add_visual(
        self,
        scene_id: UUID,
        visual_model: VisualType,
        data_store: BaseDataStore | None = None,
    ) -> VisualType:
        """Register a pre-built visual model with a scene.

        This is the canonical construction path used by ``from_model``.
        All typed convenience methods (``add_image``, ``add_mesh``, etc.)
        delegate to this method internally.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : VisualType
            Pre-built visual model. Its ``data_store_id`` must already be
            registered via ``add_data_store``, or ``data_store`` must be
            passed explicitly.
        data_store : BaseDataStore or None
            If provided, register the store first (no-op if already present),
            then use it. If ``None``, the store is looked up by
            ``visual_model.data_store_id``; a ``KeyError`` is raised if not
            found.

        Returns
        -------
        VisualType
            The same visual_model passed in.

        Raises
        ------
        KeyError
            If ``data_store`` is None and ``visual_model.data_store_id`` is not
            registered.
        TypeError
            If the visual type is not recognized.
        """
        if data_store is not None:
            if data_store.id not in self._model.data.stores:
                self._model.data.stores[data_store.id] = data_store

        if isinstance(visual_model, MultiscaleImageVisual):
            return self._add_multiscale_image_visual(scene_id, visual_model)
        elif isinstance(visual_model, ImageVisual):
            return self._add_image_visual(scene_id, visual_model)
        elif isinstance(visual_model, MultichannelImageVisual):
            return self._add_multichannel_image_memory_visual(scene_id, visual_model)
        elif isinstance(visual_model, MultichannelMultiscaleImageVisual):
            return self._add_multichannel_multiscale_image_visual(
                scene_id, visual_model
            )
        elif isinstance(visual_model, MultiscaleLabelVisual):
            return self._add_multiscale_label_visual(scene_id, visual_model)
        elif isinstance(visual_model, LabelMemoryVisual):
            return self._add_label_memory_visual(scene_id, visual_model)
        elif isinstance(visual_model, PointsVisual):
            return self._add_points_visual(scene_id, visual_model)
        elif isinstance(visual_model, LinesVisual):
            return self._add_lines_visual(scene_id, visual_model)
        elif isinstance(visual_model, MeshVisual):
            return self._add_mesh_visual(scene_id, visual_model)
        else:
            raise TypeError(
                f"Unrecognized visual type {type(visual_model)!r}. "
                "Register a handler in add_visual."
            )

    def add_image(
        self,
        data: ImageMemoryStore,
        scene_id: UUID,
        appearance: BaseImageAppearance,
        name: str = "image",
    ) -> ImageVisual:
        """Add an in-memory image visual to a scene.

        Parameters
        ----------
        data : ImageMemoryStore
            The backing data store.
        scene_id : UUID
            ID of an existing scene.
        appearance : BaseImageAppearance
            Appearance parameters.
        name : str
            Human-readable label. Default ``"image"``.

        Returns
        -------
        ImageVisual
        """
        visual_model = ImageVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_labels(
        self,
        data: LabelMemoryStore,
        scene_id: UUID,
        appearance: BaseLabelsAppearance | None = None,
        name: str = "labels",
        transform: AffineTransform | None = None,
    ) -> LabelMemoryVisual:
        """Add an in-memory label visual to a scene.

        Parameters
        ----------
        data : LabelMemoryStore
            Backing int32 label store.
        scene_id : UUID
            ID of an existing scene.
        appearance : BaseLabelsAppearance or None
            Appearance parameters. Defaults to InMemoryLabelsAppearance().
        name : str
            Human-readable label. Default ``"labels"``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when None.

        Returns
        -------
        LabelMemoryVisual
        """
        if appearance is None:
            from cellier.visuals._label_memory import InMemoryLabelsAppearance

            appearance = InMemoryLabelsAppearance()

        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=data.ndim)
        )
        visual_model = LabelMemoryVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_mesh(
        self,
        data: MeshMemoryStore,
        scene_id: UUID,
        appearance: MeshAppearance,
        name: str = "mesh",
        transform: AffineTransform | None = None,
    ) -> MeshVisual:
        """Add a mesh visual to a scene.

        Parameters
        ----------
        data : MeshMemoryStore
            In-memory mesh.  Normals are auto-computed if not supplied;
            indices are coerced to int32.
        scene_id : UUID
            ID of an existing scene.
        appearance : MeshFlatAppearance | MeshPhongAppearance
            Appearance.  Use MeshPhongAppearance with ``lighting="default"``
            on the scene for shaded rendering.
        name : str
            Human-readable label.  Default ``"mesh"``.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        MeshVisual
        """
        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=data.positions.shape[1])
        )
        visual_model = MeshVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_points(
        self,
        data: PointsMemoryStore,
        scene_id: UUID,
        appearance: PointsMarkerAppearance | None = None,
        name: str = "points",
        transform: AffineTransform | None = None,
    ) -> PointsVisual:
        """Add a points visual backed by a PointsMemoryStore.

        Parameters
        ----------
        data : PointsMemoryStore
            The backing data store.
        scene_id : UUID
            ID of the target scene.
        appearance : PointsMarkerAppearance or None
            Appearance model.  Defaults to PointsMarkerAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        PointsVisual
        """
        if appearance is None:
            appearance = PointsMarkerAppearance()

        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=data.ndim)
        )
        visual_model = PointsVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_lines(
        self,
        data: LinesMemoryStore,
        scene_id: UUID,
        appearance: LinesMemoryAppearance | None = None,
        name: str = "lines",
        transform: AffineTransform | None = None,
    ) -> LinesVisual:
        """Add a lines visual backed by a LinesMemoryStore.

        Parameters
        ----------
        data : LinesMemoryStore
            The backing data store.
        scene_id : UUID
            ID of the target scene.
        appearance : LinesMemoryAppearance or None
            Appearance model.  Defaults to LinesMemoryAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        LinesVisual
        """
        if appearance is None:
            appearance = LinesMemoryAppearance()

        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=data.ndim)
        )
        visual_model = LinesVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_image_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: MultiscaleImageAppearance,
        name: str = "image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: AffineTransform | None = None,
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual to a scene.

        Parameters
        ----------
        data : BaseDataStore
            The backing data store.
        scene_id : UUID
            ID of an existing scene.
        appearance : MultiscaleImageAppearance
            Visual appearance parameters.
        name : str
            Human-readable label. Default ``"image"``.
        render_config : MultiscaleImageRenderConfig or None
            Render-layer configuration. Defaults to
            ``MultiscaleImageRenderConfig()`` with all default values if None.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when None.

        Returns
        -------
        MultiscaleImageVisual
        """
        if render_config is None:
            render_config = MultiscaleImageRenderConfig()

        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=len(data.level_shapes[0]))
        )
        visual_model = MultiscaleImageVisual(
            name=name,
            data_store_id=str(data.id),
            level_transforms=data.level_transforms,
            appearance=appearance,
            render_config=render_config,
            transform=resolved_transform,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_labels_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: MultiscaleLabelsAppearance,
        name: str = "labels",
        render_config: MultiscaleLabelRenderConfig | None = None,
        transform: AffineTransform | None = None,
    ) -> MultiscaleLabelVisual:
        """Add a multiscale label visual to a scene.

        Parameters
        ----------
        data : BaseDataStore
            The backing label data store (e.g. ``OMEZarrLabelDataStore``).
        scene_id : UUID
            ID of an existing scene.
        appearance : MultiscaleLabelsAppearance
            Visual appearance parameters.
        name : str
            Human-readable label. Default ``"labels"``.
        render_config : MultiscaleLabelRenderConfig or None
            Render-layer configuration. Defaults to
            ``MultiscaleLabelRenderConfig()`` with all default values if None.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when None.

        Returns
        -------
        MultiscaleLabelVisual
        """
        if render_config is None:
            render_config = MultiscaleLabelRenderConfig()

        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=len(data.level_shapes[0]))
        )
        visual_model = MultiscaleLabelVisual(
            name=name,
            data_store_id=str(data.id),
            level_transforms=data.level_transforms,
            appearance=appearance,
            render_config=render_config,
            transform=resolved_transform,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_multichannel_image(
        self,
        data: ImageMemoryStore,
        scene_id: UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance],
        name: str = "multichannel_image",
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
    ) -> MultichannelImageVisual:
        """Add an in-memory multichannel image visual to a scene.

        Parameters
        ----------
        data : ImageMemoryStore
            Backing data store.
        scene_id : UUID
            Target scene.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance]
            Per-channel appearance keyed by channel index.
        name : str
            Display name for the visual.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.

        Returns
        -------
        MultichannelImageVisual
        """
        if len(channels) > max_channels_2d:
            raise ValueError(
                f"len(channels)={len(channels)} exceeds "
                f"max_channels_2d={max_channels_2d}."
            )
        visual_model = MultichannelImageVisual(
            name=name,
            data_store_id=str(data.id),
            channel_axis=channel_axis,
            channels=channels,
            max_channels_2d=max_channels_2d,
            max_channels_3d=max_channels_3d,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_multichannel_image_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance],
        name: str = "multichannel_image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: AffineTransform | None = None,
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
    ) -> MultichannelMultiscaleImageVisual:
        """Add a multiscale multichannel image visual to a scene.

        Parameters
        ----------
        data : BaseDataStore
            Backing multiscale data store.
        scene_id : UUID
            Target scene.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance]
            Per-channel appearance keyed by channel index.
        name : str
            Display name for the visual.
        render_config : MultiscaleImageRenderConfig or None
            LOD and rendering configuration; uses defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform; uses identity when ``None``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.

        Returns
        -------
        MultichannelMultiscaleImageVisual
        """
        if render_config is None:
            render_config = MultiscaleImageRenderConfig()
        if len(channels) > max_channels_2d:
            raise ValueError(
                f"len(channels)={len(channels)} exceeds "
                f"max_channels_2d={max_channels_2d}."
            )
        # Extract the spatial-only submatrix so the GFX layer can expand_dims
        # it back to full data ndim (it prepends identity axes for non-spatial dims).
        if transform is not None:
            spatial_axes = tuple(i for i in range(transform.ndim) if i != channel_axis)
            spatial_transform = transform.select_axes(spatial_axes)
        else:
            spatial_transform = None

        extra = {} if spatial_transform is None else {"transform": spatial_transform}
        visual_model = MultichannelMultiscaleImageVisual(
            name=name,
            data_store_id=str(data.id),
            channel_axis=channel_axis,
            channels=channels,
            level_transforms=data.level_transforms,
            render_config=render_config,
            max_channels_2d=max_channels_2d,
            max_channels_3d=max_channels_3d,
            **extra,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def _get_visual_model(self, visual_id: UUID) -> VisualType:
        """Return the visual model for *visual_id*.

        Raises
        ------
        KeyError
            If *visual_id* is not registered.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        for visual in scene.visuals:
            if visual.id == visual_id:
                return visual
        raise KeyError(f"Visual {visual_id} not found.")

    def add_channel(
        self,
        visual_id: UUID,
        channel_index: int,
        appearance: ChannelAppearance,
    ) -> None:
        """Add a channel to a multichannel image visual.

        Parameters
        ----------
        visual_id : UUID
            ID of a MultichannelImageVisual or MultichannelMultiscaleImageVisual.
        channel_index : int
            Index along the visual's channel_axis. Must not already be present.
        appearance : ChannelAppearance
            Colormap, clim, and opacity settings for the new channel.

        Raises
        ------
        ValueError
            If channel_index is already present.
        RuntimeError
            If the pool is full.
        """
        visual = self._get_visual_model(visual_id)
        if channel_index in visual.channels:
            raise ValueError(
                f"channel_index={channel_index} already in visual.channels."
            )
        if len(visual.channels) >= visual.max_channels_2d:
            raise RuntimeError(
                f"Pool is full ({visual.max_channels_2d} channels). "
                "Increase max_channels_2d or remove a channel first."
            )
        new_channels = dict(visual.channels)
        new_channels[channel_index] = appearance
        visual.channels = new_channels
        self.reslice_visual(visual_id)

    def remove_channel(self, visual_id: UUID, channel_index: int) -> None:
        """Remove a channel from a multichannel image visual.

        Parameters
        ----------
        visual_id : UUID
            ID of a MultichannelImageVisual or MultichannelMultiscaleImageVisual.
        channel_index : int
            Index of the channel to remove.

        Raises
        ------
        KeyError
            If channel_index is not in visual.channels.
        """
        visual = self._get_visual_model(visual_id)
        if channel_index not in visual.channels:
            raise KeyError(f"channel_index={channel_index} not in visual.channels.")
        new_channels = {k: v for k, v in visual.channels.items() if k != channel_index}
        visual.channels = new_channels

    # ------------------------------------------------------------------
    # Visual management — private dispatch methods
    # ------------------------------------------------------------------

    def _register_visual(
        self,
        scene_id: UUID,
        visual_model: BaseVisual,
        gfx_visual: Any,
        data_store: Any,
        displayed_axes: tuple[int, ...],
    ) -> None:
        """Register a pre-built (visual_model, gfx_visual) pair in one scene.

        Single point of truth for all post-construction wiring:
        - Appends the visual to the scene's visual list.
        - Registers the GFX visual with the RenderManager.
        - Records the visual→scene mapping.
        - Wires psygnal bridges (appearance, channels, aabb, transform,
          pick_write).
        - Subscribes the GFX visual to all EventBus events it handles.
        - Emits VisualAddedEvent.
        """
        scene = self._model.scenes[scene_id]
        scene.visuals.append(visual_model)
        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
        self._visual_to_scene[visual_model.id] = scene_id

        # psygnal bridges
        if hasattr(visual_model, "appearance"):
            self._wire_appearance(visual_model)
        if hasattr(visual_model, "channels"):
            self._wire_channels(visual_model)
        self._wire_aabb(visual_model)
        self._wire_transform(visual_model, scene_id)
        self._wire_pick_write(visual_model)

        # EventBus subscriptions — only subscribe when the GFX visual implements
        # the handler so new visual types get wired automatically.
        for event_type, handler_name in (
            (AppearanceChangedEvent, "on_appearance_changed"),
            (ChannelAppearanceChangedEvent, "on_channel_appearance_changed"),
            (AABBChangedEvent, "on_aabb_changed"),
            (VisualVisibilityChangedEvent, "on_visibility_changed"),
            (TransformChangedEvent, "on_transform_changed"),
            (PickWriteChangedEvent, "on_pick_write_changed"),
        ):
            handler = getattr(gfx_visual, handler_name, None)
            if handler is not None:
                self._outgoing_events.subscribe(
                    event_type,
                    handler,
                    entity_id=visual_model.id,
                    owner_id=visual_model.id,
                )

        self._outgoing_events.emit(
            VisualAddedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_model.id,
            )
        )

    def _add_multiscale_image_visual(
        self,
        scene_id: UUID,
        visual_model: MultiscaleImageVisual,
    ) -> MultiscaleImageVisual:
        """Wire and register a pre-built MultiscaleImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=list(data_store.level_shapes),
            render_modes=render_modes,
            displayed_axes=displayed_axes,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_multiscale_label_visual(
        self,
        scene_id: UUID,
        visual_model: MultiscaleLabelVisual,
    ) -> MultiscaleLabelVisual:
        """Wire and register a pre-built MultiscaleLabelVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMultiscaleLabelVisual.from_cellier_model(
            model=visual_model,
            level_shapes=list(data_store.level_shapes),
            render_modes=render_modes,
            displayed_axes=displayed_axes,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_image_visual(
        self,
        scene_id: UUID,
        visual_model: ImageVisual,
    ) -> ImageVisual:
        """Wire and register a pre-built ImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXImageMemoryVisual(
            visual_model=visual_model,
            data_store=data_store,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_multichannel_image_memory_visual(
        self,
        scene_id: UUID,
        visual_model: MultichannelImageVisual,
    ) -> MultichannelImageVisual:
        """Wire and register a pre-built MultichannelImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMultichannelImageMemoryVisual(
            visual_model=visual_model,
            data_store=data_store,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_multichannel_multiscale_image_visual(
        self,
        scene_id: UUID,
        visual_model: MultichannelMultiscaleImageVisual,
    ) -> MultichannelMultiscaleImageVisual:
        """Wire and register a pre-built MultichannelMultiscaleImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMultichannelMultiscaleImageVisual(
            visual_model=visual_model,
            level_shapes=list(data_store.level_shapes),
            render_modes=render_modes,
            displayed_axes=displayed_axes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_points_visual(
        self,
        scene_id: UUID,
        visual_model: PointsVisual,
    ) -> PointsVisual:
        """Wire and register a pre-built PointsVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXPointsMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_lines_visual(
        self,
        scene_id: UUID,
        visual_model: LinesVisual,
    ) -> LinesVisual:
        """Wire and register a pre-built LinesVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXLinesMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_mesh_visual(
        self,
        scene_id: UUID,
        visual_model: MeshVisual,
    ) -> MeshVisual:
        """Wire and register a pre-built MeshVisual."""
        import warnings

        if isinstance(visual_model.appearance, MeshPhongAppearance):
            if not self._render_manager.scene_has_lighting(scene_id):
                warnings.warn(
                    "MeshPhongAppearance requires lights in the scene. "
                    "Pass lighting='default' to the Scene model, otherwise "
                    "the mesh will render black.",
                    stacklevel=3,
                )
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMeshMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_label_memory_visual(
        self,
        scene_id: UUID,
        visual_model: LabelMemoryVisual,
    ) -> LabelMemoryVisual:
        """Wire and register a pre-built LabelMemoryVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXLabelMemoryVisual(
            visual_model=visual_model,
            data_store=data_store,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    # ------------------------------------------------------------------
    # Overlay construction
    # ------------------------------------------------------------------

    def _build_gfx_overlay(
        self,
        canvas_id: UUID,
        overlay_model: CanvasOverlay,
    ):
        """Construct the render-layer overlay for *overlay_model*.

        Mirrors the ``_add_*_visual`` pattern: the controller is responsible
        for constructing GFX objects; the render manager is a passive registrar.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas that will own the overlay.
        overlay_model : CanvasOverlay
            Model-layer overlay description.

        Raises
        ------
        TypeError
            If *overlay_model* has an unrecognised type.
        """
        canvas_view = self._render_manager._canvases[canvas_id]
        if isinstance(overlay_model, CenteredAxes2D):
            return GFXCenteredAxes2D(
                model=overlay_model,
                camera=canvas_view.camera,
            )
        raise TypeError(
            f"Unrecognised overlay type {type(overlay_model)!r}. "
            "Register a handler in _build_gfx_overlay."
        )

    # ------------------------------------------------------------------
    # Canvas management
    # ------------------------------------------------------------------

    def add_canvas(
        self,
        scene_id: UUID,
        render_modes: set[str] | None = None,
        initial_dim: str | None = None,
        fov: float = 70.0,
        depth_range_3d: tuple[float, float] = (1.0, 8000.0),
        depth_range_2d: tuple[float, float] = (-500.0, 500.0),
        canvas_size: tuple[int, int] | None = None,
    ) -> QWidget:
        """Create a canvas attached to a scene and return its embeddable widget.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        render_modes : set[str] or None
            Which camera modes to prepare on the canvas.  Each entry must be
            ``"2d"`` or ``"3d"``.  When ``None``, defaults to the scene's own
            ``render_modes``.  Pass ``{"2d", "3d"}`` for a canvas that can
            switch between views.
        initial_dim : str or None
            Which mode is active when the canvas first appears.  Must be a
            member of ``render_modes``.  When ``None``, inferred from the
            scene's current ``displayed_axes`` length (3 axes -> ``"3d"``,
            otherwise ``"2d"``).
        fov : float
            Vertical field of view in degrees for the 3D perspective camera.
            Ignored when ``"3d"`` is not in ``render_modes``.  Default ``70.0``.
        depth_range_3d : tuple[float, float]
            ``(near, far)`` clip distances for the 3D perspective camera.
            Default ``(1.0, 8000.0)``.
        depth_range_2d : tuple[float, float]
            ``(near, far)`` clip distances for the 2D orthographic camera.
            Default ``(-500.0, 500.0)``.
        canvas_size : tuple[int, int] or None
            Initial CSS pixel size for the anywidget canvas.  Ignored for the
            Qt gui (which is sized by its parent layout).  Defaults to
            ``(600, 600)`` when ``None`` for the anywidget gui.

        Returns
        -------
        QWidget
            The render widget.  Embed with ``layout.addWidget(widget)``.

        Raises
        ------
        ValueError
            If ``initial_dim`` is supplied but is not a member of
            ``render_modes``.
        """
        scene = self._model.scenes[scene_id]

        if render_modes is None:
            render_modes = scene.render_modes

        if initial_dim is None:
            initial_dim = (
                "3d" if len(scene.dims.selection.displayed_axes) == 3 else "2d"
            )

        if initial_dim not in render_modes:
            raise ValueError(
                f"initial_dim={initial_dim!r} is not in render_modes={render_modes!r}."
            )

        cameras: dict[str, CameraType] = {}
        if "3d" in render_modes:
            cameras["3d"] = PerspectiveCamera(
                fov=fov,
                near_clipping_plane=depth_range_3d[0],
                far_clipping_plane=depth_range_3d[1],
                controller=OrbitCameraController(enabled=True),
            )
        if "2d" in render_modes:
            cameras["2d"] = OrthographicCamera(
                near_clipping_plane=depth_range_2d[0],
                far_clipping_plane=depth_range_2d[1],
                controller=PanZoomCameraController(enabled=True),
            )

        canvas_model = Canvas(cameras=cameras)
        return self.add_canvas_model(
            scene_id, canvas_model, initial_dim=initial_dim, canvas_size=canvas_size
        )

    def add_canvas_model(
        self,
        scene_id: UUID,
        canvas_model: Canvas,
        initial_dim: str | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> QWidget:
        """Register a pre-built Canvas model with a scene.

        Used by ``from_model`` to restore canvases from a serialized
        ViewerModel, and called internally by ``add_canvas``.  Camera state
        (position, rotation, fov, depth range) is read from the camera models
        stored in ``canvas_model.cameras``.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        canvas_model : Canvas
            Pre-built canvas model.  Must have at least one entry in
            ``canvas_model.cameras``.
        initial_dim : str or None
            Which dim to activate first.  When ``None``, the first key in
            ``canvas_model.cameras`` is used (insertion order is preserved by
            Python dicts, so this is deterministic for serialized models).
        canvas_size : tuple[int, int] or None
            Initial CSS pixel size for the anywidget canvas.  Ignored for the
            Qt gui.  Defaults to ``(600, 600)`` for the anywidget gui.

        Returns
        -------
        QWidget
            The render widget.

        Raises
        ------
        ValueError
            If ``canvas_model.cameras`` is empty, or if ``initial_dim`` is not
            a key in ``canvas_model.cameras``.
        """
        if not canvas_model.cameras:
            raise ValueError("canvas_model.cameras must not be empty.")

        if initial_dim is None:
            initial_dim = next(iter(canvas_model.cameras))

        if initial_dim not in canvas_model.cameras:
            raise ValueError(
                f"initial_dim={initial_dim!r} is not a key in "
                f"canvas_model.cameras ({list(canvas_model.cameras)!r})."
            )

        active_camera = canvas_model.cameras[initial_dim]
        depth_range = (
            active_camera.near_clipping_plane,
            active_camera.far_clipping_plane,
        )

        # Store the model using its existing id.
        self._model.scenes[scene_id].canvases[canvas_model.id] = canvas_model

        # Use canvas_model.id as the single canonical ID for both layers.
        if isinstance(active_camera, PerspectiveCamera):
            canvas_view = self._render_manager.add_canvas(
                canvas_model.id,
                scene_id,
                parent=self._widget_parent,
                dim=initial_dim,
                fov=active_camera.fov,
                depth_range=depth_range,
                gui=self._gui,
                size=canvas_size,
            )
        else:
            # OrthographicCamera: fov is not meaningful; omit it so CanvasView
            # uses its own internal default for the 3D camera it keeps in reserve.
            canvas_view = self._render_manager.add_canvas(
                canvas_model.id,
                scene_id,
                parent=self._widget_parent,
                dim=initial_dim,
                depth_range=depth_range,
                gui=self._gui,
                size=canvas_size,
            )

        canvas_view.set_event_bus(self._outgoing_events)

        # Fix the reserve (non-active) camera's depth range.  CanvasView builds
        # both the 2D and 3D cameras up front but applies only the active
        # camera's range to both; the reserve camera (e.g. the 3D perspective
        # camera when starting in 2D) is then left with the active camera's
        # range, which may have an invalid near plane and render nothing on the
        # first toggle.  Only the reserve camera is touched here: re-setting the
        # active camera's range would change its captured state and spuriously
        # emit a CameraChangedEvent (scheduling an unwanted settle reslice).
        for dim_key, camera_model in canvas_model.cameras.items():
            if dim_key == initial_dim:
                continue
            canvas_view.set_depth_range_for_dim(
                dim_key,
                (
                    camera_model.near_clipping_plane,
                    camera_model.far_clipping_plane,
                ),
            )

        # Wire any overlays already stored on this canvas model to the render
        # layer.  For canvases created via add_canvas() this loop is a no-op
        # (overlays=[]).  For canvases restored from a serialized ViewerModel
        # the overlays list is already populated, so this call is sufficient —
        # from_model needs no additional overlay-restoration step.
        # _build_gfx_overlay is called directly rather than add_canvas_overlay_model
        # to avoid re-appending models that are already in canvas_model.overlays.
        for overlay_model in canvas_model.overlays:
            gfx_overlay = self._build_gfx_overlay(canvas_model.id, overlay_model)
            self._render_manager.add_canvas_overlay(canvas_model.id, gfx_overlay)

        self._canvas_to_scene[canvas_model.id] = scene_id
        self._scene_to_canvases[scene_id].append(canvas_model.id)

        return canvas_view.widget

    # ------------------------------------------------------------------
    # Scene and visual lookup
    # ------------------------------------------------------------------

    def get_scene(self, scene_id: UUID) -> Scene:
        """Return the live Scene model for scene_id."""
        return self._model.scenes[scene_id]

    def get_data_store(self, store_id: UUID) -> BaseDataStore:
        """Return the registered data store for store_id.

        Parameters
        ----------
        store_id : UUID
            ID of a previously registered data store.

        Returns
        -------
        BaseDataStore

        Raises
        ------
        KeyError
            If no store with store_id has been registered.
        """
        return self._model.data.stores[store_id]

    def fit_camera(self, scene_id: UUID, canvas_id: UUID | None = None) -> None:
        """Fit the camera to the current scene bounding box.

        Safe to call immediately after ``add_image`` / ``add_image_multiscale``
        and transform assignment — the node matrix is set at construction time
        so no chunk data needs to be loaded first.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene whose camera should be fitted.
        canvas_id : UUID or None
            If provided, fit only that canvas.  When ``None`` (default),
            all canvases attached to *scene_id* are fitted.
        """
        gfx_scene = self._render_manager.get_scene(scene_id)
        targets = (
            [canvas_id]
            if canvas_id is not None
            else list(self._scene_to_canvases.get(scene_id, []))
        )
        for cid in targets:
            canvas = self._render_manager._canvases.get(cid)
            if canvas is not None:
                canvas.show_object(gfx_scene)
                state = canvas.capture_camera_state()
                self._update_camera_model(scene_id, cid, state)
        # A fit already walked the scene graph, so this is the cheapest
        # moment to re-derive the ambient occlusion radius from the scene
        # bounding box.
        self._render_manager.update_ssao_radius(scene_id)

    def add_canvas_overlay_model(
        self,
        canvas_id: UUID,
        overlay: CanvasOverlay,
    ) -> CanvasOverlay:
        """Attach a screen-space overlay to a specific canvas.

        The overlay is rendered as a post-pass on top of the main scene each
        frame.  It does not participate in reslicing, has no world-space
        transform, and is not added to ``scene.visuals``.

        The overlay model is stored in the ``Canvas.overlays`` list of
        *canvas_id*, making it part of the serializable model.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas that should display the overlay.  Use
            :meth:`get_canvas_ids` to look up canvas IDs for a scene.
        overlay : CanvasOverlay
            Model-layer overlay description.  Typically a
            :class:`~cellier.visuals._canvas_overlay.CenteredAxes2D`.

        Returns
        -------
        CanvasOverlay
            The same overlay object passed in (for ID access or chaining).

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        scene_id = self._canvas_to_scene[canvas_id]
        canvas_model = self._model.scenes[scene_id].canvases[canvas_id]
        canvas_model.overlays.append(overlay)

        gfx_overlay = self._build_gfx_overlay(canvas_id, overlay)
        self._render_manager.add_canvas_overlay(canvas_id, gfx_overlay)

        return overlay

    def set_overlay_visible(self, overlay_id: UUID, visible: bool) -> None:
        """Toggle the visibility of a canvas overlay.

        Searches all canvases across all scenes for an overlay with
        ``overlay_id``.  Updates both the model field and the render layer.

        Parameters
        ----------
        overlay_id : UUID
            ID of the :class:`~cellier.visuals._canvas_overlay.CanvasOverlay`
            to toggle.
        visible : bool
            ``True`` to show the overlay, ``False`` to hide it.

        Raises
        ------
        KeyError
            If no overlay with ``overlay_id`` is found.
        """
        for scene in self._model.scenes.values():
            for canvas_model in scene.canvases.values():
                for overlay_model in canvas_model.overlays:
                    if overlay_model.id == overlay_id:
                        overlay_model.visible = visible
                        canvas_view = self._render_manager._canvases.get(
                            canvas_model.id
                        )
                        if canvas_view is not None:
                            for gfx_overlay in canvas_view._overlays:
                                model_ref = getattr(gfx_overlay, "_model", None)
                                if model_ref is overlay_model:
                                    gfx_overlay.set_visible(visible)
                        return
        raise KeyError(
            f"No canvas overlay with id={overlay_id!r} found.  "
            "Ensure add_canvas_overlay was called before set_overlay_visible."
        )

    def get_scene_by_name(self, name: str) -> Scene:
        """Return the live Scene model for the given name.

        Raises KeyError if no scene with that name exists.
        """
        for scene in self._model.scenes.values():
            if scene.name == name:
                return scene
        raise KeyError(f"No scene named {name!r}")

    def get_canvas_ids(self, scene_id: UUID) -> list[UUID]:
        """Return the IDs of all canvases registered for *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.

        Returns
        -------
        list[UUID]
            Canvas IDs in registration order.  Empty if no canvases have
            been added yet.
        """
        return list(self._scene_to_canvases.get(scene_id, []))

    def get_canvas_view(self, canvas_id: UUID) -> CanvasView:
        """Return the render-layer ``CanvasView`` for *canvas_id*.

        Provides access to the rendering backend (camera, widget, overlays)
        for a canvas registered via :meth:`add_canvas` or
        :meth:`add_canvas_model`.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas.

        Returns
        -------
        CanvasView

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        return self._render_manager._canvases[canvas_id]

    def get_camera_state(self, canvas_id: UUID) -> CameraState:
        """Return a snapshot of the current camera state for *canvas_id*.

        Useful for seeding downstream widgets (e.g. orientation overlays)
        with the post-fit camera state without constructing a synthetic
        ``CameraChangedEvent``.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas.

        Returns
        -------
        CameraState

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        return self.get_canvas_view(canvas_id).capture_camera_state()

    def screenshot(
        self,
        canvas_id: UUID,
        size: tuple[int, int] | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Capture a screenshot of the canvas as an RGBA uint8 array.

        Temporarily resizes the canvas to *size* (scaled by *scale*), renders
        one frame, grabs the framebuffer, then restores the original size and
        camera state.  Follows the same pattern as napari's ``resize_canvas``
        context manager.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas.
        size : tuple[int, int] or None
            Target ``(width, height)`` in logical pixels before applying
            *scale*.  When ``None`` the current canvas size is used.
        scale : float
            Multiplier applied to *size* (or the current size when *size* is
            ``None``).  ``scale=2`` doubles the resolution for high-DPI output.

        Returns
        -------
        np.ndarray
            RGBA uint8 array of shape ``(height, width, 4)``.
        """
        canvas_view = self._render_manager._canvases[canvas_id]
        canvas = canvas_view.widget

        prev_w, prev_h = canvas.get_logical_size()
        target_w = int((size[0] if size is not None else prev_w) * scale)
        target_h = int((size[1] if size is not None else prev_h) * scale)

        try:
            canvas.set_logical_size(target_w, target_h)
            if self._gui == "qt":
                from PySide6.QtWidgets import QApplication

                QApplication.processEvents()
            canvas_view._canvas.request_draw(canvas_view._draw_frame)
            if self._gui == "qt":
                from PySide6.QtWidgets import QApplication

                QApplication.processEvents()
            array = np.asarray(canvas_view._renderer.snapshot())
        finally:
            canvas.set_logical_size(prev_w, prev_h)

        return array

    def get_visual_model(self, visual_id: UUID) -> MultiscaleImageVisual:
        """Return the live visual model for visual_id.

        Searches all scenes.  Raises KeyError if not found.
        """
        for scene in self._model.scenes.values():
            for visual in scene.visuals:
                if visual.id == visual_id:
                    return visual
        raise KeyError(f"No visual with id {visual_id}")

    # ------------------------------------------------------------------
    # Reslicing
    # ------------------------------------------------------------------

    def _build_visual_configs_for_scene(
        self, scene_id: UUID
    ) -> dict[UUID, VisualRenderConfig]:
        """Build a VisualRenderConfig dict from all visuals in a scene."""
        scene = self._model.scenes[scene_id]
        configs: dict[UUID, VisualRenderConfig] = {}
        for visual in scene.visuals:
            if isinstance(visual, (MultiscaleImageVisual, MultiscaleLabelVisual)):
                configs[visual.id] = VisualRenderConfig(
                    lod_bias=visual.appearance.lod_bias,
                    force_level=visual.appearance.force_level,
                    frustum_cull=visual.appearance.frustum_cull,
                )
            else:
                configs[visual.id] = VisualRenderConfig()
        return configs

    def _dims_state_for_scene(self, scene_id: UUID) -> DimsState:
        """Derive a DimsState from the scene's DimsManager."""
        return self._model.scenes[scene_id].dims.to_state()

    # ------------------------------------------------------------------
    # psygnal bridges
    # ------------------------------------------------------------------

    def _wire_dims_model(self, scene: Scene) -> None:
        """Subscribe to all field changes on a scene's DimsManager."""
        self._dims_cache[scene.id] = scene.dims.selection.displayed_axes
        scene.dims.events.connect(self._make_dims_handler(scene.id))

    def _make_dims_handler(self, scene_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a scene's DimsManager."""

        def _on_dims_psygnal(info: EmissionInfo) -> None:
            new_state = self._model.scenes[scene_id].dims.to_state()
            prev_axes = self._dims_cache[scene_id]
            displayed_axes_changed = prev_axes != new_state.selection.displayed_axes
            self._dims_cache[scene_id] = new_state.selection.displayed_axes
            if displayed_axes_changed:
                self._rebuild_visuals_geometry(
                    scene_id, new_state.selection.displayed_axes
                )
                self._switch_canvas_cameras(
                    scene_id, new_state.selection.displayed_axes
                )
            resolved_source_id = _source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_dims_psygnal  scene=%s"
                "  resolved_source=%s  override_active=%s",
                scene_id,
                resolved_source_id,
                _source_id_override.get() is not None,
            )
            self._outgoing_events.emit(
                DimsChangedEvent(
                    source_id=resolved_source_id,
                    scene_id=scene_id,
                    dims_state=new_state,
                    displayed_axes_changed=displayed_axes_changed,
                )
            )

        return _on_dims_psygnal

    def _level_shapes_for(self, visual_model) -> list[tuple[int, ...]]:
        """Return the level shapes for *visual_model* from its data store."""
        data_store_id = getattr(visual_model, "data_store_id", None)
        if data_store_id:
            store = self._model.data.stores.get(UUID(data_store_id))
            if store is not None:
                # Single-node memory stores (points/lines/mesh) have no levels;
                # only image/label stores expose ``level_shapes``.
                return list(getattr(store, "level_shapes", []))
        return []

    def _rebuild_visuals_geometry(
        self, scene_id: UUID, displayed_axes: tuple[int, ...]
    ) -> None:
        """Swap each visual's active node after a displayed_axes change.

        Uses the GFXVisual protocol: calls ``build_node`` on the first visit
        to a mode (passing the model so the renderer reads appearance at call
        time without caching it), or ``rebuild_node_geometry`` when the node
        already exists.  Single-node visuals (mesh, lines) handle both as
        no-ops and return the same node, so ``swap_node`` is a no-op too.
        """
        mode = "3d" if len(displayed_axes) == 3 else "2d"
        scene = self._model.scenes[scene_id]
        scene_manager = self._render_manager._scenes[scene_id]

        for visual_model in scene.visuals:
            gfx_visual = scene_manager.get_visual(visual_model.id)
            level_shapes = self._level_shapes_for(visual_model)
            level_transforms = list(getattr(visual_model, "level_transforms", []))
            if not gfx_visual.has_node(mode):
                new_node = gfx_visual.build_node(
                    mode, visual_model, displayed_axes, level_shapes, level_transforms
                )
            else:
                new_node = gfx_visual.rebuild_node_geometry(
                    mode, displayed_axes, level_shapes, level_transforms
                )
            scene_manager.swap_node(visual_model.id, new_node)

    def _switch_canvas_cameras(
        self, scene_id: UUID, displayed_axes: tuple[int, ...]
    ) -> None:
        """Switch cameras on all canvases attached to *scene_id*.

        Calls ``show_object`` on canvases that are being activated for the
        first time in the new dimensionality.
        """
        new_dim = "3d" if len(displayed_axes) == 3 else "2d"
        gfx_scene = self._render_manager.get_scene(scene_id)
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            canvas_view = self._render_manager._canvases[canvas_id]
            first_visit = canvas_view.switch_dim(new_dim)
            if first_visit:
                canvas_view.show_object(gfx_scene)

    def _wire_transform(
        self,
        visual: BaseVisual,
        scene_id: UUID,
    ) -> None:
        """Subscribe to transform field changes on a visual model."""
        handler = self._make_transform_handler(visual.id, scene_id)
        visual.events.transform.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.events.transform, handler)
        )

    def _make_transform_handler(self, visual_id: UUID, scene_id: UUID) -> Callable:
        """Return a handler that emits TransformChangedEvent and triggers reslice.

        The reslice is skipped when ``_suppress_reslice`` is True, which is
        managed by the :meth:`suppress_reslice` context manager.
        """

        def _on_transform(new_transform: AffineTransform) -> None:
            self._outgoing_events.emit(
                TransformChangedEvent(
                    source_id=self._id,
                    scene_id=scene_id,
                    visual_id=visual_id,
                    transform=new_transform,
                )
            )
            if not self._suppress_reslice:
                self.reslice_scene(scene_id)

        return _on_transform

    def _wire_appearance(
        self,
        visual: BaseVisual,
    ) -> None:
        """Subscribe to all field changes on a visual's appearance model."""
        handler = self._make_appearance_handler(visual.id)
        visual.appearance.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.appearance.events, handler)
        )

    def _wire_channels(
        self,
        visual: MultichannelImageVisual | MultichannelMultiscaleImageVisual,
    ) -> None:
        """Subscribe to field changes on every ``ChannelAppearance`` in a visual.

        One psygnal handler is registered per channel. Handlers are stored in
        ``self._visual_psygnal_handlers`` so they can be disconnected on teardown.

        This wires per-field changes only. Whole-dict replacement
        (``visual.channels = new_dict``) is handled by a direct psygnal connect
        inside the GFX visual, because it is a structural pool-management
        operation rather than an appearance field push.
        """
        for channel_index, appearance in visual.channels.items():
            handler = self._make_channel_appearance_handler(visual.id, channel_index)
            appearance.events.connect(handler)
            self._visual_psygnal_handlers.setdefault(visual.id, []).append(
                (appearance.events, handler)
            )

    def _make_channel_appearance_handler(
        self, visual_id: UUID, channel_index: int
    ) -> Callable:
        """Return a psygnal catch-all handler for one ``ChannelAppearance``."""

        def _on_channel_appearance_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _source_id_override.get() or self._id
            self._outgoing_events.emit(
                ChannelAppearanceChangedEvent(
                    source_id=resolved_source_id,
                    visual_id=visual_id,
                    channel_index=channel_index,
                    field_name=field_name,
                    new_value=new_value,
                )
            )

        return _on_channel_appearance_psygnal

    def _wire_aabb(self, visual: BaseVisual) -> None:
        """Subscribe to all field changes on a visual's aabb model."""
        handler = self._make_aabb_handler(visual.id)
        visual.aabb.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.aabb.events, handler)
        )

    def _make_aabb_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a visual's AABBParams."""

        def _on_aabb_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _aabb_source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_aabb_psygnal  visual=%s  field=%s"
                "  resolved_source=%s  override_active=%s",
                visual_id,
                field_name,
                resolved_source_id,
                _aabb_source_id_override.get() is not None,
            )
            self._outgoing_events.emit(
                AABBChangedEvent(
                    source_id=resolved_source_id,
                    visual_id=visual_id,
                    field_name=field_name,
                    new_value=new_value,
                )
            )

        return _on_aabb_psygnal

    def _wire_pick_write(self, visual: BaseVisual) -> None:
        """Subscribe to pick_write field changes on a visual model."""
        handler = self._make_pick_write_handler(visual.id)
        visual.events.pick_write.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.events.pick_write, handler)
        )

    def _make_pick_write_handler(self, visual_id: UUID) -> Callable:
        """Return a handler that emits PickWriteChangedEvent on pick_write changes."""

        def _on_pick_write(new_value: bool) -> None:
            self._outgoing_events.emit(
                PickWriteChangedEvent(
                    source_id=self._id,
                    visual_id=visual_id,
                    pick_write=new_value,
                )
            )

        return _on_pick_write

    def _make_appearance_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for any visual's appearance model.

        Routes field changes to one of two bus events: ``visible`` field changes
        become ``VisualVisibilityChangedEvent``; all other fields become
        ``AppearanceChangedEvent`` with ``requires_reslice=True`` for fields in
        ``_RESLICE_FIELDS`` (``lod_bias``, ``force_level``, ``frustum_cull``).
        """

        def _on_appearance_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_appearance_psygnal  visual=%s  field=%s"
                "  resolved_source=%s  override_active=%s",
                visual_id,
                field_name,
                resolved_source_id,
                _source_id_override.get() is not None,
            )
            if field_name == "visible":
                self._outgoing_events.emit(
                    VisualVisibilityChangedEvent(
                        source_id=resolved_source_id,
                        visual_id=visual_id,
                        visible=new_value,
                    )
                )
            else:
                self._outgoing_events.emit(
                    AppearanceChangedEvent(
                        source_id=resolved_source_id,
                        visual_id=visual_id,
                        field_name=field_name,
                        new_value=new_value,
                        requires_reslice=(field_name in _RESLICE_FIELDS),
                    )
                )
                if field_name in _RESLICE_FIELDS:
                    self.reslice_visual(visual_id)

        return _on_appearance_psygnal

    def reslice_all(self) -> None:
        """Trigger a data load for all visuals across all scenes."""
        for scene_id in self._model.scenes:
            self.reslice_scene(scene_id)

    def reslice_scene(
        self,
        scene_id: UUID,
        *,
        on_ready: Callable[[], None] | None = None,
        owner_id: UUID | None = None,
    ) -> None:
        """Trigger a data load for all visuals in one scene.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to reslice.
        on_ready : Callable[[], None] or None
            If provided, a zero-argument callback fired exactly once after
            *all* visuals loaded by this reslice have committed to the GPU
            (across every canvas attached to the scene).  Visuals with no data
            in the current view (culled, empty, or hidden) do not delay it.
            Works uniformly for in-memory, multiscale, multichannel, and
            geometry visuals.  See :meth:`on_scene_ready`.

            The callback tracks *this* reslice generation.  If a superseding
            reslice (e.g. a camera-settle reload) cancels these in-flight reads
            before they commit, the cancelled visual never reports completion
            and the callback may not fire.  Callers that need a guaranteed
            startup signal should suppress camera-driven reslicing during the
            load (the convenience launchers do this automatically).
        owner_id : UUID or None
            Owner under which the temporary ``on_ready`` subscriptions are
            registered (for teardown).  Defaults to the controller's own id.
        """
        dims_state = self._dims_state_for_scene(scene_id)
        visual_configs = self._build_visual_configs_for_scene(scene_id)

        if on_ready is None:
            self._render_manager.reslice_scene(scene_id, dims_state, visual_configs)
            return

        self._notify_when_resliced(
            scene_id,
            on_ready,
            owner_id or self._id,
            lambda: self._render_manager.reslice_scene(
                scene_id, dims_state, visual_configs
            ),
        )

    def _notify_when_resliced(
        self,
        scene_id: UUID,
        callback: Callable[[], None],
        owner_id: UUID,
        trigger: Callable[[], None],
    ) -> None:
        """Arm a one-shot quiescence tracker, then run *trigger* to reslice.

        The tracker counts visuals announced by ``ResliceStartedEvent`` and
        decrements on each ``ResliceCompletedEvent`` for *scene_id*; *callback*
        fires once the count drains to zero.  Because planning and the
        synchronous ``ResliceStartedEvent`` emissions all run inside *trigger*
        on a synchronous bus, the pending count is fully known the instant
        *trigger* returns — so the "armed" flag suppresses any premature fire
        from synchronous empty-batch completions until the full count is in.
        """
        state = {"pending": 0, "armed": True, "fired": False}
        handles: list[SubscriptionHandle] = []

        def _finish() -> None:
            if state["fired"]:
                return
            state["fired"] = True
            for handle in handles:
                self._outgoing_events.unsubscribe(handle)
            callback()

        def _maybe_fire() -> None:
            if not state["armed"] and not state["fired"] and state["pending"] <= 0:
                _finish()

        def _on_started(event: ResliceStartedEvent) -> None:
            state["pending"] += len(event.visual_ids)

        def _on_completed(event: ResliceCompletedEvent) -> None:
            if event.scene_id != scene_id:
                return
            state["pending"] -= 1
            _maybe_fire()

        handles.append(
            self._outgoing_events.subscribe(
                ResliceStartedEvent,
                _on_started,
                entity_id=scene_id,
                owner_id=owner_id,
            )
        )
        # ResliceCompletedEvent routes by visual_id, so subscribe as a
        # catch-all and filter on scene_id inside the handler.
        handles.append(
            self._outgoing_events.subscribe(
                ResliceCompletedEvent,
                _on_completed,
                owner_id=owner_id,
            )
        )

        trigger()

        # All ResliceStartedEvents for this generation have now been emitted
        # synchronously, so the pending count is complete.  Close the arming
        # window and fire immediately if there was nothing to load.
        state["armed"] = False
        _maybe_fire()

    def reslice_visual(self, visual_id: UUID) -> None:
        """Trigger a data load for one visual."""
        scene_id = self._visual_to_scene[visual_id]
        dims_state = self._dims_state_for_scene(scene_id)
        visual = self.get_visual_model(visual_id)
        if isinstance(visual, (MultiscaleImageVisual, MultiscaleLabelVisual)):
            cfg = VisualRenderConfig(
                lod_bias=visual.appearance.lod_bias,
                force_level=visual.appearance.force_level,
                frustum_cull=visual.appearance.frustum_cull,
            )
        else:
            cfg = VisualRenderConfig()
        self._render_manager.reslice_visual(visual_id, dims_state, cfg)

    @contextmanager
    def suppress_reslice(self) -> Generator[None, None, None]:
        """Context manager that blocks reslice_scene inside transform handlers.

        Use this when updating a visual's transform without needing to reload
        its underlying data — for example, repositioning a static-geometry mesh
        by translation only.

        .. warning::
            ``_suppress_reslice`` is a flat boolean.  Nested calls or concurrent
            async tasks that mutate transforms inside overlapping
            ``suppress_reslice`` blocks will interfere.  Replace with a depth
            counter if that becomes necessary.
        """
        self._suppress_reslice = True
        try:
            yield
        finally:
            self._suppress_reslice = False

    def set_visual_transform(
        self,
        visual_id: UUID,
        transform: AffineTransform,
        *,
        reslice: bool = True,
    ) -> None:
        """Update the data-to-world transform of a visual.

        Assigns *transform* to the live visual model, which fires its psygnal
        field event and propagates the change to the render layer via the
        event bus.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to update.
        transform : AffineTransform
            New data-to-world transform.
        reslice : bool
            If True (default), a full reslice is triggered after the transform
            is applied — required for image visuals where the transform changes
            which data falls in the current slab.  Pass False for
            static-geometry visuals (mesh, points, lines) where the transform
            only repositions the node and the underlying data is unchanged.
        """
        visual_model = self.get_visual_model(visual_id)
        if reslice:
            visual_model.transform = transform
        else:
            with self.suppress_reslice():
                visual_model.transform = transform

    def _on_dims_changed_bus(self, event: DimsChangedEvent) -> None:
        """Bus handler — reslice the scene whenever its dims state changes."""
        self.reslice_scene(event.scene_id)

    # ------------------------------------------------------------------
    # Model mutation with source-ID threading
    # ------------------------------------------------------------------

    def update_slice_indices(
        self,
        scene_id: UUID,
        slice_indices: dict[int, int],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set ``slice_indices`` on a scene's dims.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``DimsChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        slice_indices :
            Mapping of axis index → slice position.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.  Defaults
            to the controller's own ID.
        """
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  scene=%s  source=%s",
            scene_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            self._model.scenes[scene_id].dims.selection.slice_indices = slice_indices
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  scene=%s", scene_id)

    def update_appearance_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on a visual's appearance model.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``AppearanceChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        visual_id :
            Target visual.
        field :
            Attribute name on the appearance model, e.g. ``"clim"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``AppearanceChangedEvent``.  Defaults
            to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  field=%s  visual=%s  source=%s",
            field,
            visual_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            setattr(visual.appearance, field, value)
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  field=%s  visual=%s", field, visual_id)

    def update_channel_appearance_field(
        self,
        visual_id: UUID,
        channel_index: int,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on one channel of a multichannel visual.

        Tags the emitted bus event with *source_id*.  GUI widgets should pass
        ``source_id=self._id`` so their own ``ChannelAppearanceChangedEvent``
        subscription can ignore the echo.

        This mutates a single visual only.  When a channel is shared in
        lock-step across several panels (e.g. an ``OrthoViewer``), calling this
        on one panel's visual leaves the sibling panels unequal until they are
        written too; use ``update_channel_group_field`` to keep the group in
        lock-step.

        A ``pydantic.ValidationError`` from a malformed *value* is allowed to
        propagate (matching ``update_appearance_field``).

        Parameters
        ----------
        visual_id :
            Target visual.
        channel_index :
            Index into ``visual.channels`` selecting the ``ChannelAppearance``.
        field :
            Attribute name on the channel appearance model, e.g. ``"clim"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``ChannelAppearanceChangedEvent``.
            Defaults to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  channel=%d  field=%s  visual=%s  source=%s",
            channel_index,
            field,
            visual_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            setattr(visual.channels[channel_index], field, value)
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug(
                "reset  channel=%d  field=%s  visual=%s",
                channel_index,
                field,
                visual_id,
            )

    def update_channel_group_field(
        self,
        visual_ids: list[UUID],
        channel_index: int,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one channel field across a group of visuals in lock-step.

        Fan-out over ``update_channel_appearance_field`` so every visual in the
        group (e.g. the per-panel visuals of an ``OrthoViewer``) receives the
        same channel change.  This is the programmatic write-side companion to
        the widget subscribe-to-all read side.

        Parameters
        ----------
        visual_ids :
            Target visuals sharing the channel set.
        channel_index :
            Index into each visual's ``channels`` mapping.
        field :
            Attribute name on the channel appearance model.
        value :
            New value for the field.
        source_id :
            UUID to stamp on each emitted ``ChannelAppearanceChangedEvent``.
            Defaults to the controller's own ID.
        """
        for visual_id in visual_ids:
            self.update_channel_appearance_field(
                visual_id, channel_index, field, value, source_id=source_id
            )

    def update_aabb_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on a visual's AABB params model.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``AABBChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        visual_id :
            Target visual.
        field :
            Attribute name on the AABB model, e.g. ``"enabled"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``AABBChangedEvent``.  Defaults
            to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  field=%s  visual=%s  source=%s",
            field,
            visual_id,
            resolved_source_id,
        )
        token = _aabb_source_id_override.set(source_id)
        try:
            setattr(visual.aabb, field, value)
        finally:
            _aabb_source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  field=%s  visual=%s", field, visual_id)

    def update_displayed_axes(
        self,
        scene_id: UUID,
        displayed_axes: tuple[int, ...],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set ``displayed_axes`` on a scene's dims.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``DimsChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        displayed_axes :
            Tuple of axis indices to display; length 2 for 2D, 3 for 3D.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.  Defaults
            to the controller's own ID.
        """
        token = _source_id_override.set(source_id)
        try:
            self._model.scenes[scene_id].dims.selection.displayed_axes = displayed_axes
        finally:
            _source_id_override.reset(token)

    def update_stacked_axes(
        self,
        scene_id: UUID,
        stacked_axes: tuple[int, ...],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set ``stacked_axes`` on a scene's dims.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``DimsChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        stacked_axes :
            Tuple of axis indices whose full extent is composited by the render
            layer (e.g. channel axis).  Pass ``()`` for no stacked axes.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.  Defaults
            to the controller's own ID.
        """
        token = _source_id_override.set(source_id)
        try:
            self._model.scenes[scene_id].dims.selection.stacked_axes = stacked_axes
        finally:
            _source_id_override.reset(token)

    def set_displayed_axes(
        self,
        scene_id: UUID,
        displayed_axes: tuple[int, ...],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set displayed axes on a scene's dims (preferred public API).

        Equivalent to :meth:`update_displayed_axes`.  The controller's
        psygnal bridge fires ``_rebuild_visuals_geometry`` and
        ``_switch_canvas_cameras`` automatically when the model field changes.

        Parameters
        ----------
        scene_id :
            Target scene.
        displayed_axes :
            Tuple of axis indices to display; length 2 for 2D, 3 for 3D.
        source_id :
            UUID stamped on the emitted ``DimsChangedEvent``.
        """
        self.update_displayed_axes(scene_id, displayed_axes, source_id=source_id)

    def set_stacked_axes(
        self,
        scene_id: UUID,
        stacked_axes: tuple[int, ...],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set stacked axes on a scene's dims (preferred public API).

        Equivalent to :meth:`update_stacked_axes`.  After updating the model,
        notifies each visual via ``on_stacked_axes_changed`` so they can
        adjust internal state (e.g. LUT channel routing) without a node swap.

        Parameters
        ----------
        scene_id :
            Target scene.
        stacked_axes :
            Tuple of axis indices whose full extent is composited by the
            render layer (e.g. channel axis).  Pass ``()`` for none.
        source_id :
            UUID stamped on the emitted ``DimsChangedEvent``.
        """
        self.update_stacked_axes(scene_id, stacked_axes, source_id=source_id)
        scene = self._model.scenes[scene_id]
        scene_manager = self._render_manager._scenes[scene_id]
        for visual_model in scene.visuals:
            gfx_visual = scene_manager.get_visual(visual_model.id)
            gfx_visual.on_stacked_axes_changed(stacked_axes)

    # ------------------------------------------------------------------
    # IncomingEventBus handlers
    # ------------------------------------------------------------------

    def _on_appearance_update(self, event: AppearanceUpdateEvent) -> None:
        self.update_appearance_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

    def _on_dims_update(self, event: DimsUpdateEvent) -> None:
        if event.displayed_axes is not None and event.slice_indices is not None:
            # Order matters: every axis must stay covered by either
            # "displayed" or "sliced" at all times (see
            # AxisAlignedSelectionState.to_index_selection). Expanding the
            # displayed set must extend displayed_axes before slice_indices
            # drops the newly-displayed axes; contracting must add the
            # newly-hidden axes to slice_indices before displayed_axes
            # shrinks. Mirrors Viewer.set_displayed_dimensions.
            current_displayed = set(
                self._model.scenes[event.scene_id].dims.selection.displayed_axes
            )
            is_expanding = bool(set(event.displayed_axes) - current_displayed)
            if is_expanding:
                self.update_displayed_axes(
                    event.scene_id, event.displayed_axes, source_id=event.source_id
                )
                self.update_slice_indices(
                    event.scene_id, event.slice_indices, source_id=event.source_id
                )
            else:
                self.update_slice_indices(
                    event.scene_id, event.slice_indices, source_id=event.source_id
                )
                self.update_displayed_axes(
                    event.scene_id, event.displayed_axes, source_id=event.source_id
                )
        else:
            if event.slice_indices is not None:
                self.update_slice_indices(
                    event.scene_id, event.slice_indices, source_id=event.source_id
                )
            if event.displayed_axes is not None:
                self.update_displayed_axes(
                    event.scene_id, event.displayed_axes, source_id=event.source_id
                )
        if event.stacked_axes is not None:
            self.update_stacked_axes(
                event.scene_id, event.stacked_axes, source_id=event.source_id
            )

    def _on_aabb_update(self, event: AABBUpdateEvent) -> None:
        self.update_aabb_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

    def _on_channel_appearance_update(
        self, event: ChannelAppearanceUpdateEvent
    ) -> None:
        self.update_channel_appearance_field(
            event.visual_id,
            event.channel_index,
            event.field,
            event.value,
            source_id=event.source_id,
        )

    # ------------------------------------------------------------------
    # Camera settle
    # ------------------------------------------------------------------

    @property
    def camera_reslice_enabled(self) -> bool:
        """Whether camera movement triggers automatic reslicing."""
        return self._render_manager.config.camera.reslice_enabled

    @camera_reslice_enabled.setter
    def camera_reslice_enabled(self, value: bool) -> None:
        self._render_manager.config.camera.reslice_enabled = value
        if not value:
            for task in self._settle_tasks.values():
                if not task.done():
                    task.cancel()
            self._settle_tasks.clear()

    @property
    def render_config(self) -> RenderManagerConfig:
        """Live rendering configuration.

        Mutating a field here changes the model but not the GPU state; for
        the outline pass, follow a mutation with
        :meth:`apply_outline_config`.  The dedicated properties
        (``outline_enabled`` and friends) do both in one step.
        """
        return self._render_manager.config

    @property
    def outline_enabled(self) -> bool:
        """Whether the screen-space outline pass is active."""
        return self._render_manager.outline_enabled

    @outline_enabled.setter
    def outline_enabled(self, value: bool) -> None:
        self._render_manager.outline_enabled = value

    @property
    def outline_boundaries_enabled(self) -> bool:
        """Whether the boundaries layer (every outlined region) draws."""
        return self._render_manager.outline_boundaries_enabled

    @outline_boundaries_enabled.setter
    def outline_boundaries_enabled(self, value: bool) -> None:
        self._render_manager.outline_boundaries_enabled = value

    @property
    def outline_selection_enabled(self) -> bool:
        """Whether the selection layer (regions with a palette slot) draws."""
        return self._render_manager.outline_selection_enabled

    @outline_selection_enabled.setter
    def outline_selection_enabled(self, value: bool) -> None:
        self._render_manager.outline_selection_enabled = value

    @property
    def ssao_enabled(self) -> bool:
        """Whether the screen-space ambient occlusion pass is active.

        Ambient occlusion darkens creases by sampling the depth buffer, and
        is the cheapest shape cue available for cellier's default unlit
        isosurfaces.  It runs in 3D only.
        """
        return self._render_manager.ssao_enabled

    @ssao_enabled.setter
    def ssao_enabled(self, value: bool) -> None:
        self._render_manager.ssao_enabled = value

    @property
    def ssao_radius(self) -> float | None:
        """Occlusion hemisphere radius in scene units, or ``None`` for auto.

        ``None`` derives the radius from the scene bounding box diagonal
        (``render_config.ssao.auto_radius_fraction``, 2 percent by
        default), which is the only default that means anything across
        cellier's coordinate systems.
        """
        return self._render_manager.ssao_radius

    @ssao_radius.setter
    def ssao_radius(self, value: float | None) -> None:
        self._render_manager.ssao_radius = value

    @property
    def ssao_strength(self) -> float:
        """How far the occlusion is applied, 0 (off) to 1 (full)."""
        return self._render_manager.ssao_strength

    @ssao_strength.setter
    def ssao_strength(self, value: float) -> None:
        self._render_manager.ssao_strength = value

    @property
    def ssao_power(self) -> float:
        """Contrast exponent applied to the occlusion before the multiply."""
        return self._render_manager.ssao_power

    @ssao_power.setter
    def ssao_power(self, value: float) -> None:
        self._render_manager.ssao_power = value

    def apply_ssao_config(self) -> None:
        """Push ``render_config.ssao`` onto every canvas's occlusion pass.

        Needed after mutating the config model in place.  ``n_samples`` and
        ``blur_radius`` are shader template vars, so changing them
        recompiles; the rest are uniforms and do not.
        """
        self._render_manager.apply_ssao_config()

    def apply_outline_config(self) -> None:
        """Push ``render_config.outline`` onto every canvas's outline pass.

        Call this after mutating thicknesses or colours in place.  Changing
        a thickness recompiles the outline shader; enables, colours and the
        palette do not.
        """
        self._render_manager.apply_outline_config()

    @property
    def camera_settle_threshold_s(self) -> float:
        """Debounce delay before reslice after camera movement."""
        return self._render_manager.config.camera.settle_threshold_s

    @camera_settle_threshold_s.setter
    def camera_settle_threshold_s(self, value: float) -> None:
        self._render_manager.config.camera.settle_threshold_s = value

    def _on_canvas_size_changed(self, event: CanvasSizeChangedEvent) -> None:
        """Update Canvas.size in the model when the backend reports a resize."""
        scene_id = self._canvas_to_scene.get(event.canvas_id)
        if scene_id is None:
            return
        canvas_model = self._model.scenes[scene_id].canvases.get(event.canvas_id)
        if canvas_model is None:
            return
        canvas_model.size = (event.width, event.height)

    def _on_camera_changed(self, event: CameraChangedEvent) -> None:
        """Synchronous bus handler: updates camera model and schedules settle task."""
        self._update_camera_model(event.scene_id, event.source_id, event.camera_state)

        if not self._render_manager.config.camera.reslice_enabled:
            return

        canvas_id = event.source_id
        existing = self._settle_tasks.get(canvas_id)
        if existing is not None and not existing.done():
            _CAMERA_LOGGER.debug(
                "settle_cancel  canvas=%s  scene=%s",
                canvas_id,
                event.scene_id,
            )
            existing.cancel()

        _CAMERA_LOGGER.debug(
            "settle_schedule  canvas=%s  scene=%s  threshold=%.3fs",
            canvas_id,
            event.scene_id,
            self._render_manager.config.camera.settle_threshold_s,
        )
        self._settle_tasks[canvas_id] = asyncio.create_task(
            self._settle_after(canvas_id, event.scene_id)
        )

    def _update_camera_model(
        self, scene_id: UUID, canvas_id: UUID, camera_state: CameraState
    ) -> None:
        """Write a CameraState snapshot back into the model-layer camera.

        Branches on the actual model type: ``PerspectiveCamera`` writes
        ``up_direction`` and ``fov``; ``OrthographicCamera`` writes ``width``
        and ``height`` (``extent``).

        Parameters
        ----------
        scene_id : UUID
            Scene that owns the canvas.
        canvas_id : UUID
            The specific canvas whose camera moved (``event.source_id``).
        camera_state : CameraState
            Snapshot to write back.
        """
        scene = self._model.scenes[scene_id]
        canvas_model = scene.canvases.get(canvas_id)
        if canvas_model is None:
            return
        canvas_view = self._render_manager._canvases.get(canvas_id)
        if canvas_view is not None:
            active_dim = canvas_view._dim
        else:
            # No render-layer canvas (e.g. headless tests): infer from type.
            active_dim = "2d" if camera_state.camera_type == "orthographic" else "3d"
        camera_model = canvas_model.cameras.get(active_dim)
        if camera_model is None:
            return

        # Common fields present on both camera model types.
        camera_model.position = np.array(camera_state.position, dtype=np.float32)
        camera_model.rotation = np.array(camera_state.rotation, dtype=np.float32)
        camera_model.zoom = camera_state.zoom
        camera_model.near_clipping_plane = camera_state.depth_range[0]
        camera_model.far_clipping_plane = camera_state.depth_range[1]

        # Type-specific fields.
        if isinstance(camera_model, PerspectiveCamera):
            camera_model.up_direction = np.array(camera_state.up, dtype=np.float32)
            camera_model.fov = camera_state.fov
        elif isinstance(camera_model, OrthographicCamera):
            camera_model.width = camera_state.extent[0]
            camera_model.height = camera_state.extent[1]

    async def _settle_after(self, canvas_id: UUID, scene_id: UUID) -> None:
        """Wait for the settle threshold, then reslice camera-sensitive visuals."""
        try:
            await asyncio.sleep(self._render_manager.config.camera.settle_threshold_s)
        except asyncio.CancelledError:
            raise

        scene = self._model.scenes[scene_id]
        target_ids = frozenset(v.id for v in scene.visuals if v.requires_camera_reslice)

        if not target_ids:
            return

        dims_state = self._dims_state_for_scene(scene_id)
        visual_configs = self._build_visual_configs_for_scene(scene_id)

        _CAMERA_LOGGER.info(
            "settle_reslice  scene=%s  visuals=%d",
            scene_id,
            len(target_ids),
        )

        self._render_manager.reslice_scene(
            scene_id=scene_id,
            dims_state=dims_state,
            visual_configs=visual_configs,
            target_visual_ids=target_ids,
        )

    # ------------------------------------------------------------------
    # Camera operations
    # ------------------------------------------------------------------

    def look_at_visual(
        self,
        visual_id: UUID,
        canvas_id: UUID,
        view_direction: tuple[float, float, float] = (-1, -1, -1),
        up: tuple[float, float, float] = (0, 0, 1),
    ) -> None:
        """Fit the camera to a visual's bounding box.

        Parameters
        ----------
        visual_id : UUID
            ID of the target visual.
        canvas_id : UUID
            ID of the canvas whose camera should be fitted.
        view_direction : tuple[float, float, float]
            Camera look direction vector (need not be normalized).
        up : tuple[float, float, float]
            Camera up vector.
        """
        self._render_manager.look_at_visual(visual_id, canvas_id, view_direction, up)
        scene_id = self._visual_to_scene[visual_id]
        state = self.get_canvas_view(canvas_id).capture_camera_state()
        self._update_camera_model(scene_id, canvas_id, state)

    def set_camera_depth_range(
        self,
        canvas_id: UUID,
        depth_range: tuple[float, float],
    ) -> None:
        """Set the near/far clip distances for a canvas camera.

        Parameters
        ----------
        canvas_id : UUID
            ID of the target canvas.
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances in world units.
        """
        self._render_manager.set_camera_depth_range(canvas_id, depth_range)

    # ------------------------------------------------------------------
    # Stubs for future features
    # ------------------------------------------------------------------

    def add_paint_controller(
        self,
        visual_id: UUID,
        canvas_id: UUID,
        brush_value: int = 1,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
        autosave_interval_s: float | None = None,
    ):
        """Create and wire a paint controller for the visual's data store.

        Parameters
        ----------
        visual_id : UUID
            Visual to paint on.
        canvas_id : UUID
            Canvas to bind to.  Its camera controller is disabled for
            the session.  Pass ``controller.get_canvas_ids(scene_id)[0]``
            for the common single-canvas case.
        brush_value : int
            Integer label ID written to every painted voxel.
        brush_radius_voxels : float
            Brush radius in level-0 voxel units.
        history_depth : int
            Maximum undoable strokes.
        autosave_interval_s : float | None
            Seconds between automatic flushes for ``MultiscalePaintController``.
            Each autosave rebuilds the pyramid and resets GPU paint textures.
            ``None`` disables autosave.  Ignored for ``SyncPaintController``.

        Returns
        -------
        AbstractPaintController
            Fully wired; caller owns the object.

        Raises
        ------
        TypeError
            If the data store type has no registered paint controller.
        """
        from cellier.data.image._image_memory_store import ImageMemoryStore
        from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore
        from cellier.data.image._zarr_multiscale_store import (
            MultiscaleZarrDataStore,
        )
        from cellier.data.label._label_memory_store import LabelMemoryStore

        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        visual_model = next(v for v in scene.visuals if v.id == visual_id)
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]

        if isinstance(data_store, (ImageMemoryStore, LabelMemoryStore)):
            from cellier.paint import SyncPaintController

            displayed_axes = scene.dims.selection.displayed_axes
            if len(displayed_axes) != 2:
                raise NotImplementedError(
                    "SyncPaintController currently only supports 2-D "
                    f"displayed-axis configurations; scene {scene_id} has "
                    f"displayed_axes={displayed_axes!r}."
                )
            return SyncPaintController(
                cellier_controller=self,
                visual_id=visual_id,
                scene_id=scene_id,
                canvas_id=canvas_id,
                data_store=data_store,
                displayed_axes=displayed_axes,
                brush_value=brush_value,
                brush_radius_voxels=brush_radius_voxels,
                history_depth=history_depth,
            )

        from cellier.data.label._ome_zarr_label_store import OMEZarrLabelDataStore

        if isinstance(
            data_store,
            (OMEZarrImageDataStore, MultiscaleZarrDataStore, OMEZarrLabelDataStore),
        ):
            from cellier.paint import MultiscalePaintController

            displayed_axes = scene.dims.selection.displayed_axes
            if len(displayed_axes) != 2:
                raise NotImplementedError(
                    "MultiscalePaintController currently only supports 2-D "
                    f"displayed-axis configurations; scene {scene_id} has "
                    f"displayed_axes={displayed_axes!r}.  3-D paint "
                    "feedback is a Phase 3 follow-up."
                )

            block_size = visual_model.render_config.block_size
            return MultiscalePaintController(
                cellier_controller=self,
                visual_id=visual_id,
                scene_id=scene_id,
                canvas_id=canvas_id,
                data_store=data_store,
                visual_block_size=block_size,
                displayed_axes=displayed_axes,
                brush_value=brush_value,
                brush_radius_voxels=brush_radius_voxels,
                history_depth=history_depth,
                autosave_interval_s=autosave_interval_s,
            )

        raise TypeError(
            f"No PaintController implementation for data store type "
            f"{type(data_store).__name__!r}.  "
            f"Supported: ImageMemoryStore, LabelMemoryStore, "
            f"OMEZarrImageDataStore, MultiscaleZarrDataStore, OMEZarrLabelDataStore."
        )

    def remove_scene(self, scene_id: UUID) -> None:
        """Remove a scene and all its visuals and canvases.

        Teardown order mirrors ``remove_visual`` for each child visual, then
        cleans up the scene-level maps and render layer.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to remove.

        Raises
        ------
        KeyError
            If ``scene_id`` is not registered.
        """
        scene = self._model.scenes[scene_id]

        # 1. Tear down all child visuals (psygnal + bus + render layer each).
        for visual_model in list(scene.visuals):
            self.remove_visual(visual_model.id)

        # 2. Cancel any pending camera-settle tasks for this scene's canvases.
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            task = self._settle_tasks.pop(canvas_id, None)
            if task is not None and not task.done():
                task.cancel()

        # 3. Bus cleanup for canvases and the scene itself.
        for canvas_id in self._scene_to_canvases.pop(scene_id, []):
            self._outgoing_events.unsubscribe_all(canvas_id)
            self._canvas_to_scene.pop(canvas_id, None)
        self._outgoing_events.unsubscribe_all(scene_id)

        # 4. Clean up controller-side scene maps.
        self._dims_cache.pop(scene_id, None)
        self._scene_render_modes.pop(scene_id, None)

        # 5. Remove from model layer.
        self._model.scenes.pop(scene_id)

        # 6. Render-layer teardown (drops gfx.Scene, canvas widgets, GPU refs).
        self._render_manager.remove_scene(scene_id)

        # 7. Notify external observers.
        self._outgoing_events.emit(
            SceneRemovedEvent(source_id=self._id, scene_id=scene_id)
        )

    def remove_canvas(self, canvas_id: UUID) -> None:
        """Remove a canvas from its scene, disconnecting all wiring.

        Teardown order mirrors ``remove_scene`` for the canvas-level steps:
        1. Cancel any pending camera-settle task.
        2. Remove bus subscriptions owned by this canvas.
        3. Update controller lookup maps.
        4. Remove from the model layer.
        5. Render-layer teardown (drops widget and GPU references).

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas to remove.

        Raises
        ------
        KeyError
            If ``canvas_id`` is not registered.
        """
        scene_id = self._canvas_to_scene[canvas_id]

        # 1. Cancel any pending camera-settle task.
        task = self._settle_tasks.pop(canvas_id, None)
        if task is not None and not task.done():
            task.cancel()

        # 2. Remove bus subscriptions owned by this canvas.
        self._outgoing_events.unsubscribe_all(canvas_id)
        self._pick_subscriber_counts.pop(canvas_id, None)

        # 3. Update controller lookup maps.
        self._canvas_to_scene.pop(canvas_id)
        self._scene_to_canvases[scene_id].remove(canvas_id)

        # 4. Remove from the model layer.
        self._model.scenes[scene_id].canvases.pop(canvas_id)

        # 5. Render-layer teardown.
        self._render_manager.remove_canvas(canvas_id)

    def remove_visual(self, visual_id: UUID) -> None:
        """Remove a visual from its scene, disconnecting all wiring.

        Teardown order:
        1. Psygnal bridge handlers disconnected first — prevents the
           bridge closures from firing during any subsequent model access.
        2. Bus subscriptions removed — prevents dangling GFX-layer handlers
           from receiving events after the node is gone from the scene graph.
        3. Render-layer removal — drops scene-graph node and GPU references.
        4. ``VisualRemovedEvent`` emitted for external observers.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.

        Raises
        ------
        KeyError
            If ``visual_id`` is not registered.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        visual_model = self.get_visual_model(visual_id)

        # 1. Remove from model layer.
        scene.visuals.remove(visual_model)

        # 2. Disconnect psygnal bridge handlers.
        for signal, handler in self._visual_psygnal_handlers.pop(visual_id, []):
            signal.disconnect(handler)

        # 3. Remove bus subscriptions for this visual's GFX handlers.
        self._outgoing_events.unsubscribe_all(visual_id)

        # 4. Remove from controller lookup maps.
        self._visual_to_scene.pop(visual_id)

        # 5. Render-layer teardown.
        self._render_manager.remove_visual(visual_id)

        # 6. Notify external observers.
        self._outgoing_events.emit(
            VisualRemovedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_id,
            )
        )

    def remove_data_store(self, data_store_id: UUID) -> None:
        """Remove a data store from the model.

        Raises ``ValueError`` if any live visual still references the store.
        Call ``remove_visual`` for each referencing visual first.

        Parameters
        ----------
        data_store_id : UUID
            ID of the data store to remove.

        Raises
        ------
        ValueError
            If one or more visuals still reference the store.  The error
            message names each visual so the caller can identify them.
        KeyError
            If ``data_store_id`` is not registered.
        """
        referencing = [
            v
            for scene in self._model.scenes.values()
            for v in scene.visuals
            if UUID(v.data_store_id) == data_store_id
        ]
        if referencing:
            names = ", ".join(f"{v.name!r} ({v.id})" for v in referencing)
            raise ValueError(
                f"Cannot remove data store {data_store_id}: "
                f"still referenced by visuals: {names}"
            )
        self._model.data.stores.pop(data_store_id)

    # ------------------------------------------------------------------
    # External event subscriptions
    # ------------------------------------------------------------------

    def on_dims_changed(
        self,
        scene_id: UUID,
        callback: Callable[[DimsChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the dims for *scene_id* change.

        The callback receives the full ``DimsChangedEvent``, which includes
        ``source_id`` for echo-filtering and ``dims_state`` for the new state.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``DimsChangedEvent`` on each dims change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.  Use for
            transient widgets that may be destroyed outside the controller's
            teardown path.  Cannot be used with lambdas.

        Returns
        -------
        SubscriptionHandle
            Pass to ``EventBus.unsubscribe()`` for individual removal.
        """
        return self._outgoing_events.subscribe(
            DimsChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_camera_changed(
        self,
        scene_id: UUID,
        callback: Callable[[CameraChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the camera for *scene_id* changes.

        The callback receives a ``CameraChangedEvent`` carrying the latest
        ``CameraState``, including ``extent`` (width, height) for
        ``OrthographicCamera`` scenes.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``CameraChangedEvent`` on each camera change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            CameraChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def _on_raw_pointer_event(self, event: _CanvasRawPointerEvent) -> None:
        """Embed the 2D camera position in N-dimensional world space.

        Reads displayed_axes and slice_indices from the scene model to fill
        the full world coordinate, then emits the public canvas mouse event.
        No render-layer access occurs here.
        """
        from cellier.events._events import (
            CanvasPickInfo,
            ImagePickInfo,
            LabelsPickInfo,
        )
        from cellier.render.render_manager import (
            _ImageDisplayedDataCoord,
            _LabelsDisplayedDataCoord,
        )

        scene_model = self._model.scenes[event.scene_id]
        dims = scene_model.dims
        displayed_axes = dims.selection.displayed_axes
        slice_indices = dims.selection.slice_indices
        axis_labels = dims.coordinate_system.axis_labels
        n_dims = len(axis_labels)

        # Promote render-layer intermediates to full-N-dim public pick types.
        # The render layer decodes the displayed axes into level-0 data
        # coordinates; the controller fills the remaining (non-displayed) axes
        # from the dims slice state — which are already data indices, so every
        # axis of the result is in the same (data) coordinate system.
        raw_pick = event.pick_details
        if isinstance(raw_pick, (_ImageDisplayedDataCoord, _LabelsDisplayedDataCoord)):
            data_coord_pick = np.empty(n_dims, dtype=np.float64)
            # The render layer decodes displayed-axis coordinates in pygfx
            # (x, y[, z]) order, which is the reverse of cellier's ascending
            # ``displayed_axes`` (..., row, col) order.  Reverse so each value
            # lands on its true data axis (e.g. the column coordinate on the x
            # axis, not on the first displayed axis) — otherwise the displayed
            # axes come out transposed.
            disp_coord = tuple(raw_pick.displayed_data_coord)[::-1]
            for i, axis in enumerate(displayed_axes):
                data_coord_pick[axis] = disp_coord[i]
            for axis, idx in slice_indices.items():
                data_coord_pick[axis] = float(idx)
            full_coord: tuple[float, ...] = tuple(float(v) for v in data_coord_pick)
            if isinstance(raw_pick, _ImageDisplayedDataCoord):
                promoted_pick = ImagePickInfo(data_coordinate=full_coord)
            else:
                promoted_pick = LabelsPickInfo(data_coordinate=full_coord)
        else:
            promoted_pick = raw_pick

        pick_info = CanvasPickInfo(
            hit_visual_id=event.hit_visual_id, details=promoted_pick
        )

        if event.camera_type == "2d":
            world_coord = np.empty(n_dims, dtype=np.float64)
            # Embed pygfx position_2d into world_coord verbatim:
            #   world_coord[displayed_axes[0]] = position_2d[0]  (pygfx-x)
            #   world_coord[displayed_axes[1]] = position_2d[1]  (pygfx-y)
            # Both image visuals render data[r, c] at pygfx (x=c, y=r), so
            # world_coord[ax_row] == pygfx_x (column) and
            # world_coord[ax_col] == pygfx_y (row) after this assignment.
            # Paint controllers swap ax_row ↔ ax_col to recover (row, col)
            # voxel order before calling imap_coordinates.
            for i, axis in enumerate(displayed_axes):
                world_coord[axis] = event.position_2d[i]  # type: ignore[index]
            for axis, idx in slice_indices.items():
                world_coord[axis] = float(idx)

            _CLS_2D = {
                "press": CanvasMousePress2DEvent,
                "move": CanvasMouseMove2DEvent,
                "release": CanvasMouseRelease2DEvent,
            }
            self._outgoing_events.emit(
                _CLS_2D[event.action](
                    source_id=event.canvas_id,
                    scene_id=event.scene_id,
                    world_coordinate=world_coord,
                    pick_info=pick_info,
                    button=event.button,
                    buttons=event.buttons,
                    modifiers=event.modifiers,
                    gesture_id=event.gesture_id,
                )
            )
        else:
            _CLS_3D = {
                "press": CanvasMousePress3DEvent,
                "move": CanvasMouseMove3DEvent,
                "release": CanvasMouseRelease3DEvent,
            }
            self._outgoing_events.emit(
                _CLS_3D[event.action](
                    source_id=event.canvas_id,
                    scene_id=event.scene_id,
                    ray=event.ray,
                    pick_info=pick_info,
                    button=event.button,
                    buttons=event.buttons,
                    modifiers=event.modifiers,
                    gesture_id=event.gesture_id,
                )
            )

    def _register_pick_subscriber(
        self, canvas_id: UUID, handle: SubscriptionHandle
    ) -> SubscriptionHandle:
        """Record a mouse subscription and enable pick details for its canvas.

        Increments the per-canvas picking-subscriber count; the first
        subscriber flips on element-detail extraction in the render layer.

        Parameters
        ----------
        canvas_id : UUID
            The canvas the subscription watches.
        handle : SubscriptionHandle
            The handle returned by the bus subscription.

        Returns
        -------
        SubscriptionHandle
            The same handle, for convenient return from the callers.
        """
        self._mouse_handle_canvas[id(handle)] = canvas_id
        count = self._pick_subscriber_counts.get(canvas_id, 0)
        self._pick_subscriber_counts[canvas_id] = count + 1
        if count == 0:
            self._render_manager.set_pick_details_enabled(canvas_id, True)
        return handle

    def unsubscribe_mouse(self, handle: SubscriptionHandle) -> None:
        """Remove a mouse subscription created by an ``on_mouse_*`` method.

        Use this in place of ``EventBus.unsubscribe`` for handles returned by
        the six ``on_mouse_*`` methods so the per-canvas picking-subscriber
        count stays accurate; the last unsubscribe disables element-detail
        extraction for that canvas.

        Parameters
        ----------
        handle : SubscriptionHandle
            A handle returned by one of the ``on_mouse_*`` methods.
        """
        canvas_id = self._mouse_handle_canvas.pop(id(handle), None)
        if canvas_id is not None:
            count = self._pick_subscriber_counts.get(canvas_id, 0) - 1
            if count <= 0:
                self._pick_subscriber_counts.pop(canvas_id, None)
                self._render_manager.set_pick_details_enabled(canvas_id, False)
            else:
                self._pick_subscriber_counts[canvas_id] = count
        self._outgoing_events.unsubscribe(handle)

    def on_mouse_press_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMousePress2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-down event on a 2D canvas.

        Parameters
        ----------
        canvas_id :
            The canvas to watch.
        callback :
            Called with the CanvasMousePress2DEvent on each press.
        owner_id :
            UUID under which this subscription is registered for bulk
            removal via unsubscribe_all(owner_id).
        weak :
            If True, hold only a weak reference to *callback*.
        """
        handle = self._outgoing_events.subscribe(
            CanvasMousePress2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return self._register_pick_subscriber(canvas_id, handle)

    def on_mouse_move_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseMove2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-move event on a 2D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseMove2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return self._register_pick_subscriber(canvas_id, handle)

    def on_mouse_release_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseRelease2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-up event on a 2D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseRelease2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return self._register_pick_subscriber(canvas_id, handle)

    def on_mouse_press_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMousePress3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-down event on a 3D canvas.

        Parameters
        ----------
        canvas_id :
            The canvas to watch.
        callback :
            Called with the CanvasMousePress3DEvent on each press.
        owner_id :
            UUID under which this subscription is registered for bulk
            removal via unsubscribe_all(owner_id).
        weak :
            If True, hold only a weak reference to *callback*.
        """
        handle = self._outgoing_events.subscribe(
            CanvasMousePress3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return self._register_pick_subscriber(canvas_id, handle)

    def on_mouse_move_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseMove3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-move event on a 3D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseMove3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return self._register_pick_subscriber(canvas_id, handle)

    def on_mouse_release_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseRelease3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-up event on a 3D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseRelease3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return self._register_pick_subscriber(canvas_id, handle)

    def set_camera_controller_enabled(self, canvas_id: UUID, enabled: bool) -> None:
        """Enable or disable the camera controller for one canvas.

        Parameters
        ----------
        canvas_id : UUID
            The canvas whose controller state should change.
        enabled : bool
            False disables the controller (paint session active).
            True restores normal camera interaction (session ended).
        """
        self._render_manager._canvases[canvas_id].set_controller_enabled(enabled)

    def _invalidate_painted_tiles_2d(
        self, visual_id: UUID, dirty_grid_coords: set[tuple[int, int]]
    ) -> int:
        """Evict the visible 2D-cache tiles for *visual_id* listed in dirty grid coords.

        Used by :class:`MultiscalePaintController` after staging level-0
        writes so the next reslice re-fetches through the open paint
        transaction.  No-op for visuals that don't own a 2D tile cache.

        Parameters
        ----------
        visual_id :
            The painted multiscale visual.
        dirty_grid_coords :
            Set of ``(gy, gx)`` tile-grid coordinates whose finest-level
            tiles should be evicted.

        Returns
        -------
        int
            Number of tiles evicted (0 if the visual has no 2D cache or
            no entries matched).
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "invalidate_painted_tiles_2d"):
            return 0
        return gfx_visual.invalidate_painted_tiles_2d(dirty_grid_coords)

    def _patch_painted_tiles_2d(
        self,
        visual_id: UUID,
        voxel_indices: np.ndarray,
        values: np.ndarray,
        displayed_axes: tuple[int, int],
    ) -> int:
        """Write paint into the visual's GPU paint cache (Phase-2 fast path).

        Used by :class:`MultiscalePaintController._write_values` for
        sub-frame visible feedback in 2-D paint sessions.  Mirrors the
        architectural pattern of :meth:`_invalidate_painted_tiles_2d`.

        Parameters
        ----------
        visual_id :
            The painted multiscale visual.
        voxel_indices :
            Shape ``(N, ndim)`` int64.  Level-0 voxel indices in
            data-array axis order.
        values :
            Shape ``(N,)`` float32.  Brush values.
        displayed_axes :
            ``(row_axis, col_axis)`` for the 2-D display.

        Returns
        -------
        int
            Number of tiles successfully patched.  Returns 0 if the visual
            has no 2-D paint resources or every tile was rejected by the
            slot manager.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "patch_paint_texture"):
            return 0
        return gfx_visual.patch_paint_texture(voxel_indices, values, displayed_axes)

    def _clear_painted_tiles_2d(self, visual_id: UUID) -> None:
        """Clear the GPU paint textures for *visual_id*.

        Called by :class:`MultiscalePaintController.commit` and ``abort`` at
        session end.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "clear_paint_textures"):
            return
        gfx_visual.clear_paint_textures()

    def on_appearance_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AppearanceChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the appearance of *visual_id* changes.

        The callback receives the full ``AppearanceChangedEvent``, which
        includes ``source_id`` for echo-filtering, ``field_name``, and
        ``new_value``.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``AppearanceChangedEvent`` on each appearance change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            AppearanceChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_visual_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AppearanceChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Deprecated. Use ``on_appearance_changed`` instead."""
        import warnings

        warnings.warn(
            "on_visual_changed is deprecated; use on_appearance_changed instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.on_appearance_changed(
            visual_id, callback, owner_id=owner_id, weak=weak
        )

    def on_visibility_changed(
        self,
        visual_id: UUID,
        callback: Callable[[VisualVisibilityChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the visibility of *visual_id* changes.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``VisualVisibilityChangedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            VisualVisibilityChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def set_visual_visible(self, visual_id: UUID, visible: bool) -> None:
        """Show or hide a visual.

        Parameters
        ----------
        visual_id :
            Target visual.
        visible :
            ``True`` to show, ``False`` to hide.
        """
        visual = self.get_visual_model(visual_id)
        if isinstance(
            visual, (MultichannelImageVisual, MultichannelMultiscaleImageVisual)
        ):
            for ch in visual.channels.values():
                ch.visible = visible
        else:
            self.update_appearance_field(visual_id, "visible", visible)

    def set_visual_outline(
        self,
        visual_id: UUID,
        slot: int = 1,
        placement: str | None = None,
    ) -> None:
        """Outline a visual with a screen-space contour.

        Requires ``render_config.outline.enabled`` (or
        ``controller.render_manager.outline_enabled = True``); the outline
        pass is off by default.

        Parameters
        ----------
        visual_id :
            Target visual.
        slot :
            ``0`` removes the outline.  ``1..15`` selects palette entry
            ``slot - 1`` from ``OutlineConfig.palette`` for the selection
            layer; any nonzero slot also makes the visual visible to the
            boundaries layer.
        placement :
            ``"inward"`` puts the band inside the region's own footprint,
            so the region never appears to grow but one thinner than twice
            the thickness is consumed entirely.  ``"outward"`` puts it
            outside, reading as a halo and leaving the region intact.
            Defaults by visual type: ``"outward"`` for lines and points,
            whose screen-space defaults (2 px thickness, 5 px marker size)
            are too thin to survive an inward band at any zoom level;
            ``"inward"`` for everything else.

        Raises
        ------
        ValueError
            If *slot* or *placement* is out of range.

        Notes
        -----
        A nonzero *slot* sets ``pick_write = True`` on the visual, since
        outlines are derived from the pick buffer.  **This also enables
        mouse picking for that visual**, which is a side effect worth
        knowing about if you had deliberately turned picking off.

        Labels visuals are outlined *per label* rather than as one
        silhouette, provided the canvas was built with outlines enabled.
        **Their selection colour comes from the label, not from** *slot*:
        a nonzero *slot* makes the visual eligible for the boundaries
        layer, and :meth:`set_label_selection` is what puts a palette
        colour on individual labels.  Without a selection a labels visual
        shows boundaries only.
        """
        slot = int(slot)
        visual = self._get_visual_model(visual_id)
        if placement is None:
            placement = (
                "outward"
                if isinstance(visual, (LinesVisual, PointsVisual))
                else "inward"
            )
        if slot >= 1 and not visual.pick_write:
            # Set it on the model, not the material: the PickWriteChangedEvent
            # wiring propagates it to the pygfx material for us.
            visual.pick_write = True
        kind = (
            KIND_LABEL
            if isinstance(visual, (LabelMemoryVisual, MultiscaleLabelVisual))
            else KIND_WHOLE_OBJECT
        )
        self._render_manager.set_visual_outline(
            visual_id, slot=slot, placement=placement, kind=kind
        )
        self._request_draw_for_visual(visual_id)

    def set_visual_ambient_occlusion(
        self, visual_id: UUID, enabled: bool | None = None
    ) -> None:
        """Choose whether one visual receives ambient occlusion.

        Requires ``render_config.ssao.enabled`` (or
        ``controller.ssao_enabled = True``); the occlusion pass is off by
        default, and off in 2D always.

        Parameters
        ----------
        visual_id :
            Target visual.
        enabled :
            ``None`` (the default) restores the automatic rule: excluded
            while the visual renders in a MIP-family mode (``mip``,
            ``attenuated_mip``, ``minip``), included otherwise, re-derived
            whenever the render mode changes.  ``True`` and ``False`` are
            explicit and survive a render-mode change.

        Notes
        -----
        A MIP-family mode writes the depth of the brightest sample along
        the ray rather than of a surface, and that depth jumps between
        neighbouring pixels, so occlusion computed from it shimmers.  That
        is what the automatic rule exists for; an explicit ``True`` is
        available for anyone who wants it anyway.

        Excluding a visual sets ``pick_write = True`` on it, because the
        occlusion pass identifies pixels through the pick buffer -- the
        same side effect :meth:`set_visual_outline` has, and worth knowing
        if you had deliberately turned picking off.  **The automatic rule
        needs picking too**: a MIP visual with ``pick_write = False``
        cannot be identified per pixel, so it receives occlusion despite
        the default.

        This controls whether the visual *receives* occlusion, not whether
        it *casts* it: the occlusion loop reads raw depth, so an excluded
        visual's depth still darkens its neighbours.
        """
        visual = self._get_visual_model(visual_id)
        if enabled is False and not visual.pick_write:
            # Set it on the model, not the material: the PickWriteChangedEvent
            # wiring propagates it to the pygfx material for us.
            visual.pick_write = True
        self._render_manager.set_visual_ambient_occlusion(visual_id, enabled)
        self._request_draw_for_visual(visual_id)

    def get_visual_ambient_occlusion(self, visual_id: UUID) -> bool | None:
        """Return the explicit occlusion setting for *visual_id*.

        Parameters
        ----------
        visual_id :
            Target visual.

        Returns
        -------
        bool or None
            ``None`` when the visual is on the automatic rule, which is
            the default.
        """
        return self._render_manager.get_visual_ambient_occlusion(visual_id)

    def get_visual_outline(self, visual_id: UUID) -> tuple[int, str] | None:
        """Return ``(slot, placement)`` for *visual_id*, or ``None``.

        Parameters
        ----------
        visual_id :
            Target visual.

        Returns
        -------
        tuple[int, str] or None
            ``None`` when the visual is not outlined.
        """
        entry = self._render_manager.get_visual_outline(visual_id)
        if entry is None:
            return None
        slot, placement, _kind = entry
        return slot, "outward" if placement == PLACEMENT_OUTWARD else "inward"

    def set_label_selection(self, visual_id: UUID, selection: dict[int, int]) -> None:
        """Choose which label values the selection layer outlines.

        Parameters
        ----------
        visual_id :
            A labels visual, already given an outline with
            :meth:`set_visual_outline`.
        selection :
            ``{label value: palette slot}``.  Slots are clamped into
            ``1..15`` and index ``OutlineConfig.palette`` as ``slot - 1``.
            An empty dict clears the selection, leaving the boundaries layer
            to draw every label boundary.

        Notes
        -----
        Selection is **exact**: the label key is range-partitioned, with
        ``1..15`` reserved for selected labels and everything above for
        unselected ones, so an unselected label can never be mistaken for a
        selected one.

        Requires a canvas built with outlines enabled -- the per-label key
        lives in a render target that is only allocated then.  Without it
        the visual still gets a whole-object silhouette.
        """
        self._render_manager.set_label_selection(visual_id, selection)
        self._request_draw_for_visual(visual_id)

    def _request_draw_for_visual(self, visual_id: UUID) -> None:
        """Ask every canvas showing *visual_id*'s scene to redraw."""
        scene_id = self._visual_to_scene.get(visual_id)
        if scene_id is None:
            return
        for canvas_id in self.get_canvas_ids(scene_id):
            canvas_view = self._render_manager._canvases.get(canvas_id)
            if canvas_view is not None:
                canvas_view.request_draw()

    def on_scene_added(
        self,
        scene_id: UUID,
        callback: Callable[[SceneAddedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *scene_id* is added.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``SceneAddedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            SceneAddedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_scene_removed(
        self,
        scene_id: UUID,
        callback: Callable[[SceneRemovedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *scene_id* is removed.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``SceneRemovedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            SceneRemovedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_visual_added(
        self,
        visual_id: UUID,
        callback: Callable[[VisualAddedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *visual_id* is added.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``VisualAddedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            VisualAddedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_visual_removed(
        self,
        visual_id: UUID,
        callback: Callable[[VisualRemovedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *visual_id* is removed.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``VisualRemovedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            VisualRemovedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def cancel_pending_slices(self, scene_id: UUID) -> None:
        """Cancel all in-flight slice requests for *scene_id*.

        Parameters
        ----------
        scene_id :
            ID of the scene whose pending slices to cancel.
        """
        self._render_manager._slice_coordinator.cancel_scene(scene_id)

    def close(self) -> None:
        """Cancel in-flight slices and close every canvas this viewer owns.

        Releases the render surfaces and GPU resources held by the controller.
        Closing is explicit because the canvases are owned by the GUI backend,
        not by Python refcounting, so dropping the controller alone leaks them
        (see :meth:`CanvasView.close`).

        Safe to call more than once; the controller must not be used afterwards.
        """
        for scene_id in list(self._scene_to_canvases):
            self.cancel_pending_slices(scene_id)
        self._render_manager.close()
        self._scene_to_canvases.clear()

    def on_aabb_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AABBChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the AABB params of *visual_id* change.

        The callback receives the full ``AABBChangedEvent``, which includes
        ``source_id`` for echo-filtering, ``field_name``, and ``new_value``.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``AABBChangedEvent`` on each AABB change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            AABBChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_reslice_started(
        self,
        scene_id: UUID,
        callback: Callable[[ResliceStartedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when a reslice cycle begins for *scene_id*.

        Useful for showing a loading indicator.  Fired once per reslice
        submission, before any async data fetching starts.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``ResliceStartedEvent``.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            ResliceStartedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_reslice_completed(
        self,
        visual_id: UUID,
        callback: Callable[[ResliceCompletedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when a reslice cycle completes for *visual_id*.

        Useful for hiding a loading indicator.  Fired once per visual per
        reslice cycle, after all bricks/tiles in the batch are committed.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``ResliceCompletedEvent``.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            ResliceCompletedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_scene_ready(
        self,
        scene_id: UUID,
        callback: Callable[[], None],
        *,
        owner_id: UUID | None = None,
    ) -> None:
        """Reslice *scene_id* and fire *callback* once all its data is on the GPU.

        This triggers a reslice of every visual in the scene and invokes
        *callback* exactly once, after all visuals loaded by that reslice have
        committed to the GPU across every attached canvas.  Visuals with no
        data in the current view (frustum-culled, empty slab, or hidden) do not
        delay the callback.

        Unlike :meth:`on_reslice_completed` (which fires per-visual on every
        cycle), this is a scene-level, one-shot readiness signal — the right
        hook for fitting the camera or hiding a startup spinner once a mixed
        scene of multiscale images, in-memory images, and geometry has fully
        loaded.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to reslice and watch.
        callback : Callable[[], None]
            Zero-argument callback fired once the scene's data is resident.
        owner_id : UUID or None
            Owner under which the temporary subscriptions are registered.
            Defaults to the controller's own id.
        """
        self.reslice_scene(scene_id, on_ready=callback, owner_id=owner_id)

    def on_canvas_first_frame(
        self,
        canvas_id: UUID,
        callback: Callable[[], None],
        *,
        owner_id: UUID | None = None,
    ) -> None:
        """Fire *callback* once *canvas_id* has rendered its first frame.

        The first rendered frame guarantees the canvas has reached its final
        logical size and its camera matrix has been applied — the precondition
        for fitting the camera and computing view-dependent (multiscale) slice
        requests at the correct level of detail.  This is a timer-free
        replacement for deferring startup work with ``QTimer.singleShot(0)``.

        A draw is requested immediately so the frame is guaranteed to arrive
        whether or not the event loop is already running.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas to watch.
        callback : Callable[[], None]
            Zero-argument callback fired once, on the first frame.
        owner_id : UUID or None
            Owner under which the temporary subscription is registered.
            Defaults to the controller's own id.
        """
        owner = owner_id or self._id
        state: dict[str, Any] = {"fired": False, "handle": None}

        def _on_frame(event: FrameRenderedEvent) -> None:
            if state["fired"]:
                return
            state["fired"] = True
            if state["handle"] is not None:
                self._outgoing_events.unsubscribe(state["handle"])
            callback()

        state["handle"] = self._outgoing_events.subscribe(
            FrameRenderedEvent,
            _on_frame,
            entity_id=canvas_id,
            owner_id=owner,
        )

        # Request a draw so a FrameRenderedEvent is guaranteed to be emitted.
        canvas_view = self._render_manager._canvases.get(canvas_id)
        if canvas_view is not None:
            canvas_view.request_draw()

    def unsubscribe_owner(self, owner_id: UUID) -> None:
        """Remove all event subscriptions registered under *owner_id*.

        GUI widgets should call this from their Qt ``closeEvent`` or
        ``destroyed`` signal handler to deterministically clean up their
        bus subscriptions.

        Parameters
        ----------
        owner_id :
            The UUID used as ``owner_id`` when the subscriptions were
            registered (typically the widget's own ``self._id``).
        """
        self._outgoing_events.unsubscribe_all(owner_id)

    def connect_widget(
        self,
        widget: WidgetView,
        *,
        subscription_specs: list[SubscriptionSpec] | None = None,
    ) -> None:
        """Wire a widget's psygnal signals to the bus and register subscriptions.

        Widgets declare their intent through two psygnal signals and an optional
        list of ``SubscriptionSpec`` objects, so they never import or hold a
        reference to ``CellierController``.

        The caller is responsible for constructing the widget and passing the
        specs — typically obtained from ``widget.subscription_specs()``.

        Parameters
        ----------
        widget :
            Any object exposing:

            * ``widget._id`` — a ``UUID`` identifying the widget.
            * ``widget.changed`` — a psygnal ``Signal`` that emits a
              ``CellierUpdateEventTypes`` instance when the user changes a
              value.  Connected to ``incoming_events.emit``.
            * ``widget.closed`` — a psygnal ``Signal`` (no arguments) emitted
              when the widget is closed.  Triggers
              ``unsubscribe_owner(widget._id)``.

        subscription_specs :
            Optional list of ``SubscriptionSpec`` entries describing which
            outgoing bus events the widget wants to receive.  Pass ``None``
            (or omit the argument) for pure-output widgets that do not need
            model-driven updates.
        """
        widget.changed.connect(self._incoming_events.emit)
        owner_id = widget._id
        widget.closed.connect(lambda: self.unsubscribe_owner(owner_id))
        for spec in subscription_specs or []:
            self._outgoing_events.subscribe(
                spec.event_type,
                spec.handler,
                entity_id=spec.entity_id,
                owner_id=owner_id,
                weak=spec.weak,
            )
