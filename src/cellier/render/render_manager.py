"""RenderManager — single top-level render-layer object."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, NamedTuple
from uuid import uuid4

import numpy as np

from cellier.events._events import AABBChangedEvent, ViewRay, _CanvasRawPointerEvent
from cellier.render._config import RenderManagerConfig
from cellier.render._scene_config import VisualRenderConfig
from cellier.render._visual_lut import (
    AO_EXCLUDED_BIT,
    KIND_WHOLE_OBJECT,
    MAX_SLOT,
    PLACEMENT_INWARD,
    PLACEMENT_NAMES,
    PLACEMENT_OUTWARD,
    encode_entry,
    get_shared_visual_lut,
    peek_shared_visual_lut,
)
from cellier.render.canvas_view import CanvasView
from cellier.render.scene_manager import SceneManager
from cellier.render.scheduling import ChunkScheduler
from cellier.render.slice_coordinator import SliceCoordinator
from cellier.slicer import AsyncSlicer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    import pygfx as gfx
    from PySide6.QtWidgets import QWidget

    from cellier.data._base_data_store import BaseDataStore
    from cellier.events import EventBus
    from cellier.events._events import LoadingProgress, VisualPickDetails
    from cellier.render._requests import DimsState
    from cellier.render.visuals._canvas_overlay import GFXCanvasOverlay
    from cellier.render.visuals._image import GFXMultiscaleImageVisual
    from cellier.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
    from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
    from cellier.render.visuals._points_memory import GFXPointsMemoryVisual
    from cellier.render.visuals._scene_overlay import GFXSceneOverlay
    from cellier.scene._background import BackgroundAppearance
    from cellier.transform import RegionSelection

    _GFXVisual = (
        GFXMultiscaleImageVisual
        | GFXImageMemoryVisual
        | GFXPointsMemoryVisual
        | GFXLinesMemoryVisual
        | GFXMeshMemoryVisual
    )


# ---------------------------------------------------------------------------
# Internal render-layer intermediates (not exported)
# ---------------------------------------------------------------------------


class _ImageDisplayedDataCoord(NamedTuple):
    """Render-layer intermediate for image/volume pick — displayed axes only.

    ``_extract_pick_details`` can only decode the rendered axes from the pick
    payload; the visual supplies the rest from what it last drew, and the
    controller joins the two into a full-rank ``ImagePickInfo`` in
    ``_on_raw_pointer_event``.

    Parameters
    ----------
    displayed_data_coord : tuple[float, ...]
        Level-0 data-array position on the displayed axes only (``floor`` gives
        the index).  Length 2 for a 2-D canvas, 3 for a 3-D canvas.
    collapsed_data_indices : tuple[tuple[int, int], ...] or None
        ``(data axis, level-0 voxel index)`` for every axis the visual
        collapsed, taken from the plan it last drew.  ``None`` when the visual
        cannot say -- it has never been resliced, or it is a headlessly
        constructed one with no region.

        Pairs rather than a mapping so the tuple stays comparable and
        hashable like every other pick payload.  These are the planes actually
        on screen, which is not the same as the planes the current dims state
        implies: a slider moved since the last reslice changes the second and
        not the first, and a pick is a question about the first.
    channel_index : int or None
        The channel of the slot whose node was hit, or ``None`` when the
        visual has no channel slots or the node is not one of them.
    drawn_channels : tuple[int, ...]
        The channel indices the visual's last plan draws, ascending.
    """

    displayed_data_coord: tuple[float, ...]
    collapsed_data_indices: tuple[tuple[int, int], ...] | None = None
    channel_index: int | None = None
    drawn_channels: tuple[int, ...] = ()


class _LabelsDisplayedDataCoord(NamedTuple):
    """Render-layer intermediate for labels pick — displayed axes only.

    Same semantics as ``_ImageDisplayedDataCoord``; separate type so the
    controller can produce the correct public ``LabelsPickInfo``.
    """

    displayed_data_coord: tuple[float, ...]
    collapsed_data_indices: tuple[tuple[int, int], ...] | None = None


#: Render modes that project along the ray instead of finding a surface.
#: Their depth is the depth of the extremum sample, so ambient occlusion
#: derived from it is noise; visuals rendering this way are excluded from
#: receiving occlusion unless the caller says otherwise.
MIP_RENDER_MODES: frozenset[str] = frozenset({"mip", "attenuated_mip", "minip"})


class VisualFlags(NamedTuple):
    """Everything the shared per-visual LUT carries for one cellier visual.

    One record rather than one per feature, because ``VisualLut.apply`` is
    a whole-state sync: two independent maps syncing the same table would
    each clear the other's entries.

    Parameters
    ----------
    outline : tuple[int, int, int] or None
        ``(slot, placement, kind)`` for the screen-space outline pass, or
        ``None`` when the visual is not outlined.
    ao : bool or None
        Whether the visual receives ambient occlusion.  ``None`` is auto:
        excluded when it renders in the MIP family, which writes the depth
        of an extremum sample rather than of a surface.  ``True`` and
        ``False`` are explicit and survive a render-mode change.
    """

    outline: tuple[int, int, int] | None = None
    ao: bool | None = None


# Union used to annotate _CanvasRawPointerEvent.pick_details
_RawPickDetails = (
    "_ImageDisplayedDataCoord | _LabelsDisplayedDataCoord | VisualPickDetails | None"
)


def _slice_check_extents(data_store: Any) -> Any:
    """The extents the out-of-domain slice check uses for *data_store*.

    Gridded stores only: a geometry store's vertices are filtered by the
    slab itself, so it reports ``None`` ("not known") and is never skipped.
    """
    return data_store.axis_extents if hasattr(data_store, "level_shapes") else None


class RenderManager:
    """Single top-level render-layer object.

    Owns the scene registry, canvas registry, shared async slicer, and
    slice coordinator.  Exposes three reslicing entry points that cover
    the common triggers: all scenes, one scene, or one visual.

    Construction is parameter-free; scenes, canvases, and visuals are
    registered via the ``add_*`` methods.
    """

    def __init__(self, config: RenderManagerConfig | None = None) -> None:
        if config is None:
            config = RenderManagerConfig()
        self._config = config
        self._id: UUID = uuid4()
        self._scenes: dict[UUID, SceneManager] = {}
        self._canvases: dict[UUID, CanvasView] = {}
        self._canvas_to_scene: dict[UUID, UUID] = {}
        self._visual_to_scene: dict[UUID, UUID] = {}
        self._data_stores: dict[UUID, BaseDataStore] = {}
        self._event_bus: EventBus | None = None
        self._active_gestures: dict[UUID, UUID] = {}
        self._pick_details_enabled: dict[UUID, bool] = {}
        # The single authoritative per-visual flag map.  Both the outline
        # pass and the ambient occlusion pass read the same GPU table, which
        # is derived from this every frame; see ``_sync_visual_lut``.
        self._visual_flags: dict[UUID, VisualFlags] = {}
        self._slicer = AsyncSlicer(
            batch_size=config.slicing.batch_size,
            render_every=config.slicing.render_every,
        )
        # Multiscale visuals load through the chunk scheduler, 2D and 3D
        # (plans/progressive_loading_design_v3.md); everything else through
        # the slicer.
        self._scheduler = ChunkScheduler(
            config.scheduler,
            request_draw=self._request_draw_scene,
            on_complete=self._on_cache_complete,
            on_backstop_complete=self._on_cache_backstop_complete,
            on_progress=self._on_cache_progress,
        )
        self._slice_coordinator = SliceCoordinator(
            scenes=self._scenes,
            slicer=self._slicer,
            data_stores=self._data_stores,
            scheduler=self._scheduler,
        )
        # canvas id -> its ``before_draw`` handler, removed with the canvas.
        self._commit_hooks: dict[UUID, Any] = {}

    def connect_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe internal components to *event_bus*."""
        self._event_bus = event_bus
        # Let the coordinator emit ResliceStartedEvent / ResliceCompletedEvent.
        self._slice_coordinator._event_bus = event_bus
        # The ambient-occlusion radius is derived from the scene bounding
        # box, so it has to be recomputed when that box moves.  Walking the
        # scene graph per frame is not an option -- a multiscale visual is a
        # gfx.Group with a child per brick -- so it happens here and on a
        # camera fit, the only two moments the answer can change.
        event_bus.subscribe(
            AABBChangedEvent,
            self._on_aabb_changed_for_ssao,
            owner_id=self._id,
        )

    @property
    def scheduler(self) -> ChunkScheduler:
        """The chunk scheduler shared by every multiscale visual."""
        return self._scheduler

    def _on_cache_complete(self, cache_id: int, generation: int) -> None:
        self._slice_coordinator.on_cache_complete(cache_id, generation)

    def _on_cache_backstop_complete(self, cache_id: int, generation: int) -> None:
        self._slice_coordinator.on_cache_backstop_complete(cache_id, generation)

    def _on_cache_progress(self, cache_id: int) -> None:
        self._slice_coordinator.on_cache_progress(cache_id)

    def loading_progress(self, visual_id: UUID) -> LoadingProgress | None:
        """A multiscale visual's loading progress, or ``None`` (design 5.13)."""
        return self._slice_coordinator.visual_progress(visual_id)

    def _request_draw_scene(self, scene_id: UUID | None) -> None:
        """Scheduler callback: data for *scene_id* changed; draw its canvases."""
        for canvas_id, canvas_scene_id in list(self._canvas_to_scene.items()):
            if canvas_scene_id == scene_id:
                canvas_view = self._canvases.get(canvas_id)
                if canvas_view is not None:
                    canvas_view.request_draw()

    def invalidate_store(
        self, store_id: UUID, regions: tuple | None = None
    ) -> list[int]:
        """Forget GPU data read from a store, where it changed (design 5.14).

        Every scheduled atlas reading the store drops the bricks that overlap
        *regions* (all of them for ``None``); wanted ones are fetched again.

        Parameters
        ----------
        store_id : UUID
            The store that changed.
        regions : tuple of DataRegion or None
            Level-0 data regions, as ``StoreChange.regions`` carries them.

        Returns
        -------
        list[int]
            The atlases touched.
        """
        return self._scheduler.invalidate(store_id, regions)

    @property
    def config(self) -> RenderManagerConfig:
        """Current rendering performance configuration.

        Reflects live state: mutations via ``temporal_blend_weight`` and
        ``temporal_enabled`` setters are visible here immediately.
        """
        return self._config

    @property
    def temporal_blend_weight(self) -> float:
        """EMA floor weight for temporal accumulation."""
        return self._config.temporal.blend_weight

    @temporal_blend_weight.setter
    def temporal_blend_weight(self, value: float) -> None:
        self._config.temporal.blend_weight = value
        for canvas in self._canvases.values():
            canvas._accum_pass.blend_weight = value

    @property
    def temporal_enabled(self) -> bool:
        """Whether the temporal accumulation pass is active."""
        return self._config.temporal.enabled

    @temporal_enabled.setter
    def temporal_enabled(self, value: bool) -> None:
        self._config.temporal.enabled = value
        for canvas in self._canvases.values():
            canvas._accum_pass.enabled = value
            # A disabled pass is skipped outright, so its history and frame
            # count freeze at whatever they held.  Switching back on without
            # this would blend ``alpha`` of that stale image into the first
            # frame, with no warm-up, however long the pass was off.
            canvas.invalidate_accumulation()

    @property
    def temporal_frame_count(self) -> int | None:
        """Frames accumulated on the least-converged canvas.

        ``None`` when there is no canvas.  The minimum rather than any one
        canvas's count, so a GUI reading it reports "settled" only once
        every canvas is.
        """
        counts = [c._accum_pass.frame_count for c in self._canvases.values()]
        return min(counts) if counts else None

    def reset_temporal_accumulation(self) -> None:
        """Discard the accumulated history on every canvas."""
        for canvas in self._canvases.values():
            canvas.invalidate_accumulation()

    # ------------------------------------------------------------------
    # Screen-space ambient occlusion
    # ------------------------------------------------------------------

    @property
    def ambient_occlusion_enabled(self) -> bool:
        """Whether the ambient occlusion pass is active.

        A canvas in 2D never runs the pass whatever this says; see
        ``CanvasView.set_ssao_enabled``.
        """
        return self._config.ambient_occlusion.enabled

    @ambient_occlusion_enabled.setter
    def ambient_occlusion_enabled(self, value: bool) -> None:
        value = bool(value)
        self._config.ambient_occlusion.enabled = value
        for canvas in self._canvases.values():
            if value:
                # Only on the way on, and only where it is missing: a canvas
                # built without the ``normal`` target would otherwise run the
                # pass against normals reconstructed from depth, which is
                # 34 degrees out on a raymarched isosurface.
                canvas.ensure_render_targets(ssao=True)
            canvas.set_ssao_enabled(value)

    @property
    def ambient_occlusion_radius(self) -> float | None:
        """Hemisphere radius in scene units, or ``None`` for auto."""
        return self._config.ambient_occlusion.radius

    @ambient_occlusion_radius.setter
    def ambient_occlusion_radius(self, value: float | None) -> None:
        value = None if value is None else float(value)
        self._config.ambient_occlusion.radius = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.radius = value

    @property
    def ambient_occlusion_strength(self) -> float:
        """How far the occlusion multiply is applied, 0 (off) to 1 (full)."""
        return self._config.ambient_occlusion.strength

    @ambient_occlusion_strength.setter
    def ambient_occlusion_strength(self, value: float) -> None:
        value = float(value)
        self._config.ambient_occlusion.strength = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.strength = value

    @property
    def ambient_occlusion_power(self) -> float:
        """Contrast exponent applied to the occlusion before the multiply."""
        return self._config.ambient_occlusion.power

    @ambient_occlusion_power.setter
    def ambient_occlusion_power(self, value: float) -> None:
        value = float(value)
        self._config.ambient_occlusion.power = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.power = value

    @property
    def ambient_occlusion_bias(self) -> float:
        """Depth-comparison bias, as a fraction of the effective radius."""
        return self._config.ambient_occlusion.bias

    @ambient_occlusion_bias.setter
    def ambient_occlusion_bias(self, value: float) -> None:
        value = float(value)
        self._config.ambient_occlusion.bias = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.bias = value

    @property
    def ambient_occlusion_n_samples(self) -> int:
        """Hemisphere samples per pixel.  Changing this recompiles."""
        return self._config.ambient_occlusion.n_samples

    @ambient_occlusion_n_samples.setter
    def ambient_occlusion_n_samples(self, value: int) -> None:
        value = int(value)
        self._config.ambient_occlusion.n_samples = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.n_samples = value

    @property
    def ambient_occlusion_blur_radius(self) -> int:
        """Box-blur half-width in internal pixels.  Changing this recompiles."""
        return self._config.ambient_occlusion.blur_radius

    @ambient_occlusion_blur_radius.setter
    def ambient_occlusion_blur_radius(self, value: int) -> None:
        value = int(value)
        self._config.ambient_occlusion.blur_radius = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.blur_radius = value

    @property
    def ambient_occlusion_auto_radius_fraction(self) -> float:
        """Fraction of the scene bounding box diagonal used when radius is auto."""
        return self._config.ambient_occlusion.auto_radius_fraction

    @ambient_occlusion_auto_radius_fraction.setter
    def ambient_occlusion_auto_radius_fraction(self, value: float) -> None:
        value = float(value)
        self._config.ambient_occlusion.auto_radius_fraction = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.auto_radius_fraction = value

    @property
    def ambient_occlusion_effective_radius(self) -> float | None:
        """The occlusion radius actually in use, in scene units.

        The explicit ``config.ambient_occlusion.radius`` when one is set, otherwise the
        value derived from the scene bounding box diagonal.  ``None`` when
        there is no canvas to ask.  Read-only.
        """
        for canvas in self._canvases.values():
            return canvas._ssao_pass.effective_radius
        return None

    def apply_ambient_occlusion_config(self) -> None:
        """Push the current ``config.ambient_occlusion`` onto every canvas's pass."""
        for canvas in self._canvases.values():
            canvas.apply_ambient_occlusion_config(self._config.ambient_occlusion)

    def update_ssao_radius(self, scene_id: UUID) -> None:
        """Recompute the auto occlusion radius for one scene.

        Reads the scene's world bounding box once and hands its diagonal to
        every canvas showing that scene.  An explicit ``ssao.radius``
        overrides the result, so this is safe to call unconditionally.

        Parameters
        ----------
        scene_id : UUID
            The scene whose bounding box should be measured.  Unknown ids
            are ignored.
        """
        scene_manager = self._scenes.get(scene_id)
        if scene_manager is None:
            return
        canvases = [
            canvas
            for canvas_id, canvas in self._canvases.items()
            if self._canvas_to_scene.get(canvas_id) == scene_id
        ]
        if not canvases:
            return
        try:
            box = scene_manager.scene.get_world_bounding_box()
        except (AttributeError, IndexError, TypeError, ValueError):
            # An empty or half-built scene graph is not an error here:
            # the previous radius stays in place until the next fit.
            return
        if box is None:
            return
        box = np.asarray(box, dtype=np.float64)
        diagonal = float(np.linalg.norm(box[1] - box[0]))
        for canvas in canvases:
            canvas.set_scene_extent(diagonal)

    def _on_aabb_changed_for_ssao(self, event: AABBChangedEvent) -> None:
        """Re-derive the auto occlusion radius when a visual's box moves."""
        scene_id = self._visual_to_scene.get(event.visual_id)
        if scene_id is not None:
            self.update_ssao_radius(scene_id)

    # ------------------------------------------------------------------
    # Screen-space outlines
    # ------------------------------------------------------------------

    @property
    def outline_enabled(self) -> bool:
        """Whether the screen-space outline pass is active."""
        return self._config.outline.enabled

    @outline_enabled.setter
    def outline_enabled(self, value: bool) -> None:
        value = bool(value)
        self._config.outline.enabled = value
        if value:
            self._warn_if_outline_unavailable()
        for canvas in self._canvases.values():
            if value:
                # Only on the way on, and only where it is missing.  Without
                # the ``outline_id`` target a labels visual is outlined as one
                # silhouette rather than per label.
                canvas.ensure_render_targets(outline=True)
            canvas._outline_pass.enabled = value

    @property
    def outline_boundaries_enabled(self) -> bool:
        """Whether the boundaries layer (every outlined region) draws."""
        return self._config.outline.boundaries.enabled

    @outline_boundaries_enabled.setter
    def outline_boundaries_enabled(self, value: bool) -> None:
        value = bool(value)
        self._config.outline.boundaries.enabled = value
        for canvas in self._canvases.values():
            canvas._outline_pass._quad_pass.set_uniform(
                "boundaries_enabled", int(value)
            )

    @property
    def outline_selection_enabled(self) -> bool:
        """Whether the selection layer (regions with a palette slot) draws."""
        return self._config.outline.selection.enabled

    @outline_selection_enabled.setter
    def outline_selection_enabled(self, value: bool) -> None:
        value = bool(value)
        self._config.outline.selection.enabled = value
        for canvas in self._canvases.values():
            canvas._outline_pass._quad_pass.set_uniform("selection_enabled", int(value))

    def apply_outline_config(self) -> None:
        """Push the current ``config.outline`` onto every canvas's pass."""
        for canvas in self._canvases.values():
            canvas._outline_pass.apply_config(self._config.outline)

    def set_visual_outline(
        self,
        visual_id: UUID,
        slot: int = 1,
        placement: str | int = PLACEMENT_INWARD,
        kind: int = KIND_WHOLE_OBJECT,
    ) -> None:
        """Record the outline assignment for one visual.

        The GPU table is not written here: it is derived from these
        assignments once per frame, because cellier rebuilds world objects
        (2D/3D switches, multiscale brick groups, channel changes) and each
        rebuild hands out fresh ``global_id``s.

        Parameters
        ----------
        visual_id : UUID
            The cellier visual to outline.
        slot : int
            0 removes the outline.  ``1..15`` selects palette entry
            ``slot - 1`` for the selection layer; any nonzero slot also
            makes the visual visible to the boundaries layer.
        placement : str or int
            ``"inward"`` or ``"outward"`` (or the corresponding constant).
        kind : int
            ``KIND_WHOLE_OBJECT`` keys the edge test on the pygfx object id,
            giving one silhouette per visual.  ``KIND_LABEL`` keys it on the
            per-pixel label field instead, so boundaries appear *between*
            labels inside one volume, coloured per label.  ``KIND_LABEL_ALL``
            keys on the label too but colours every one of them from *slot*.
            Both label kinds require the ``outline_id`` target; without it a
            label visual falls back to a whole-object silhouette.

        Raises
        ------
        ValueError
            If *slot* or *placement* is out of range.
        """
        slot = int(slot)
        if not 0 <= slot <= MAX_SLOT:
            raise ValueError(f"slot must be in [0, {MAX_SLOT}], got {slot}")
        if isinstance(placement, str):
            try:
                placement = PLACEMENT_NAMES[placement]
            except KeyError:
                raise ValueError(
                    f"placement must be one of {sorted(PLACEMENT_NAMES)}, "
                    f"got {placement!r}"
                ) from None
        if placement not in (PLACEMENT_INWARD, PLACEMENT_OUTWARD):
            raise ValueError(f"unknown outline placement: {placement}")

        outline = None if slot == 0 else (slot, int(placement), int(kind))
        self._update_visual_flags(visual_id, outline=outline)
        self._sync_visual_lut()

    def get_visual_outline(self, visual_id: UUID) -> tuple[int, int, int] | None:
        """Return ``(slot, placement, kind)`` for *visual_id*, or ``None``."""
        flags = self._visual_flags.get(visual_id)
        return None if flags is None else flags.outline

    def set_visual_ambient_occlusion(
        self, visual_id: UUID, enabled: bool | None = None
    ) -> None:
        """Choose whether one visual receives ambient occlusion.

        Parameters
        ----------
        visual_id : UUID
            The cellier visual.
        enabled : bool or None
            ``None`` (the default) restores the automatic rule: excluded
            while the visual renders in the MIP family, included
            otherwise, re-derived whenever the render mode changes.
            ``True`` and ``False`` are explicit and survive a render-mode
            change.

        Notes
        -----
        This controls whether the visual *receives* occlusion, not whether
        it *casts* it: the occlusion loop reads raw depth, so an excluded
        visual's depth still occludes its neighbours.
        """
        if enabled is not None:
            enabled = bool(enabled)
        self._update_visual_flags(visual_id, ao=enabled)
        self._sync_visual_lut()

    def get_visual_ambient_occlusion(self, visual_id: UUID) -> bool | None:
        """Return the explicit occlusion setting, or ``None`` for auto."""
        flags = self._visual_flags.get(visual_id)
        return None if flags is None else flags.ao

    def _update_visual_flags(self, visual_id: UUID, **changes) -> None:
        """Merge *changes* into one visual's flags, dropping empty records."""
        flags = self._visual_flags.get(visual_id, VisualFlags())
        flags = flags._replace(**changes)
        if flags == VisualFlags():
            self._visual_flags.pop(visual_id, None)
        else:
            self._visual_flags[visual_id] = flags

    def set_label_selection(self, visual_id: UUID, selection: dict[int, int]) -> None:
        """Set which label values the selection layer outlines.

        Parameters
        ----------
        visual_id : UUID
            A labels visual.
        selection : dict[int, int]
            ``{label value: palette slot}``, slots in ``1..15``.  An empty
            dict clears the selection.

        Notes
        -----
        Writes into the material's fixed-capacity selection texture rather
        than replacing it, so a selection change is a data upload and never
        a pipeline rebuild.  Materials that carry no such texture (every
        non-label visual) are skipped.
        """
        from cellier.render.shaders._label_colormap import (
            update_outline_selection,
        )

        scene_id = self._visual_to_scene.get(visual_id)
        if scene_id is None:
            return
        scene = self._scenes.get(scene_id)
        if scene is None:
            return
        try:
            gfx_visual = scene.get_visual(visual_id)
        except KeyError:
            return

        # A visual that can rebuild its materials records the selection and
        # re-applies it for itself; one whose materials are built once takes
        # the direct write.  Duck-typed rather than keyed on class, the way
        # ``_label_materials`` is.
        recorder = getattr(gfx_visual, "set_outline_selection", None)
        if recorder is not None:
            recorder(selection)
            return

        for material in self._label_materials(gfx_visual):
            count = update_outline_selection(
                material.outline_selection_texture, selection
            )
            buffer = material.label_params_buffer
            buffer.data["n_outline_entries"] = np.uint32(count)
            buffer.update_full()

    @staticmethod
    def _label_materials(gfx_visual: _GFXVisual):
        """Yield each material on *gfx_visual* that carries a label key.

        Deliberately duck-typed rather than keyed on visual class: the 2D
        and 3D nodes of one labels visual hold different material types, and
        multiscale visuals nest theirs inside a ``gfx.Group``.
        """
        seen: set[int] = set()
        for mode in ("2d", "3d"):
            try:
                node = gfx_visual.get_node(mode)
            except (AttributeError, KeyError, ValueError):
                node = None
            if node is None:
                continue
            for obj in node.iter():
                material = getattr(obj, "material", None)
                if material is None or id(material) in seen:
                    continue
                if getattr(material, "outline_selection_texture", None) is None:
                    continue
                if getattr(material, "label_params_buffer", None) is None:
                    continue
                seen.add(id(material))
                yield material

    def _warn_if_outline_unavailable(self) -> None:
        """Warn once if no canvas could grant the pick texture binding."""
        canvases = list(self._canvases.values())
        if canvases and not any(c._outline_available for c in canvases):
            warnings.warn(
                "screen-space outlines are unavailable: the pick texture could "
                "not be granted TEXTURE_BINDING on any canvas. This usually "
                "means the pinned pygfx no longer exposes the blender internals "
                "cellier.render._pick_buffer relies on.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _sync_visual_lut(self) -> None:
        """Rebuild the shared GPU table from ``_visual_flags``.

        Runs once per canvas draw, because cellier rebuilds world objects
        (2D/3D switches, multiscale brick groups, channel changes) and each
        rebuild hands out fresh ``global_id``s, so a write-once table would
        silently lose its entries.  Only texels that actually differ are
        uploaded, so a steady frame does no GPU work.

        **One sync for both features.**  ``VisualLut.apply`` is a
        whole-state sync: ids present in the table but absent from the
        mapping are cleared.  If outlines and ambient occlusion kept
        separate maps and synced independently, enabling outlines would
        wipe every occlusion exclusion.  That is why this walks one
        authoritative map and writes one byte per object.
        """
        ssao_on = self._config.ambient_occlusion.enabled
        if not self._visual_flags and not ssao_on:
            if peek_shared_visual_lut() is None:
                # Neither feature has ever needed the table on this
                # process, so there is nothing to sync and no reason to
                # allocate 1 MB.
                return

        entries: dict[int, int] = {}
        has_inward = False
        has_outward = False
        n_excluded = 0

        for visual_id, scene_id in self._visual_to_scene.items():
            flags = self._visual_flags.get(visual_id)
            if flags is None and not ssao_on:
                continue
            scene = self._scenes.get(scene_id)
            if scene is None:
                continue
            try:
                gfx_visual = scene.get_visual(visual_id)
            except KeyError:
                continue

            if flags is not None and flags.outline is not None:
                slot, placement, kind = flags.outline
                value = encode_entry(slot, kind, placement)
                for object_id in self._world_object_ids(gfx_visual):
                    entries[object_id] = entries.get(object_id, 0) | value
                if placement == PLACEMENT_OUTWARD:
                    has_outward = True
                else:
                    has_inward = True

            if not ssao_on:
                continue
            ao = flags.ao if flags is not None else None
            if ao is True:
                # Explicitly opted in: never excluded, whatever it renders.
                continue
            if ao is False:
                excluded_ids = self._world_object_ids(gfx_visual)
            else:
                # Auto: the MIP family only.  Derived per world object
                # rather than per visual, so one channel of a multichannel
                # visual can be excluded while another is not.
                excluded_ids = self._mip_object_ids(gfx_visual)
            for object_id in excluded_ids:
                entries[object_id] = entries.get(object_id, 0) | AO_EXCLUDED_BIT
                n_excluded += 1

        get_shared_visual_lut().apply(entries)
        for canvas in self._canvases.values():
            canvas._outline_pass.set_placements(
                has_inward=has_inward, has_outward=has_outward
            )
            canvas._ssao_pass.set_has_exclusions(n_excluded > 0)

    @staticmethod
    def _mip_object_ids(gfx_visual: _GFXVisual) -> set[int]:
        """Return the ids of world objects rendering in the MIP family.

        A MIP-family mode writes the depth of the *extremum sample along
        the ray*, not of a surface.  That depth jumps discontinuously
        wherever the brightest sample moves, which for a noisy volume is
        most neighbouring pixels, so occlusion derived from it shimmers
        under camera motion.  Those objects are excluded by default.

        Keyed on the material's ``render_mode`` rather than on the visual
        class: every volume material that has a projection mode carries
        one, label materials carry a categorical mode that is not in the
        family, and mesh, line and point materials have no such attribute
        at all -- so the same rule covers every visual type without a
        lookup table of classes.
        """
        ids: set[int] = set()
        for mode in ("2d", "3d"):
            try:
                node = gfx_visual.get_node(mode)
            except (AttributeError, KeyError, ValueError):
                node = None
            if node is None:
                continue
            for obj in node.iter():
                material = getattr(obj, "material", None)
                render_mode = getattr(material, "render_mode", None)
                if render_mode in MIP_RENDER_MODES:
                    ids.add(int(obj.id))
        ids.discard(0)
        return ids

    @staticmethod
    def _world_object_ids(gfx_visual: _GFXVisual) -> set[int]:
        """Return every pygfx id the pick buffer can carry for *gfx_visual*.

        One cellier visual owns several world objects: the 2D and 3D node
        pair, ``gfx.Group`` children for multiscale bricks, per-channel
        objects for multichannel visuals.  All of them get the same entry.

        Instanced objects are the exception: ``mesh.wgsl`` writes
        ``instance_info.global_id``, not the world object's own id, so the
        instance ids are collected instead.  Writing the world-object id
        for an instanced mesh would leave it with no outline at all.
        """
        ids: set[int] = set()
        for mode in ("2d", "3d"):
            try:
                node = gfx_visual.get_node(mode)
            except (AttributeError, KeyError, ValueError):
                node = None
            if node is None:
                continue
            for obj in node.iter():
                instance_buffer = getattr(obj, "instance_buffer", None)
                if instance_buffer is not None:
                    ids.update(
                        int(v) for v in instance_buffer.data["global_id"].ravel()
                    )
                else:
                    ids.add(int(obj.id))
        ids.discard(0)
        return ids

    def add_scene(
        self,
        scene_id: UUID,
        lighting: str = "none",
        background: BackgroundAppearance | None = None,
    ) -> SceneManager:
        """Create and register a new scene.

        Parameters
        ----------
        scene_id : UUID
            Unique identifier for the scene.
        lighting : str
            ``"none"`` (default) or ``"default"``.  Pass ``"default"`` to
            add ambient and directional lights — required for
            ``MeshPhongAppearance``.
        background : BackgroundAppearance or None
            Background appearance for the scene.  ``None`` uses the model
            defaults.

        Returns
        -------
        SceneManager
            The newly created scene manager.
        """
        scene_manager = SceneManager(
            scene_id=scene_id, lighting=lighting, background=background
        )
        self._scenes[scene_id] = scene_manager
        return scene_manager

    def scene_has_lighting(self, scene_id: UUID) -> bool:
        """Return True if *scene_id* was created with lighting enabled."""
        return self._scenes[scene_id].has_lighting

    def set_scene_background(
        self, scene_id: UUID, background: BackgroundAppearance
    ) -> None:
        """Apply *background* to the scene registered under *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to update.
        background : BackgroundAppearance
            The background appearance to apply.

        Raises
        ------
        KeyError
            If ``scene_id`` is not registered.
        """
        self._scenes[scene_id].set_background(background)

    def add_canvas(
        self,
        canvas_id: UUID,
        scene_id: UUID,
        parent: QWidget | None = None,
        **canvas_view_kwargs,
    ) -> CanvasView:
        """Create a ``CanvasView``, register it, and return it.

        The caller embeds ``canvas_view.widget`` in their Qt layout.

        Parameters
        ----------
        canvas_id : UUID
            Unique identifier for this canvas.
        scene_id : UUID
            ID of the scene this canvas should render.
        parent : QWidget or None
            Parent widget for the underlying ``QRenderWidget``.
        **canvas_view_kwargs
            Additional keyword arguments forwarded to ``CanvasView.__init__``
            (e.g. ``dim``, ``fov``, ``depth_range``, ``gui``, ``size``).

        Returns
        -------
        CanvasView
            The newly created canvas view.
        """
        canvas_view = CanvasView(
            canvas_id=canvas_id,
            scene_id=scene_id,
            get_scene_fn=self.get_scene,
            parent=parent,
            outline_enabled=self._config.outline.enabled,
            ambient_occlusion_enabled=self._config.ambient_occlusion.enabled,
            **canvas_view_kwargs,
        )
        # Apply temporal config to the canvas's accumulation pass.
        canvas_view._accum_pass.blend_weight = self._config.temporal.blend_weight
        if not self._config.temporal.enabled:
            canvas_view._accum_pass.enabled = False
        # Apply the ambient occlusion config.  The pass stays off in 2D
        # whatever the config says.
        canvas_view.apply_ambient_occlusion_config(self._config.ambient_occlusion)
        # Apply outline config, and wire the per-frame LUT re-sync.
        canvas_view._outline_pass.apply_config(self._config.outline)
        canvas_view._visual_lut_sync_fn = self._sync_visual_lut
        if self._config.outline.enabled:
            self._warn_if_outline_unavailable()
        # Wire up per-frame tick for visuals (e.g. jitter seed advance).
        canvas_view._tick_visuals_fn = self._make_tick_fn(scene_id)
        # Commit what the chunk scheduler has received just before each frame
        # of this canvas is drawn, scoped to its scene (design 5.7).  Every
        # canvas of the scene is hooked, including capture canvases.
        scheduler = self._scheduler

        def _commit_before_draw(event, scene_id=scene_id) -> None:
            scheduler.commit_round(scene_id)

        canvas_view.widget.add_event_handler(_commit_before_draw, "before_draw")
        self._commit_hooks[canvas_id] = _commit_before_draw
        self._canvases[canvas_id] = canvas_view
        self._canvas_to_scene[canvas_id] = scene_id
        canvas_view._renderer.add_event_handler(
            lambda ev, cid=canvas_id: self._on_canvas_pointer_event(ev, cid),
            "pointer_down",
            "pointer_up",
            "pointer_move",
        )
        return canvas_view

    def add_visual(
        self,
        scene_id: UUID,
        visual: _GFXVisual,
        data_store: BaseDataStore,
        displayed_axes: tuple[int, ...],
    ) -> None:
        """Register a visual with a scene and its associated data store.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to add the visual to.
        visual : _GFXVisual
            The render-layer visual object.
        data_store : BaseDataStore
            The data store that will serve chunk data for this visual.
        displayed_axes : tuple[int, ...]
            Current displayed axes from the scene's dims selection.  Passed to
            ``SceneManager.add_visual`` to select the initial node.
        """
        # Extents are passed only for **gridded** stores, and that scoping is
        # deliberate.  The out-of-domain check exists to replace *clamping*,
        # and clamping only happens where a world position becomes a scalar
        # index into a grid (``round_world_to_voxel``).  A geometry store
        # selects by proximity and already reports ``is_empty`` when nothing
        # is near, so it never pins a stale plane and needs no short-circuit;
        # skipping it early would only suppress the per-visual bookkeeping its
        # planner does on the way.
        self._scenes[scene_id].add_visual(
            visual,
            displayed_axes,
            axis_extents=_slice_check_extents(data_store),
        )
        self._visual_to_scene[visual.visual_model_id] = scene_id
        self._data_stores[visual.visual_model_id] = data_store

    def refresh_visual_axis_extents(self, visual_id: UUID) -> None:
        """Re-read a visual's store extent after the store's extent changed.

        The scene manager keeps each visual's extent for the out-of-domain
        check in slice planning, copied when the visual was added; an
        ``"extent"`` store change makes that copy stale.  An unknown visual
        is ignored.
        """
        scene_id = self._visual_to_scene.get(visual_id)
        data_store = self._data_stores.get(visual_id)
        if scene_id is None or data_store is None:
            return
        self._scenes[scene_id].set_axis_extents(
            visual_id, _slice_check_extents(data_store)
        )

    def add_canvas_overlay(
        self,
        canvas_id: UUID,
        gfx_overlay: GFXCanvasOverlay,
    ) -> None:
        """Attach a pre-built GFX overlay to *canvas_id*.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas that should receive the overlay.
        gfx_overlay : GFXCanvasOverlay
            The fully-constructed render-layer overlay.

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        self._canvases[canvas_id].add_overlay(gfx_overlay)

    def remove_canvas_overlay(
        self,
        canvas_id: UUID,
        gfx_overlay: GFXCanvasOverlay,
    ) -> None:
        """Detach a GFX overlay from *canvas_id*.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas the overlay is attached to.  An unknown canvas is
            ignored -- it has already been torn down.
        gfx_overlay : GFXCanvasOverlay
            The render-layer overlay to detach.
        """
        canvas = self._canvases.get(canvas_id)
        if canvas is not None:
            canvas.remove_overlay(gfx_overlay)

    def add_scene_overlay(
        self,
        scene_id: UUID,
        overlay_id: UUID,
        gfx_overlay: GFXSceneOverlay,
    ) -> None:
        """Attach a pre-built GFX scene overlay to *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene that should receive the overlay.
        overlay_id : UUID
            ID of the overlay's model.
        gfx_overlay : GFXSceneOverlay
            The fully-constructed render-layer overlay.

        Raises
        ------
        KeyError
            If *scene_id* is not registered.
        """
        self._scenes[scene_id].add_overlay(overlay_id, gfx_overlay)

    def remove_scene_overlay(self, scene_id: UUID, overlay_id: UUID) -> None:
        """Detach a scene overlay from *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene.  An unknown scene is ignored -- it has already
            been torn down.
        overlay_id : UUID
            ID of the overlay's model.
        """
        scene_manager = self._scenes.get(scene_id)
        if scene_manager is not None:
            scene_manager.remove_overlay(overlay_id)

    def _on_canvas_pointer_event(
        self, event: gfx.PointerEvent, canvas_id: UUID
    ) -> None:
        """Translate a pygfx pointer event to a model-layer internal event.

        Performs three render-layer operations:
          1. NDC computation from screen pixel coordinates.
          2. Camera unprojection of NDC to 2D world position.
          3. gfx.WorldObject -> visual_id lookup via SceneManager.

        The resulting _CanvasRawPointerEvent contains no gfx.* types.
        """
        if self._event_bus is None:
            return

        scene_id = self._canvas_to_scene[canvas_id]
        canvas_view = self._canvases[canvas_id]

        w, h = canvas_view._canvas.get_logical_size()
        ndc_x = event.x / w * 2.0 - 1.0
        ndc_y = -(event.y / h * 2.0 - 1.0)

        cam = canvas_view._camera
        hit_object = event.pick_info.get("world_object")
        hit_visual_id: UUID | None = None
        pick_details: VisualPickDetails | None = None
        if hit_object is not None:
            hit_visual_id = self._scenes[scene_id].get_visual_id_for_node(hit_object)
            # Gate the cheap per-type extraction on whether any picking
            # subscriber exists for this canvas (Decision 4).  hit_visual_id
            # is always computed; it is already paid for upstream and is
            # needed for visual-level hits and misses.
            if self._pick_details_enabled.get(canvas_id, False):
                pick_details = self._extract_pick_details(
                    scene_id, hit_object, event.pick_info
                )

        _ACTION = {
            "pointer_down": "press",
            "pointer_up": "release",
            "pointer_move": "move",
        }
        action = _ACTION[event.type]
        if action == "press":
            gesture_id: UUID | None = uuid4()
            self._active_gestures[canvas_id] = gesture_id
        elif action == "release":
            gesture_id = self._active_gestures.pop(canvas_id, None)
        else:  # move
            gesture_id = self._active_gestures.get(canvas_id)
        buttons = tuple(event.buttons)

        if canvas_view._dim == "2d":
            # OrthographicCamera with maintain_aspect=True stretches one axis
            # to match the canvas aspect; cam.width / cam.height alone are the
            # *minimum* visible extent.  Mirror the logic in
            # CanvasView._capture_orthographic to recover the true visible
            # extent before unprojecting NDC coordinates.
            vw = float(w) if w > 0 else 800.0
            vh = float(h) if h > 0 else 600.0
            canvas_aspect = vw / vh
            cam_w = float(cam.width) if cam.width > 0 else 1.0
            cam_h = float(cam.height) if cam.height > 0 else 1.0
            cam_aspect = cam_w / cam_h
            if canvas_aspect >= cam_aspect:
                world_height = cam_h
                world_width = cam_h * canvas_aspect
            else:
                world_width = cam_w
                world_height = cam_w / canvas_aspect

            position_2d = np.array(
                [
                    cam.local.position[0] + ndc_x * (world_width / 2.0),
                    cam.local.position[1] + ndc_y * (world_height / 2.0),
                ],
                dtype=np.float64,
            )

            self._event_bus.emit(
                _CanvasRawPointerEvent(
                    canvas_id=canvas_id,
                    scene_id=scene_id,
                    action=action,
                    camera_type="2d",
                    position_2d=position_2d,
                    ray=None,
                    hit_visual_id=hit_visual_id,
                    button=event.button,
                    modifiers=tuple(event.modifiers),
                    buttons=buttons,
                    gesture_id=gesture_id,
                    pick_details=pick_details,
                )
            )
        else:
            # Perspective camera: bilinearly interpolate the near-plane frustum
            # corners at (ndc_x, ndc_y) to get the ray origin, then compute the
            # unit direction from the camera world position through that point.
            # pygfx frustum[0] = near plane corners: [left-bottom, right-bottom,
            # right-top, left-top] in world space.
            near_corners = np.asarray(cam.frustum[0], dtype=np.float64)
            tx = (ndc_x + 1.0) / 2.0  # 0 = left, 1 = right
            ty = (ndc_y + 1.0) / 2.0  # 0 = bottom, 1 = top
            bottom = near_corners[0] + tx * (near_corners[1] - near_corners[0])
            top = near_corners[3] + tx * (near_corners[2] - near_corners[3])
            origin = bottom + ty * (top - bottom)
            cam_pos = np.array(cam.world.position, dtype=np.float64)
            d = origin - cam_pos
            direction = d / np.linalg.norm(d)
            ray = ViewRay(origin=origin, direction=direction)

            self._event_bus.emit(
                _CanvasRawPointerEvent(
                    canvas_id=canvas_id,
                    scene_id=scene_id,
                    action=action,
                    camera_type="3d",
                    position_2d=None,
                    ray=ray,
                    hit_visual_id=hit_visual_id,
                    button=event.button,
                    modifiers=tuple(event.modifiers),
                    buttons=buttons,
                    gesture_id=gesture_id,
                    pick_details=pick_details,
                )
            )

    def submit_pick_read(
        self,
        requests: list,
        fetch_fn: Callable,
        callback: Callable[[list], None],
        on_complete: Callable[[], None],
    ) -> UUID | None:
        """Read pick values through the slicer, cancellable by the returned id.

        Multiscale pick values are read at level 0 through the same cancellable
        async service as slicing (unified image design 3.7).

        Parameters
        ----------
        requests : list[ChunkRequest]
            One point request per value, all sharing one ``slice_request_id``.
        fetch_fn : Callable
            The store's ``get_data`` coroutine.
        callback : Callable[[list], None]
            Receives each batch of ``(request, data)`` pairs.
        on_complete : Callable[[], None]
            Called once every batch has arrived; never after a cancel.

        Returns
        -------
        UUID or None
            The read's id, for :meth:`cancel_pick_read`.
        """
        return self._slicer.submit(
            requests, fetch_fn, callback, consumer_id="pick", on_complete=on_complete
        )

    def cancel_pick_read(self, read_id: UUID) -> None:
        """Cancel an in-flight pick read.  A finished or unknown id is a no-op."""
        self._slicer.cancel(read_id)

    def set_pick_details_enabled(self, canvas_id: UUID, enabled: bool) -> None:
        """Enable or disable element-level pick extraction for one canvas.

        When disabled (the default), pointer events still carry
        ``hit_visual_id`` but ``pick_details`` is left ``None`` so non-picking
        consumers do not pay for the per-type dispatch.

        Parameters
        ----------
        canvas_id : UUID
            The canvas whose extraction state should change.
        enabled : bool
            True to extract typed ``VisualPickDetails`` on each pointer event.
        """
        self._pick_details_enabled[canvas_id] = enabled

    def _extract_pick_details(
        self,
        scene_id: UUID,
        hit_object: gfx.WorldObject,
        pick_info: dict,
    ) -> (
        _ImageDisplayedDataCoord | _LabelsDisplayedDataCoord | VisualPickDetails | None
    ):
        """Translate a pygfx pick payload into a typed, gfx-free pick detail.

        All ``gfx.*`` and pick-dict access stays inside this helper so nothing
        leaks past the render layer.  Returns None when the hit object has no
        element-level identity yet (stubbed visual kinds) or when there is no
        usable index in the payload.

        For image and labels visuals this returns the intermediate
        ``_ImageDisplayedDataCoord`` / ``_LabelsDisplayedDataCoord`` types
        carrying level-0 data coordinates on the displayed axes; the per-visual
        ``pick_data_coordinate`` does the node-specific decode so this helper
        never touches the world matrix.  The controller promotes them to
        full-N-dim public types in ``_on_raw_pointer_event`` once it has access
        to the dims state.

        Parameters
        ----------
        scene_id : UUID
            Scene owning the hit object.
        hit_object : gfx.WorldObject
            The picked world object from the pick buffer.
        pick_info : dict
            The pygfx pick payload (``event.pick_info``).

        Returns
        -------
        _ImageDisplayedDataCoord | _LabelsDisplayedDataCoord | VisualPickDetails | None
        """
        from cellier.events._events import (
            LinesPickInfo,
            MeshPickInfo,
            PointsPickInfo,
        )
        from cellier.render.visuals._graph_memory import GFXGraphMemoryVisual
        from cellier.render.visuals._image import GFXMultiscaleImageVisual
        from cellier.render.visuals._image_memory import GFXImageMemoryVisual
        from cellier.render.visuals._label_memory import GFXLabelMemoryVisual
        from cellier.render.visuals._label_multiscale import GFXMultiscaleLabelVisual
        from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
        from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
        from cellier.render.visuals._points_memory import GFXPointsMemoryVisual

        scene_manager = self._scenes[scene_id]
        visual_id = scene_manager.get_visual_id_for_node(hit_object)
        if visual_id is None:
            return None
        gfx_visual = scene_manager.get_visual(visual_id)

        # ── Points / Lines / Mesh ──────────────────────────────────────────
        if isinstance(gfx_visual, GFXPointsMemoryVisual):
            index = pick_info.get("vertex_index")
            if index is None:
                return None
            return PointsPickInfo(
                point_index=gfx_visual.point_index_for_vertex(int(index))
            )
        if isinstance(gfx_visual, GFXLinesMemoryVisual):
            index = pick_info.get("vertex_index")
            if index is None:
                return None
            return LinesPickInfo(
                edge_index=gfx_visual.edge_index_for_vertex(int(index))
            )
        if isinstance(gfx_visual, GFXMeshMemoryVisual):
            index = pick_info.get("face_index")
            if index is None:
                return None
            return MeshPickInfo(face_index=gfx_visual.face_index_for_pick(int(index)))
        if isinstance(gfx_visual, GFXGraphMemoryVisual):
            # The compound visual discriminates on which child was hit, so
            # the branch delegates rather than reading vertex_index here.
            return gfx_visual.decode_pick(
                hit_object, pick_info, self._data_stores.get(visual_id)
            )

        # ── Image / Labels ─────────────────────────────────────────────────
        # Each visual decodes its own node payload into level-0 data coordinates
        # on the displayed axes; no world-matrix or proxy/norm details here.
        _IMAGE_TYPES = (
            GFXImageMemoryVisual,
            GFXMultiscaleImageVisual,
        )
        _LABELS_TYPES = (
            GFXLabelMemoryVisual,
            GFXMultiscaleLabelVisual,
        )
        if not isinstance(gfx_visual, _IMAGE_TYPES + _LABELS_TYPES):
            return None

        coord = gfx_visual.pick_data_coordinate(hit_object, pick_info)
        if coord is None:
            return None
        # The visual knows which planes it drew; asking it is the only way to
        # get an answer that cannot disagree with what is on screen.
        collapsed = gfx_visual.pick_collapsed_indices()
        collapsed_pairs = (
            tuple(sorted(collapsed.items())) if collapsed is not None else None
        )
        if isinstance(gfx_visual, _LABELS_TYPES):
            return _LabelsDisplayedDataCoord(
                displayed_data_coord=coord, collapsed_data_indices=collapsed_pairs
            )
        return _ImageDisplayedDataCoord(
            displayed_data_coord=coord,
            collapsed_data_indices=collapsed_pairs,
            # The pick-buffer winner in composite mode (design 3.7); the
            # controller reads channel values from these.
            channel_index=gfx_visual.pick_channel_index(hit_object),
            drawn_channels=gfx_visual.drawn_channel_indices(),
        )

    def remove_visual(self, visual_id: UUID) -> None:
        """Remove a visual from its scene and deregister it.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.
        """
        scene_id = self._visual_to_scene.pop(visual_id)
        # Cancel this visual's in-flight slices on every canvas before its
        # slots are released, so no batch lands on a closed visual.
        for canvas_id, canvas_scene_id in list(self._canvas_to_scene.items()):
            if canvas_scene_id == scene_id:
                self._slice_coordinator.cancel_visual(scene_id, canvas_id, visual_id)
        # And drop its atlases from the chunk scheduler (design 5.4 remove).
        self._slice_coordinator.forget_visual(visual_id)
        self._data_stores.pop(visual_id)
        # Drop the per-visual render flags too.  Leaving them behind leaks,
        # and -- because the map is keyed by cellier visual id rather than by
        # anything pygfx hands out -- a visual re-registered under the same
        # id would silently inherit the old outline.
        self._visual_flags.pop(visual_id, None)
        self._scenes[scene_id].remove_visual(visual_id)

    def remove_scene(self, scene_id: UUID) -> None:
        """Remove a scene and all its visuals and canvases.

        Each visual and each canvas is closed explicitly: see
        :meth:`SceneManager.close` and :meth:`CanvasView.close`, which dropping
        references cannot substitute for.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to remove.
        """
        scene_manager = self._scenes.pop(scene_id)
        for vid in scene_manager.visual_ids:
            self._slice_coordinator.forget_visual(vid)
            self._visual_to_scene.pop(vid, None)
            self._data_stores.pop(vid, None)
            self._visual_flags.pop(vid, None)
        scene_manager.close()

        canvas_ids = [
            cid for cid, sid in self._canvas_to_scene.items() if sid == scene_id
        ]
        for cid in canvas_ids:
            self._canvas_to_scene.pop(cid)
            self._unhook_canvas(cid)
            self._canvases.pop(cid).close()
            self._active_gestures.pop(cid, None)
            self._pick_details_enabled.pop(cid, None)

    def remove_canvas(self, canvas_id: UUID) -> None:
        """Remove a single canvas, closing it and dropping its references.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas to remove.

        Raises
        ------
        KeyError
            If ``canvas_id`` is not registered.
        """
        self._canvas_to_scene.pop(canvas_id)
        self._unhook_canvas(canvas_id)
        self._canvases.pop(canvas_id).close()
        self._active_gestures.pop(canvas_id, None)
        self._pick_details_enabled.pop(canvas_id, None)

    def _unhook_canvas(self, canvas_id: UUID) -> None:
        """Remove the canvas's commit hook (see ``add_canvas``)."""
        hook = self._commit_hooks.pop(canvas_id, None)
        canvas_view = self._canvases.get(canvas_id)
        if hook is None or canvas_view is None:
            return
        try:
            canvas_view.widget.remove_event_handler(hook, "before_draw")
        except Exception:  # a canvas already torn down by its backend
            pass

    def close(self) -> None:
        """Close every canvas and scene, and drop the render references.

        The scenes are closed rather than merely forgotten: a closed controller
        can stay referenced (a reference cycle, a pending task, a traceback),
        and a multiscale visual's brick cache alone can be a gigabyte.  Closing
        each visual releases its textures now, by refcount, instead of whenever
        the cyclic garbage collector next gets to the controller.

        Safe to call more than once.
        """
        # The scheduler first: cancel its reads and drop every atlas, so the
        # visuals' brick caches are released with the scenes below.
        self._scheduler.close()
        for canvas_id in list(self._canvases):
            self._unhook_canvas(canvas_id)
        for canvas_view in list(self._canvases.values()):
            canvas_view.close()
        self._canvases.clear()
        self._canvas_to_scene.clear()
        self._active_gestures.clear()
        self._pick_details_enabled.clear()

        # Cleared in place: the slice coordinator holds these same dicts.
        for scene_manager in list(self._scenes.values()):
            scene_manager.close()
        self._scenes.clear()
        self._visual_to_scene.clear()
        self._data_stores.clear()
        self._visual_flags.clear()

    def reset_frame_counters(self, scene_id: UUID) -> None:
        """Rewind every per-frame counter that feeds *scene_id*'s pixels.

        Three things advance once per rendered frame and so make frame N
        differ from frame N+1: the temporal accumulation history, the SSAO
        kernel rotation, and the multiscale brick shader's ray-start jitter.
        All three are counters rather than random draws, so rewinding them is
        what turns "draw N frames" into a reproducible picture.

        Called before a capture (see ``cellier.render._capture``).  The visual
        counters are shared state -- the scene graph is shared across canvases
        -- so this also rewinds the jitter sequence any live canvas is drawing.
        That is harmless: the sequence is dither, not content.
        """
        for canvas_view in self._canvases.values():
            if self._canvas_to_scene.get(canvas_view.canvas_id) != scene_id:
                continue
            canvas_view._accum_pass.reset()
            canvas_view._ssao_pass.reset_frame_index()

        scene_manager = self._scenes.get(scene_id)
        if scene_manager is None:
            return
        for visual_id in scene_manager.visual_ids:
            visual = scene_manager.get_visual(visual_id)
            reset = getattr(visual, "reset_tick", None)
            if reset is not None:
                reset()

    def _make_tick_fn(self, scene_id: UUID):
        """Return a callable that ticks all visuals in *scene_id*."""

        def _tick():
            sm = self._scenes.get(scene_id)
            if sm is None:
                return
            for vid in sm.visual_ids:
                vis = sm.get_visual(vid)
                vis.tick()

        return _tick

    def get_scene(self, scene_id: UUID) -> gfx.Scene:
        """Return the pygfx Scene for ``scene_id``.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to retrieve.

        Returns
        -------
        gfx.Scene
        """
        return self._scenes[scene_id].scene

    def reslice_scene(
        self,
        scene_id: UUID,
        dims_state: DimsState,
        visual_configs: dict[UUID, VisualRenderConfig] | None = None,
        target_visual_ids: frozenset[UUID] | None = None,
        selections: Mapping[UUID, RegionSelection] | None = None,
    ) -> None:
        """Reslice all visuals in one scene.

        One reslicing request is submitted per registered canvas so that each
        canvas uses its own camera state for LOD and frustum-culling decisions.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to reslice.
        dims_state : DimsState
            Current dimension display state.
        visual_configs : dict[UUID, VisualRenderConfig] or None
            Per-visual render configuration.  ``None`` falls back to defaults.
        selections : Mapping[UUID, RegionSelection] or None
            The region each canvas is showing, keyed by canvas id.  Built by
            the controller, which owns the rendered coordinate systems; the
            render manager only routes them.
        target_visual_ids : frozenset[UUID] or None
            ``None`` reslices all visuals in the scene.
        """
        if visual_configs is None:
            visual_configs = {}
        canvases = self._find_canvases_for_scene(scene_id)
        for canvas in canvases:
            request = canvas.capture_reslicing_request(
                dims_state,
                selection=(selections or {}).get(canvas.canvas_id),
                target_visual_ids=target_visual_ids,
            )
            self._slice_coordinator.submit(request, visual_configs)

    def reslice_visual(
        self,
        visual_id: UUID,
        dims_state: DimsState,
        visual_config: VisualRenderConfig | None = None,
        selections: Mapping[UUID, RegionSelection] | None = None,
    ) -> None:
        """Reslice one visual.

        Looks up which scene owns ``visual_id``, then submits one
        ``ReslicingRequest`` per registered canvas so that each canvas uses
        its own camera state.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to reslice.
        dims_state : DimsState
            Current dimension display state.
        visual_config : VisualRenderConfig or None
            Render configuration for this visual.  ``None`` uses defaults.
        selections : Mapping[UUID, RegionSelection] or None
            The region each canvas is showing, keyed by canvas id.
        """
        cfg = visual_config if visual_config is not None else VisualRenderConfig()
        scene_id = self._visual_to_scene[visual_id]
        canvases = self._find_canvases_for_scene(scene_id)
        for canvas in canvases:
            request = canvas.capture_reslicing_request(
                dims_state,
                selection=(selections or {}).get(canvas.canvas_id),
                target_visual_ids=frozenset({visual_id}),
            )
            self._slice_coordinator.submit(request, {visual_id: cfg})

    def look_at_visual(
        self,
        visual_id: UUID,
        canvas_id: UUID,
        view_direction: tuple[float, float, float] = (-1, -1, -1),
        up: tuple[float, float, float] = (0, 0, 1),
    ) -> None:
        """Fit a canvas camera to a visual's bounding box.

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
        scene_id = self._visual_to_scene[visual_id]
        gfx_scene = self.get_scene(scene_id)
        self._canvases[canvas_id]._camera.show_object(
            gfx_scene, view_dir=view_direction, up=up
        )

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
        self._canvases[canvas_id].set_depth_range(depth_range)

    def _find_canvases_for_scene(self, scene_id: UUID) -> list[CanvasView]:
        """Return all canvases registered for *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to look up.

        Returns
        -------
        list[CanvasView]
            All canvas views rendering the scene.  Empty if none registered.
        """
        return [
            self._canvases[cid]
            for cid, sid in self._canvas_to_scene.items()
            if sid == scene_id
        ]
