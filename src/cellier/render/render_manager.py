"""RenderManager — single top-level render-layer object."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

import numpy as np

from cellier.events import DimsChangedEvent, EventBus
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
from cellier.render.slice_coordinator import SliceCoordinator
from cellier.slicer import AsyncSlicer

if TYPE_CHECKING:
    from uuid import UUID

    import pygfx as gfx
    from PySide6.QtWidgets import QWidget

    from cellier.data._base_data_store import BaseDataStore
    from cellier.events._events import VisualPickDetails
    from cellier.render._requests import DimsState
    from cellier.render.visuals._canvas_overlay import GFXCanvasOverlay
    from cellier.render.visuals._image import GFXMultiscaleImageVisual
    from cellier.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
    from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
    from cellier.render.visuals._points_memory import GFXPointsMemoryVisual

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

    ``_extract_pick_details`` can only decode the rendered axes; the
    controller promotes this to a full-N-dim ``ImagePickInfo`` in
    ``_on_raw_pointer_event`` by filling non-displayed axes from dims state.

    Parameters
    ----------
    displayed_data_coord : tuple[float, ...]
        Level-0 data-array position on the displayed axes only (``floor`` gives
        the index).  Length 2 for a 2-D canvas, 3 for a 3-D canvas.
    """

    displayed_data_coord: tuple[float, ...]


class _LabelsDisplayedDataCoord(NamedTuple):
    """Render-layer intermediate for labels pick — displayed axes only.

    Same semantics as ``_ImageDisplayedDataCoord``; separate type so the
    controller can produce the correct public ``LabelsPickInfo``.
    """

    displayed_data_coord: tuple[float, ...]


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
        self._slice_coordinator = SliceCoordinator(
            scenes=self._scenes,
            slicer=self._slicer,
            data_stores=self._data_stores,
        )

    def connect_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe internal components to *event_bus*.

        Must be called before the caller registers its own DimsChangedEvent
        handler so the SliceCoordinator invalidates stale 2D caches first.
        """
        self._event_bus = event_bus
        # Let the coordinator emit ResliceStartedEvent / ResliceCompletedEvent.
        self._slice_coordinator._event_bus = event_bus
        event_bus.subscribe(
            DimsChangedEvent,
            self._slice_coordinator._on_dims_changed,
            owner_id=self._slice_coordinator.id,
        )
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
    def config(self) -> RenderManagerConfig:
        """Current rendering performance configuration.

        Reflects live state: mutations via ``temporal_alpha`` and
        ``temporal_enabled`` setters are visible here immediately.
        """
        return self._config

    @property
    def temporal_alpha(self) -> float:
        """EMA floor weight for temporal accumulation."""
        return self._config.temporal.alpha

    @temporal_alpha.setter
    def temporal_alpha(self, value: float) -> None:
        self._config.temporal.alpha = value
        for canvas in self._canvases.values():
            canvas._accum_pass.alpha = value

    @property
    def temporal_enabled(self) -> bool:
        """Whether the temporal accumulation pass is active."""
        return self._config.temporal.enabled

    @temporal_enabled.setter
    def temporal_enabled(self, value: bool) -> None:
        self._config.temporal.enabled = value
        for canvas in self._canvases.values():
            canvas._accum_pass.enabled = value

    # ------------------------------------------------------------------
    # Screen-space ambient occlusion
    # ------------------------------------------------------------------

    @property
    def ssao_enabled(self) -> bool:
        """Whether the ambient occlusion pass is active.

        A canvas in 2D never runs the pass whatever this says; see
        ``CanvasView.set_ssao_enabled``.
        """
        return self._config.ssao.enabled

    @ssao_enabled.setter
    def ssao_enabled(self, value: bool) -> None:
        value = bool(value)
        self._config.ssao.enabled = value
        for canvas in self._canvases.values():
            canvas.set_ssao_enabled(value)

    @property
    def ssao_radius(self) -> float | None:
        """Hemisphere radius in scene units, or ``None`` for auto."""
        return self._config.ssao.radius

    @ssao_radius.setter
    def ssao_radius(self, value: float | None) -> None:
        value = None if value is None else float(value)
        self._config.ssao.radius = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.radius = value

    @property
    def ssao_strength(self) -> float:
        """How far the occlusion multiply is applied, 0 (off) to 1 (full)."""
        return self._config.ssao.strength

    @ssao_strength.setter
    def ssao_strength(self, value: float) -> None:
        value = float(value)
        self._config.ssao.strength = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.strength = value

    @property
    def ssao_power(self) -> float:
        """Contrast exponent applied to the occlusion before the multiply."""
        return self._config.ssao.power

    @ssao_power.setter
    def ssao_power(self, value: float) -> None:
        value = float(value)
        self._config.ssao.power = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.power = value

    @property
    def ssao_bias(self) -> float:
        """Depth-comparison bias, as a fraction of the effective radius."""
        return self._config.ssao.bias

    @ssao_bias.setter
    def ssao_bias(self, value: float) -> None:
        value = float(value)
        self._config.ssao.bias = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.bias = value

    @property
    def ssao_n_samples(self) -> int:
        """Hemisphere samples per pixel.  Changing this recompiles."""
        return self._config.ssao.n_samples

    @ssao_n_samples.setter
    def ssao_n_samples(self, value: int) -> None:
        value = int(value)
        self._config.ssao.n_samples = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.n_samples = value

    @property
    def ssao_blur_radius(self) -> int:
        """Box-blur half-width in internal pixels.  Changing this recompiles."""
        return self._config.ssao.blur_radius

    @ssao_blur_radius.setter
    def ssao_blur_radius(self, value: int) -> None:
        value = int(value)
        self._config.ssao.blur_radius = value
        for canvas in self._canvases.values():
            canvas._ssao_pass.blur_radius = value

    def apply_ssao_config(self) -> None:
        """Push the current ``config.ssao`` onto every canvas's pass."""
        for canvas in self._canvases.values():
            canvas.apply_ssao_config(self._config.ssao)

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
            labels inside one volume.  Requires the ``outline_id`` target;
            without it a label visual falls back to a whole-object
            silhouette.

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
        ssao_on = self._config.ssao.enabled
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

    def add_scene(self, scene_id: UUID, lighting: str = "none") -> SceneManager:
        """Create and register a new scene.

        Parameters
        ----------
        scene_id : UUID
            Unique identifier for the scene.
        lighting : str
            ``"none"`` (default) or ``"default"``.  Pass ``"default"`` to
            add ambient and directional lights — required for
            ``MeshPhongAppearance``.

        Returns
        -------
        SceneManager
            The newly created scene manager.
        """
        scene_manager = SceneManager(scene_id=scene_id, lighting=lighting)
        self._scenes[scene_id] = scene_manager
        return scene_manager

    def scene_has_lighting(self, scene_id: UUID) -> bool:
        """Return True if *scene_id* was created with lighting enabled."""
        return self._scenes[scene_id].has_lighting

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
            ssao_enabled=self._config.ssao.enabled,
            **canvas_view_kwargs,
        )
        # Apply temporal config to the canvas's accumulation pass.
        canvas_view._accum_pass.alpha = self._config.temporal.alpha
        if not self._config.temporal.enabled:
            canvas_view._accum_pass.enabled = False
        # Apply the ambient occlusion config.  The pass stays off in 2D
        # whatever the config says.
        canvas_view.apply_ssao_config(self._config.ssao)
        # Apply outline config, and wire the per-frame LUT re-sync.
        canvas_view._outline_pass.apply_config(self._config.outline)
        canvas_view._visual_lut_sync_fn = self._sync_visual_lut
        if self._config.outline.enabled:
            self._warn_if_outline_unavailable()
        # Wire up per-frame tick for visuals (e.g. jitter seed advance).
        canvas_view._tick_visuals_fn = self._make_tick_fn(scene_id)
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
        self._scenes[scene_id].add_visual(visual, displayed_axes)
        self._visual_to_scene[visual.visual_model_id] = scene_id
        self._data_stores[visual.visual_model_id] = data_store

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

        # ── Image / Labels ─────────────────────────────────────────────────
        # Each visual decodes its own node payload into level-0 data coordinates
        # on the displayed axes; no world-matrix or proxy/norm details here.
        _IMAGE_TYPES = (
            GFXImageMemoryVisual,
            GFXMultiscaleImageVisual,
            GFXMultichannelImageMemoryVisual,
            GFXMultichannelMultiscaleImageVisual,
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
        if isinstance(gfx_visual, _LABELS_TYPES):
            return _LabelsDisplayedDataCoord(displayed_data_coord=coord)
        return _ImageDisplayedDataCoord(displayed_data_coord=coord)

    def remove_visual(self, visual_id: UUID) -> None:
        """Remove a visual from its scene and deregister it.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.
        """
        scene_id = self._visual_to_scene.pop(visual_id)
        self._data_stores.pop(visual_id)
        self._scenes[scene_id].remove_visual(visual_id)

    def remove_scene(self, scene_id: UUID) -> None:
        """Remove a scene and all its visuals and canvases.

        Visuals are released by dropping references (pygfx has no explicit
        destroy API), but each canvas is closed explicitly -- see
        :meth:`CanvasView.close`, which GC alone cannot substitute for.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to remove.
        """
        scene_manager = self._scenes.pop(scene_id)
        for vid in scene_manager.visual_ids:
            self._visual_to_scene.pop(vid, None)
            self._data_stores.pop(vid, None)
        # scene_manager goes out of scope here; GC drops gfx.Scene + all nodes.

        canvas_ids = [
            cid for cid, sid in self._canvas_to_scene.items() if sid == scene_id
        ]
        for cid in canvas_ids:
            self._canvas_to_scene.pop(cid)
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
        self._canvases.pop(canvas_id).close()
        self._active_gestures.pop(canvas_id, None)
        self._pick_details_enabled.pop(canvas_id, None)

    def close(self) -> None:
        """Close every registered canvas and drop the render references.

        Safe to call more than once.
        """
        for canvas_view in list(self._canvases.values()):
            canvas_view.close()
        self._canvases.clear()
        self._canvas_to_scene.clear()
        self._active_gestures.clear()
        self._pick_details_enabled.clear()

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
        target_visual_ids : frozenset[UUID] or None
            ``None`` reslices all visuals in the scene.
        """
        if visual_configs is None:
            visual_configs = {}
        canvases = self._find_canvases_for_scene(scene_id)
        for canvas in canvases:
            request = canvas.capture_reslicing_request(
                dims_state, target_visual_ids=target_visual_ids
            )
            self._slice_coordinator.submit(request, visual_configs)

    def reslice_visual(
        self,
        visual_id: UUID,
        dims_state: DimsState,
        visual_config: VisualRenderConfig | None = None,
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
        """
        cfg = visual_config if visual_config is not None else VisualRenderConfig()
        scene_id = self._visual_to_scene[visual_id]
        canvases = self._find_canvases_for_scene(scene_id)
        for canvas in canvases:
            request = canvas.capture_reslicing_request(
                dims_state, target_visual_ids=frozenset({visual_id})
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
