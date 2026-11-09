"""Render visual of a MaskedImage.

A MaskedImage is two multiscale image visuals.  The **mask** side is this
visual itself: its node is what the scene draws, and its slots build the mask
material.  The **value** side is an inner :class:`GFXMultiscaleImageVisual`
over the value store, whose node is never in the scene (a hidden child would
still count toward the bounding box); it only plans and loads, and its atlas
is bound into the mask's material.

Both are built for the brick route only (``{"3d"}``): every view of a
MaskedImage, 2D included, is drawn from 3D bricks, so no 2D tile atlas is
allocated.  Both paint only the current slice's bricks, so a surface of
one time point is never coloured with another's values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from cellier.render._backstop import backstop_level
from cellier.render.block_cache._cell_ranges import CellRanges
from cellier.render.block_cache._surface_map import (
    SurfaceMap,
    mask_bricks_near_surface,
    value_bricks_near_surface,
)
from cellier.render.scheduling import PlanMode
from cellier.render.shaders._masked_volume_brick import MaskedVolumeBrickMaterial
from cellier.render.shaders._multiscale_volume_brick import _norm_to_data_params
from cellier.render.visuals._image import (
    GFXMultiscaleImageVisual,
    _MultiscaleImageSlot,
)
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleImageVisual,
)

if TYPE_CHECKING:
    import pygfx as gfx

    from cellier.events import (
        AppearanceChangedEvent,
        DataStoreContentsChangedEvent,
        DataStoreMetadataChangedEvent,
        TransformChangedEvent,
    )
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render._spaces import RenderSpaces
    from cellier.render.scheduling import DesiredSet
    from cellier.transform import AffineTransform
    from cellier.visuals._masked_image import MaskedImageVisual

__all__ = [
    "GFXMaskedImageVisual",
    "mask_to_value",
    "surface_interval",
    "value_clipping_planes",
    "value_model_view",
]

#: Samples along the normal before the step widens.
PROJECTION_CAP = 64


def surface_interval(thickness: float, placement: str) -> tuple[float, float]:
    """The placed interval along ``inward``, world units.

    Parameters
    ----------
    thickness : float
        ``surface_thickness``.
    placement : str
        ``"inward"`` is ``[0, T]``, ``"centred"`` ``[-T/2, T/2]`` and
        ``"outward"`` ``[-T, 0]``.
    """
    thickness = float(thickness)
    if placement == "inward":
        return 0.0, thickness
    if placement == "outward":
        return -thickness, 0.0
    return -0.5 * thickness, 0.5 * thickness


def sliced_shader_axis(slot) -> int | None:
    """The data axis, shader order, a 2D slice node's plane is normal to.

    The slice node's plane is rendered ``z = 0`` (``slice_render_plane``),
    so its normal in the slot's voxels is the third row of the voxel ->
    rendered map.  ``None`` when the plane is tilted against the slot's
    grid: the side then reads at its true depth.
    """
    world = np.asarray(slot._inner_node_3d.world.matrix, np.float64)[:3, :3]
    voxel_to_norm = np.asarray(slot._norm_size, np.float64) / np.asarray(
        slot._dataset_size, np.float64
    )
    normal = np.abs((world * voxel_to_norm)[2])
    axis = int(np.argmax(normal))
    if np.delete(normal, axis).max() > 1e-6 * normal[axis]:
        return None
    return axis


def mask_to_value(mask_slot, value_slot) -> np.ndarray:
    """The mask node's normalized space to value level-0 voxels (data order xyz).

    ``norm_to_data(value) @ inv(value node world) @ mask node world``: the
    value node is never in the scene, so its world matrix is its own.
    """
    mask_world = np.asarray(mask_slot._inner_node_3d.world.matrix, np.float64)
    value_world = np.asarray(value_slot._inner_node_3d.world.matrix, np.float64)
    scale, offset = _norm_to_data_params(
        value_slot._dataset_size, value_slot._norm_size
    )
    norm_to_data = np.eye(4)
    norm_to_data[0, 0], norm_to_data[1, 1], norm_to_data[2, 2] = scale
    norm_to_data[:3, 3] = offset
    return norm_to_data @ np.linalg.inv(value_world) @ mask_world


def _norm_to_data(slot) -> np.ndarray:
    """A slot's normalized node space to its level-0 voxels (xyz), 4x4."""
    scale, offset = _norm_to_data_params(slot._dataset_size, slot._norm_size)
    matrix = np.eye(4)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale
    matrix[:3, 3] = offset
    return matrix


#: Shader order (x, y, z) to fetch order (z, y, x), and back: its own inverse.
_FLIP = np.eye(4)[[2, 1, 0, 3]]


def value_to_mask_voxels(mask_slot, value_slot) -> np.ndarray:
    """Value level-0 voxels to mask level-0 voxels, fetch order, 4x4.

    The inverse of :func:`mask_to_value` taken from mask voxels, so the
    surface filter maps value bricks through the same matrix the shader
    reads the value with.
    """
    to_value = mask_to_value(mask_slot, value_slot) @ np.linalg.inv(
        _norm_to_data(mask_slot)
    )
    return _FLIP @ np.linalg.inv(to_value) @ _FLIP


def mask_voxel_lengths(mask_slot) -> np.ndarray:
    """The world length of one mask level-0 voxel step on each fetched axis."""
    world = np.asarray(mask_slot._inner_node_3d.world.matrix, np.float64)
    to_world = world @ np.linalg.inv(_norm_to_data(mask_slot))
    lengths = np.linalg.norm(to_world[:3, :3], axis=0)
    return lengths[::-1]


def surface_reach(thickness: float, placement: str) -> float:
    """How far from the surface the value is read, world units."""
    low, high = surface_interval(thickness, placement)
    return max(abs(low), abs(high))


class _MaskSlot(_MultiscaleImageSlot):
    """A mask slot: the masked material; paints only the current slice.

    Its atlas keeps each brick's range for the 2D band's skip.
    """

    _material_3d_class = MaskedVolumeBrickMaterial
    _current_slice_only = True

    def _make_surface_map(self) -> SurfaceMap:
        return SurfaceMap(
            self._volume_geometry.level_shapes,
            self._volume_geometry.block_size,
            int(self._block_cache_3d.info.overlap),
        )

    def _make_cell_ranges(self) -> CellRanges:
        info = self._block_cache_3d.info
        return CellRanges(
            self._volume_geometry.level_shapes,
            self._volume_geometry.block_size,
            int(info.overlap),
            int(info.n_slots) - 1,
            int(info.grid_side),
            tuple(self._lut_manager_3d.lut_data.shape[:3]),
        )


class _ValueSlot(_MultiscaleImageSlot):
    """A value slot: paints only the current slice."""

    _current_slice_only = True


class _ValueImageVisual(GFXMultiscaleImageVisual):
    """The value side: plans and loads, never drawn.

    Its 3D target is filtered by the mask visual's ``target_filter``,
    set when the mask builds it.
    """

    _slot_class = _ValueSlot
    target_filter = None

    def _filter_target_3d(self, planner, arr, config, fill):
        if self.target_filter is None:
            return arr
        return self.target_filter(planner, arr)


def value_clipping_planes(planes, model: MaskedImageVisual) -> tuple:
    """The mask's clipping planes in the value's level-0 data coordinates.

    Parameters
    ----------
    planes : Sequence[ClippingPlane]
        The planes, in the mask's level-0 data coordinates.
    model : MaskedImageVisual
        The MaskedImage, for its ``value_to_mask``; ``None`` there means
        the two share a grid and the planes pass through.

    Returns
    -------
    tuple[ClippingPlane, ...]
        Each plane with its id, pulled back to the value's system.
    """
    value_to_mask = model.value_to_mask
    if value_to_mask is None:
        return tuple(planes)
    return tuple(
        item.model_copy(update={"plane": value_to_mask.imap_plane(item.plane)})
        for item in planes
    )


def _value_appearance(model: MaskedImageVisual) -> MultiscaleImageAppearance:
    """The value side's appearance: the shared one, its own forced level."""
    appearance = model.appearance
    fields = {
        name: getattr(appearance, name)
        for name in MultiscaleImageAppearance.model_fields
        if name != "force_level"
    }
    return MultiscaleImageAppearance(**fields, force_level=appearance.value_force_level)


def value_model_view(
    model: MaskedImageVisual, value_transform: AffineTransform
) -> MultiscaleImageVisual:
    """The model the value side reads: the value store, placed and budgeted.

    Not registered anywhere: the controller knows the MaskedImage only, and
    the value side's systems come from the controller under the
    MaskedImage's own id.

    Parameters
    ----------
    model : MaskedImageVisual
        The MaskedImage.
    value_transform : AffineTransform
        The value's world transform, ``transform @ value_to_mask``.
    """
    config = model.render_config
    render_config = MultiscaleImageRenderConfig(
        **{
            name: getattr(config, name)
            for name in MultiscaleImageRenderConfig.model_fields
            if name != "gpu_budget_bytes"
        },
        gpu_budget_bytes=config.value_gpu_budget_bytes,
    )
    single = model.single
    return MultiscaleImageVisual(
        name=f"{model.name} (value)",
        data_store_id=model.value_store_id,
        level_transforms=list(model.value_level_transforms),
        appearance=_value_appearance(model),
        single=MultiscaleImageSingleAppearance(
            render_mode="iso",
            color_map=single.color_map,
            clim=single.clim,
            opacity=single.opacity,
        ),
        render_config=render_config,
        transform=value_transform,
        pick_write=False,
    )


class GFXMaskedImageVisual(GFXMultiscaleImageVisual):
    """Render-layer visual for one ``MaskedImageVisual``.

    Parameters
    ----------
    visual_model : MaskedImageVisual
        The model, held for the life of the visual.
    level_shapes : list[tuple[int, ...]]
        The mask store's shape per level, finest first.
    value_level_shapes : list[tuple[int, ...]]
        The value store's.
    value_store : Any
        The value store, for the value side's desired sets and its level-0
        system.
    value_transform : AffineTransform
        The value's world transform, ``transform @ value_to_mask``.
    world : CoordinateSystem
        The scene's world system, to compose the value's transform again
        when the visual moves.
    displayed_axes : tuple[int, ...]
        The world axes displayed at construction.
    """

    _slot_class = _MaskSlot

    def __init__(
        self,
        visual_model: MaskedImageVisual,
        level_shapes: list[tuple[int, ...]],
        value_level_shapes: list[tuple[int, ...]],
        value_store: Any,
        value_transform: AffineTransform,
        world: Any,
        displayed_axes: tuple[int, ...],
    ) -> None:
        super().__init__(visual_model, level_shapes, {"3d"}, displayed_axes)
        self.value_store_id = UUID(str(value_store.id))
        self._value_store = value_store
        self._world = world
        self._value_model = value_model_view(visual_model, value_transform)
        self._value = _ValueImageVisual(
            self._value_model,
            [tuple(shape) for shape in value_level_shapes],
            {"3d"},
            displayed_axes,
        )
        self._value_spaces: RenderSpaces | None = None
        # target_bricks="surface": the slice whose coarse level a target
        # waits for, and this plan's crossing cells for the value.
        self._deferred_surface_selection: Any = None
        self._surface_crossing: tuple | None = None
        self._value.target_filter = self._filter_value_target
        self._bind_materials()

    @property
    def value_side(self) -> GFXMultiscaleImageVisual:
        """The value side's inner visual (never in the scene)."""
        return self._value

    # -- nodes -------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """The 3D node, also for a 2D view before the route is set.

        Every 2D view of a MaskedImage is drawn by the slice node, so there
        is no 2D node; until the controller puts the visual on the route the
        3D group stands in, empty.
        """
        if len(displayed_axes) == 2 and self._slice_axes is None:
            return self.node_3d
        return super().get_node_for_dims(displayed_axes)

    # -- the route and the spaces ----------------------------------------------

    def set_slice_route(self, axes: tuple[int, int, int] | None) -> bool:
        """Put both sides on the route; the slice axes are world axes."""
        changed = super().set_slice_route(axes)
        self._value.set_slice_route(axes)
        return changed

    def prepare_slice_route(self, axes: tuple[int, int, int]) -> None:
        """Build both sides' slice resources."""
        super().prepare_slice_route(axes)
        self._value.prepare_slice_route(axes)
        self._bind_materials()

    def set_slice_footprint(self, polygon) -> None:
        """Both sides plan against the mask's footprint."""
        super().set_slice_footprint(polygon)
        self._value.set_slice_footprint(polygon)

    def set_render_spaces(
        self,
        spaces: RenderSpaces | None,
        value_spaces: RenderSpaces | None = None,
    ) -> None:
        """Hand each side its own systems.

        Parameters
        ----------
        spaces : RenderSpaces or None
            The mask's: ``controller.render_spaces``.
        value_spaces : RenderSpaces or None
            The value's: ``controller.value_render_spaces``, ``None``
            whenever the mask's are.
        """
        super().set_render_spaces(spaces)
        self._value_spaces = None if spaces is None else value_spaces
        self._value.set_render_spaces(self._value_spaces)
        self._bind_materials()

    # -- clipping ---------------------------------------------------------------

    def set_clipping_planes(self, planes) -> None:
        """Clip the mask; the value side culls its bricks by the same planes.

        The planes are in the mask's data coordinates.  The value side gets
        them pulled back through ``value_to_mask``, so a value brick wholly
        on the clipped side is not fetched; its node is never drawn, so no
        material of it clips anything.
        """
        super().set_clipping_planes(planes)
        self._value.set_clipping_planes(
            value_clipping_planes(self._clipping_planes, self._visual_model)
        )

    # -- planning --------------------------------------------------------------

    def plan(
        self,
        request,
        config: VisualRenderConfig,
        mode: PlanMode = PlanMode.FULL,
    ) -> list[DesiredSet]:
        """Both sides' desired sets, each carrying its own store.

        The mask's sets leave ``store`` unset for the coordinator to fill in
        with the visual's store; the value's carry the value store.  The
        value side plans with ``value_force_level`` in place of
        ``force_level``.
        """
        import dataclasses

        self._surface_crossing = None
        if mode is PlanMode.FULL:
            # Set again by the mask's filter if the target has to wait.
            self._deferred_surface_selection = None
        mask_sets = super().plan(request, config, mode)
        if not mask_sets:
            # The mask draws nothing here: the value has nothing to colour.
            return mask_sets
        value_config = dataclasses.replace(
            config, force_level=self._visual_model.appearance.value_force_level
        )
        value_sets = [
            dataclasses.replace(desired, store=self._value_store)
            for desired in self._value.plan(request, value_config, mode)
        ]
        # Planning can build either side's 3D resources.
        self._bind_materials()
        return [*mask_sets, *value_sets]

    # -- target_bricks="surface" ---------------------------------------------------

    @property
    def target_deferred(self) -> bool:
        """Whether the last full plan held its target back for the coarse mask.

        The coordinator reports the visual's target as deferred meanwhile,
        so its progress is not complete; the controller re-plans on the
        visual's ``BackstopCompleteEvent`` (:meth:`settle_surface`).
        """
        return self._deferred_surface_selection is not None

    def settle_surface(self) -> bool:
        """The backstop landed: plan from the coarse mask the map now holds.

        Returns
        -------
        bool
            True if a target was deferred, so the caller re-plans.
        """
        selection = self._deferred_surface_selection
        self._deferred_surface_selection = None
        if selection is None or not self._slots:
            return False
        residency = self._slots[0]._residency_3d
        if residency is not None and residency.surface_map is not None:
            residency.surface_map.settle(selection)
        return True

    def clear_surface(self) -> None:
        """Forget the coarse mask (the mask store changed)."""
        for slot in self._slots:
            residency = slot._residency_3d
            if residency is not None and residency.surface_map is not None:
                residency.surface_map.clear()

    def _filter_target_3d(self, planner, arr, config, fill):
        """Keep the mask's target bricks near the surface.

        Every 3D full plan keeps the map on the backstop level, whatever
        ``target_bricks`` is, so switching to "surface" finds the coarse
        mask already there: a backstop already resident is not written
        again.  Under "surface" the target waits until the mask's coarse
        level for this slice is known: ``None`` (no target) until then, and
        the visual is planned again when it is.  Every other plan keeps
        ``arr``.
        """
        residency = planner.residency_3d()
        surface_map = None if residency is None else residency.surface_map
        geometry = planner._volume_geometry
        if surface_map is None or geometry is None:
            return arr
        n_levels = len(geometry.level_shapes)
        loading = config.loading
        level = n_levels if loading is None else backstop_level(loading, n_levels)
        surface_map.set_level(level)
        if self._visual_model.appearance.target_bricks != "surface":
            return arr
        selection = planner._level_slice_selection(level - 1, fill)
        if not surface_map.ready(selection):
            self._deferred_surface_selection = selection
            return None
        crossing = surface_map.crossing(
            selection, self._visual_model.single.iso_threshold
        )
        if crossing is None:
            # Settled with no brick of it written: nothing to filter by.
            return arr
        scale = np.asarray(geometry._scale_vecs_data[level - 1], np.float64)
        translation = np.asarray(geometry._translation_vecs_data[level - 1], np.float64)
        self._surface_crossing = (crossing, scale, translation)
        if arr is None:
            return arr
        return mask_bricks_near_surface(
            arr,
            crossing,
            block_size=geometry.block_size,
            level_shapes=geometry.level_shapes,
            scales=geometry._scale_vecs_data,
            translations=geometry._translation_vecs_data,
            coarse_level=level,
        )

    def _filter_value_target(self, planner, arr):
        """Keep the value's target bricks within reach of the surface."""
        if self._visual_model.appearance.target_bricks != "surface" or arr is None:
            return arr
        if self._deferred_surface_selection is not None:
            return None
        if self._surface_crossing is None or not self._slots:
            return arr
        mask_slot = self._slots[0]
        geometry = planner._volume_geometry
        if (
            geometry is None
            or mask_slot._inner_node_3d is None
            or planner._inner_node_3d is None
        ):
            return arr
        crossing, scale, translation = self._surface_crossing
        single = self._visual_model.single
        reach = surface_reach(single.surface_thickness, single.surface_placement)
        return value_bricks_near_surface(
            arr,
            crossing,
            block_size=geometry.block_size,
            level_shapes=geometry.level_shapes,
            scales=geometry._scale_vecs_data,
            translations=geometry._translation_vecs_data,
            value_to_mask=value_to_mask_voxels(mask_slot, planner),
            mask_scale=scale,
            mask_translation=translation,
            reach_voxels=reach / mask_voxel_lengths(mask_slot),
        )

    def residencies(self) -> dict:
        """``cache_id -> adapter`` for both sides' atlases."""
        out = dict(super().residencies())
        out.update(self._value.residencies())
        return out

    # -- binding -----------------------------------------------------------------

    def _apply_materials(self) -> None:
        """Restyle as an image does, then the masked material's own settings."""
        super()._apply_materials()
        self._bind_materials()

    def _bind_materials(self) -> None:
        """Push the value side's atlas and the surface settings to the mask.

        Runs whenever either side may have built or rebuilt its resources
        (planning, the route, the spaces) and whenever the appearance
        changes; every setter is a no-op when nothing changed.  Until both
        sides have their 3D resources the mask draws as a plain image of
        nothing: it paints no brick of its own before then either.
        """
        value = getattr(self, "_value", None)
        if value is None or not self._slots or not value._slots:
            return
        mask_slot = self._slots[0]
        value_slot = value._slots[0]
        material = mask_slot.material_3d
        if (
            not isinstance(material, MaskedVolumeBrickMaterial)
            or value_slot._block_cache_3d is None
            or value_slot._inner_node_3d is None
            or mask_slot._inner_node_3d is None
        ):
            return
        # Never drawn itself, the value atlas is not uploaded by a draw.
        value_slot._block_cache_3d.allocate_on_gpu()
        material.set_value_source(
            value_slot._block_cache_3d.cache_tex,
            value_slot._lut_manager_3d.lut_tex,
            value_slot._vol_params_buffer,
            value_slot._brick_scales_buffer,
        )
        material.set_level_bounds("mask", mask_slot._volume_geometry.level_shapes)
        material.set_level_bounds("value", value_slot._volume_geometry.level_shapes)
        material.set_mask_to_value(mask_to_value(mask_slot, value_slot))
        model = self._visual_model
        appearance, single = model.appearance, model.single
        if material.interpolation != appearance.interpolation:
            material.interpolation = appearance.interpolation
        material.surface_op = single.surface_projection
        material.set_surface(
            *surface_interval(single.surface_thickness, single.surface_placement),
            float(appearance.projection_steps_per_voxel),
            PROJECTION_CAP,
        )
        material.set_band(
            float(single.surface_min_width_px), single.band_rays, single.band_cap
        )
        mask_axis = None
        if self._slice_axes is not None:
            mask_axis = sliced_shader_axis(mask_slot)
            material.set_read(mask_axis, sliced_shader_axis(value_slot))
        residency = mask_slot._residency_3d
        ranges = None if residency is None else residency.cell_ranges
        if ranges is not None:
            ranges.set_threshold(single.iso_threshold)
            # Shader order is x, y, z; the table is z, y, x.
            ranges.set_layer_axis(None if mask_axis is None else 2 - mask_axis)
            material.set_mask_ranges(ranges.texture)

    # -- events ----------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Move both images: the value's transform is ``transform @ value_to_mask``."""
        from cellier._masked_image import value_world_transform

        super().on_transform_changed(event)
        transform = value_world_transform(
            event.transform,
            self._visual_model.value_to_mask,
            self._value_store.data_coordinate_systems[0],
            self._world,
        )
        self._value_model.transform = transform
        self._value.on_transform_changed(event._replace(transform=transform))
        self._bind_materials()

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Restyle the mask; keep the value side's appearance in step."""
        super().on_appearance_changed(event)
        name = event.field_name
        if name == "value_force_level":
            name = "force_level"
        elif name == "force_level":
            return
        if name not in MultiscaleImageAppearance.model_fields:
            return
        setattr(self._value_model.appearance, name, event.new_value)
        self._value.on_appearance_changed(
            event._replace(field_name=name, new_value=event.new_value)
        )

    def on_data_store_contents_changed(
        self, event: DataStoreContentsChangedEvent
    ) -> None:
        """Hand a store's change to the side that reads it."""
        if UUID(str(event.data_store_id)) == self.value_store_id:
            self._value.on_data_store_contents_changed(event)
        else:
            self.clear_surface()
            super().on_data_store_contents_changed(event)

    def on_data_store_metadata_changed(
        self, event: DataStoreMetadataChangedEvent
    ) -> None:
        """Hand a store's change to the side that reads it."""
        if UUID(str(event.data_store_id)) == self.value_store_id:
            self._value.on_data_store_metadata_changed(event)
        else:
            self.clear_surface()
            super().on_data_store_metadata_changed(event)
        self._bind_materials()

    def tick(self) -> None:
        """Advance both sides' jitter seeds; the band's skip catches up.

        A threshold change zeroes the skip's clearance at once; it is
        computed again here, once a frame however many ticks a drag sends.
        """
        super().tick()
        self._value.tick()
        if self._slots:
            residency = self._slots[0]._residency_3d
            if residency is not None and residency.cell_ranges is not None:
                residency.cell_ranges.refresh()

    def reset_tick(self) -> None:
        """Rewind both sides' jitter seeds."""
        super().reset_tick()
        self._value.reset_tick()

    # -- lifetime --------------------------------------------------------------

    def close(self) -> None:
        """Release both sides' slots and atlases.  Unusable afterwards."""
        value = self._value
        if value is not None:
            value.target_filter = None
            value.close()
        self._value = None
        self._value_model = None
        self._value_store = None
        self._value_spaces = None
        super().close()
