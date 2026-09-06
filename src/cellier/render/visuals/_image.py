"""GFXMultiscaleImageVisual — render-layer visual for multiscale images."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.data.image import ChunkRequest
from cellier.logging import _GPU_LOGGER, _PERF_LOGGER
from cellier.render._frustum import (
    bricks_in_frustum_arr,
    frustum_planes_from_corners,
)
from cellier.render._level_of_detail import (
    arr_to_brick_keys,
    build_level_grids,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.render._level_of_detail_2d import (
    arr_to_block_keys_2d,
    build_tile_grids_2d,
    select_lod_2d,
    sort_tiles_by_distance_2d,
    viewport_cull_2d,
)
from cellier.render.block_cache import (
    BlockCache3D,
    BlockKey3D,
    TileSlot,
    compute_block_cache_parameters_3d,
)
from cellier.render.block_cache._block_cache_2d import BlockCache2D
from cellier.render.block_cache._cache_parameters_2d import (
    compute_block_cache_parameters_2d,
)
from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.render.lut_indirection._layout_2d import BlockLayout2D
from cellier.render.lut_indirection._lut_buffers_2d import (
    build_block_scales_buffer_2d,
    build_lut_params_buffer_2d,
)
from cellier.render.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)
from cellier.render.shaders._block_image import ImageBlockMaterial
from cellier.render.shaders._multiscale_volume_brick import (
    MultiscaleVolumeBrickMaterial,
    build_brick_scales_buffer,
    build_vol_params_buffer,
    compose_world_transform,
    compute_normalized_size,
    norm_full_extent_box,
)
from cellier.render.visuals._image_memory import (
    _box_wireframe_positions,
    _make_aabb_line,
    _pygfx_matrix,
    _rect_wireframe_positions,
)
from cellier.render.visuals._paint_tile_slot_manager import (
    PaintTileSlotManager,
)
from cellier.render.visuals._pick import (
    multiscale_image_data_coordinate,
    multiscale_volume_data_coordinate,
)
from cellier.render.visuals._slicing import map_world_slice_to_voxel
from cellier.transform import AffineTransform
from cellier.transform._axis_order import select_axes, swap_axes

if TYPE_CHECKING:
    from pygfx.resources import Buffer

    from cellier._state import AxisAlignedSelectionState, DimsState
    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        DataStoreContentsChangedEvent,
        DataStoreMetadataChangedEvent,
        PickWriteChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.render.block_cache._tile_manager_2d import (
        BlockKey2D,
    )
    from cellier.render.block_cache._tile_manager_2d import (
        TileSlot as TileSlot2D,
    )
    from cellier.visuals._image import MultiscaleImageVisual

# Importing this module registers the shader class with pygfx via the
# @register_wgpu_render_function decorator.
import cellier.render.shaders._multiscale_volume_brick as _brick_reg  # noqa: F401


class NormSizedVolume(gfx.Volume):
    """gfx.Volume subclass with a norm_size-aware bounding box.

    The standard gfx.Volume derives its local bounding box from the proxy
    texture dimensions, which for a 2x2x2 dummy produces an asymmetric box
    that offsets the orbit center by half the scene size.  This subclass
    overrides get_bounding_box() to return the full voxel-extent box in
    normalized local space, i.e. the box that maps (via the node's
    norm->world matrix) to data ``[-0.5, N-0.5]`` so the corners sit on the
    *outer edges* of the boundary voxels (see ``norm_full_extent_box``).
    """

    def __init__(
        self,
        geometry,
        material,
        *,
        norm_size: np.ndarray | None = None,
        dataset_size: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(geometry, material, **kwargs)
        self._norm_size: np.ndarray | None = (
            np.asarray(norm_size, dtype=np.float64) if norm_size is not None else None
        )
        # Voxel counts are constant across transform changes, so store once.
        self._dataset_size: np.ndarray | None = (
            np.asarray(dataset_size, dtype=np.float64)
            if dataset_size is not None
            else None
        )

    def update_norm_size(self, norm_size: np.ndarray) -> None:
        """Update the bounding box extents after a transform change."""
        self._norm_size = np.asarray(norm_size, dtype=np.float64)

    def get_bounding_box(self) -> np.ndarray | None:
        if self._norm_size is None or self._dataset_size is None:
            return super().get_bounding_box()
        return norm_full_extent_box(self._dataset_size, self._norm_size)

    def _wgpu_get_pick_info(self, pick_value: int) -> dict:
        """Decode the custom 3x14-bit pick encoding written by the brick shader.

        The inherited ``gfx.Volume._wgpu_get_pick_info`` normalises by
        ``tex.size`` (the 2x2x2 proxy), which destroys all positional
        information.  This override decodes the three 14-bit fields directly
        as normalised [0, 1] floats and converts them to centred normalised
        object space using ``self._norm_size``.

        The shader writes::

            pick_pack(global_id, 20)
            +pick_pack((pos / norm_size + 0.5).x * 16383, 14)
            +pick_pack((pos / norm_size + 0.5).y * 16383, 14)
            +pick_pack((pos / norm_size + 0.5).z * 16383, 14)

        where ``pos`` is the surface position in centred normalised space.

        Returns a dict with keys ``"world_object"`` and ``"norm_pos"``.
        ``"norm_pos"`` is a 3-tuple of floats in centred normalised object
        space (same space that ``u_wobject.world_transform`` acts on).
        ``"index"`` is intentionally absent so callers cannot accidentally
        use the wrong decode path.
        """
        from pygfx.utils import unpack_bitfield

        values = unpack_bitfield(pick_value, wobject_id=20, x=14, y=14, z=14)
        nx = values["x"] / 16383.0
        ny = values["y"] / 16383.0
        nz = values["z"] / 16383.0
        if self._norm_size is not None:
            px = (nx - 0.5) * float(self._norm_size[0])
            py = (ny - 0.5) * float(self._norm_size[1])
            pz = (nz - 0.5) * float(self._norm_size[2])
        else:
            px, py, pz = nx - 0.5, ny - 0.5, nz - 0.5
        return {"world_object": self, "norm_pos": (px, py, pz)}


def _extract_scale_and_translation(
    level_transforms: list[AffineTransform],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract per-axis scale and translation vectors from level transforms.

    Vectors are returned in the same axis order as the input transforms.
    For VolumeGeometry / ImageGeometry2D the transforms are already
    projected onto the displayed axes via ``select_axes``, so the
    output is in displayed-axis order over the displayed subset (e.g.
    ``(z, y, x)`` when displayed_axes=(0, 1, 2)).  Callers that feed
    these into shader-space geometry must reverse the local axis order
    via ``swap_axes`` first.

    Parameters
    ----------
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0.

    Returns
    -------
    scale_vecs : list[np.ndarray]
        ``(ndim,)`` per level — diagonal of the spatial block.
    translation_vecs : list[np.ndarray]
        ``(ndim,)`` per level — translation column.
    """
    scale_vecs: list[np.ndarray] = []
    translation_vecs: list[np.ndarray] = []
    for t in level_transforms:
        nd = t.ndim
        scale_vecs.append(np.diag(t.matrix[:nd, :nd]).copy())
        translation_vecs.append(t.matrix[:nd, nd].copy())
    return scale_vecs, translation_vecs


# ---------------------------------------------------------------------------
# Brick-shader transform helpers
# ---------------------------------------------------------------------------


def _check_transform_no_rotation(transform: AffineTransform) -> None:
    """Raise ValueError if the transform contains rotation or shear.

    The brick shader only supports scale and translation.  A transform with
    off-diagonal entries in the linear submatrix (i.e. rotation or shear)
    would require ``norm_to_voxel`` to be a full affine inverse, which the
    current WGSL does not implement.

    Parameters
    ----------
    transform : AffineTransform
        The data-to-world transform to validate.

    Raises
    ------
    ValueError
        If the linear part of the transform is not diagonal.
    """
    nd = transform.ndim
    linear = transform.matrix[:nd, :nd]
    diagonal_only = np.diag(np.diag(linear))
    if not np.allclose(linear, diagonal_only, atol=1e-5):
        raise ValueError(
            "The brick shader only supports scale and translation transforms. "
            "The provided transform contains rotation or shear components. "
            "Support for general affine transforms requires changes to the "
            "WGSL norm_to_voxel function and is not yet implemented."
        )


def _norm_size_from_transform(
    transform: AffineTransform,
    displayed_axes: tuple[int, ...],
    dataset_size_xyz: np.ndarray,
) -> np.ndarray:
    """Compute normalised physical size from a scale+translation transform.

    Projects ``transform`` onto the 3 displayed data axes via
    ``select_axes``, extracts per-axis scale factors from the column
    norms of the linear submatrix (now in displayed-axis order over the
    3 displayed axes), reverses to shader order via ``swap_axes``,
    multiplies by ``dataset_size_xyz``, and normalises so the longest
    axis equals 1.0.

    For a pure diagonal (scale-only) transform the column norms equal the
    absolute diagonal values, so this is exact.

    Parameters
    ----------
    transform : AffineTransform
        The full data-to-world transform (any ``ndim >= 3``).  Must be
        scale + translation only (validated separately by
        ``_check_transform_no_rotation``).
    displayed_axes : tuple[int, ...]
        The 3 displayed data axes, in display order.
    dataset_size_xyz : ndarray, shape (3,)
        Finest-level voxel counts in shader order (x=W, y=H, z=D).

    Returns
    -------
    ndarray, shape (3,)
        Normalised physical size in shader order, with the longest axis
        equal to 1.0.
    """
    if len(displayed_axes) != 3:
        raise ValueError(
            f"_norm_size_from_transform requires 3 displayed axes, "
            f"got {len(displayed_axes)} ({displayed_axes})"
        )
    sub = transform.select_axes(displayed_axes)
    nd = sub.ndim
    linear = sub.matrix[:nd, :nd]
    # Column norms are in displayed-axis order over the 3 displayed axes.
    col_norms = np.array(
        [np.linalg.norm(linear[:, i]) for i in range(nd)], dtype=np.float64
    )
    # Convert displayed-axis order → shader order via explicit reversal.
    per_axis_scale_xyz = np.asarray(swap_axes(col_norms, (2, 1, 0)))
    return compute_normalized_size(dataset_size_xyz, per_axis_scale_xyz)


# ---------------------------------------------------------------------------
# VolumeGeometry
# ---------------------------------------------------------------------------


class MultiscaleBrickLayout3D:
    """Pre-built metadata cache for a multiscale 3-D volume.

    Holds per-level ``BlockLayout3D`` objects and the precomputed coarse
    grid arrays used by the LOAD selection pipeline.  Never touches the
    live ``DataStore`` after construction.

    Parameters
    ----------
    level_shapes : list[tuple[int, ...]]
        Displayed-axis shape at each scale level, finest first.
        For 3D rendering this is ``(D, H, W)``; for 2D it is ``(H, W)``.
        The caller extracts the displayed dimensions before passing.
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0.
        ``level_transforms[0]`` must be the identity.
    block_size : int
        Rendering brick side length in voxels.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, ...]],
        level_transforms: list[AffineTransform],
        block_size: int,
    ) -> None:
        self.level_transforms = list(level_transforms)
        self.block_size = block_size
        self.n_levels = len(level_shapes)

        ndim = level_transforms[0].ndim
        assert np.allclose(level_transforms[0].matrix, np.eye(ndim + 1)), (
            "level_transforms[0] must be the identity"
        )

        # Inputs are in displayed-axis order over the 3 displayed axes
        # (e.g. (z, y, x) when displayed_axes=(0, 1, 2)).
        sv_data, tv_data = _extract_scale_and_translation(level_transforms)
        self._scale_vecs_data = sv_data
        self._translation_vecs_data = tv_data

        # Shader / pygfx order is the reversal of displayed-axis order.
        # Always a full reversal of the local 3 axes, regardless of which
        # data axes were selected upstream in ``from_cellier_model``.
        _to_shader_3d = (2, 1, 0)
        self._scale_vecs_shader = [
            np.asarray(swap_axes(sv, _to_shader_3d)) for sv in sv_data
        ]
        self._translation_vecs_shader = [
            np.asarray(swap_axes(tv, _to_shader_3d)) for tv in tv_data
        ]

        # (n_levels, 3) arrays for vectorised hot-path lookups.
        self._scale_arr_shader = np.stack(self._scale_vecs_shader, axis=0)
        self._translation_arr_shader = np.stack(self._translation_vecs_shader, axis=0)

        # Scalar LOD factor per level (geometric mean of the 3 per-axis scales).
        self._level_scale_factors = [
            float(np.prod(sv) ** (1.0 / len(sv))) for sv in sv_data
        ]

        self._rebuild(level_shapes)

    def _rebuild(self, level_shapes: list[tuple[int, ...]]) -> None:
        self.level_shapes = list(level_shapes)
        self.layouts = [
            BlockLayout3D(volume_shape=shape, block_size=self.block_size)
            for shape in level_shapes
        ]
        self.base_layout = self.layouts[0]
        self._level_grids = build_level_grids(
            self.base_layout,
            self.n_levels,
            self._scale_vecs_shader,
            self._translation_vecs_shader,
            level_shapes=self.level_shapes,
        )

    def update(self, level_shapes: list[tuple[int, ...]]) -> None:
        """Rebuild from new level shapes after a DataStoreMutated event."""
        self._rebuild(level_shapes)


class ImageGeometry3D:
    """Pre-built metadata cache for a multiscale 2-D image.

    Analogous to ``VolumeGeometry`` but for 2D tile grids.

    Parameters
    ----------
    level_shapes : list[tuple[int, int]]
        Image shape ``(H, W)`` at each scale level, finest first.
        The caller extracts the two displayed dimensions before passing.
    block_size : int
        Tile side length in pixels.
    n_levels : int
        Number of LOD levels.
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0
        in the 2D displayed-axis sub-space.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, int]],
        block_size: int,
        n_levels: int,
        level_transforms: list[AffineTransform] | None = None,
    ) -> None:
        self.block_size = block_size
        self.n_levels = n_levels
        self.level_shapes = list(level_shapes)

        # Build 2D transforms if not provided (fallback to identity).
        if level_transforms is None:
            level_transforms = [
                AffineTransform.identity(ndim=2) for _ in range(n_levels)
            ]
        self.level_transforms = list(level_transforms)

        # Inputs are in displayed-axis order over the 2 displayed axes
        # (e.g. (H, W) when displayed_axes=(1, 2)).
        sv_data, tv_data = _extract_scale_and_translation(self.level_transforms)
        self._scale_vecs_data = sv_data
        self._translation_vecs_data = tv_data

        # Shader order is the reversal of displayed-axis order.
        _to_shader_2d = (1, 0)
        self._scale_vecs_shader = [
            np.asarray(swap_axes(sv, _to_shader_2d)) for sv in sv_data
        ]
        self._translation_vecs_shader = [
            np.asarray(swap_axes(tv, _to_shader_2d)) for tv in tv_data
        ]

        self._scale_arr_shader = np.stack(self._scale_vecs_shader, axis=0)
        self._translation_arr_shader = np.stack(self._translation_vecs_shader, axis=0)

        # Scalar LOD factor per level (geometric mean of per-axis scales).
        self._level_scale_factors = [float(np.sqrt(np.prod(sv))) for sv in sv_data]

        # Build 2D base layout from finest level (H, W).
        self.base_layout = BlockLayout2D.from_shape(
            shape=tuple(level_shapes[0]),
            block_size=block_size,
        )
        self._level_grids = build_tile_grids_2d(
            self.base_layout,
            n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )

    def update(self, level_shapes: list[tuple[int, int]]) -> None:
        """Rebuild from new level shapes after displayed axes change."""
        self.level_shapes = list(level_shapes)
        self.base_layout = BlockLayout2D.from_shape(
            shape=tuple(level_shapes[0]),
            block_size=self.block_size,
        )
        self._level_grids = build_tile_grids_2d(
            self.base_layout,
            self.n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _brick_key_to_padded_coords(
    key: BlockKey3D,
    block_size: int,
    overlap: int,
) -> tuple[int, int, int, int, int, int]:
    """Translate a BlockKey3D to padded voxel coordinates.

    Returns z0, y0, x0, z1, y1, x1.  May extend outside store bounds;
    the DataStore is responsible for clamping and zero-padding.
    """
    padded = block_size + 2 * overlap
    z0 = key.g0 * block_size - overlap
    y0 = key.g1 * block_size - overlap
    x0 = key.g2 * block_size - overlap
    return z0, y0, x0, z0 + padded, y0 + padded, x0 + padded


def _build_axis_selections_multiscale(
    sel: AxisAlignedSelectionState,
    ndim: int,
    display_coords: list[tuple[int, int]],
    level_shape: tuple[int, ...],
    world_to_level_k: AffineTransform,
) -> tuple[int | tuple[int, int], ...]:
    """Map brick/tile window coords and slice indices onto the full nD axis list.

    ``display_coords`` are in the order of ``sel.displayed_axes``.
    Non-displayed axes use ``world_to_level_k.map_coordinates`` to
    map slice indices from world space to level-k voxel space.

    Parameters
    ----------
    sel : AxisAlignedSelectionState
        Current selection state.
    ndim : int
        Number of data dimensions.
    display_coords : list[tuple[int, int]]
        Padded coordinate ranges for displayed axes.
    level_shape : tuple[int, ...]
        Shape of the data at this level.
    world_to_level_k : AffineTransform
        Composed world→level-k transform (data-axis order).
    """
    display_pos = {ax: i for i, ax in enumerate(sel.displayed_axes)}

    # Map every non-displayed axis from world to level-k voxel space via the
    # shared world->voxel mapper (round-half-up + clamp).  Any non-displayed
    # axis absent from ``slice_indices`` defaults to world position 0, matching
    # the previous zero-filled-point behaviour.
    sliced_indices = {
        ax: sel.slice_indices.get(ax, 0) for ax in range(ndim) if ax not in display_pos
    }
    voxel_indices = map_world_slice_to_voxel(
        sliced_indices, ndim, world_to_level_k, level_shape
    )

    result: list[int | tuple[int, int]] = []
    for data_axis in range(ndim):
        if data_axis in display_pos:
            result.append(display_coords[display_pos[data_axis]])
        else:
            result.append(voxel_indices[data_axis])
    return tuple(result)


def _block_key_2d_to_padded_coords(
    key: BlockKey2D,
    block_size: int,
    overlap: int,
) -> tuple[int, int, int, int]:
    """Translate a BlockKey2D to padded pixel coordinates.

    Returns y0, x0, y1, x1.  May extend outside store bounds.
    """
    padded = block_size + 2 * overlap
    y0 = key.g0 * block_size - overlap
    x0 = key.g1 * block_size - overlap
    return y0, x0, y0 + padded, x0 + padded


# ---------------------------------------------------------------------------
# GFXMultiscaleImageVisual
# ---------------------------------------------------------------------------


class GFXMultiscaleImageVisual:
    """Render-layer wrapper for one logical multiscale image visual.

    Owns GPU resources (brick caches, LUT textures, pygfx nodes) for both
    2D and 3D rendering.  Implements ``build_slice_request()`` /
    ``build_slice_request_2d()`` and ``on_data_ready()`` /
    ``on_data_ready_2d()`` for use by a ``SliceCoordinator``.

    Parameters
    ----------
    visual_model_id : UUID
        ID of the associated ``MultiscaleImageVisual`` model.
    volume_geometry : MultiscaleBrickLayout3D
        Pre-built metadata cache (level shapes, LOAD grids).
    render_modes : set[str]
        Which nodes to build: ``{"3d"}``, ``{"2d"}``, or
        ``{"2d", "3d"}``.  Non-applicable nodes are ``None``.
    colormap : gfx.TextureMap or None
        Colourmap for the volume.  Defaults to viridis.
    clim : tuple[float, float]
        Contrast limits passed to the material.
    threshold : float
        Isosurface threshold.
    interpolation : str
        Sampler filter (``"linear"`` or ``"nearest"``).
    gpu_budget_bytes_3d : int
        Maximum GPU memory for the 3D brick cache texture.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2D tile cache texture.
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model_id: UUID,
        volume_geometry: MultiscaleBrickLayout3D | None,
        image_geometry_2d: ImageGeometry3D | None,
        render_modes: set[str],
        displayed_axes: tuple[int, ...] | None = None,
        colormap: gfx.TextureMap | None = None,
        clim: tuple[float, float] = (0.0, 1.0),
        threshold: float = 0.5,
        attenuation: float = 1.0,
        render_mode: str = "iso",
        interpolation: str = "nearest",
        gpu_budget_bytes_3d: int = 1 * 1024**3,
        gpu_budget_bytes_2d: int = 64 * 1024**2,
        transform: AffineTransform | None = None,
        full_level_transforms: list[AffineTransform] | None = None,
        full_level_shapes: list[tuple[int, ...]] | None = None,
        aabb_enabled: bool = False,
        aabb_color: str = "#ffffff",
        aabb_line_width: float = 2.0,
        render_order: int = 0,
        pick_write: bool = True,
        paint_max_tiles: int = 512,
    ) -> None:
        self.visual_model_id = visual_model_id

        # ndim of the original data (not the displayed subspace).
        if full_level_shapes is not None:
            self._ndim = len(full_level_shapes[0])
        elif volume_geometry is not None:
            self._ndim = len(volume_geometry.level_shapes[0])
        elif image_geometry_2d is not None:
            self._ndim = len(image_geometry_2d.level_shapes[0])
        else:
            self._ndim = 3

        if transform is None:
            transform = AffineTransform.identity(ndim=self._ndim)
        elif transform.ndim < self._ndim:
            transform = transform.expand_dims(self._ndim)
        self._transform: AffineTransform = transform
        self.render_modes = render_modes
        self._volume_geometry = volume_geometry
        self._image_geometry_2d = image_geometry_2d

        # Full-ndim level transforms for composed world→level-k mapping.
        # These are always in the original data dimensionality, separate
        # from the displayed-subspace transforms on the geometry objects.
        if full_level_transforms is not None:
            self._level_transforms = list(full_level_transforms)
        elif volume_geometry is not None:
            self._level_transforms = volume_geometry.level_transforms
        elif image_geometry_2d is not None:
            self._level_transforms = image_geometry_2d.level_transforms
        else:
            self._level_transforms = []

        # Full-ndim level shapes for _build_axis_selections_multiscale (clamping
        # non-displayed slice indices against the correct dimension).
        if full_level_shapes is not None:
            self._full_level_shapes = list(full_level_shapes)
        elif volume_geometry is not None:
            self._full_level_shapes = list(volume_geometry.level_shapes)
        elif image_geometry_2d is not None:
            self._full_level_shapes = list(image_geometry_2d.level_shapes)
        else:
            self._full_level_shapes = []

        self._world_to_level_transforms = self._build_world_to_level_transforms()
        # Track displayed axes for node matrix updates.
        self._last_displayed_axes: tuple[int, ...] | None = displayed_axes
        self._gpu_budget_bytes = gpu_budget_bytes_3d
        self._frame_number = 0
        self._pending_slot_map: dict[UUID, tuple[BlockKey3D, TileSlot]] = {}
        self._pending_slot_map_2d: dict[UUID, tuple[BlockKey2D, TileSlot2D]] = {}
        self._last_plan_stats: dict = {}

        # Data-ready flags: AABB visibility is suppressed until the first
        # brick/tile batch arrives.
        self._data_ready_3d: bool = False
        self._data_ready_2d: bool = False

        # Cache AABB params so on_aabb_changed can update them.
        self._aabb_enabled: bool = aabb_enabled
        self._aabb_color: str = aabb_color
        self._aabb_line_width: float = aabb_line_width

        # _visible is kept so that visibility set via on_visibility_changed
        # before a lazy-init'd node is built can be applied when it is built.
        # All other appearance values are read from the model at build time
        # via build_node / _lazy_init_*.
        self._visible: bool = True
        # Construction-time config: not appearance, not event-driven, safe to cache.
        self._block_size: int = (
            volume_geometry.block_size
            if volume_geometry is not None
            else image_geometry_2d.block_size
            if image_geometry_2d is not None
            else 32
        )
        self._gpu_budget_bytes_2d: int = gpu_budget_bytes_2d

        # ── 3D GPU resources (only when volume_geometry is provided) ───
        self._block_cache_3d: BlockCache3D | None = None
        self._lut_manager_3d: LutIndirectionManager3D | None = None
        if volume_geometry is not None:
            cache_parameters_3d = compute_block_cache_parameters_3d(
                block_size=volume_geometry.block_size,
                gpu_budget_bytes=gpu_budget_bytes_3d,
                overlap=3,
            )
            self._block_cache_3d = BlockCache3D(cache_parameters=cache_parameters_3d)
            self._lut_manager_3d = LutIndirectionManager3D(
                base_layout=volume_geometry.base_layout,
                n_levels=volume_geometry.n_levels,
                level_scale_vecs_data=volume_geometry._scale_vecs_data,
            )

        # ── 2D GPU resources (only when image_geometry_2d is provided) ─
        self._block_cache_2d: BlockCache2D | None = None
        self._lut_manager_2d: LutIndirectionManager2D | None = None
        self._lut_params_buffer_2d = None
        self._block_scales_buffer_2d = None
        # Sorted (axis_index, world_value) pairs for the most-recently-requested slice.
        # Used by the two-phase LUT rebuild to separate current-slice tiles from
        # old-slice fallback tiles.
        self._current_slice_coord: tuple[tuple[int, int], ...] | None = None
        # Viewport base-grid cell bounds (gy0, gx0, gy1, gx1), half-open, from the
        # most-recent plan.  Used to clip stale-slice background tiles out of the
        # LUT so out-of-view old data is not referenced.  None disables clipping.
        self._current_viewport_cells: tuple[int, int, int, int] | None = None
        # 3D equivalent: embeds non-displayed axis positions into BlockKey3D so
        # that bricks from different slice positions are not treated as cache hits.
        self._current_slice_coord_3d: tuple[tuple[int, int], ...] | None = None
        if image_geometry_2d is not None:
            cache_parameters_2d = compute_block_cache_parameters_2d(
                gpu_budget_bytes=gpu_budget_bytes_2d,
                block_size=image_geometry_2d.block_size,
            )
            self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
            self._lut_manager_2d = LutIndirectionManager2D(
                base_layout=image_geometry_2d.base_layout,
                n_levels=image_geometry_2d.n_levels,
                scale_vecs_data=image_geometry_2d._scale_vecs_data,
            )
            self._lut_params_buffer_2d = build_lut_params_buffer_2d(
                image_geometry_2d.base_layout, cache_parameters_2d
            )
            self._block_scales_buffer_2d = build_block_scales_buffer_2d(
                level_scale_vecs_data=image_geometry_2d._scale_vecs_data,
            )

        # ── 2D paint resources (Phase 2 GPU fast path) ──────────────────
        self._paint_slot_manager: PaintTileSlotManager | None = None
        self._t_paint_cache: gfx.Texture | None = None
        self._t_paint_lut: gfx.Texture | None = None
        self._paint_max_tiles: int = int(paint_max_tiles)
        if image_geometry_2d is not None:
            self._allocate_paint_resources_2d()

        # ── Brick-shader-specific buffers (3D only) ──────────────────────
        self._vol_params_buffer: Buffer | None = None
        self._brick_scales_buffer: Buffer | None = None
        self._norm_size: np.ndarray | None = None
        self._dataset_size: np.ndarray | None = None
        self._norm_size_axes: tuple[int, ...] | None = None
        if volume_geometry is not None:
            # ``volume_geometry.level_shapes[0]`` is in displayed-axis order
            # over the 3 displayed axes (e.g. (D, H, W) when displayed=(0,1,2)).
            # Reverse to shader order (x=W, y=H, z=D) via ``swap_axes``.
            ds = volume_geometry.level_shapes[0]
            self._dataset_size = np.asarray(
                swap_axes(tuple(float(s) for s in ds), (2, 1, 0)),
                dtype=np.float64,
            )
            _check_transform_no_rotation(self._transform)
            # 3D rendering requires 3 displayed axes.  ``displayed_axes`` is
            # set when constructed via ``from_cellier_model``; if a caller
            # uses the raw constructor with a 3D-equal-ndim transform, fall
            # back to the trailing 3 data axes (preserves prior behaviour).
            axes_for_norm = (
                displayed_axes
                if displayed_axes is not None and len(displayed_axes) == 3
                else tuple(range(self._ndim))[-3:]
            )
            self._norm_size = _norm_size_from_transform(
                self._transform, axes_for_norm, self._dataset_size
            )
            self._norm_size_axes = axes_for_norm
            self._vol_params_buffer = build_vol_params_buffer(
                norm_size=self._norm_size,
                dataset_size=self._dataset_size,
                base_layout=volume_geometry.base_layout,
                cache_info=self._block_cache_3d.info,
            )
            self._brick_scales_buffer = build_brick_scales_buffer(
                volume_geometry._scale_vecs_data
            )

        if colormap is None:
            colormap = gfx.cm.viridis

        # ── 3D node ─────────────────────────────────────────────────────
        self.node_3d: gfx.Group | None = None
        self._inner_node_3d: NormSizedVolume | None = None
        self.material_3d: MultiscaleVolumeBrickMaterial | None = None
        self._proxy_tex_3d: gfx.Texture | None = None
        self._aabb_line_3d: gfx.Line | None = None
        if "3d" in render_modes and volume_geometry is not None:
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                attenuation=attenuation,
                pick_write=pick_write,
            )
            self._inner_node_3d = inner
            self.material_3d.render_mode = render_mode
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            # Build AABB line with known geometry (geometry available at construction).
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d.add(self._aabb_line_3d)

        # ── 2D node ─────────────────────────────────────────────────────
        self.node_2d: gfx.Group | None = None
        self._inner_node_2d: gfx.Image | None = None
        self.material_2d: ImageBlockMaterial | None = None
        self._proxy_tex_2d: gfx.Texture | None = None
        self._aabb_line_2d: gfx.Line | None = None
        if "2d" in render_modes and image_geometry_2d is not None:
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap,
                clim=clim,
                interpolation=interpolation,
                pick_write=pick_write,
            )
            self._inner_node_2d = inner
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            # Build AABB line with known geometry (geometry available at construction).
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d.add(self._aabb_line_2d)

        if self.node_3d is not None:
            self.node_3d.render_order = render_order
        if self.node_2d is not None:
            self.node_2d.render_order = render_order

        # Apply node matrices now if displayed_axes are already known.
        # Without this, the matrices stay at identity until the first
        # displayed-axes change, which may never happen in a fixed viewer.
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    @classmethod
    def from_cellier_model(
        cls,
        model: MultiscaleImageVisual,
        level_shapes: list[tuple[int, ...]],
        render_modes: set[str],
        displayed_axes: tuple[int, ...],
    ) -> GFXMultiscaleImageVisual:
        """Build a ``GFXMultiscaleImageVisual`` from a ``MultiscaleImageVisual`` model.

        Block size and GPU budgets are read from ``model.render_config``.
        Interpolation is read from ``model.appearance.interpolation``.

        Parameters
        ----------
        model : MultiscaleImageVisual
            Source visual model.  Must have a ``render_config`` field.
        level_shapes : list[tuple[int, ...]]
            Full nD shape per level, finest first.
        render_modes : set[str]
            Which nodes to build: ``{"3d"}``, ``{"2d"}``, or ``{"2d", "3d"}``.
        displayed_axes : tuple[int, ...]
            The axes currently displayed.  ``len == 3`` means the initial mode
            is 3D; ``len == 2`` means 2D.  In dual-mode (``{"2d", "3d"}``),
            when starting in 3D the 2D geometry is built from
            ``displayed_axes[-2:]``.

        Returns
        -------
        GFXMultiscaleImageVisual
        """
        render_config = model.render_config
        block_size = render_config.block_size
        interpolation = model.appearance.interpolation
        gpu_budget_bytes = render_config.gpu_budget_bytes
        gpu_budget_bytes_2d = render_config.gpu_budget_bytes_2d
        threshold = model.appearance.iso_threshold
        attenuation = model.appearance.attenuation
        render_mode = model.appearance.render_mode

        if len(displayed_axes) == 3:
            axes_3d: tuple[int, ...] | None = displayed_axes
            axes_2d: tuple[int, ...] | None = None
        else:
            axes_3d = None
            axes_2d = displayed_axes

        # Build 3D geometry when 3D rendering is requested and axes are available.
        # ``select_axes`` projects the per-data-axis level shape onto the
        # 3 displayed data axes; ``AffineTransform.select_axes`` does the
        # same for the per-level transforms.  Inputs to VolumeGeometry are
        # therefore in displayed-axis order over the displayed subset.
        volume_geometry: MultiscaleBrickLayout3D | None = None
        if "3d" in render_modes and axes_3d is not None:
            shapes_3d = [select_axes(s, axes_3d) for s in level_shapes]
            transforms_3d = [t.select_axes(axes_3d) for t in model.level_transforms]
            volume_geometry = MultiscaleBrickLayout3D(
                level_shapes=shapes_3d,
                level_transforms=transforms_3d,
                block_size=block_size,
            )

        # Build 2D geometry when 2D rendering is requested and axes are known.
        # When starting in 3D mode, axes_2d is None and geometry is deferred
        # to the first call to rebuild_geometry with len(displayed_axes)==2.
        image_geometry_2d: ImageGeometry3D | None = None
        if "2d" in render_modes and axes_2d is not None:
            shapes_2d = [select_axes(s, axes_2d) for s in level_shapes]
            transforms_2d = [t.select_axes(axes_2d) for t in model.level_transforms]
            image_geometry_2d = ImageGeometry3D(
                level_shapes=shapes_2d,
                block_size=block_size,
                n_levels=len(level_shapes),
                level_transforms=transforms_2d,
            )

        colormap = model.appearance.color_map.to_pygfx(N=256)
        clim = model.appearance.clim

        instance = cls(
            visual_model_id=model.id,
            volume_geometry=volume_geometry,
            image_geometry_2d=image_geometry_2d,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
            colormap=colormap,
            clim=clim,
            threshold=threshold,
            attenuation=attenuation,
            render_mode=render_mode,
            interpolation=interpolation,
            gpu_budget_bytes_3d=gpu_budget_bytes,
            gpu_budget_bytes_2d=gpu_budget_bytes_2d,
            aabb_enabled=model.aabb.enabled,
            aabb_color=model.aabb.color,
            aabb_line_width=model.aabb.line_width,
            render_order=model.appearance.render_order,
            transform=model.transform,
            full_level_transforms=list(model.level_transforms),
            full_level_shapes=list(level_shapes),
            pick_write=model.pick_write,
            paint_max_tiles=render_config.paint_max_tiles,
        )
        for mat in (instance.material_3d, instance.material_2d):
            if mat is not None:
                mat.opacity = model.appearance.opacity
                mat.depth_test = model.appearance.depth_test
                mat.depth_write = model.appearance.depth_write
                mat.depth_compare = model.appearance.depth_compare
                mat.alpha_mode = model.appearance.transparency_mode
        instance._visible = model.appearance.visible
        for node in (instance.node_3d, instance.node_2d):
            if node is not None:
                node.visible = model.appearance.visible
        return instance

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of LOD levels."""
        if self._volume_geometry is not None:
            return self._volume_geometry.n_levels
        if self._image_geometry_2d is not None:
            return self._image_geometry_2d.n_levels
        if self._full_level_shapes:
            return len(self._full_level_shapes)
        raise RuntimeError("No geometry available")

    # ── Node selection ───────────────────────────────────────────────

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Rebuild geometry if needed and return the node for *displayed_axes*.

        Calls ``self.rebuild_geometry`` internally using the already-stored
        ``_full_level_shapes``.  The controller no longer needs to supply
        level shapes or call ``rebuild_geometry`` directly.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            The new set of displayed axes.

        Returns
        -------
        gfx.Group or None
            The new active node after geometry rebuild, or ``None`` if the
            required render mode was not built.
        """
        _old_node, new_node = self.rebuild_geometry(
            self._full_level_shapes, displayed_axes
        )
        return new_node

    # ── Geometry rebuild ─────────────────────────────────────────────

    def rebuild_geometry(
        self,
        level_shapes: list[tuple[int, ...]],
        displayed_axes: tuple[int, ...],
    ) -> tuple[gfx.WorldObject | None, gfx.WorldObject | None]:
        """Rebuild geometry and GPU resources after a displayed_axes change.

        Returns ``(old_node, new_node)`` for the active node so the caller
        can swap it in the scene graph.  Derives the 3D and 2D axis subsets
        from ``ndim`` (last-three and last-two axes respectively).

        Parameters
        ----------
        level_shapes : list[tuple[int, ...]]
            Full nD shape per level from the data store.
        displayed_axes : tuple[int, ...]
            The new set of displayed axes.  ``len == 3`` rebuilds the 3D
            geometry using exactly those axes; ``len == 2`` rebuilds the 2D
            geometry.  Also determines which node is the active one to swap.

        Returns
        -------
        tuple[old_node, new_node]
            The previous and replacement pygfx nodes (may be ``None``).
        """
        self._full_level_shapes = list(level_shapes)
        self._last_displayed_axes = displayed_axes

        old_node: gfx.WorldObject | None = None
        new_node: gfx.WorldObject | None = None

        if "3d" in self.render_modes and len(displayed_axes) == 3:
            old_node = self.node_3d
            if self._volume_geometry is None:
                self._lazy_init_3d(displayed_axes)
            else:
                shapes_3d = [
                    tuple(s[ax] for ax in displayed_axes) for s in level_shapes
                ]
                if shapes_3d != self._volume_geometry.level_shapes:
                    self._volume_geometry.update(shapes_3d)
                    self._rebuild_3d_resources()
            new_node = self.node_3d

        if "2d" in self.render_modes and len(displayed_axes) == 2:
            old_node = self.node_2d
            if self._image_geometry_2d is None:
                self._lazy_init_2d(displayed_axes)
            else:
                shapes_2d_full = [
                    tuple(s[ax] for ax in displayed_axes) for s in level_shapes
                ]
                shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
                if shapes_2d != self._image_geometry_2d.level_shapes:
                    self._image_geometry_2d.update(shapes_2d)
                    self._rebuild_2d_resources()
            new_node = self.node_2d

        return old_node, new_node

    # ── GFXVisual protocol ──────────────────────────────────────────────

    def has_node(self, mode: str) -> bool:
        """Return True if the node for *mode* has already been built."""
        if mode == "3d":
            return self.node_3d is not None
        return self.node_2d is not None

    def get_node(self, mode: str) -> gfx.WorldObject | None:
        """Return the already-built node for *mode*, or None if not built."""
        if mode == "3d":
            return self.node_3d
        return self.node_2d

    def build_node(
        self,
        mode: str,
        visual_model,
        displayed_axes: tuple[int, ...],
        level_shapes: list[tuple[int, ...]],
        level_transforms: list,
    ) -> gfx.WorldObject | None:
        """Build the 2D or 3D node for the first time.

        Reads appearance and render_config from *visual_model* at call time.
        Does NOT store *visual_model*.
        """
        self._full_level_shapes = list(level_shapes)
        if mode == "3d":
            if self._volume_geometry is None:
                self._lazy_init_3d(displayed_axes, visual_model)
            return self.node_3d
        else:
            if self._image_geometry_2d is None:
                self._lazy_init_2d(displayed_axes, visual_model)
            return self.node_2d

    def rebuild_node_geometry(
        self,
        mode: str,
        displayed_axes: tuple[int, ...],
        level_shapes: list[tuple[int, ...]],
        level_transforms: list,
    ) -> gfx.WorldObject | None:
        """Rebuild geometry on an already-built node after a dims change."""
        self._full_level_shapes = list(level_shapes)
        self._last_displayed_axes = displayed_axes
        if mode == "3d" and self._volume_geometry is not None:
            shapes_3d = [tuple(s[ax] for ax in displayed_axes) for s in level_shapes]
            if shapes_3d != self._volume_geometry.level_shapes:
                self._volume_geometry.update(shapes_3d)
                self._rebuild_3d_resources()
            return self.node_3d
        if mode == "2d" and self._image_geometry_2d is not None:
            shapes_2d_full = [
                tuple(s[ax] for ax in displayed_axes) for s in level_shapes
            ]
            shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
            if shapes_2d != self._image_geometry_2d.level_shapes:
                self._image_geometry_2d.update(shapes_2d)
                self._rebuild_2d_resources()
            return self.node_2d
        return None

    def on_stacked_axes_changed(self, stacked_axes: tuple[int, ...]) -> None:
        pass

    def _lazy_init_3d(self, displayed_axes: tuple[int, ...], visual_model=None) -> None:
        """Build 3D GPU resources on first entry into 3D mode.

        When *visual_model* is provided (controller-driven path), appearance
        values are read from it at call time and not stored.  When it is
        ``None`` (e.g. multichannel slot path), safe defaults are used and
        the caller is responsible for applying per-channel appearance afterward.

        Sets ``node_3d.local.matrix`` directly — does NOT call
        ``_update_node_matrix`` so ``node_2d`` (if already built) is left
        unchanged.
        """
        if visual_model is not None:
            threshold = float(visual_model.appearance.iso_threshold)
            attenuation = float(visual_model.appearance.attenuation)
            render_mode = visual_model.appearance.render_mode
            pick_write = visual_model.pick_write
        else:
            threshold = 0.5
            attenuation = 1.0
            render_mode = "mip"
            pick_write = False

        shapes_3d = [select_axes(s, displayed_axes) for s in self._full_level_shapes]
        transforms_3d = [t.select_axes(displayed_axes) for t in self._level_transforms]
        self._volume_geometry = MultiscaleBrickLayout3D(
            level_shapes=shapes_3d,
            level_transforms=transforms_3d,
            block_size=self._block_size,
        )
        cache_parameters_3d = compute_block_cache_parameters_3d(
            block_size=self._volume_geometry.block_size,
            gpu_budget_bytes=self._gpu_budget_bytes,
            overlap=3,
        )
        self._block_cache_3d = BlockCache3D(cache_parameters=cache_parameters_3d)
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=self._volume_geometry.base_layout,
            n_levels=self._volume_geometry.n_levels,
            level_scale_vecs_data=self._volume_geometry._scale_vecs_data,
        )
        ds = self._volume_geometry.level_shapes[0]
        self._dataset_size = np.asarray(
            swap_axes(tuple(float(s) for s in ds), (2, 1, 0)),
            dtype=np.float64,
        )
        _check_transform_no_rotation(self._transform)
        self._norm_size = _norm_size_from_transform(
            self._transform, displayed_axes, self._dataset_size
        )
        self._norm_size_axes = displayed_axes
        self._vol_params_buffer = build_vol_params_buffer(
            norm_size=self._norm_size,
            dataset_size=self._dataset_size,
            base_layout=self._volume_geometry.base_layout,
            cache_info=self._block_cache_3d.info,
        )
        self._brick_scales_buffer = build_brick_scales_buffer(
            self._volume_geometry._scale_vecs_data
        )
        if self.material_2d is not None:
            colormap = self.material_2d.map
            clim = self.material_2d.clim
        else:
            colormap = gfx.cm.viridis
            clim = (0.0, 1.0)
        inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
            colormap=colormap,
            clim=clim,
            threshold=threshold,
            attenuation=attenuation,
            pick_write=pick_write,
        )
        self.material_3d.render_mode = render_mode
        self._inner_node_3d = inner
        self._aabb_line_3d = self._build_aabb_line_3d()
        self.node_3d = gfx.Group()
        self.node_3d.add(inner)
        self.node_3d.add(self._aabb_line_3d)
        sub = self._transform.select_axes(displayed_axes)
        data_to_world = _pygfx_matrix(sub)
        m = compose_world_transform(data_to_world, self._dataset_size, self._norm_size)
        self.node_3d.local.matrix = m
        if visual_model is not None:
            self.material_3d.opacity = visual_model.appearance.opacity
            self.material_3d.depth_test = visual_model.appearance.depth_test
            self.material_3d.depth_write = visual_model.appearance.depth_write
            self.material_3d.depth_compare = visual_model.appearance.depth_compare
            self.material_3d.alpha_mode = visual_model.appearance.transparency_mode
            self.node_3d.visible = visual_model.appearance.visible
        else:
            self.node_3d.visible = self._visible

    def _lazy_init_2d(self, displayed_axes: tuple[int, ...], visual_model=None) -> None:
        """Build 2D GPU resources on first entry into 2D mode.

        When *visual_model* is provided (controller-driven path), appearance
        values are read from it at call time and not stored.  When it is
        ``None`` (e.g. multichannel slot path), safe defaults are used and
        the caller is responsible for applying per-channel appearance afterward.

        Sets ``node_2d.local.matrix`` directly — does NOT call
        ``_update_node_matrix`` so ``node_3d`` (if already built) is left
        unchanged.
        """
        if visual_model is not None:
            interpolation = visual_model.appearance.interpolation
            pick_write = visual_model.pick_write
        else:
            interpolation = "linear"
            pick_write = False

        shapes_2d_full = [
            select_axes(s, displayed_axes) for s in self._full_level_shapes
        ]
        shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
        transforms_2d = [t.select_axes(displayed_axes) for t in self._level_transforms]
        self._image_geometry_2d = ImageGeometry3D(
            level_shapes=shapes_2d,
            block_size=self._block_size,
            n_levels=len(self._full_level_shapes),
            level_transforms=transforms_2d,
        )
        cache_parameters_2d = compute_block_cache_parameters_2d(
            gpu_budget_bytes=self._gpu_budget_bytes_2d,
            block_size=self._image_geometry_2d.block_size,
        )
        self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=self._image_geometry_2d.base_layout,
            n_levels=self._image_geometry_2d.n_levels,
            scale_vecs_data=self._image_geometry_2d._scale_vecs_data,
        )
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            self._image_geometry_2d.base_layout, cache_parameters_2d
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            level_scale_vecs_data=self._image_geometry_2d._scale_vecs_data,
        )
        self._allocate_paint_resources_2d()
        if self.material_3d is not None:
            colormap = self.material_3d.map
            clim = self.material_3d.clim
        else:
            colormap = gfx.cm.viridis
            clim = (0.0, 1.0)
        inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
            colormap=colormap,
            clim=clim,
            interpolation=interpolation,
            pick_write=pick_write,
        )
        self._inner_node_2d = inner
        self._aabb_line_2d = self._build_aabb_line_2d()
        self.node_2d = gfx.Group()
        self.node_2d.add(inner)
        self.node_2d.add(self._aabb_line_2d)
        sub = self._transform.select_axes(displayed_axes)
        self.node_2d.local.matrix = _pygfx_matrix(sub)
        if visual_model is not None:
            self.material_2d.opacity = visual_model.appearance.opacity
            self.material_2d.depth_test = visual_model.appearance.depth_test
            self.material_2d.depth_write = visual_model.appearance.depth_write
            self.material_2d.depth_compare = visual_model.appearance.depth_compare
            self.material_2d.alpha_mode = visual_model.appearance.transparency_mode
            self.node_2d.visible = visual_model.appearance.visible
        else:
            self.node_2d.visible = self._visible

    def _rebuild_3d_resources(self) -> None:
        """Rebuild 3D GPU resources after geometry update."""
        geo = self._volume_geometry
        # Clear cache
        self._block_cache_3d.tile_manager.release_all_in_flight()
        # Rebuild LUT manager
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=geo.base_layout,
            n_levels=geo.n_levels,
            level_scale_vecs_data=geo._scale_vecs_data,
        )
        # Rebuild node preserving current appearance
        if self.node_3d is not None:
            colormap = self.material_3d.map
            clim = self.material_3d.clim
            threshold = self.material_3d.threshold
            attenuation = self.material_3d.attenuation
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                attenuation=attenuation,
            )
            self._inner_node_3d = inner
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            self.node_3d.add(self._aabb_line_3d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map = {}

    def _rebuild_2d_resources(self) -> None:
        """Rebuild 2D GPU resources after geometry update."""
        geo2d = self._image_geometry_2d
        # Clear cache
        self._block_cache_2d.tile_manager.release_all_in_flight()
        # Rebuild LUT manager
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=geo2d.base_layout,
            n_levels=geo2d.n_levels,
            scale_vecs_data=geo2d._scale_vecs_data,
        )
        # Rebuild param buffers
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            geo2d.base_layout, self._block_cache_2d.info
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            level_scale_vecs_data=geo2d._scale_vecs_data,
        )
        # Re-allocate paint resources so they track the new geometry.
        self._allocate_paint_resources_2d()
        # Rebuild node preserving current appearance
        if self.node_2d is not None:
            colormap = self.material_2d.map
            clim = self.material_2d.clim
            interpolation = self.material_2d.interpolation
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )
            self._inner_node_2d = inner
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            self.node_2d.add(self._aabb_line_2d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map_2d = {}

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Recompute and apply the pygfx node matrix for *displayed_axes*."""
        self._last_displayed_axes = displayed_axes
        sub = self._transform.select_axes(displayed_axes)

        if self.node_3d is not None:
            data_to_world = _pygfx_matrix(sub)
            m = compose_world_transform(
                data_to_world, self._dataset_size, self._norm_size
            )
            self.node_3d.local.matrix = m

        if self.node_2d is not None:
            self.node_2d.local.matrix = _pygfx_matrix(sub)

    def _build_world_to_level_transforms(
        self,
    ) -> list[AffineTransform]:
        """Precompute composed world→level-k transforms.

        All transforms operate in data-axis order (z, y, x).
        ``map_coordinates`` on a world-space point gives the level-k
        voxel index.

        Composition: ``inv_level @ inv_visual`` so that
        ``map_coordinates(world_pt)`` applies inv_visual first
        (world → level-0), then inv_level (level-0 → level-k).
        """
        inv_visual = AffineTransform(matrix=self._transform.inverse_matrix)
        result: list[AffineTransform] = []
        for lt in self._level_transforms:
            inv_level = AffineTransform(matrix=lt.inverse_matrix)
            # world → level-0 → level-k
            composed = inv_level @ inv_visual
            result.append(composed)
        return result

    # ── 3D planning helpers ───────────────────────────────────────────

    def _plan_bricks(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        fov_y_rad: float,
        screen_height_px: float,
        lod_bias: float,
        force_level: int | None,
    ) -> np.ndarray:
        """Run LOD selection, distance sort, frustum cull, and budget truncation.

        Does **not** call ``stage()`` on the block cache.  Call
        ``_materialize_brick_requests`` to stage and build ``ChunkRequest``
        objects.

        Parameters
        ----------
        camera_pos_world : np.ndarray
            Camera position in world space.
        frustum_corners_world : np.ndarray or None
            Frustum corner points for culling, or ``None`` to skip culling.
        fov_y_rad : float
            Vertical field of view in radians.
        screen_height_px : float
            Viewport height in pixels.
        lod_bias : float
            Bias applied to LOD distance thresholds. Values > 1 prefer coarser
            levels; values < 1 prefer finer. Clamped to a minimum of 1e-6.
        force_level : int or None
            Override level; ``None`` lets LOD selection choose.

        Returns
        -------
        np.ndarray
            The planned brick array after all filtering (may be empty).
        """
        geo = self._volume_geometry
        if geo is None or self._block_cache_3d is None:
            return np.empty((0, 4), dtype=np.int64)

        if self._last_displayed_axes is None:
            raise RuntimeError("_plan_bricks requires displayed_axes to be set.")
        if len(self._last_displayed_axes) != 3:
            raise ValueError(
                f"_plan_bricks expects 3D display, got "
                f"displayed_axes={self._last_displayed_axes}"
            )

        sub_3d = self._transform.select_axes(self._last_displayed_axes)
        cam_zyx = camera_pos_world[[2, 1, 0]]
        cam_data_zyx = sub_3d.imap_coordinates(cam_zyx.reshape(1, -1)).flatten()
        camera_pos_data = cam_data_zyx[[2, 1, 0]]

        if force_level is None and fov_y_rad > 0:
            safe_bias = max(lod_bias, 1e-6)
            focal_half_height_world = (screen_height_px / 2.0) / np.tan(fov_y_rad / 2.0)
            thresholds: list[float] | None = [
                geo._level_scale_factors[k - 1] * focal_half_height_world / safe_bias
                for k in range(1, geo.n_levels)
            ]
        else:
            thresholds = None

        if frustum_corners_world is not None:
            corners_flat = frustum_corners_world.reshape(-1, 3)
            corners_flat_zyx = corners_flat[:, [2, 1, 0]]
            corners_data_flat_zyx = sub_3d.imap_coordinates(corners_flat_zyx)
            corners_data = corners_data_flat_zyx[:, [2, 1, 0]].reshape(
                frustum_corners_world.shape
            )
            frustum_planes = frustum_planes_from_corners(corners_data)
        else:
            frustum_planes = None

        # 1. LOD selection
        if force_level is not None:
            brick_arr = select_levels_arr_forced(
                geo.base_layout, force_level, geo._level_grids
            )
        else:
            brick_arr = select_levels_from_cache(
                geo._level_grids,
                geo.n_levels,
                camera_pos_data,
                thresholds=thresholds,
                base_layout=geo.base_layout,
            )

        # 2. Distance sort
        brick_arr = sort_arr_by_distance(
            brick_arr,
            camera_pos_data,
            geo.block_size,
            scale_vecs_shader=geo._scale_arr_shader,
            translation_vecs_shader=geo._translation_arr_shader,
        )

        # 3. Frustum cull
        if frustum_planes is not None:
            brick_arr, _ = bricks_in_frustum_arr(
                brick_arr,
                geo.block_size,
                frustum_planes,
                level_scale_arr_shader=geo._scale_arr_shader,
                level_translation_arr_shader=geo._translation_arr_shader,
            )

        # 4. Budget truncation
        n_budget = self._block_cache_3d.info.n_slots - 1
        if len(brick_arr) > n_budget:
            _GPU_LOGGER.warning(
                "budget_truncation  pre=%d  budget=%d  dropped=%d",
                len(brick_arr),
                n_budget,
                len(brick_arr) - n_budget,
            )
            brick_arr = brick_arr[:n_budget]

        return brick_arr

    def _materialize_brick_requests(
        self,
        brick_arr: np.ndarray,
        slice_request_id: UUID,
        dims_state: DimsState | None,
        fill: dict[int, int] | None = None,
        slice_coord: tuple[tuple[int, int], ...] = (),
    ) -> list[ChunkRequest]:
        """Stage ``brick_arr`` in ``self._block_cache_3d`` and build ChunkRequests.

        Also populates ``self._pending_slot_map``.

        Parameters
        ----------
        brick_arr : np.ndarray
            Output of ``_plan_bricks``.
        slice_request_id : UUID
            Shared ID for all requests produced in this batch.
        dims_state : DimsState or None
            Used to map slice indices to data-space coords for non-displayed
            axes.  ``None`` → only display ranges are used.
        fill : dict[int, int] or None
            Axis index → scalar value overrides applied after
            ``_build_axis_selections_multiscale``.  For multichannel use, pass
            ``{channel_axis: channel_index}``.
        slice_coord : tuple of (axis_index, world_value) pairs
            Sorted slice-position encoding to embed in every ``BlockKey3D``
            so that bricks from different non-displayed-axis positions are
            not treated as cache hits.
        """
        geo = self._volume_geometry
        sorted_required = arr_to_brick_keys(brick_arr, slice_coord=slice_coord)
        fill_plan = self._block_cache_3d.tile_manager.stage(
            sorted_required, self._frame_number
        )

        if not fill_plan:
            self._lut_manager_3d.rebuild(
                self._block_cache_3d.tile_manager,
                current_slice_coord=self._current_slice_coord_3d,
            )

        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map = {}

        for brick_key, slot in fill_plan:
            chunk_id = uuid4()
            z0, y0, x0, z1, y1, x1 = _brick_key_to_padded_coords(
                brick_key, geo.block_size, self._block_cache_3d.info.overlap
            )
            level_index = brick_key.level - 1
            display_coords = [(z0, z1), (y0, y1), (x0, x1)]
            if dims_state is not None:
                ndim = len(dims_state.axis_labels)
                axis_selections = _build_axis_selections_multiscale(
                    dims_state.selection,
                    ndim,
                    display_coords,
                    level_shape=self._full_level_shapes[level_index],
                    world_to_level_k=self._world_to_level_transforms[level_index],
                )
            else:
                axis_selections = tuple(display_coords)

            if fill:
                axis_selections = tuple(
                    fill[ax] if ax in fill else val
                    for ax, val in enumerate(axis_selections)
                )

            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_request_id,
                scale_index=brick_key.level - 1,
                axis_selections=axis_selections,
            )
            chunk_requests.append(req)
            self._pending_slot_map[chunk_id] = (brick_key, slot)

        return chunk_requests

    # ── 2D planning helpers ───────────────────────────────────────────

    def _viewport_cells_from_view_bounds(
        self,
        view_min: np.ndarray | None,
        view_max: np.ndarray | None,
        block_size: int,
    ) -> tuple[int, int, int, int] | None:
        """Convert a data-space viewport AABB to base-grid cell bounds.

        Parameters
        ----------
        view_min, view_max : ndarray, shape (2,) or None
            Viewport AABB in level-0 voxel space, ``(gx, gy)`` order (as
            produced by the culling block).  ``None`` returns ``None``.
        block_size : int
            Finest-level tile side length in voxels.

        Returns
        -------
        tuple[int, int, int, int] or None
            Half-open cell bounds ``(gy0, gx0, gy1, gx1)`` clamped to the base
            grid, or ``None`` when no viewport is available.
        """
        if view_min is None or view_max is None:
            return None
        gh_grid, gw_grid = self._image_geometry_2d.base_layout.grid_dims
        cx0 = max(0, int(np.floor(view_min[0] / block_size)))
        cx1 = min(gw_grid, int(np.ceil(view_max[0] / block_size)))
        cy0 = max(0, int(np.floor(view_min[1] / block_size)))
        cy1 = min(gh_grid, int(np.ceil(view_max[1] / block_size)))
        return (cy0, cx0, cy1, cx1)

    def _plan_tiles_2d(
        self,
        camera_pos_world: np.ndarray,
        viewport_width_px: float,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
        lod_bias: float,
        force_level: int | None,
        use_culling: bool,
    ) -> tuple[dict, int]:
        """Run 2D LOD selection, sort, viewport cull, and budget truncation.

        Updates ``self._current_slice_coord`` as a side effect.
        Does **not** stage in the block cache.

        Returns
        -------
        tuple[dict, int]
            ``(required, target_level)`` where ``required`` maps
            ``BlockKey2D → None`` and ``target_level`` is the coarsest level
            requested (used for eviction).
        """
        geo2d = self._image_geometry_2d
        if geo2d is None or self._block_cache_2d is None:
            return {}, 1

        block_size = geo2d.block_size
        n_levels = geo2d.n_levels

        displayed = self._last_displayed_axes
        sub_2d = self._transform.select_axes(displayed)

        nd2 = sub_2d.ndim
        world_units_per_voxel_2d = np.abs(np.diag(sub_2d.matrix[:nd2, :nd2]))
        world_to_voxel_scale_2d = float(
            np.prod(1.0 / world_units_per_voxel_2d) ** (1.0 / nd2)
        )
        voxel_width = world_width * world_to_voxel_scale_2d

        camera_pos_2d = sub_2d.imap_coordinates(
            camera_pos_world[[1, 0]].reshape(1, -1)
        ).flatten()[[1, 0]]
        camera_pos = np.array(
            [camera_pos_2d[0], camera_pos_2d[1], 0.0], dtype=np.float32
        )

        if use_culling and view_min_world is not None and view_max_world is not None:
            cx = float(camera_pos_world[0])
            cy = float(camera_pos_world[1])
            half_w = world_width / 2.0
            half_h = (float(view_max_world[1]) - float(view_min_world[1])) / 2.0
            corners_world_2d = np.array(
                [
                    [cy - half_h, cx - half_w],
                    [cy - half_h, cx + half_w],
                    [cy + half_h, cx + half_w],
                    [cy + half_h, cx - half_w],
                ],
                dtype=np.float32,
            )
            corners_data_2d = sub_2d.imap_coordinates(corners_world_2d)[:, [1, 0]]
            view_min = corners_data_2d.min(axis=0)
            view_max = corners_data_2d.max(axis=0)
        else:
            view_min = None
            view_max = None

        self._current_viewport_cells = self._viewport_cells_from_view_bounds(
            view_min, view_max, block_size
        )

        # 1. LOD selection
        tile_arr = select_lod_2d(
            geo2d._level_grids,
            n_levels,
            viewport_width_px=viewport_width_px,
            voxel_width=voxel_width,
            lod_bias=lod_bias,
            force_level=force_level,
            level_scale_factors=geo2d._level_scale_factors,
        )

        # 2. Distance sort
        tile_arr = sort_tiles_by_distance_2d(
            tile_arr,
            camera_pos,
            block_size,
            level_scale_arr_shader=geo2d._scale_arr_shader,
            level_translation_arr_shader=geo2d._translation_arr_shader,
        )

        required = arr_to_block_keys_2d(tile_arr, slice_coord=self._current_slice_coord)

        # 3. Viewport culling
        if use_culling and view_min is not None and view_max is not None:
            required, _ = viewport_cull_2d(
                required,
                block_size,
                view_min,
                view_max,
                level_scale_arr_shader=geo2d._scale_arr_shader,
                level_translation_arr_shader=geo2d._translation_arr_shader,
            )

        # 4. Budget truncation
        n_budget = self._block_cache_2d.info.n_slots - 1
        if len(required) > n_budget:
            keys_to_keep = list(required.keys())[:n_budget]
            required = {k: required[k] for k in keys_to_keep}

        target_level = int(tile_arr[0, 0]) if len(tile_arr) > 0 else 1
        return required, target_level

    def _materialize_tile_requests(
        self,
        required: dict,
        target_level: int,
        dims_state: DimsState,
        slice_request_id: UUID,
        fill: dict[int, int] | None = None,
    ) -> list[ChunkRequest]:
        """Evict, stage, and build 2D tile ChunkRequests.

        Also populates ``self._pending_slot_map_2d``.

        Parameters
        ----------
        required : dict
            Mapping from ``BlockKey2D`` to ``None``.
        target_level : int
            Coarsest LOD level — tiles finer than this are evicted first.
        dims_state : DimsState
            For building axis_selections.
        slice_request_id : UUID
            Shared ID embedded in every returned ``ChunkRequest``.
        fill : dict[int, int] or None
            Axis index → scalar overrides (e.g. ``{channel_axis: ch_idx}``).
        """
        geo2d = self._image_geometry_2d
        block_size = geo2d.block_size
        overlap = self._block_cache_2d.info.overlap

        self._block_cache_2d.tile_manager.evict_finer_than(target_level)
        fill_plan = self._block_cache_2d.tile_manager.stage(
            required, self._frame_number
        )

        if not fill_plan:
            self._lut_manager_2d.rebuild(
                self._block_cache_2d.tile_manager,
                current_slice_coord=self._current_slice_coord,
                viewport_cells=self._current_viewport_cells,
            )

        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map_2d = {}

        ndim = len(dims_state.axis_labels)
        sel = dims_state.selection

        for tile_key, slot in fill_plan:
            chunk_id = uuid4()
            y0, x0, y1, x1 = _block_key_2d_to_padded_coords(
                tile_key, block_size, overlap
            )
            level_index = tile_key.level - 1
            display_coords = [(y0, y1), (x0, x1)]
            axis_selections = _build_axis_selections_multiscale(
                sel,
                ndim,
                display_coords,
                level_shape=self._full_level_shapes[level_index],
                world_to_level_k=self._world_to_level_transforms[level_index],
            )

            if fill:
                axis_selections = tuple(
                    fill[ax] if ax in fill else val
                    for ax, val in enumerate(axis_selections)
                )

            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_request_id,
                scale_index=tile_key.level - 1,
                axis_selections=axis_selections,
            )
            chunk_requests.append(req)
            self._pending_slot_map_2d[chunk_id] = (tile_key, slot)

        return chunk_requests

    # ── 3D SliceCoordinator interface ──────────────────────────────────

    def build_slice_request(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        fov_y_rad: float,
        screen_height_px: float,
        lod_bias: float = 1.0,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        """Run the synchronous 3D planning phase and return ChunkRequests.

        World-space inputs are transformed to data space before planning.

        Pipeline: LOAD select -> distance sort -> optional frustum cull ->
        budget cap -> stage() -> build ``ChunkRequest`` objects.

        Parameters
        ----------
        camera_pos_world : np.ndarray
            Camera position in world coordinates.
        frustum_corners_world : np.ndarray or None
            Frustum corner points in world coordinates. ``None`` disables
            frustum culling.
        fov_y_rad : float
            Vertical field-of-view in radians (perspective camera).
        screen_height_px : float
            Viewport height in logical pixels.
        lod_bias : float
            Bias applied to LOD distance thresholds. Values > 1 prefer coarser
            levels; values < 1 prefer finer. Clamped to a minimum of 1e-6.
        dims_state : DimsState or None
            Current dimension state.
        force_level : int or None
            Override LOD level.

        Returns
        -------
        list[ChunkRequest]
        """
        t_plan_start = time.perf_counter()
        self._frame_number += 1
        geo = self._volume_geometry
        if geo is None or self._block_cache_3d is None:
            return []

        # Record the current slice coordinate so non-displayed axis positions
        # are embedded in every BlockKey3D (mirrors the 2D path).
        if dims_state is not None:
            self._current_slice_coord_3d = tuple(
                sorted(dims_state.selection.slice_indices.items())
            )

        # Lazy node matrix update when displayed axes change.
        if dims_state is not None:
            displayed = dims_state.selection.displayed_axes
            if displayed != self._last_displayed_axes:
                self._update_node_matrix(displayed)

        if self._last_displayed_axes is None:
            raise RuntimeError(
                "build_slice_request requires displayed_axes to be set. "
                "Ensure the dims_state has been propagated to the visual."
            )
        if len(self._last_displayed_axes) != 3:
            raise ValueError(
                f"build_slice_request expects 3D display, got "
                f"displayed_axes={self._last_displayed_axes}"
            )

        # Steps 1-4: LOD selection, distance sort, frustum cull, budget truncation.
        brick_arr = self._plan_bricks(
            camera_pos_world,
            frustum_corners_world,
            fov_y_rad,
            screen_height_px,
            lod_bias,
            force_level,
        )
        if not len(brick_arr):
            return []

        # 5. Stage: find cache hits/misses, reserve slots for misses.
        # When force_level is set every brick is at a single level; evict
        # finer bricks proactively so their slots are immediately reusable.
        if force_level is not None:
            self._block_cache_3d.tile_manager.evict_finer_than(force_level)
        t0 = time.perf_counter()
        sorted_required = arr_to_brick_keys(
            brick_arr, slice_coord=self._current_slice_coord_3d or ()
        )
        fill_plan = self._block_cache_3d.tile_manager.stage(
            sorted_required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        # Rebuild the LUT immediately so that:
        #   • reserve-promoted bricks (moved to tilemap by stage()) are
        #     visible right away, not only after the first on_data_ready batch.
        #   • when there are no misses at all, on_data_ready never fires so
        #     this is the only rebuild opportunity for that plan.
        # When there are misses, on_data_ready will rebuild again after each
        # arriving batch, progressively filling the remaining positions.
        #
        # COUPLING: the 2D path (build_slice_request_2d) rebuilds only when
        # fill_plan is empty, because TileManager2D has no reserve tier and so
        # nothing is promoted into tilemap during stage().  See the longer note
        # there before changing either rebuild policy.
        self._lut_manager_3d.rebuild(
            self._block_cache_3d.tile_manager,
            current_slice_coord=self._current_slice_coord_3d,
        )

        # 6. Build ChunkRequests and populate the pending slot map.
        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map = {}

        for brick_key, slot in fill_plan:
            chunk_id = uuid4()
            z0, y0, x0, z1, y1, x1 = _brick_key_to_padded_coords(
                brick_key, geo.block_size, self._block_cache_3d.info.overlap
            )
            level_index = brick_key.level - 1
            display_coords = [(z0, z1), (y0, y1), (x0, x1)]
            if dims_state is not None:
                ndim = len(dims_state.axis_labels)
                axis_selections = _build_axis_selections_multiscale(
                    dims_state.selection,
                    ndim,
                    display_coords,
                    level_shape=self._full_level_shapes[level_index],
                    world_to_level_k=self._world_to_level_transforms[level_index],
                )
            else:
                axis_selections = tuple(display_coords)
            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_id,
                scale_index=brick_key.level - 1,
                axis_selections=axis_selections,
            )
            chunk_requests.append(req)
            self._pending_slot_map[chunk_id] = (brick_key, slot)

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        self._last_plan_stats = stats = {
            "hits": len(sorted_required) - len(fill_plan),
            "misses": len(fill_plan),
            "fills": len(fill_plan),
            "total_required": len(brick_arr),
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }

        _PERF_LOGGER.info(
            "[frame %d]  plan=%.1fms  stage=%.1fms  |  required=%d  hits=%d  misses=%d",
            self._frame_number,
            plan_total_ms,
            stage_ms,
            stats["total_required"],
            stats["hits"],
            stats["misses"],
        )

        if len(brick_arr):
            level_col = brick_arr[:, 0]
            requested_by_level = {
                int(lv): int((level_col == lv).sum()) for lv in np.unique(level_col)
            }
        else:
            requested_by_level = {}
        _PERF_LOGGER.debug(
            "[frame %d]  requested_by_level=%s",
            self._frame_number,
            requested_by_level,
        )

        return chunk_requests

    def on_data_ready(
        self,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        """Commit an arriving batch of 3D bricks to the GPU cache."""
        for req, data in batch:
            entry = self._pending_slot_map.get(req.chunk_request_id)
            if entry is None:
                continue
            brick_key, slot = entry
            slot.brick_max = float(data.max())
            self._block_cache_3d.write_brick(slot, data, key=brick_key)
            self._block_cache_3d.tile_manager.commit(brick_key, slot)

        _GPU_LOGGER.info(
            "gpu_flush  bricks_in_batch=%d  resident=%d",
            len(batch),
            self._block_cache_3d.n_resident,
        )

        self._lut_manager_3d.rebuild(
            self._block_cache_3d.tile_manager,
            current_slice_coord=self._current_slice_coord_3d,
        )

        _GPU_LOGGER.info(
            "lut_rebuilt  resident=%d  frame=%d",
            self._block_cache_3d.n_resident,
            self._frame_number,
        )

        # On first brick batch, reveal the AABB line if enabled.
        if not self._data_ready_3d and self._aabb_line_3d is not None:
            self._data_ready_3d = True
            self._aabb_line_3d.visible = self._aabb_enabled

    def cancel_pending(self) -> None:
        """Release all in-flight 3D slots."""
        if self._block_cache_3d is None:
            return
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._pending_slot_map = {}

    # ── 2D SliceCoordinator interface ──────────────────────────────────

    def build_slice_request_2d(
        self,
        camera_pos_world: np.ndarray,
        viewport_width_px: float,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
        dims_state: DimsState,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> list[ChunkRequest]:
        """Run the synchronous 2D planning phase and return ChunkRequests.

        World-space inputs are transformed to data space before planning.

        Pipeline: LOD select -> distance sort -> optional viewport cull ->
        budget cap -> stage() -> build ``ChunkRequest`` objects.

        Parameters
        ----------
        camera_pos_world : ndarray, shape (3,)
            Camera world-space position ``(x, y, z)``.
        viewport_width_px : float
            Viewport width in logical pixels.
        world_width : float
            Visible world width in world units.
        view_min_world : ndarray, shape (2,) or None
            Viewport AABB minimum ``(x, y)`` in world space.
            ``None`` disables culling.
        view_max_world : ndarray, shape (2,) or None
            Viewport AABB maximum ``(x, y)`` in world space.
            ``None`` disables culling.
        dims_state : DimsState
            Current dimension display state.
        lod_bias : float
            Multiplicative LOD bias.
        force_level : int or None
            Override: all tiles assigned this 1-based level.
        use_culling : bool
            Enable viewport culling.

        Returns
        -------
        list[ChunkRequest]
            Nearest-first.  Empty when all required tiles are cached.
        """
        t_plan_start = time.perf_counter()
        self._frame_number += 1

        # Record the current slice coordinate for two-phase LUT rebuild (Step 3).
        self._current_slice_coord = tuple(
            sorted(dims_state.selection.slice_indices.items())
        )

        geo2d = self._image_geometry_2d
        block_size = geo2d.block_size
        n_levels = geo2d.n_levels

        # Lazy node matrix update when displayed axes change.
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)

        # Camera / viewport are 2D from pygfx -- use the 2D
        # sub-transform for inverse mapping.
        #
        # Axis-order note: pygfx delivers world coords as (X, Y) where
        #   X = second displayed axis,  Y = first displayed axis.
        # But select_axes(displayed=(a, b)).imap_coordinates expects input in
        #   (a-world, b-world) = (first-displayed, second-displayed) order
        # and returns (a-data, b-data) = (gy, gx) order.
        # Downstream consumers (sort_tiles_by_distance_2d, viewport_cull_2d)
        # expect (gx, gy) = (second-displayed, first-displayed) order.
        # So we swap the input and swap the output.
        sub_2d = self._transform.select_axes(displayed)

        # Convert world_width to level-0 voxels using the geometric mean of
        # the per-axis world-to-voxel scale from the forward (data→world)
        # transform.  sub_2d.matrix diagonal = world_units/voxel per axis;
        # the inverse geometric mean gives voxels/world_unit.
        nd2 = sub_2d.ndim
        world_units_per_voxel_2d = np.abs(np.diag(sub_2d.matrix[:nd2, :nd2]))
        world_to_voxel_scale_2d = float(
            np.prod(1.0 / world_units_per_voxel_2d) ** (1.0 / nd2)
        )
        voxel_width = world_width * world_to_voxel_scale_2d

        # camera_pos_world[:2] is (X_world, Y_world); imap expects (Y_world, X_world).
        camera_pos_2d = sub_2d.imap_coordinates(
            camera_pos_world[[1, 0]].reshape(1, -1)
        ).flatten()[[1, 0]]  # swap output (gy, gx) -> (gx, gy)
        # Pad to 3D for compatibility with downstream code.
        camera_pos = np.array(
            [camera_pos_2d[0], camera_pos_2d[1], 0.0], dtype=np.float32
        )

        # Transform viewport AABB to data space via corners.
        if use_culling and view_min_world is not None and view_max_world is not None:
            cx = float(camera_pos_world[0])  # X_world = second-displayed axis
            cy = float(camera_pos_world[1])  # Y_world = first-displayed axis
            half_w = world_width / 2.0
            half_h = (float(view_max_world[1]) - float(view_min_world[1])) / 2.0
            # imap expects (first-displayed, second-displayed) = (cy, cx) order.
            corners_world_2d = np.array(
                [
                    [cy - half_h, cx - half_w],
                    [cy - half_h, cx + half_w],
                    [cy + half_h, cx + half_w],
                    [cy + half_h, cx - half_w],
                ],
                dtype=np.float32,
            )
            # imap returns (gy, gx); swap columns to (gx, gy) for culling.
            corners_data_2d = sub_2d.imap_coordinates(corners_world_2d)[:, [1, 0]]
            view_min = corners_data_2d.min(axis=0)
            view_max = corners_data_2d.max(axis=0)
        else:
            view_min = None
            view_max = None

        self._current_viewport_cells = self._viewport_cells_from_view_bounds(
            view_min, view_max, block_size
        )

        # ── DEBUG: print camera & viewport in level-0 voxel space ──────────
        import logging as _logging

        _DBG = _logging.getLogger("cellier.2d_reslice_debug")
        if _DBG.isEnabledFor(_logging.DEBUG):
            _DBG.debug(
                "[2D plan] displayed=%s  camera_L0_vox(gx,gy)=(%.2f, %.2f)  "
                "viewport_L0_vox: min=%s  max=%s",
                displayed,
                float(camera_pos[0]),
                float(camera_pos[1]),
                (
                    f"({view_min[0]:.2f},{view_min[1]:.2f})"
                    if view_min is not None
                    else "None"
                ),
                (
                    f"({view_max[0]:.2f},{view_max[1]:.2f})"
                    if view_max is not None
                    else "None"
                ),
            )

        # Slice indices mapped per-level via composed transforms.

        # 1. LOD selection
        t0 = time.perf_counter()
        tile_arr = select_lod_2d(
            geo2d._level_grids,
            n_levels,
            viewport_width_px=viewport_width_px,
            voxel_width=voxel_width,
            lod_bias=lod_bias,
            force_level=force_level,
            level_scale_factors=geo2d._level_scale_factors,
        )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # 2. Distance sort
        t0 = time.perf_counter()
        tile_arr = sort_tiles_by_distance_2d(
            tile_arr,
            camera_pos,
            block_size,
            level_scale_arr_shader=geo2d._scale_arr_shader,
            level_translation_arr_shader=geo2d._translation_arr_shader,
        )
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        # Convert to dict (embed current slice coord into every key).
        required = arr_to_block_keys_2d(tile_arr, slice_coord=self._current_slice_coord)
        n_total = len(required)

        # 3. Viewport culling
        n_culled = 0
        cull_ms = 0.0
        if use_culling and view_min is not None and view_max is not None:
            t0 = time.perf_counter()
            required, n_culled = viewport_cull_2d(
                required,
                block_size,
                view_min,
                view_max,
                level_scale_arr_shader=geo2d._scale_arr_shader,
                level_translation_arr_shader=geo2d._translation_arr_shader,
            )
            cull_ms = (time.perf_counter() - t0) * 1000

        # 4. Budget truncation
        n_needed = len(required)
        n_budget = self._block_cache_2d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            keys_to_keep = list(required.keys())[:n_budget]
            required = {k: required[k] for k in keys_to_keep}

        # 5. Evict finer-than-target tiles, then stage.
        # Evicting first returns slots to free_slots so the incoming coarser
        # tiles can claim them without triggering unnecessary LRU evictions.
        target_level = int(tile_arr[0, 0]) if len(tile_arr) > 0 else 1

        self._block_cache_2d.tile_manager.evict_finer_than(target_level)

        t0 = time.perf_counter()
        fill_plan = self._block_cache_2d.tile_manager.stage(
            required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        # When all required tiles are cache hits, on_data_ready_2d never fires
        # so the LUT would remain stale (pointing to a different slice position
        # or freed slots).  Rebuild immediately in that case.
        #
        # This is intentionally *conditional*, unlike the 3D path
        # (build_slice_request), which rebuilds the LUT unconditionally after
        # stage().  The difference is load-bearing and stems from the caches'
        # residency models: TileManager3D has a reserve tier whose hits are
        # promoted into ``tilemap`` *during* stage() with no fetch and no
        # fill_plan entry (_tile_manager_3d.py stage()), so 3D needs an eager
        # rebuild or those bricks would be invisible until the next miss-batch
        # (which may never come).  TileManager2D has no reserve tier: stage()
        # only refreshes existing hits or reserves slots for misses, so the
        # only new ``tilemap`` content arrives via commit() -> on_data_ready_2d,
        # which already rebuilds.  The lone gap is the all-hits frame above.
        #
        # COUPLING: if a reserve tier is ever added to TileManager2D, this
        # conditional MUST become unconditional to match the 3D path, or
        # reserve-promoted tiles will silently fail to render.  Update both
        # sites together.  See also build_slice_request and the audit notes
        # in slice_coordinator._on_dims_changed / invalidate_2d_cache.
        if not fill_plan:
            self._lut_manager_2d.rebuild(
                self._block_cache_2d.tile_manager,
                current_slice_coord=self._current_slice_coord,
                viewport_cells=self._current_viewport_cells,
            )

        # 6. Build ChunkRequests
        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map_2d = {}

        overlap = self._block_cache_2d.info.overlap

        ndim = len(dims_state.axis_labels)
        sel = dims_state.selection

        _debug_tile_count = 0
        for tile_key, slot in fill_plan:
            chunk_id = uuid4()
            y0, x0, y1, x1 = _block_key_2d_to_padded_coords(
                tile_key, block_size, overlap
            )
            level_index = tile_key.level - 1
            display_coords = [(y0, y1), (x0, x1)]
            axis_selections = _build_axis_selections_multiscale(
                sel,
                ndim,
                display_coords,
                level_shape=self._full_level_shapes[level_index],
                world_to_level_k=self._world_to_level_transforms[level_index],
            )

            # ── DEBUG: print tile region and data selection ─────────────────
            if _DBG.isEnabledFor(_logging.DEBUG) and _debug_tile_count < 4:
                _sv = geo2d._scale_arr_shader[level_index]
                _cx_l0 = (tile_key.g1 + 0.5) * block_size * float(_sv[0])
                _cy_l0 = (tile_key.g0 + 0.5) * block_size * float(_sv[1])
                _DBG.debug(
                    "  tile level=%d g0=%d g1=%d  "
                    "centre_L0_vox(g1,g0)=(%.2f,%.2f)  axis_sel=%s",
                    tile_key.level,
                    tile_key.g0,
                    tile_key.g1,
                    _cx_l0,
                    _cy_l0,
                    axis_selections,
                )
                _debug_tile_count += 1

            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_id,
                scale_index=tile_key.level - 1,
                axis_selections=axis_selections,
            )
            chunk_requests.append(req)
            self._pending_slot_map_2d[chunk_id] = (tile_key, slot)

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        self._last_plan_stats = stats = {
            "hits": len(required) - len(fill_plan),
            "misses": len(fill_plan),
            "fills": len(fill_plan),
            "total_required": n_total,
            "n_culled": n_culled,
            "n_needed": n_needed,
            "n_budget": n_budget,
            "n_dropped": n_dropped,
            "lod_select_ms": lod_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "cull_ms": cull_ms,
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }

        if n_dropped > 0:
            _PERF_LOGGER.warning(
                "budget_exceeded  required=%d  budget=%d  dropped=%d  "
                "(consider larger cache or tighter LOD thresholds)",
                n_needed,
                n_budget,
                n_dropped,
            )

        _PERF_LOGGER.info(
            "[frame %d]  lod_select=%.1fms  dist_sort=%.1fms  cull=%.1fms  "
            "stage=%.1fms  |  required=%d  culled=%d  hits=%d  misses=%d",
            self._frame_number,
            stats["lod_select_ms"],
            stats["distance_sort_ms"],
            stats.get("cull_ms", 0.0),
            stats.get("stage_ms", 0.0),
            stats["total_required"],
            stats.get("n_culled", 0),
            stats["hits"],
            stats["misses"],
        )

        return chunk_requests

    def on_data_ready_2d(
        self,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        """Commit an arriving batch of 2D tiles to the GPU cache.

        Parameters
        ----------
        batch : list of (ChunkRequest, ndarray)
            Each item is a ``(request, data)`` pair where ``data`` has
            shape ``(pbs, pbs)``, dtype float32.
        """
        for req, data in batch:
            entry = self._pending_slot_map_2d.get(req.chunk_request_id)
            if entry is None:
                continue
            tile_key, slot = entry
            self._block_cache_2d.write_tile(slot, data, key=tile_key)
            self._block_cache_2d.tile_manager.commit(tile_key, slot)

        n_resident = len(self._block_cache_2d.tile_manager.tilemap)

        _GPU_LOGGER.info(
            "gpu_flush  tiles_in_batch=%d  resident=%d",
            len(batch),
            n_resident,
        )

        self._lut_manager_2d.rebuild(
            self._block_cache_2d.tile_manager,
            current_slice_coord=self._current_slice_coord,
            viewport_cells=self._current_viewport_cells,
        )

        _GPU_LOGGER.info(
            "lut_rebuilt  resident=%d  frame=%d",
            n_resident,
            self._frame_number,
        )

        # On first tile batch, reveal the AABB line if enabled.
        if not self._data_ready_2d and self._aabb_line_2d is not None:
            self._data_ready_2d = True
            self._aabb_line_2d.visible = self._aabb_enabled

    def cancel_pending_2d(self) -> None:
        """Release all in-flight 2D slots."""
        if self._block_cache_2d is None:
            return
        self._block_cache_2d.tile_manager.release_all_in_flight()
        self._pending_slot_map_2d = {}

    def invalidate_2d_cache(self) -> None:
        """Cancel in-flight 2D requests when the slice position changes.

        Old committed tiles are intentionally kept alive as a visible fallback
        while new tiles load.  Because ``BlockKey2D`` now encodes
        ``slice_coord``, tiles from different slice positions cannot collide, so
        keeping them does not cause rendering artefacts.

        A full LUT rebuild is not needed here — the LUT remains current from
        the last ``on_data_ready_2d`` call and old tiles are the correct thing
        to display until new ones arrive.
        """
        if self._block_cache_2d is None or self._lut_manager_2d is None:
            return
        self.cancel_pending_2d()

    def _allocate_paint_resources_2d(self) -> None:
        """Allocate the 2D paint cache + LUT textures and the slot manager.

        Called from ``__init__`` and from ``_rebuild_2d_resources`` so the
        paint resources track the underlying 2D image geometry.
        """
        if self._image_geometry_2d is None:
            return
        bs = self._image_geometry_2d.block_size
        gh, gw = self._image_geometry_2d.base_layout.grid_dims

        # Stripes-of-tiles layout: row-stripe per slot of height bs.
        # Two channels: (value, alpha).  Alpha=0 ⇒ this voxel is unpainted.
        paint_cache_data = np.zeros(
            (self._paint_max_tiles * bs, bs, 2), dtype=np.float32
        )
        self._t_paint_cache = gfx.Texture(paint_cache_data, dim=2, format="2xf4")

        # Same grid as base LUT: per-tile (slot_index, alpha).
        # alpha=0 ⇒ no slot allocated for this tile (skip the cache lookup).
        paint_lut_data = np.zeros((gh, gw, 2), dtype=np.float32)
        self._t_paint_lut = gfx.Texture(paint_lut_data, dim=2, format="2xf4")

        self._paint_slot_manager = PaintTileSlotManager(self._paint_max_tiles)

    def patch_paint_texture(
        self,
        voxel_indices: np.ndarray,
        values: np.ndarray,
        displayed_axes: tuple[int, int],
    ) -> int:
        """Write painted voxels into the GPU paint cache + LUT.

        Used by ``MultiscalePaintController._write_values`` for sub-frame
        paint feedback in 2-D.  No-op if the visual has no 2-D paint
        resources (e.g. 3-D-only visual).

        Parameters
        ----------
        voxel_indices :
            Shape ``(N, ndim)`` int64.  Level-0 voxel indices in
            data-array axis order.
        values :
            Shape ``(N,)`` float32.  Brush values to write.
        displayed_axes :
            Two-element tuple ``(row_axis, col_axis)`` describing which
            data-array axes correspond to the displayed (row, col) of
            the 2-D tile grid.  ``row_axis`` ↔ pygfx-y, ``col_axis`` ↔
            pygfx-x; this matches the upload convention of the multiscale
            2-D image (no transpose on upload).

        Returns
        -------
        int
            Number of distinct tiles successfully patched.  Tiles that
            could not be allocated a slot (pool exhaustion) are silently
            skipped; check ``self._paint_slot_manager.exhausted`` to
            detect this.
        """
        if (
            self._t_paint_cache is None
            or self._t_paint_lut is None
            or self._paint_slot_manager is None
            or self._image_geometry_2d is None
        ):
            return 0
        if voxel_indices.shape[0] == 0:
            return 0

        bs = int(self._image_geometry_2d.block_size)
        ax_row, ax_col = displayed_axes

        rows = voxel_indices[:, ax_row].astype(np.int64)
        cols = voxel_indices[:, ax_col].astype(np.int64)
        vals = values.astype(np.float32, copy=False)

        gy = rows // bs
        gx = cols // bs
        ty = rows % bs
        tx = cols % bs

        cache_data = self._t_paint_cache.data
        lut_data = self._t_paint_lut.data

        cache_y_min = cache_data.shape[0]
        cache_y_max = 0
        cache_x_min = cache_data.shape[1]
        cache_x_max = 0
        lut_y_min = lut_data.shape[0]
        lut_y_max = 0
        lut_x_min = lut_data.shape[1]
        lut_x_max = 0
        n_tiles_patched = 0

        # Group voxels by (gy, gx) so each tile gets one slot allocation.
        pairs = np.stack([gy, gx], axis=1)
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)

        for tile_i in range(unique_pairs.shape[0]):
            gy_t = int(unique_pairs[tile_i, 0])
            gx_t = int(unique_pairs[tile_i, 1])
            slot = self._paint_slot_manager.get_or_allocate((gy_t, gx_t))
            if slot is None:
                continue
            n_tiles_patched += 1

            sel = inverse == tile_i
            ty_sel = ty[sel]
            tx_sel = tx[sel]
            v_sel = vals[sel]

            # Stripes layout: this slot's pixel rows are
            # cache_data[slot*bs : (slot+1)*bs, :, :]
            slot_y0 = slot * bs
            cache_data[slot_y0 + ty_sel, tx_sel, 0] = v_sel
            cache_data[slot_y0 + ty_sel, tx_sel, 1] = 1.0

            lut_data[gy_t, gx_t, 0] = float(slot)
            lut_data[gy_t, gx_t, 1] = 1.0

            cache_y_min = min(cache_y_min, slot_y0)
            cache_y_max = max(cache_y_max, slot_y0 + bs)
            cache_x_min = 0
            cache_x_max = bs
            lut_y_min = min(lut_y_min, gy_t)
            lut_y_max = max(lut_y_max, gy_t + 1)
            lut_x_min = min(lut_x_min, gx_t)
            lut_x_max = max(lut_x_max, gx_t + 1)

        if n_tiles_patched == 0:
            return 0

        # pygfx Texture.update_range takes (offset, size) in (x, y, z) order.
        self._t_paint_cache.update_range(
            (cache_x_min, cache_y_min, 0),
            (cache_x_max - cache_x_min, cache_y_max - cache_y_min, 1),
        )
        self._t_paint_lut.update_range(
            (lut_x_min, lut_y_min, 0),
            (lut_x_max - lut_x_min, lut_y_max - lut_y_min, 1),
        )
        return n_tiles_patched

    def clear_paint_textures(self) -> None:
        """Reset all paint-cache slots and LUT entries to "no paint".

        Called by ``MultiscalePaintController.commit`` and ``abort`` at
        session end.
        """
        if (
            self._t_paint_cache is None
            or self._t_paint_lut is None
            or self._paint_slot_manager is None
        ):
            return
        self._t_paint_cache.data[:] = 0.0
        self._t_paint_lut.data[:] = 0.0
        # data.shape[:2] is (rows, cols); update_range size is (x, y, z)
        # = (cols, rows, 1).
        cache_h, cache_w = self._t_paint_cache.data.shape[:2]
        lut_h, lut_w = self._t_paint_lut.data.shape[:2]
        self._t_paint_cache.update_range((0, 0, 0), (cache_w, cache_h, 1))
        self._t_paint_lut.update_range((0, 0, 0), (lut_w, lut_h, 1))
        self._paint_slot_manager.clear()

    def invalidate_painted_tiles_2d(
        self, dirty_grid_coords: set[tuple[int, int]]
    ) -> int:
        """Evict committed finest-level tiles whose ``(gy, gx)`` is dirty.

        Used by ``MultiscalePaintController`` to drop the visible cached
        copy of a tile after a brush write so the next reslice re-fetches
        through the open paint transaction (read-your-writes).

        Only level-1 tiles (== finest-data level == level 0) are evicted.
        Coarser levels show pre-paint data until the next paint-pyramid
        rebuild (deferred to a follow-up).  In-flight slots are also
        released so any pending stale fetches don't overwrite freshly
        painted state.

        Parameters
        ----------
        dirty_grid_coords :
            Set of ``(gy, gx)`` finest-level tile-grid coordinates whose
            corresponding tiles should be evicted.  Coordinates that
            don't currently have a committed tile are silently ignored.

        Returns
        -------
        int
            Number of tiles evicted from the committed tilemap.
        """
        if self._block_cache_2d is None:
            return 0
        tm = self._block_cache_2d.tile_manager
        # Always release in-flight to avoid races with stale fetches.
        tm.release_all_in_flight()
        self._pending_slot_map_2d = {}
        evicted = 0
        for key in list(tm.tilemap.keys()):
            if key.level != 1:
                continue
            if (key.g0, key.g1) in dirty_grid_coords:
                slot = tm.tilemap.pop(key)
                tm.slot_index[slot.index] = None
                tm.free_slots.append(slot.index)
                evicted += 1
        return evicted

    # ── EventBus handler methods ─────────────────────────────────────────

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update stored transform, norm_size uniform, and pygfx node matrix."""
        self._transform = event.transform
        self._world_to_level_transforms = self._build_world_to_level_transforms()

        # Recompute norm_size and push the updated uniform buffer so the
        # proxy cube shape stays consistent with the new transform.
        if self._dataset_size is not None:
            _check_transform_no_rotation(self._transform)
            self._norm_size = _norm_size_from_transform(
                self._transform, self._norm_size_axes, self._dataset_size
            )
            if self._vol_params_buffer is not None:
                buf_data = self._vol_params_buffer.data
                buf_data["norm_size_x"] = float(self._norm_size[0])
                buf_data["norm_size_y"] = float(self._norm_size[1])
                buf_data["norm_size_z"] = float(self._norm_size[2])
                self._vol_params_buffer.update_full()
            if self._inner_node_3d is not None:
                self._inner_node_3d.update_norm_size(self._norm_size)
            # The wireframe lives in normalized space, which rescales with
            # norm_size, so its geometry must be rebuilt (not just the matrix).
            if self._aabb_line_3d is not None and self._dataset_size is not None:
                box = norm_full_extent_box(self._dataset_size, self._norm_size)
                self._aabb_line_3d.geometry = gfx.Geometry(
                    positions=_box_wireframe_positions(box[0], box[1])
                )

        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply GPU-only appearance changes."""
        if event.field_name == "color_map":
            from cmap import Colormap

            colormap = (
                Colormap(event.new_value)
                if isinstance(event.new_value, str)
                else event.new_value
            )
            new_map = colormap.to_pygfx(N=256)
            if self.material_3d is not None:
                self.material_3d.map = new_map
            if self.material_2d is not None:
                self.material_2d.map = new_map
        elif event.field_name == "clim":
            if self.material_3d is not None:
                self.material_3d.clim = event.new_value
            if self.material_2d is not None:
                self.material_2d.clim = event.new_value
        elif event.field_name == "iso_threshold":
            if self.material_3d is not None:
                self.material_3d.threshold = float(event.new_value)
        elif event.field_name == "attenuation":
            if self.material_3d is not None:
                self.material_3d.attenuation = float(event.new_value)
        elif event.field_name == "render_mode":
            if self.material_3d is not None:
                self.material_3d.render_mode = event.new_value
        elif event.field_name == "render_order":
            if self.node_3d is not None:
                self.node_3d.render_order = event.new_value
            if self.node_2d is not None:
                self.node_2d.render_order = event.new_value
        elif event.field_name == "opacity":
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    mat.opacity = event.new_value
        elif event.field_name in ("depth_test", "depth_write", "depth_compare"):
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    setattr(mat, event.field_name, event.new_value)
        elif event.field_name == "interpolation":
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    mat.interpolation = event.new_value
        elif event.field_name == "transparency_mode":
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    mat.alpha_mode = event.new_value

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Apply visibility change to all render nodes."""
        self._visible = event.visible
        if self.node_3d is not None:
            self.node_3d.visible = event.visible
        if self.node_2d is not None:
            self.node_2d.visible = event.visible

    def pick_data_coordinate(
        self, hit_object, pick_info: dict
    ) -> tuple[float, ...] | None:
        """Level-0 data coordinate of a pick on this visual (displayed axes).

        3-D bricks decode a centred-normalised surface hit (``norm_pos``) mapped
        back to voxels via ``norm_size`` / ``dataset_size``; the 2-D node is a
        tile-grid proxy whose ``index`` is rescaled to level-0 pixels.  See
        :mod:`cellier.render.visuals._pick`.
        """
        if isinstance(hit_object, NormSizedVolume):
            if self._norm_size is None or self._dataset_size is None:
                return None
            return multiscale_volume_data_coordinate(
                pick_info, self._norm_size, self._dataset_size
            )
        if isinstance(hit_object, gfx.Image):
            if self._image_geometry_2d is None:
                return None
            return multiscale_image_data_coordinate(
                pick_info,
                self._image_geometry_2d.level_shapes[0],
                self._image_geometry_2d.base_layout.grid_dims,
            )
        return None

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        """Update pick_write on all active materials."""
        for mat in (self.material_3d, self.material_2d):
            if mat is not None:
                mat.pick_write = event.pick_write

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Apply an AABB parameter change.

        ``enabled`` toggles AABB line visibility (guarded by data-ready
        flags so the line cannot appear before the first brick/tile batch).
        ``color`` updates the line material.

        Parameters
        ----------
        event : AABBChangedEvent
            Carries ``field_name`` and ``new_value``.
        """
        if event.field_name == "enabled":
            self._aabb_enabled = event.new_value
            if self._aabb_line_3d is not None:
                self._aabb_line_3d.visible = event.new_value and self._data_ready_3d
            if self._aabb_line_2d is not None:
                self._aabb_line_2d.visible = event.new_value and self._data_ready_2d
        elif event.field_name == "color":
            self._aabb_color = event.new_value
            for line in (self._aabb_line_3d, self._aabb_line_2d):
                if line is not None:
                    line.material.color = event.new_value
        elif event.field_name == "line_width":
            self._aabb_line_width = event.new_value
            for line in (self._aabb_line_3d, self._aabb_line_2d):
                if line is not None:
                    line.material.thickness = event.new_value

    def on_data_store_contents_changed(
        self, event: DataStoreContentsChangedEvent
    ) -> None:
        """Stub — brick eviction deferred to a future phase."""
        pass

    def on_data_store_metadata_changed(
        self, event: DataStoreMetadataChangedEvent
    ) -> None:
        """Stub — geometry rebuild deferred to a future phase."""
        pass

    def tick(self) -> None:
        """Advance jitter seed for the brick shader."""
        if self.material_3d is not None:
            self.material_3d.tick()

    def reset_tick(self) -> None:
        """Rewind the per-frame jitter seed to its construction value.

        The counterpart to :meth:`tick`, called before a reproducible capture.
        This is the only visual with per-frame state; every other ``tick`` is
        a no-op and needs no reset.
        """
        if self.material_3d is not None:
            self.material_3d.reset_frame_index()

    # ── Private helpers ─────────────────────────────────────────────────

    def _build_aabb_line_3d(self) -> gfx.Line:
        """Build the 3D AABB wireframe line in normalized space.

        The wireframe lives in the node's normalized local space, so it uses
        the same ``norm_full_extent_box`` as ``NormSizedVolume.get_bounding_box``
        — both wrap the full voxel extent (data ``[-0.5, N-0.5]``).
        """
        if self._norm_size is not None and self._dataset_size is not None:
            box = norm_full_extent_box(self._dataset_size, self._norm_size)
            positions = _box_wireframe_positions(box[0], box[1])
        else:
            positions = _box_wireframe_positions(np.zeros(3), np.ones(3))
        return _make_aabb_line(positions, self._aabb_color, self._aabb_line_width)

    def _build_aabb_line_2d(self) -> gfx.Line:
        """Build the 2D AABB wireframe rect for the current geometry."""
        if self._image_geometry_2d is not None:
            h, w = self._image_geometry_2d.level_shapes[0]
            # pixel i center at i, edges at i±0.5: rect spans [-0.5, N-0.5],
            # matching the center-at-integer node placement in _build_2d_node.
            positions = _rect_wireframe_positions(
                np.array([-0.5, -0.5]),
                np.array([float(w) - 0.5, float(h) - 0.5]),
            )
        else:
            positions = _rect_wireframe_positions(np.zeros(2), np.ones(2))
        return _make_aabb_line(positions, self._aabb_color, self._aabb_line_width)

    def _build_3d_node(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        threshold: float,
        attenuation: float = 1.0,
        pick_write: bool = True,
    ) -> tuple[NormSizedVolume, MultiscaleVolumeBrickMaterial, gfx.Texture]:
        """Construct the proxy texture, brick material, and Volume node.

        The brick shader generates its own box geometry from
        ``u_vol_params.norm_size_*``, so the proxy texture is a small
        dummy (2x2x2) and the inner Volume node has identity local
        transform.  The Group node's matrix (set by
        ``_update_node_matrix``) maps normalized -> world.

        NormSizedVolume is used so that pygfx's bounding box machinery sees
        the correct [-norm_size/2, norm_size/2] bounds rather than the
        asymmetric box derived from the 2x2x2 proxy texture dimensions.
        """
        proxy_data = np.zeros((2, 2, 2), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=3)

        material = MultiscaleVolumeBrickMaterial(
            cache_texture=self._block_cache_3d.cache_tex,
            lut_texture=self._lut_manager_3d.lut_tex,
            brick_max_texture=self._lut_manager_3d.brick_max_tex,
            vol_params_buffer=self._vol_params_buffer,
            block_scales_buffer=self._brick_scales_buffer,
            clim=clim,
            map=colormap,
            threshold=threshold,
            attenuation=attenuation,
            pick_write=pick_write,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        vol = NormSizedVolume(
            geometry,
            material,
            norm_size=self._norm_size,
            dataset_size=self._dataset_size,
        )
        # No inner transform — vertex shader uses normalized space.

        return vol, material, proxy_tex

    def _build_2d_node(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        interpolation: str,
        pick_write: bool = True,
    ) -> tuple[gfx.Image, ImageBlockMaterial, gfx.Texture]:
        gh, gw = self._image_geometry_2d.base_layout.grid_dims

        proxy_data = np.zeros((gh, gw), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=2)

        material = ImageBlockMaterial(
            cache_texture=self._block_cache_2d.cache_tex,
            lut_texture=self._lut_manager_2d.lut_tex,
            lut_params_buffer=self._lut_params_buffer_2d,
            block_scales_buffer=self._block_scales_buffer_2d,
            paint_cache_texture=self._t_paint_cache,
            paint_lut_texture=self._t_paint_lut,
            clim=clim,
            map=colormap,
            pick_write=pick_write,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        image = gfx.Image(geometry, material)

        bs = self._image_geometry_2d.block_size
        h, w = self._image_geometry_2d.level_shapes[0]
        # Proxy grid rounds up to whole tiles. Apply a per-axis correction so
        # the displayed quad footprint matches the true finest-level (W, H).
        sx = float(w) / float(gw * bs)
        sy = float(h) / float(gh * bs)
        scale_x = float(bs) * sx
        scale_y = float(bs) * sy
        image.local.scale = (scale_x, scale_y, 1.0)
        # pygfx places proxy texel 0's center at the local origin, so the scaled
        # quad spans [-0.5*scale, W-0.5*scale]. Shift by (scale - 1)/2 so the
        # footprint is [-0.5, W-0.5]: data pixel i centered on world i, matching
        # the memory gfx.Image and the multiscale volume (center-at-integer).
        image.local.position = ((scale_x - 1.0) * 0.5, (scale_y - 1.0) * 0.5, 0.0)

        return image, material, proxy_tex
