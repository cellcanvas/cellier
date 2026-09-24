"""GFXMultiscaleImageVisual -- render-layer visual for multiscale images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.render._backstop import (
    backstop_bricks_3d,
    backstop_level,
    backstop_tiles_2d,
)
from cellier.render._frustum import (
    bricks_in_frustum_arr,
    frustum_planes_from_corners,
)
from cellier.render._gpu_lifetime import destroy_textures, weak_callback
from cellier.render._level_mapping import base_cell_range
from cellier.render._level_of_detail import (
    build_level_grids,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.render._level_of_detail_2d import (
    build_tile_grids_2d,
    select_lod_2d,
    sort_tiles_by_distance_2d,
    viewport_cull_2d,
)
from cellier.render._spaces import (
    RenderSpaces,
    axis_correspondence,
    cellier_to_pygfx_order,
    node_matrix,
    pygfx_to_cellier_order,
    select_axes,
    swap_axes,
)
from cellier.render.block_cache import (
    BlockCache3D,
    compute_block_cache_parameters_3d,
)
from cellier.render.block_cache._block_cache_2d import BlockCache2D
from cellier.render.block_cache._cache_parameters_2d import (
    TILE_BORDER_2D,
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
from cellier.render.scheduling import ChunkClass, DesiredSet, PlanMode
from cellier.render.shaders._block_image import ImageBlockMaterial
from cellier.render.shaders._multiscale_volume_brick import (
    MultiscaleVolumeBrickMaterial,
    _norm_to_data_params,
    build_brick_scales_buffer,
    build_vol_params_buffer,
    compute_normalized_size,
    norm_full_extent_box,
)
from cellier.render.visuals._chunked import (
    backstop_cap_for,
    desired_bricks,
    log_backstop_cap_once,
    make_residency_2d,
    make_residency_3d,
    viewport_2d,
)
from cellier.render.visuals._image_memory import (
    _box_wireframe_positions,
    _make_aabb_line,
    _make_colormap,
    _rect_wireframe_positions,
)
from cellier.render.visuals._pick import (
    multiscale_image_data_coordinate,
    multiscale_volume_data_coordinate,
)
from cellier.render.visuals._slicing import (
    axis_selections_from_box,
    image_plane_selection,
)
from cellier.visuals._image_memory import effective_transparency_mode

if TYPE_CHECKING:
    from uuid import UUID

    from pygfx.resources import Buffer

    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        ChannelAppearanceChangedEvent,
        DataStoreContentsChangedEvent,
        DataStoreMetadataChangedEvent,
        ImageCompositeChangedEvent,
        PickWriteChangedEvent,
        SingleAppearanceChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.render._requests import ReslicingRequest
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render.block_cache._image_residency import (
        ImageResidency2D,
        ImageResidency3D,
    )
    from cellier.transform import AffineTransform
    from cellier.visuals._image import MultiscaleImageVisual
    from cellier.visuals._loading import ProgressiveLoadingConfig

# Importing this module registers the shader class with pygfx via the
# @register_wgpu_render_function decorator.
import cellier.render.shaders._multiscale_volume_brick as _brick_reg  # noqa: F401

#: How far past a sample position each image path reads, in level-k voxels,
#: at the default ``ray_steps_per_voxel`` (plan v2, "Padding budgets"): the
#: brick-rule padding warning subtracts it from the ghost border.  3D: the
#: gradient probe (+-1.5) outreaches one bisection step (1 at K = 1); 2D:
#: linear filtering.
_IMAGE_SAMPLING_MARGIN_3D = 1.5
_IMAGE_SAMPLING_MARGIN_2D = 0.5


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


def _level_scale_and_translation(
    level_transforms: list[AffineTransform],
    fetch_axes: tuple[int, ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-axis scale and translation of each level, over the fetch axes.

    ``level_transforms`` are **full rank**: one per level, mapping level-k
    voxel coordinates to level-0, over every data axis.  *fetch_axes* names
    the ascending data axes the brick grid is built over, and the vectors come
    back in that order -- which is the order the fetched array carries
    (:func:`_fetch_order`).  Callers that feed these into shader-space
    geometry must reverse the local axis order via ``swap_axes`` first.

    Until v1 was retired the projection was a separate step: the caller built
    a rank-reduced ``select_axes(fetch_axes)`` transform and this read its
    whole diagonal.  A v2 transform names its endpoints and cannot be
    rank-reduced without minting two coordinate systems that describe nothing,
    so the axis selection happens here instead -- reading the same entries
    ``select_axes`` would have kept.

    Parameters
    ----------
    level_transforms : list[AffineTransform]
        Full-rank per-level transforms, level-k voxel to level-0 voxel.
    fetch_axes : tuple[int, ...]
        The data axes to read, ascending.

    Returns
    -------
    scale_vecs : list[np.ndarray]
        ``(len(fetch_axes),)`` per level -- the diagonal entry per axis.
    translation_vecs : list[np.ndarray]
        ``(len(fetch_axes),)`` per level -- the translation entry per axis.
    """
    scale_vecs: list[np.ndarray] = []
    translation_vecs: list[np.ndarray] = []
    for transform in level_transforms:
        matrix = np.asarray(transform.matrix, dtype=np.float64)
        rank = matrix.shape[0] - 1
        scale_vecs.append(
            np.array([matrix[axis, axis] for axis in fetch_axes], dtype=np.float64)
        )
        translation_vecs.append(
            np.array([matrix[axis, rank] for axis in fetch_axes], dtype=np.float64)
        )
    return scale_vecs, translation_vecs


# ---------------------------------------------------------------------------
# Brick-shader transform helpers
# ---------------------------------------------------------------------------


def _brick_max_image(data: np.ndarray) -> float:
    """An image brick's MIP early-out value: its maximum."""
    return float(np.max(data)) if data.size else 0.0


def _check_transform_no_rotation(transform: AffineTransform | None) -> None:
    """Raise ValueError if the transform contains rotation or shear.

    The brick shader only supports scale and translation.  A transform with
    off-diagonal entries in the linear submatrix (i.e. rotation or shear)
    would require ``norm_to_voxel`` to be a full affine inverse, which the
    current WGSL does not implement.

    Parameters
    ----------
    transform : AffineTransform or None
        The ``data -> world`` transform to validate.  ``None`` means the
        visual has not been placed in a world yet.

    Raises
    ------
    ValueError
        If the linear part of the transform is not diagonal.
    """
    if transform is None:
        # The visual has not been placed in a world yet, so there is no
        # transform to constrain.  It is checked again when one arrives.
        return
    # A ``data -> world`` transform need not be square, so "is the linear
    # block diagonal" is not a well-formed question (P10 / F0.6).  The
    # equivalent, and what the shader actually needs, is: each input axis
    # reaches at most one output axis, each output axis is reached by at most
    # one input axis, and the correspondence preserves order.  For a square
    # transform that is exactly "the linear block is diagonal", so this is a
    # generalisation and not a relaxation.
    try:
        correspondence = axis_correspondence(transform)
    except ValueError as error:
        raise ValueError(
            "The brick shader only supports scale and translation transforms. "
            "The provided transform contains rotation or shear components. "
            "Support for general affine transforms requires changes to the "
            "WGSL norm_to_voxel function and is not yet implemented."
        ) from error
    outputs = [correspondence[axis] for axis in sorted(correspondence)]
    if outputs != sorted(outputs):
        raise ValueError(
            f"The brick shader only supports scale and translation transforms. "
            f"This one permutes its axes: data axes {sorted(correspondence)} "
            f"map to world axes {outputs}.  Support for general affine "
            f"transforms requires changes to the WGSL norm_to_voxel function "
            f"and is not yet implemented."
        )


def _norm_to_data_params_for(
    spaces: RenderSpaces, dataset_size: np.ndarray, norm_size: np.ndarray
) -> tuple[dict[int, float], dict[int, float]]:
    """The 3D node's ``visual -> data`` factors, keyed by data axis.

    The 3D node's geometry is a proxy box in **normalized** space, not in
    voxel indices, which is what ``VisualCoordinateSystem`` names (design 3.11
    B).  Its transform into data space is what ``compose_world_transform``
    used to build by hand as ``norm_to_data``: the box ``[-norm/2, +norm/2]``
    spans the voxel extent ``[-0.5, N - 0.5]``, so the scale is
    ``dataset_size / norm_size`` and the translation ``0.5 * dataset_size -
    0.5``.  The ``-0.5`` is the existing centre-at-integer convention,
    unchanged.

    ``dataset_size`` and ``norm_size`` are in shader ``(x, y, z)`` order while
    the retained axes are ascending data axes, so the two are zipped in
    reverse.

    Parameters
    ----------
    spaces : RenderSpaces
        The systems this visual is placed with.
    dataset_size : np.ndarray
        Finest-level voxel counts in shader order.
    norm_size : np.ndarray
        Normalized physical extent in shader order.

    Returns
    -------
    tuple[dict[int, float], dict[int, float]]
        ``(scale, translation)``, both keyed by data axis.
    """
    scale_xyz, offset_xyz = _norm_to_data_params(dataset_size, norm_size)
    retained = spaces.retained_axes
    last = len(retained) - 1
    scale = {
        axis: float(scale_xyz[last - index]) for index, axis in enumerate(retained)
    }
    translation = {
        axis: float(offset_xyz[last - index]) for index, axis in enumerate(retained)
    }
    return scale, translation


def _displayed_submatrix(
    transform: AffineTransform, displayed_axes: tuple[int, ...]
) -> np.ndarray:
    """The square displayed-axes block of a ``data -> world`` transform.

    A **bridge**, not an end state.  Two callers are left: the image
    family's 2-D planner (``_plan_tiles_2d``) and the multiscale labels'
    own ``_plan_tiles_2d`` (its ``voxel_width`` computation), both of which
    still do the pre-migration
    ``select_axes`` + reverse + ``imap_coordinates`` dance that the
    single-channel *3D* path replaced with one
    ``rendered_to_level0.map_coordinates`` (F8.1).  Migrating either is
    Phase 5 work that was never done and is not this phase's to do, so the
    arithmetic is preserved exactly -- **float32 included**, which is what a
    v1 transform stored.

    The numbers are identical to what ``select_axes(displayed_axes)``
    produced on an equal-rank transform: row ``i`` is world axis
    ``displayed_axes[i]``, column ``j`` is the data axis that feeds world axis
    ``displayed_axes[j]``.  Unlike ``select_axes`` it does not read the
    homogeneous row when the ranks disagree (design 3.8 finding 2) -- it
    raises instead.

    Parameters
    ----------
    transform : AffineTransform
        The ``data -> world`` transform.
    displayed_axes : tuple[int, ...]
        The displayed **world** axes, in display order.

    Returns
    -------
    np.ndarray
        A square ``(n + 1, n + 1)`` homogeneous **float32** matrix, where
        ``n`` is ``len(displayed_axes)``.

    Raises
    ------
    ValueError
        If a displayed world axis has no data axis feeding it.
    """
    # ``.linear`` / ``.translation`` exist only on ``AffineTransform``; a
    # ``ByDimensionTransform`` (any visual sharing the time-axis leaf) has
    # neither.  Restricting to the displayed axes first gives back a plain
    # affine with world-axis numbering intact -- see
    # ``_restricted_affine_for_display``.  A no-op, and no behaviour change,
    # for the already-affine transforms every caller here was written
    # against.
    affine = _restricted_affine_for_display(transform, displayed_axes)
    correspondence = axis_correspondence(affine)
    world_to_data = {world: data for data, world in correspondence.items()}
    linear = np.asarray(affine.linear)
    translation = np.asarray(affine.translation)
    missing = [axis for axis in displayed_axes if axis not in world_to_data]
    if missing:
        raise ValueError(
            f"World axes {missing} are displayed but no data axis of this "
            f"visual maps to them, so it has no extent along them."
        )
    n = len(displayed_axes)
    matrix = np.eye(n + 1, dtype=np.float64)
    for column, world_axis in enumerate(displayed_axes):
        data_axis = world_to_data[world_axis]
        for row, out_axis in enumerate(displayed_axes):
            matrix[row, column] = linear[out_axis, data_axis]
    for row, out_axis in enumerate(displayed_axes):
        matrix[row, n] = translation[out_axis]
    return matrix.astype(np.float32)


def _imap_square(matrix: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Pull points back through a square homogeneous matrix.

    The inverse half of what a v1 ``AffineTransform`` did, kept for
    ``_plan_tiles_2d`` alone (see ``_displayed_submatrix``).  The inverse is
    taken and then narrowed to float32, and the product is formed in float32,
    because that is what v1 did and this phase changes no numbers.

    Parameters
    ----------
    matrix : np.ndarray
        A square ``(n + 1, n + 1)`` homogeneous matrix.
    coordinates : np.ndarray
        ``(m, n)`` points in the matrix's output space.

    Returns
    -------
    np.ndarray
        ``(m, n)`` points in its input space.
    """
    ndim = matrix.shape[0] - 1
    inverse = np.linalg.inv(matrix).astype(np.float32)
    points = np.atleast_2d(coordinates)
    if points.shape[1] == ndim:
        points = np.pad(points, pad_width=((0, 0), (0, 1)), constant_values=1)
    return np.dot(points, inverse.T)[:, :ndim]


def _restricted_affine_for_display(
    transform: AffineTransform, displayed_axes: tuple[int, ...]
) -> AffineTransform:
    """The transform's affine form once every non-displayed axis is fixed.

    ``.linear`` / ``.translation`` exist only on ``AffineTransform``.  A
    ``ByDimensionTransform`` -- which is what any visual sharing a
    ``NonUniformAxisTransform`` time axis carries, broadcast or not -- has
    neither, even when every block it actually holds is affine.  Reading
    ``.linear`` off it directly is the same category of mistake
    ``_check_transform_no_rotation`` already guards against for a rotated
    transform, just for a different reason: there is no matrix to read yet.

    The fix mirrors the one the design doc gives for the GPU node-matrix
    boundary ("The GPU still gets a matrix"): every data axis this
    transform does *not* map to a displayed world axis is pinned via
    ``restrict`` -- the container's job, since it evaluates a fully-fixed
    non-affine block to a constant rather than delegating to the block's own
    ``restrict`` -- and what remains is asked for its affine form.  The
    value each axis is pinned to does not matter for the *scale* this is
    used to compute: a collapsed axis contributes only to translation
    (``_check_transform_no_rotation`` already forbids any cross-term
    reaching a displayed axis), so ``0.0`` is as good as the real slice
    position.

    A transform that is already affine is returned unchanged -- the only
    case every existing call site was written against, so this is a pure
    widening with no behaviour change for it.

    Parameters
    ----------
    transform : AffineTransform
        The visual's ``data -> world`` transform, of any concrete type.
    displayed_axes : tuple[int, ...]
        The displayed **world** axes.

    Returns
    -------
    AffineTransform
        Ready for ``.linear`` / ``.translation`` / ``axis_correspondence``,
        the last of which now answers in the restricted transform's own
        (renumbered) data-axis indices -- exactly what every caller here
        already re-derives via ``axis_correspondence`` rather than assuming.

    Raises
    ------
    NonAffineTransformError
        If a non-affine block reaches a *displayed* axis.  There is no
        matrix that expresses that, which is the honest outcome (design
        doc, "The GPU still gets a matrix").
    """
    from cellier.transform import AffineTransform as _AffineTransform
    from cellier.transform import NonAffineTransformError

    if isinstance(transform, _AffineTransform):
        return transform
    correspondence = axis_correspondence(transform)
    displayed_set = set(displayed_axes)
    fixed = {
        data_axis: 0.0
        for data_axis, world_axis in correspondence.items()
        if world_axis not in displayed_set
    }
    affine = transform.restrict(fixed).to_affine()
    if affine is None:
        raise NonAffineTransformError(
            f"{type(transform).__name__} has a non-affine block reaching a "
            f"displayed axis (world axes {displayed_axes}); there is no "
            f"matrix for the multiscale brick shader to use.  Only a "
            f"collapsed (sliced) axis may be non-affine."
        )
    return affine


def _norm_size_from_transform(
    transform: AffineTransform | None,
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
    transform : AffineTransform or None
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
    # The scale of one displayed world axis is the magnitude of the single
    # entry its data axis contributes.  On a square diagonal transform that is
    # the column norm the old ``select_axes`` path computed, so the numbers are
    # unchanged; unlike that path it is well defined when the two ranks differ
    # (P10 / F0.6), because it never asks for a square sub-block that does not
    # exist.
    if transform is None:
        # Unplaced: every axis is at unit scale, which is what a placed
        # identity would have given.
        return compute_normalized_size(dataset_size_xyz, np.ones(3))
    affine = _restricted_affine_for_display(transform, displayed_axes)
    correspondence = axis_correspondence(affine)
    world_to_data = {world: data for data, world in correspondence.items()}
    linear = np.asarray(affine.linear)
    scales = []
    for world_axis in displayed_axes:
        data_axis = world_to_data.get(world_axis)
        if data_axis is None:
            raise ValueError(
                f"World axis {world_axis} is displayed but no data axis maps "
                f"to it, so this visual has no extent along it and no scale "
                f"to normalise by."
            )
        scales.append(abs(float(linear[world_axis, data_axis])))
    # Convert displayed-axis order to shader order via explicit reversal.
    per_axis_scale_xyz = np.asarray(
        swap_axes(np.array(scales, dtype=np.float64), (2, 1, 0))
    )
    return compute_normalized_size(dataset_size_xyz, per_axis_scale_xyz)


def _world_axes_to_data_axes(
    transform: AffineTransform | None, displayed_axes: tuple[int, ...]
) -> tuple[int, ...]:
    """Translate WORLD displayed axes into this visual's own DATA axes.

    ``select_axes`` / ``_fetch_order`` index a **per-data-axis** sequence --
    a store's own ``level_shapes`` entry -- so they need this visual's data
    axis indices, not world ones.  The two coincide only when the data and
    the world share rank with no broadcast axis in between, which is the
    case every call site here was written against; a store with fewer axes
    than the world -- a ``tzyx`` labels volume broadcast over a world ``c``,
    say -- needs the translation this function performs, via the same
    ``axis_correspondence`` the already-migrated per-reslice helpers
    (``_norm_size_from_transform``, ``_displayed_submatrix``,
    ``_check_transform_no_rotation``) already read.  Those three take
    *world* axes deliberately and translate internally; this one exists
    because ``select_axes``/``_fetch_order`` do not, and cannot be taught
    to without knowing the transform themselves.

    ``transform=None`` (an unplaced visual) returns *displayed_axes*
    unchanged: there is no correspondence to consult yet, and the caller is
    only sizing a placeholder geometry that is rebuilt once the visual is
    placed.

    Parameters
    ----------
    transform : AffineTransform or None
        The visual's ``data -> world`` transform.
    displayed_axes : tuple[int, ...]
        The displayed **world** axes, in display order.

    Returns
    -------
    tuple[int, ...]
        The corresponding **data** axes, in the same order.

    Raises
    ------
    ValueError
        If a displayed world axis has no data axis feeding it -- a
        multiscale visual cannot render along a broadcast axis, since there
        is no data varying along it to draw.
    """
    if transform is None:
        return displayed_axes
    correspondence = axis_correspondence(transform)
    world_to_data = {world: data for data, world in correspondence.items()}
    missing = [axis for axis in displayed_axes if axis not in world_to_data]
    if missing:
        raise ValueError(
            f"World axis/axes {missing} are displayed but no data axis of "
            f"this visual maps to them (they are broadcast).  A multiscale "
            f"visual has no data to render along a broadcast axis."
        )
    return tuple(world_to_data[axis] for axis in displayed_axes)


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
        **Full-rank** per-level transforms mapping level-k voxel coords to
        level-0.  ``level_transforms[0]`` must be the identity.
    block_size : int
        Rendering brick side length in voxels.
    fetch_axes : tuple[int, ...]
        The ascending data axes this grid is built over, which is the order
        the fetched array carries (:func:`_fetch_order`).  The transforms are
        projected onto them here; before v1 was retired the caller projected
        them with ``select_axes`` and passed a rank-reduced list.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, ...]],
        level_transforms: list[AffineTransform],
        block_size: int,
        fetch_axes: tuple[int, ...],
    ) -> None:
        self.level_transforms = list(level_transforms)
        self.fetch_axes = tuple(fetch_axes)
        self.block_size = block_size
        self.n_levels = len(level_shapes)

        identity = np.asarray(level_transforms[0].matrix, dtype=np.float64)
        assert np.allclose(identity, np.eye(identity.shape[0])), (
            "level_transforms[0] must be the identity"
        )

        # Vectors come back in fetch order over the displayed subset
        # (e.g. (z, y, x) when displayed_axes=(0, 1, 2)).
        sv_data, tv_data = _level_scale_and_translation(
            level_transforms, self.fetch_axes
        )
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
        **Full-rank** per-level transforms mapping level-k voxel coords to
        level-0.
    fetch_axes : tuple[int, ...]
        The ascending data axes this grid is built over.  The transforms are
        projected onto them here; before v1 was retired the caller projected
        them with ``select_axes`` and passed a rank-reduced list.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, int]],
        block_size: int,
        n_levels: int,
        level_transforms: list[AffineTransform],
        fetch_axes: tuple[int, ...],
    ) -> None:
        self.block_size = block_size
        self.n_levels = n_levels
        self.level_shapes = list(level_shapes)
        self.level_transforms = list(level_transforms)
        self.fetch_axes = tuple(fetch_axes)

        # Vectors come back in fetch order over the 2 displayed axes
        # (e.g. (H, W) when displayed_axes=(1, 2)).
        sv_data, tv_data = _level_scale_and_translation(
            self.level_transforms, self.fetch_axes
        )
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


def _fetch_order(displayed_axes: tuple[int, ...]) -> tuple[int, ...]:
    """The displayed axes in the order the fetched array carries them.

    ``displayed_axes`` is a **display** order.  ``axis_selections`` is
    assembled per data axis ascending, and ``get_data`` hands back an array
    whose axes are therefore ascending -- numpy fixes that and nothing
    downstream can negotiate it.  So a brick grid, a level shape projection
    and a per-level transform projection all index the *fetch* order, and the
    display permutation lives entirely in the node matrix (design 3.14).

    Before this, the grid was built in display order and a transposed tuple
    rotated it against its own uploaded array.

    Parameters
    ----------
    displayed_axes : tuple[int, ...]
        The displayed axes, in display order.

    Returns
    -------
    tuple[int, ...]
        The same axes, ascending.
    """
    return tuple(sorted(displayed_axes))


# ---------------------------------------------------------------------------
# GFXMultiscaleImageVisual
# ---------------------------------------------------------------------------


class MultiscaleRegionPlanner:
    """Pulls the selected region back to each pyramid level (design 3.11 A).

    Mixed into both multiscale families, which are near-copies of each other
    everywhere else too.

    The pull-back composes cleanly: level-0 first, through the visual's own
    ``data -> world``, then each level's own transform.  Every step is
    ``imap_region``, which is ``A^T`` on the normals and needs no inverse at
    all (D39) -- so this replaces ``_build_world_to_level_transforms`` and the
    precomputed ``inv_level_k @ inv_visual`` list it held.

    The per-level answer now falls out of the per-level transform rather than
    from a ``2 ** (level - 1)`` assumption applied to every axis.  That
    assumption is the one this repo has already paid for once, on a pyramid
    whose ``z`` was not downsampled.

    Boxes are memoised for the duration of one planning call: a frame plans
    hundreds of bricks across a handful of levels, and each level's answer is
    the same for all of them.
    """

    def _begin_region_planning(self, selection) -> None:
        """Start a planning call: adopt the selection and drop the memo."""
        self._selection = selection
        self._level0_region = None
        self._level_boxes = {}

    def _level_box(self, level_index: int):
        """The selection pulled back to level *level_index*, as a box.

        ``None`` when there is nothing to pull back -- no region, or a visual
        the controller has not placed -- in which case the caller falls back to
        the ``dims_state`` path.
        """
        selection = getattr(self, "_selection", None)
        spaces = self._spaces
        if selection is None or spaces is None or self._transform is None:
            return None
        if level_index >= len(spaces.level_transforms):
            return None
        cached = self._level_boxes.get(level_index)
        if cached is not None:
            return cached
        if self._level0_region is None:
            self._level0_region = self._transform.imap_region(
                selection.region, spaces.world
            )
        box = (
            spaces.level_transforms[level_index]
            .imap_region(self._level0_region, spaces.data)
            .simplify()
            .bounding_box()
        )
        self._level_boxes[level_index] = box
        return box

    def _level0_axis_selections(self) -> tuple[int | tuple[int, int], ...] | None:
        """The level-0 selection this plan fetches, per data axis.

        ``axis_selections_from_box`` on the memoised level-0 box, so the same
        round-half-up and the same clamp the fetch itself applies.

        Returns
        -------
        tuple[int | tuple[int, int], ...] or None
            One entry per data axis, ascending.  ``None`` when the visual has
            nothing planned.
        """
        box = self._level_box(0)
        if box is None:
            return None
        level_shape = getattr(self, "_full_level_shapes", None)
        if not level_shape:
            return None
        return axis_selections_from_box(box, tuple(level_shape[0]))

    def _block_key_slice_coord(self) -> tuple[tuple[int, int | tuple[int, int]], ...]:
        """Where this plan's collapsed axes sit, for the block cache key.

        Bricks from different slice positions must not collide in the cache,
        so the position is embedded in every ``BlockKey``.  The key is the
        level-0 selection the fetch actually uses on each collapsed axis -- an
        integer plane, or a ``(start, stop)`` window for a slab -- not the
        continuous pulled-back position.  Two slider positions that fetch the
        same plane therefore share a key and reuse each other's bricks.

        A continuous key re-fetched every visible brick on every sub-plane
        slider move; on a non-uniform time axis spanning hundreds of thousands
        of seconds over a few hundred frames that was nearly every move.  See
        ``docs/Explanations/multiscale_brick_lookup.md``.

        Returns
        -------
        tuple[tuple[int, int | tuple[int, int]], ...]
            ``(data axis, selection)`` pairs for the collapsed axes, sorted by
            axis.  Empty when the visual has nothing planned.
        """
        selections = self._level0_axis_selections()
        if selections is None or self._spaces is None:
            return ()
        retained = set(self._spaces.retained_axes)
        return tuple(
            (axis, value)
            for axis, value in enumerate(selections)
            if axis not in retained
        )

    def _level_slice_selection(
        self, level_index: int, fill: dict[int, int] | None = None
    ) -> tuple[int | tuple[int, int] | None, ...]:
        """The collapsed-axis selection level *level_index* fetches (design 5.1).

        What a brick key's slice id interns: the plane or slab the fetch
        reads on each collapsed axis at this level, ``None`` on the displayed
        axes (their windows come from the brick grid).  Per level, so two
        level-0 planes that read the same coarse plane share coarse bricks.

        Parameters
        ----------
        level_index : int
            0-based level.
        fill : dict[int, int] or None
            Per-axis overrides, e.g. a composite channel's own index on the
            channel axis.

        Returns
        -------
        tuple
            One entry per data axis.

        Raises
        ------
        RuntimeError
            If the level has no pulled-back region (no selection, or a visual
            not placed in a world): there is no fallback (R8.3).
        """
        box = self._level_box(level_index)
        if box is None:
            raise RuntimeError(
                f"Level {level_index} of this visual has no pulled-back "
                f"region, so its bricks cannot be addressed.  Either the "
                f"visual has not been placed in a world or the reslicing "
                f"request carried no region."
            )
        selection = list(
            axis_selections_from_box(box, tuple(self._full_level_shapes[level_index]))
        )
        for axis in self._spaces.retained_axes:
            selection[axis] = None
        for axis, value in (fill or {}).items():
            selection[axis] = value
        return tuple(selection)

    def pick_collapsed_indices(self) -> dict[int, int] | None:
        """The level-0 planes this visual last drew, per collapsed data axis.

        Answers the pick path's question -- "which plane is the user looking
        at?" -- from the plan rather than from the dims state, so the answer
        cannot drift from the screen while a reslice is in flight.

        Derived from the level-0 box rather than stored, because these visuals
        plan per level and per brick; ``_level_box`` is already memoised for
        the planning call, and ``axis_selections_from_box`` applies the same
        round-half-up and the same clamp the fetch itself used.  The
        ``_collapsed_indices`` attribute both multiscale families declare is
        deliberately left alone: it feeds the node matrix, and populating it
        here would move geometry rather than answer a pick.

        Returns
        -------
        dict[int, int] or None
            Data axis to level-0 voxel index, for collapsed axes only.
            ``None`` when the visual has nothing planned to report.
        """
        selections = self._level0_axis_selections()
        if selections is None:
            return None
        return {
            axis: int(value)
            for axis, value in enumerate(selections)
            if not isinstance(value, tuple)
        }

    def _to_level0_displayed(self, points: np.ndarray) -> np.ndarray:
        """Map pygfx-order points to level-0 voxels on the displayed axes.

        Returns them back in pygfx order, which is what the LOD thresholds,
        the distance sort and the frustum culler take.  The pulled-back point
        also carries the collapsed axes -- the ``t`` the camera is looking
        at, say -- which are harmless and dropped here.

        Design 3.11 C.  The camera and the frustum arrive in **pygfx**
        coordinates -- ``ReslicingRequest`` says so in a docstring and nowhere
        in the type system -- so they are reversed once, by name, mapped
        **forward** to world through ``rendered_to_world`` (always affine, so
        always a matrix), and pulled **back** to level-0 data through
        ``imap_coordinates`` rather than a materialized inverse.

        That second step used to be one composed matrix,
        ``rendered_to_world.then(data_to_world.inverse())``, built from
        ``data_to_world.inverse()`` directly.  A ``ByDimensionTransform``
        carrying a ``NonUniformAxisTransform`` block has no such inverse *as
        a transform* -- the map is invertible, but its inverse is not itself
        "a world position per data index", which is what that class models
        (design doc, "the leaf has no inverse *transform*").  Every such
        transform still answers ``imap_coordinates`` directly, which is the
        one path guaranteed to exist regardless of what blocks the transform
        holds, so this composes in two steps instead of one -- exactly the
        two-step shape ``data_slice_positions`` and every other post-Phase-4
        consumer already uses, and the class of bug the one-step matrix
        version reintroduced for any visual sharing the time axis.

        Raises
        ------
        RuntimeError
            If the visual has not been placed in a world.
        """
        spaces = self._spaces
        if spaces is None or self._transform is None:
            raise RuntimeError(
                "This visual has not been placed in a world, so a camera "
                "position cannot be pulled back to a voxel index.  A "
                "multiscale visual must be given its coordinate systems "
                "before it is planned."
            )
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, points.shape[-1])
        world = spaces.rendered_to_world.map_coordinates(pygfx_to_cellier_order(flat))
        level0 = self._transform.imap_coordinates(world)
        displayed = level0[:, list(spaces.retained_axes)]
        return cellier_to_pygfx_order(displayed).reshape(points.shape)

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
        (cx0, cy0), (cx1, cy1) = base_cell_range(view_min[:2], view_max[:2], block_size)
        return (
            max(0, int(cy0)),
            max(0, int(cx0)),
            min(gh_grid, int(cy1)),
            min(gw_grid, int(cx1)),
        )

    def _adopt_viewport_2d(
        self,
        camera_pos_world: np.ndarray,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
    ) -> None:
        """Set the background clip for a plan that skips the target (2D).

        ``_plan_tiles_2d`` sets ``_current_viewport_cells`` as it culls; a
        backstop-only plan does not run it, so it takes the viewport here.
        """
        geo2d = self._image_geometry_2d
        if geo2d is None:
            return
        _, view_min, view_max, _ = self._view_2d(
            camera_pos_world, world_width, view_min_world, view_max_world
        )
        self._current_viewport_cells = self._viewport_cells_from_view_bounds(
            view_min, view_max, geo2d.block_size
        )

    def _plan_backstop_3d(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        loading: ProgressiveLoadingConfig,
    ) -> np.ndarray | None:
        """The backstop bricks for this view, nearest first (design 5.9).

        ``None`` when the backstop is off.  Extent ``"view"`` culls to the
        request's frustum whether or not the target is frustum-culled.
        """
        geo = self._volume_geometry
        if not loading.backstop or geo is None:
            return None
        camera_pos_data = self._to_level0_displayed(
            np.asarray(camera_pos_world).reshape(1, -1)
        ).flatten()
        planes = None
        if loading.backstop_extent == "view" and frustum_corners_world is not None:
            planes = frustum_planes_from_corners(
                self._to_level0_displayed(frustum_corners_world)
            )
        return backstop_bricks_3d(
            geo._level_grids,
            backstop_level(loading, geo.n_levels),
            camera_pos_data,
            geo.block_size,
            geo._scale_arr_shader,
            geo._translation_arr_shader,
            frustum_planes=planes,
        )

    def _plan_backstop_2d(
        self,
        camera_pos_world: np.ndarray,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
        loading: ProgressiveLoadingConfig,
    ) -> np.ndarray | None:
        """The backstop tiles for this view, centre first (design 5.9).

        ``None`` when the backstop is off.  Extent ``"view"`` culls to the
        viewport plus one backstop tile, whether or not the target is
        viewport-culled.
        """
        geo2d = self._image_geometry_2d
        if not loading.backstop or geo2d is None:
            return None
        camera_pos, view_min, view_max, _ = self._view_2d(
            camera_pos_world, world_width, view_min_world, view_max_world
        )
        if loading.backstop_extent != "view":
            view_min = view_max = None
        return backstop_tiles_2d(
            geo2d._level_grids,
            backstop_level(loading, geo2d.n_levels),
            camera_pos,
            geo2d.block_size,
            geo2d._scale_arr_shader,
            geo2d._translation_arr_shader,
            view_min=view_min,
            view_max=view_max,
        )


def _plan_stats(n_target: int, desired: DesiredSet) -> dict:
    """What one plan asked for: ``total_required`` target bricks, the rest kept.

    ``total_required`` counts the target before truncation; ``n_backstop``
    and ``n_target`` count the desired set's two classes after it.
    """
    n_backstop = int((desired.cls == ChunkClass.BACKSTOP).sum())
    return {
        "total_required": int(n_target),
        "n_backstop": n_backstop,
        "n_target": len(desired.keys) - n_backstop,
        "n_truncated_target": desired.n_truncated_target,
        "n_truncated_backstop": desired.n_truncated_backstop,
    }


class _MultiscaleImageSlot(MultiscaleRegionPlanner):
    """One channel's GPU resources on a multiscale image visual.

    Owns GPU resources (brick caches, LUT textures, pygfx nodes) for both
    2D and 3D rendering, and the chunk scheduler's adapter for each atlas
    (``residency_3d`` / ``residency_2d``).  The public
    :class:`GFXMultiscaleImageVisual` plans once and turns the plan into a
    desired set per drawn channel through its slots (unified image design
    3.8).  Appearance
    is applied by that wrapper: a slot builds its nodes with neutral
    defaults.

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
    #: Applies the image slicing rule (design 3.2) itself, so the scene
    #: manager's data-coverage pre-check does not skip it.
    decides_empty_slices: bool = True

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

        # The data -> world transform.  There is no coordinate-system-less
        # identity to fall back on (D18); the controller supplies one.
        self._transform: AffineTransform | None = transform
        # The systems this visual's geometry is placed with, pushed by the
        # controller.  ``None`` until the scene has a canvas.
        self._spaces: RenderSpaces | None = None
        # The collapsed voxel index per dropped data axis, refreshed on every
        # planned request (design 3.9).
        self._collapsed_indices: dict[int, float] = {}
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

        # Full-ndim level shapes, which ``axis_selections_from_box`` clamps
        # each collapsed axis against.
        if full_level_shapes is not None:
            self._full_level_shapes = list(full_level_shapes)
        elif volume_geometry is not None:
            self._full_level_shapes = list(volume_geometry.level_shapes)
        elif image_geometry_2d is not None:
            self._full_level_shapes = list(image_geometry_2d.level_shapes)
        else:
            self._full_level_shapes = []

        # Track displayed axes for node matrix updates.
        self._last_displayed_axes: tuple[int, ...] | None = displayed_axes
        self._gpu_budget_bytes = gpu_budget_bytes_3d
        self._frame_number = 0
        self._last_plan_stats: dict = {}
        # The chunk scheduler's view of the 3D atlas; rebuilt whenever the
        # atlas, its LUT or the displayed axes change (a new cache id).
        self._residency_3d: ImageResidency3D | None = None
        self._residency_axes: tuple[int, ...] | None = None
        # Called on this slot's first 3D brick (the wrapper reveals the
        # visual's one bounding box).
        self._data_ready_listener = None
        # The 2D atlas's adapter; rebuilt when the atlas or its LUT changes.
        self._residency_2d: ImageResidency2D | None = None
        self._data_ready_listener_2d = None

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
        # Kept so a node built lazily on first entry into a mode writes pick.
        self._pick_write: bool = pick_write
        # True while a sliced axis selects no level-0 sample (design 3.2): the
        # inner data nodes are hidden and nothing is planned.
        self._slice_empty: bool = False
        # Data axes the image slicing rule skips.  A multichannel wrapper sets
        # its channel axis here, because each of its requests overwrites it.
        self._unsliced_data_axes: tuple[int, ...] = ()
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
                level_translation_vecs_data=volume_geometry._translation_vecs_data,
                sampling_margin=_IMAGE_SAMPLING_MARGIN_3D,
                level_shapes=volume_geometry.level_shapes,
                border=cache_parameters_3d.overlap,
            )

        # ── 2D GPU resources (only when image_geometry_2d is provided) ─
        self._block_cache_2d: BlockCache2D | None = None
        self._lut_manager_2d: LutIndirectionManager2D | None = None
        self._lut_params_buffer_2d = None
        self._block_scales_buffer_2d = None
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
                overlap=TILE_BORDER_2D,
            )
            self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
            self._lut_manager_2d = LutIndirectionManager2D(
                base_layout=image_geometry_2d.base_layout,
                n_levels=image_geometry_2d.n_levels,
                scale_vecs_data=image_geometry_2d._scale_vecs_data,
                translation_vecs_data=image_geometry_2d._translation_vecs_data,
                sampling_margin=_IMAGE_SAMPLING_MARGIN_2D,
                level_shapes=image_geometry_2d.level_shapes,
                border=cache_parameters_2d.overlap,
            )
            self._lut_params_buffer_2d = build_lut_params_buffer_2d(
                image_geometry_2d.base_layout, cache_parameters_2d
            )
            self._block_scales_buffer_2d = build_block_scales_buffer_2d(
                level_scale_vecs_data=image_geometry_2d._scale_vecs_data,
                level_shapes=image_geometry_2d.level_shapes,
                block_size=image_geometry_2d.block_size,
                level_translation_vecs_data=image_geometry_2d._translation_vecs_data,
            )

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
                volume_geometry._scale_vecs_data,
                level_shapes=volume_geometry.level_shapes,
                block_size=volume_geometry.block_size,
                level_translation_vecs_data=volume_geometry._translation_vecs_data,
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
                fetch_3d = _fetch_order(
                    _world_axes_to_data_axes(self._transform, displayed_axes)
                )
                shapes_3d = [tuple(s[ax] for ax in fetch_3d) for s in level_shapes]
                if shapes_3d != self._volume_geometry.level_shapes:
                    self._volume_geometry.update(shapes_3d)
                    self._rebuild_3d_resources()
            new_node = self.node_3d

        if "2d" in self.render_modes and len(displayed_axes) == 2:
            old_node = self.node_2d
            if self._image_geometry_2d is None:
                self._lazy_init_2d(displayed_axes)
            else:
                fetch_2d = _fetch_order(
                    _world_axes_to_data_axes(self._transform, displayed_axes)
                )
                shapes_2d_full = [tuple(s[ax] for ax in fetch_2d) for s in level_shapes]
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
            fetch_3d = _fetch_order(
                _world_axes_to_data_axes(self._transform, displayed_axes)
            )
            shapes_3d = [tuple(s[ax] for ax in fetch_3d) for s in level_shapes]
            if shapes_3d != self._volume_geometry.level_shapes:
                self._volume_geometry.update(shapes_3d)
                self._rebuild_3d_resources()
            return self.node_3d
        if mode == "2d" and self._image_geometry_2d is not None:
            fetch_2d = _fetch_order(
                _world_axes_to_data_axes(self._transform, displayed_axes)
            )
            shapes_2d_full = [tuple(s[ax] for ax in fetch_2d) for s in level_shapes]
            shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
            if shapes_2d != self._image_geometry_2d.level_shapes:
                self._image_geometry_2d.update(shapes_2d)
                self._rebuild_2d_resources()
            return self.node_2d
        return None

    def _lazy_init_3d(self, displayed_axes: tuple[int, ...], visual_model=None) -> None:
        """Build 3D GPU resources on first entry into 3D mode.

        Built with neutral appearance defaults; the wrapper applies the
        model's appearance afterwards.  *visual_model* is accepted for the
        protocol and ignored.

        Sets ``node_3d.local.matrix`` directly — does NOT call
        ``_update_node_matrix`` so ``node_2d`` (if already built) is left
        unchanged.
        """
        threshold = 0.5
        attenuation = 1.0
        render_mode = "mip"
        pick_write = self._pick_write

        fetch_axes = _fetch_order(
            _world_axes_to_data_axes(self._transform, displayed_axes)
        )
        shapes_3d = [select_axes(s, fetch_axes) for s in self._full_level_shapes]
        self._volume_geometry = MultiscaleBrickLayout3D(
            level_shapes=shapes_3d,
            level_transforms=self._level_transforms,
            block_size=self._block_size,
            fetch_axes=fetch_axes,
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
            level_translation_vecs_data=self._volume_geometry._translation_vecs_data,
            sampling_margin=_IMAGE_SAMPLING_MARGIN_3D,
            level_shapes=self._volume_geometry.level_shapes,
            border=cache_parameters_3d.overlap,
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
            self._volume_geometry._scale_vecs_data,
            level_shapes=self._volume_geometry.level_shapes,
            block_size=self._volume_geometry.block_size,
            level_translation_vecs_data=self._volume_geometry._translation_vecs_data,
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
        self._last_displayed_axes = displayed_axes
        if self._spaces is not None and self._transform is not None:
            self.node_3d.local.matrix = self._node_matrices()[0]
        self.node_3d.visible = self._visible

    def _lazy_init_2d(self, displayed_axes: tuple[int, ...], visual_model=None) -> None:
        """Build 2D GPU resources on first entry into 2D mode.

        Built with neutral appearance defaults; the wrapper applies the
        model's appearance afterwards.  *visual_model* is accepted for the
        protocol and ignored.

        Sets ``node_2d.local.matrix`` directly — does NOT call
        ``_update_node_matrix`` so ``node_3d`` (if already built) is left
        unchanged.
        """
        interpolation = "linear"
        pick_write = self._pick_write

        fetch_axes = _fetch_order(
            _world_axes_to_data_axes(self._transform, displayed_axes)
        )
        shapes_2d_full = [select_axes(s, fetch_axes) for s in self._full_level_shapes]
        shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
        self._image_geometry_2d = ImageGeometry3D(
            level_shapes=shapes_2d,
            block_size=self._block_size,
            n_levels=len(self._full_level_shapes),
            level_transforms=self._level_transforms,
            fetch_axes=fetch_axes,
        )
        cache_parameters_2d = compute_block_cache_parameters_2d(
            gpu_budget_bytes=self._gpu_budget_bytes_2d,
            block_size=self._image_geometry_2d.block_size,
            overlap=TILE_BORDER_2D,
        )
        self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=self._image_geometry_2d.base_layout,
            n_levels=self._image_geometry_2d.n_levels,
            scale_vecs_data=self._image_geometry_2d._scale_vecs_data,
            translation_vecs_data=self._image_geometry_2d._translation_vecs_data,
            sampling_margin=_IMAGE_SAMPLING_MARGIN_2D,
            level_shapes=self._image_geometry_2d.level_shapes,
            border=cache_parameters_2d.overlap,
        )
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            self._image_geometry_2d.base_layout, cache_parameters_2d
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            level_scale_vecs_data=self._image_geometry_2d._scale_vecs_data,
            level_shapes=self._image_geometry_2d.level_shapes,
            block_size=self._image_geometry_2d.block_size,
            level_translation_vecs_data=self._image_geometry_2d._translation_vecs_data,
        )
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
        self._last_displayed_axes = displayed_axes
        if self._spaces is not None and self._transform is not None:
            self.node_2d.local.matrix = self._node_matrices()[1]
        self.node_2d.visible = self._visible

    def _rebuild_3d_resources(self) -> None:
        """Rebuild 3D GPU resources after geometry update."""
        geo = self._volume_geometry
        # The bricks were keyed on the old geometry: a new LUT manager makes
        # residency_3d() start a new atlas registry (a new cache id).
        self._block_cache_3d.clear()
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=geo.base_layout,
            n_levels=geo.n_levels,
            level_scale_vecs_data=geo._scale_vecs_data,
            level_translation_vecs_data=geo._translation_vecs_data,
            sampling_margin=_IMAGE_SAMPLING_MARGIN_3D,
            level_shapes=geo.level_shapes,
            border=self._block_cache_3d.info.overlap,
        )
        # The brick counts in the scales buffer follow the level shapes.
        self._brick_scales_buffer = build_brick_scales_buffer(
            geo._scale_vecs_data,
            level_shapes=geo.level_shapes,
            block_size=geo.block_size,
            level_translation_vecs_data=geo._translation_vecs_data,
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

    def _rebuild_2d_resources(self) -> None:
        """Rebuild 2D GPU resources after geometry update."""
        geo2d = self._image_geometry_2d
        # A new LUT means a new residency (and cache id) on the next plan.
        self._block_cache_2d.clear()
        # Rebuild LUT manager
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=geo2d.base_layout,
            n_levels=geo2d.n_levels,
            scale_vecs_data=geo2d._scale_vecs_data,
            translation_vecs_data=geo2d._translation_vecs_data,
            sampling_margin=_IMAGE_SAMPLING_MARGIN_2D,
            level_shapes=geo2d.level_shapes,
            border=self._block_cache_2d.info.overlap,
        )
        # Rebuild param buffers
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            geo2d.base_layout, self._block_cache_2d.info
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            level_scale_vecs_data=geo2d._scale_vecs_data,
            level_shapes=geo2d.level_shapes,
            block_size=geo2d.block_size,
            level_translation_vecs_data=geo2d._translation_vecs_data,
        )
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

    def set_render_spaces(self, spaces: RenderSpaces | None) -> None:
        """Receive the coordinate systems this visual is placed with."""
        self._spaces = spaces
        if spaces is not None and self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def _collapsed_origin(self) -> dict[int, float]:
        """Where the dropped data axes sit, for the node matrix (design 3.9)."""
        return {
            axis: float(self._collapsed_indices.get(axis, 0.0))
            for axis in self._spaces.collapsed_axes
        }

    def _node_matrices(self) -> tuple[object, object]:
        """``(matrix for the 3D node, matrix for the 2D node)``.

        Two matrices because the two nodes are indexed in different spaces:
        the 3D node in the normalized proxy box, the 2D node in level-0
        pixels.  Either entry is ``None`` when the systems cannot express it.
        """
        plain = node_matrix(self._spaces, self._transform, self._collapsed_origin())
        if len(self._spaces.retained_axes) != 3 or self._norm_size is None:
            return plain, plain
        scale, translation = _norm_to_data_params_for(
            self._spaces, self._dataset_size, self._norm_size
        )
        composed = node_matrix(
            self._spaces,
            self._transform,
            self._collapsed_origin(),
            scale,
            translation,
        )
        return composed, plain

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Recompute and apply the pygfx node matrices for *displayed_axes*.

        Design 3.9's composition, ``visual -> data -> world -> rendered``,
        replacing ``select_axes`` plus a hand-built ``norm_to_data``.  The
        reversal now happens once, at the end, instead of in the middle.
        A no-op until the controller supplies the systems.
        """
        self._last_displayed_axes = displayed_axes
        if self._spaces is None or self._transform is None:
            return
        composed, plain = self._node_matrices()
        if self.node_3d is not None:
            self.node_3d.local.matrix = composed
        if self.node_2d is not None:
            self.node_2d.local.matrix = plain

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

        Pure with respect to the GPU: it reads the camera and the level
        grids and returns an array.  ``desired_set_3d`` turns the array into
        the atlas's desired set for the chunk scheduler.

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

        camera_pos_data = self._to_level0_displayed(
            np.asarray(camera_pos_world).reshape(1, -1)
        ).flatten()

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
            corners_data = self._to_level0_displayed(frustum_corners_world)
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

        # Truncation to the atlas's budget happens in desired_set_3d, after
        # the backstop has taken its share (design 5.3).
        return brick_arr

    def residency_3d(self) -> ImageResidency3D | None:
        """The chunk scheduler's adapter for this slot's 3D atlas.

        ``None`` until the 3D resources exist.  A new adapter (with a new
        cache id) replaces the old one when the atlas, its LUT or the
        displayed axes change: keys from the old layout mean nothing in the
        new one, so the scheduler drops the old registry.
        """
        if (
            self._block_cache_3d is None
            or self._lut_manager_3d is None
            or self._volume_geometry is None
        ):
            return None
        residency = self._residency_3d
        if (
            residency is None
            or residency.block_cache is not self._block_cache_3d
            or residency.lut_manager is not self._lut_manager_3d
            or (
                # Only a 3D layout change re-keys the atlas; 2D leaves it be.
                len(self._last_displayed_axes or ()) == 3
                and self._residency_axes != self._last_displayed_axes
            )
        ):
            self._block_cache_3d.clear()
            residency = make_residency_3d(
                self._block_cache_3d,
                self._lut_manager_3d,
                self._volume_geometry.block_size,
                self._level_transforms,
                self._ndim,
                brick_max=_brick_max_image,
                on_write=self._on_brick_written_3d,
            )
            self._residency_3d = residency
            if len(self._last_displayed_axes or ()) == 3:
                self._residency_axes = self._last_displayed_axes
        return residency

    def _on_brick_written_3d(self) -> None:
        """Reveal the bounding box once the first brick is on the GPU."""
        if self._data_ready_3d:
            return
        self._data_ready_3d = True
        if self._aabb_line_3d is not None:
            self._aabb_line_3d.visible = self._aabb_enabled
        if self._data_ready_listener is not None:
            self._data_ready_listener()

    def desired_set_3d(
        self,
        brick_arr: np.ndarray | None,
        fill: dict[int, int] | None = None,
        backstop_arr: np.ndarray | None = None,
        loading: ProgressiveLoadingConfig | None = None,
    ) -> DesiredSet | None:
        """The desired set of this slot's atlas for a planned brick array (5.3).

        Parameters
        ----------
        brick_arr : np.ndarray or None
            Output of ``_plan_bricks`` (the target), from this slot or the
            wrapper's planning slot.  ``None`` plans no target.
        fill : dict[int, int] or None
            Selection overrides, e.g. ``{channel_axis: channel}``.
        backstop_arr : np.ndarray or None
            Output of ``_plan_backstop_3d``.
        loading : ProgressiveLoadingConfig or None
            For the backstop cap.

        Returns
        -------
        DesiredSet or None
            ``None`` without 3D resources.
        """
        residency = self.residency_3d()
        if residency is None:
            return None
        return desired_bricks(
            self,
            residency,
            brick_arr,
            backstop_arr=backstop_arr,
            backstop_cap=backstop_cap_for(loading, residency),
            fill=fill,
        )

    # ── 2D planning helpers ───────────────────────────────────────────

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
    ) -> np.ndarray:
        """Run 2D LOD selection, sort and viewport cull.

        Pure with respect to the GPU.  Sets ``_current_viewport_cells`` (the
        background clip); ``desired_set_2d`` truncates to the atlas and
        turns the array into its desired set.

        Returns
        -------
        np.ndarray
            ``(N, 3)`` rows ``[level, g0, g1]`` in load order (nearest the
            camera first).
        """
        geo2d = self._image_geometry_2d
        if geo2d is None or self._block_cache_2d is None:
            return np.empty((0, 3), dtype=np.int64)

        block_size = geo2d.block_size
        n_levels = geo2d.n_levels
        camera_pos, view_min, view_max, voxel_width = self._view_2d(
            camera_pos_world, world_width, view_min_world, view_max_world
        )
        if not use_culling:
            view_min = view_max = None

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

        n_total = len(tile_arr)

        # 3. Viewport culling
        if view_min is not None:
            tile_arr, _ = viewport_cull_2d(
                tile_arr,
                block_size,
                view_min,
                view_max,
                level_scale_arr_shader=geo2d._scale_arr_shader,
                level_translation_arr_shader=geo2d._translation_arr_shader,
            )
        self._last_plan_stats = {"total_required": n_total, "n_needed": len(tile_arr)}
        return tile_arr

    def _view_2d(
        self,
        camera_pos_world: np.ndarray,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
        """The 2D view in level-0 data space.

        Returns
        -------
        camera_pos : np.ndarray
            ``(x, y, 0)``, the canvas centre.
        view_min, view_max : np.ndarray or None
            The viewport's bounds ``(x, y)``; ``None`` without a viewport.
        voxel_width : float
            Visible width in level-0 voxels, for LOD selection.
        """
        displayed = self._last_displayed_axes
        sub_2d = _displayed_submatrix(self._transform, displayed)

        nd2 = sub_2d.shape[0] - 1
        world_units_per_voxel_2d = np.abs(np.diag(sub_2d[:nd2, :nd2]))
        world_to_voxel_scale_2d = float(
            np.prod(1.0 / world_units_per_voxel_2d) ** (1.0 / nd2)
        )
        voxel_width = world_width * world_to_voxel_scale_2d

        camera_pos_2d = _imap_square(
            sub_2d, camera_pos_world[[1, 0]].reshape(1, -1)
        ).flatten()[[1, 0]]
        camera_pos = np.array(
            [camera_pos_2d[0], camera_pos_2d[1], 0.0], dtype=np.float32
        )

        if view_min_world is not None and view_max_world is not None:
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
            corners_data_2d = _imap_square(sub_2d, corners_world_2d)[:, [1, 0]]
            view_min = corners_data_2d.min(axis=0)
            view_max = corners_data_2d.max(axis=0)
        else:
            view_min = None
            view_max = None
        return camera_pos, view_min, view_max, voxel_width

    def residency_2d(self) -> ImageResidency2D | None:
        """The chunk scheduler's adapter for this slot's 2D atlas.

        ``None`` until the 2D resources exist.  A new adapter (with a new
        cache id) replaces the old one when the atlas or its LUT changes.
        Displayed axes need no re-key: a tile's slice id records which axes
        were collapsed.
        """
        if self._block_cache_2d is None or self._lut_manager_2d is None:
            return None
        residency = self._residency_2d
        if (
            residency is None
            or residency.block_cache is not self._block_cache_2d
            or residency.lut_manager is not self._lut_manager_2d
        ):
            self._block_cache_2d.clear()
            residency = make_residency_2d(
                self._block_cache_2d,
                self._lut_manager_2d,
                self._image_geometry_2d.block_size,
                self._level_transforms,
                self._ndim,
                on_write=self._on_tile_written_2d,
            )
            self._residency_2d = residency
        return residency

    def _on_tile_written_2d(self) -> None:
        """Reveal the bounding box once the first tile is on the GPU."""
        if self._data_ready_2d:
            return
        self._data_ready_2d = True
        if self._aabb_line_2d is not None:
            self._aabb_line_2d.visible = self._aabb_enabled
        if self._data_ready_listener_2d is not None:
            self._data_ready_listener_2d()

    def desired_set_2d(
        self,
        tile_arr: np.ndarray | None,
        fill: dict[int, int] | None = None,
        backstop_arr: np.ndarray | None = None,
        loading: ProgressiveLoadingConfig | None = None,
    ) -> DesiredSet | None:
        """The desired set of this slot's 2D atlas for a planned tile array.

        Parameters
        ----------
        tile_arr : np.ndarray or None
            Output of ``_plan_tiles_2d`` (the target), from this slot or the
            wrapper's planning slot.  ``None`` plans no target.
        fill : dict[int, int] or None
            Selection overrides, e.g. ``{channel_axis: channel}``.
        backstop_arr : np.ndarray or None
            Output of ``_plan_backstop_2d``.
        loading : ProgressiveLoadingConfig or None
            For the backstop cap.

        Returns
        -------
        DesiredSet or None
            ``None`` without 2D resources.
        """
        residency = self.residency_2d()
        if residency is None:
            return None
        residency.viewport_cells = self._current_viewport_cells
        return desired_bricks(
            self,
            residency,
            tile_arr,
            backstop_arr=backstop_arr,
            backstop_cap=backstop_cap_for(loading, residency),
            fill=fill,
        )

    # ── EventBus handler methods ─────────────────────────────────────────

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update stored transform, norm_size uniform, and pygfx node matrix."""
        self._transform = event.transform

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
        elif event.field_name == "ray_steps_per_voxel":
            if self.material_3d is not None:
                self.material_3d.ray_steps_per_voxel = float(event.new_value)
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

    def _begin_region_planning(self, selection) -> None:
        """Start a planning call, applying the image slicing rule first.

        Design 3.2, decided on **level 0**: each sliced axis draws its nearest
        level-0 sample within the thickness, and if any sliced axis has none
        the visual draws nothing -- ``_slice_empty`` is set and the inner data
        nodes are hidden until a later plan lands in the data.

        Otherwise planning continues from a selection whose sliced slabs are
        planes at the slice position.  Level 0 rounds that position to the
        chosen sample; coarser levels keep their own rounding of it.  Labels
        share the base planner and keep clamping.
        """
        planned = selection
        empty = False
        if (
            selection is not None
            and self._spaces is not None
            and self._transform is not None
            and self._full_level_shapes
        ):
            planned = image_plane_selection(
                selection,
                self._transform,
                self._spaces,
                tuple(self._full_level_shapes[0]),
                exempt_data_axes=self._unsliced_data_axes,
            )
            empty = planned is None
        self._slice_empty = empty
        for inner in (
            getattr(self, "_inner_node_2d", None),
            getattr(self, "_inner_node_3d", None),
        ):
            if inner is not None:
                inner.visible = not self._slice_empty
        super()._begin_region_planning(selection if planned is None else planned)

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

    def close(self) -> None:
        """Free this slot's GPU textures and drop its caches.  Unusable afterwards.

        The textures are destroyed rather than dropped: pygfx's bind-group
        cache would otherwise keep their GPU memory after the slot is gone.
        """
        textures = [self._proxy_tex_3d, self._proxy_tex_2d]
        for cache in (self._block_cache_3d, self._block_cache_2d):
            if cache is not None:
                textures.append(cache.cache_tex)
        if self._lut_manager_3d is not None:
            textures += [
                self._lut_manager_3d.lut_tex,
                self._lut_manager_3d.brick_max_tex,
            ]
        if self._lut_manager_2d is not None:
            textures.append(self._lut_manager_2d.lut_tex)
        destroy_textures(*textures)
        self._residency_3d = self._residency_2d = None
        self._data_ready_listener = self._data_ready_listener_2d = None
        self._block_cache_3d = self._lut_manager_3d = None
        self._block_cache_2d = self._lut_manager_2d = None
        self._proxy_tex_3d = self._proxy_tex_2d = None

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


# ---------------------------------------------------------------------------
# GFXMultiscaleImageVisual
# ---------------------------------------------------------------------------


def _slot_geometries(
    transform,
    level_shapes: list[tuple[int, ...]],
    level_transforms: list,
    render_modes: set[str],
    displayed_axes: tuple[int, ...],
    block_size: int,
) -> tuple[MultiscaleBrickLayout3D | None, ImageGeometry3D | None]:
    """Build a slot's geometry for the mode *displayed_axes* selects.

    The other mode's geometry is built lazily on first entry into it.
    ``select_axes`` projects each level shape onto the data axes the
    displayed world axes map to; the level transforms are handed over whole.
    """
    volume_geometry = None
    image_geometry_2d = None
    fetch = _fetch_order(_world_axes_to_data_axes(transform, displayed_axes))
    if len(displayed_axes) == 3 and "3d" in render_modes:
        volume_geometry = MultiscaleBrickLayout3D(
            level_shapes=[select_axes(s, fetch) for s in level_shapes],
            level_transforms=list(level_transforms),
            block_size=block_size,
            fetch_axes=fetch,
        )
    elif len(displayed_axes) == 2 and "2d" in render_modes:
        image_geometry_2d = ImageGeometry3D(
            level_shapes=[select_axes(s, fetch) for s in level_shapes],
            block_size=block_size,
            n_levels=len(level_shapes),
            level_transforms=list(level_transforms),
            fetch_axes=fetch,
        )
    return volume_geometry, image_geometry_2d


class GFXMultiscaleImageVisual:
    """Render-layer visual for one ``MultiscaleImageVisual``.

    Draws the image single-channel or composited from a pool of
    :class:`_MultiscaleImageSlot` (unified image design 3.8).  The pool holds
    one slot when the visual has no ``channel_axis``, so a plain image keeps
    the whole GPU budget, and ``max_channels`` slots when it does, each with
    ``budget / pool size``.  Slots are keyed by channel index; single mode on
    index ``k`` reuses ``k``'s slot, and a full pool reassigns the least
    recently drawn slot the current mode does not need.

    Each plan runs LOD selection once, on the first drawn slot, and
    materializes the result per drawn channel.  Every channel's cache keys
    carry its own index on the channel axis, so a switch between the modes
    finds the bricks either mode already loaded.

    Parameters
    ----------
    visual_model : MultiscaleImageVisual
        The model, held for the life of the visual.
    level_shapes : list[tuple[int, ...]]
        Full nD shape per level, finest first.
    render_modes : set[str]
        Which nodes to build: ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    displayed_axes : tuple[int, ...]
        The world axes displayed at construction.
    """

    #: 3D loads go through the chunk scheduler (``plan`` / ``residencies``).
    chunked: bool = True
    cancellable: bool = True
    #: Applies the image slicing rule (design 3.2) itself, so the scene
    #: manager's data-coverage pre-check does not skip it.
    decides_empty_slices: bool = True

    def __init__(
        self,
        visual_model: MultiscaleImageVisual,
        level_shapes: list[tuple[int, ...]],
        render_modes: set[str],
        displayed_axes: tuple[int, ...],
    ) -> None:
        invalid = render_modes - {"2d", "3d"}
        if invalid or not render_modes:
            raise ValueError(
                f"render_modes must be a non-empty subset of {{'2d', '3d'}}, "
                f"got {render_modes!r}"
            )
        self.visual_model_id: UUID = visual_model.id
        self.render_modes = render_modes
        self._visual_model = visual_model
        self._channel_axis: int | None = visual_model.channel_axis
        self._full_level_shapes = [tuple(shape) for shape in level_shapes]
        self._level_transforms = list(visual_model.level_transforms)
        self._transform = visual_model.transform
        self._spaces: RenderSpaces | None = None
        self._last_displayed_axes: tuple[int, ...] = tuple(displayed_axes)
        self._pick_write: bool = visual_model.pick_write
        self._visible: bool = visual_model.appearance.visible
        self._slice_empty: bool = False

        config = visual_model.render_config
        n_slots = 1 if self._channel_axis is None else visual_model.max_channels
        # The budget is split between the channels the visual can *draw*, not
        # between the pool's slots: the pool is sized by ``max_channels`` so a
        # channel can be added later, but an unused slot's cache would only
        # take budget away from the ones drawing.  A visual with no channel
        # axis, or none configured, keeps the whole budget -- which is what a
        # single-channel image had before the pool existed.
        n_budget = max(1, len(visual_model.channels))
        budget_3d = max(1, config.gpu_budget_bytes // n_budget)
        budget_2d = max(1, config.gpu_budget_bytes_2d // n_budget)
        self._slots: list[_MultiscaleImageSlot] = []
        for index in range(n_slots):
            volume_geometry, image_geometry_2d = _slot_geometries(
                self._transform,
                self._full_level_shapes,
                self._level_transforms,
                render_modes,
                self._last_displayed_axes,
                config.block_size,
            )
            slot = _MultiscaleImageSlot(
                visual_model_id=visual_model.id,
                volume_geometry=volume_geometry,
                image_geometry_2d=image_geometry_2d,
                render_modes=render_modes,
                displayed_axes=self._last_displayed_axes,
                interpolation=visual_model.appearance.interpolation,
                gpu_budget_bytes_3d=budget_3d,
                gpu_budget_bytes_2d=budget_2d,
                transform=self._transform,
                full_level_transforms=self._level_transforms,
                full_level_shapes=self._full_level_shapes,
                # One wireframe per visual: the first slot draws it.
                aabb_enabled=visual_model.aabb.enabled if index == 0 else False,
                aabb_color=visual_model.aabb.color,
                aabb_line_width=visual_model.aabb.line_width,
                render_order=visual_model.appearance.render_order,
                pick_write=self._pick_write,
            )
            slot._block_size = config.block_size
            # Weakly: the slots are this visual's, so a strong bound method
            # would be a cycle keeping every slot's caches alive.
            slot._data_ready_listener = weak_callback(self._reveal_aabb_3d)
            slot._data_ready_listener_2d = weak_callback(self._reveal_aabb_2d)
            slot.key = None
            slot.last_drawn = 0
            slot.color_map_source = None
            slot.colormap = None
            self._slots.append(slot)

        self._slot_for_key: dict[int, int] = {}
        self._drawn: dict[int, int] = {}
        self._clock = 0

        self.node_3d: gfx.Group | None = gfx.Group() if "3d" in render_modes else None
        self.node_2d: gfx.Group | None = gfx.Group() if "2d" in render_modes else None
        for slot in self._slots:
            if self.node_3d is not None and slot.node_3d is not None:
                self.node_3d.add(slot.node_3d)
            if self.node_2d is not None and slot.node_2d is not None:
                self.node_2d.add(slot.node_2d)
        for group in (self.node_3d, self.node_2d):
            if group is not None:
                group.render_order = visual_model.appearance.render_order
                group.visible = self._visible

        keys = self._initial_keys()
        self._drawn = dict(zip(keys, self._assign_slots(keys), strict=True))
        self._apply_materials()
        self._apply_slot_visibility()

    @classmethod
    def from_cellier_model(
        cls,
        model: MultiscaleImageVisual,
        level_shapes: list[tuple[int, ...]],
        render_modes: set[str],
        displayed_axes: tuple[int, ...],
    ) -> GFXMultiscaleImageVisual:
        """Build the render visual for *model*.

        Parameters
        ----------
        model : MultiscaleImageVisual
            Source visual model.
        level_shapes : list[tuple[int, ...]]
            Full nD shape per level, finest first.
        render_modes : set[str]
            Which nodes to build.
        displayed_axes : tuple[int, ...]
            The axes currently displayed.

        Returns
        -------
        GFXMultiscaleImageVisual
        """
        return cls(model, level_shapes, render_modes, displayed_axes)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def slots(self) -> tuple[_MultiscaleImageSlot, ...]:
        """The slot pool."""
        return tuple(self._slots)

    @property
    def n_levels(self) -> int:
        """Number of LOD levels."""
        return len(self._full_level_shapes)

    # The first slot is the one a plain image draws with; these name its
    # resources for callers that inspect a single-channel visual.

    @property
    def material_3d(self) -> MultiscaleVolumeBrickMaterial | None:
        """The first slot's 3D material."""
        return self._slots[0].material_3d

    @property
    def material_2d(self) -> ImageBlockMaterial | None:
        """The first slot's 2D material."""
        return self._slots[0].material_2d

    @property
    def _inner_node_3d(self):
        return self._slots[0]._inner_node_3d

    @property
    def _inner_node_2d(self):
        return self._slots[0]._inner_node_2d

    @property
    def _block_cache_3d(self):
        return self._slots[0]._block_cache_3d

    @property
    def _block_cache_2d(self):
        return self._slots[0]._block_cache_2d

    @property
    def _volume_geometry(self):
        return self._slots[0]._volume_geometry

    @property
    def _image_geometry_2d(self):
        return self._slots[0]._image_geometry_2d

    @property
    def _last_plan_stats(self) -> dict:
        return self._slots[0]._last_plan_stats

    # ── Slots ───────────────────────────────────────────────────────────

    def _channel_size(self) -> int | None:
        if self._channel_axis is None:
            return None
        return int(self._full_level_shapes[0][self._channel_axis])

    def _composite(self) -> bool:
        return bool(self._visual_model.composite) and self._channel_axis is not None

    def _initial_keys(self) -> list[int]:
        if self._composite():
            return list(self._visual_model.drawn_channels(self._channel_size()))
        return [0]

    def _assign_slots(self, keys: list[int]) -> list[int]:
        """Give every key a slot, reusing a key's own slot first (design 3.8)."""
        self._clock += 1
        wanted = set(keys)
        taken: set[int] = set()
        assigned: list[int] = []
        for key in keys:
            index = self._slot_for_key.get(key)
            if index is None or index in taken:
                free = [
                    i
                    for i, slot in enumerate(self._slots)
                    if slot.key is None and i not in taken
                ]
                if free:
                    index = free[0]
                else:
                    index = min(
                        (
                            i
                            for i, slot in enumerate(self._slots)
                            if i not in taken and slot.key not in wanted
                        ),
                        key=lambda i: self._slots[i].last_drawn,
                    )
                    self._slot_for_key.pop(self._slots[index].key, None)
                self._slots[index].key = key
                self._slot_for_key[key] = index
            taken.add(index)
            self._slots[index].last_drawn = self._clock
            assigned.append(index)
        return assigned

    def _mode_appearance(self, key: int):
        if self._composite():
            return self._visual_model.channels.get(key)
        return self._visual_model.single

    def _colormap_for(self, slot: _MultiscaleImageSlot, color_map) -> gfx.TextureMap:
        if color_map is not slot.color_map_source or slot.colormap is None:
            slot.colormap = _make_colormap(color_map)
            slot.color_map_source = color_map
        return slot.colormap

    def _apply_materials(self) -> None:
        """Re-apply every drawn slot's appearance from the model."""
        model = self._visual_model
        if model is None:
            return
        shared = model.appearance
        alpha_mode = effective_transparency_mode(model)
        overlap = len(self._drawn) > 1
        for key, index in self._drawn.items():
            mode_appearance = self._mode_appearance(key)
            if mode_appearance is None:
                continue
            slot = self._slots[index]
            colormap = self._colormap_for(slot, mode_appearance.color_map)
            for node in (slot.node_3d, slot.node_2d):
                if node is not None:
                    node.render_order = shared.render_order
            material = slot.material_3d
            if material is not None:
                material.map = colormap
                material.clim = mode_appearance.clim
                material.threshold = float(mode_appearance.iso_threshold)
                material.attenuation = float(shared.attenuation)
                material.ray_steps_per_voxel = float(shared.ray_steps_per_voxel)
                material.render_mode = mode_appearance.render_mode
                material.opacity = mode_appearance.opacity
                material.interpolation = shared.interpolation
                material.alpha_mode = alpha_mode
                material.pick_write = self._pick_write
                material.depth_test = shared.depth_test
                # Overlapping channel volumes: the first to draw would write
                # its hit depth and clip the rest into speckle.
                material.depth_write = shared.depth_write and not overlap
                material.depth_compare = shared.depth_compare
            material = slot.material_2d
            if material is not None:
                material.map = colormap
                material.clim = mode_appearance.clim
                material.opacity = mode_appearance.opacity
                material.interpolation = shared.interpolation
                material.alpha_mode = alpha_mode
                material.pick_write = self._pick_write
                material.depth_compare = shared.depth_compare
                # Every channel plane sits at the same depth: with more than
                # one drawn, the first would hide the rest.
                material.depth_test = shared.depth_test and not overlap
                material.depth_write = shared.depth_write and not overlap

    def _apply_slot_visibility(self) -> None:
        drawn = set(self._drawn.values())
        for index, slot in enumerate(self._slots):
            visible = index in drawn and not self._slice_empty
            for inner in (slot._inner_node_3d, slot._inner_node_2d):
                if inner is not None:
                    inner.visible = visible
            for node in (slot.node_3d, slot.node_2d):
                if node is not None:
                    node.visible = True

    def _prune_undrawn(self) -> None:
        if not self._composite():
            return
        drawn = set(self._visual_model.drawn_channels(self._channel_size()))
        self._drawn = {k: i for k, i in self._drawn.items() if k in drawn}

    def _slice_coord_for(self, base: tuple, key: int) -> tuple:
        """*base* with this channel's own index on the channel axis."""
        if self._channel_axis is None:
            return base
        return tuple(
            (axis, key if axis == self._channel_axis else value) for axis, value in base
        )

    def _plan_keys(self, selection) -> list[int]:
        """Apply the slicing rule and name the channels to draw (design 3.3)."""
        composite = self._composite()
        exempt = (self._channel_axis,) if composite else ()
        for slot in self._slots:
            slot._unsliced_data_axes = exempt
        probe = self._slots[0]
        probe._begin_region_planning(selection)
        self._slice_empty = probe._slice_empty
        if self._slice_empty:
            return []
        if composite:
            return list(self._visual_model.drawn_channels(self._channel_size()))
        if self._channel_axis is None:
            return [0]
        selections = probe._level0_axis_selections()
        value = 0 if selections is None else selections[self._channel_axis]
        return [int(value) if not isinstance(value, tuple) else int(value[0])]

    def _begin_plan(self, selection) -> tuple[list[int], list[_MultiscaleImageSlot]]:
        keys = self._plan_keys(selection)
        indices = self._assign_slots(keys)
        self._drawn = dict(zip(keys, indices, strict=True))
        self._apply_materials()
        self._apply_slot_visibility()
        slots = [self._slots[i] for i in indices]
        for slot in slots:
            slot._begin_region_planning(selection)
        # _begin_region_planning shows or hides a slot's data node from its
        # own slicing verdict; the pool's visibility is the wrapper's call.
        self._apply_slot_visibility()
        return keys, slots

    # ── Node selection ─────────────────────────────────────────────────

    def _rebuild_slot_geometries(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = tuple(displayed_axes)
        group = self.node_3d if len(displayed_axes) == 3 else self.node_2d
        for slot in self._slots:
            old_node, new_node = slot.rebuild_geometry(
                self._full_level_shapes, displayed_axes
            )
            if group is None or old_node is new_node:
                continue
            if old_node is not None and old_node.parent is group:
                group.remove(old_node)
            if new_node is not None:
                group.add(new_node)
        self._apply_materials()
        self._apply_slot_visibility()

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Rebuild the slots' geometry if needed and return the group."""
        mode_attr = "node_3d" if len(displayed_axes) == 3 else "node_2d"
        if tuple(displayed_axes) != self._last_displayed_axes or any(
            getattr(slot, mode_attr) is None for slot in self._slots
        ):
            self._rebuild_slot_geometries(displayed_axes)
        return self.node_3d if len(displayed_axes) == 3 else self.node_2d

    def has_node(self, mode: str) -> bool:
        """Return True if the group for *mode* exists."""
        return (self.node_3d if mode == "3d" else self.node_2d) is not None

    def get_node(self, mode: str) -> gfx.WorldObject | None:
        """Return the group for *mode*."""
        return self.node_3d if mode == "3d" else self.node_2d

    def build_node(
        self,
        mode: str,
        visual_model,
        displayed_axes: tuple[int, ...],
        level_shapes: list[tuple[int, ...]],
        level_transforms: list,
    ) -> gfx.WorldObject | None:
        """Build the slots' nodes for *mode* and return the group."""
        self._full_level_shapes = [tuple(s) for s in level_shapes]
        return self.get_node_for_dims(displayed_axes)

    def rebuild_node_geometry(
        self,
        mode: str,
        displayed_axes: tuple[int, ...],
        level_shapes: list[tuple[int, ...]],
        level_transforms: list,
    ) -> gfx.WorldObject | None:
        """Rebuild the slots' geometry after a dims change."""
        self._full_level_shapes = [tuple(s) for s in level_shapes]
        return self.get_node_for_dims(displayed_axes)

    # ── Planning ───────────────────────────────────────────────────────

    def plan(
        self,
        request: ReslicingRequest,
        config: VisualRenderConfig,
        mode: PlanMode = PlanMode.FULL,
    ) -> list[DesiredSet]:
        """Plan once, and return a desired set per drawn channel.

        The chunk scheduler's planner contract (design 5.3): LOD selection,
        distance sort, frustum or viewport cull and truncation run once, on
        the first drawn slot; each drawn channel's atlas gets the same bricks
        (3D) or tiles (2D) on its own slice.  Nothing on the GPU is touched.

        Parameters
        ----------
        request : ReslicingRequest
            A 3D request (camera, frustum) or a 2D one (camera, world
            extent), with dims and region.
        config : VisualRenderConfig
            LOD bias, forced level and frustum (viewport) culling.
        mode : PlanMode
            Accepted for the contract; this phase plans the target only.

        Returns
        -------
        list[DesiredSet]
            One per drawn channel (``store`` unset).  Empty when the slice
            misses the data, so the caller retires the atlases.
        """
        dims_state = request.dims_state
        displayed = tuple(dims_state.selection.displayed_axes)
        if displayed != self._last_displayed_axes:
            self._rebuild_slot_geometries(displayed)
        keys, slots = self._begin_plan(request.selection)
        if not keys:
            return []
        planner = slots[0]
        is_2d = len(displayed) == 2
        if is_2d:
            if planner._image_geometry_2d is None or planner._block_cache_2d is None:
                return []
        elif planner._volume_geometry is None or planner._block_cache_3d is None:
            return []
        for slot in slots:
            if self._last_displayed_axes != slot._last_displayed_axes:
                slot._update_node_matrix(self._last_displayed_axes)
        loading = config.loading
        plan_target = mode == PlanMode.FULL
        arr = None
        if is_2d:
            view_min, view_max = viewport_2d(request)
            if plan_target:
                arr = planner._plan_tiles_2d(
                    camera_pos_world=request.camera_pos,
                    viewport_width_px=request.screen_size_px[0],
                    world_width=request.world_extent[0],
                    view_min_world=view_min if config.frustum_cull else None,
                    view_max_world=view_max if config.frustum_cull else None,
                    lod_bias=config.lod_bias,
                    force_level=config.force_level,
                    use_culling=config.frustum_cull,
                )
            else:
                planner._adopt_viewport_2d(
                    request.camera_pos,
                    request.world_extent[0],
                    view_min if config.frustum_cull else None,
                    view_max if config.frustum_cull else None,
                )
            backstop = planner._plan_backstop_2d(
                request.camera_pos,
                request.world_extent[0],
                view_min,
                view_max,
                loading,
            )
        else:
            if plan_target:
                arr = planner._plan_bricks(
                    request.camera_pos,
                    request.frustum_corners if config.frustum_cull else None,
                    request.fov_y_rad,
                    request.screen_size_px[1],
                    config.lod_bias,
                    config.force_level,
                )
            backstop = planner._plan_backstop_3d(
                request.camera_pos, request.frustum_corners, loading
            )
        base_coord = planner._block_key_slice_coord()
        composite = self._composite()
        n_target = 0 if arr is None else len(arr)
        desired: list[DesiredSet] = []
        for key, slot in zip(keys, slots, strict=True):
            fill = {self._channel_axis: key} if composite else None
            if is_2d:
                slot._current_viewport_cells = planner._current_viewport_cells
                wanted = slot.desired_set_2d(
                    arr, fill=fill, backstop_arr=backstop, loading=loading
                )
            else:
                slot._current_slice_coord_3d = self._slice_coord_for(base_coord, key)
                wanted = slot.desired_set_3d(
                    arr, fill=fill, backstop_arr=backstop, loading=loading
                )
            if wanted is not None:
                slot._last_plan_stats = _plan_stats(n_target, wanted)
                desired.append(wanted)
        log_backstop_cap_once(self, desired)
        return desired

    def residencies(self) -> dict[int, ImageResidency3D | ImageResidency2D]:
        """``cache_id -> adapter`` for every slot's 3D and 2D atlas."""
        out: dict[int, ImageResidency3D | ImageResidency2D] = {}
        for slot in self._slots:
            for residency in (slot.residency_3d(), slot.residency_2d()):
                if residency is not None:
                    out[residency.cache_id] = residency
        return out

    def _reveal_aabb_3d(self) -> None:
        """A slot's first brick landed: show the visual's one wireframe."""
        owner = self._slots[0] if self._slots else None
        if owner is not None and owner._aabb_line_3d is not None:
            owner._data_ready_3d = True
            owner._aabb_line_3d.visible = owner._aabb_enabled

    def _reveal_aabb_2d(self) -> None:
        """A slot's first tile landed: show the visual's one wireframe."""
        owner = self._slots[0] if self._slots else None
        if owner is not None and owner._aabb_line_2d is not None:
            owner._data_ready_2d = True
            owner._aabb_line_2d.visible = owner._aabb_enabled

    def close(self) -> None:
        """Release the slots, their caches and nodes.  Unusable afterwards.

        The chunk scheduler forgets the 3D atlases first (the render manager
        removes them), so nothing lands on a closed slot.
        """
        for slot in self._slots:
            slot.close()
        for group in (self.node_3d, self.node_2d):
            if group is not None:
                group.clear()
        self._slots = []
        self._slot_for_key = {}
        self._drawn = {}
        self._visual_model = None

    # ── Event handlers ─────────────────────────────────────────────────

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Hand the new transform to every slot."""
        self._transform = event.transform
        for slot in self._slots:
            slot.on_transform_changed(event)

    def set_render_spaces(self, spaces: RenderSpaces | None) -> None:
        """Hand the coordinate systems to every slot."""
        self._spaces = spaces
        for slot in self._slots:
            slot.set_render_spaces(spaces)

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """A shared field changed: restyle every drawn slot."""
        if event.field_name == "render_order":
            for group in (self.node_3d, self.node_2d):
                if group is not None:
                    group.render_order = event.new_value
        self._apply_materials()

    def on_single_appearance_changed(self, event: SingleAppearanceChangedEvent) -> None:
        """A single-mode field changed: restyle when single mode is drawn."""
        if not self._composite():
            self._apply_materials()

    def on_channel_appearance_changed(
        self, event: ChannelAppearanceChangedEvent
    ) -> None:
        """A channel field changed: restyle, and hide a channel switched off."""
        if not self._composite():
            return
        if event.field_name == "visible" and not event.new_value:
            self._prune_undrawn()
            self._apply_slot_visibility()
        self._apply_materials()

    def on_image_composite_changed(self, event: ImageCompositeChangedEvent) -> None:
        """The mode switched: restyle now; the controller's reslice follows."""
        self._prune_undrawn()
        self._apply_materials()
        self._apply_slot_visibility()

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Toggle the whole visual."""
        self._visible = event.visible
        for group in (self.node_3d, self.node_2d):
            if group is not None:
                group.visible = event.visible

    def _slot_owning(self, hit_object) -> _MultiscaleImageSlot | None:
        for slot in self._slots:
            if hit_object in (slot._inner_node_3d, slot._inner_node_2d):
                return slot
        return None

    def pick_channel_index(self, hit_object) -> int | None:
        """The channel index of the slot whose node *hit_object* is, if any."""
        slot = self._slot_owning(hit_object)
        return None if slot is None else slot.key

    def drawn_channel_indices(self) -> tuple[int, ...]:
        """The channel indices the last plan draws, ascending."""
        return tuple(sorted(self._drawn))

    def pick_collapsed_indices(self) -> dict[int, int] | None:
        """The level-0 planes the drawn slots last planned.

        In composite mode the channel axis is left out: each drawn channel
        sits at its own index.

        Returns
        -------
        dict[int, int] or None
            Data axis to level-0 voxel index.
        """
        if not self._drawn:
            return None
        slot = self._slots[next(iter(self._drawn.values()))]
        collapsed = slot.pick_collapsed_indices()
        if collapsed is None:
            return None
        skip = self._channel_axis if self._composite() else None
        return {axis: value for axis, value in collapsed.items() if axis != skip}

    def pick_data_coordinate(
        self, hit_object, pick_info: dict
    ) -> tuple[float, ...] | None:
        """Level-0 data coordinate of a pick on one of the slots' nodes."""
        slot = self._slot_owning(hit_object) or (
            self._slots[0] if self._slots else None
        )
        if slot is None:
            return None
        return slot.pick_data_coordinate(hit_object, pick_info)

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        """Update pick_write on every slot."""
        self._pick_write = event.pick_write
        for slot in self._slots:
            slot._pick_write = event.pick_write
            slot.on_pick_write_changed(event)

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Apply an AABB change to the visual's one wireframe (the first slot's)."""
        if self._slots:
            self._slots[0].on_aabb_changed(event)

    def on_data_store_contents_changed(
        self, event: DataStoreContentsChangedEvent
    ) -> None:
        """Stub -- brick eviction deferred to a future phase."""

    def on_data_store_metadata_changed(
        self, event: DataStoreMetadataChangedEvent
    ) -> None:
        """Stub -- geometry rebuild deferred to a future phase."""

    def tick(self) -> None:
        """Advance every slot's jitter seed."""
        for slot in self._slots:
            slot.tick()

    def reset_tick(self) -> None:
        """Rewind every slot's jitter seed."""
        for slot in self._slots:
            slot.reset_tick()
