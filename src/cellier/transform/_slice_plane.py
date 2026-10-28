"""The geometry of one 2D slice through raster data (oblique 2D design v3).

Section 4.1 of ``plans/oblique_2d_raster_design_v3.md``.  For one visual in
one 2D view, ``L`` and ``t`` are the effective ``data -> world`` linear
block and translation, ``D`` the two displayed world axes, ``S`` the sliced
ones and ``s`` the slice position:

* the **frame** (:func:`slice_plane_frame`) parameterises the pullback
  ``P = {p : L[S, :] p + t[S] = s[S]}`` by the displayed world coordinates:
  ``p = p0 + c0 u0 + c1 u1`` lands on world ``c`` in the displayed axes;
* the **projection geometry** (:func:`classify_projection`) describes a
  slab over one sliced world axis ``a``: the data axes it varies over and
  the data direction of a pixel's ray, which runs along world ``a``;
* the **ray step** (:func:`ray_step`) is one drawn-level voxel along that
  ray, the Euclidean voxel length (never the max-norm, K6);
* the **footprint** (:func:`slice_footprint`) is the data box cut by the
  slice or the slab, as a convex polygon in the displayed world axes (D15);
* :func:`projection_axis` and :func:`choose_slice_route` are the rules of
  4.2 and 5.2.2, as pure functions.

Everything is float64 and reads ``classification.effective_transform``, so
it agrees with :func:`~cellier.transform.classify_view` about what is zero
(D7).  A non-affine (non-uniform) axis may be sliced: it is pinned to the
data value its slice position maps back to.  It may not be displayed or
projected along, because nothing here is affine in it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from cellier.transform._alignment import (
    COUPLING_TOLERANCE,
    SliceAlignment,
    ViewClassification,
    classify_view,
    coupling_pattern,
)
from cellier.transform._geometry_ops import NonAffineTransformError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.transform._base import BaseTransform
    from cellier.transform._selection import RegionSelection

__all__ = [
    "BRICK_ROUTE_AXES",
    "DepthAlignment",
    "ProjectionGeometry",
    "ProjectionRefusal",
    "ProjectionRefusalReason",
    "RouteDecision",
    "RouteRefusalReason",
    "SlicePlaneFrame",
    "SliceRoute",
    "choose_slice_route",
    "classify_projection",
    "projection_axis",
    "ray_step",
    "slice_footprint",
    "slice_plane_frame",
]

#: The relative residual above which a solved frame is reported as wrong
#: rather than returned: a pullback that is not what the classifier said.
_SOLVE_TOLERANCE = 1e-9


# -- the affine part ------------------------------------------------------------


@dataclass(frozen=True)
class _AffineParts:
    """``L`` and ``t`` of a transform, with its non-affine leaves set apart.

    ``linear`` and ``translation`` hold the affine blocks only (snapped,
    D7); a non-affine block's rows and columns are zero there and the block
    is listed in ``leaves`` as ``(data axis, world axis, transform)``.
    """

    linear: np.ndarray
    translation: np.ndarray
    leaves: tuple[tuple[int, int, BaseTransform], ...]


def _affine_parts(transform: BaseTransform) -> _AffineParts:
    """Split *transform* into its affine blocks and its 1-D non-affine leaves.

    Raises
    ------
    NonAffineTransformError
        If a non-affine block is not 1-D in and out.
    """
    from cellier.transform._by_dimension import ByDimensionTransform

    affine = transform.to_affine()
    if affine is not None:
        linear = np.asarray(affine.linear, dtype=np.float64)
        return _AffineParts(
            linear=np.where(coupling_pattern(linear), linear, 0.0),
            translation=np.asarray(affine.translation, dtype=np.float64),
            leaves=(),
        )
    n_world, n_data = transform.output_ndim, transform.input_ndim
    linear = np.zeros((n_world, n_data))
    translation = np.zeros(n_world)
    leaves: list[tuple[int, int, BaseTransform]] = []
    if isinstance(transform, ByDimensionTransform):
        blocks = [
            (block.transform, block.input_axes, block.output_axes)
            for block in transform.blocks
        ]
    else:  # a 1-D non-affine leaf on its own
        blocks = [(transform, (0,), (0,))]
    for inner, inputs, outputs in blocks:
        part = inner.to_affine()
        if part is None:
            if len(inputs) != 1 or len(outputs) != 1:
                raise NonAffineTransformError(
                    f"A non-affine block over data axes {list(inputs)} is not "
                    f"1-D, so a slice through it has no affine description."
                )
            leaves.append((int(inputs[0]), int(outputs[0]), inner))
            continue
        block_linear = np.asarray(part.linear, dtype=np.float64)
        rows = list(outputs)
        for column, axis in enumerate(inputs):
            linear[rows, axis] = np.where(
                coupling_pattern(block_linear)[:, column],
                block_linear[:, column],
                0.0,
            )
        translation[rows] = part.translation
    return _AffineParts(linear=linear, translation=translation, leaves=tuple(leaves))


def _pin_leaves(
    parts: _AffineParts, free_world_axes: Sequence[int], world_point: np.ndarray
) -> dict[int, float]:
    """The data value of every non-affine leaf, all of which must be pinned.

    Raises
    ------
    NonAffineTransformError
        If a leaf's world axis is free (displayed, or projected along).
    """
    pinned: dict[int, float] = {}
    free = set(free_world_axes)
    for data_axis, world_axis, leaf in parts.leaves:
        if world_axis in free:
            raise NonAffineTransformError(
                f"World axis {world_axis} maps through a non-affine transform; "
                f"it can be sliced but not displayed or projected along."
            )
        value = leaf.imap_coordinates(np.array([[world_point[world_axis]]]))
        pinned[data_axis] = float(np.asarray(value).reshape(-1)[0])
    return pinned


@dataclass(frozen=True)
class _Solve:
    """``p = offset + basis @ z`` for world coordinates ``z`` on free axes."""

    offset: np.ndarray
    basis: np.ndarray


def _solve_free(
    parts: _AffineParts,
    pinned_data: Mapping[int, float],
    free_world_axes: Sequence[int],
    world_point: np.ndarray,
) -> _Solve:
    """Data points as an affine function of the free world coordinates.

    Every world axis not in *free_world_axes* is held at *world_point*; the
    free ones are the parameters.  The rows the data reaches (a broadcast
    axis's row is zero) and the data axes that are not pinned leaves must
    form a square, invertible system, which is what a pullback of the
    expected dimension means.

    Raises
    ------
    ValueError
        If a free world axis is one the data does not reach, or the system
        is not square and invertible.
    """
    n_world, n_data = parts.linear.shape
    columns = [axis for axis in range(n_data) if axis not in pinned_data]
    block = parts.linear[:, columns]
    rows = [row for row in range(n_world) if np.any(block[row] != 0.0)]
    free = [int(axis) for axis in free_world_axes]
    unreached = [axis for axis in free if axis not in rows]
    if unreached:
        raise ValueError(
            f"The data does not vary along world axes {unreached}, so a slice "
            f"displaying or projecting along them has no extent there."
        )
    square = block[rows, :]
    if square.shape[0] != square.shape[1] or np.linalg.matrix_rank(square) < len(
        columns
    ):
        raise ValueError(
            "The slice does not pull back to a set the free world axes "
            "parameterise: the data axes and the world axes the data reaches "
            "do not correspond one to one."
        )
    inverse = np.linalg.inv(square)
    held = np.array(world_point, dtype=np.float64)
    held[free] = 0.0
    constant = held - parts.translation
    for data_axis, value in pinned_data.items():
        constant -= parts.linear[:, data_axis] * value
    offset = np.zeros(n_data)
    offset[columns] = inverse @ constant[rows]
    for data_axis, value in pinned_data.items():
        offset[data_axis] = value
    basis = np.zeros((n_data, len(free)))
    for index, axis in enumerate(free):
        basis[columns, index] = inverse[:, rows.index(axis)]
    return _Solve(offset=offset, basis=basis)


def _check_slice_position(position: np.ndarray, n_world: int) -> np.ndarray:
    values = np.asarray(position, dtype=np.float64).reshape(-1)
    if values.shape[0] != n_world:
        raise ValueError(
            f"The slice position has {values.shape[0]} axes but the world has "
            f"{n_world}."
        )
    return values


def _displayed(classification: ViewClassification) -> tuple[int, ...]:
    n_world = classification.effective_transform.output_ndim
    sliced = set(classification.sliced_world_axes)
    return tuple(axis for axis in range(n_world) if axis not in sliced)


# -- the frame ------------------------------------------------------------------


@dataclass(frozen=True)
class SlicePlaneFrame:
    """The pullback of a 2D slice, parameterised by the displayed axes.

    ``origin + c[0] * axes[0] + c[1] * axes[1]`` is the data point the
    slice puts at displayed world coordinates ``c`` (in
    ``displayed_world_axes`` order); every such point satisfies the slice.

    Attributes
    ----------
    origin : np.ndarray
        ``p0``, ``(n_data,)``: the data point at displayed world zero.
    axes : np.ndarray
        ``(2, n_data)``: ``u0`` and ``u1``, the data step per world unit
        along each displayed axis.
    displayed_world_axes : tuple[int, int]
        ``D``, ascending.
    varying_data_axes : tuple[int, ...]
        ``V``: the data axes on which ``axes`` is non-zero.
    """

    origin: np.ndarray
    axes: np.ndarray
    displayed_world_axes: tuple[int, int]
    varying_data_axes: tuple[int, ...]

    def data_points(self, displayed: np.ndarray) -> np.ndarray:
        """Map displayed world coordinates ``(..., 2)`` to data points."""
        values = np.asarray(displayed, dtype=np.float64)
        return self.origin + values @ self.axes


def slice_plane_frame(
    classification: ViewClassification, slice_position: np.ndarray
) -> SlicePlaneFrame:
    """The frame of the slice *classification* describes, at *slice_position*.

    Parameters
    ----------
    classification : ViewClassification
        The visual's classification against a 2D view; its effective
        transform is the one solved.
    slice_position : np.ndarray
        A world point, ``(n_world,)``; only the sliced axes are read
        (``RegionSelection.slice_position``).

    Returns
    -------
    SlicePlaneFrame
        ``p0``, ``u0``, ``u1`` in float64.

    Raises
    ------
    ValueError
        If the view is not 2D, or the pullback is not 2D (a tilted 2D store,
        O18), or the displayed axes do not parameterise it.
    NonAffineTransformError
        If a displayed axis maps through a non-affine transform.
    """
    displayed = _displayed(classification)
    if len(displayed) != 2:
        raise ValueError(f"A slice frame needs a 2D view; this one shows {displayed}.")
    if classification.pullback_dim != 2:
        raise ValueError(
            f"The slice pulls back to a set of dimension "
            f"{classification.pullback_dim}; a frame needs a plane."
        )
    transform = classification.effective_transform
    position = _check_slice_position(slice_position, transform.output_ndim)
    parts = _affine_parts(transform)
    pinned = _pin_leaves(parts, displayed, position)
    solved = _solve_free(parts, pinned, displayed, position)
    _check_solution(parts, solved, displayed, position)
    return SlicePlaneFrame(
        origin=solved.offset,
        axes=solved.basis.T.copy(),
        displayed_world_axes=(displayed[0], displayed[1]),
        varying_data_axes=classification.varying_data_axes,
    )


def _check_solution(
    parts: _AffineParts,
    solved: _Solve,
    free: Sequence[int],
    position: np.ndarray,
) -> None:
    """Raise when the solved points miss the held or the free world axes.

    The origin must land on the held coordinates and at zero on the free
    ones, and each basis vector must move only its own free axis, by one.
    """
    n_world = parts.linear.shape[0]
    free_axes = [int(axis) for axis in free]
    reached = [row for row in range(n_world) if np.any(parts.linear[row] != 0.0)]
    held = [row for row in reached if row not in set(free_axes)]
    world = parts.linear @ solved.offset + parts.translation
    expected = position.copy()
    expected[free_axes] = 0.0
    scale = 1.0 + float(np.max(np.abs(expected[reached]), initial=0.0))
    steps = parts.linear @ solved.basis
    identity = np.zeros_like(steps)
    identity[free_axes, range(len(free_axes))] = 1.0
    if (
        np.max(np.abs(world[reached] - expected[reached])) > _SOLVE_TOLERANCE * scale
        or np.max(np.abs(steps[held + free_axes] - identity[held + free_axes]))
        > _SOLVE_TOLERANCE
    ):
        raise ValueError("The solved slice frame does not satisfy the slice.")


# -- projection -------------------------------------------------------------------


class DepthAlignment(str, Enum):
    """How a projection's rays meet the data (4.1).

    ``ALONG_AXIS``: the ray runs along one data axis, so the projection is
    an exact voxel reduction.  ``TILTED``: it crosses several.
    ``CONSTANT``: the data does not vary along the projection axis (a
    broadcast world axis), so every sample on a ray reads the same value.
    """

    ALONG_AXIS = "along_axis"
    TILTED = "tilted"
    CONSTANT = "constant"


@dataclass(frozen=True)
class ProjectionGeometry:
    """A slab over one sliced world axis, for one visual and one 2D view.

    Attributes
    ----------
    projection_world_axis : int
        ``a``.
    depth_alignment : DepthAlignment
        The case.
    slab_pullback_dim : int
        The dimension of the slab's pullback: ``n_data`` minus the rank of
        the other sliced rows.  The brick route draws when it is 3.
    slab_varying_axes : tuple[int, ...]
        ``V_h``, ascending: the data axes the slab varies over.
    ray_direction_data : np.ndarray
        ``d``, ``(n_data,)``: the data step per world unit along ``a`` with
        every other world coordinate held (snapped, D7).
    """

    projection_world_axis: int
    depth_alignment: DepthAlignment
    slab_pullback_dim: int
    slab_varying_axes: tuple[int, ...]
    ray_direction_data: np.ndarray


def classify_projection(
    transform: BaseTransform,
    displayed_world_axes: Sequence[int],
    projection_world_axis: int,
) -> ProjectionGeometry:
    """The geometry of a projection along *projection_world_axis*.

    Parameters
    ----------
    transform : BaseTransform
        A visual's ``data -> world`` transform; D7's snapping applies.
    displayed_world_axes : Sequence[int]
        The view's displayed world axes.
    projection_world_axis : int
        ``a``, which must be sliced.

    Returns
    -------
    ProjectionGeometry
        Depth alignment, ``V_h`` and the ray's data direction.

    Raises
    ------
    ValueError
        If ``a`` is displayed or out of range.
    NonAffineTransformError
        If ``a`` maps through a non-affine transform.
    """
    projection_axis_index = int(projection_world_axis)
    classification = classify_view(transform, displayed_world_axes)
    effective = classification.effective_transform
    n_world, n_data = effective.output_ndim, effective.input_ndim
    if not 0 <= projection_axis_index < n_world:
        raise ValueError(
            f"World axis {projection_axis_index} is not an axis of a "
            f"{n_world}-axis world."
        )
    if projection_axis_index not in classification.sliced_world_axes:
        raise ValueError(
            f"World axis {projection_axis_index} is displayed; a projection "
            f"runs along a sliced axis."
        )
    slab = classify_view(
        effective, (*_displayed(classification), projection_axis_index)
    )
    parts = _affine_parts(effective)
    if any(world == projection_axis_index for _, world, _ in parts.leaves):
        raise NonAffineTransformError(
            f"World axis {projection_axis_index} maps through a non-affine "
            f"transform; a projection along it has no constant ray step."
        )
    leaf_axes = {data for data, _, _ in parts.leaves}
    columns = [axis for axis in range(n_data) if axis not in leaf_axes]
    block = parts.linear[:, columns]
    direction = np.zeros(n_data)
    if np.any(block[projection_axis_index] != 0.0):
        target = np.zeros(n_world)
        target[projection_axis_index] = 1.0
        solution, *_ = np.linalg.lstsq(block, target, rcond=None)
        direction[columns] = solution
        magnitude = np.abs(direction)
        direction = np.where(
            magnitude > COUPLING_TOLERANCE * magnitude.max(), direction, 0.0
        )
    nonzero = int(np.count_nonzero(direction))
    if nonzero == 0:
        alignment = DepthAlignment.CONSTANT
    elif nonzero == 1:
        alignment = DepthAlignment.ALONG_AXIS
    else:
        alignment = DepthAlignment.TILTED
    return ProjectionGeometry(
        projection_world_axis=projection_axis_index,
        depth_alignment=alignment,
        slab_pullback_dim=slab.pullback_dim,
        slab_varying_axes=slab.varying_data_axes,
        ray_direction_data=direction,
    )


def ray_step(
    ray_direction_data: np.ndarray,
    level_scale: Sequence[float] | None = None,
    steps_per_voxel: float = 1.0,
) -> float:
    """World units between samples one drawn-level voxel apart along a ray.

    ``1 / |diag(level_scale)^-1 d| / steps_per_voxel`` (4.1): the Euclidean
    length, in drawn-level voxels, of one world unit along the ray (K6).

    Parameters
    ----------
    ray_direction_data : np.ndarray
        ``d``, level-0 data units per world unit
        (:attr:`ProjectionGeometry.ray_direction_data`).
    level_scale : Sequence[float] or None
        The drawn level's voxel size in level-0 voxels, per data axis;
        ``None`` is level 0.
    steps_per_voxel : float
        Samples per drawn-level voxel (``projection_steps_per_voxel``).

    Returns
    -------
    float
        The step in world units; ``inf`` when the ray does not move through
        the data (``DepthAlignment.CONSTANT``).

    Raises
    ------
    ValueError
        If *steps_per_voxel* is not positive or *level_scale* has the wrong
        length or a non-positive entry.
    """
    if steps_per_voxel <= 0.0:
        raise ValueError(f"steps_per_voxel must be positive; got {steps_per_voxel}.")
    direction = np.asarray(ray_direction_data, dtype=np.float64).reshape(-1)
    scale = (
        np.ones_like(direction)
        if level_scale is None
        else np.asarray(level_scale, dtype=np.float64).reshape(-1)
    )
    if scale.shape != direction.shape or np.any(scale <= 0.0):
        raise ValueError(
            f"level_scale must hold one positive size per data axis "
            f"({direction.shape[0]}); got {scale.tolist()}."
        )
    voxels_per_world = float(np.linalg.norm(direction / scale))
    if voxels_per_world == 0.0:
        return float("inf")
    return 1.0 / voxels_per_world / steps_per_voxel


# -- which axis projects (4.2, G0-3) ------------------------------------------


class ProjectionRefusalReason(str, Enum):
    """Why a projection setting draws the plane instead (5.2.5)."""

    MULTIPLE_AXES = "multiple_axes"
    DISPLAYED_AXIS = "displayed_axis"
    UNKNOWN_AXIS = "unknown_axis"


@dataclass(frozen=True)
class ProjectionRefusal:
    """A projection setting that cannot be honoured in this view.

    Attributes
    ----------
    reason : ProjectionRefusalReason
        The case.
    axes : tuple[str, ...]
        The ``projection_axes`` names it concerns.
    """

    reason: ProjectionRefusalReason
    axes: tuple[str, ...]


def projection_axis(
    slab_projection: str,
    projection_axes: Sequence[str],
    world_axis_names: Sequence[str],
    displayed_world_axes: Sequence[int],
    thickness: Mapping[int, float],
) -> int | None | ProjectionRefusal:
    """The world axis a visual projects along in one view, if any (4.2).

    Parameters
    ----------
    slab_projection : str
        ``"none"``, ``"mip"``, ``"minip"`` or ``"mean"``.
    projection_axes : Sequence[str]
        The appearance's ``projection_axes``: world axis names.
    world_axis_names : Sequence[str]
        The world's axis names, by index.
    displayed_world_axes : Sequence[int]
        The view's displayed world axes.
    thickness : Mapping[int, float]
        The scene's half-thickness per world axis (``dims.selection``);
        an absent axis has thickness 0.

    Returns
    -------
    int or None or ProjectionRefusal
        The axis index when the visual projects; ``None`` when it draws the
        plane by today's rule with no status (no operator, no axis, or zero
        thickness); a refusal when it draws the plane with a status.
    """
    if slab_projection == "none" or not projection_axes:
        return None
    names = tuple(projection_axes)
    unknown = tuple(name for name in names if name not in world_axis_names)
    if unknown:
        return ProjectionRefusal(ProjectionRefusalReason.UNKNOWN_AXIS, unknown)
    if len(names) > 1:
        return ProjectionRefusal(ProjectionRefusalReason.MULTIPLE_AXES, names)
    axis = list(world_axis_names).index(names[0])
    if axis in {int(shown) for shown in displayed_world_axes}:
        return ProjectionRefusal(ProjectionRefusalReason.DISPLAYED_AXIS, names)
    if float(thickness.get(axis, 0.0)) == 0.0:
        return None
    return axis


# -- the route (5.2.2) ---------------------------------------------------------


class SliceRoute(str, Enum):
    """Which path draws a raster visual in a view (5.2.2)."""

    TODAY = "today"
    BRICK = "brick"
    NONE = "none"


class RouteRefusalReason(str, Enum):
    """Why a raster visual draws nothing in a view (5.2.5)."""

    PULLBACK_NOT_2D = "pullback_not_2d"
    TOO_MANY_VARYING_AXES = "too_many_varying_axes"
    NORMAL_NOT_ONE_AXIS = "normal_not_one_axis"


@dataclass(frozen=True)
class RouteDecision:
    """The route, and the reason when it is ``NONE``.

    Attributes
    ----------
    route : SliceRoute
        The path.
    reason : RouteRefusalReason or None
        Set exactly when ``route`` is ``NONE``.
    dimension : int or None
        With a refusal: the pullback's dimension, ``|V|``, or the number of
        sliced world axes the slice's normal reads, whichever refused.
    normal_world_axis : int or None
        With ``BRICK``: the world axis the slice node's depth runs along --
        the one sliced axis whose row reads the varying data axes (a plain
        slice), or the projection axis.
    """

    route: SliceRoute
    reason: RouteRefusalReason | None = None
    dimension: int | None = None
    normal_world_axis: int | None = None


#: The number of data axes the brick route can draw a slice through (O19).
BRICK_ROUTE_AXES = 3


def choose_slice_route(
    classification: ViewClassification,
    projection: ProjectionGeometry | None = None,
) -> RouteDecision:
    """The route for a raster visual in one view (5.2.2's table).

    Parameters
    ----------
    classification : ViewClassification
        The visual against the view.
    projection : ProjectionGeometry or None
        Set when the visual projects in this view (:func:`projection_axis`
        returned an axis); labels never pass one.

    Returns
    -------
    RouteDecision
        ``TODAY`` for 3D views and for aligned slices that do not project;
        ``BRICK`` for a tilted slice through three data axes or any
        projection over a three-axis slab; ``NONE`` with a reason otherwise.

    Notes
    -----
    A plain 2D slice whose pullback is a line or a point (``pullback_dim <
    2``: a tilted 2D store, or a flat one seen edge-on) is refused in every
    alignment (O18, U13). It has no area to draw, and today's path raises
    on it when the render spaces are pushed.
    """
    if len(_displayed(classification)) != 2:
        return RouteDecision(SliceRoute.TODAY)
    if projection is not None:
        decision = _brick_or_refusal(
            projection.slab_pullback_dim, projection.slab_varying_axes, 3
        )
        if decision.route is not SliceRoute.BRICK:
            return decision
        return RouteDecision(
            SliceRoute.BRICK, normal_world_axis=projection.projection_world_axis
        )
    if classification.pullback_dim < 2:
        return RouteDecision(
            SliceRoute.NONE,
            RouteRefusalReason.PULLBACK_NOT_2D,
            classification.pullback_dim,
        )
    if classification.alignment is not SliceAlignment.OBLIQUE:
        return RouteDecision(SliceRoute.TODAY)
    decision = _brick_or_refusal(
        classification.pullback_dim, classification.varying_data_axes, 2
    )
    if decision.route is not SliceRoute.BRICK:
        return decision
    coupling = classification.effective_transform.axis_coupling()
    varying = list(classification.varying_data_axes)
    normals = [
        axis
        for axis in classification.sliced_world_axes
        if coupling[axis, varying].any()
    ]
    if len(normals) != 1:
        return RouteDecision(
            SliceRoute.NONE, RouteRefusalReason.NORMAL_NOT_ONE_AXIS, len(normals)
        )
    return RouteDecision(SliceRoute.BRICK, normal_world_axis=normals[0])


def _brick_or_refusal(
    pullback_dim: int, varying: tuple[int, ...], expected_dim: int
) -> RouteDecision:
    if pullback_dim != expected_dim:
        return RouteDecision(
            SliceRoute.NONE, RouteRefusalReason.PULLBACK_NOT_2D, pullback_dim
        )
    if len(varying) != BRICK_ROUTE_AXES:
        return RouteDecision(
            SliceRoute.NONE, RouteRefusalReason.TOO_MANY_VARYING_AXES, len(varying)
        )
    return RouteDecision(SliceRoute.BRICK)


# -- the footprint (D15) -------------------------------------------------------


def slice_footprint(
    data_shape: Sequence[int],
    classification: ViewClassification,
    selection: RegionSelection,
    projection_world_axis: int | None = None,
) -> np.ndarray:
    """The data box cut by a slice or a slab, in the displayed world axes.

    The data box is ``[-0.5, n - 0.5]`` per data axis (voxel edges, voxel
    centres at integers).  Without a projection, every sliced world axis is
    held at the selection's slice position and only the selection's
    constraints on displayed axes cut the polygon; with one, the
    projection axis is free as well and the selection's constraints on it
    (the slab) apply, and the polygon is the slab's cut projected onto the
    displayed axes.

    Parameters
    ----------
    data_shape : Sequence[int]
        The data's shape, one entry per data axis.
    classification : ViewClassification
        The visual against a 2D view.
    selection : RegionSelection
        The view's selection; its region is read, never rebuilt (5.1).
    projection_world_axis : int or None
        ``a`` when the visual projects.

    Returns
    -------
    np.ndarray
        ``(M, 2)`` vertices in ``displayed_world_axes`` order (ascending),
        counter-clockwise, without repeats; ``(0, 2)`` when the slice
        misses the data.

    Raises
    ------
    ValueError
        If the view is not 2D, the shape has the wrong rank, the region is
        in another coordinate system, or the pullback is not parameterised
        by the free axes.
    """
    displayed = _displayed(classification)
    if len(displayed) != 2:
        raise ValueError(f"A footprint needs a 2D view; this one shows {displayed}.")
    transform = classification.effective_transform
    if selection.region.coordinate_system != transform.output_coordinate_system:
        raise ValueError(
            "The selection's region is not in the visual's world coordinate system."
        )
    shape = np.asarray(data_shape, dtype=np.float64).reshape(-1)
    if shape.shape[0] != transform.input_ndim:
        raise ValueError(
            f"data_shape has {shape.shape[0]} axes but the data has "
            f"{transform.input_ndim}."
        )
    position = _check_slice_position(selection.slice_position, transform.output_ndim)
    parts = _affine_parts(transform)
    free = list(displayed)
    if projection_world_axis is not None and np.any(
        parts.linear[int(projection_world_axis)] != 0.0
    ):
        # Along a world axis the data does not reach (a broadcast axis), the
        # data is the same at every depth: the slab's cut is the plane's.
        free.append(int(projection_world_axis))
    pinned = _pin_leaves(parts, free, position)
    for data_axis, value in pinned.items():
        if not -0.5 <= value <= shape[data_axis] - 0.5:
            return np.zeros((0, 2))
    solved = _solve_free(parts, pinned, free, position)

    rows: list[np.ndarray] = []
    bounds: list[float] = []
    # The data box: -0.5 <= offset + basis z <= n - 0.5.
    for axis in range(shape.shape[0]):
        if axis in pinned:
            continue
        rows.append(solved.basis[axis])
        bounds.append(shape[axis] - 0.5 - solved.offset[axis])
        rows.append(-solved.basis[axis])
        bounds.append(0.5 + solved.offset[axis])
    # The region, with the held world axes substituted.
    held = [axis for axis in range(position.shape[0]) if axis not in set(free)]
    for normal, offset in zip(
        selection.region.normals, selection.region.offsets, strict=True
    ):
        on_free = normal[free]
        remainder = float(offset - normal[held] @ position[held])
        if np.any(on_free != 0.0):
            rows.append(on_free)
            bounds.append(remainder)
        elif remainder < -_SOLVE_TOLERANCE * (1.0 + abs(float(offset))):
            return np.zeros((0, 2))
    return _polygon(np.array(rows), np.array(bounds), len(free))


def _polygon(normals: np.ndarray, offsets: np.ndarray, ndim: int) -> np.ndarray:
    """``{z : N z <= b}``'s vertices projected onto ``z[:2]``, as a hull."""
    scale = 1.0 + float(np.max(np.abs(offsets), initial=0.0))
    tolerance = _SOLVE_TOLERANCE * scale
    vertices: list[np.ndarray] = []
    for combination in itertools.combinations(range(normals.shape[0]), ndim):
        system = normals[list(combination)]
        if abs(np.linalg.det(system)) < 1e-12 * max(
            1.0, float(np.prod(np.linalg.norm(system, axis=1)))
        ):
            continue
        point = np.linalg.solve(system, offsets[list(combination)])
        if np.all(normals @ point <= offsets + tolerance):
            vertices.append(point[:2])
    if len(vertices) < 3:
        return np.zeros((0, 2))
    return _convex_hull(np.array(vertices), tolerance)


def _convex_hull(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Andrew's monotone chain, counter-clockwise, collinear points dropped."""
    unique = {
        (round(float(x) / tolerance), round(float(y) / tolerance)): (
            float(x),
            float(y),
        )
        for x, y in points
    }
    ordered = sorted(unique.values())
    if len(ordered) < 3:
        return np.zeros((0, 2))
    span = float(np.ptp(np.array(ordered), axis=0).max())
    area_tolerance = tolerance * max(span, 1.0)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= area_tolerance:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= area_tolerance:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return np.zeros((0, 2))
    return np.array(hull, dtype=np.float64)
