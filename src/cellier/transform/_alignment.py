"""How a transform lines data up with one view (full affine design v2, 3.1 and 4.1).

Whether a ``data -> world`` transform is easy to draw is a question about a
**view**: which world axes are displayed and which are sliced.  The same
rotation is harmless in a 3D view and tilts the slice through the data in a
2D one.

For one visual in one view, with ``L`` the ``(n_world, n_data)`` linear block
of ``data -> world``, ``D`` the displayed world axes and ``S`` the sliced
ones:

* ``C`` (**collapsed** data axes) -- data axes with a non-zero entry in any
  sliced row;
* ``R`` (**retained** data axes) -- the others.

The world slice ``{w : w[S] = s}`` pulls back to a data-aligned plane spanned
by the ``R`` axes exactly when ``|R| = n_data - rank(L[S, :])``.  Then:

=================  =========================================================
``OBLIQUE``        ``|R| < n_data - rank(L[S, :])``: the slice is a tilted
                   plane through the data.
``SLICE_SHIFT``    not oblique, and ``L[D, C] != 0``: the slice position
                   translates the displayed slab.
``AXIS_ALIGNED``   not oblique, ``L[D, C] = 0`` and ``L[D, R]`` a scaled
                   permutation: what worked before rotation support.
``IN_PLANE``       not oblique, ``L[D, C] = 0``, ``L[D, R]`` dense: only the
                   placement of the fetched slab changes.
=================  =========================================================

The slice pulls back to a flat set of dimension
``pullback_dim = n_data - rank(L[S, :])``; the data axes it moves along --
those with a non-zero component in the null space of ``L[S, :]`` -- are the
**varying** data axes ``V`` (oblique 2D design v3, 4.1).  ``V`` is ``R``
unless the view is ``OBLIQUE``, where it has more axes than the pullback has
dimensions.

**"Non-zero" follows decision D7**: an entry counts as zero when
``|L_ij| <= tau * min(max|L[i, :]|, max|L[:, j]|)`` with
``tau = COUPLING_TOLERANCE``.  Floating-point noise -- a 90 degree rotation
built from ``cos`` and ``sin`` carries ``6e-17`` -- would otherwise turn an
axis-aligned view oblique.  The snapped entries are zeroed in
:attr:`ViewClassification.effective_transform`, which is what every consumer
places and slices with, so the classification and the drawing agree; the
user's transform is never modified.

Containers answer structurally: a ``ByDimensionTransform`` reports its
blocks' patterns without forming a matrix, and a non-affine block is dense
within itself.  Blocks partition both axis sets, so a row's or a column's
entries all live in one block, and the per-block tolerance is the global one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellier.transform._base import BaseTransform

__all__ = [
    "COUPLING_TOLERANCE",
    "SliceAlignment",
    "ViewClassification",
    "classify_view",
    "coupling_pattern",
    "effective_transform",
    "null_space_support",
]

#: D7: the relative size below which a matrix entry counts as zero.  Float32
#: pipelines need at least ``1e-6``; a snapped genuine coupling moves a world
#: axis by at most ``tau * extent`` voxels (0.1 voxel across 100k voxels).
COUPLING_TOLERANCE = 1e-6


class SliceAlignment(str, Enum):
    """How one visual's data lines up with one view (module docstring)."""

    AXIS_ALIGNED = "axis_aligned"
    IN_PLANE = "in_plane"
    SLICE_SHIFT = "slice_shift"
    OBLIQUE = "oblique"


@dataclass(frozen=True)
class ViewClassification:
    """The alignment of one transform with one view, and the axes behind it.

    Attributes
    ----------
    alignment : SliceAlignment
        The case.
    retained_data_axes : tuple[int, ...]
        ``R``, ascending.
    collapsed_data_axes : tuple[int, ...]
        ``C``, ascending.
    sliced_world_axes : tuple[int, ...]
        ``S``, ascending: every world axis that is not displayed.
    effective_transform : BaseTransform
        The transform with D7's snapped entries set to exactly zero -- the
        one consumers place and slice with.  The very same object as the
        input when nothing snaps.
    pullback_dim : int
        ``n_data - rank(L[S, :])``: the dimension of the set of data points
        the slice selects.  Raster draws only when it is 2.
    varying_data_axes : tuple[int, ...]
        ``V``, ascending: the data axes the slice moves along.
    """

    alignment: SliceAlignment
    retained_data_axes: tuple[int, ...]
    collapsed_data_axes: tuple[int, ...]
    sliced_world_axes: tuple[int, ...]
    effective_transform: BaseTransform
    pullback_dim: int
    varying_data_axes: tuple[int, ...]


def coupling_pattern(
    linear: np.ndarray, tolerance: float = COUPLING_TOLERANCE
) -> np.ndarray:
    """The non-zero pattern of a linear block under D7's relative tolerance.

    Parameters
    ----------
    linear : np.ndarray
        ``(n_out, n_in)`` matrix.
    tolerance : float
        ``tau``; ``0`` compares exactly.

    Returns
    -------
    np.ndarray
        Booleans of the same shape: ``True`` where the entry is a coupling.
    """
    magnitude = np.abs(np.asarray(linear, dtype=np.float64))
    if magnitude.size == 0:
        return magnitude > 0.0
    if tolerance == 0.0:
        return magnitude > 0.0
    row = magnitude.max(axis=1, keepdims=True)
    column = magnitude.max(axis=0, keepdims=True)
    return magnitude > tolerance * np.minimum(row, column)


def _rank_of_rows(transform: BaseTransform, rows: Sequence[int]) -> int:
    """Rank of the snapped linear block restricted to output *rows*.

    Numeric for an affine; summed over blocks for a container, whose blocks
    partition both axis sets (so its matrix is block-structured); a
    non-affine leaf is 1-D and contributes 1 when its row is asked for.
    """
    rows = [int(row) for row in rows]
    if not rows:
        return 0
    from cellier.transform._affine import AffineTransform
    from cellier.transform._by_dimension import ByDimensionTransform

    if isinstance(transform, AffineTransform):
        linear = np.asarray(transform.linear, dtype=np.float64)
        snapped = np.where(coupling_pattern(linear), linear, 0.0)
        return int(np.linalg.matrix_rank(snapped[rows, :]))
    if isinstance(transform, ByDimensionTransform):
        total = 0
        for block in transform.blocks:
            local = [
                index
                for index, axis in enumerate(block.output_axes)
                if axis in set(rows)
            ]
            total += _rank_of_rows(block.transform, local)
        return total
    # A 1-D non-affine leaf.
    return int(transform.axis_coupling()[rows].any())


#: The size, relative to 1, below which a component of an orthonormal
#: null-space basis counts as zero.  Far above SVD noise (``1e-15``) on the
#: column-normalised block, far below any coupling D7 keeps.
_NULL_SPACE_TOLERANCE = 1e-9


def null_space_support(linear: np.ndarray) -> tuple[int, ...]:
    """The columns with a non-zero component in the null space of *linear*.

    Columns are scaled to unit maximum first, which leaves the support
    unchanged (``L x = 0`` exactly when ``(L D)(D^-1 x) = 0``) and keeps the
    tolerance independent of each data axis's units; rows are scaled
    likewise.  A zero column is in the null space on its own.

    Parameters
    ----------
    linear : np.ndarray
        ``(n_rows, n_columns)``, already snapped.

    Returns
    -------
    tuple[int, ...]
        Column indices, ascending.
    """
    matrix = np.asarray(linear, dtype=np.float64)
    n_columns = matrix.shape[1]
    if matrix.shape[0] == 0:
        return tuple(range(n_columns))
    column = np.abs(matrix).max(axis=0)
    matrix = matrix / np.where(column > 0.0, column, 1.0)
    row = np.abs(matrix).max(axis=1, keepdims=True)
    matrix = matrix / np.where(row > 0.0, row, 1.0)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank == n_columns:
        return ()
    _, _, vh = np.linalg.svd(matrix)
    basis = vh[rank:].T
    weight = np.linalg.norm(basis, axis=1)
    return tuple(int(axis) for axis in np.flatnonzero(weight > _NULL_SPACE_TOLERANCE))


def _varying_axes(transform: BaseTransform, rows: Sequence[int]) -> tuple[int, ...]:
    """``V``: the input axes the slice of output *rows* moves along.

    Numeric for an affine; assembled from the blocks for a container (the
    null space of a block-diagonal matrix is the product of the blocks'); a
    non-affine leaf is 1-D and varies when its row is not pinned.
    """
    from cellier.transform._affine import AffineTransform
    from cellier.transform._by_dimension import ByDimensionTransform

    pinned = {int(row) for row in rows}
    if isinstance(transform, AffineTransform):
        linear = np.asarray(transform.linear, dtype=np.float64)
        snapped = np.where(coupling_pattern(linear), linear, 0.0)
        return null_space_support(snapped[sorted(pinned), :])
    if isinstance(transform, ByDimensionTransform):
        varying: list[int] = []
        for block in transform.blocks:
            local = [
                index for index, axis in enumerate(block.output_axes) if axis in pinned
            ]
            varying.extend(
                block.input_axes[index]
                for index in _varying_axes(block.transform, local)
            )
        return tuple(sorted(varying))
    # A 1-D non-affine leaf.
    return () if transform.axis_coupling()[sorted(pinned)].any() else (0,)


def _effective(transform: BaseTransform) -> BaseTransform:
    """*transform* with D7's snapped entries zeroed; itself if none snap."""
    from cellier.transform._affine import AffineTransform
    from cellier.transform._by_dimension import ByDimensionTransform

    if isinstance(transform, AffineTransform):
        linear = np.asarray(transform.linear, dtype=np.float64)
        pattern = coupling_pattern(linear)
        if np.array_equal(pattern, linear != 0.0):
            return transform
        matrix = np.array(transform.matrix, dtype=np.float64)
        matrix[:-1, :-1] = np.where(pattern, linear, 0.0)
        return AffineTransform(
            name=transform.name,
            transform=matrix,
            input_coordinate_system=transform.input_coordinate_system,
            output_coordinate_system=transform.output_coordinate_system,
            broadcast_axes=transform.broadcast_axes,
        )
    if isinstance(transform, ByDimensionTransform):
        blocks = []
        changed = False
        for block in transform.blocks:
            inner = _effective(block.transform)
            changed = changed or inner is not block.transform
            blocks.append(
                block
                if inner is block.transform
                else block.model_copy(update={"transform": inner})
            )
        if not changed:
            return transform
        return ByDimensionTransform(
            name=transform.name,
            blocks=tuple(blocks),
            input_coordinate_system=transform.input_coordinate_system,
            output_coordinate_system=transform.output_coordinate_system,
        )
    return transform


def effective_transform(transform: BaseTransform | None) -> BaseTransform | None:
    """*transform* with D7's snapped entries set to exactly zero.

    What every render and slicing consumer places and slices with, so that
    they agree with :func:`classify_view` -- a 90 degree rotation built from
    ``cos`` and ``sin`` carries ``6e-17`` where the classifier sees zero, and
    an axis-aligned path reading the raw matrix would see a rotation.  The
    view does not matter: the tolerance is relative to the whole matrix's
    rows and columns.  ``ViewClassification.effective_transform`` is the
    same object.

    Parameters
    ----------
    transform : BaseTransform or None
        A ``data -> world`` transform.

    Returns
    -------
    BaseTransform or None
        The very same object when nothing snaps (and ``None`` for ``None``);
        otherwise a new transform with the same coordinate systems.
    """
    if transform is None:
        return None
    return _effective(transform)


def classify_view(
    transform: BaseTransform, displayed_world_axes: Sequence[int]
) -> ViewClassification:
    """Classify *transform* against the view that displays *displayed_world_axes*.

    Parameters
    ----------
    transform : BaseTransform
        A visual's ``data -> world`` transform.
    displayed_world_axes : Sequence[int]
        The displayed **world** axes, by index, in any order.

    Returns
    -------
    ViewClassification
        The case, ``R``, ``C``, ``S``, the effective transform, the
        pullback's dimension and ``V``.

    Raises
    ------
    ValueError
        If a displayed axis is out of range or repeated.

    Notes
    -----
    "Zero" follows decision D7: an entry is a coupling only when it exceeds
    ``COUPLING_TOLERANCE`` (``1e-6``) times the smaller of its row's and its
    column's largest magnitude, so a rotation built with ``cos`` and ``sin``
    (``6e-17`` off) or a float32 pipeline (``1e-7`` off) classifies as the
    exact matrix would.  The snapped entries are zero in
    ``effective_transform``, which drawing and slicing use; the worst a
    snapped genuine coupling moves data is ``1e-6`` times the axis extent.

    See Also
    --------
    cellier.transform.BaseTransform.axis_coupling : the pattern classified.
    cellier.transform.effective_transform : the snapped transform.
    """
    pattern = transform.axis_coupling()
    n_world, n_data = pattern.shape
    displayed = [int(axis) for axis in displayed_world_axes]
    if len(set(displayed)) != len(displayed):
        raise ValueError(f"Displayed world axes repeat: {displayed}.")
    out_of_range = [axis for axis in displayed if not 0 <= axis < n_world]
    if out_of_range:
        raise ValueError(
            f"Displayed world axes {out_of_range} are not axes of a "
            f"{n_world}-axis world."
        )
    shown = sorted(displayed)
    sliced = [axis for axis in range(n_world) if axis not in shown]
    collapsed = [axis for axis in range(n_data) if pattern[sliced, axis].any()]
    retained = [axis for axis in range(n_data) if axis not in collapsed]
    pullback_dim = n_data - _rank_of_rows(transform, sliced)

    if len(retained) < pullback_dim:
        alignment = SliceAlignment.OBLIQUE
    elif collapsed and pattern[np.ix_(shown, collapsed)].any():
        alignment = SliceAlignment.SLICE_SHIFT
    else:
        block = pattern[np.ix_(shown, retained)]
        permutation = bool(
            (block.sum(axis=1) <= 1).all() and (block.sum(axis=0) == 1).all()
        )
        alignment = (
            SliceAlignment.AXIS_ALIGNED if permutation else SliceAlignment.IN_PLANE
        )
    return ViewClassification(
        alignment=alignment,
        retained_data_axes=tuple(retained),
        collapsed_data_axes=tuple(collapsed),
        sliced_world_axes=tuple(sliced),
        effective_transform=_effective(transform),
        pullback_dim=pullback_dim,
        varying_data_axes=_varying_axes(transform, sliced),
    )
