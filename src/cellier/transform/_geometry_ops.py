"""Array math for affine geometry (design section 4).

Pure functions over numpy arrays.  Nothing here constructs a pydantic
model, knows about a coordinate system, or touches a ``transformnd``
type: every function takes a linear block ``A`` and a translation ``t``
(or the pseudo-inverse of one) and returns arrays.  That is what makes
every closed form in design section 4.1 testable on its own.

The naming convention encodes which block a rule needs, which is the
substance of D27 and D39.  ``map_*`` pushes forward, ``imap_*`` pulls
back, and the argument name says whether that requires the inverse:

- ``map_points``, ``map_directions``, ``imap_normals`` and
  ``imap_half_spaces`` take ``linear``, so they work on a transform whose
  inverse does not exist.
- ``imap_points``, ``imap_directions``, ``map_normals`` and
  ``map_half_spaces`` take ``inverse_linear`` and therefore do not.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


class NonInvertibleTransformError(RuntimeError):
    """Raised when an inverse is required but cannot be computed."""


class DegenerateNormalError(RuntimeError):
    """Raised when a normal or plane maps to the zero vector (D32)."""


class NonAffineTransformError(RuntimeError):
    """Raised where a transform must be affine and is not.

    Two kinds of place raise it.  The first is the GPU boundary: a pygfx
    node transform is a 4x4 matrix, and there is no matrix that expresses a
    non-uniform axis, so a non-affine block on a *displayed* axis has no
    honest answer -- the message names the axis, because the fix is to stop
    displaying it.  The second is an operation that needs a spatially
    constant Jacobian -- a direction, a normal, a plane, a half-space --
    which an irregularly spaced axis does not have.

    It is deliberately one error rather than two: the underlying reason is
    the same in both, that there is no affine answer to hand back.
    """


# ---------------------------------------------------------------------
# input handling (D12)
# ---------------------------------------------------------------------


def _as_2d(array: np.ndarray, ndim: int, name: str) -> tuple[np.ndarray, bool]:
    """Normalize ``(N, ndim)`` or ``(ndim,)`` input to 2-D.

    Homogeneous ``(N, ndim + 1)`` input is rejected: it is an encoding
    detail of the affine implementation with no meaning for a
    displacement field or a spline (D12).

    Parameters
    ----------
    array : np.ndarray
        The input array.
    ndim : int
        The expected number of components per element.
    name : str
        The argument name, used in error messages.

    Returns
    -------
    tuple[np.ndarray, bool]
        The array as ``(N, ndim)``, and whether the input was 1-D and so
        the result should be demoted back.

    Raises
    ------
    ValueError
        If the input is not 1-D or 2-D, or its component count is not
        ``ndim``.
    """
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        if values.shape[0] != ndim:
            raise ValueError(
                f"{name} must have {ndim} components, got {values.shape[0]}."
            )
        return values.reshape(1, ndim), True
    if values.ndim == 2:
        if values.shape[1] != ndim:
            raise ValueError(
                f"{name} must have shape (N, {ndim}) or ({ndim},), got "
                f"{values.shape}.  Homogeneous input is not accepted."
            )
        return values, False
    raise ValueError(f"{name} must be 1-D or 2-D, got {values.ndim} dimensions.")


def _restore_rank(values: np.ndarray, was_1d: bool) -> np.ndarray:
    """Demote a ``(1, D)`` result back to ``(D,)`` when the input was 1-D."""
    return values[0] if was_1d else values


# ---------------------------------------------------------------------
# the inverse rule (D24)
# ---------------------------------------------------------------------


def pseudo_inverse_affine(
    linear: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the inverse affine ``(A+, -A+ t)``, or ``None`` (D24).

    Three cases, in order:

    1. Square and non-singular -> the exact inverse.
    2. Full column rank with more outputs than inputs (an embedding) ->
       the Moore-Penrose pseudo-inverse of the *linear block*, which is
       an exact **left** inverse: ``imap(map(p)) == p``.
    3. Anything else -- rank deficient, or a genuine dimension-reducing
       projection -> ``None``.

    Case 3 is not given a right-inverse.  One exists, but it returns a
    different point than the caller put in, and silently.

    The pseudo-inverse is applied to the linear block and never to the
    augmented matrix, whose pseudo-inverse does not generally carry the
    affine bottom row.

    Parameters
    ----------
    linear : np.ndarray
        The ``(D_out, D_in)`` linear block.
    translation : np.ndarray
        The ``(D_out,)`` translation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        The inverse linear block and its translation, or ``None`` when
        no left inverse exists.
    """
    matrix = np.asarray(linear, dtype=float)
    offset = np.asarray(translation, dtype=float)
    n_out, n_in = matrix.shape

    if n_out == n_in:
        if np.linalg.matrix_rank(matrix) < n_in:
            return None
        inverse = np.linalg.inv(matrix)
        return inverse, -inverse @ offset

    if n_out > n_in and np.linalg.matrix_rank(matrix) == n_in:
        inverse = np.linalg.pinv(matrix)
        return inverse, -inverse @ offset

    return None


# ---------------------------------------------------------------------
# points and vectors
# ---------------------------------------------------------------------


def map_points(
    points: np.ndarray, linear: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Push points forward: ``A p + t``.

    The result is always a distinct array, even when the affine is the
    identity (D11), so a caller may mutate it without aliasing its input.

    Parameters
    ----------
    points : np.ndarray
        ``(N, D_in)`` or ``(D_in,)`` points.
    linear : np.ndarray
        The ``(D_out, D_in)`` linear block.
    translation : np.ndarray
        The ``(D_out,)`` translation.

    Returns
    -------
    np.ndarray
        ``(N, D_out)`` or ``(D_out,)``, matching the input's rank.
    """
    matrix = np.asarray(linear, dtype=float)
    values, was_1d = _as_2d(points, matrix.shape[1], "points")
    mapped = values @ matrix.T + np.asarray(translation, dtype=float)
    return _restore_rank(mapped, was_1d)


def imap_points(
    points: np.ndarray, inverse_linear: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Pull points back: ``A+ (w - t)``.

    Parameters
    ----------
    points : np.ndarray
        ``(N, D_out)`` or ``(D_out,)`` points in the output space.
    inverse_linear : np.ndarray
        The ``(D_in, D_out)`` inverse linear block, from
        :func:`pseudo_inverse_affine`.
    translation : np.ndarray
        The **forward** ``(D_out,)`` translation.

    Returns
    -------
    np.ndarray
        ``(N, D_in)`` or ``(D_in,)``, matching the input's rank.
    """
    matrix = np.asarray(inverse_linear, dtype=float)
    values, was_1d = _as_2d(points, matrix.shape[1], "points")
    mapped = (values - np.asarray(translation, dtype=float)) @ matrix.T
    return _restore_rank(mapped, was_1d)


def map_directions(directions: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Push displacement vectors forward: ``A v`` (contravariant, D27).

    Nothing is normalized (D28): a displacement's magnitude is
    meaningful.

    Parameters
    ----------
    directions : np.ndarray
        ``(N, D_in)`` or ``(D_in,)`` vectors.
    linear : np.ndarray
        The ``(D_out, D_in)`` linear block.

    Returns
    -------
    np.ndarray
        ``(N, D_out)`` or ``(D_out,)``, matching the input's rank.
    """
    matrix = np.asarray(linear, dtype=float)
    values, was_1d = _as_2d(directions, matrix.shape[1], "directions")
    return _restore_rank(values @ matrix.T, was_1d)


def imap_directions(directions: np.ndarray, inverse_linear: np.ndarray) -> np.ndarray:
    """Pull displacement vectors back: ``A+ v`` (D27).

    Parameters
    ----------
    directions : np.ndarray
        ``(N, D_out)`` or ``(D_out,)`` vectors in the output space.
    inverse_linear : np.ndarray
        The ``(D_in, D_out)`` inverse linear block.

    Returns
    -------
    np.ndarray
        ``(N, D_in)`` or ``(D_in,)``, matching the input's rank.
    """
    matrix = np.asarray(inverse_linear, dtype=float)
    values, was_1d = _as_2d(directions, matrix.shape[1], "directions")
    return _restore_rank(values @ matrix.T, was_1d)


def _check_non_degenerate(normals: np.ndarray) -> None:
    """Raise if any row is the zero vector (D32).

    Parameters
    ----------
    normals : np.ndarray
        ``(N, D)`` mapped normals.

    Raises
    ------
    DegenerateNormalError
        If any row has zero length.
    """
    zero_rows = np.flatnonzero(~np.any(normals != 0.0, axis=1))
    if zero_rows.size:
        raise DegenerateNormalError(
            f"Normal(s) at index {zero_rows.tolist()} mapped to the zero "
            f"vector.  The preimage of that plane is the whole space, not a "
            f"plane, so there is no normal to return."
        )


def map_normals(normals: np.ndarray, inverse_linear: np.ndarray) -> np.ndarray:
    """Push plane normals forward: ``(A+)^T n`` (covariant, D27).

    A normal is not a displacement.  Using ``A`` here is correct only for
    a rigid transform and is wrong for every scale.  Nothing is
    normalized (D28).

    Parameters
    ----------
    normals : np.ndarray
        ``(N, D_in)`` or ``(D_in,)`` normals.
    inverse_linear : np.ndarray
        The ``(D_in, D_out)`` inverse linear block.

    Returns
    -------
    np.ndarray
        ``(N, D_out)`` or ``(D_out,)``, matching the input's rank.

    Raises
    ------
    DegenerateNormalError
        If a normal maps to the zero vector (D32).
    """
    matrix = np.asarray(inverse_linear, dtype=float)
    values, was_1d = _as_2d(normals, matrix.shape[0], "normals")
    mapped = values @ matrix
    _check_non_degenerate(mapped)
    return _restore_rank(mapped, was_1d)


def imap_normals(normals: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Pull plane normals back: ``A^T n'`` (D27).

    This needs only the forward linear block, so it works on a transform
    whose inverse does not exist.

    The map is **not injective**: a normal tilted along an axis the
    transform has no extent in comes back with that tilt discarded.  So
    ``map_normals(imap_normals(n)) == n`` does not hold in general.

    Parameters
    ----------
    normals : np.ndarray
        ``(N, D_out)`` or ``(D_out,)`` normals in the output space.
    linear : np.ndarray
        The ``(D_out, D_in)`` forward linear block.

    Returns
    -------
    np.ndarray
        ``(N, D_in)`` or ``(D_in,)``, matching the input's rank.

    Raises
    ------
    DegenerateNormalError
        If a normal maps to the zero vector (D32).
    """
    matrix = np.asarray(linear, dtype=float)
    values, was_1d = _as_2d(normals, matrix.shape[0], "normals")
    mapped = values @ matrix
    _check_non_degenerate(mapped)
    return _restore_rank(mapped, was_1d)


# ---------------------------------------------------------------------
# planes and half-spaces
# ---------------------------------------------------------------------


def map_plane(
    normal: np.ndarray,
    offset: float,
    inverse_linear: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Push a plane forward: ``n' = (A+)^T n``, ``d' = d + n' . t``.

    The plane is the set ``normal . p == offset``.

    Parameters
    ----------
    normal : np.ndarray
        The ``(D_in,)`` normal.
    offset : float
        The plane offset.
    inverse_linear : np.ndarray
        The ``(D_in, D_out)`` inverse linear block.
    translation : np.ndarray
        The forward ``(D_out,)`` translation.

    Returns
    -------
    tuple[np.ndarray, float]
        The mapped normal and offset.

    Raises
    ------
    DegenerateNormalError
        If the normal maps to the zero vector (D32).
    """
    mapped_normal = map_normals(normal, inverse_linear)
    mapped_offset = float(offset) + float(
        mapped_normal @ np.asarray(translation, dtype=float)
    )
    return mapped_normal, mapped_offset


def imap_plane(
    normal: np.ndarray,
    offset: float,
    linear: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Pull a plane back: ``n = A^T m``, ``d = e - m . t``.

    Needs only the forward blocks, so it works on a transform whose
    inverse does not exist.  Like :func:`imap_normals` it is not
    injective.

    Parameters
    ----------
    normal : np.ndarray
        The ``(D_out,)`` normal in the output space.
    offset : float
        The plane offset in the output space.
    linear : np.ndarray
        The ``(D_out, D_in)`` forward linear block.
    translation : np.ndarray
        The forward ``(D_out,)`` translation.

    Returns
    -------
    tuple[np.ndarray, float]
        The pulled-back normal and offset.

    Raises
    ------
    DegenerateNormalError
        If the normal maps to the zero vector (D32).
    """
    values = np.asarray(normal, dtype=float)
    mapped_normal = imap_normals(values, linear)
    mapped_offset = float(offset) - float(values @ np.asarray(translation, dtype=float))
    return mapped_normal, mapped_offset


def map_half_spaces(
    normals: np.ndarray,
    offsets: np.ndarray,
    inverse_linear: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Push half-spaces forward, preserving the inequality direction.

    The rule is the plane rule of :func:`map_plane` with ``<=`` carried
    through, which it survives.  Unlike the plane version this does
    **not** raise on a degenerate normal (D39): a half-space whose normal
    maps to zero becomes either vacuous or infeasible, and both of those
    are correct, useful answers rather than errors.

    Parameters
    ----------
    normals : np.ndarray
        ``(M, D_in)`` normals, one per constraint.
    offsets : np.ndarray
        ``(M,)`` offsets.
    inverse_linear : np.ndarray
        The ``(D_in, D_out)`` inverse linear block.
    translation : np.ndarray
        The forward ``(D_out,)`` translation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(M, D_out)`` mapped normals and ``(M,)`` mapped offsets.
    """
    matrix = np.asarray(inverse_linear, dtype=float)
    values = np.asarray(normals, dtype=float).reshape(-1, matrix.shape[0])
    mapped = values @ matrix
    mapped_offsets = np.asarray(offsets, dtype=float) + mapped @ np.asarray(
        translation, dtype=float
    )
    return mapped, mapped_offsets


def drop_broadcast_constraints(
    normals: np.ndarray,
    offsets: np.ndarray,
    broadcast_axes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop half-spaces that constrain a broadcast axis (D8).

    A broadcast axis is a **free variable**: the source occupies
    ``{M p + t + s e_b : p in data, s in R}`` in the output space, so the
    set of input points visible in ``{w : N w <= o}`` is

    ``{ p : EXISTS s,  (N M) p + N t + N[:, b] s <= o }``

    which is Fourier-Motzkin elimination of ``s`` *before* ``A^T`` is
    applied.  Eliminating ``s`` passes through every constraint with a
    zero coefficient on ``b``, combines each ``(+b, -b)`` pair into one
    new constraint, and drops constraints carrying only one sign.

    For an **axis-aligned** selection there is exactly one ``+b`` and one
    ``-b`` constraint, and their combination is
    ``0 <= (c + h) - (c - h) = 2h``, trivially true for any non-negative
    half thickness.  So the elimination degenerates exactly to "drop
    every constraint whose normal touches a broadcast axis", which is
    what this function implements.

    That is exact for every axis-aligned selection and for an oblique
    slab tilted through a broadcast axis.  It **diverges** from the true
    elimination only for an oblique slab tilted through a broadcast axis
    *intersected with* a bound on that same axis, where the pairwise
    combination produces a constraint this rule discards.  The general
    pairwise form is deferred until such a consumer exists; no shipping
    selection is one.

    Parameters
    ----------
    normals : np.ndarray
        ``(M, D_out)`` normals in the output space.
    offsets : np.ndarray
        ``(M,)`` offsets.
    broadcast_axes : tuple[int, ...]
        Output-space axis indices that are broadcast (free).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The surviving normals and offsets.
    """
    indices = tuple(broadcast_axes)
    if not indices or normals.shape[0] == 0:
        return normals, offsets
    touches = np.any(normals[:, indices] != 0.0, axis=1)
    return normals[~touches], offsets[~touches]


def imap_half_spaces(
    normals: np.ndarray,
    offsets: np.ndarray,
    linear: np.ndarray,
    translation: np.ndarray,
    broadcast_axes: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Pull half-spaces back: ``n = A^T m``, ``f = e - m . t`` (D39).

    This is the operation the slicer actually runs -- a world-space
    selection into a datastore's voxel space -- and it needs nothing but
    ``A^T``.  It is exact, cheap, and available on a transform whose
    inverse is ``None``.

    Constraints touching a broadcast axis are removed first, by
    :func:`drop_broadcast_constraints` (D8).  Without that step a
    selection on an axis the source is broadcast over pulls back through
    an all-zero row to an infeasible ``0 <= negative`` and the region
    comes back empty -- the wrong answer for a source that by definition
    exists at every position on that axis.

    A constraint on an axis the transform has no extent in, and which is
    *not* broadcast, still comes back with a zero normal.  That is not an
    error: with a non-negative offset it is vacuous ("fetch everything")
    and with a negative one it is infeasible ("fetch nothing").
    Resolving which is the region's job.

    Parameters
    ----------
    normals : np.ndarray
        ``(M, D_out)`` normals in the output space.
    offsets : np.ndarray
        ``(M,)`` offsets.
    linear : np.ndarray
        The ``(D_out, D_in)`` forward linear block.
    translation : np.ndarray
        The forward ``(D_out,)`` translation.
    broadcast_axes : tuple[int, ...]
        Output-space axis indices that are broadcast (free).  Empty
        leaves the pull-back at raw ``A^T``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(M, D_in)`` pulled-back normals and ``(M,)`` offsets.
    """
    matrix = np.asarray(linear, dtype=float)
    values = np.asarray(normals, dtype=float).reshape(-1, matrix.shape[0])
    kept_offsets = np.asarray(offsets, dtype=float)
    values, kept_offsets = drop_broadcast_constraints(
        values, kept_offsets, broadcast_axes
    )
    mapped = values @ matrix
    mapped_offsets = kept_offsets - values @ np.asarray(translation, dtype=float)
    return mapped, mapped_offsets


# ---------------------------------------------------------------------
# bounding boxes
# ---------------------------------------------------------------------


def affine_bounding_box(
    min_coordinate: np.ndarray,
    max_coordinate: np.ndarray,
    linear: np.ndarray,
    translation: np.ndarray,
    unbounded_output_axes: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Return the smallest AABB enclosing the image of a box (D30, D31).

    Computed from the centre and half-extent, ``c -> A c + t`` and
    ``h'_i = sum_j abs(A_ij) h_j``, never from the corners: transforming
    only the min and max corners yields ``min > max`` under a reflection
    and the wrong box under a rotation.

    The result is a **conservative superset**.  It equals the exact image
    box only when the linear block is diagonal or a signed permutation,
    so ``affine_bounding_box`` composed with its inverse inflates rather
    than returning the original box.

    The half-extent contraction skips zero coefficients rather than
    multiplying by them, because ``0 * inf`` is ``nan`` and one ``nan``
    poisons every output axis, not just the unbounded one (D31).

    A semi-infinite input interval (one finite bound and one infinite)
    makes every output axis it touches fully unbounded.  That is a
    superset of the true image, and so within this function's contract,
    but it is looser than the exact answer.

    Parameters
    ----------
    min_coordinate : np.ndarray
        ``(D_in,)`` lower bounds; ``-inf`` is allowed.
    max_coordinate : np.ndarray
        ``(D_in,)`` upper bounds; ``+inf`` is allowed.
    linear : np.ndarray
        The ``(D_out, D_in)`` linear block.
    translation : np.ndarray
        The ``(D_out,)`` translation.
    unbounded_output_axes : tuple[int, ...]
        Output axis indices that are unbounded regardless of the
        arithmetic -- the broadcast axes of D25.  These come back
        ``(-inf, +inf)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The ``(D_out,)`` lower and upper bounds of the enclosing box.
    """
    matrix = np.asarray(linear, dtype=float)
    lower = np.asarray(min_coordinate, dtype=float)
    upper = np.asarray(max_coordinate, dtype=float)

    # An axis with a non-finite bound has an infinite half-extent, which
    # already carries it to infinity, so its centre is unused.  It is set
    # to zero rather than computed: (+inf) + (-inf) is nan, and evaluating
    # it would emit a RuntimeWarning that this repo's pytest turns into an
    # error.
    finite = np.isfinite(lower) & np.isfinite(upper)
    centre = np.zeros_like(lower)
    half_extent = np.full(lower.shape, np.inf)
    centre[finite] = (upper[finite] + lower[finite]) / 2.0
    half_extent[finite] = (upper[finite] - lower[finite]) / 2.0

    magnitudes = np.abs(matrix)
    mapped_half = np.array(
        [np.sum(row[row != 0.0] * half_extent[row != 0.0]) for row in magnitudes]
    )
    mapped_centre = matrix @ centre + np.asarray(translation, dtype=float)

    result_min = mapped_centre - mapped_half
    result_max = mapped_centre + mapped_half
    for axis in unbounded_output_axes:
        result_min[axis] = -np.inf
        result_max[axis] = np.inf
    return result_min, result_max


# ---------------------------------------------------------------------
# polytope bounds (D40)
# ---------------------------------------------------------------------


def is_axis_aligned(normals: np.ndarray) -> bool:
    """Whether every constraint normal lies along a single axis (D40).

    A normal with at most one non-zero entry is axis-aligned; its
    magnitude does not matter, since the fast path divides it out.  A
    zero normal counts as aligned: it is a vacuous or infeasible
    constraint, which the fast path resolves without a solver.

    Parameters
    ----------
    normals : np.ndarray
        ``(M, D)`` constraint normals.

    Returns
    -------
    bool
        ``True`` when the bounds can be read off without a solver.
    """
    values = np.atleast_2d(np.asarray(normals, dtype=float))
    if values.size == 0:
        return True
    return bool(np.all(np.count_nonzero(values, axis=1) <= 1))


def axis_aligned_bounds(
    normals: np.ndarray, offsets: np.ndarray, ndim: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the bounds of an axis-aligned polytope off directly (D40).

    The fast path of :func:`polytope_bounds`, worth roughly 1000x and
    covering every axis-aligned selection shipping today.  Requires
    :func:`is_axis_aligned` to hold.

    Parameters
    ----------
    normals : np.ndarray
        ``(M, D)`` constraint normals, each with at most one non-zero.
    offsets : np.ndarray
        ``(M,)`` offsets; the constraints are ``normal . p <= offset``.
    ndim : int
        The rank of the space.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        The lower and upper bounds, with ``-inf`` / ``+inf`` on
        unbounded axes, or ``None`` when the constraints are infeasible.

    Raises
    ------
    ValueError
        If the normals are not axis-aligned.
    """
    values = np.asarray(normals, dtype=float).reshape(-1, ndim)
    bounds = np.asarray(offsets, dtype=float).reshape(-1)
    if not is_axis_aligned(values):
        raise ValueError(
            "axis_aligned_bounds requires every normal to lie along a single "
            "axis; use polytope_bounds instead."
        )

    lower = np.full(ndim, -np.inf)
    upper = np.full(ndim, np.inf)
    for normal, offset in zip(values, bounds, strict=True):
        nonzero = np.flatnonzero(normal)
        if nonzero.size == 0:
            # 0 <= offset: vacuous when satisfiable, infeasible otherwise.
            if offset < 0:
                return None
            continue
        axis = int(nonzero[0])
        coefficient = normal[axis]
        limit = offset / coefficient
        if coefficient > 0:
            upper[axis] = min(upper[axis], limit)
        else:
            lower[axis] = max(lower[axis], limit)

    if np.any(lower > upper):
        return None
    return lower, upper


def polytope_bounds(
    normals: np.ndarray, offsets: np.ndarray, ndim: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact bounds of ``{p : N p <= d}`` by ``2 * ndim`` LPs (D40).

    Unboundedness falls out of the solver rather than being special-cased
    and maps to ``-inf`` / ``+inf``, which is what makes this usable where
    vertex enumeration is not: a region with unbounded displayed axes is
    the common case, and enumeration cannot represent it.

    Measured at roughly 3 ms for ``ndim == 5``, inside a 16 ms frame
    budget.  The cost is per-solver-call overhead and is linear in
    ``2 * ndim``, not in the constraint count.  Prefer
    :func:`axis_aligned_bounds` when :func:`is_axis_aligned` holds.

    Parameters
    ----------
    normals : np.ndarray
        ``(M, ndim)`` constraint normals.
    offsets : np.ndarray
        ``(M,)`` offsets.
    ndim : int
        The rank of the space.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        The lower and upper bounds, or ``None`` when the constraints are
        infeasible.
    """
    values = np.asarray(normals, dtype=float).reshape(-1, ndim)
    bounds = np.asarray(offsets, dtype=float).reshape(-1)
    if values.shape[0] == 0:
        return np.full(ndim, -np.inf), np.full(ndim, np.inf)

    free = [(None, None)] * ndim
    lower = np.empty(ndim)
    upper = np.empty(ndim)
    feasible: bool | None = None
    for axis in range(ndim):
        objective = np.zeros(ndim)
        objective[axis] = 1.0
        for destination, sign in ((lower, 1.0), (upper, -1.0)):
            # Without presolve: HiGHS's presolve (scipy 1.17) reports some
            # unbounded problems as infeasible -- a slab tilted through
            # every axis, as a rotated 2D view pulls back to.
            result = linprog(
                sign * objective,
                A_ub=values,
                b_ub=bounds,
                bounds=free,
                options={"presolve": False},
            )
            if result.status == 0:
                destination[axis] = result.fun * sign
            elif result.status == 3:  # unbounded
                destination[axis] = -np.inf if sign > 0 else np.inf
            elif result.status == 2:
                # "Infeasible" is settled once, by a zero-objective solve: a
                # feasible region read as infeasible is unbounded this way.
                if feasible is None:
                    check = linprog(
                        np.zeros(ndim), A_ub=values, b_ub=bounds, bounds=free
                    )
                    feasible = check.status == 0
                if not feasible:
                    return None
                destination[axis] = -np.inf if sign > 0 else np.inf
            else:  # the solver gave up
                return None
    return lower, upper
