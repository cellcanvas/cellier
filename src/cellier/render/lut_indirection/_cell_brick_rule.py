"""The one rule mapping a base LUT cell to the level-k brick that owns it.

A multiscale LUT is a grid of *base cells*, one per finest-level brick (or
tile, in 2D).  Coarser levels are written into it, and a shader reads it back.
Both sides must agree on which level-k brick owns each base cell.  On a
pyramid whose level ratio is not an integer -- 389 voxels halved to 194 is a
ratio of 2.005 -- no whole number of base cells matches a level-k brick
exactly, so the answer has to be *chosen*, once, and used everywhere::

    span = max(1, round_half_up(scale))  # base cells per level-k brick
    bricks = ceil(level_shape / block_size)  # level-k brick count
    brick = min(cell // span, bricks - 1)  # the last brick owns the tail

Python writes the LUT with it and uploads ``span`` and ``bricks`` so the
shaders compute the same integer answer (``cellier.brick_rule.wgsl`` and
``cellier.tile_rule.wgsl``).  Positions *inside* a brick still use the float
scale, so geometry stays exact; the price is that a sample can land slightly
outside the brick that owns its cell and read that brick's padding.
:func:`max_out_of_brick` measures how far, and :func:`brick_rule_issues`
reports levels where that exceeds the padding.

See ``docs/Explanations/multiscale_brick_lookup.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cellier._rounding import round_half_up

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Brick count uploaded for a level whose shape is unknown.  Large enough that
#: the shader's clamp never binds, which is the same answer the LUT writer's
#: fallback count (``ceil(grid / span)``) gives for every cell in the grid.
UNBOUNDED_BRICK_COUNT = 65535


def level_cell_spans(
    n_levels: int,
    ndim: int,
    scale_vecs_data: Sequence[Sequence[float]] | None = None,
) -> list[tuple[int, ...]]:
    """Base cells per level-k brick, per level and axis.

    Parameters
    ----------
    n_levels : int
        Number of levels.  Entry ``k`` of the result is level ``k + 1``.
    ndim : int
        Number of grid axes (3 for bricks, 2 for tiles).
    scale_vecs_data : sequence of sequences of float or None
        Per-level downscale factor relative to the finest level, in the brick
        grid's axis order.  Levels without an entry fall back to the uniform
        ``2 ** (level - 1)``.

    Returns
    -------
    list[tuple[int, ...]]
        ``max(1, round_half_up(scale))`` per level and axis.
    """
    spans: list[tuple[int, ...]] = []
    for index in range(n_levels):
        if scale_vecs_data is not None and index < len(scale_vecs_data):
            scale = scale_vecs_data[index]
            spans.append(
                tuple(max(1, round_half_up(float(scale[axis]))) for axis in range(ndim))
            )
        else:
            spans.append((2**index,) * ndim)
    return spans


def level_brick_counts(
    spans: Sequence[tuple[int, ...]],
    grid_dims: Sequence[int],
    block_size: int,
    level_shapes: Sequence[Sequence[int]] | None = None,
) -> list[tuple[int, ...]]:
    """Level-k brick count, per level and axis.

    Parameters
    ----------
    spans : sequence of tuple of int
        Output of :func:`level_cell_spans`.
    grid_dims : sequence of int
        The base LUT grid's dimensions.  Used only for levels without a shape.
    block_size : int
        Brick side length in voxels.
    level_shapes : sequence of sequences of int or None
        Voxel shape of each level in the brick grid's axis order.

    Returns
    -------
    list[tuple[int, ...]]
        ``ceil(level_shape / block_size)`` when the shape is known, otherwise
        ``ceil(grid_dim / span)`` -- the count at which the clamp in
        :func:`brick_for_cell` never binds inside the grid.
    """
    counts: list[tuple[int, ...]] = []
    for index, span in enumerate(spans):
        if level_shapes is not None and index < len(level_shapes):
            shape = level_shapes[index]
            counts.append(
                tuple(
                    max(1, math.ceil(int(shape[axis]) / block_size))
                    for axis in range(len(span))
                )
            )
        else:
            counts.append(
                tuple(
                    max(1, math.ceil(int(grid_dims[axis]) / span[axis]))
                    for axis in range(len(span))
                )
            )
    return counts


def brick_for_cell(cell: int, span: int, count: int) -> int:
    """The level-k brick that owns base cell *cell* along one axis.

    This is the reference the shaders mirror.
    """
    return min(cell // span, count - 1)


def cell_range(brick: int, span: int, count: int, grid_dim: int) -> tuple[int, int]:
    """The half-open base-cell range brick *brick* owns along one axis.

    The inverse of :func:`brick_for_cell`: every cell in the range maps back
    to *brick*, and the ranges of ``0 .. count - 1`` partition the grid.  The
    last brick extends to the end of the grid so no cell is left unwritten.
    A brick whose range starts past the grid owns nothing (see
    :func:`max_out_of_brick`).
    """
    start = min(brick * span, grid_dim)
    stop = grid_dim if brick >= count - 1 else min((brick + 1) * span, grid_dim)
    return start, stop


def max_out_of_brick(
    scale: float,
    span: int,
    count: int,
    level0_extent: int,
    block_size: int,
    translation: float | None = None,
) -> tuple[float, tuple[int, ...]]:
    """How far a sample can land outside the brick that owns its cell.

    In the centre convention (plan v2, D1; ``cellier.render._level_mapping``):
    the cells brick ``g`` owns cover level-0 voxels ``[start * bs, stop *
    bs)`` (clipped to the data), i.e. data ``p`` in ``[start * bs - 0.5, stop
    * bs - 0.5]``, which is level-k ``u = (p - t) / s``; the brick's own
    interior is ``u`` in ``[g * bs - 0.5, (g + 1) * bs - 0.5]``.  The
    difference is what the brick's padding has to absorb.

    Parameters
    ----------
    scale : float
        Level-k downscale factor along this axis.
    span : int
        Base cells per level-k brick along this axis.
    count : int
        Level-k brick count along this axis.
    level0_extent : int
        Finest-level voxel count along this axis.
    block_size : int
        Brick side length in voxels.
    translation : float or None
        The level's translation ``t`` in level-0 voxels.  ``None`` assumes
        block averaging, ``(scale - 1) / 2`` -- the placement the rule was
        first written for, which gives the same numbers as before
        translations were taken into account.

    Returns
    -------
    out_of_brick : float
        The worst distance, in level-k voxels, over every brick.
    unreachable : tuple[int, ...]
        Bricks that own no cell, so their data is never drawn.
    """
    t = (scale - 1.0) / 2.0 if translation is None else float(translation)
    grid_dim = math.ceil(level0_extent / block_size)
    worst = 0.0
    unreachable: list[int] = []
    for brick in range(count):
        start, stop = cell_range(brick, span, count, grid_dim)
        if start >= stop:
            unreachable.append(brick)
            continue
        low = (block_size * start - 0.5 - t) / scale
        high = (min(block_size * stop, level0_extent) - 0.5 - t) / scale
        worst = max(
            worst,
            block_size * brick - 0.5 - low,
            high - (block_size * (brick + 1) - 0.5),
        )
    return worst, tuple(unreachable)


@dataclass(frozen=True)
class BrickRuleIssue:
    """A level and axis where the cell -> brick rule does not fit the padding.

    Attributes
    ----------
    level : int
        1-indexed level (1 = finest).
    axis : int
        Axis in the brick grid's order.
    scale : float
        The level's downscale factor along the axis.
    span : int
        Base cells per brick the rule uses.
    out_of_brick : float
        Worst distance, in level-k voxels, a sample lands outside its brick.
    allowed : float
        The distance the padding absorbs.
    unreachable : tuple[int, ...]
        Bricks that own no cell.
    """

    level: int
    axis: int
    scale: float
    span: int
    out_of_brick: float
    allowed: float
    unreachable: tuple[int, ...] = ()
    translation: float | None = None

    def describe(self) -> str:
        """A one-line human-readable description."""
        shift = (
            "" if self.translation is None else f", translation {self.translation:.4g}"
        )
        where = (
            f"level {self.level} axis {self.axis} "
            f"(scale {self.scale:.4f}{shift}, {self.span} base cells per brick)"
        )
        if self.unreachable:
            return (
                f"{where}: bricks {list(self.unreachable)} own no LUT cell, so "
                f"their data is never drawn"
            )
        return (
            f"{where}: samples land up to {self.out_of_brick:.2f} voxels outside "
            f"their brick but the padding absorbs {self.allowed:.2f}; edge "
            f"texels are repeated there"
        )


def brick_rule_issues(
    scale_vecs_data: Sequence[Sequence[float]] | None,
    level_shapes: Sequence[Sequence[int]] | None,
    block_size: int,
    border: float,
    sampling_margin: float = 0.5,
    translation_vecs_data: Sequence[Sequence[float]] | None = None,
) -> list[BrickRuleIssue]:
    """Levels and axes where the rule needs more padding than the cache has.

    Parameters
    ----------
    scale_vecs_data : sequence of sequences of float or None
        Per-level downscale factors in the brick grid's axis order.
    level_shapes : sequence of sequences of int or None
        Per-level voxel shapes in the same order.
    block_size : int
        Brick side length in voxels.
    border : float
        The cache's padding (``overlap``) in level-k voxels.
    sampling_margin : float
        Padding reserved for the sampler itself: how far past a sample
        position the path reads, in level-k voxels (plan v2, "Padding
        budgets": 0.5 for linear filtering, 0 for nearest, more for gradient
        or bisection probes).
    translation_vecs_data : sequence of sequences of float or None
        Per-level translations in level-0 voxels, same order as the scales.
        ``None`` assumes block averaging (see :func:`max_out_of_brick`).

    Returns
    -------
    list[BrickRuleIssue]
        Empty when either input is ``None`` or everything fits.
    """
    if scale_vecs_data is None or not level_shapes:
        return []
    n_levels = min(len(scale_vecs_data), len(level_shapes))
    ndim = len(level_shapes[0])
    allowed = float(border) - sampling_margin
    spans = level_cell_spans(n_levels, ndim, scale_vecs_data)
    grid_dims = tuple(math.ceil(int(n) / block_size) for n in level_shapes[0])
    counts = level_brick_counts(spans, grid_dims, block_size, level_shapes)
    issues: list[BrickRuleIssue] = []
    for index in range(n_levels):
        for axis in range(ndim):
            scale = float(scale_vecs_data[index][axis])
            translation = (
                None
                if translation_vecs_data is None
                else float(translation_vecs_data[index][axis])
            )
            out_of_brick, unreachable = max_out_of_brick(
                scale,
                spans[index][axis],
                counts[index][axis],
                int(level_shapes[0][axis]),
                block_size,
                translation,
            )
            if unreachable or out_of_brick > allowed + 1e-9:
                issues.append(
                    BrickRuleIssue(
                        level=index + 1,
                        axis=axis,
                        scale=scale,
                        span=spans[index][axis],
                        out_of_brick=out_of_brick,
                        allowed=allowed,
                        unreachable=unreachable,
                        translation=translation,
                    )
                )
    return issues
