"""Painter's-order LUT writes from arrays (design v3, section 5.8; D6).

A multiscale LUT is rebuilt by painting resident bricks into a grid of base
cells, coarsest first, so a finer brick covers the coarser fallback under
it.  The per-brick loop that did this cost 5 ms (3D) and 7 ms (2D) on a full
atlas, because a rebuild while loading draws the whole atlas.  Here each
(phase, level) group of bricks that covers exactly one ``span`` of cells per
axis is written with a single fancy-index assignment into a block view of
the LUT, ``lut.reshape(n0, s0, n1, s1, ...)[g0, :, g1, :, ...]``.  Tail bricks
(the last brick along an axis owns the rest of the grid), clipped bricks and
small groups take the per-brick loop.  The output is identical to the loop's.

The cell -> brick rule is :mod:`._cell_brick_rule`'s, so the LUT agrees with
the shaders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Groups smaller than this use the per-brick loop: the block view's setup
#: costs more than a handful of slice assignments.
LOOP_BELOW: int = 48


def _block_view(arr: np.ndarray, span: Sequence[int], ndim: int) -> np.ndarray:
    """``arr`` cropped to whole spans and viewed as ``(n0, s0, n1, s1, ...)``."""
    whole = [(arr.shape[a] // span[a]) * span[a] for a in range(ndim)]
    crop = arr[tuple(slice(0, w) for w in whole)]
    shape: list[int] = []
    for a in range(ndim):
        shape += [whole[a] // span[a], span[a]]
    return crop.reshape(*shape, *arr.shape[ndim:])


def paint_lut(
    lut: np.ndarray,
    values: np.ndarray,
    levels: np.ndarray,
    grids: np.ndarray,
    phases: Sequence[np.ndarray],
    spans: Sequence[Sequence[int]],
    counts: Sequence[Sequence[int]],
    *,
    extra: np.ndarray | None = None,
    extra_values: np.ndarray | None = None,
    clips: Sequence[tuple[int, ...] | None] | None = None,
    loop_below: int = LOOP_BELOW,
) -> None:
    """Clear *lut* and paint bricks into it, phase by phase.

    Parameters
    ----------
    lut : np.ndarray
        ``(*grid, C)`` (or ``grid`` shaped): the base-cell table to write.
    values : np.ndarray
        ``(N, C)``: what brick ``i`` writes into each cell it owns.
    levels : np.ndarray
        ``(N,)`` 1-based level of each brick (1 is the finest).
    grids : np.ndarray
        ``(N, ndim)`` brick grid position at its level, in the LUT's axis
        order.
    phases : sequence of np.ndarray
        Index arrays into the bricks, in painting order: a later phase covers
        an earlier one.  Within a phase, coarser levels are painted first.
    spans, counts : sequence of sequence of int
        Per level (entry ``k`` is level ``k + 1``): base cells per brick and
        brick count per axis, from ``level_cell_spans`` and
        ``level_brick_counts``.
    extra, extra_values : np.ndarray or None
        A second target painted with the same cells (3D brick max): ``grid``
        shaped, and ``(N,)``.
    clips : sequence of tuple or None
        Per phase, ``(lo_0, ..., lo_n, hi_0, ..., hi_n)`` base-cell bounds
        (half open) that phase's writes are clipped to, or ``None``.
    loop_below : int
        Groups smaller than this use the per-brick loop.  ``0`` vectorises
        everything that can be; a huge value forces the loop (tests compare
        the two).
    """
    ndim = grids.shape[1] if grids.ndim == 2 else len(spans[0])
    grid = lut.shape[:ndim]
    grid_arr = np.asarray(grid)
    whole_all = [
        np.asarray([(grid[a] // span[a]) * span[a] for a in range(ndim)])
        for span in spans
    ]
    lut[:] = 0
    if extra is not None:
        extra[:] = 0
    for phase_index, phase in enumerate(phases):
        phase = np.asarray(phase, dtype=np.int64)
        if phase.size == 0:
            continue
        clip = None if clips is None else clips[phase_index]
        phase_levels = levels[phase]
        for level in np.unique(phase_levels)[::-1].tolist():
            sel = phase[phase_levels == level]
            span = np.asarray(spans[level - 1])
            count = np.asarray(counts[level - 1])
            gs = grids[sel]
            start = gs * span
            stop = np.minimum(start + span, grid_arr)
            stop = np.where(gs >= count - 1, grid_arr, stop)
            start = np.minimum(start, grid_arr)
            regular = ((stop - start) == span).all(1) & (
                stop <= whole_all[level - 1]
            ).all(1)
            if clip is not None:
                lo = np.asarray(clip[:ndim])
                hi = np.asarray(clip[ndim:])
                keep = ~((stop <= lo) | (start >= hi)).any(1)
                regular &= ((start >= lo) & (stop <= hi)).all(1)
            else:
                keep = np.ones(len(sel), dtype=bool)
            if len(sel) < loop_below:
                regular[:] = False
            for j in np.flatnonzero(keep & ~regular).tolist():
                ranges = [(int(start[j, a]), int(stop[j, a])) for a in range(ndim)]
                if clip is not None:
                    ranges = [
                        (max(r0, clip[a]), min(r1, clip[a + ndim]))
                        for a, (r0, r1) in enumerate(ranges)
                    ]
                if any(r0 >= r1 for r0, r1 in ranges):
                    continue  # a brick that owns no cell
                cells = tuple(slice(r0, r1) for r0, r1 in ranges)
                lut[cells] = values[sel[j]]
                if extra is not None:
                    extra[cells] = extra_values[sel[j]]
            reg = np.flatnonzero(regular)
            if reg.size == 0:
                continue
            index: list = []
            for a in range(ndim):
                index += [gs[reg, a], slice(None)]
            index_t = tuple(index)
            pad = (slice(None),) + (None,) * ndim
            span_t = tuple(int(s) for s in span)
            _block_view(lut, span_t, ndim)[index_t] = values[sel[reg]][pad]
            if extra is not None:
                _block_view(extra, span_t, ndim)[index_t] = extra_values[sel[reg]][pad]
