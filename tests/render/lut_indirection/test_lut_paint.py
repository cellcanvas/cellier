"""The vectorised painter's-order LUT write (design v3 5.8, D6).

``paint_lut`` writes regular bricks with one fancy-index assignment per
(phase, level) and the rest with a per-brick loop.  Two checks:

* the vectorised path equals the loop, on random non-power-of-2 pyramids
  (odd tails, ratios like 2.005), random resident sets and phases, and random
  clips -- D6 ran the same check against today's tilemap rebuild;
* the loop equals a per-cell oracle written from the cell -> brick rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellier.render.lut_indirection._cell_brick_rule import (
    brick_for_cell,
    level_brick_counts,
    level_cell_spans,
)
from cellier.render.lut_indirection._lut_paint import paint_lut

BLOCK = 32


def _random_case(rng: np.random.Generator, ndim: int, max_bricks: int = 400):
    n_levels = int(rng.integers(2, 6))
    shape0 = tuple(int(x) for x in rng.integers(40, 900 if ndim == 2 else 500, ndim))
    ratio = rng.uniform(1.6, 2.4, (n_levels, ndim))
    ratio[0] = 1.0
    scale = np.cumprod(ratio, axis=0)
    shapes = [
        tuple(max(1, round(float(shape0[a] / scale[k, a]))) for a in range(ndim))
        for k in range(n_levels)
    ]
    scales = [tuple(float(x) for x in scale[k]) for k in range(n_levels)]
    grid = tuple(math.ceil(n / BLOCK) for n in shape0)
    spans = level_cell_spans(n_levels, ndim, scales)
    counts = level_brick_counts(spans, grid, BLOCK, shapes)
    candidates = []
    for k in range(n_levels):
        cells = np.stack(
            np.meshgrid(*[np.arange(c) for c in counts[k]], indexing="ij"), -1
        ).reshape(-1, ndim)
        candidates.append(np.column_stack([np.full(len(cells), k + 1), cells]))
    candidates = np.concatenate(candidates)
    n = int(rng.integers(1, min(len(candidates), max_bricks) + 1))
    rows = candidates[rng.choice(len(candidates), n, replace=False)]
    # Random phases: a partition of the bricks, painted in order.  Bricks of
    # one level in one phase never overlap (distinct grid positions).
    n_phases = int(rng.integers(1, 4))
    phase_of = rng.integers(0, n_phases, n)
    phases = [np.flatnonzero(phase_of == p) for p in range(n_phases)]
    values = rng.integers(1, 250, (n, 4)).astype(np.uint8)
    extra = rng.random(n).astype(np.float32)
    return grid, rows, phases, values, extra, spans, counts


def _paint(grid, rows, phases, values, extra, spans, counts, loop_below, clips=None):
    lut = np.full((*grid, 4), 77, np.uint8)
    bmax = np.full(grid, 5.0, np.float32)
    paint_lut(
        lut,
        values,
        rows[:, 0],
        rows[:, 1:],
        phases,
        spans,
        counts,
        extra=bmax,
        extra_values=extra,
        clips=clips,
        loop_below=loop_below,
    )
    return lut, bmax


@pytest.mark.parametrize("ndim", [2, 3])
def test_vectorised_equals_the_loop_on_random_pyramids(ndim: int) -> None:
    rng = np.random.default_rng(ndim)
    for _ in range(150):
        grid, rows, phases, values, extra, spans, counts = _random_case(rng, ndim)
        clips = None
        if rng.random() < 0.5:
            lo = [int(rng.integers(0, d)) for d in grid]
            hi = [int(rng.integers(lo[a] + 1, grid[a] + 1)) for a in range(ndim)]
            clips = [tuple(lo + hi) if rng.random() < 0.5 else None for _ in phases]
        vec = _paint(grid, rows, phases, values, extra, spans, counts, 0, clips)
        loop = _paint(grid, rows, phases, values, extra, spans, counts, 10**9, clips)
        np.testing.assert_array_equal(vec[0], loop[0])
        np.testing.assert_array_equal(vec[1], loop[1])


def test_the_loop_matches_a_per_cell_oracle() -> None:
    """Each cell shows the last brick, in painting order, whose rule owns it."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        grid, rows, phases, values, extra, spans, counts = _random_case(
            rng, 2, max_bricks=60
        )
        lut, _ = _paint(grid, rows, phases, values, extra, spans, counts, 10**9)
        expected = np.zeros((*grid, 4), np.uint8)
        order = [
            i
            for phase in phases
            for i in sorted(phase.tolist(), key=lambda i: -rows[i, 0])
        ]
        for i in order:
            level = int(rows[i, 0])
            span, count = spans[level - 1], counts[level - 1]
            for cell in np.ndindex(grid):
                owner = tuple(
                    brick_for_cell(cell[a], span[a], count[a]) for a in range(2)
                )
                if owner == tuple(int(g) for g in rows[i, 1:]):
                    expected[cell] = values[i]
        np.testing.assert_array_equal(lut, expected)


def test_everything_is_cleared_first() -> None:
    lut = np.full((4, 4, 4), 9, np.uint8)
    paint_lut(
        lut,
        np.empty((0, 4), np.uint8),
        np.empty(0, np.int64),
        np.empty((0, 2), np.int64),
        [],
        [(1, 1)],
        [(4, 4)],
    )
    assert not lut.any()
