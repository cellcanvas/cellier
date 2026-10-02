"""The per-axis bounds index (``plans/mesh_refactor_v3.md`` S4).

Its queries are checked against a pass over every item, on random items:
points (``min == max``), short segments, and a few long outliers.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data._bounds_index import AxisBoundsIndex


def _items(rng, n: int, *, long: int = 0, points: bool = False):
    """Random float32 ``(min, max)`` pairs; *long* of them span everything."""
    low = rng.uniform(-50.0, 50.0, n).astype(np.float32)
    extent = np.zeros(n) if points else rng.uniform(0.0, 2.0, n)
    high = (low + extent).astype(np.float32)
    if long:
        which = rng.choice(n, long, replace=False)
        low[which] = -60.0
        high[which] = 60.0
    return low, high


def _index(low, high, **kwargs) -> AxisBoundsIndex:
    return AxisBoundsIndex.build(low, high, lambda ids: high[ids], **kwargs)


def _intervals(rng, low, high, n: int):
    """Query intervals: random, zero-width, on item bounds, and open-ended."""
    out = []
    for _ in range(n):
        a, b = np.sort(rng.uniform(-55.0, 55.0, 2))
        out.append((float(a), float(b)))
        out.append((float(a), float(a)))
    for i in rng.choice(len(low), min(n, len(low)), replace=False):
        # Exactly on an item's bounds: the comparisons are inclusive.
        out.append((float(low[i]), float(high[i])))
        out.append((float(high[i]), float(high[i])))
    out += [(-np.inf, 0.0), (0.0, np.inf), (-np.inf, np.inf)]
    return out


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("long", [0, 3])
@pytest.mark.parametrize("points", [False, True])
def test_queries_equal_a_pass_over_every_item(seed, long, points):
    rng = np.random.default_rng(seed)
    low, high = _items(rng, 400, long=long, points=points)
    index = _index(low, high)
    for lo, hi in _intervals(rng, low, high, 25):
        contained = np.flatnonzero((low >= lo) & (high <= hi))
        overlapping = np.flatnonzero((low <= hi) & (high >= lo))
        between = np.flatnonzero((low >= lo) & (low <= hi))
        assert np.array_equal(np.sort(index.contained(lo, hi)), contained)
        assert np.array_equal(np.sort(index.overlapping(lo, hi)), overlapping)
        assert np.array_equal(np.sort(index.min_between(lo, hi)), between)
        assert index.count_min_between(lo, hi) >= len(between)


def test_a_long_item_goes_in_the_long_list():
    rng = np.random.default_rng(0)
    low, high = _items(rng, 200, long=2)
    index = _index(low, high)
    assert len(index.long_ids) == 2
    assert index.longest < 2.1
    # The long items are found by a query far from any short item's reach.
    assert set(index.long_ids) <= set(index.overlapping(59.0, 59.5))


def test_without_a_long_list_the_window_widens():
    rng = np.random.default_rng(0)
    low, high = _items(rng, 200, long=2)
    index = _index(low, high, long_factor=np.inf)
    assert len(index.long_ids) == 0
    assert index.longest >= 120.0
    overlapping = np.flatnonzero((low <= 1.0) & (high >= 0.0))
    assert np.array_equal(np.sort(index.overlapping(0.0, 1.0)), overlapping)


def test_bounds_not_representable_in_float32_are_exact():
    """A bound between two float32 values must not round onto an item."""
    value = np.float32(0.1)
    low = np.array([value], dtype=np.float32)
    index = _index(low, low.copy())
    just_above = float(value) + 1e-12
    just_below = float(value) - 1e-12
    assert len(index.contained(just_above, 1.0)) == 0
    assert len(index.contained(just_below, 1.0)) == 1
    assert len(index.contained(-1.0, just_below)) == 0
    assert len(index.contained(-1.0, just_above)) == 1
    assert len(index.overlapping(just_above, 1.0)) == 0
    assert len(index.overlapping(-1.0, just_below)) == 0


def test_an_empty_index_answers_nothing():
    empty = np.zeros(0, dtype=np.float32)
    index = _index(empty, empty)
    assert len(index.contained(-1.0, 1.0)) == 0
    assert len(index.overlapping(-1.0, 1.0)) == 0
    assert index.nbytes == 0


def test_eight_bytes_per_item():
    rng = np.random.default_rng(1)
    low, high = _items(rng, 1000)
    assert _index(low, high).nbytes == 8 * 1000
