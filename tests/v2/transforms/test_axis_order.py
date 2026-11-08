"""Tests for the axis-order primitives in cellier.render._spaces.

They lived in v1's ``cellier.transform._axis_order`` until Phase 8 and moved
v1's retirement: they subset and permute plain value sequences -- shape
tuples, scale vectors -- and only shared their names with the
``AffineTransform`` methods, which went with v1.
"""

from __future__ import annotations

import pytest

from cellier.render._spaces import select_axes, swap_axes

# ---------------------------------------------------------------------------
# select_axes (sequence)
# ---------------------------------------------------------------------------


def test_select_axes_subset_preserves_order():
    values = (10, 20, 30, 40, 50)
    assert select_axes(values, (0, 4)) == (10, 50)
    assert select_axes(values, (4, 0)) == (50, 10)
    assert select_axes(values, (1, 3, 4)) == (20, 40, 50)


def test_select_axes_empty():
    assert select_axes((1, 2, 3), ()) == ()


def test_select_axes_full_is_identity_when_in_order():
    values = (1, 2, 3)
    assert select_axes(values, (0, 1, 2)) == values


# ---------------------------------------------------------------------------
# swap_axes (sequence)
# ---------------------------------------------------------------------------


def test_swap_axes_reverse():
    assert swap_axes((10, 20, 30), (2, 1, 0)) == (30, 20, 10)


def test_swap_axes_identity():
    assert swap_axes((10, 20, 30), (0, 1, 2)) == (10, 20, 30)


def test_swap_axes_arbitrary_permutation():
    assert swap_axes(("a", "b", "c", "d"), (3, 0, 2, 1)) == ("d", "a", "c", "b")


def test_swap_axes_rejects_non_permutation():
    # repeated index
    with pytest.raises(ValueError, match="permutation"):
        swap_axes((10, 20, 30), (0, 0, 1))
    # wrong length
    with pytest.raises(ValueError, match="permutation"):
        swap_axes((10, 20, 30), (0, 1))
    # out-of-range index
    with pytest.raises(ValueError, match="permutation"):
        swap_axes((10, 20, 30), (0, 1, 5))
