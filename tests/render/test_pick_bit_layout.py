"""The pick-position layout: how 44 bits are shared between a volume's axes.

The table cases are worked by hand from the rule in ``pick_bit_layout``'s
docstring.  The sweep checks the rule's guarantees on many shapes, and the
round-trip test stores and reads back positions as the shader and decoder
will, in f32, to check that a pick inside the budget decodes to its own voxel.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._pick_layout import (
    MAX_FIELD_BITS,
    PICK_POSITION_BITS,
    PickBitLayout,
    pick_bit_layout,
)

#: ``(x, y, z) voxels -> (index bits, fraction bits, widths)``.
TABLE = [
    # A small test volume: all 44 bits used, steps of 1/1024 and 1/512 voxel.
    ((40, 32, 24), (6, 5, 5), (10, 9, 9), (16, 14, 14)),
    # The largest volume the default block_size loads: the spare bit goes to
    # the longest axes, ties in x, y, z order.
    ((8160, 8160, 8160), (13, 13, 13), (2, 2, 1), (15, 15, 14)),
    ((16384, 16384, 16384), (14, 14, 14), (1, 1, 1 - 1), (15, 15, 14)),
    # Lightsheet-like: the short axis gives its bits to the long ones.
    ((2048, 10000, 40000), (11, 14, 16), (1, 1, 1), (12, 15, 17)),
    ((2, 2, 2), (1, 1, 1), (14, 14, 13), (15, 15, 14)),
    # The long axis is capped at 24 bits; three bits stay unused.
    ((1, 7, 300000), (0, 3, 19), (7, 7, 5), (7, 10, 24)),
    # Over budget by one bit: the widest field gives it up (ties: x first).
    ((32640, 32640, 32640), (15, 15, 15), (-1, 0, 0), (14, 15, 15)),
    ((100000, 100000, 2000), (17, 17, 11), (-1, 0, 0), (16, 17, 11)),
]


@pytest.mark.parametrize(("size", "index", "frac", "widths"), TABLE)
def test_worked_layouts(size, index, frac, widths):
    layout = pick_bit_layout(size)
    assert layout == PickBitLayout(
        dataset_size=size, index_bits=index, frac_bits=frac, widths=widths
    )
    assert layout.step == tuple(2.0**-f for f in frac)
    assert layout.over_budget == any(f < 0 for f in frac)


def test_over_budget_matches_the_documented_examples():
    assert not pick_bit_layout((2048, 10000, 40000)).over_budget
    assert pick_bit_layout((32640, 32640, 32640)).step == (2.0, 1.0, 1.0)
    # One axis longer than 2 ** 24 voxels cannot fit a 24-bit field.
    assert pick_bit_layout((2**24 + 1, 1, 1)).over_budget
    assert not pick_bit_layout((2**24, 1, 1)).over_budget


def _sweep_sizes():
    rng = np.random.default_rng(0)
    sizes = [tuple(int(v) for v in rng.integers(1, 2**26, 3)) for _ in range(300)]
    sizes += [
        tuple(int(2 ** rng.uniform(0, 26)) + 1 for _ in range(3)) for _ in range(300)
    ]
    for p in (0, 1, 13, 14, 15, 16, 24, 25):
        for d in (-1, 0, 1):
            n = max(1, 2**p + d)
            sizes += [(n, n, n), (n, 1, 1), (1, n, 3), (5, 9, n)]
    return sizes


def test_rule_guarantees_on_a_sweep_of_shapes():
    for size in _sweep_sizes():
        layout = pick_bit_layout(size)
        index = tuple((n - 1).bit_length() for n in size)
        assert layout.index_bits == index, size
        assert layout.widths == tuple(
            i + f for i, f in zip(layout.index_bits, layout.frac_bits, strict=True)
        ), size
        assert sum(layout.widths) <= PICK_POSITION_BITS, size
        assert all(0 <= w <= MAX_FIELD_BITS for w in layout.widths), size
        fits = sum(index) <= PICK_POSITION_BITS and max(index) <= MAX_FIELD_BITS
        assert layout.over_budget == (not fits), size
        if not any(w == MAX_FIELD_BITS for w in layout.widths):
            # No cap was hit, so no bit is wasted.
            assert sum(layout.widths) == PICK_POSITION_BITS, size


def test_inputs_are_checked():
    assert pick_bit_layout(np.array([40, 32, 24])) == pick_bit_layout((40, 32, 24))
    with pytest.raises(ValueError, match="3 entries"):
        pick_bit_layout((40, 32))
    with pytest.raises(ValueError, match="at least 1"):
        pick_bit_layout((40, 0, 24))


def _store(layout, v):
    """Store positions *v* as the shader will: ``floor(v * 2 ** f)`` in f32."""
    scale = np.exp2(np.asarray(layout.frac_bits, dtype=np.float32))
    return np.floor(v.astype(np.float32) * scale).astype(np.int64)


@pytest.mark.parametrize(
    "size",
    [s for s, *_ in TABLE] + [(3, 50000, 70000), (16777216, 1, 1), (3, 5, 2**15 + 1)],
)
def test_a_stored_pick_reads_back_inside_its_voxel(size):
    layout = pick_bit_layout(size)
    rng = np.random.default_rng(1)
    n = np.asarray(size, dtype=np.float64)
    n32 = n.astype(np.float32)
    below_n = np.nextafter(n32, np.float32(0.0))
    # Points anywhere, points just either side of a voxel face, and the ends.
    faces = rng.integers(0, n, (2000, 3)).astype(np.float64)
    v = np.concatenate(
        [
            rng.random((4000, 3)) * n,
            faces + rng.uniform(1e-3, 0.1, (2000, 3)),
            np.maximum(faces - rng.uniform(1e-3, 0.1, (2000, 3)), 0.0),
            np.zeros((1, 3)),
            below_n[None, :],
        ]
    ).astype(np.float32)
    v = np.minimum(v, below_n).astype(np.float64)
    q = _store(layout, v)
    # Every point inside the volume fits its field, so the shader's clamp to
    # the field never changes one.
    assert (q >= 0).all()
    assert (q < 2 ** np.asarray(layout.widths, dtype=np.int64)).all()
    decoded = layout.decode(q)
    assert ((decoded >= 0) & (decoded < n)).all()
    # The decoded point is within half a step of the pick.
    assert (np.abs(decoded - v) <= 0.5 * np.asarray(layout.step)).all()
    off = np.abs(np.floor(decoded) - np.floor(v))
    if layout.over_budget:
        assert (off <= np.asarray(layout.step) - 1).all()
    else:
        assert not off.any(), f"{(off > 0).sum(axis=0)} picks read another voxel"


@pytest.mark.parametrize(
    "size", [(40, 32, 24), (3, 5, 7), (32640, 32640, 32640), (3, 3, 2**40)]
)
def test_a_value_past_the_volume_reads_as_its_last_voxel(size):
    """A hit on the far face, ``v == n``, stores a value past the last step."""
    layout = pick_bit_layout(size)
    n = np.asarray(size, dtype=np.float64)
    q = _store(layout, n[None, :])
    top = 2 ** np.asarray(layout.widths, dtype=np.int64) - 1
    for values in (q, np.minimum(q, top), top[None, :]):
        decoded = layout.decode(values)[0]
        assert (decoded < n).all()
        assert (np.floor(decoded) >= n - np.maximum(layout.step, 1.0)).all()
    assert (layout.decode([[-5, -5, -5]])[0] >= 0).all()
