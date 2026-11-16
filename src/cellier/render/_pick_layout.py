"""How a 3D brick-shader pick shares its 44 position bits between the axes.

pygfx's pick texel is 64 bits.  The low 20 hold the object id, which pygfx's
``get_pick_info`` and cellier's outline and ambient-occlusion passes read, so
the pick position has the other 44.  A fixed 14 bits per axis can address at
most 16383 positions along any axis, so on an axis longer than that a pick
cannot name its voxel.  Here the bits follow the axis lengths instead: each
axis first gets the bits it needs to name every voxel, and what is left over
splits a voxel into finer steps.

Each axis ``a`` of ``n_a`` voxels gets a field of ``w_a`` bits:

- ``i_a = bit_length(n_a - 1)`` index bits name the voxel;
- ``f_a = w_a - i_a`` fraction bits split it into ``2 ** f_a`` steps.

A pick at continuous voxel position ``v`` (voxel ``j`` spans ``[j, j + 1)``) is
stored as ``q = floor(v * 2 ** f_a)`` and read back as the centre of its step,
``(q + 0.5) * 2 ** -f_a`` (`PickBitLayout.decode`).  The steps are powers
of two, so with ``f_a >= 0`` each step lies inside one voxel and the decoded
point floors to the voxel the pick hit.  When the voxels need more than 44
bits, ``f_a`` goes negative on the widest field: a step is then ``2 ** -f_a``
voxels and the decoded voxel can be off by up to ``2 ** -f_a - 1``.

The shader that writes the field and the decoder that reads it both take
their layout from `pick_bit_layout`, so they cannot disagree.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Bits of the pick texel left for the position after the 20-bit object id.
PICK_POSITION_BITS = 44

#: The widest field.  The shader computes ``v * 2 ** f`` in f32, whose integer
#: part is exact up to 24 bits.
MAX_FIELD_BITS = 24


@dataclass(frozen=True)
class PickBitLayout:
    """The pick-position fields of one volume, per axis in pygfx ``(x, y, z)``.

    Attributes
    ----------
    dataset_size : tuple[int, int, int]
        The level-0 voxel counts the layout was made for.
    index_bits : tuple[int, int, int]
        Bits needed to name every voxel on the axis, ``bit_length(n - 1)``.
    frac_bits : tuple[int, int, int]
        Bits past the voxel index.  Negative when a step spans several voxels.
    widths : tuple[int, int, int]
        Field widths, ``index_bits + frac_bits``; they sum to at most 44 and
        each is between 0 and 24.
    """

    dataset_size: tuple[int, int, int]
    index_bits: tuple[int, int, int]
    frac_bits: tuple[int, int, int]
    widths: tuple[int, int, int]

    @property
    def step(self) -> tuple[float, float, float]:
        """The size of one encoded step on each axis, in level-0 voxels."""
        return tuple(2.0**-f for f in self.frac_bits)

    @property
    def over_budget(self) -> bool:
        """Whether a step spans more than one voxel on any axis.

        True when the volume's index bits add up to more than 44, or an axis
        is longer than ``2 ** 24`` voxels.  A pick can then read a neighbour
        of the voxel it hit.
        """
        return any(f < 0 for f in self.frac_bits)

    def decode(self, values) -> np.ndarray:
        """Continuous level-0 voxel positions of stored field *values*.

        Parameters
        ----------
        values : array_like of int, shape (..., 3)
            The ``(x, y, z)`` fields read from the pick texel.

        Returns
        -------
        numpy.ndarray of float64, shape (..., 3)
            The centre of each value's step, ``(q + 0.5) * step``.  A value
            past the volume's last step (a hit on the far face, which a field
            wider than the volume can hold) reads as the last step, and the
            last step's centre is taken over the voxels inside the volume, so
            every result lies in ``[0, n)`` and floors to a voxel that exists.
        """
        q = np.asarray(values, dtype=np.int64)
        size = np.asarray(self.dataset_size, dtype=np.int64)
        frac = np.asarray(self.frac_bits, dtype=np.int64)
        up = np.maximum(frac, 0)
        down = np.maximum(-frac, 0)
        last = np.where(frac >= 0, (size << up) - 1, (size - 1) >> down)
        q = np.clip(q, 0, last)
        step = np.asarray(self.step)
        low = q * step
        high = np.minimum(low + step, size)
        return 0.5 * (low + high)


def pick_bit_layout(dataset_size: Sequence[int]) -> PickBitLayout:
    """The pick-position layout for a volume of *dataset_size* voxels.

    Parameters
    ----------
    dataset_size : Sequence[int]
        Level-0 voxel counts in pygfx ``(x, y, z)`` order, each at least 1.

    Returns
    -------
    PickBitLayout
        The per-axis fields.  The rule:

        1. Each axis gets ``i = bit_length(n - 1)`` index bits.
        2. If they fit in 44 bits, the spare bits are shared as fraction
           bits: ``spare // 3`` each, and the ``spare % 3`` left over to the
           longest axes first (ties in ``x, y, z`` order).
        3. If they do not fit, fraction bits go negative one bit at a time,
           each taken from the widest field (ties: the longer axis, then
           ``x, y, z`` order), until the widths sum to 44.
        4. Every field is then capped at 24 bits; bits freed by the cap stay
           unused.
    """
    sizes = tuple(int(n) for n in dataset_size)
    if len(sizes) != 3:
        raise ValueError(f"dataset_size must have 3 entries, got {dataset_size!r}")
    if min(sizes) < 1:
        raise ValueError(f"dataset_size entries must be at least 1, got {sizes}")

    index = [(n - 1).bit_length() for n in sizes]
    spare = PICK_POSITION_BITS - sum(index)
    frac = [0, 0, 0]
    if spare >= 0:
        frac = [spare // 3] * 3
        longest_first = sorted(range(3), key=lambda a: (-sizes[a], a))
        for a in longest_first[: spare % 3]:
            frac[a] += 1
    else:
        for _ in range(-spare):
            a = max(range(3), key=lambda a: (index[a] + frac[a], sizes[a], -a))
            frac[a] -= 1
    frac = [min(frac[a], MAX_FIELD_BITS - index[a]) for a in range(3)]
    widths = tuple(index[a] + frac[a] for a in range(3))
    return PickBitLayout(
        dataset_size=sizes,
        index_bits=tuple(index),
        frac_bits=tuple(frac),
        widths=widths,
    )


#: ``(visual model id, dataset size)`` pairs already warned about.
_WARNED: set[tuple[object, tuple[int, int, int]]] = set()

_AXES = ("x", "y", "z")


def warn_if_pick_over_budget(
    dataset_size: Sequence[float], visual_model_id: object, visual_name: str
) -> PickBitLayout:
    """The pick layout of *dataset_size*, warning once if it is over budget.

    An over-budget volume cannot name every voxel in a pick: on some axes a
    step spans several voxels (`PickBitLayout.over_budget`).  The
    warning (a ``UserWarning``) names the visual, the axes and the step, and
    is given once per visual and 3D shape: a multichannel image has one slot
    per channel, all with the same id, and a 3D view that returns to a shape
    already warned about stays quiet.

    Parameters
    ----------
    dataset_size : sequence of float
        The level-0 voxel counts over the displayed axes, pygfx ``(x, y, z)``.
    visual_model_id : object
        The id of the visual model the volume belongs to.
    visual_name : str
        The visual's name, for the message.

    Returns
    -------
    PickBitLayout
        ``pick_bit_layout(dataset_size)``.
    """
    layout = pick_bit_layout(dataset_size)
    if not layout.over_budget:
        return layout
    key = (visual_model_id, layout.dataset_size)
    if key in _WARNED:
        return layout
    _WARNED.add(key)
    warnings.warn(over_budget_message(layout, visual_name), UserWarning, stacklevel=2)
    return layout


def _voxels(count: int) -> str:
    return f"{count} voxel" if count == 1 else f"{count} voxels"


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def over_budget_message(layout: PickBitLayout, visual_name: str) -> str:
    """The warning text for an over-budget *layout* of visual *visual_name*."""
    coarse = [a for a in range(3) if layout.frac_bits[a] < 0]
    steps = _join([f"{_voxels(int(layout.step[a]))} along {_AXES[a]}" for a in coarse])
    off = _join(
        [f"{_voxels(int(layout.step[a]) - 1)} along {_AXES[a]}" for a in coarse]
    )
    needed = sum(layout.index_bits)
    if needed > PICK_POSITION_BITS:
        shape = " x ".join(str(n) for n in layout.dataset_size)
        why = (
            f"its {shape} voxels (x, y, z) need {needed} bits to address and "
            f"the pick buffer has {PICK_POSITION_BITS}"
        )
    else:
        a = coarse[0]
        why = (
            f"its {layout.dataset_size[a]} voxels along {_AXES[a]} need "
            f"{layout.index_bits[a]} bits to address and one axis can have at "
            f"most {MAX_FIELD_BITS}"
        )
    return (
        f"3D picks on visual {visual_name!r} resolve to {steps}: {why}.  A "
        f"pick's data_coordinate and value can be off by up to {off}."
    )
