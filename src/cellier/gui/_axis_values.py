"""The values a dims slider can take on one world axis.

A dims panel draws one slider per sliced axis, and what that slider looks
like depends on the axis: a spatial axis can be sliced at any world position,
while a channel axis only has a handful of valid values.  The two models here
say which, per axis, and every dims front end reads the same mapping.

Nothing here infers which kind an axis is.  The helpers that derive the
values from the loaded data (:func:`cellier.convenience.axis_values_from_viewer`)
decide from the data's axis sampling: a channel or time axis that every
store samples discretely gets a :class:`DiscreteAxisValues`, and everything
else a :class:`ContinuousAxisValues`.  A caller can replace any entry.

Toolkit-free on purpose: both the Qt and the anywidget dims panels import it.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

__all__ = [
    "AxisValues",
    "ContinuousAxisValues",
    "DiscreteAxisValues",
    "coerce_axis_values",
    "nearest_value_index",
]


class ContinuousAxisValues(BaseModel):
    """An axis that can be sliced at any world position in ``[min, max]``.

    Parameters
    ----------
    min : float
        Lowest world position the slider reaches.
    max : float
        Highest world position the slider reaches.  Must be ``>= min``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["continuous"] = "continuous"
    min: float
    max: float

    @model_validator(mode="after")
    def _check_bounds(self) -> ContinuousAxisValues:
        if not (math.isfinite(self.min) and math.isfinite(self.max)):
            raise ValueError(
                f"min and max must be finite; got min={self.min}, max={self.max}."
            )
        if self.min > self.max:
            raise ValueError(
                f"min must not exceed max; got min={self.min}, max={self.max}."
            )
        return self


class DiscreteAxisValues(BaseModel):
    """An axis that can only be sliced at the listed world positions.

    The slider steps through ``values`` by position, so irregular spacing is
    fine.  A position set from elsewhere that falls between two values is
    shown as the nearest one (see :func:`nearest_value_index`); it is not
    written back, because the renderer resolves it with the same rule.

    Parameters
    ----------
    values : tuple[float, ...]
        World positions, strictly increasing.  A tuple rather than an array
        so that equality stays total -- an ndarray field makes ``__eq__``
        return an array, which breaks every containing model's comparison.
    labels : tuple[str, ...] or None
        Readout text for each value, e.g. channel names.  ``None`` shows the
        value itself.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["discrete"] = "discrete"
    values: tuple[float, ...]
    labels: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _check_values(self) -> DiscreteAxisValues:
        if not self.values:
            raise ValueError("values must not be empty.")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError(f"values must be finite; got {self.values}.")
        if any(b <= a for a, b in zip(self.values, self.values[1:])):
            raise ValueError(f"values must be strictly increasing; got {self.values}.")
        if self.labels is not None and len(self.labels) != len(self.values):
            raise ValueError(
                f"labels must have one entry per value; got {len(self.labels)} "
                f"labels for {len(self.values)} values."
            )
        return self

    @property
    def min(self) -> float:
        """The first value."""
        return self.values[0]

    @property
    def max(self) -> float:
        """The last value."""
        return self.values[-1]


#: One axis's slider values, discriminated by ``kind``.
AxisValues = Annotated[
    Union[ContinuousAxisValues, DiscreteAxisValues], Field(discriminator="kind")
]

_AXIS_VALUES_ADAPTER: TypeAdapter = TypeAdapter(AxisValues)


def coerce_axis_values(
    axis_values: Mapping[int, ContinuousAxisValues | DiscreteAxisValues],
) -> dict[int, ContinuousAxisValues | DiscreteAxisValues]:
    """Validate an axis-values mapping and normalise its keys to ``int``.

    Entries must already be :class:`ContinuousAxisValues` or
    :class:`DiscreteAxisValues`.  A dict in their serialised form (with a
    ``kind`` key) is also accepted, which is what lets a mapping round-trip
    through ``model_dump``.  A bare ``(min, max)`` tuple is rejected with a
    message naming the replacement.

    Parameters
    ----------
    axis_values : Mapping[int, AxisValues]
        Axis index to that axis's slider values.

    Returns
    -------
    dict[int, AxisValues]
        A new dict with ``int`` keys.

    Raises
    ------
    TypeError
        If an entry is a tuple, list or other non-model value.
    """
    coerced: dict[int, ContinuousAxisValues | DiscreteAxisValues] = {}
    for axis, entry in axis_values.items():
        if isinstance(entry, (ContinuousAxisValues, DiscreteAxisValues)):
            coerced[int(axis)] = entry
        elif isinstance(entry, Mapping):
            coerced[int(axis)] = _AXIS_VALUES_ADAPTER.validate_python(entry)
        elif isinstance(entry, Sequence) and not isinstance(entry, str):
            raise TypeError(
                f"axis_values[{axis!r}] is a {type(entry).__name__}; bare "
                f"(min, max) pairs are no longer accepted.  Use "
                f"ContinuousAxisValues(min=..., max=...) or "
                f"DiscreteAxisValues(values=...)."
            )
        else:
            raise TypeError(
                f"axis_values[{axis!r}] must be a ContinuousAxisValues or "
                f"DiscreteAxisValues; got {type(entry).__name__}."
            )
    return coerced


def nearest_value_index(values: Sequence[float], position: float) -> int:
    """Return the index of the value in *values* nearest to *position*.

    A position exactly halfway between two values goes to the higher one,
    and positions beyond either end clamp to it.  For evenly spaced integer
    values that is ``round_half_up_clamped`` from :mod:`cellier._rounding`,
    so a discrete slider shows the same sample the renderer selects.

    Parameters
    ----------
    values : Sequence[float]
        Strictly increasing, non-empty.
    position : float
        World position to resolve.

    Returns
    -------
    int
        An index in ``[0, len(values) - 1]``.
    """
    upper = bisect.bisect_left(values, position)
    if upper <= 0:
        return 0
    if upper >= len(values):
        return len(values) - 1
    lower = upper - 1
    midpoint = (values[lower] + values[upper]) / 2.0
    return upper if position >= midpoint else lower
