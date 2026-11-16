"""The ``np.ndarray`` field contract shared by every model here.

Three things go wrong with the obvious spelling of a numpy field on a
pydantic model, none of them visible from reading the model:

1. A ``field_serializer`` alone does not round-trip.  It emits a
   ``list``, and validation rejects that on the way back in, so every
   array field also needs a coercing ``field_validator(mode="before")``.
2. Infinity does not survive JSON.  ``model_dump_json`` writes ``null``
   and reads it back as ``nan``, and ``nan`` is worse than an error:
   every comparison against it is ``False``, so an unbounded axis
   silently becomes one that excludes everything.  Bounds are therefore
   written as the sentinel strings ``"Infinity"`` / ``"-Infinity"``,
   which are unambiguous, symmetric, and valid under strict JSON.
3. A model holding a bare ``np.ndarray`` gets a default ``__eq__`` that
   **raises** ``ValueError``, because an elementwise comparison has no
   truth value.  A raising ``__eq__`` degrades an owning class to
   identity comparison process-wide, so every such model defines
   ``__eq__`` and ``__hash__`` explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

POSITIVE_INFINITY = "Infinity"
NEGATIVE_INFINITY = "-Infinity"


def coerce_float_array(value: Any) -> np.ndarray:
    """Coerce a field value to a 1-D float array.

    Accepts an array, a sequence of numbers, or a sequence mixing numbers
    with the ``"Infinity"`` / ``"-Infinity"`` sentinels that
    `serialize_bounds` emits.

    Parameters
    ----------
    value : Any
        The incoming field value.

    Returns
    -------
    np.ndarray
        A 1-D ``float64`` array.
    """
    if isinstance(value, str):
        # a bare string would otherwise be read as a character sequence
        raise ValueError(f"Expected a sequence of numbers, got the string {value!r}.")
    return np.asarray(value, dtype=float)


def reject_non_finite(array: np.ndarray, field: str) -> np.ndarray:
    """Reject ``inf`` and ``nan`` in a field where neither has a meaning.

    Parameters
    ----------
    array : np.ndarray
        The coerced array.
    field : str
        The field name, used in the error message.

    Returns
    -------
    np.ndarray
        The array, unchanged.

    Raises
    ------
    ValueError
        If any entry is not finite.
    """
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{field} must be finite; got {array.tolist()}.  Infinity is legal "
            f"only in an AxisAlignedBoundingBox bound, and nan is legal nowhere."
        )
    return array


def reject_nan(array: np.ndarray, field: str) -> np.ndarray:
    """Reject ``nan`` while allowing ``+-inf``.

    Parameters
    ----------
    array : np.ndarray
        The coerced array.
    field : str
        The field name, used in the error message.

    Returns
    -------
    np.ndarray
        The array, unchanged.

    Raises
    ------
    ValueError
        If any entry is ``nan``.
    """
    if np.any(np.isnan(array)):
        raise ValueError(f"{field} must not contain nan; got {array.tolist()}.")
    return array


def serialize_array(array: np.ndarray) -> list[float]:
    """Serialize a finite array to a list of floats."""
    return [float(value) for value in array]


def serialize_bounds(array: np.ndarray) -> list[float | str]:
    """Serialize an array that may hold ``+-inf``, using the sentinels.

    Parameters
    ----------
    array : np.ndarray
        The array to serialize.

    Returns
    -------
    list[float | str]
        Finite entries as floats, infinities as ``"Infinity"`` or
        ``"-Infinity"``.  The result is valid under strict JSON, which
        cannot represent infinity at all.
    """
    out: list[float | str] = []
    for value in array:
        if np.isposinf(value):
            out.append(POSITIVE_INFINITY)
        elif np.isneginf(value):
            out.append(NEGATIVE_INFINITY)
        else:
            out.append(float(value))
    return out
