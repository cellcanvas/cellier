"""Tests for the dims slider value models and their shared helpers."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from cellier._rounding import round_half_up_clamped
from cellier.gui._axis_values import (
    TICK_WARNING_LIMIT,
    ContinuousAxisValues,
    DiscreteAxisValues,
    coerce_axis_values,
    nearest_value_index,
)
from cellier.gui._dims import initial_slice_indices


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": 1.0, "max": 0.0},
        {"min": 0.0, "max": float("inf")},
        {"min": float("nan"), "max": 1.0},
    ],
)
def test_continuous_rejects_bad_bounds(kwargs):
    with pytest.raises(ValidationError):
        ContinuousAxisValues(**kwargs)


def test_continuous_allows_a_single_point():
    spec = ContinuousAxisValues(min=2.0, max=2.0)
    assert (spec.min, spec.max) == (2.0, 2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": ()},
        {"values": (0.0, 0.0)},
        {"values": (1.0, 0.0)},
        {"values": (0.0, float("inf"))},
        {"values": (0.0, 1.0), "labels": ("only one",)},
    ],
)
def test_discrete_rejects_bad_values(kwargs):
    with pytest.raises(ValidationError):
        DiscreteAxisValues(**kwargs)


def test_discrete_min_and_max_are_the_end_values():
    spec = DiscreteAxisValues(values=(0.5, 2.0, 9.0))
    assert (spec.min, spec.max) == (0.5, 9.0)


def test_models_compare_and_hash_by_value():
    first = DiscreteAxisValues(values=(0.0, 1.0), labels=("a", "b"))
    second = DiscreteAxisValues(values=(0.0, 1.0), labels=("a", "b"))
    assert first == second
    assert hash(first) == hash(second)
    assert first != DiscreteAxisValues(values=(0.0, 1.0))


def test_coerce_normalises_keys_and_accepts_serialised_models():
    continuous = ContinuousAxisValues(min=0.0, max=4.0)
    discrete = DiscreteAxisValues(values=(0.0, 1.0), labels=("a", "b"))

    coerced = coerce_axis_values({"0": continuous, 1: discrete.model_dump(mode="json")})

    assert coerced == {0: continuous, 1: discrete}


@pytest.mark.parametrize("entry", [(0.0, 1.0), [0.0, 1.0]])
def test_coerce_rejects_bare_pairs_with_a_pointer_to_the_models(entry):
    with pytest.raises(TypeError, match="ContinuousAxisValues"):
        coerce_axis_values({0: entry})


def test_coerce_rejects_other_types():
    with pytest.raises(TypeError, match="must be a ContinuousAxisValues"):
        coerce_axis_values({0: 3.0})


@pytest.mark.parametrize(
    ("values", "position", "expected"),
    [
        ((0.0, 1.0), -1.0, 0),
        ((0.0, 1.0), 0.49, 0),
        ((0.0, 1.0), 0.5, 1),
        ((0.0, 1.0), 7.0, 1),
        ((0.0, 10.0, 40.0), 4.9, 0),
        ((0.0, 10.0, 40.0), 5.0, 1),
        ((0.0, 10.0, 40.0), 24.9, 1),
        ((0.0, 10.0, 40.0), 25.0, 2),
        ((3.0,), -100.0, 0),
    ],
)
def test_nearest_value_index(values, position, expected):
    assert nearest_value_index(values, position) == expected


def test_nearest_value_index_matches_the_renderer_on_integer_values():
    """A discrete slider must show the sample the renderer selects."""
    values = (0.0, 1.0, 2.0, 3.0, 4.0)
    for position in np.linspace(-2.0, 6.0, 81):
        assert nearest_value_index(values, position) == round_half_up_clamped(
            position, len(values)
        )


def test_initial_slice_indices_seeds_by_kind():
    axis_values = {
        0: DiscreteAxisValues(values=(0.0, 1.0)),
        1: ContinuousAxisValues(min=-0.5, max=15.5),
    }

    seeded = initial_slice_indices(SimpleNamespace(slice_indices={}), axis_values)

    # Discrete starts on its first value, not on the 0.5 tie between channels.
    assert seeded == {0: 0.0, 1: 7.5}


@pytest.mark.parametrize(("known", "expected"), [(0.2, 0.0), (0.5, 1.0), (1.0, 1.0)])
def test_initial_slice_indices_moves_a_discrete_scene_value_to_a_listed_one(
    known, expected
):
    axis_values = {0: DiscreteAxisValues(values=(0.0, 1.0))}
    selection = SimpleNamespace(slice_indices={0: known})

    assert initial_slice_indices(selection, axis_values) == {0: expected}


# ---------------------------------------------------------------------------
# draw_ticks
# ---------------------------------------------------------------------------


def test_draw_ticks_is_off_by_default():
    assert DiscreteAxisValues(values=(0.0, 1.0)).draw_ticks is False


def test_draw_ticks_survives_a_serialisation_round_trip():
    spec = DiscreteAxisValues(values=(0.0, 1.0), draw_ticks=True)

    coerced = coerce_axis_values({0: spec.model_dump(mode="json")})

    assert coerced[0].draw_ticks is True


@pytest.mark.parametrize("n_values", [TICK_WARNING_LIMIT, TICK_WARNING_LIMIT - 1])
def test_draw_ticks_is_quiet_up_to_the_limit(n_values):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = DiscreteAxisValues(
            values=tuple(float(i) for i in range(n_values)), draw_ticks=True
        )

    assert spec.draw_ticks is True


def test_draw_ticks_warns_past_the_limit():
    n_values = TICK_WARNING_LIMIT + 1

    with pytest.warns(UserWarning, match=f"{n_values} values"):
        DiscreteAxisValues(
            values=tuple(float(i) for i in range(n_values)), draw_ticks=True
        )


def test_a_long_axis_without_ticks_is_quiet():
    # The default is what every derived axis gets, however long it is.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = DiscreteAxisValues(
            values=tuple(float(i) for i in range(10 * TICK_WARNING_LIMIT))
        )

    assert spec.draw_ticks is False
