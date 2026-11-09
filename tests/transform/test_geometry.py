"""Tests for cellier.transform._geometry."""

from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.transform import AxisAlignedBoundingBox, Plane


def box(**kwargs):
    defaults = {
        "coordinate_system": uuid4(),
        "min_coordinate": [0.0, 0.0, 0.0],
        "max_coordinate": [1.0, 2.0, 3.0],
    }
    return AxisAlignedBoundingBox(**{**defaults, **kwargs})


def plane(**kwargs):
    defaults = {
        "coordinate_system": uuid4(),
        "normal": [0.0, 0.0, 1.0],
        "offset": 2.0,
    }
    return Plane(**{**defaults, **kwargs})


# --- AxisAlignedBoundingBox ------------------------------------------


def test_bounds_are_coerced_from_a_list():
    result = box()
    assert isinstance(result.min_coordinate, np.ndarray)
    assert result.min_coordinate.dtype == np.float64
    assert result.ndim == 3


def test_infinite_bounds_are_first_class():
    result = box(min_coordinate=[-np.inf, 0.0, 0.0], max_coordinate=[np.inf, 2.0, 3.0])
    assert result.min_coordinate[0] == -np.inf
    assert result.max_coordinate[0] == np.inf


def test_nan_bounds_are_rejected():
    with pytest.raises(ValidationError, match="nan"):
        box(min_coordinate=[np.nan, 0.0, 0.0])


def test_min_above_max_is_rejected():
    with pytest.raises(ValidationError, match="must not exceed"):
        box(min_coordinate=[5.0, 0.0, 0.0])


def test_mismatched_rank_is_rejected():
    with pytest.raises(ValidationError, match="same shape"):
        box(min_coordinate=[0.0, 0.0])


def test_zero_rank_is_rejected():
    with pytest.raises(ValidationError):
        box(min_coordinate=[], max_coordinate=[])


def test_zero_thickness_box_is_allowed():
    result = box(min_coordinate=[1.0, 0.0, 0.0], max_coordinate=[1.0, 2.0, 3.0])
    assert result.min_coordinate[0] == result.max_coordinate[0] == 1.0


def test_box_is_frozen():
    with pytest.raises(ValidationError):
        box().min_coordinate = np.zeros(3)


def test_box_equality_and_hash():
    coordinate_system = uuid4()
    first = box(coordinate_system=coordinate_system)
    second = box(coordinate_system=coordinate_system)
    assert first == second
    assert hash(first) == hash(second)
    assert first != box()  # a different coordinate system


def test_box_equality_against_a_foreign_type_is_false_not_an_exception():
    """The default __eq__ on a bare-array model raises ValueError here."""
    assert (box() == "not a box") is False
    assert (box() != "not a box") is True


# --- Plane -----------------------------------------------------------


def test_plane_normal_need_not_be_unit_length():
    result = plane(normal=[0.0, 0.0, 5.0])
    assert np.linalg.norm(result.normal) == pytest.approx(5.0)


def test_plane_rejects_a_zero_normal():
    with pytest.raises(ValidationError, match="zero vector"):
        plane(normal=[0.0, 0.0, 0.0])


def test_plane_rejects_a_non_finite_normal():
    with pytest.raises(ValidationError, match="finite"):
        plane(normal=[np.inf, 0.0, 1.0])
    with pytest.raises(ValidationError, match="finite"):
        plane(normal=[np.nan, 0.0, 1.0])


def test_plane_rejects_a_non_finite_offset():
    with pytest.raises(ValidationError, match="finite"):
        plane(offset=np.inf)
    with pytest.raises(ValidationError, match="finite"):
        plane(offset=np.nan)


def test_plane_is_frozen():
    with pytest.raises(ValidationError):
        plane().offset = 3.0


def test_plane_equality_and_hash():
    coordinate_system = uuid4()
    first = plane(coordinate_system=coordinate_system)
    second = plane(coordinate_system=coordinate_system)
    assert first == second
    assert hash(first) == hash(second)
    assert (first == "not a plane") is False


def test_plane_carries_its_coordinate_system():
    coordinate_system = uuid4()
    assert plane(coordinate_system=coordinate_system).coordinate_system == (
        coordinate_system
    )
