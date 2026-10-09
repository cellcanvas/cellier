"""``PlaneOutline`` and the ``outline`` field of the two kinds of plane.

Plane outline design (``plans/plane_outline_design.md``), section 4.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from cellier.transform import Axis, CoordinateSystem
from cellier.visuals import ClippingPlane, PlaneOutline, RenderPlane
from cellier.visuals._clipping import validate_clipping_planes
from cellier.visuals._plane_outline import NO_OUTLINE, without_outlines
from cellier.visuals._render_plane import validate_render_planes

WORLD = CoordinateSystem(
    name="world", axes=tuple(Axis(name=n, axis_type="space") for n in "zyx")
)
RED = PlaneOutline(enabled=True, color=(1.0, 0.0, 0.0, 1.0), width=3.0)


def _render_plane(**kwargs) -> RenderPlane:
    return RenderPlane.from_point_normal(WORLD, (4, 0, 0), (1, 0, 0), **kwargs)


def _clipping_plane(**kwargs) -> ClippingPlane:
    return ClippingPlane.from_point_normal(WORLD, (4, 0, 0), (1, 0, 0), **kwargs)


# -- the outline ----------------------------------------------------------------


def test_an_outline_is_off_white_and_two_pixels_wide_by_default():
    outline = PlaneOutline()
    assert outline.enabled is False
    assert outline.color == (1.0, 1.0, 1.0, 1.0)
    assert outline.width == 2.0
    assert outline == NO_OUTLINE


def test_an_outline_is_frozen():
    with pytest.raises(ValidationError):
        RED.enabled = False


@pytest.mark.parametrize("width", [0.0, -1.0, float("inf"), float("nan")])
def test_a_width_is_a_positive_finite_number(width):
    with pytest.raises(ValidationError):
        PlaneOutline(width=width)


@pytest.mark.parametrize(
    "color",
    [
        (1.0, 0.0, 0.0),
        (1.5, 0.0, 0.0, 1.0),
        (0.0, -0.1, 0.0, 1.0),
        (0, 0, 0, float("nan")),
    ],
)
def test_a_colour_is_four_numbers_between_zero_and_one(color):
    with pytest.raises(ValidationError):
        PlaneOutline(color=color)


# -- on the planes ----------------------------------------------------------------


@pytest.mark.parametrize("make", [_render_plane, _clipping_plane])
def test_a_plane_has_no_outline_until_given_one(make):
    assert make().outline == NO_OUTLINE
    assert make(outline=RED).outline == RED


def test_from_rotation_takes_an_outline():
    plane = RenderPlane.from_rotation(
        WORLD, (4, 0, 0), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], outline=RED
    )
    assert plane.outline == RED


@pytest.mark.parametrize("make", [_render_plane, _clipping_plane])
def test_an_outline_is_changed_with_a_copy_that_keeps_the_id(make):
    plane = make()
    outlined = plane.model_copy(update={"outline": RED})
    assert outlined.id == plane.id
    assert outlined != plane
    assert outlined.model_copy(update={"outline": NO_OUTLINE}) == plane


@pytest.mark.parametrize(
    ("make", "validate"),
    [
        (_render_plane, validate_render_planes),
        (_clipping_plane, validate_clipping_planes),
    ],
)
def test_an_outline_survives_a_json_round_trip(make, validate):
    plane = make(outline=RED)
    loaded = type(plane).model_validate_json(plane.model_dump_json())
    assert loaded.outline == RED
    assert loaded.id == plane.id
    # And as an item of a visual's tuple, from plain values.
    assert validate([plane.model_dump(mode="json")])[0].outline == RED


@pytest.mark.parametrize("make", [_render_plane, _clipping_plane])
def test_a_plane_saved_before_outlines_loads_with_none(make):
    """A scene file written before the field has no ``outline`` key."""
    saved = make(outline=RED).model_dump(mode="json")
    del saved["outline"]
    assert type(make()).model_validate(saved).outline == NO_OUTLINE


# -- setting the outlines aside ---------------------------------------------------


@pytest.mark.parametrize("make", [_render_plane, _clipping_plane])
def test_without_outlines_tells_a_restyle_from_a_move(make):
    plane = make()
    restyled = plane.model_copy(update={"outline": RED})
    assert without_outlines((plane,)) == without_outlines((restyled,))
    assert without_outlines((restyled,))[0].id == plane.id

    disabled = restyled.model_copy(update={"enabled": False})
    assert without_outlines((plane,)) != without_outlines((disabled,))
    other = make().model_copy(update={"id": uuid4()})
    assert without_outlines((plane,)) != without_outlines((other,))
    assert without_outlines((plane,)) != without_outlines((plane, other))
