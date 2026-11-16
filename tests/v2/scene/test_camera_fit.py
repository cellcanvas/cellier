"""Tests for the camera fit sphere and the far plane it needs."""

import numpy as np
import pygfx as gfx
import pytest
from pygfx.cameras._perspective import fov_distance_factor as pygfx_factor

from cellier.scene._camera_fit import (
    FAR_MARGIN,
    fit_distance,
    fit_sphere,
    fitted_far,
    fov_distance_factor,
    grow_far,
)

#: A ``(100, 200, 400)`` volume, cellier displayed order ``(z, y, x)``; not a
#: cube, so a missed reversal to pygfx ``(x, y, z)`` changes the answers.
BOX = ((0.0, 0.0, 0.0), (100.0, 200.0, 400.0))
DEFAULT_VIEW = (-1.0, -1.0, -1.0)


def test_the_sphere_is_the_box_centre_and_half_diagonal_in_pygfx_order():
    centre, radius = fit_sphere(BOX)
    assert centre == (200.0, 100.0, 50.0)
    assert radius == pytest.approx(np.sqrt(100**2 + 200**2 + 400**2) / 2)


def test_a_2d_box_is_fitted_at_z_0():
    centre, radius = fit_sphere(((0.0, 0.0), (10.0, 20.0)))
    assert centre == (10.0, 5.0, 0.0)
    assert radius == pytest.approx(np.sqrt(10**2 + 20**2) / 2)


@pytest.mark.parametrize(
    ("fov", "margin", "expected"),
    [
        # A (100, 200, 400) box seen along (-1, -1, -1).
        (70.0, 1.0, 529.3),
        (70.0, 2.0, 856.5),
        (70.0, 4.0, 1511.0),
        (10.0, 2.0, 5440.0),
    ],
)
def test_the_worked_example(fov, margin, expected):
    assert fitted_far(BOX, fov, DEFAULT_VIEW, margin=margin) == pytest.approx(
        expected, abs=0.1
    )


def test_the_default_margin_is_4():
    assert FAR_MARGIN == 4.0
    assert fitted_far(BOX, 70.0, DEFAULT_VIEW) == pytest.approx(
        fitted_far(BOX, 70.0, DEFAULT_VIEW, margin=4.0)
    )


@pytest.mark.parametrize(
    ("view", "reach"),
    [
        # pygfx z is the first cellier axis (extent 100), x the last (400).
        ((0.0, 0.0, -1.0), 50.0),
        ((0.0, 0.0, 1.0), 50.0),
        ((1.0, 0.0, 0.0), 200.0),
        ((0.0, -3.0, 0.0), 100.0),
    ],
)
def test_the_reach_is_along_the_view_direction_in_pygfx_order(view, reach):
    _, radius = fit_sphere(BOX)
    far = fitted_far(BOX, 70.0, view, margin=1.0)
    assert far - fit_distance(radius, 70.0) == pytest.approx(reach)


def test_an_orthographic_camera_needs_only_the_reach():
    # pygfx puts a fov 0 camera at the centre of what it fits.
    assert fit_distance(229.1, 0.0) == 0.0
    assert fitted_far(BOX, 0.0, (0.0, 0.0, -1.0)) == pytest.approx(50.0)


def test_a_zero_radius_is_fitted_as_1():
    point = ((5.0, 5.0, 5.0), (5.0, 5.0, 5.0))
    assert fit_sphere(point) == ((5.0, 5.0, 5.0), 0.0)
    assert fit_distance(0.0, 70.0) == pytest.approx(fov_distance_factor(70.0) * 2)
    assert fitted_far(point, 70.0, DEFAULT_VIEW, margin=1.0) == pytest.approx(
        fov_distance_factor(70.0) * 2
    )


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3, 1e6])
def test_the_far_plane_scales_with_the_box(scale):
    low, high = (np.asarray(side) * scale for side in BOX)
    far = fitted_far((low, high), 70.0, DEFAULT_VIEW)
    assert np.isfinite(far)
    assert far == pytest.approx(fitted_far(BOX, 70.0, DEFAULT_VIEW) * scale)


def test_the_far_plane_needs_a_3d_box():
    with pytest.raises(ValueError, match="3D box"):
        fitted_far(((0.0, 0.0), (1.0, 1.0)), 70.0, DEFAULT_VIEW)


@pytest.mark.parametrize("view", [(0.0, 0.0, 0.0), (np.nan, 0.0, 1.0)])
def test_a_degenerate_view_direction_is_refused(view):
    with pytest.raises(ValueError, match="view_direction"):
        fitted_far(BOX, 70.0, view)


def test_a_box_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="finite"):
        fit_sphere(((0.0, np.nan, 0.0), (1.0, 1.0, 1.0)))


def test_grow_far_only_raises_far_and_keeps_near():
    assert grow_far((1.0, 8000.0), 500.0) == (1.0, 8000.0)
    assert grow_far((1.0, 8000.0), 40000.0) == (1.0, 40000.0)
    assert grow_far((0.25, 10.0), 10.0) == (0.25, 10.0)


def test_grow_far_leaves_an_automatic_range_alone():
    assert grow_far(None, 40000.0) is None


# -- against pygfx: the rule predicts where its fit puts the camera ------------


@pytest.mark.parametrize("fov", [0.0, 1.0, 10.0, 45.0, 70.0, 120.0, 179.0])
def test_the_distance_factor_is_pygfx_s(fov):
    assert fov_distance_factor(fov) == pytest.approx(pygfx_factor(fov))


@pytest.mark.parametrize("fov", [10.0, 70.0])
@pytest.mark.parametrize("aspect", [16 / 9, 1.0, 0.3])
@pytest.mark.parametrize("view", [DEFAULT_VIEW, (0.2, -1.0, 0.5)])
def test_pygfx_fits_where_the_rule_says(fov, aspect, view):
    centre, radius = fit_sphere(BOX)
    camera = gfx.PerspectiveCamera(fov, aspect, depth_range=(1.0, 8000.0))
    camera.show_object((*centre, radius), view_dir=view, up=(0.0, 1.0, 0.0))

    position = np.asarray(camera.world.position)
    direction = np.asarray(view) / np.linalg.norm(view)
    assert np.linalg.norm(np.asarray(centre) - position) == pytest.approx(
        fit_distance(radius, fov)
    )
    # With margin 1 the far plane is the depth of the farthest corner.
    low, high = np.asarray(BOX[0])[::-1], np.asarray(BOX[1])[::-1]
    corners = np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    deepest = float(((corners - position) @ direction).max())
    assert fitted_far(BOX, fov, view, margin=1.0) == pytest.approx(deepest)
