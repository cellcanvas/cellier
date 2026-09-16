"""Tests for ``cellier.convenience.Viewer.set_displayed_dimensions``.

Covers the two validation guards and the 2D <-> 3D roundtrip.  Every axis
keeps its slice position whether or not it is displayed (D36), so an axis
cycling from displayed back to sliced needs no saving or restoring.
"""

from __future__ import annotations

import pytest

from cellier.convenience import Viewer
from cellier.scene.dims import spatial_axes


def test_rejects_wrong_arity():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    with pytest.raises(ValueError, match="2 or 3 entries"):
        viewer.set_displayed_dimensions(("x",))
    with pytest.raises(ValueError, match="2 or 3 entries"):
        viewer.set_displayed_dimensions(("t", "z", "y", "x"))


def test_rejects_unknown_axis_name():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    with pytest.raises(ValueError, match="Unknown axis names"):
        viewer.set_displayed_dimensions(("q", "x"))


def test_switch_2d_to_3d_sets_displayed_axes():
    viewer = Viewer(
        [("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")], dim="2d"
    )
    scene = viewer.scene
    assert tuple(scene.dims.selection.displayed_axes) == (2, 3)

    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert tuple(scene.dims.selection.displayed_axes) == (1, 2, 3)
    # z (axis 1) is now displayed and keeps its position.
    assert scene.dims.selection.slice_indices[1] == 0


def test_roundtrip_restores_saved_slice_position():
    viewer = Viewer(
        [("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")], dim="2d"
    )
    scene = viewer.scene

    # Seed a non-default slice position on z (axis 1) while it is sliced.
    viewer.controller.update_slice_indices(scene.id, {1: 5})
    assert scene.dims.selection.slice_indices[1] == 5

    # Expand to 3D: z becomes displayed and keeps its position.
    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert scene.dims.selection.slice_indices[1] == 5

    # Contract back to 2D: z is sliced again at the same position.
    viewer.set_displayed_dimensions(("y", "x"))
    assert tuple(scene.dims.selection.displayed_axes) == (2, 3)
    assert scene.dims.selection.slice_indices[1] == pytest.approx(5)


def test_unseen_axis_defaults_to_zero_on_contract():
    viewer = Viewer(
        [("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")], dim="3d"
    )
    scene = viewer.scene
    # Displaying z, y, x leaves t (axis 0) sliced.
    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert tuple(scene.dims.selection.displayed_axes) == (1, 2, 3)

    # Contract to y, x: z (axis 1) was never seeded, so it restores to 0.
    viewer.set_displayed_dimensions(("y", "x"))
    assert scene.dims.selection.slice_indices[1] == pytest.approx(0)
