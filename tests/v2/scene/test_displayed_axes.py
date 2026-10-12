"""``displayed_axes`` is required and resolved in one place.

``plans/displayed_axes_required.md`` section 5.1: ``Viewer`` and
``CellierController.add_scene`` take the displayed axes as names or axis ids,
and ``Viewer.set_displayed_dimensions`` reads them with the same resolver.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.controller import CellierController
from cellier.convenience import Viewer
from cellier.scene.dims import (
    resolve_displayed_axes,
    spatial_axes,
    world_coordinate_system,
)
from cellier.transform import Axis

_TZYX = world_coordinate_system([("t", "time"), *spatial_axes("z", "y", "x")])


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (("y", "x"), (2, 3)),
        (("x", "y"), (3, 2)),
        (("z", "y", "x"), (1, 2, 3)),
        # No axis-type check: time can be displayed (D8).
        (("t", "x"), (0, 3)),
        (["t", "z", "y"], (0, 1, 2)),
    ],
)
def test_names_resolve_in_the_order_given(given, expected):
    assert resolve_displayed_axes(_TZYX, given) == expected


def test_ids_resolve_and_mix_with_names():
    t, _z, _y, x = _TZYX.axes

    assert resolve_displayed_axes(_TZYX, (x.id, t.id)) == (3, 0)
    assert resolve_displayed_axes(_TZYX, ("y", x.id)) == (2, 3)


def test_a_shared_name_is_an_error_that_asks_for_the_id():
    first = Axis(name="x", axis_type="space")
    second = Axis(name="x", axis_type="space")
    world = world_coordinate_system([Axis(name="y", axis_type="space"), first, second])

    with pytest.raises(ValueError, match="shared by 2 axes") as error:
        resolve_displayed_axes(world, ("y", "x"))
    assert "UUID" in str(error.value)
    assert str(second.id) in str(error.value)
    # The id is never ambiguous.
    assert resolve_displayed_axes(world, ("y", second.id)) == (0, 2)


@pytest.mark.parametrize(
    ("given", "error", "match"),
    [
        (("x",), ValueError, "2 or 3 entries"),
        (("t", "z", "y", "x"), ValueError, "2 or 3 entries"),
        ((), ValueError, "2 or 3 entries"),
        (("y", "q"), ValueError, "Unknown axis name 'q'"),
        (("y", "y"), ValueError, "more than once"),
        ((2, 3), TypeError, "Axis indices are not accepted"),
        (("y", None), TypeError, "axis name"),
        ("yx", TypeError, "single value"),
    ],
)
def test_bad_displayed_axes_are_rejected(given, error, match):
    with pytest.raises(error, match=match):
        resolve_displayed_axes(_TZYX, given)


def test_an_unknown_id_is_rejected():
    with pytest.raises(ValueError, match="No axis has the id"):
        resolve_displayed_axes(_TZYX, ("y", uuid4()))


def test_a_name_and_the_id_of_the_same_axis_are_the_same_axis():
    with pytest.raises(ValueError, match="more than once"):
        resolve_displayed_axes(_TZYX, ("x", _TZYX.axes[3].id))


@pytest.mark.parametrize(
    ("given", "modes", "mode"),
    [(("y", "x"), {"3d"}, "2d"), (("z", "y", "x"), {"2d"}, "3d")],
)
def test_the_rank_must_be_a_mode_the_scene_renders(given, modes, mode):
    with pytest.raises(ValueError, match=f"in {mode}, but render_modes"):
        resolve_displayed_axes(_TZYX, given, modes)

    assert resolve_displayed_axes(_TZYX, given, {"2d", "3d"})


# ---------------------------------------------------------------------------
# The constructors
# ---------------------------------------------------------------------------


@pytest.fixture
def controller():
    ctrl = CellierController(gui="offscreen")
    yield ctrl
    ctrl.close()


def test_add_scene_requires_displayed_axes(controller):
    with pytest.raises(TypeError, match="displayed_axes"):
        controller.add_scene()


def test_add_scene_no_longer_takes_dim(controller):
    with pytest.raises(TypeError, match="dim"):
        controller.add_scene(displayed_axes=("y", "x"), dim="2d")


@pytest.mark.parametrize(
    ("given", "expected"),
    [(("x", "y"), (0, 1)), (("y", "z"), (1, 2)), (("z", "y", "x"), (2, 1, 0))],
)
def test_add_scene_displays_the_axes_it_is_given(controller, given, expected):
    scene = controller.add_scene(
        displayed_axes=given, coordinate_system=spatial_axes("x", "y", "z")
    )

    assert tuple(scene.dims.selection.displayed_axes) == expected
    # Every axis keeps a position, displayed ones included.
    assert set(scene.dims.selection.slice_indices) == {0, 1, 2}


def test_add_scene_checks_the_rank_against_render_modes(controller):
    with pytest.raises(ValueError, match="render_modes"):
        controller.add_scene(displayed_axes=("y", "x"), render_modes={"3d"})


def test_the_canvas_starts_in_the_mode_of_the_displayed_axes(controller):
    flat = controller.add_scene(displayed_axes=("y", "x"), name="flat")
    volume = controller.add_scene(displayed_axes=("z", "y", "x"), name="volume")
    controller.add_canvas(flat.id)
    controller.add_canvas(volume.id)

    def canvas_dim(scene):
        canvas_id = controller.get_canvas_ids(scene.id)[0]
        return controller._render_manager.canvas_dim(canvas_id)

    assert canvas_dim(flat) == "2d"
    assert canvas_dim(volume) == "3d"


def test_viewer_requires_displayed_axes():
    with pytest.raises(TypeError, match="displayed_axes"):
        Viewer(spatial_axes("z", "y", "x"), gui="offscreen")


def test_viewer_no_longer_takes_dim():
    with pytest.raises(TypeError, match="dim"):
        Viewer(
            spatial_axes("z", "y", "x"),
            displayed_axes=("y", "x"),
            dim="2d",
            gui="offscreen",
        )


def test_viewer_displays_the_axes_it_is_given():
    viewer = Viewer(
        spatial_axes("x", "y", "z"), displayed_axes=("x", "y"), gui="offscreen"
    )

    assert tuple(viewer.scene.dims.selection.displayed_axes) == (0, 1)


def test_viewer_takes_axis_ids():
    axes = spatial_axes("x", "y", "z")
    viewer = Viewer(axes, displayed_axes=(axes[2].id, axes[0].id), gui="offscreen")

    assert tuple(viewer.scene.dims.selection.displayed_axes) == (2, 0)


def test_set_displayed_dimensions_uses_the_same_resolver():
    axes = spatial_axes("x", "y", "z")
    viewer = Viewer(axes, displayed_axes=("x", "y"), gui="offscreen")

    viewer.set_displayed_dimensions((axes[1].id, "z"))
    assert tuple(viewer.scene.dims.selection.displayed_axes) == (1, 2)

    with pytest.raises(ValueError, match="Unknown axis name"):
        viewer.set_displayed_dimensions(("x", "q"))
    with pytest.raises(ValueError, match="2 or 3 entries"):
        viewer.set_displayed_dimensions(("x",))
    # A refused change leaves the scene where it was.
    assert tuple(viewer.scene.dims.selection.displayed_axes) == (1, 2)
