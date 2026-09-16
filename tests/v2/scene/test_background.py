"""Tests for the BackgroundAppearance model and its relay onto Scene."""

from cellier.scene._background import (
    DEFAULT_BOTTOM_COLOR,
    DEFAULT_TOP_COLOR,
    BackgroundAppearance,
)
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.scene.scene import Scene


def _make_scene() -> Scene:
    return Scene(
        name="main",
        dims=DimsManager(
            world_coordinate_system=world_coordinate_system(
                spatial_axes("z", "y", "x"), name="world"
            ),
            selection=AxisAlignedSelection(
                displayed_axes=(1, 2), slice_indices={0: 0, 1: 0, 2: 0}
            ),
        ),
    )


def test_default_is_the_historical_gradient():
    """The default reproduces the gradient that used to be hard coded."""
    background = BackgroundAppearance()
    assert background.mode == "vertical_gradient"
    assert background.visible is True
    assert background.to_colors() == (DEFAULT_BOTTOM_COLOR, DEFAULT_TOP_COLOR)


def test_to_colors_uniform_returns_one_color():
    background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 1.0, 1.0))
    assert background.to_colors() == ((0.0, 0.0, 1.0, 1.0),)


def test_to_colors_ignores_the_inactive_mode_colors():
    """Switching modes leaves the other mode's colors set but unused."""
    background = BackgroundAppearance(
        mode="uniform",
        color=(1.0, 0.0, 0.0, 1.0),
        top_color=(0.0, 1.0, 0.0, 1.0),
    )
    assert background.to_colors() == ((1.0, 0.0, 0.0, 1.0),)

    background.mode = "vertical_gradient"
    assert background.to_colors() == (DEFAULT_BOTTOM_COLOR, (0.0, 1.0, 0.0, 1.0))


def test_scene_background_roundtrip(tmp_path):
    original = _make_scene()
    original.background.mode = "uniform"
    original.background.color = (0.25, 0.5, 0.75, 1.0)

    path = tmp_path / "scene.json"
    path.write_text(original.model_dump_json())
    deserialized = Scene.model_validate_json(path.read_text())

    assert deserialized.background.mode == "uniform"
    assert deserialized.background.color == (0.25, 0.5, 0.75, 1.0)
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_field_change_relays_to_scene():
    scene = _make_scene()
    seen = []
    scene.events.background.connect(seen.append)

    scene.background.top_color = (1.0, 0.0, 0.0, 1.0)

    assert len(seen) == 1
    assert seen[0] is scene.background


def test_replacing_the_model_moves_the_relay():
    """A new background model takes over; the old one goes quiet."""
    scene = _make_scene()
    original = scene.background
    seen = []
    scene.events.background.connect(seen.append)

    scene.background = BackgroundAppearance(mode="uniform")
    assert len(seen) == 1

    scene.background.color = (0.0, 1.0, 0.0, 1.0)
    assert len(seen) == 2

    original.color = (1.0, 1.0, 1.0, 1.0)
    assert len(seen) == 2
