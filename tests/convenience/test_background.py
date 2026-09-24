"""Tests for the background API on the convenience viewers."""

import numpy as np

from cellier.convenience import OrthoViewer, Viewer
from cellier.scene import BackgroundAppearance
from cellier.scene.dims import spatial_axes


def _top_color(viewer_controller, scene_id) -> np.ndarray:
    material = viewer_controller._render_manager._scenes[scene_id]._background_material
    return np.asarray(material.color_top_left)


_BLACK = (0.0, 0.0, 0.0, 1.0)


def test_viewer_background_defaults_to_uniform_black():
    viewer = Viewer(spatial_axes("y", "x"), dim="2d")

    assert viewer.background.mode == "uniform"
    assert viewer.background.color == _BLACK
    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id), _BLACK, atol=1e-6
    )


def test_ortho_panels_default_to_uniform_black():
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))

    backgrounds = [scene.background for scene in viewer.scenes.values()]
    assert all(b.mode == "uniform" and b.color == _BLACK for b in backgrounds)
    # One model per panel, so editing one leaves the others alone.
    assert len({id(b) for b in backgrounds}) == len(backgrounds)
    for scene in viewer.scenes.values():
        np.testing.assert_allclose(
            _top_color(viewer.controller, scene.id), _BLACK, atol=1e-6
        )


def test_a_directly_built_scene_keeps_the_gradient_default():
    """Only the convenience viewers default to black."""
    from cellier.controller import CellierController

    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=spatial_axes("y", "x"), dim="2d")

    assert scene.background.mode == "vertical_gradient"


def test_switching_the_default_to_a_gradient_shows_the_gray_gradient():
    from cellier.scene._background import DEFAULT_TOP_COLOR

    viewer = Viewer(spatial_axes("y", "x"), dim="2d")

    viewer.background.mode = "vertical_gradient"

    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id), DEFAULT_TOP_COLOR, atol=1e-6
    )


def test_viewer_background_property_is_the_scene_model():
    viewer = Viewer(spatial_axes("y", "x"), dim="2d")
    assert viewer.background is viewer.scene.background


def test_viewer_background_field_change_reaches_the_render_layer():
    viewer = Viewer(spatial_axes("y", "x"), dim="2d")

    viewer.background.color = (1.0, 0.0, 0.0, 1.0)

    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id),
        (1.0, 0.0, 0.0, 1.0),
        atol=1e-6,
    )


def test_viewer_background_can_be_replaced():
    viewer = Viewer(spatial_axes("y", "x"), dim="2d")

    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 1.0, 0.0, 1.0))

    assert viewer.scene.background.mode == "uniform"
    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id),
        (0.0, 1.0, 0.0, 1.0),
        atol=1e-6,
    )
    # The new model stays connected.
    viewer.background.color = (0.0, 0.0, 1.0, 1.0)
    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id),
        (0.0, 0.0, 1.0, 1.0),
        atol=1e-6,
    )


def test_ortho_set_background_applies_to_all_panels():
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))

    viewer.set_background(
        BackgroundAppearance(mode="uniform", color=(1.0, 0.0, 0.0, 1.0))
    )

    for scene in viewer.scenes.values():
        np.testing.assert_allclose(
            _top_color(viewer.controller, scene.id), (1.0, 0.0, 0.0, 1.0), atol=1e-6
        )


def test_ortho_panels_get_independent_copies():
    """Editing one panel's background afterwards leaves the others alone."""
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    viewer.set_background(
        BackgroundAppearance(mode="uniform", color=(1.0, 0.0, 0.0, 1.0))
    )

    viewer.scenes["xy"].background.color = (0.0, 0.0, 1.0, 1.0)

    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scenes["xy"].id),
        (0.0, 0.0, 1.0, 1.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scenes["vol"].id),
        (1.0, 0.0, 0.0, 1.0),
        atol=1e-6,
    )
