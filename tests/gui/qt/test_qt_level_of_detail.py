"""Tests for the Qt "Level of detail" control (plane rendering design 9.3)."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.events import AppearanceChangedEvent
from cellier.gui._image_controls import display_seed
from cellier.gui._level_of_detail import level_of_detail_values
from cellier.gui.qt.visuals._level_of_detail import QtLevelOfDetailControls
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.visuals import MultiscaleImageAppearance
from tests._v2 import pyramid_levels


def _controller_with_visual(small_zarr_store, dim="3d", **appearance):
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim=dim, coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store),
        scale_names=["s0", "s1"],
        **pyramid_levels(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
    )
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(**appearance),
        name="vol",
    )
    return controller, scene, visual


def _control(qtbot, controller, visual):
    scene_ids, n_displayed = display_seed(controller, [visual.id])
    control = QtLevelOfDetailControls(
        visual.id,
        level_of_detail_values(visual.appearance),
        n_displayed_dimensions=n_displayed,
        scene_ids=scene_ids,
    )
    qtbot.addWidget(control.widget)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())
    return control


def test_instantiate_smoke(qtbot):
    control = QtLevelOfDetailControls(
        object(),
        {
            "settled_lod_bias": 2.0,
            "coarsest_while_moving_3d": False,
            "coarsest_while_moving_2d": True,
        },
    )
    qtbot.addWidget(control.widget)
    assert control._slider.value() == pytest.approx(2.0)
    assert control.moving_field == "coarsest_while_moving_3d"
    assert control._moving_check.isChecked() is False
    assert control.widget.title() == "Level of detail"


def test_defaults_follow_the_model(qtbot):
    control = QtLevelOfDetailControls(object())
    qtbot.addWidget(control.widget)
    defaults = level_of_detail_values(MultiscaleImageAppearance())
    assert control._slider.value() == pytest.approx(defaults["settled_lod_bias"])
    assert control._moving == {
        key: value for key, value in defaults.items() if key != "settled_lod_bias"
    }


def test_slider_release_reaches_the_model(qtbot, small_zarr_store):
    controller, _scene, visual = _controller_with_visual(small_zarr_store)
    control = _control(qtbot, controller, visual)
    control._slider.setValue(3.5)
    assert visual.appearance.settled_lod_bias == pytest.approx(1.0)  # not yet
    control._on_slider_released()
    assert visual.appearance.settled_lod_bias == pytest.approx(3.5)


def test_the_checkbox_edits_the_3d_setting_in_a_3d_view(qtbot, small_zarr_store):
    controller, _scene, visual = _controller_with_visual(small_zarr_store)
    control = _control(qtbot, controller, visual)
    assert control._moving_check.isChecked() is True
    control._moving_check.setChecked(False)
    assert visual.appearance.coarsest_while_moving_3d is False
    assert visual.appearance.coarsest_while_moving_2d is False  # untouched


def test_the_checkbox_edits_the_2d_setting_in_a_2d_view(qtbot, small_zarr_store):
    controller, _scene, visual = _controller_with_visual(small_zarr_store, dim="2d")
    control = _control(qtbot, controller, visual)
    assert control.moving_field == "coarsest_while_moving_2d"
    assert control._moving_check.isChecked() is False
    control._moving_check.setChecked(True)
    assert visual.appearance.coarsest_while_moving_2d is True
    assert visual.appearance.coarsest_while_moving_3d is True  # its default


def test_the_checkbox_follows_the_view_between_2d_and_3d(qtbot, small_zarr_store):
    controller, scene, visual = _controller_with_visual(small_zarr_store)
    control = _control(qtbot, controller, visual)
    emitted = []
    control.changed.connect(emitted.append)
    assert control._moving_check.isChecked() is True

    controller.set_displayed_axes(scene.id, (1, 2))
    assert control.n_displayed_dimensions == 2
    assert control._moving_check.isChecked() is False
    controller.set_displayed_axes(scene.id, (0, 1, 2))
    assert control._moving_check.isChecked() is True
    # Showing another view's setting is not an edit.
    assert emitted == []
    assert visual.appearance.coarsest_while_moving_3d is True
    assert visual.appearance.coarsest_while_moving_2d is False


def test_a_model_change_updates_the_control_without_reemitting(qtbot, small_zarr_store):
    controller, _scene, visual = _controller_with_visual(small_zarr_store)
    control = _control(qtbot, controller, visual)
    emitted = []
    control.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "settled_lod_bias", 4.0)
    controller.update_appearance_field(visual.id, "coarsest_while_moving_3d", False)
    assert control._slider.value() == pytest.approx(4.0)
    assert control._moving_check.isChecked() is False
    assert emitted == []

    # The other view's setting is remembered but not shown.
    controller.update_appearance_field(visual.id, "coarsest_while_moving_2d", True)
    assert control._moving_check.isChecked() is False
    control.n_displayed_dimensions = 2
    assert control._moving_check.isChecked() is True
    assert emitted == []


def test_the_echo_of_its_own_write_is_ignored(qtbot):
    control = QtLevelOfDetailControls(object(), {"settled_lod_bias": 1.0})
    qtbot.addWidget(control.widget)
    for field, value in (
        ("settled_lod_bias", 9.0),
        ("coarsest_while_moving_3d", False),
    ):
        control._on_appearance_changed(
            AppearanceChangedEvent(
                source_id=control._id,
                visual_id=control.visual_ids[0],
                field_name=field,
                new_value=value,
                requires_reslice=True,
            )
        )
    assert control._slider.value() == pytest.approx(1.0)
    assert control._moving_check.isChecked() is True


def test_an_unrelated_field_is_ignored(qtbot, small_zarr_store):
    controller, _scene, visual = _controller_with_visual(small_zarr_store)
    control = _control(qtbot, controller, visual)
    controller.update_appearance_field(visual.id, "interpolation", "linear")
    assert control._slider.value() == pytest.approx(1.0)
    assert control._moving_check.isChecked() is True


def test_n_displayed_dimensions_is_validated(qtbot):
    control = QtLevelOfDetailControls(object())
    qtbot.addWidget(control.widget)
    with pytest.raises(ValueError, match="2 or 3"):
        control.n_displayed_dimensions = 4
