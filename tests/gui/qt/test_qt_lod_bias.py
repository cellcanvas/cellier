"""Tests for the Qt ``QtLodBiasSlider`` LOD-bias-control widget."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController
from cellier.data.image._zarr_multiscale_store import (
    MultiscaleZarrDataStore,
)
from cellier.gui.qt.visuals._lod_bias import QtLodBiasSlider
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.visuals import MultiscaleImageAppearance
from tests._v2 import pyramid_levels


def _make_store(small_zarr_store, **kwargs) -> MultiscaleZarrDataStore:
    defaults = {
        "zarr_path": str(small_zarr_store),
        "scale_names": ["s0", "s1"],
        **pyramid_levels(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
    }
    defaults.update(kwargs)
    return MultiscaleZarrDataStore(**defaults)


def _make_controller_with_visual(small_zarr_store, initial_lod_bias=1.0):
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    appearance = MultiscaleImageAppearance(lod_bias=initial_lod_bias)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=appearance, name="vol"
    )
    return controller, visual


def test_instantiate_smoke(qtbot):
    slider = QtLodBiasSlider(visual_id=object(), initial_lod_bias=2.0)
    qtbot.addWidget(slider.widget)
    assert slider._slider.value() == pytest.approx(2.0)


def test_slider_release_reaches_model(qtbot, small_zarr_store):
    controller, visual = _make_controller_with_visual(small_zarr_store)
    slider = QtLodBiasSlider(visual_id=visual.id)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    slider._slider.setValue(3.5)
    slider._on_slider_released()  # simulate the user releasing the handle

    assert visual.appearance.lod_bias == pytest.approx(3.5)


def test_model_push_updates_slider_without_reemit(qtbot, small_zarr_store):
    controller, visual = _make_controller_with_visual(small_zarr_store)
    slider = QtLodBiasSlider(visual_id=visual.id)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    emitted = []
    slider.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "lod_bias", 4.0)

    assert slider._slider.value() == pytest.approx(4.0)
    assert emitted == []


def test_inbound_echo_filtered_by_source_id(qtbot, small_zarr_store):
    """A widget must ignore the bus echo of its own write.

    The anywidget twin has always covered this; the Qt module did not, which
    is the kind of one-sided gap ``plans/gui_backend_unification.md`` D14 is
    about -- the two front ends implement the same ``source_id`` filter, so
    testing it on one side only leaves the other free to lose it.

    Driven through the handler rather than through the controller because the
    controller stamps its own ``source_id``; the case under test is
    specifically an event carrying *this widget's* id.
    """
    from cellier.events import AppearanceChangedEvent

    _controller, visual = _make_controller_with_visual(small_zarr_store)
    slider = QtLodBiasSlider(visual_id=visual.id, initial_lod_bias=1.0)
    qtbot.addWidget(slider.widget)

    slider._on_visual_changed(
        AppearanceChangedEvent(
            source_id=slider._id,  # our own echo -> ignored
            visual_id=visual.id,
            field_name="lod_bias",
            new_value=9.0,
            requires_reslice=True,
        )
    )

    assert slider._slider.value() == pytest.approx(1.0)  # unchanged


def test_unrelated_field_change_ignored(qtbot, small_zarr_store):
    controller, visual = _make_controller_with_visual(small_zarr_store)
    slider = QtLodBiasSlider(visual_id=visual.id, initial_lod_bias=1.0)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    controller.update_appearance_field(visual.id, "interpolation", "linear")

    assert slider._slider.value() == pytest.approx(1.0)
