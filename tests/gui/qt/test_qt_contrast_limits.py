"""Tests for the Qt ``QtClimRangeSlider`` contrast-limits-control widget."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.gui.qt.visuals._contrast_limits import QtClimRangeSlider
from cellier.scene.dims import CoordinateSystem
from cellier.visuals._image_memory import InMemoryImageAppearance


def _make_controller_with_visual(clim=(0.0, 1.0)):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d", coordinate_system=cs, name="main", render_modes={"2d"}
    )
    data = np.zeros((16, 16), dtype=np.float32)
    store = ImageMemoryStore(data=data)
    appearance = InMemoryImageAppearance(color_map="viridis", clim=clim)
    visual = controller.add_image(data=store, scene_id=scene.id, appearance=appearance)
    return controller, visual


def test_instantiate_smoke(qtbot):
    slider = QtClimRangeSlider(
        visual_id=object(), clim_range=(0.0, 255.0), initial_clim=(10.0, 200.0)
    )
    qtbot.addWidget(slider.widget)
    assert tuple(slider._slider.value()) == pytest.approx((10.0, 200.0))


def test_slider_edit_reaches_model(qtbot):
    controller, visual = _make_controller_with_visual()
    slider = QtClimRangeSlider(
        visual_id=visual.id, clim_range=(0.0, 1000.0), initial_clim=(0.0, 1.0)
    )
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    slider._on_slider_changed((100.0, 900.0))

    assert visual.appearance.clim == (100.0, 900.0)


def test_model_push_updates_slider_without_reemit(qtbot):
    controller, visual = _make_controller_with_visual()
    slider = QtClimRangeSlider(
        visual_id=visual.id, clim_range=(0.0, 1000.0), initial_clim=(0.0, 1.0)
    )
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    emitted = []
    slider.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "clim", (50.0, 500.0))

    assert tuple(slider._slider.value()) == pytest.approx((50.0, 500.0))
    assert emitted == []


def test_unrelated_field_change_ignored(qtbot):
    controller, visual = _make_controller_with_visual(clim=(0.0, 1.0))
    slider = QtClimRangeSlider(
        visual_id=visual.id, clim_range=(0.0, 1000.0), initial_clim=(0.0, 1.0)
    )
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    controller.update_appearance_field(visual.id, "interpolation", "linear")

    assert tuple(slider._slider.value()) == pytest.approx((0.0, 1.0))


def test_inbound_echo_filtered_by_source_id(qtbot):
    """A widget must ignore the bus echo of its own write.

    The core of the bus contract, and the one thing that stops a control and
    its model oscillating.  Covered on every anywidget module and, until this,
    on only three Qt ones -- the systematic half of D14
    (``plans/gui_backend_unification.md``).

    Driven through the handler rather than the controller, because the
    controller stamps its own ``source_id``; the case under test is an event
    carrying *this widget's* id.
    """
    from cellier.events import AppearanceChangedEvent

    controller, visual = _make_controller_with_visual(clim=(0.0, 1.0))
    slider = QtClimRangeSlider(
        visual_id=visual.id, clim_range=(0.0, 1000.0), initial_clim=(0.0, 1.0)
    )
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    slider._on_visual_changed(
        AppearanceChangedEvent(
            source_id=slider._id,  # our own echo -> ignored
            visual_id=visual.id,
            field_name="clim",
            new_value=(10.0, 900.0),
            requires_reslice=False,
        )
    )

    assert tuple(slider.control.value()) == (0.0, 1.0)
