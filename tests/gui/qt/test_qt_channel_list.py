"""Tests for the Qt ``QtChannelList`` composite channel-controls widget."""

from __future__ import annotations

import numpy as np
import pytest
from cmap import Colormap

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.gui.qt.visuals import QtChannelList
from cellier.scene.dims import CoordinateSystem
from cellier.visuals._channel_appearance import ChannelAppearance


def _make_channel_appearance(**kwargs) -> ChannelAppearance:
    defaults = {"color_map": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return ChannelAppearance(**defaults)


def _make_4d_store(shape=(3, 2, 16, 16)) -> ImageMemoryStore:
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    return ImageMemoryStore(data=data)


def _make_controller_with_multichannel(n_channels=2):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "c", "y", "x"))
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    store = _make_4d_store()
    channels = {i: _make_channel_appearance() for i in range(n_channels)}
    visual = controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
    )
    return controller, visual


def _find_control(widget: QtChannelList, channel_index: int, field: str):
    # The applier closes over the target control; reach it for driving in tests.
    applier = widget._appliers[(channel_index, field)]
    return applier.__closure__[0].cell_contents


def test_checkbox_edit_reaches_model(qtbot):
    controller, visual = _make_controller_with_multichannel()
    widget = QtChannelList([visual.id], visual.channels)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    checkbox = _find_control(widget, 0, "visible")
    assert visual.channels[0].visible is True
    checkbox.setChecked(False)
    assert visual.channels[0].visible is False


def test_slider_edit_reaches_model(qtbot):
    controller, visual = _make_controller_with_multichannel()
    widget = QtChannelList([visual.id], visual.channels, clim_range=(0.0, 1000.0))
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    clim = _find_control(widget, 1, "clim")
    clim.setValue((100.0, 900.0))
    assert visual.channels[1].clim == (100.0, 900.0)


def test_model_push_updates_control_without_reemit(qtbot):
    controller, visual = _make_controller_with_multichannel()
    widget = QtChannelList([visual.id], visual.channels)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    emitted = []
    widget.changed.connect(emitted.append)

    # Programmatic push (default source_id != widget) -> control reflects it.
    controller.update_channel_appearance_field(visual.id, 0, "opacity", 0.25)

    opacity = _find_control(widget, 0, "opacity")
    assert opacity.value() == pytest.approx(0.25)
    # The inbound apply must not echo back onto the bus.
    assert emitted == []


def test_inbound_colormap_object_shows_name_string(qtbot):
    controller, visual = _make_controller_with_multichannel()
    widget = QtChannelList([visual.id], visual.channels)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    # new_value arrives as a cmap.Colormap object; combo should show the name.
    controller.update_channel_appearance_field(
        visual.id, 0, "color_map", Colormap("magma")
    )

    combo = _find_control(widget, 0, "color_map")
    # colormap_to_str yields the canonical (possibly namespaced) name string.
    assert "magma" in combo.currentColormap().name


def test_edit_fans_out_to_all_visual_ids(qtbot):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "c", "y", "x"))
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")

    def _add():
        store = _make_4d_store()
        return controller.add_multichannel_image(
            data=store,
            scene_id=scene.id,
            channel_axis=1,
            channels={0: _make_channel_appearance(), 1: _make_channel_appearance()},
        )

    visual0 = _add()
    visual1 = _add()

    widget = QtChannelList([visual0.id, visual1.id], visual0.channels)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    checkbox = _find_control(widget, 1, "visible")
    checkbox.setChecked(False)

    assert visual0.channels[1].visible is False
    assert visual1.channels[1].visible is False


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
    from cellier.events import ChannelAppearanceChangedEvent

    controller, visual = _make_controller_with_multichannel()
    widget = QtChannelList([visual.id], visual.channels)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    before = _find_control(widget, 0, "opacity").value()

    widget._on_changed(
        ChannelAppearanceChangedEvent(
            source_id=widget._id,  # our own echo -> ignored
            visual_id=visual.id,
            channel_index=0,
            field_name="opacity",
            new_value=0.25,
        )
    )

    assert _find_control(widget, 0, "opacity").value() == pytest.approx(before)
