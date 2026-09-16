"""The unified image control on both toolkits (unified image design 3.10).

One control per image visual: the shared section, a composite switch shown
only when the visual has a channel axis, and a page per mode.  These tests
drive the control through a real controller, so a mode switch travels the bus
both ways.

``color_map`` is never written here: writing it poisons ``==`` on the
appearance class for the rest of the process (see the convenience cleanup
notes), which would break unrelated round-trip tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.gui._image_controls import image_control_values
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    ImageVisual,
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageVisual,
)
from tests._v2 import level_transforms

_CYX = [("c", "channel"), *spatial_axes("y", "x")]
_FIELDS = ["visible", "clim", "opacity", "render_mode", "iso_threshold"]


def _image(*, channel_axis=0, composite=False, n_channels=2):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_CYX, dim="2d")
    store = ImageMemoryStore(data=np.zeros((3, 8, 8), dtype=np.float32))
    visual = controller.add_image(
        store,
        scene.id,
        channel_axis=channel_axis,
        composite=composite,
        channels={i: InMemoryImageChannelAppearance() for i in range(n_channels)},
    )
    return controller, visual


def _values(visual):
    return image_control_values(
        visual, fields=_FIELDS, clim_range=(0.0, 1.0), channel_labels={1: "GFP"}
    )


def _connect(controller, widget):
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    return widget


def _multiscale_model(**kwargs):
    return MultiscaleImageVisual(
        name="ms",
        data_store_id="store",
        level_transforms=level_transforms(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.5]],
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Toolkit-neutral values
# ---------------------------------------------------------------------------


def test_values_follow_the_storage_type():
    memory = _values(ImageVisual(name="m", data_store_id="s", channel_axis=0))
    multiscale = image_control_values(
        _multiscale_model(channel_axis=0), fields=[*_FIELDS, "attenuation"]
    )

    assert memory["storage"] == "memory"
    assert "attenuation" not in memory["shared"]
    assert set(memory["render_modes"]) == {"mip", "iso", "minip"}
    assert multiscale["storage"] == "multiscale"
    assert "attenuation" in multiscale["shared"]
    assert {"smooth_iso", "attenuated_mip"} <= set(multiscale["render_modes"])


def test_values_without_a_channel_axis_say_so():
    values = _values(ImageVisual(name="m", data_store_id="s"))
    assert values["has_channel_axis"] is False
    assert values["channels"] == {}


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


@pytest.fixture
def qt_controls(qtbot):
    pytest.importorskip("superqt")
    from cellier.gui.qt.visuals import QtImageControls

    return QtImageControls


def test_qt_mode_switch_round_trips_through_the_bus(qt_controls):
    controller, visual = _image()
    widget = _connect(controller, qt_controls(visual.id, _values(visual)))
    assert widget.composite is False

    widget._composite_box.setChecked(True)
    assert visual.composite is True
    assert widget._pages.currentIndex() == 1

    controller.set_image_composite(visual.id, False)
    assert widget.composite is False
    assert widget._composite_box.isChecked() is False


def test_qt_a_refused_toggle_leaves_the_widget_consistent(qtbot, qt_controls):
    """Compositing a displayed axis raises; the switch and page go back."""
    controller, visual = _image(channel_axis=1)
    widget = _connect(controller, qt_controls(visual.id, _values(visual)))

    with qtbot.capture_exceptions() as exceptions:
        widget._composite_box.setChecked(True)

    assert exceptions, "the refusal must surface, not be swallowed"
    assert visual.composite is False
    assert widget._composite_box.isChecked() is False
    assert widget.composite is False


def test_qt_switch_is_hidden_without_a_channel_axis(qt_controls):
    _controller, visual = _image(channel_axis=None, n_channels=0)
    widget = qt_controls(visual.id, _values(visual))
    assert widget._composite_box.isHidden()

    _controller, visual = _image()
    widget = qt_controls(visual.id, _values(visual))
    assert not widget._composite_box.isHidden()


def test_qt_an_empty_composite_shows_an_empty_channel_list(qt_controls):
    _controller, visual = _image(composite=True, n_channels=0)
    widget = qt_controls(visual.id, _values(visual))

    assert widget.composite is True
    assert widget._channel_groups == {}
    assert widget._empty_label.text() == "No channels"


def test_qt_pages_hold_the_right_fields(qt_controls):
    _controller, visual = _image()
    widget = qt_controls(visual.id, _values(visual))

    single = {f for page, _ch, f in widget._controls if page == "single"}
    channel = {(ch, f) for page, ch, f in widget._controls if page == "channel"}
    assert single == {"clim", "opacity", "render_mode", "iso_threshold"}
    assert {f for ch, f in channel if ch == 1} == single | {"visible"}
    assert ("shared", None, "attenuation") not in widget._controls

    multiscale = _multiscale_model(
        channel_axis=0, channels={0: MultiscaleImageChannelAppearance()}
    )
    ms_widget = qt_controls(
        multiscale.id,
        image_control_values(multiscale, fields=[*_FIELDS, "attenuation"]),
    )
    assert ("shared", None, "attenuation") in ms_widget._controls


def test_qt_single_and_channel_edits_reach_the_model_and_back(qt_controls):
    controller, visual = _image()
    widget = _connect(controller, qt_controls(visual.id, _values(visual)))

    widget._controls[("single", None, "render_mode")].setCurrentText("iso")
    widget._controls[("channel", 1, "clim")].setValue((0.2, 0.4))
    assert visual.single.render_mode == "iso"
    assert visual.channels[1].clim == pytest.approx((0.2, 0.4))

    controller.update_single_appearance_field(visual.id, "iso_threshold", 0.75)
    value = widget._controls[("single", None, "iso_threshold")].value()
    assert value == pytest.approx(0.75)


def test_qt_shows_a_colormap_that_no_name_can_reconstruct(qt_controls):
    """The lightsheet case: channels built with an inline ``Colormap``.

    ``Colormap([...], name="white_green")`` is not in cmap's catalogue, so
    superqt cannot resolve that name -- the control has to hand it the model's
    own object.  Built at construction and never assigned, and no model is
    compared here: assigning ``color_map`` with a listener attached poisons
    model equality process-wide.
    """
    from cmap import Colormap

    white_green = Colormap(["#ffffff", "#00ff00"], name="white_green")
    white_blue = Colormap(["#ffffff", "#0000ff"], name="white_blue")
    visual = ImageVisual(
        name="m",
        data_store_id="s",
        channel_axis=0,
        composite=True,
        single=InMemoryImageSingleAppearance(color_map=white_blue),
        channels={0: InMemoryImageChannelAppearance(color_map=white_green)},
    )
    values = image_control_values(visual, fields=["color_map"], clim_range=(0.0, 1.0))
    assert values["colormaps"]["channels"][0] is white_green

    widget = qt_controls(visual.id, values)

    assert widget._controls[("channel", 0, "color_map")].currentColormap().name == (
        "white_green"
    )
    assert widget._controls[("single", None, "color_map")].currentColormap().name == (
        "white_blue"
    )


def test_qt_an_unresolvable_inbound_colormap_name_does_not_raise(qt_controls):
    """An inline colormap that arrives as its name leaves the combo alone.

    Raising here would travel back out through the bus and take down whatever
    edit emitted the event.
    """
    _controller, visual = _image()
    values = image_control_values(visual, fields=["color_map"], clim_range=(0.0, 1.0))
    widget = qt_controls(visual.id, values)
    control = widget._controls[("single", None, "color_map")]
    before = control.currentColormap().name

    widget._appliers[("single", None, "color_map")]("white_green")

    assert control.currentColormap().name == before


# ---------------------------------------------------------------------------
# anywidget
# ---------------------------------------------------------------------------


@pytest.fixture
def any_controls():
    pytest.importorskip("anywidget")
    from cellier.gui.anywidget.visuals import AnywidgetImageControls

    return AnywidgetImageControls


def test_anywidget_mode_switch_round_trips_through_the_bus(any_controls):
    controller, visual = _image()
    widget = _connect(controller, any_controls(visual.id, _values(visual)))

    widget.composite = True
    assert visual.composite is True

    controller.set_image_composite(visual.id, False)
    assert widget.composite is False


def test_anywidget_a_refused_toggle_rolls_the_trait_back(any_controls):
    controller, visual = _image(channel_axis=1)
    widget = _connect(controller, any_controls(visual.id, _values(visual)))

    with pytest.raises(Exception, match="displays"):
        widget.composite = True

    assert visual.composite is False
    assert widget.composite is False


def test_anywidget_state_for_no_axis_and_an_empty_composite(any_controls):
    _controller, visual = _image(channel_axis=None, n_channels=0)
    assert any_controls(visual.id, _values(visual)).has_channel_axis is False

    _controller, visual = _image(composite=True, n_channels=0)
    widget = any_controls(visual.id, _values(visual))
    assert widget.has_channel_axis is True
    assert widget.composite is True
    assert widget.channels == {}


def test_anywidget_channel_edits_reach_the_model_and_back(any_controls):
    controller, visual = _image()
    widget = _connect(controller, any_controls(visual.id, _values(visual)))
    assert widget.channel_labels["1"] == "GFP"

    channels = {key: dict(fields) for key, fields in widget.channels.items()}
    channels["1"]["clim"] = [0.2, 0.4]
    widget.channels = channels
    assert visual.channels[1].clim == pytest.approx((0.2, 0.4))

    controller.update_channel_appearance_field(visual.id, 0, "opacity", 0.5)
    assert widget.channels["0"]["opacity"] == pytest.approx(0.5)
