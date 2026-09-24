"""Which rows of the image control are shown, on both toolkits.

The render mode, the iso threshold and the attenuation only mean something in
3D, and the last two only for some render modes, so their rows hide
themselves.  The rule is decided once
(:func:`cellier.gui._image_controls.row_visible`) and each toolkit re-applies
it in full on every trigger: startup, a render-mode change from either side,
a composite switch and a 2D/3D switch on a followed scene.

The anywidget front end applies the rule in JavaScript, which has no test
runner here; these tests pin the traits it reads.  The JS itself is checked
through ``scripts/anywidget_js_harness.py``.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import DimsChangedEvent
from cellier.gui._image_controls import (
    ATTENUATION_MODES,
    THRESHOLD_MODES,
    display_seed,
    image_control_values,
    row_visible,
    validate_n_displayed_dimensions,
)
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageChannelAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageSingleAppearance,
    MultiscaleImageVisual,
)
from tests._v2 import level_transforms

_FIELDS = ["visible", "clim", "opacity", "render_mode", "iso_threshold"]
_MS_FIELDS = [*_FIELDS, "attenuation"]


def _multiscale(*, render_mode="mip", channel_modes=None):
    """A multiscale visual model; no controller needed to seed a control."""
    channels = {
        i: MultiscaleImageChannelAppearance(render_mode=mode)
        for i, mode in enumerate(channel_modes or [])
    }
    return MultiscaleImageVisual(
        name="ms",
        data_store_id="store",
        level_transforms=level_transforms(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.5]],
        ),
        single=MultiscaleImageSingleAppearance(render_mode=render_mode),
        channel_axis=0,
        composite=bool(channels),
        channels=channels,
    )


def _values(visual):
    return image_control_values(visual, fields=_MS_FIELDS)


# ---------------------------------------------------------------------------
# The rule, without a toolkit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "modes", "n", "expected"),
    [
        ("clim", ["mip"], 2, True),  # a row that never hides
        ("opacity", ["iso"], 2, True),
        ("render_mode", ["mip"], 3, True),
        ("render_mode", ["mip"], 2, False),
        ("iso_threshold", ["iso"], 3, True),
        ("iso_threshold", ["smooth_iso"], 3, True),
        ("iso_threshold", ["mip"], 3, False),
        ("iso_threshold", ["iso"], 2, False),
        ("attenuation", ["attenuated_mip"], 3, True),
        ("attenuation", ["mip"], 3, False),
        ("attenuation", ["attenuated_mip"], 2, False),
        # The composite page's attenuation row serves every channel.
        ("attenuation", ["mip", "attenuated_mip"], 3, True),
        ("attenuation", [], 3, False),
    ],
)
def test_row_visible(field, modes, n, expected):
    assert row_visible(field, modes, n) is expected


def test_the_mode_lists():
    assert THRESHOLD_MODES == ("iso", "smooth_iso")
    assert ATTENUATION_MODES == ("attenuated_mip",)
    values = _values(_multiscale())
    assert values["threshold_modes"] == ["iso", "smooth_iso"]
    assert values["attenuation_modes"] == ["attenuated_mip"]


@pytest.mark.parametrize("value", [2, 3])
def test_a_display_dimensionality_is_2_or_3(value):
    assert validate_n_displayed_dimensions(value) == value


@pytest.mark.parametrize(("value", "error"), [(1, ValueError), (4, ValueError)])
def test_other_dimensionalities_raise(value, error):
    with pytest.raises(error, match="must be 2 or 3"):
        validate_n_displayed_dimensions(value)


@pytest.mark.parametrize("value", [True, 3.0, "3"])
def test_a_dimensionality_must_be_an_int(value):
    with pytest.raises(TypeError, match="must be an int"):
        validate_n_displayed_dimensions(value)


def _scene_image(dim):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=spatial_axes("z", "y", "x"), dim=dim)
    store = ImageMemoryStore(data=np.zeros((4, 8, 8), dtype=np.float32))
    visual = controller.add_image(store, scene.id)
    return controller, scene, visual


@pytest.mark.parametrize(("dim", "n"), [("2d", 2), ("3d", 3)])
def test_the_seed_reads_the_scene(dim, n):
    controller, scene, visual = _scene_image(dim)

    assert display_seed(controller, [visual.id]) == ((scene.id,), n)


def test_without_a_controller_the_seed_shows_everything():
    assert display_seed(None, [uuid4()]) == ((), 3)


def test_the_controller_names_a_visuals_scene():
    controller, scene, visual = _scene_image("2d")

    assert controller.get_visual_scene_id(visual.id) == scene.id
    with pytest.raises(KeyError, match="No visual"):
        controller.get_visual_scene_id(uuid4())


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


@pytest.fixture
def qt_controls(qtbot):
    pytest.importorskip("superqt")
    from cellier.gui.qt.visuals import QtImageControls

    return QtImageControls


def _shown(widget, page, channel, field) -> bool:
    return not widget._controls[(page, channel, field)].isHidden()


def test_qt_3d_rows_follow_the_render_mode(qt_controls):
    widget = qt_controls(uuid4(), _values(_multiscale(render_mode="mip")))
    render_mode = widget._controls[("single", None, "render_mode")]

    assert _shown(widget, "single", None, "render_mode")
    assert not _shown(widget, "single", None, "iso_threshold")
    assert not _shown(widget, "single", None, "attenuation")

    render_mode.setCurrentText("iso")
    assert _shown(widget, "single", None, "iso_threshold")
    assert not _shown(widget, "single", None, "attenuation")

    render_mode.setCurrentText("attenuated_mip")
    assert not _shown(widget, "single", None, "iso_threshold")
    assert _shown(widget, "single", None, "attenuation")


def test_qt_attenuation_sits_under_the_render_mode(qt_controls):
    widget = qt_controls(uuid4(), _values(_multiscale()))
    form = widget._single_page.layout()

    def row_of(field):
        return form.getWidgetPosition(widget._controls[("single", None, field)])[0]

    assert row_of("attenuation") == row_of("render_mode") + 1


def test_qt_2d_hides_every_3d_row(qt_controls):
    widget = qt_controls(
        uuid4(), _values(_multiscale(render_mode="iso")), n_displayed_dimensions=2
    )

    for field in ("render_mode", "iso_threshold", "attenuation"):
        assert not _shown(widget, "single", None, field)
    assert _shown(widget, "single", None, "clim")

    widget.n_displayed_dimensions = 3
    assert _shown(widget, "single", None, "render_mode")
    assert _shown(widget, "single", None, "iso_threshold")


def test_qt_the_setter_validates(qt_controls):
    widget = qt_controls(uuid4(), _values(_multiscale()))

    with pytest.raises(ValueError, match="2 or 3"):
        widget.n_displayed_dimensions = 4
    with pytest.raises(ValueError, match="2 or 3"):
        qt_controls(uuid4(), _values(_multiscale()), n_displayed_dimensions=1)


def test_qt_channel_rows_follow_their_own_mode(qt_controls):
    widget = qt_controls(uuid4(), _values(_multiscale(channel_modes=["iso", "mip"])))

    assert _shown(widget, "channel", 0, "iso_threshold")
    assert not _shown(widget, "channel", 1, "iso_threshold")
    # One attenuation row below the channels, shown when any channel uses it.
    assert not _shown(widget, "composite", None, "attenuation")

    widget._controls[("channel", 1, "render_mode")].setCurrentText("attenuated_mip")
    assert _shown(widget, "composite", None, "attenuation")
    assert _shown(widget, "channel", 0, "iso_threshold")


def test_qt_both_attenuation_rows_show_one_value(qt_controls):
    visual_id = uuid4()
    widget = qt_controls(visual_id, _values(_multiscale()))
    single = widget._controls[("single", None, "attenuation")]
    composite = widget._controls[("composite", None, "attenuation")]
    emitted: list = []
    widget.changed.connect(emitted.append)

    composite.setValue(4.0)

    assert emitted[-1].field == "attenuation"
    assert emitted[-1].value == pytest.approx(4.0)
    # Inbound, both are set.
    widget._appliers[("shared", None, "attenuation")](2.5)
    assert single.value() == pytest.approx(2.5)
    assert composite.value() == pytest.approx(2.5)


def _in_memory(dim="3d"):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")], dim=dim
    )
    store = ImageMemoryStore(data=np.zeros((2, 4, 8, 8), dtype=np.float32))
    visual = controller.add_image(
        store,
        scene.id,
        channel_axis=0,
        channels={i: InMemoryImageChannelAppearance() for i in range(2)},
    )
    return controller, scene, visual


def _wired(controller, widget):
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    return widget


def test_qt_an_inbound_render_mode_updates_the_rows(qt_controls):
    controller, scene, visual = _in_memory()
    values = image_control_values(visual, fields=_FIELDS)
    widget = _wired(controller, qt_controls(visual.id, values, scene_ids=[scene.id]))
    assert not _shown(widget, "single", None, "iso_threshold")

    controller.update_single_appearance_field(visual.id, "render_mode", "iso")
    assert _shown(widget, "single", None, "iso_threshold")

    controller.update_channel_appearance_field(visual.id, 1, "render_mode", "iso")
    assert _shown(widget, "channel", 1, "iso_threshold")
    assert not _shown(widget, "channel", 0, "iso_threshold")


def test_qt_follows_a_2d_3d_switch_on_its_scene(qt_controls):
    controller, scene, visual = _in_memory(dim="2d")
    values = image_control_values(visual, fields=_FIELDS)
    scene_ids, n = display_seed(controller, [visual.id])
    widget = _wired(
        controller,
        qt_controls(visual.id, values, n_displayed_dimensions=n, scene_ids=scene_ids),
    )
    assert widget.n_displayed_dimensions == 2
    assert not _shown(widget, "single", None, "render_mode")

    controller.set_displayed_axes(scene.id, (1, 2, 3))
    assert widget.n_displayed_dimensions == 3
    assert _shown(widget, "single", None, "render_mode")

    controller.set_displayed_axes(scene.id, (2, 3))
    assert widget.n_displayed_dimensions == 2
    assert not _shown(widget, "single", None, "render_mode")


def test_qt_a_slice_move_leaves_a_manual_setting(qt_controls):
    """Only a change of displayed axes reaches the control."""
    widget = qt_controls(uuid4(), _values(_multiscale()), n_displayed_dimensions=2)
    scene_id = uuid4()
    event = DimsChangedEvent(
        source_id=uuid4(),
        scene_id=scene_id,
        dims_state=_dims_state(3),
        displayed_axes_changed=False,
    )

    widget._on_dims_changed(event)

    assert widget.n_displayed_dimensions == 2


def _dims_state(n):
    from types import SimpleNamespace

    return SimpleNamespace(selection=SimpleNamespace(displayed_axes=tuple(range(n))))


def test_qt_subscribes_to_each_scene(qt_controls):
    scenes = [uuid4(), uuid4()]
    widget = qt_controls(uuid4(), _values(_multiscale()), scene_ids=scenes)

    dims_specs = [
        s for s in widget.subscription_specs() if s.event_type is DimsChangedEvent
    ]
    assert [s.entity_id for s in dims_specs] == scenes


# ---------------------------------------------------------------------------
# anywidget
# ---------------------------------------------------------------------------


@pytest.fixture
def any_controls():
    pytest.importorskip("anywidget")
    from cellier.gui.anywidget.visuals import AnywidgetImageControls

    return AnywidgetImageControls


def test_anywidget_carries_the_rule_inputs(any_controls):
    widget = any_controls(uuid4(), _values(_multiscale()), n_displayed_dimensions=2)

    assert widget.n_displayed_dimensions == 2
    assert widget.threshold_modes == ["iso", "smooth_iso"]
    assert widget.attenuation_modes == ["attenuated_mip"]


def test_anywidget_defaults_to_3d(any_controls):
    assert any_controls(uuid4(), _values(_multiscale())).n_displayed_dimensions == 3


def test_anywidget_the_trait_validates(any_controls):
    widget = any_controls(uuid4(), _values(_multiscale()))

    with pytest.raises(ValueError, match="2 or 3"):
        widget.n_displayed_dimensions = 4


def test_anywidget_follows_a_2d_3d_switch_on_its_scene(any_controls):
    controller, scene, visual = _in_memory(dim="2d")
    values = image_control_values(visual, fields=_FIELDS)
    scene_ids, n = display_seed(controller, [visual.id])
    widget = _wired(
        controller,
        any_controls(visual.id, values, n_displayed_dimensions=n, scene_ids=scene_ids),
    )

    controller.set_displayed_axes(scene.id, (1, 2, 3))
    assert widget.n_displayed_dimensions == 3

    controller.set_displayed_axes(scene.id, (2, 3))
    assert widget.n_displayed_dimensions == 2
