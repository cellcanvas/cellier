"""Tests for the Qt ``QtDimsControl`` widget (slider panel + 2D/3D toggle)."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.controller import CellierController
from cellier.events import DimsChangedEvent
from cellier.gui.qt._scene import QtDimsControl
from cellier.scene.dims import CoordinateSystem


def _make_controller_with_scene(*, dim="2d"):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim=dim, coordinate_system=cs, name="main")
    return controller, scene


def _make_control(scene, qtbot, *, with_toggle=True) -> QtDimsControl:
    selection = scene.dims.selection
    control = QtDimsControl(
        scene_id=scene.id,
        axis_ranges={0: (0, 9), 1: (0, 99), 2: (0, 99)},
        axis_labels={0: "z", 1: "y", 2: "x"},
        initial_slice_indices=dict(selection.slice_indices),
        initial_displayed_axes=selection.displayed_axes,
        axes_2d=(1, 2) if with_toggle else None,
        axes_3d=(0, 1, 2) if with_toggle else None,
    )
    qtbot.addWidget(control.widget)
    return control


def _dims_changed_event(source_id, scene_id, *, displayed, slices, stacked=()):
    selection = AxisAlignedSelectionState(
        displayed_axes=displayed, slice_indices=slices, stacked_axes=stacked
    )
    state = DimsState(axis_labels=("z", "y", "x"), selection=selection)
    return DimsChangedEvent(
        source_id=source_id,
        scene_id=scene_id,
        dims_state=state,
        displayed_axes_changed=False,
    )


def test_slider_drag_updates_model(qtbot):
    controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())

    control._sliders[0].setValue(5)
    assert scene.dims.selection.slice_indices[0] == 5


def test_toggle_button_omitted_without_axes(qtbot):
    _controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot, with_toggle=False)
    assert control._toggle_button is None


def test_toggle_click_uses_live_slider_value_and_updates_model(qtbot):
    controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())

    assert control._toggle_button.text() == "Switch to 3D"

    # Move axis 0's slider before toggling to 3D -- it becomes displayed and
    # so has nothing to restore yet.
    control._sliders[0].setValue(5)
    assert scene.dims.selection.slice_indices[0] == 5

    control._on_toggle_click()  # 2D -> 3D
    assert len(scene.dims.selection.displayed_axes) == 3
    assert control._toggle_button.text() == "Switch to 2D"

    control._on_toggle_click()  # 3D -> 2D
    assert len(scene.dims.selection.displayed_axes) == 2
    assert control._toggle_button.text() == "Switch to 3D"
    # Regression guard: axis 0's slice value survives the round trip through
    # 3D instead of being reset to a hardcoded default.
    assert scene.dims.selection.slice_indices[0] == 5


def test_external_dims_change_resyncs_sliders_and_toggle(qtbot):
    controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())

    # Driven by something else entirely (not this widget's own button/slider).
    controller.set_displayed_axes(scene.id, (0, 1, 2))

    assert control._toggle_button.text() == "Switch to 2D"


def test_echoed_event_is_ignored(qtbot):
    _controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)

    control._sliders[0].setValue(3)
    event = _dims_changed_event(
        source_id=control._id,
        scene_id=scene.id,
        displayed=(1, 2),
        slices={0: 999},
    )
    control._on_dims_changed(event)

    # An event stamped with our own id is our own echo; it must not reapply.
    assert control._sliders[0].value() == 3


# ---------------------------------------------------------------------------
# D14: parity with the anywidget dims panel, which had twice this coverage --
# in the area that produced D16-D18 (``plans/gui_backend_unification.md``).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dim", "expected"), [("2d", "Switch to 3D"), ("3d", "Switch to 2D")]
)
def test_the_toggle_is_labelled_for_the_mode_it_switches_to(qtbot, dim, expected):
    """At construction, before anything has been clicked.

    A button that describes the mode it is *in* rather than the one it offers
    is the failure D17 produced after a raising handler, and nothing pinned
    the un-clicked case at all.
    """
    _controller, scene = _make_controller_with_scene(dim=dim)
    control = _make_control(scene, qtbot)

    assert control._toggle_button.text() == expected


def test_a_slider_edit_reports_only_the_hidden_axes(qtbot):
    """Displayed axes are drawn, not sliced, so they carry no slice index.

    Reporting one would ask the slicer to slice an axis it is displaying.
    """
    _controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    emitted = []
    control.changed.connect(emitted.append)

    control._on_slider_changed(0, 3)

    assert emitted, "sanity: the edit reached the bus"
    displayed = set(scene.dims.selection.displayed_axes)
    assert set(emitted[-1].slice_indices).isdisjoint(displayed)


def test_toggling_twice_returns_to_the_starting_mode(qtbot):
    """The round trip, which no Qt test covered.

    Its anywidget twin has had ``toggle_click_round_trip`` since it was
    written; the Qt side checked one direction only.
    """
    _controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    start = tuple(control._displayed_axes)
    start_label = control._toggle_button.text()

    control._on_toggle_click()
    assert tuple(control._displayed_axes) != start

    control._on_toggle_click()
    assert tuple(control._displayed_axes) == start
    assert control._toggle_button.text() == start_label


def test_a_dims_change_does_not_relabel_a_control_with_no_toggle(qtbot):
    """A control built without a toggle must not grow one, or crash reaching for it."""
    controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot, with_toggle=False)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())

    assert control.has_toggle is False

    control._on_dims_changed(
        _dims_changed_event(controller._id, scene.id, displayed=(0, 1, 2), slices={})
    )

    assert control.has_toggle is False
