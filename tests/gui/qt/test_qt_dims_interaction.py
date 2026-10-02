"""The Qt dims control's press and release scopes (tracker design 4.8).

A slider press opens a dims interaction scope and its release closes it, so
a scrub ends on release.  The release flushes the throttle first: the end
plans in full, and it has to plan the final position.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QStyle, QStyleOptionSlider

from cellier.controller import CellierController
from cellier.events import DimsInteractionUpdateEvent, DimsUpdateEvent
from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.gui.qt._scene import QtDimsControl
from cellier.scene.dims import spatial_axes

DISCRETE, CONTINUOUS = 0, 1


def _make_control(qtbot, *, scene_id=None, shown=True) -> QtDimsControl:
    control = QtDimsControl(
        scene_id=scene_id or uuid4(),
        axis_values={
            DISCRETE: DiscreteAxisValues(values=tuple(float(i) for i in range(20))),
            CONTINUOUS: ContinuousAxisValues(min=0.0, max=100.0),
        },
        axis_labels={DISCRETE: "t", CONTINUOUS: "z"},
        initial_slice_indices={DISCRETE: 10.0, CONTINUOUS: 50.0},
        initial_displayed_axes=(),
        # A wide throttle, so a drag always leaves its last position pending.
        debounce_ms=10_000,
    )
    qtbot.addWidget(control.widget)
    if shown:
        control.widget.resize(500, 120)
        control.widget.show()
        qtbot.waitExposed(control.widget)
    return control


def _record(control) -> list[tuple]:
    """``("begin",)``, ``("end",)`` and ``("tick", positions, interactive)``."""
    log: list[tuple] = []

    def on_changed(event) -> None:
        if isinstance(event, DimsInteractionUpdateEvent):
            log.append((event.phase,))
        elif isinstance(event, DimsUpdateEvent):
            log.append(("tick", dict(event.slice_indices), event.interactive))

    control.changed.connect(on_changed)
    return log


def _inner_slider(control, axis):
    # QLabeledDoubleSlider wraps the real slider; a discrete row is a QSlider.
    slider = control._sliders[axis]
    return getattr(slider, "_slider", slider)


def _handle_center(slider) -> QPoint:
    if hasattr(slider, "_styleOption"):  # superqt's generic slider
        option = slider._styleOption
    else:
        option = QStyleOptionSlider()
        slider.initStyleOption(option)
    rect = slider.style().subControlRect(
        QStyle.ComplexControl.CC_Slider,
        option,
        QStyle.SubControl.SC_SliderHandle,
        slider,
    )
    return rect.center()


def _drag(slider, *, steps: int = 6, dx: int = 8) -> None:
    position = _handle_center(slider)
    QTest.mousePress(
        slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, position
    )
    for _ in range(steps):
        position = QPoint(position.x() + dx, position.y())
        QTest.mouseMove(slider, position, 2)
    QTest.mouseRelease(
        slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, position
    )


@pytest.mark.parametrize("axis", [DISCRETE, CONTINUOUS])
def test_a_drag_emits_begin_ticks_flush_then_end(qtbot, axis):
    control = _make_control(qtbot)
    log = _record(control)
    _drag(_inner_slider(control, axis))

    kinds = [entry[0] for entry in log]
    assert kinds.count("begin") == 1
    assert kinds.count("end") == 1
    # The end is last, and what comes right before it is the flushed tick
    # for the position the slider was released at.
    assert kinds[-1] == "end"
    assert kinds[-2] == "tick"
    final = control._world_value(axis)
    assert log[-2] == ("tick", {axis: final}, True)
    assert final != {DISCRETE: 10.0, CONTINUOUS: 50.0}[axis]
    # Every tick is interactive, and nothing waits in the throttle to land
    # after the release (where it would start a new scrub).
    ticks = [entry for entry in log if entry[0] == "tick"]
    assert len(ticks) >= 2  # the leading edge, and the flush
    assert all(interactive for _kind, _positions, interactive in ticks)
    assert not control._slider_dirty
    assert not control._scope_open
    # The press may come after the first tick (a superqt press nudges the
    # value first); nothing may assume it comes before.
    assert kinds.index("begin") <= 1


def test_the_release_does_not_resubmit_a_position_already_sent(qtbot):
    control = _make_control(qtbot, shown=False)
    log = _record(control)
    slider = control._sliders[DISCRETE]
    slider.setSliderDown(True)  # emits sliderPressed
    slider.setValue(12)  # the leading edge submits it: nothing is pending
    slider.setSliderDown(False)  # emits sliderReleased
    assert log == [("begin",), ("tick", {DISCRETE: 12.0}, True), ("end",)]


def test_a_press_and_release_in_place_opens_and_closes_a_scope(qtbot):
    control = _make_control(qtbot, shown=False)
    log = _record(control)
    slider = control._sliders[DISCRETE]
    slider.setSliderDown(True)
    slider.setSliderDown(False)
    assert log == [("begin",), ("end",)]


def test_a_keyboard_step_ticks_with_no_scope(qtbot):
    control = _make_control(qtbot)
    log = _record(control)
    slider = control._sliders[DISCRETE]
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_Right)
    # Interactive, so it scrubs; with no press it ends on the stillness timer.
    assert log == [("tick", {DISCRETE: 11.0}, True)]


def test_close_ends_an_open_scope(qtbot):
    control = _make_control(qtbot, shown=False)
    log = _record(control)
    closed: list[int] = []
    control.closed.connect(lambda: closed.append(len(log)))
    control._sliders[DISCRETE].setSliderDown(True)
    control.close()  # a control destroyed mid-drag
    assert log == [("begin",), ("end",)]
    assert closed == [2]  # the end went out before the unsubscription
    control.close()
    assert log == [("begin",), ("end",)]


async def test_a_release_ends_the_scrub_through_the_controller(qtbot):
    """Wired to a controller: begin, interactive ticks, then a release end."""
    controller = CellierController()
    try:
        scene = controller.add_scene(
            dim="2d",
            coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
            name="scene",
        )
        control = _make_control(qtbot, scene_id=scene.id, shown=False)
        controller.connect_widget(
            control, subscription_specs=control.subscription_specs()
        )
        events: list[tuple] = []
        controller.on_dims_interaction(
            scene.id,
            lambda event: events.append(
                (event.phase, event.reason, scene.dims.selection.slice_indices[0])
            ),
            owner_id=controller._id,
        )
        slider = control._sliders[DISCRETE]

        slider.setSliderDown(True)
        slider.setValue(12)  # submitted at once (the leading edge)
        slider.setValue(15)  # waits in the throttle
        assert scene.dims.selection.slice_indices[0] == 12.0
        assert controller.dims_interaction_state(scene.id) == "active"
        slider.setSliderDown(False)

        # The scrub ended on release, at the final position, with no timer.
        # (The start saw the scene's position before the first tick.)
        assert events == [("start", None, 0.0), ("end", "release", 15.0)]
        assert controller.dims_interaction_state(scene.id) == "idle"
        assert not controller._deferred_reslice_tasks()
    finally:
        controller.close()
