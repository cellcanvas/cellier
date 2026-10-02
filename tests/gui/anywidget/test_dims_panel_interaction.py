"""The anywidget dims panel's press and release messages (tracker design 4.8).

The browser sends ``{"type": "interaction", "phase": "begin"}`` on a slider's
``pointerdown`` and ``{"type": "interaction", "phase": "end",
"slice_indices": {...}}`` on its release.  The end **carries the final
position**: a custom message can overtake the traitlet sync sent before it
(Phase 0 D2), so Python applies the position itself before ending the scrub.

The browser side is stood in for the way ``tests/v2/test_anywidget.py`` does
it: a trait assignment is a model sync, and ``_handle_custom_msg`` is a
message arriving.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier.controller import CellierController
from cellier.events import DimsInteractionUpdateEvent, DimsUpdateEvent
from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel
from cellier.scene.dims import spatial_axes


def _make_panel(scene_id=None) -> AnywidgetDimsPanel:
    return AnywidgetDimsPanel(
        scene_id=scene_id or uuid4(),
        axis_values={
            0: DiscreteAxisValues(values=tuple(float(i) for i in range(20))),
            1: ContinuousAxisValues(min=0.0, max=100.0),
        },
        axis_labels={0: "t", 1: "z"},
        slice_indices={0: 0.0, 1: 0.0},
        displayed_axes=(),
    )


def _record(panel) -> list[tuple]:
    """``("begin",)``, ``("end",)`` and ``("tick", positions, interactive)``."""
    log: list[tuple] = []

    def on_changed(event) -> None:
        if isinstance(event, DimsInteractionUpdateEvent):
            assert event.source_id == panel._id
            log.append((event.phase,))
        elif isinstance(event, DimsUpdateEvent):
            log.append(("tick", dict(event.slice_indices), event.interactive))

    panel.changed.connect(on_changed)
    return log


def _message(panel, phase: str, positions: dict | None = None) -> None:
    content = {"type": "interaction", "phase": phase}
    if positions is not None:
        content["slice_indices"] = positions
    panel._handle_custom_msg(content, [])


def _sync(panel, positions: dict) -> None:
    """A ``slice_indices`` sync from the browser."""
    panel.slice_indices = {**panel.slice_indices, **positions}


def test_a_press_opens_one_scope():
    panel = _make_panel()
    log = _record(panel)
    _message(panel, "begin")
    _message(panel, "begin")
    assert log == [("begin",)]


def test_a_drag_is_begin_interactive_ticks_then_end():
    panel = _make_panel()
    log = _record(panel)
    _message(panel, "begin")
    _sync(panel, {"0": 3.0})
    _sync(panel, {"0": 5.0})
    _message(panel, "end", {"0": 5.0})  # Python has seen 5 already
    assert log == [
        ("begin",),
        ("tick", {0: 3.0}, True),
        ("tick", {0: 5.0}, True),
        ("end",),
    ]


def test_the_end_applies_a_position_python_has_not_seen_yet():
    """The end message overtook the sync of the final position."""
    panel = _make_panel()
    log = _record(panel)
    _message(panel, "begin")
    _sync(panel, {"0": 3.0})
    _message(panel, "end", {"0": 7.0})
    # Applied as an interactive tick, before the scope closes.
    assert log[-2:] == [("tick", {0: 7.0}, True), ("end",)]
    assert panel.slice_indices["0"] == 7.0
    assert panel.discrete_index["0"] == 7

    # The overtaken sync arrives now: the position is already applied, so it
    # moves nothing and is not a tick.
    before = list(log)
    _sync(panel, {"0": 7.0})
    assert log == before


def test_a_begin_after_the_first_tick_is_harmless():
    panel = _make_panel()
    log = _record(panel)
    _sync(panel, {"1": 2.5})
    _message(panel, "begin")
    _message(panel, "end", {"1": 2.5})
    assert log == [("tick", {1: 2.5}, True), ("begin",), ("end",)]


def test_an_end_with_no_scope_applies_its_position_and_ends_nothing():
    """A keyboard step on a discrete row fires ``change`` with no press."""
    panel = _make_panel()
    log = _record(panel)
    _message(panel, "end", {"0": 4.0})
    assert log == [("tick", {0: 4.0}, True)]


def test_an_end_with_no_position_just_closes_the_scope():
    panel = _make_panel()
    log = _record(panel)
    _message(panel, "begin")
    _message(panel, "end")
    assert log == [("begin",), ("end",)]


def test_other_messages_are_ignored():
    panel = _make_panel()
    log = _record(panel)
    panel._handle_custom_msg({"type": "something-else"}, [])
    panel._handle_custom_msg("not a dict", [])
    panel._handle_custom_msg({"type": "interaction", "phase": "sideways"}, [])
    assert log == []


def test_close_ends_an_open_scope():
    """A panel destroyed mid-drag (a dropped comm) must not leak its scope."""
    panel = _make_panel()
    log = _record(panel)
    _message(panel, "begin")
    panel.close()
    assert log == [("begin",), ("end",)]


async def test_a_release_ends_the_scrub_at_the_final_position():
    """Wired to a controller, with the end message overtaking the last sync."""
    controller = CellierController(gui="offscreen")
    try:
        scene = controller.add_scene(
            dim="2d",
            coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
            name="scene",
        )
        panel = _make_panel(scene.id)
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())
        events: list[tuple] = []
        controller.on_dims_interaction(
            scene.id,
            lambda event: events.append(
                (event.phase, event.reason, scene.dims.selection.slice_indices[0])
            ),
            owner_id=controller._id,
        )

        _message(panel, "begin")
        _sync(panel, {"0": 3.0})
        assert controller.dims_interaction_state(scene.id) == "active"
        _message(panel, "end", {"0": 7.0})

        assert events == [("start", None, 0.0), ("end", "release", 7.0)]
        assert not controller._deferred_reslice_tasks()
        _sync(panel, {"0": 7.0})  # the late sync: not a tick, no new scrub
        assert controller.dims_interaction_state(scene.id) == "idle"
        assert len(events) == 2
        panel.close()
    finally:
        controller.close()
