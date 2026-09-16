"""Trail-window controls: the widgets, the controller seam and the dock wiring.

A graph's ``trail`` lives on the visual rather than on its appearance, so it
has its own update event (``TrailUpdateEvent``) and controller method
(``update_visual_trail``).  Both toolkits' widgets are pinned together here,
since they draw from one vocabulary in ``cellier.gui._trail``.
"""

from __future__ import annotations

import warnings
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier.convenience import AppearanceControls, GraphControlsConfig, Viewer
from cellier.convenience.layout._shared import appearance_specs
from cellier.convenience.layout._walk import render_dock
from cellier.data import GraphMemoryStore, PointsMemoryStore
from cellier.events import TrailChangedEvent, TrailUpdateEvent
from cellier.gui._protocol import WidgetView
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import GraphAppearance, PointsMarkerAppearance, TrailConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _world() -> list:
    return [Axis(name="t", axis_type="time"), *spatial_axes("z", "y", "x")]


def _graph_store(first_axis: tuple[str, str] = ("t", "time")) -> GraphMemoryStore:
    """A three-node chain on ``(t, z, y, x)``; ``t`` holds frame indices.

    *first_axis* is the ``(name, axis_type)`` of the leading axis.
    """
    positions = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    axes = (
        Axis(name=first_axis[0], axis_type=first_axis[1], sampling="discrete"),
        *(Axis(name=name, axis_type="space") for name in ("z", "y", "x")),
    )
    return GraphMemoryStore.from_arrays(
        positions,
        edges,
        data_coordinate_system=DataCoordinateSystem(
            name="tracks", axes=axes, datastore_id=uuid4()
        ),
        name="tracks",
    )


def _add_graph(viewer, *, trail=None, controls=None, store=None):
    return viewer.add_graph(
        data=_graph_store() if store is None else store,
        appearance=GraphAppearance(),
        name="tracks",
        trail=trail,
        controls=controls,
    )


def _trail_events(viewer, visual) -> list:
    events: list = []
    viewer.controller._outgoing_events.subscribe(
        TrailChangedEvent, events.append, entity_id=visual.id, owner_id=uuid4()
    )
    return events


# ---------------------------------------------------------------------------
# The controller seam
# ---------------------------------------------------------------------------


def test_setting_a_window_on_an_axis_without_one_adds_it():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer)

    viewer.controller.update_visual_trail(graph.id, 0, TrailConfig(before=2.0))

    assert set(graph.trail) == {0}
    assert graph.trail[0].before == 2.0


def test_editing_an_existing_window_keeps_the_object_and_reslices_once():
    """In place, so one spin box nudge is one reslice rather than a rebuild."""
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer, trail={0: TrailConfig(before=1.0)})
    window = graph.trail[0]
    reslices: list = []
    original = viewer.controller.reslice_visual
    viewer.controller.reslice_visual = lambda vid: (reslices.append(vid), original(vid))

    viewer.controller.update_visual_trail(graph.id, 0, TrailConfig(before=3.0))

    assert graph.trail[0] is window
    assert window.before == 3.0
    assert reslices.count(graph.id) == 1


def test_none_removes_the_window():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer, trail={0: TrailConfig()})

    viewer.controller.update_visual_trail(graph.id, 0, None)

    assert graph.trail == {}


def test_none_on_an_axis_without_a_window_changes_nothing():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer)
    events = _trail_events(viewer, graph)

    viewer.controller.update_visual_trail(graph.id, 0, None)

    assert graph.trail == {}
    assert events == []


def test_the_change_event_carries_the_callers_source_id():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer, trail={0: TrailConfig()})
    events = _trail_events(viewer, graph)
    source = uuid4()

    viewer.controller.update_visual_trail(
        graph.id, 0, TrailConfig(after=4.0), source_id=source
    )

    assert events
    assert all(event.source_id == source for event in events)


def test_an_out_of_range_axis_raises_a_plain_value_error():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer)

    with pytest.raises(ValueError, match="out of range"):
        viewer.controller.update_visual_trail(graph.id, 9, TrailConfig())


def test_a_visual_without_a_trail_is_refused():
    viewer = Viewer(_world(), gui="offscreen")
    points = viewer.add_points(
        PointsMemoryStore(positions=np.zeros((2, 4), dtype=np.float32)),
        appearance=PointsMarkerAppearance(),
    )

    with pytest.raises(TypeError, match="graph"):
        viewer.controller.update_visual_trail(points.id, 0, TrailConfig())


def test_the_update_event_reaches_the_controller():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer)

    viewer.controller.incoming_events.emit(
        TrailUpdateEvent(
            source_id=uuid4(), visual_id=graph.id, axis=0, config=TrailConfig(after=4.0)
        )
    )

    assert graph.trail[0].after == 4.0


# ---------------------------------------------------------------------------
# The widgets, on both toolkits
# ---------------------------------------------------------------------------


@pytest.fixture(params=["qt", "anywidget"])
def trail_widget(request):
    """``(toolkit, widget class)``."""
    if request.param == "qt":
        pytest.importorskip("qtpy")
        request.getfixturevalue("qtbot")
        from cellier.gui.qt.visuals import QtTrailControls

        return "qt", QtTrailControls
    pytest.importorskip("anywidget")
    from cellier.gui.anywidget.visuals import AnywidgetTrailControls

    return "anywidget", AnywidgetTrailControls


_AXES = [(0, "t"), (1, "z")]


def _seeded(widget_cls, visual_ids=None):
    trail = {0: TrailConfig(before=2.0, after=0.0, fade=True, min_alpha=0.25)}
    widget = widget_cls(visual_ids or [uuid4()], _AXES, trail)
    emitted: list = []
    widget.changed.connect(emitted.append)
    return widget, emitted


def _edit(toolkit: str, widget, axis: int, field: str, value) -> None:
    """Change one control the way a user would."""
    if toolkit == "qt":
        control = widget._controls[axis][field]
        if field in ("enabled", "fade"):
            control.setChecked(value)
        else:
            control.setValue(value)
        return
    windows = dict(widget.windows)
    windows[str(axis)] = {**windows[str(axis)], field: value}
    widget.windows = windows


def _shown(toolkit: str, widget, axis: int, field: str):
    """What the control for *field* on *axis* currently displays."""
    if toolkit == "qt":
        control = widget._controls[axis][field]
        if field in ("enabled", "fade"):
            return control.isChecked()
        return control.value()
    return widget.windows[str(axis)][field]


def test_a_trail_widget_satisfies_the_contract(trail_widget):
    _toolkit, widget_cls = trail_widget
    widget, _emitted = _seeded(widget_cls)

    assert isinstance(widget, WidgetView)


def test_it_seeds_every_axis_from_the_trail(trail_widget):
    toolkit, widget_cls = trail_widget
    widget, _emitted = _seeded(widget_cls)

    assert widget.is_enabled(0)
    assert not widget.is_enabled(1)
    assert _shown(toolkit, widget, 0, "before") == pytest.approx(2.0)
    assert _shown(toolkit, widget, 0, "fade") is True
    assert _shown(toolkit, widget, 1, "enabled") is False


def test_an_edit_emits_the_whole_window_once_per_visual(trail_widget):
    toolkit, widget_cls = trail_widget
    ids = [uuid4(), uuid4()]
    widget, emitted = _seeded(widget_cls, ids)

    _edit(toolkit, widget, 0, "before", 5.0)

    assert [event.visual_id for event in emitted] == ids
    assert all(isinstance(event, TrailUpdateEvent) for event in emitted)
    assert all(event.source_id == widget._id for event in emitted)
    assert all(event.axis == 0 for event in emitted)
    for event in emitted:
        assert event.config.before == 5.0
        # Fields the widget shows and fields it does not both ride along.
        assert event.config.fade is True
        assert event.config.min_alpha == 0.25
    # The controller adopts the object, so visuals must never share one.
    assert emitted[0].config is not emitted[1].config


def test_an_axis_that_is_off_emits_nothing_until_switched_on(trail_widget):
    toolkit, widget_cls = trail_widget
    widget, emitted = _seeded(widget_cls)

    _edit(toolkit, widget, 1, "before", 3.0)
    assert emitted == []

    _edit(toolkit, widget, 1, "enabled", True)

    assert len(emitted) == 1
    assert emitted[0].axis == 1
    assert emitted[0].config.before == 3.0


def test_switching_off_clears_and_switching_on_restores(trail_widget):
    toolkit, widget_cls = trail_widget
    widget, emitted = _seeded(widget_cls)

    _edit(toolkit, widget, 0, "enabled", False)
    assert emitted[-1].config is None
    assert not widget.is_enabled(0)

    _edit(toolkit, widget, 0, "enabled", True)
    assert emitted[-1].config.before == 2.0
    assert emitted[-1].config.fade is True


def test_an_inbound_change_updates_the_widget_without_emitting(trail_widget):
    toolkit, widget_cls = trail_widget
    widget, emitted = _seeded(widget_cls)

    widget._on_trail_changed(
        TrailChangedEvent(
            source_id=uuid4(),
            visual_id=uuid4(),
            trail={1: TrailConfig(after=7.0)},
        )
    )

    assert not widget.is_enabled(0)
    assert widget.is_enabled(1)
    assert widget.window(1).after == 7.0
    assert _shown(toolkit, widget, 1, "after") == pytest.approx(7.0)
    # Switched off from outside, axis 0 still remembers its window.
    assert widget.window(0).before == 2.0
    assert emitted == []


def test_the_widget_ignores_its_own_echo(trail_widget):
    _toolkit, widget_cls = trail_widget
    widget, _emitted = _seeded(widget_cls)

    widget._on_trail_changed(
        TrailChangedEvent(source_id=widget._id, visual_id=uuid4(), trail={})
    )

    assert widget.is_enabled(0)


def test_a_widget_with_no_axes_is_refused(trail_widget):
    _toolkit, widget_cls = trail_widget

    with pytest.raises(ValueError, match="axis"):
        widget_cls([uuid4()], [], {})


def test_the_qt_extents_are_disabled_while_an_axis_is_off(qtbot):
    from cellier.gui.qt.visuals import QtTrailControls

    widget, _emitted = _seeded(QtTrailControls)

    assert widget._controls[0]["before"].isEnabled()
    assert not widget._controls[1]["before"].isEnabled()
    assert not widget._controls[1]["fade"].isEnabled()


# ---------------------------------------------------------------------------
# GraphControlsConfig.trail_controls and the shared spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["t", [True], [1.5]])
def test_a_malformed_trail_controls_value_is_refused(value):
    with pytest.raises(TypeError, match="trail_controls"):
        GraphControlsConfig(appearance=["node_visible"], trail_controls=value)


def _trail_spec(viewer, graph, config):
    store = viewer.controller._model.data.stores[UUID(graph.data_store_id)]
    specs, _skipped = appearance_specs(graph, config, store)
    return next((spec for spec in specs if spec.kind == "trail"), None)


def _config(trail_controls, appearance=("node_visible",)):
    return GraphControlsConfig(
        appearance=list(appearance) if appearance else False,
        trail_controls=trail_controls,
    )


def test_true_offers_the_axes_that_have_a_window():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer, trail={2: TrailConfig()})

    spec = _trail_spec(viewer, graph, _config(True))

    assert spec.values["axes"] == [(2, "y")]
    assert spec.title == "Trail"


def test_true_falls_back_to_the_time_axes():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer)

    spec = _trail_spec(viewer, graph, _config(True))

    assert spec.values["axes"] == [(0, "t")]


def test_true_with_nothing_to_offer_warns_and_builds_nothing():
    # An all-space world, so the graph has no time axis to fall back to.
    viewer = Viewer(spatial_axes("w", "z", "y", "x"), gui="offscreen")
    graph = _add_graph(viewer, store=_graph_store(first_axis=("w", "space")))

    with pytest.warns(UserWarning, match="no axis to offer"):
        spec = _trail_spec(viewer, graph, _config(True))

    assert spec is None


def test_a_list_names_axes_by_name_or_index_in_order():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer)

    spec = _trail_spec(viewer, graph, _config(["z", 0]))

    assert spec.values["axes"] == [(1, "z"), (0, "t")]


@pytest.mark.parametrize(
    ("axes", "message"), [(["w"], "is not one of"), ([7], "out of range")]
)
def test_an_axis_the_store_does_not_have_is_refused(axes, message):
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer)

    with pytest.raises(ValueError, match=message):
        _trail_spec(viewer, graph, _config(axes))


def test_without_appearance_there_is_no_trail_control():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer, trail={0: TrailConfig()})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _trail_spec(viewer, graph, _config(True, appearance=None)) is None


def test_the_spec_holds_copies_of_the_windows():
    viewer = Viewer(_world(), gui="offscreen")
    graph = _add_graph(viewer, trail={0: TrailConfig(before=1.0)})

    spec = _trail_spec(viewer, graph, _config(True))

    assert spec.values["trail"][0].before == 1.0
    assert spec.values["trail"][0] is not graph.trail[0]


# ---------------------------------------------------------------------------
# End to end: a dock builds the control, and it drives the graph
# ---------------------------------------------------------------------------


@pytest.fixture(params=["qt", "jupyter"])
def toolkit(request):
    """``(gui, host class)`` for one toolkit."""
    if request.param == "qt":
        pytest.importorskip("qtpy")
        request.getfixturevalue("qtbot")
        from cellier.convenience._hosts import QtLayoutHost

        return "qt", QtLayoutHost
    pytest.importorskip("anywidget")
    from cellier.convenience._hosts import JupyterHost

    return "anywidget", JupyterHost


def test_a_dock_trail_control_drives_the_graph_and_follows_it(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(_world(), gui=gui)
    graph = _add_graph(
        viewer,
        trail={0: TrailConfig(before=1.0)},
        controls=_config(True),
    )
    closeables: list = []
    render_dock(AppearanceControls(), viewer, host_cls(), closeables)
    (dock,) = closeables
    (widget,) = [w for w in dock.widgets if type(w).__name__.endswith("TrailControls")]
    toolkit_name = "qt" if gui == "qt" else "anywidget"

    _edit(toolkit_name, widget, 0, "before", 4.0)
    assert graph.trail[0].before == 4.0

    graph.trail[0].after = 9.0
    assert widget.window(0).after == 9.0
    assert _shown(toolkit_name, widget, 0, "after") == pytest.approx(9.0)

    _edit(toolkit_name, widget, 0, "enabled", False)
    assert graph.trail == {}

    dock.close()
