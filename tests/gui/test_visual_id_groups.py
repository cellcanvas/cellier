"""The sequence form on the pre-existing appearance and AABB widgets.

Stage 2 step 1 of ``plans/convenience_cleanup.md`` (section 8.3): every widget
that used to hold a single ``self._visual_id`` now accepts a ``UUID`` **or** a
sequence of them and drives the whole group in lock-step.  That is what lets
one dock widget serve an ``OrthoViewer``'s four sibling visuals.

Both toolkits are covered here rather than in two parallel modules, because
the behaviour under test is ``VisualIdGroup``'s and is shared: the widgets
differ only in how the user edit is injected.  The single-id path stays
covered by the existing per-widget test modules, unchanged -- which is the
point of the compatible signature.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.events import AABBUpdateEvent, AppearanceUpdateEvent

qtpy = pytest.importorskip("qtpy")
pytest.importorskip("superqt")


def _ids(n: int = 4) -> list:
    return [uuid4() for _ in range(n)]


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def _qt_widgets(visual_id):
    """One instance of every Qt widget stage 2 widened, with its edit driver.

    Returns ``(name, widget, edit, expected_field)`` where ``edit()`` performs
    a single user-level change.
    """
    from cmap import Colormap

    from cellier.gui.qt.visuals import (
        QtAABBWidget,
        QtClimRangeSlider,
        QtColormapCombo,
        QtLodBiasSlider,
        QtVolumeRenderControls,
    )

    colormap = QtColormapCombo(visual_id, initial_colormap="grays")
    clim = QtClimRangeSlider(visual_id, clim_range=(0.0, 1.0), initial_clim=(0.0, 1.0))
    lod = QtLodBiasSlider(visual_id, initial_lod_bias=1.0)
    render = QtVolumeRenderControls(
        visual_id,
        dtype_max=1.0,
        initial_render_mode="iso",
        initial_threshold=0.2,
    )
    aabb = QtAABBWidget(visual_id)

    return [
        (
            "colormap",
            colormap,
            lambda: colormap._on_combo_changed(Colormap("viridis")),
            "color_map",
        ),
        ("clim", clim, lambda: clim._slider.setValue((0.2, 0.8)), "clim"),
        (
            "lod_bias",
            lod,
            lambda: (lod._slider.setValue(2.5), lod._on_slider_released()),
            "lod_bias",
        ),
        ("render", render, lambda: render._on_combo_changed("mip"), "render_mode"),
        ("aabb", aabb, lambda: aabb._enabled_check.setChecked(True), "enabled"),
    ]


@pytest.mark.parametrize("n_ids", [1, 4])
def test_qt_widgets_emit_one_event_per_visual(qtbot, n_ids):
    visual_ids = _ids(n_ids)

    for name, widget, edit, field in _qt_widgets(visual_ids):
        emitted: list = []
        widget.changed.connect(emitted.append)

        edit()

        matching = [e for e in emitted if e.field == field]
        assert len(matching) == n_ids, f"{name}: {len(matching)} events for {n_ids} ids"
        assert [e.visual_id for e in matching] == visual_ids, name
        # One source id for the whole group, so every echo is filtered by the
        # same check that filters a single-visual widget's one echo.
        assert len({e.source_id for e in matching}) == 1, name


@pytest.mark.parametrize("n_ids", [1, 4])
def test_qt_widgets_subscribe_to_every_visual(qtbot, n_ids):
    visual_ids = _ids(n_ids)

    for name, widget, _edit, _field in _qt_widgets(visual_ids):
        specs = widget.subscription_specs()
        assert len(specs) == n_ids, name
        assert [s.entity_id for s in specs] == visual_ids, name
        # Subscribe-to-all, not subscribe-to-first: a sibling written by
        # something other than this widget must still reach it.
        assert len({s.handler for s in specs}) == 1, name


def test_qt_a_single_uuid_still_works_unchanged(qtbot):
    """The compatible signature: one id in, one event out, one subscription."""
    from cellier.gui.qt.visuals import QtColormapCombo

    visual_id = uuid4()
    widget = QtColormapCombo(visual_id, initial_colormap="grays")

    assert widget.visual_ids == (visual_id,)
    assert widget._visual_id == visual_id
    assert len(widget.subscription_specs()) == 1


def test_qt_aabb_emits_per_visual_for_every_field(qtbot):
    """AABB writes three fields; each must fan out."""
    from cellier.gui.qt.visuals import QtAABBWidget

    visual_ids = _ids(4)
    widget = QtAABBWidget(visual_ids)
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget._enabled_check.setChecked(True)
    widget._line_width_spin.setValue(5.0)

    assert all(isinstance(e, AABBUpdateEvent) for e in emitted)
    for field in ("enabled", "line_width"):
        matching = [e for e in emitted if e.field == field]
        assert [e.visual_id for e in matching] == visual_ids


# ---------------------------------------------------------------------------
# anywidget
# ---------------------------------------------------------------------------


def _anywidget_widgets(visual_id):
    from cellier.gui.anywidget.visuals import (
        AnywidgetAABBWidget,
        AnywidgetClimRangeSlider,
        AnywidgetColormapCombo,
        AnywidgetLodBiasSlider,
        AnywidgetVolumeRenderControls,
    )

    colormap = AnywidgetColormapCombo(visual_id, initial_colormap="grays")
    clim = AnywidgetClimRangeSlider(
        visual_id, clim_range=(0.0, 1.0), initial_clim=(0.0, 1.0)
    )
    lod = AnywidgetLodBiasSlider(visual_id, initial_lod_bias=1.0)
    render = AnywidgetVolumeRenderControls(visual_id)
    aabb = AnywidgetAABBWidget(visual_id)

    def _set(widget, name, value):
        return lambda: setattr(widget, name, value)

    return [
        ("colormap", colormap, _set(colormap, "color_map", "viridis"), "color_map"),
        ("clim", clim, _set(clim, "clim", [0.2, 0.8]), "clim"),
        ("lod_bias", lod, _set(lod, "lod_bias", 2.5), "lod_bias"),
        # Not "mip": that is the trait's initial value, and a no-op write
        # fires no change and would read as a pass (the section 6.4.1 harness
        # lesson, which applies to trait-driven tests too).
        ("render", render, _set(render, "render_mode", "iso"), "render_mode"),
        ("aabb", aabb, _set(aabb, "enabled", True), "enabled"),
    ]


@pytest.mark.parametrize("n_ids", [1, 4])
def test_anywidget_widgets_emit_one_event_per_visual(n_ids):
    visual_ids = _ids(n_ids)

    for name, widget, edit, field in _anywidget_widgets(visual_ids):
        emitted: list = []
        widget.changed.connect(emitted.append)

        edit()

        matching = [e for e in emitted if e.field == field]
        assert len(matching) == n_ids, f"{name}: {len(matching)} events for {n_ids} ids"
        assert [e.visual_id for e in matching] == visual_ids, name
        assert len({e.source_id for e in matching}) == 1, name


@pytest.mark.parametrize("n_ids", [1, 4])
def test_anywidget_widgets_subscribe_to_every_visual(n_ids):
    visual_ids = _ids(n_ids)

    for name, widget, _edit, _field in _anywidget_widgets(visual_ids):
        specs = widget.subscription_specs()
        assert len(specs) == n_ids, name
        assert [s.entity_id for s in specs] == visual_ids, name


def test_anywidget_a_single_uuid_still_works_unchanged():
    from cellier.gui.anywidget.visuals import AnywidgetLodBiasSlider

    visual_id = uuid4()
    widget = AnywidgetLodBiasSlider(visual_id, initial_lod_bias=1.0)

    assert widget.visual_ids == (visual_id,)
    assert widget._visual_id == visual_id
    assert len(widget.subscription_specs()) == 1


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------


def test_an_empty_sequence_is_rejected():
    """A widget driving nothing would silently do nothing on every edit."""
    from cellier.gui.anywidget.visuals import AnywidgetLodBiasSlider

    with pytest.raises(ValueError, match="must not be empty"):
        AnywidgetLodBiasSlider([], initial_lod_bias=1.0)


def test_the_event_type_is_unchanged_by_the_group_form():
    """Fanning out changes how many events, never which kind."""
    from cellier.gui.anywidget.visuals import (
        AnywidgetAABBWidget,
        AnywidgetClimRangeSlider,
    )

    clim = AnywidgetClimRangeSlider(
        _ids(3), clim_range=(0.0, 1.0), initial_clim=(0.0, 1.0)
    )
    aabb = AnywidgetAABBWidget(_ids(3))
    clim_events: list = []
    aabb_events: list = []
    clim.changed.connect(clim_events.append)
    aabb.changed.connect(aabb_events.append)

    clim.clim = [0.1, 0.9]
    aabb.enabled = True

    assert all(isinstance(e, AppearanceUpdateEvent) for e in clim_events)
    assert all(isinstance(e, AABBUpdateEvent) for e in aabb_events)
