"""Tests for the per-visual outline / occlusion / picking widgets.

Both toolkits at once where the behaviour is shared, since they build from
one ``VISUAL_RENDER_CONTROLS`` spec.  The split into three outline-ish
widgets is the thing most worth pinning: ``outline.slot`` means the colour
on a mesh and only participation on a labels visual, so a test that let one
widget serve both would be pinning the bug.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.events import (
    PickWriteChangedEvent,
    RenderConfigChangedEvent,
    VisualRenderChangedEvent,
    VisualRenderUpdateEvent,
)
from cellier.gui._protocol import WidgetView
from cellier.gui._render_controls import (
    VISUAL_RENDER_CONTROLS,
    VISUAL_RENDER_TITLES,
)
from cellier.render._config import OutlineConfig

_PALETTE = OutlineConfig().palette

_OUTLINE_VALUES = {
    "outline.slot": 2,
    "outline.placement": None,
    "default_placement": "inward",
}
_LABELS_VALUES = {
    **_OUTLINE_VALUES,
    "outline_mode": "per_label",
    "outline_selected_labels": {1: 1, 3: 2},
}


def _qt(kind, visual_id):
    from cellier.gui.qt.render import (
        QtLabelsOutlineControls,
        QtVisualOcclusionControls,
        QtVisualOutlineControls,
        QtVisualPickingControls,
    )

    if kind == "visual_outline":
        return QtVisualOutlineControls(
            [visual_id], dict(_OUTLINE_VALUES), palette=_PALETTE
        )
    if kind == "labels_outline":
        return QtLabelsOutlineControls(
            [visual_id], dict(_LABELS_VALUES), palette=_PALETTE
        )
    if kind == "visual_occlusion":
        return QtVisualOcclusionControls([visual_id], {"ambient_occlusion": None})
    return QtVisualPickingControls([visual_id], {"pick_write": True})


def _any(kind, visual_id):
    from cellier.gui.anywidget.render import (
        AnywidgetLabelsOutlineControls,
        AnywidgetVisualOcclusionControls,
        AnywidgetVisualOutlineControls,
        AnywidgetVisualPickingControls,
    )

    if kind == "visual_outline":
        return AnywidgetVisualOutlineControls(
            [visual_id], dict(_OUTLINE_VALUES), palette=_PALETTE
        )
    if kind == "labels_outline":
        return AnywidgetLabelsOutlineControls(
            [visual_id], dict(_LABELS_VALUES), palette=_PALETTE
        )
    if kind == "visual_occlusion":
        return AnywidgetVisualOcclusionControls(
            [visual_id], {"ambient_occlusion": None}
        )
    return AnywidgetVisualPickingControls([visual_id], {"pick_write": True})


_KINDS = ("visual_outline", "labels_outline", "visual_occlusion", "visual_picking")


# ---------------------------------------------------------------------------
# The spec, and parity between the two front ends
# ---------------------------------------------------------------------------


def test_the_spec_covers_every_widget():
    assert set(VISUAL_RENDER_CONTROLS) == set(_KINDS)
    assert set(VISUAL_RENDER_TITLES) == set(_KINDS)


@pytest.mark.parametrize("kind", _KINDS)
def test_every_spec_field_is_settable(kind):
    """A control over an unroutable field looks like it works and then raises."""
    from cellier.controller import VISUAL_RENDER_FIELDS

    for control in VISUAL_RENDER_CONTROLS[kind]:
        assert control.field in VISUAL_RENDER_FIELDS


@pytest.mark.parametrize("kind", _KINDS)
def test_both_front_ends_show_the_same_controls(kind, qtbot):
    pytest.importorskip("anywidget")
    qt_widget = _qt(kind, uuid4())
    qtbot.addWidget(qt_widget.widget)
    any_widget = _any(kind, uuid4())

    assert set(qt_widget._appliers) == {c["field"] for c in any_widget.controls}


@pytest.mark.parametrize("kind", _KINDS)
def test_both_front_ends_agree_on_the_title(kind, qtbot):
    pytest.importorskip("anywidget")
    qt_widget = _qt(kind, uuid4())
    qtbot.addWidget(qt_widget.widget)
    assert qt_widget.DEFAULT_TITLE == VISUAL_RENDER_TITLES[kind]
    assert _any(kind, uuid4()).DEFAULT_TITLE == VISUAL_RENDER_TITLES[kind]


# ---------------------------------------------------------------------------
# The split that motivates three widgets
# ---------------------------------------------------------------------------


def test_a_labels_visual_gets_no_slot_swatches(qtbot):
    """``outline.slot`` picks no colour on a labels visual.

    Its selection colour comes from ``outline_selected_labels``, per label,
    so a swatch picker over the visual's own slot would let a user choose a
    colour that does nothing.
    """
    labels = _qt("labels_outline", uuid4())
    qtbot.addWidget(labels.widget)
    plain = _qt("visual_outline", uuid4())
    qtbot.addWidget(plain.widget)

    assert not hasattr(labels, "_swatch_group")
    assert hasattr(plain, "_swatch_group")
    assert "outline_selected_labels" in labels._appliers
    assert "outline_selected_labels" not in plain._appliers


# ---------------------------------------------------------------------------
# Driving the real controls
#
# Every test in this section clicks or sets an actual widget rather than
# calling the method behind it.  Two bugs got past the first round because
# the tests took the shorter route: a user's own edit never reaches the
# applier -- the echo of it comes back stamped with the widget's own source
# id and is discarded, correctly, as its own -- so anything wired only to
# the applier silently did not happen.
# ---------------------------------------------------------------------------


def _combo(widget, index: int = 0):
    from PySide6.QtWidgets import QComboBox

    return widget.widget.findChildren(QComboBox)[index]


def test_picking_a_swatch_moves_the_chosen_marking(qtbot):
    """The marking is the only feedback for which slot is selected.

    An explicit background colour hides Qt's own checked indicator, so the
    heavy border does that job -- and it has to move on the user's click,
    not only on a change arriving from elsewhere.
    """
    widget = _qt("visual_outline", uuid4())
    qtbot.addWidget(widget.widget)

    def chosen() -> list[int]:
        return [
            slot
            for slot, button, rgba in widget._swatch_buttons
            if rgba is not None and "2px solid black" in button.styleSheet()
        ]

    assert chosen() == [2]

    widget._swatch_group.buttons()[3].click()  # slot 3

    assert widget._values["outline.slot"] == 3
    assert chosen() == [3], "the marking stayed on the old slot"


def test_picking_off_clears_the_chosen_marking(qtbot):
    widget = _qt("visual_outline", uuid4())
    qtbot.addWidget(widget.widget)

    widget._swatch_group.buttons()[0].click()  # Off

    assert widget._values["outline.slot"] == 0
    assert not any(
        rgba is not None and "2px solid black" in button.styleSheet()
        for _slot, button, rgba in widget._swatch_buttons
    )


def test_choosing_whole_volume_reveals_the_swatch_row(qtbot):
    """The bug this section exists for.

    Selecting whole-volume mode left the panel showing the per-label
    checkbox, so there was no way to set the outline colour -- the model
    was in one mode and the panel in the other.
    """
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    assert not hasattr(widget, "_swatch_group") or not widget._swatch_group.buttons()

    _combo(widget).setCurrentIndex(1)  # Whole volume

    assert widget._values["outline_mode"] == "whole_object"
    assert widget._swatch_group.buttons(), "no swatch row after switching mode"
    assert "outline_selected_labels" not in widget._appliers


def test_choosing_all_boundaries_reveals_the_swatch_row(qtbot):
    """All boundaries takes its colour from the slot, exactly as whole volume.

    Same shape as the whole-volume case above: the panel must offer a
    colour, not a participation checkbox, or the mode has no way to say
    what colour to band every label in.
    """
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)

    _combo(widget).setCurrentIndex(2)  # All boundaries

    assert widget._values["outline_mode"] == "all_boundaries"
    assert widget._swatch_group.buttons(), "no swatch row after switching mode"
    assert "outline_selected_labels" not in widget._appliers


def test_choosing_per_label_brings_the_rows_back(qtbot):
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    _combo(widget).setCurrentIndex(1)

    _combo(widget).setCurrentIndex(0)  # Per label

    assert widget._values["outline_mode"] == "per_label"
    assert "outline_selected_labels" in widget._appliers


def test_a_swatch_in_whole_volume_mode_sets_the_slot(qtbot):
    """End of the chain: the colour a user picks reaches the model."""
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)
    _combo(widget).setCurrentIndex(1)

    widget._swatch_group.buttons()[2].click()  # slot 2

    assert emitted[-1].field == "outline.slot"
    assert emitted[-1].value == 2


def test_the_labels_panel_reshapes_with_its_mode(qtbot):
    """``outline.slot`` means two things, so it gets two controls.

    In whole-volume mode it chooses the colour, exactly as on a mesh, so it
    gets the swatch row and the per-label rows go away.  In per-label mode
    it only decides participation, so it gets a checkbox and the rows come
    back -- a swatch there would let a user pick a colour the rows below
    immediately override.
    """
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    assert "outline_selected_labels" in widget._appliers

    widget._values["outline_mode"] = "whole_object"
    widget._build_mode_section()

    assert "outline_selected_labels" not in widget._appliers
    assert "outline.slot" in widget._appliers

    widget._values["outline_mode"] = "all_boundaries"
    widget._build_mode_section()

    assert "outline_selected_labels" not in widget._appliers
    assert "outline.slot" in widget._appliers
    assert widget._swatch_group.buttons()

    widget._values["outline_mode"] = "per_label"
    widget._build_mode_section()

    assert "outline_selected_labels" in widget._appliers


def test_the_anywidget_labels_panel_reshapes_with_its_mode():
    pytest.importorskip("anywidget")
    widget = _any("labels_outline", uuid4())
    kinds = {c["field"]: c["kind"] for c in widget.controls}
    assert kinds["outline.slot"] == "bool"
    assert "outline_selected_labels" in kinds

    widget.values = {**widget.values, "outline_mode": "whole_object"}

    kinds = {c["field"]: c["kind"] for c in widget.controls}
    assert kinds["outline.slot"] == "slot"
    assert "outline_selected_labels" not in kinds

    widget.values = {**widget.values, "outline_mode": "all_boundaries"}

    kinds = {c["field"]: c["kind"] for c in widget.controls}
    assert kinds["outline.slot"] == "slot"
    assert "outline_selected_labels" not in kinds


def test_an_inbound_mode_change_reshapes_the_panel(qtbot):
    """A mode set from code has to reshape the panel too, not just the model."""
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)

    widget._on_visual_render_changed(
        VisualRenderChangedEvent(
            source_id=uuid4(),
            visual_id=uuid4(),
            field_name="outline_mode",
            new_value="whole_object",
        )
    )

    assert "outline_selected_labels" not in widget._appliers


def test_the_labels_toggle_writes_a_slot_not_a_bool(qtbot):
    """The checkbox drives the same integer field, as 0 or 1."""
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)

    checkbox = widget._appliers["outline.slot"].__closure__[0].cell_contents
    checkbox.setChecked(False)

    assert emitted[-1].field == "outline.slot"
    assert emitted[-1].value == 0


# ---------------------------------------------------------------------------
# The bus contract, both toolkits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _KINDS)
def test_a_qt_widget_satisfies_the_contract(kind, qtbot):
    widget = _qt(kind, uuid4())
    qtbot.addWidget(widget.widget)
    assert isinstance(widget, WidgetView)
    assert widget.subscription_specs()


@pytest.mark.parametrize("kind", _KINDS)
def test_an_anywidget_satisfies_the_contract(kind):
    pytest.importorskip("anywidget")
    widget = _any(kind, uuid4())
    assert isinstance(widget, WidgetView)
    assert widget.subscription_specs()


@pytest.mark.parametrize(
    "kind", ("visual_outline", "labels_outline", "visual_occlusion")
)
def test_the_qt_subscription_is_keyed_by_visual(kind, qtbot):
    """Unlike the global panels, these do have an entity to filter on."""
    visual_id = uuid4()
    widget = _qt(kind, visual_id)
    qtbot.addWidget(widget.widget)
    render_specs = [
        s
        for s in widget.subscription_specs()
        if s.event_type is VisualRenderChangedEvent
    ]
    assert [s.entity_id for s in render_specs] == [visual_id]


def test_the_picking_widget_follows_its_own_event(qtbot):
    """``pick_write`` already had an outgoing event; one field, one event."""
    visual_id = uuid4()
    widget = _qt("visual_picking", visual_id)
    qtbot.addWidget(widget.widget)
    (spec,) = widget.subscription_specs()
    assert spec.event_type is PickWriteChangedEvent
    assert spec.entity_id == visual_id


def test_the_picking_widget_applies_its_own_event(qtbot):
    visual_id = uuid4()
    widget = _qt("visual_picking", visual_id)
    qtbot.addWidget(widget.widget)
    checkbox = widget._appliers["pick_write"].__closure__[0].cell_contents

    widget._on_pick_write_changed(
        PickWriteChangedEvent(source_id=uuid4(), visual_id=visual_id, pick_write=False)
    )

    assert checkbox.isChecked() is False


@pytest.mark.parametrize("kind", ("visual_outline", "visual_occlusion"))
def test_a_qt_edit_emits_for_every_driven_visual(kind, qtbot):
    """An ``OrthoViewer`` drives four panel siblings from one control."""
    ids = [uuid4(), uuid4()]
    widget = _qt(kind, ids[0])
    widget._visual_ids = ids
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)

    widget._emit(
        "ambient_occlusion" if kind == "visual_occlusion" else "outline.slot", 1
    )

    assert [e.visual_id for e in emitted] == ids


@pytest.mark.parametrize(
    "kind", ("visual_outline", "labels_outline", "visual_occlusion")
)
def test_a_qt_widget_ignores_its_own_echo(kind, qtbot):
    widget = _qt(kind, uuid4())
    qtbot.addWidget(widget.widget)
    applied: list = []
    widget._appliers[
        "outline.placement" if "outline" in kind else "ambient_occlusion"
    ] = applied.append

    widget._on_visual_render_changed(
        VisualRenderChangedEvent(
            source_id=widget._id,
            visual_id=uuid4(),
            field_name="outline.placement"
            if "outline" in kind
            else "ambient_occlusion",
            new_value="outward",
        )
    )

    assert applied == []


# ---------------------------------------------------------------------------
# Following the palette
# ---------------------------------------------------------------------------


def test_the_qt_swatches_follow_the_palette(qtbot):
    """The slots on offer are the palette's, so they must track it.

    This is what makes the control unable to choose a slot the palette
    cannot colour -- the failure the "no palette entry" warning covers when
    the choice is made in code instead.
    """
    widget = _qt("visual_outline", uuid4())
    qtbot.addWidget(widget.widget)
    assert len(widget._palette) == len(_PALETTE)

    grown = [*_PALETTE, (0.1, 0.2, 0.3, 1.0)]
    widget._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section="outline",
            config=OutlineConfig(palette=grown),
            field_name="palette",
            new_value=grown,
        )
    )

    assert len(widget._palette) == len(grown)


def test_the_qt_swatches_ignore_another_section(qtbot):
    from cellier.render._config import AmbientOcclusionConfig

    widget = _qt("visual_outline", uuid4())
    qtbot.addWidget(widget.widget)

    widget._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section="ambient_occlusion",
            config=AmbientOcclusionConfig(),
            field_name="power",
            new_value=2.0,
        )
    )

    assert len(widget._palette) == len(_PALETTE)


def test_the_anywidget_swatches_follow_the_palette():
    pytest.importorskip("anywidget")
    widget = _any("visual_outline", uuid4())
    grown = [*_PALETTE, (0.1, 0.2, 0.3, 1.0)]

    widget._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section="outline",
            config=OutlineConfig(palette=grown),
            field_name="palette",
            new_value=grown,
        )
    )

    assert len(widget.palette) == len(grown)


# ---------------------------------------------------------------------------
# The label row editor
# ---------------------------------------------------------------------------


def test_adding_a_label_row_extends_the_selection(qtbot):
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)

    widget._on_add_row()

    assert emitted[-1].field == "outline_selected_labels"
    assert emitted[-1].value == {1: 1, 3: 2, 4: 1}


def test_removing_a_label_row_shrinks_the_selection(qtbot):
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)

    widget._remove_row(3)

    assert emitted[-1].value == {1: 1}


def test_cycling_a_row_walks_the_palette(qtbot):
    """One click per slot, wrapping -- a menu per row would dominate the panel."""
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)

    widget._cycle_slot(1)
    assert emitted[-1].value[1] == 2

    for _ in range(len(_PALETTE) - 1):
        widget._cycle_slot(1)
    assert emitted[-1].value[1] == 1, "did not wrap at the end of the palette"


def test_renaming_a_label_keeps_its_slot(qtbot):
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)

    widget._on_label_renamed(3, 7)

    assert emitted[-1].value == {1: 1, 7: 2}


def test_renaming_onto_an_existing_label_is_refused(qtbot):
    """Two rows for one label would make the map lossy."""
    widget = _qt("labels_outline", uuid4())
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)

    widget._on_label_renamed(3, 1)

    assert emitted == []


# ---------------------------------------------------------------------------
# Depth threshold: a fourth mode, and the row for its depth range
# ---------------------------------------------------------------------------


def _range_boxes(widget):
    return widget._depth_range_boxes


def _labels_qt(qtbot, **values):
    from cellier.gui.qt.render import QtLabelsOutlineControls

    widget = QtLabelsOutlineControls(
        [uuid4()], {**_LABELS_VALUES, **values}, palette=_PALETTE
    )
    qtbot.addWidget(widget.widget)
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)
    return widget, emitted


def _labels_any(**values):
    from cellier.gui.anywidget.render import AnywidgetLabelsOutlineControls

    widget = AnywidgetLabelsOutlineControls(
        [uuid4()], {**_LABELS_VALUES, **values}, palette=_PALETTE
    )
    emitted: list[VisualRenderUpdateEvent] = []
    widget.changed.connect(emitted.append)
    return widget, emitted


def _foreign(field, value):
    return VisualRenderChangedEvent(
        source_id=uuid4(), visual_id=uuid4(), field_name=field, new_value=value
    )


def test_depth_threshold_is_the_fourth_mode_of_the_menu():
    """Appended, not inserted: the tests above drive the menu by index."""
    from cellier.gui._render_controls import OUTLINE_MODE_CHOICES

    assert [value for _label, value in OUTLINE_MODE_CHOICES] == [
        "per_label",
        "whole_object",
        "all_boundaries",
        "depth_threshold",
    ]
    assert OUTLINE_MODE_CHOICES[3][0] == "Depth threshold"


def test_choosing_depth_threshold_reveals_the_swatches_and_the_range(qtbot):
    """The slot is the colour in this mode, and the range row appears."""
    widget, emitted = _labels_qt(qtbot)
    assert "outline_depth_range" not in widget._appliers

    _combo(widget).setCurrentIndex(3)  # Depth threshold

    assert emitted[-1].field == "outline_mode"
    assert emitted[-1].value == "depth_threshold"
    assert widget._swatch_group.buttons()
    assert "outline_selected_labels" not in widget._appliers
    assert "outline_depth_range" in widget._appliers
    # No range was given, so the row shows the model's default.
    start_box, solid_box = _range_boxes(widget)
    assert (start_box.value(), solid_box.value()) == (0.01, 0.06)
    assert start_box.decimals() == 3
    assert start_box.minimum() == 0.001


@pytest.mark.parametrize("index", [0, 1, 2])
def test_leaving_depth_threshold_removes_the_range_row(qtbot, index):
    widget, _emitted = _labels_qt(qtbot, outline_mode="depth_threshold")
    assert "outline_depth_range" in widget._appliers

    _combo(widget).setCurrentIndex(index)

    assert "outline_depth_range" not in widget._appliers


def test_the_range_row_shows_the_visuals_range(qtbot):
    widget, _emitted = _labels_qt(
        qtbot, outline_mode="depth_threshold", outline_depth_range=(0.02, 0.12)
    )
    start_box, solid_box = _range_boxes(widget)
    assert (start_box.value(), solid_box.value()) == (0.02, 0.12)


def test_editing_a_box_sends_the_whole_pair(qtbot):
    """One field, one value: the model never sees half a change."""
    widget, emitted = _labels_qt(qtbot, outline_mode="depth_threshold")
    start_box, solid_box = _range_boxes(widget)

    solid_box.setValue(0.2)
    assert (emitted[-1].field, emitted[-1].value) == (
        "outline_depth_range",
        (0.01, 0.2),
    )

    start_box.setValue(0.05)
    assert emitted[-1].value == (0.05, 0.2)
    assert len(emitted) == 2


def test_raising_the_start_past_the_solid_value_carries_it_up(qtbot):
    widget, emitted = _labels_qt(qtbot, outline_mode="depth_threshold")
    start_box, solid_box = _range_boxes(widget)

    start_box.setValue(0.3)

    assert emitted[-1].value == (0.3, 0.3)
    assert solid_box.value() == 0.3


def test_the_solid_value_cannot_be_set_below_the_start(qtbot):
    widget, emitted = _labels_qt(qtbot, outline_mode="depth_threshold")
    start_box, solid_box = _range_boxes(widget)

    solid_box.setValue(0.004)

    assert emitted[-1].value == (0.01, 0.01)
    assert solid_box.value() == 0.01
    assert start_box.value() == 0.01


def test_an_inbound_range_updates_the_boxes_without_an_echo(qtbot):
    widget, emitted = _labels_qt(qtbot, outline_mode="depth_threshold")

    widget._on_visual_render_changed(_foreign("outline_depth_range", (0.03, 0.09)))

    start_box, solid_box = _range_boxes(widget)
    assert (start_box.value(), solid_box.value()) == (0.03, 0.09)
    assert emitted == []


def test_a_range_set_while_the_row_is_hidden_is_shown_on_entering_the_mode(qtbot):
    """The range is kept in the other modes, so the panel has to keep it too."""
    widget, _emitted = _labels_qt(qtbot)

    widget._on_visual_render_changed(_foreign("outline_depth_range", (0.03, 0.09)))
    widget._on_visual_render_changed(_foreign("outline_mode", "depth_threshold"))

    start_box, solid_box = _range_boxes(widget)
    assert (start_box.value(), solid_box.value()) == (0.03, 0.09)


def test_both_front_ends_show_the_same_controls_in_depth_threshold(qtbot):
    pytest.importorskip("anywidget")
    qt_widget, _e = _labels_qt(qtbot, outline_mode="depth_threshold")
    any_widget, _e = _labels_any(outline_mode="depth_threshold")

    fields = {c["field"] for c in any_widget.controls}
    assert set(qt_widget._appliers) == fields
    assert "outline_depth_range" in fields


def test_the_anywidget_range_row_follows_the_mode():
    pytest.importorskip("anywidget")
    widget, emitted = _labels_any()
    assert "outline_depth_range" not in {c["field"] for c in widget.controls}
    # The default is there for the script to show when the row appears.
    assert widget.values["outline_depth_range"] == [0.01, 0.06]

    widget.values = {**widget.values, "outline_mode": "depth_threshold"}

    by_field = {c["field"]: c for c in widget.controls}
    assert by_field["outline.slot"]["kind"] == "slot"
    assert "outline_selected_labels" not in by_field
    entry = by_field["outline_depth_range"]
    assert entry["kind"] == "range"
    assert entry["labels"] == ["Start", "Solid"]
    assert (entry["minimum"], entry["first_maximum"], entry["decimals"]) == (
        0.001,
        1.0,
        3,
    )
    assert [(e.field, e.value) for e in emitted] == [
        ("outline_mode", "depth_threshold")
    ]

    widget.values = {**widget.values, "outline_mode": "all_boundaries"}
    assert "outline_depth_range" not in {c["field"] for c in widget.controls}


def test_the_anywidget_range_is_sent_as_a_pair():
    pytest.importorskip("anywidget")
    widget, emitted = _labels_any(outline_mode="depth_threshold")

    widget.values = {**widget.values, "outline_depth_range": [0.02, 0.12]}

    assert [(e.field, e.value) for e in emitted] == [
        ("outline_depth_range", (0.02, 0.12))
    ]
    assert isinstance(emitted[-1].value, tuple)


def test_the_anywidget_never_sends_a_pair_out_of_order():
    """The script keeps the pair in order; Python does not rely on it."""
    pytest.importorskip("anywidget")
    widget, emitted = _labels_any(outline_mode="depth_threshold")

    widget.values = {**widget.values, "outline_depth_range": [0.3, 0.06]}

    assert emitted[-1].value == (0.3, 0.3)
    assert widget.values["outline_depth_range"] == [0.3, 0.3]
    assert len(emitted) == 1

    widget.values = {**widget.values, "outline_depth_range": ["x", 0.06]}
    assert len(emitted) == 1


def test_an_inbound_range_reaches_the_anywidget_in_any_mode():
    pytest.importorskip("anywidget")
    for mode in ("per_label", "depth_threshold"):
        widget, emitted = _labels_any(outline_mode=mode)

        widget._on_visual_render_changed(_foreign("outline_depth_range", (0.03, 0.09)))

        assert widget.values["outline_depth_range"] == [0.03, 0.09]
        assert emitted == []


def test_the_range_coercion_is_what_the_model_accepts():
    """Whatever the boxes say, the pair the row holds is a valid range."""
    from cellier.gui._render_controls import coerce_depth_range
    from cellier.visuals._label_memory import validate_outline_depth_range

    for start, solid in [(0.0, 0.0), (-1.0, 5.0), (0.5, 0.1), (3.0, 0.2), (0.01, 99)]:
        pair = coerce_depth_range(start, solid)
        assert validate_outline_depth_range(pair) == pair
    assert coerce_depth_range(0.02, 0.12) == (0.02, 0.12)
