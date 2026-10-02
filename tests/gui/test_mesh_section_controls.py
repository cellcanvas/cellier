"""The mesh "2D section" control on both toolkits (plan 5.11).

The control sends ``MeshSectionUpdateEvent`` and follows
``MeshSectionChangedEvent``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.gui._mesh_section import MESH_SECTION_TITLE, mesh_section_fields
from cellier.visuals import MeshFlatAppearance, MeshSectionConfig
from tests._meshes import uv_sphere

_SERIALS = itertools.count()
_QTBOT: list = []


@pytest.fixture(autouse=True)
def _own_qt_controls(qtbot):
    """Give every Qt control a test builds to ``qtbot``, which closes it."""
    _QTBOT.append(qtbot)
    yield
    _QTBOT.clear()


def _owned(control):
    """Register a Qt control's widget for closing; anywidgets pass through.

    A control built outside a layout is a parentless Qt widget: nothing
    closes or deletes it when the test ends unless ``qtbot`` is told of it.
    (The anywidget ones are closed by the ``_close_cellier_objects`` fixture.)
    """
    if not hasattr(control, "comm"):
        _QTBOT[-1].addWidget(control.widget)
    return control


def _make(toolkit, visual_ids, section):
    if toolkit == "qt":
        from cellier.gui.qt.visuals import QtMeshSectionControls

        return _owned(QtMeshSectionControls(visual_ids, section=section.model_dump()))
    from cellier.gui.anywidget.visuals import AnywidgetMeshSectionControls

    return AnywidgetMeshSectionControls(visual_ids, section=section.model_dump())


def _user_edit(widget, field, value) -> None:
    """Change one control as a user would."""
    if hasattr(widget, "input"):  # Qt
        control = widget.input(field)
        if hasattr(control, "setChecked"):
            control.setChecked(value)
        elif hasattr(control, "setCurrentText"):
            control.setCurrentText(value)
        else:
            control.setValue(value)
    else:  # anywidget: what the front end sets, a fresh serial per edit
        widget.edit = {"field": field, "value": value, "serial": next(_SERIALS)}


def _shown(widget) -> dict:
    if hasattr(widget, "input"):
        out = {}
        for name in MeshSectionConfig.model_fields:
            control = widget.input(name)
            if hasattr(control, "isChecked"):
                out[name] = control.isChecked()
            elif hasattr(control, "currentText"):
                out[name] = control.currentText()
            else:
                out[name] = control.value()
        return out
    return dict(widget.config)


@pytest.fixture
def controller(qtbot):
    controller = CellierController(gui="offscreen")
    yield controller
    controller.close()


def _add_mesh(controller, name="mesh"):
    scene = controller.add_scene(dim="2d", name=f"scene-{name}")
    positions, indices = uv_sphere(5.0, (8.0, 8.0, 8.0))
    visual = controller.add_mesh(
        MeshMemoryStore(positions=positions, indices=indices, name=name),
        scene.id,
        MeshFlatAppearance(),
        name,
    )
    return scene, visual


def test_the_rows_are_read_off_the_model():
    fields = {field.name: field for field in mesh_section_fields()}
    assert list(fields) == ["outline", "fill", "outline_width", "mode"]
    assert set(fields) == set(MeshSectionConfig.model_fields)
    assert fields["outline"].kind == fields["fill"].kind == "bool"
    assert fields["mode"].choices == ("cut", "slab")
    assert fields["outline_width"].minimum > 0


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_it_shows_the_config_and_edits_it(controller, toolkit):
    _scene, visual = _add_mesh(controller)
    widget = _make(toolkit, [visual.id], visual.section)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    assert _shown(widget) == {
        "mode": "cut",
        "outline": True,
        "fill": True,
        "outline_width": 2.0,
    }
    _user_edit(widget, "fill", False)
    assert visual.section.fill is False
    _user_edit(widget, "mode", "slab")
    assert visual.section.mode == "slab"
    _user_edit(widget, "outline_width", 5.0)
    assert visual.section.outline_width == pytest.approx(5.0)
    assert widget.error == ""

    # A change from elsewhere is shown.
    controller.update_section_field(visual.id, "outline", False)
    assert _shown(widget)["outline"] is False
    assert _shown(widget)["fill"] is False

    widget.close()


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_refused_edit_shows_the_config_again_and_why(controller, toolkit):
    _scene, visual = _add_mesh(controller)
    widget = _make(toolkit, [visual.id], visual.section)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    # Sent as the front end would send a value its own range does not allow.
    widget._editor.edit("outline_width", -1.0)
    assert visual.section.outline_width == pytest.approx(2.0)
    assert _shown(widget)["outline_width"] == pytest.approx(2.0)
    assert widget.error != ""
    widget.close()


def test_a_group_control_edits_every_visual(controller):
    _s1, first = _add_mesh(controller, "a")
    _s2, second = _add_mesh(controller, "b")
    widget = _make("qt", [first.id, second.id], first.section)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _user_edit(widget, "outline", False)
    assert first.section.outline is False
    assert second.section.outline is False


def test_both_toolkits_carry_the_shared_title():
    from cellier.gui.anywidget.visuals import AnywidgetMeshSectionControls
    from cellier.gui.qt.visuals import QtMeshSectionControls

    assert QtMeshSectionControls.DEFAULT_TITLE == MESH_SECTION_TITLE == "2D section"
    assert AnywidgetMeshSectionControls.DEFAULT_TITLE == MESH_SECTION_TITLE


# -- the panel ------------------------------------------------------------------


def test_the_panel_offers_it_only_when_asked(controller):
    from cellier.convenience import MeshControlsConfig
    from cellier.convenience.layout._shared import appearance_specs

    _scene, visual = _add_mesh(controller)

    def kinds(config):
        return [spec.kind for spec in appearance_specs(visual, config).specs]

    assert "mesh_section" not in kinds(MeshControlsConfig(appearance=True))
    asked = MeshControlsConfig(appearance=True, section_controls=True)
    assert "mesh_section" in kinds(asked)
    spec = next(
        spec
        for spec in appearance_specs(visual, asked).specs
        if spec.kind == "mesh_section"
    )
    assert spec.title == "2D section"
    assert spec.values["section"] == visual.section.model_dump()


def test_the_viewers_take_a_section(qtbot):
    from cellier.convenience import OrthoViewer, Viewer
    from cellier.scene.dims import spatial_axes

    positions, indices = uv_sphere(5.0, (8.0, 8.0, 8.0))
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="2d", gui="offscreen")
    visual = viewer.add_mesh(
        MeshMemoryStore(positions=positions, indices=indices),
        MeshFlatAppearance(),
        section=MeshSectionConfig(fill=False),
    )
    assert visual.section.fill is False
    viewer.controller.close()

    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    section = MeshSectionConfig(mode="slab")
    visuals = ortho.add_mesh(
        MeshMemoryStore(positions=positions, indices=np.asarray(indices)),
        MeshFlatAppearance(),
        section=section,
    )
    assert {visual.section.mode for visual in visuals.values()} == {"slab"}
    # Each panel has its own config object.
    assert len({id(visual.section) for visual in visuals.values()}) == len(visuals)
    assert all(visual.section is not section for visual in visuals.values())
    ortho.controller.close()


def _is_shown(widget) -> bool:
    if hasattr(widget, "input"):  # Qt
        # The holder is what a layout gets; it must never open as a window.
        assert not widget.widget.isVisible()
        assert widget.shown == (not widget._group.isHidden())
        return widget.shown
    return not widget.hidden


def _wired_section(controller, toolkit, visual_ids):
    from cellier.convenience.gui._appearance_widgets import _any_mesh_section
    from cellier.convenience.gui._appearance_widgets_qt import _qt_mesh_section
    from cellier.convenience.layout._shared import ControlSpec

    section = controller.get_visual_model(visual_ids[0]).section.model_dump()
    spec = ControlSpec("mesh_section", "2D section", {"section": section})
    build = _qt_mesh_section if toolkit == "qt" else _any_mesh_section
    widget = _owned(build(spec, visual_ids, controller))
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    return widget


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_it_is_shown_only_while_the_scene_is_2d(controller, toolkit):
    scene, visual = _add_mesh(controller)
    widget = _wired_section(controller, toolkit, [visual.id])
    assert _is_shown(widget)

    controller.set_displayed_axes(scene.id, (0, 1, 2))
    assert not _is_shown(widget)

    controller.set_displayed_axes(scene.id, (1, 2))
    assert _is_shown(widget)


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_it_starts_hidden_in_a_3d_scene(controller, toolkit):
    scene, visual = _add_mesh(controller)
    controller.set_displayed_axes(scene.id, (0, 1, 2))

    assert not _is_shown(_wired_section(controller, toolkit, [visual.id]))


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_group_over_2d_and_3d_scenes_stays_shown(controller, toolkit):
    scene_a, visual_a = _add_mesh(controller, "a")
    scene_b, visual_b = _add_mesh(controller, "b")
    controller.set_displayed_axes(scene_b.id, (0, 1, 2))
    widget = _wired_section(controller, toolkit, [visual_a.id, visual_b.id])
    assert _is_shown(widget)

    controller.set_displayed_axes(scene_a.id, (0, 1, 2))
    assert not _is_shown(widget)


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_control_given_no_scene_is_always_shown(controller, toolkit):
    _scene, visual = _add_mesh(controller)

    assert _is_shown(_make(toolkit, visual.id, visual.section))
