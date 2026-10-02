"""The "LOD" control of a multiscale mesh, on both toolkits.

``plans/mesh_refactor_v3.md`` Phase 9 (5.11).  The control sends
``LodConfigUpdateEvent`` and follows ``LodConfigChangedEvent``; the viewers
mirror ``add_multiscale_mesh`` and ``set_lod_config``.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.mesh import MeshLevel, MeshMemoryStore, MultiscaleMeshStore
from cellier.gui._lod import LOD_CONFIG_TITLE, lod_config_fields
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    GeometryLodConfig,
    MeshFlatAppearance,
    MeshSectionConfig,
    MultiscaleMeshVisual,
)
from tests._meshes import uv_sphere

if TYPE_CHECKING:
    from cellier.events import LodConfigChangedEvent

_SERIALS = itertools.count()
_ROWS = ("dims_drag", "dims_drag_draw", "camera_motion")
_QTBOT: list = []


@pytest.fixture(autouse=True)
def _own_qt_controls(qtbot):
    """Give every Qt control a test builds to ``qtbot``, which closes it."""
    _QTBOT.append(qtbot)
    yield
    _QTBOT.clear()


def _owned(control):
    if not hasattr(control, "comm"):
        _QTBOT[-1].addWidget(control.widget)
    return control


def _store(*n_lats: int, name: str = "sphere") -> MultiscaleMeshStore:
    levels = [
        MeshLevel(positions=positions, indices=indices)
        for positions, indices in (
            uv_sphere(5.0, (8.0, 8.0, 8.0), n_lat=n, n_lon=2 * n) for n in n_lats
        )
    ]
    return MultiscaleMeshStore(levels=levels, name=name)


def _make(toolkit, visual_ids, lod):
    if toolkit == "qt":
        from cellier.gui.qt.visuals import QtLodConfigControls

        return _owned(QtLodConfigControls(visual_ids, lod=lod.model_dump()))
    from cellier.gui.anywidget.visuals import AnywidgetLodConfigControls

    return AnywidgetLodConfigControls(visual_ids, lod=lod.model_dump())


def _user_edit(widget, field, value) -> None:
    """Change one control as a user would."""
    if hasattr(widget, "input"):  # Qt
        widget.input(field).setCurrentText(value)
    else:  # anywidget: what the front end sets, a fresh serial per edit
        widget.edit = {"field": field, "value": value, "serial": next(_SERIALS)}


def _shown(widget) -> dict:
    if hasattr(widget, "input"):
        return {name: widget.input(name).currentText() for name in _ROWS}
    return dict(widget.config)


@pytest.fixture
def controller(qtbot):
    controller = CellierController(gui="offscreen")
    yield controller
    controller.close()


def _add_mesh(controller, name="mesh", lod=None) -> MultiscaleMeshVisual:
    scene = controller.add_scene(dim="3d", name=f"scene-{name}")
    return controller.add_multiscale_mesh(
        _store(8, 4, name=name), scene.id, MeshFlatAppearance(), name, lod=lod
    )


def test_the_rows_are_read_off_the_model():
    fields = {field.name: field for field in lod_config_fields()}
    assert tuple(fields) == _ROWS
    # Every field but the one fixed when the visual is added.
    assert set(fields) == set(GeometryLodConfig.model_fields) - {"coarse_level"}
    assert all(field.kind == "choice" for field in fields.values())
    assert all(field.choices == ("coarse", "full") for field in fields.values())


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_it_shows_the_config_and_edits_it(controller, toolkit):
    visual = _add_mesh(controller, lod=GeometryLodConfig(dims_drag_draw="full"))
    widget = _make(toolkit, [visual.id], visual.lod)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    seen: list[LodConfigChangedEvent] = []
    controller.on_lod_config_changed(visual.id, seen.append, owner_id=uuid4())

    assert _shown(widget) == {
        "dims_drag": "coarse",
        "dims_drag_draw": "full",
        "camera_motion": "coarse",
    }
    _user_edit(widget, "camera_motion", "full")
    assert visual.lod.camera_motion == "full"
    _user_edit(widget, "dims_drag", "full")
    assert visual.lod.dims_drag == "full"
    assert widget.error == ""
    # One event per edit, stamped with the widget's id.
    assert [event.lod for event in seen] == [
        GeometryLodConfig(dims_drag_draw="full", camera_motion="full"),
        visual.lod,
    ]
    assert {event.source_id for event in seen} == {widget._id}

    # A change from elsewhere is shown, and stamped as the controller's.
    controller.set_lod_config(visual.id, dims_drag_draw="coarse")
    assert _shown(widget)["dims_drag_draw"] == "coarse"
    assert _shown(widget)["camera_motion"] == "full"
    assert seen[-1].source_id == controller._id

    widget.close()


def test_setting_the_same_config_emits_nothing(controller):
    visual = _add_mesh(controller)
    seen: list[LodConfigChangedEvent] = []
    controller.on_lod_config_changed(visual.id, seen.append, owner_id=uuid4())

    controller.set_lod_config(visual.id, camera_motion="coarse")

    assert seen == []


def test_assigning_the_config_on_the_model_is_announced(controller):
    visual = _add_mesh(controller)
    widget = _make("qt", [visual.id], visual.lod)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    visual.lod = GeometryLodConfig(camera_motion="full")

    assert _shown(widget)["camera_motion"] == "full"


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_refused_edit_shows_the_config_again_and_why(controller, toolkit):
    visual = _add_mesh(controller)
    widget = _make(toolkit, [visual.id], visual.lod)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    # Sent as a front end would send a value its own choices do not offer.
    widget._editor.edit("camera_motion", "sometimes")
    assert visual.lod.camera_motion == "coarse"
    assert _shown(widget)["camera_motion"] == "coarse"
    assert widget.error != ""
    widget.close()


def test_a_group_control_edits_every_visual(controller):
    first = _add_mesh(controller, "a")
    second = _add_mesh(controller, "b")
    widget = _make("qt", [first.id, second.id], first.lod)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _user_edit(widget, "dims_drag", "full")
    assert first.lod.dims_drag == "full"
    assert second.lod.dims_drag == "full"


def test_both_toolkits_carry_the_shared_title():
    from cellier.gui.anywidget.visuals import AnywidgetLodConfigControls
    from cellier.gui.qt.visuals import QtLodConfigControls

    assert QtLodConfigControls.DEFAULT_TITLE == LOD_CONFIG_TITLE == "LOD"
    assert AnywidgetLodConfigControls.DEFAULT_TITLE == LOD_CONFIG_TITLE


# -- the panel ------------------------------------------------------------------


def _kinds(visual, config) -> list[str]:
    from cellier.convenience.layout._shared import appearance_specs

    return [spec.kind for spec in appearance_specs(visual, config).specs]


def test_the_panel_offers_it_only_when_asked(controller):
    from cellier.convenience import MeshControlsConfig
    from cellier.convenience.layout._shared import appearance_specs

    visual = _add_mesh(controller)

    assert "lod_config" not in _kinds(visual, MeshControlsConfig(appearance=True))
    asked = MeshControlsConfig(appearance=True, lod_controls=True)
    spec = next(
        spec
        for spec in appearance_specs(visual, asked).specs
        if spec.kind == "lod_config"
    )
    assert spec.title == "LOD"
    assert spec.values["lod"] == visual.lod.model_dump()


def test_a_mesh_with_one_level_has_no_lod_group(controller):
    from cellier.convenience import MeshControlsConfig

    scene = controller.add_scene(dim="3d", name="scene")
    positions, indices = uv_sphere(5.0, (8.0, 8.0, 8.0))
    visual = controller.add_mesh(
        MeshMemoryStore(positions=positions, indices=indices),
        scene.id,
        MeshFlatAppearance(),
    )

    config = MeshControlsConfig(appearance=True, lod_controls=True)
    assert "lod_config" not in _kinds(visual, config)


def test_a_mesh_panel_offers_the_fetch_status_when_asked(controller):
    from cellier.convenience import MeshControlsConfig
    from cellier.convenience.layout._shared import appearance_specs

    visual = _add_mesh(controller)

    assert "loading" not in _kinds(visual, MeshControlsConfig(appearance=True))
    asked = MeshControlsConfig(appearance=True, loading_indicator=True)
    spec = next(
        spec for spec in appearance_specs(visual, asked).specs if spec.kind == "loading"
    )
    assert spec.title == "Data fetch status"
    assert spec.values == {"levels": True}


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_the_builders_make_both_controls(controller, toolkit):
    from cellier.convenience import MeshControlsConfig
    from cellier.convenience.gui._appearance_widgets import ANYWIDGET_BUILDERS
    from cellier.convenience.gui._appearance_widgets_qt import QT_BUILDERS
    from cellier.convenience.layout._shared import appearance_specs

    visual = _add_mesh(controller)
    config = MeshControlsConfig(
        appearance=True, lod_controls=True, loading_indicator=True
    )
    builders = QT_BUILDERS if toolkit == "qt" else ANYWIDGET_BUILDERS
    built = {
        spec.kind: _owned(builders[spec.kind](spec, [visual.id], controller))
        for spec in appearance_specs(visual, config).specs
        if spec.kind in ("lod_config", "loading")
    }

    assert _shown(built["lod_config"])["camera_motion"] == "coarse"
    # Nothing was planned yet, and the indicator words a mesh in levels.
    assert built["loading"].state.text == "Not loaded"
    assert built["loading"]._model._levels is True


# -- the viewers ----------------------------------------------------------------


def test_the_viewer_changes_the_lod_config(qtbot):
    from cellier.convenience import Viewer

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    mesh = viewer.add_multiscale_mesh(_store(8, 4), MeshFlatAppearance())

    new = viewer.set_lod_config(mesh, camera_motion="full")

    assert new == mesh.lod == GeometryLodConfig(camera_motion="full")
    with pytest.raises(ValueError, match="coarse_level cannot be changed"):
        viewer.set_lod_config(mesh.id, coarse_level=2)
    viewer.controller.close()


def test_the_ortho_viewer_adds_a_multiscale_mesh_to_every_panel(qtbot):
    from cellier.convenience import MeshControlsConfig, OrthoViewer
    from cellier.convenience.layout._shared import appearance_targets

    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    section = MeshSectionConfig(fill=False)
    visuals = ortho.add_multiscale_mesh(
        _store(8, 6, 4),
        MeshFlatAppearance(),
        name="cell",
        section=section,
        lod=GeometryLodConfig(coarse_level=2, dims_drag="full"),
        controls=MeshControlsConfig(appearance=True, lod_controls=True),
    )

    assert set(visuals) == set(ortho.scenes)
    assert all(isinstance(v, MultiscaleMeshVisual) for v in visuals.values())
    assert {v.name for v in visuals.values()} == {f"cell_{key}" for key in visuals}
    # One store, read by every panel.
    assert len({v.data_store_id for v in visuals.values()}) == 1
    assert {v.lod.coarse_level for v in visuals.values()} == {2}
    assert {v.section.fill for v in visuals.values()} == {False}
    assert all(v.section is not section for v in visuals.values())
    # The 2D panels and the 3D panel each get a control group.
    targets = appearance_targets(ortho)
    assert sorted(len(target.visual_ids) for target in targets) == [1, 3]

    # Every panel changes together, from any panel's visual or the dict.
    ortho.set_lod_config(next(iter(visuals.values())), camera_motion="full")
    assert {v.lod.camera_motion for v in visuals.values()} == {"full"}
    ortho.set_lod_config(visuals, dims_drag="coarse")
    assert {v.lod.dims_drag for v in visuals.values()} == {"coarse"}
    with pytest.raises(ValueError):
        ortho.set_lod_config(visuals, camera_motion="sometimes")
    assert {v.lod.camera_motion for v in visuals.values()} == {"full"}
    ortho.controller.close()


# -- the file round trip --------------------------------------------------------


def test_a_viewer_with_a_multiscale_mesh_round_trips_through_a_file(qtbot, tmp_path):
    from cellier.convenience import Viewer

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="2d", gui="offscreen")
    store = _store(8, 6, 4, name="cell")
    viewer.add_multiscale_mesh(
        store,
        MeshFlatAppearance(color=(0.0, 1.0, 0.0, 1.0)),
        name="cell",
        section=MeshSectionConfig(mode="slab", outline_width=3.0),
        lod=GeometryLodConfig(coarse_level=2, dims_drag="full", camera_motion="full"),
    )
    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    loaded = Viewer.from_file(path)

    (mesh,) = loaded.scene.visuals
    assert isinstance(mesh, MultiscaleMeshVisual)
    assert mesh.lod == GeometryLodConfig(
        coarse_level=2, dims_drag="full", camera_motion="full"
    )
    assert mesh.section == MeshSectionConfig(mode="slab", outline_width=3.0)
    again = loaded.controller.get_data_store(store.id)
    assert isinstance(again, MultiscaleMeshStore)
    assert again.level_count == 3
    for ours, theirs in zip(again.levels, store.levels):
        np.testing.assert_array_equal(ours.positions, theirs.positions)
        np.testing.assert_array_equal(ours.indices, theirs.indices)
    # The loaded visual keeps the level the file named.
    scenes = loaded.controller._render_manager._scenes
    assert scenes[loaded.scene.id].get_visual(mesh.id).resident_levels == (0, 1)
    viewer.controller.close()
    loaded.controller.close()


# -- tooltips -------------------------------------------------------------------


def test_every_row_says_what_it_does():
    from cellier.gui._mesh_section import mesh_section_fields

    for field in (*lod_config_fields(), *mesh_section_fields()):
        assert field.tooltip, field.name
    mode = next(f for f in mesh_section_fields() if f.name == "mode")
    # A cut draws one plane whatever the spatial thickness (plan Phase 9).
    assert "thickness" in mode.tooltip and "ignored" in mode.tooltip


def test_the_qt_rows_carry_their_tooltips(controller):
    from cellier.gui.qt.visuals import QtMeshSectionControls

    visual = _add_mesh(controller)
    lod = _make("qt", [visual.id], visual.lod)
    section = _owned(
        QtMeshSectionControls([visual.id], section=visual.section.model_dump())
    )

    for field in lod_config_fields():
        assert lod.input(field.name).toolTip() == field.tooltip
    assert "ignored" in section.input("mode").toolTip()


def test_the_anywidget_rows_carry_their_tooltips(controller):
    visual = _add_mesh(controller)
    widget = _make("anywidget", [visual.id], visual.lod)

    assert [row["tooltip"] for row in widget.fields] == [
        field.tooltip for field in lod_config_fields()
    ]
