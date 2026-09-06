"""Convenience-layer tests for the render-settings dock and viewer surface.

Two things are covered here that the widget tests cannot see: that a
``RenderControls`` dock actually renders and wires its panels in both
hosts, and that the viewer surface is complete enough that nothing needs
``viewer.controller._render_manager``.
"""

from __future__ import annotations

import pytest

from cellier.controller import _RENDER_CONFIG_ROUTES
from cellier.convenience import OrthoViewer, RenderControls, Viewer
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.layout import Layout
from cellier.convenience.layout._shared import (
    render_panel_kwargs,
    render_panel_sections,
)
from cellier.convenience.layout._walk import render_dock

# ---------------------------------------------------------------------------
# The dock spec
# ---------------------------------------------------------------------------


def test_the_dock_defaults_to_every_section():
    assert RenderControls().sections == ("ambient_occlusion", "outline", "temporal")


def test_the_dock_accepts_a_subset():
    assert render_panel_sections(RenderControls(sections=("ambient_occlusion",))) == (
        "ambient_occlusion",
    )


def test_an_unknown_section_raises():
    """A typo would otherwise render an empty dock and no error."""
    with pytest.raises(ValueError, match="unknown render control section"):
        render_panel_sections(RenderControls(sections=("sao",)))


def test_the_single_layout_preset_places_the_dock():
    layout = Layout.single("canvas", render="right")
    assert isinstance(layout.right_dock, RenderControls)


def test_the_single_layout_preset_stacks_with_the_appearance_dock():
    """Two docks in one region compose rather than one overwriting the other."""
    from cellier.convenience.layout import AppearanceControls, VStack

    layout = Layout.single("canvas", appearance="right", render="right")
    assert isinstance(layout.right_dock, VStack)
    kinds = [type(item) for item in layout.right_dock.items]
    assert kinds == [AppearanceControls, RenderControls]


# ---------------------------------------------------------------------------
# The shared wiring
# ---------------------------------------------------------------------------


def test_the_ssao_panel_is_given_a_live_radius_reader():
    """A value would be a snapshot; the radius is derived and moves."""
    viewer = Viewer(("z", "y", "x"), dim="3d")
    kwargs = render_panel_kwargs("ambient_occlusion", viewer.controller)
    assert callable(kwargs["effective_radius"])
    assert (
        kwargs["effective_radius"]()
        == viewer.controller.ambient_occlusion_effective_radius
    )


def test_the_temporal_panel_is_given_a_live_frame_count_and_a_reset():
    viewer = Viewer(("z", "y", "x"), dim="3d")
    kwargs = render_panel_kwargs("temporal", viewer.controller)
    assert callable(kwargs["frame_count"])
    kwargs["on_reset"]()  # must not raise with no canvas


def test_the_outline_panel_is_given_live_slot_usage():
    """``{slot: how many visuals use it}``, annotating the palette swatches.

    Derived state, like the occlusion radius and the frame count, so it is a
    callable rather than a value: a visual moving between slots changes it
    with no config field changing.

    Both panel bases have accepted ``slot_usage`` since they were written and
    both front ends draw it -- but nothing passed it and the Qt panel did not
    even forward the keyword, so the readout was dead on both toolkits.
    """
    viewer = Viewer(("z", "y", "x"), dim="3d")
    kwargs = render_panel_kwargs("outline", viewer.controller)
    assert callable(kwargs["slot_usage"])
    assert kwargs["slot_usage"]() == {}  # no visuals yet, and it must not raise


# ---------------------------------------------------------------------------
# Rendering the dock
# ---------------------------------------------------------------------------


def test_the_qt_dock_renders_and_wires_every_panel(qtbot):
    """No configured visual needed: render settings belong to the renderer."""

    viewer = Viewer(("z", "y", "x"), dim="3d")
    closeables: list = []
    dock = render_dock(RenderControls(), viewer, QtLayoutHost(), closeables)

    assert dock is not None
    qtbot.addWidget(dock)
    assert len(closeables) == 3
    assert {panel.section for panel in closeables} == {
        "ambient_occlusion",
        "outline",
        "temporal",
    }


def test_a_qt_dock_panel_edit_reaches_the_render_config(qtbot):
    """The dock connects its panels, so an edit lands without further wiring."""

    viewer = Viewer(("z", "y", "x"), dim="3d")
    closeables: list = []
    dock = render_dock(
        RenderControls(sections=("ambient_occlusion",)),
        viewer,
        QtLayoutHost(),
        closeables,
    )
    qtbot.addWidget(dock)

    panel = closeables[0]
    checkbox = panel._appliers["enabled"].__closure__[0].cell_contents
    checkbox.setChecked(True)

    assert viewer.render_config.ambient_occlusion.enabled is True


def test_the_qt_dock_honours_a_section_subset(qtbot):

    viewer = Viewer(("z", "y", "x"), dim="3d")
    closeables: list = []
    dock = render_dock(
        RenderControls(sections=("temporal",)), viewer, QtLayoutHost(), closeables
    )
    qtbot.addWidget(dock)

    assert [panel.section for panel in closeables] == ["temporal"]


def test_the_qt_dock_works_on_an_ortho_viewer(qtbot):
    """An ``OrthoViewer`` has no ``scene``, which used to defeat other docks."""

    viewer = OrthoViewer(("z", "y", "x"))
    closeables: list = []
    dock = render_dock(RenderControls(), viewer, QtLayoutHost(), closeables)

    assert dock is not None
    qtbot.addWidget(dock)
    assert len(closeables) == 3


def test_the_anywidget_dock_renders_and_wires_every_panel():
    pytest.importorskip("anywidget")
    from cellier.convenience._hosts import JupyterHost

    viewer = Viewer(("z", "y", "x"), dim="3d", gui="anywidget")
    closeables: list = []
    dock = render_dock(RenderControls(), viewer, JupyterHost(), closeables)

    assert dock is not None
    assert {panel.section for panel in closeables} == {
        "ambient_occlusion",
        "outline",
        "temporal",
    }


def test_an_anywidget_dock_panel_edit_reaches_the_render_config():
    pytest.importorskip("anywidget")
    from cellier.convenience._hosts import JupyterHost

    viewer = Viewer(("z", "y", "x"), dim="3d", gui="anywidget")
    closeables: list = []
    render_dock(
        RenderControls(sections=("ambient_occlusion",)),
        viewer,
        JupyterHost(),
        closeables,
    )

    closeables[0].enabled = True

    assert viewer.render_config.ambient_occlusion.enabled is True


# ---------------------------------------------------------------------------
# The viewer surface
# ---------------------------------------------------------------------------


_VIEWER_PROPERTIES = [
    ("outline_enabled", True),
    ("outline_boundaries_enabled", False),
    ("outline_selection_enabled", False),
    ("ambient_occlusion_enabled", True),
    ("ambient_occlusion_radius", 8.0),
    ("ambient_occlusion_auto_radius_fraction", 0.04),
    ("ambient_occlusion_strength", 0.25),
    ("ambient_occlusion_power", 2.0),
    ("ambient_occlusion_bias", 0.2),
    ("ambient_occlusion_n_samples", 24),
    ("ambient_occlusion_blur_radius", 1),
    ("temporal_enabled", False),
    ("temporal_blend_weight", 0.5),
]


@pytest.mark.parametrize(("prop", "value"), _VIEWER_PROPERTIES)
def test_the_viewer_exposes_every_setting(prop, value):
    viewer = Viewer(("z", "y", "x"), dim="3d")
    setattr(viewer, prop, value)
    assert getattr(viewer, prop) == value
    assert getattr(viewer.controller, prop) == value


@pytest.mark.parametrize(("prop", "value"), _VIEWER_PROPERTIES)
def test_the_ortho_viewer_exposes_every_setting(prop, value):
    viewer = OrthoViewer(("z", "y", "x"))
    setattr(viewer, prop, value)
    assert getattr(viewer, prop) == value


def test_the_viewer_reaches_every_settable_field():
    """Nothing should need ``viewer.controller._render_manager`` to change.

    The demos used to reach through it for the sample count, the blur
    radius and the temporal pair, and into ``canvas._ssao_pass`` for the
    auto radius fraction.
    """
    viewer = Viewer(("z", "y", "x"), dim="3d")
    for section, field in _RENDER_CONFIG_ROUTES:
        if field in {"palette", "inner_color", "inner_thickness"} or "." in field:
            # Reachable through ``viewer.render_config`` plus the
            # controller's seam rather than as a named property; a
            # dedicated property per outline thickness would be eight more
            # names for no gain.
            continue
        assert hasattr(viewer, f"{section}_{field}") or hasattr(viewer, field), field


def test_the_viewer_exposes_the_read_only_effective_radius():
    viewer = Viewer(("z", "y", "x"), dim="3d")
    assert viewer.ambient_occlusion_effective_radius is None  # no canvas yet
    with pytest.raises(AttributeError):
        viewer.ambient_occlusion_effective_radius = 1.0


def test_the_viewer_can_reset_accumulation():
    viewer = Viewer(("z", "y", "x"), dim="3d")
    viewer.reset_temporal_accumulation()  # must not raise with no canvas


def test_a_viewer_edit_announces_itself(qtbot):
    """A notebook-cell assignment has to reach a connected panel."""
    from uuid import uuid4

    from cellier.events import RenderConfigChangedEvent

    viewer = Viewer(("z", "y", "x"), dim="3d")
    seen: list = []
    viewer.controller._outgoing_events.subscribe(
        RenderConfigChangedEvent, seen.append, owner_id=uuid4()
    )

    viewer.ambient_occlusion_power = 3.0

    assert [(e.section, e.field_name) for e in seen] == [("ambient_occlusion", "power")]


# ---------------------------------------------------------------------------
# The add_* render-settings keywords on the convenience viewers
# ---------------------------------------------------------------------------

#: The ``add_*`` methods both viewers forward to the controller.  Checked
#: parametrically because the forwarding is positional -- a keyword left off
#: one call site is silently dropped rather than raising.
_ADD_METHODS = (
    "add_image",
    "add_labels",
    "add_mesh",
    "add_points",
    "add_lines",
    "add_graph",
)


def _call_add(viewer, method: str, **render_kwargs):
    """Call one ``add_*`` on *viewer* with a minimal payload."""
    import numpy as np

    from cellier.data import (
        GraphMemoryStore,
        ImageMemoryStore,
        LabelMemoryStore,
        LinesMemoryStore,
        MeshMemoryStore,
        PointsMemoryStore,
    )
    from cellier.visuals import (
        GraphAppearance,
        InMemoryImageAppearance,
        InMemoryLabelsAppearance,
        LinesMemoryAppearance,
        MeshFlatAppearance,
        PointsMarkerAppearance,
    )

    if method == "add_image":
        return viewer.add_image(
            ImageMemoryStore(data=np.zeros((8, 8, 8), dtype=np.float32), name="i"),
            InMemoryImageAppearance(color_map="gray", clim=(0.0, 1.0)),
            **render_kwargs,
        )
    if method == "add_labels":
        return viewer.add_labels(
            LabelMemoryStore(data=np.zeros((8, 8, 8), dtype=np.int32), name="l"),
            InMemoryLabelsAppearance(colormap_mode="random"),
            **render_kwargs,
        )
    if method == "add_mesh":
        return viewer.add_mesh(
            MeshMemoryStore(
                positions=np.zeros((3, 3), dtype=np.float32),
                indices=np.array([[0, 1, 2]], dtype=np.int32),
                name="m",
            ),
            MeshFlatAppearance(color=(1.0, 1.0, 1.0, 1.0)),
            **render_kwargs,
        )
    if method == "add_points":
        return viewer.add_points(
            PointsMemoryStore(positions=np.zeros((3, 3), dtype=np.float32), name="p"),
            PointsMarkerAppearance(),
            **render_kwargs,
        )
    if method == "add_lines":
        return viewer.add_lines(
            LinesMemoryStore(positions=np.zeros((4, 3), dtype=np.float32), name="ln"),
            LinesMemoryAppearance(),
            **render_kwargs,
        )
    return viewer.add_graph(
        GraphMemoryStore(
            positions=np.zeros((3, 3), dtype=np.float32),
            edges=np.array([[0, 1]], dtype=np.int64),
            name="g",
        ),
        GraphAppearance(),
        **render_kwargs,
    )


def _outlining_config():
    """A config with the outline pass on, so outlining does not warn."""
    from cellier.render import OutlineConfig, RenderManagerConfig

    return RenderManagerConfig(outline=OutlineConfig(enabled=True))


@pytest.mark.parametrize("method", _ADD_METHODS)
def test_the_viewer_forwards_the_render_settings(method):
    from cellier.visuals._base_visual import VisualOutline

    viewer = Viewer(("z", "y", "x"), dim="3d", render_config=_outlining_config())
    visual = _call_add(
        viewer,
        method,
        outline=VisualOutline(slot=3, placement="inward"),
        ambient_occlusion=True,
    )

    assert visual.outline.slot == 3
    assert visual.outline.placement == "inward"
    assert visual.ambient_occlusion is True


@pytest.mark.parametrize("method", _ADD_METHODS)
def test_the_ortho_viewer_forwards_them_to_every_panel(method):
    """One call, four panels, all carrying the same settings."""
    from cellier.visuals._base_visual import VisualOutline

    viewer = OrthoViewer(("z", "y", "x"), render_config=_outlining_config())
    visuals = _call_add(viewer, method, outline=VisualOutline(slot=2))

    assert len(visuals) == 4
    assert all(v.outline.slot == 2 for v in visuals.values())


def test_the_viewer_forwards_a_label_selection():
    import numpy as np

    from cellier.data import LabelMemoryStore
    from cellier.visuals import InMemoryLabelsAppearance
    from cellier.visuals._base_visual import VisualOutline

    viewer = Viewer(("z", "y", "x"), dim="3d", render_config=_outlining_config())
    visual = viewer.add_labels(
        LabelMemoryStore(data=np.zeros((8, 8, 8), dtype=np.int32), name="l"),
        InMemoryLabelsAppearance(colormap_mode="random"),
        outline=VisualOutline(slot=1),
        outline_selected_labels={2: 1},
    )

    assert visual.outline_selected_labels == {2: 1}


def test_the_global_dock_names_its_scope(qtbot):
    """One heading over the whole dock, saying these are the renderer's.

    Without it the per-visual groups across the canvas and these read as the
    same kind of thing: "Outline" beside "Outlines" is not a distinction
    anyone should have to notice.
    """
    from cellier.gui._render_controls import RENDER_DOCK_TITLE

    viewer = Viewer(("z", "y", "x"), dim="3d")
    dock = render_dock(RenderControls(), viewer, QtLayoutHost(), [])
    qtbot.addWidget(dock)

    assert dock.title() == RENDER_DOCK_TITLE == "Renderer effects"
