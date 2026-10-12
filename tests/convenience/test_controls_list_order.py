"""The dock stacks its controls in the order of the ``controls=`` list.

``plans/ordered_controls_list_implementation.md`` (Phase 2, items 7 and 8).
``test_appearance_specs.py`` shows that the specs follow the list.  Here the
built dock is read back, on Qt and on anywidget (the Jupyter host), with the
selector and with collapsible sections: the controls name themselves in the
order of the list, and the toolkit stacks them in that order.

No window is opened; Qt runs offscreen.
"""

from __future__ import annotations

import warnings

import pytest

from cellier.convenience import (
    AppearanceControls,
    ClippingPlanesControl,
    DatasetInfoControl,
    OrthoClippingControls,
    OrthoViewer,
    Viewer,
)
from cellier.convenience._hosts import JupyterHost, QtLayoutHost
from cellier.convenience.gui._controls_config import validate_controls
from cellier.convenience.layout._shared import appearance_specs, appearance_targets
from cellier.convenience.layout._walk import render_dock
from cellier.scene.dims import spatial_axes
from tests.convenience._controls_lists import PARITY_CONTROLS, PARITY_KINDS
from tests.convenience._qt_acceptance import (
    anywidget_stacking,
    control_labels_anywidget,
    qt_control_name,
    qt_stacking,
)
from tests.convenience._spec_parity_cases import build_case

CASES = ["multiscale_image/all", "multiscale_mesh/all", "graph/all"]
TOOLKITS = {"qt": QtLayoutHost, "anywidget": JupyterHost}

#: How the list of a case is rearranged, so that it is not in the old order.
ORDERS = {
    "as_written": lambda controls: controls,
    "reversed": lambda controls: controls[::-1],
    "rotated": lambda controls: [*controls[3:], *controls[:3]],
    "interleaved": lambda controls: [*controls[1::2], *controls[0::2]],
}


def _built(case, controls, toolkit, presentation):
    """Render the appearance dock; return ``(titles, stacking, closeables)``."""
    viewer = case.viewer
    closeables: list = []
    root = render_dock(
        AppearanceControls(presentation=presentation),
        viewer,
        TOOLKITS[toolkit](),
        closeables,
    )
    (dock,) = closeables
    widgets = dock.widgets
    if toolkit == "qt":
        titles = [qt_control_name(getattr(w, "widget", w)) for w in widgets]
        stacking = qt_stacking(root, widgets)
    else:
        titles = control_labels_anywidget(widgets)
        stacking = anywidget_stacking(dock, root, widgets)
    return titles, stacking, closeables


@pytest.fixture
def parity_case(request, tmp_path, qtbot):
    case = build_case(request.param.split("/")[0], tmp_path)
    # The clipping planes control names the canvas of its gizmo.
    case.viewer.add_canvas()
    yield request.param, case
    case.viewer.controller.close()


@pytest.mark.parametrize("order", list(ORDERS))
@pytest.mark.parametrize("presentation", ["selector", "collapsible_sections"])
@pytest.mark.parametrize("toolkit", list(TOOLKITS))
@pytest.mark.parametrize("parity_case", CASES, indirect=True)
def test_the_dock_follows_the_list(parity_case, toolkit, presentation, order):
    name, case = parity_case
    kind = PARITY_KINDS[name.split("/")[0]]
    written = PARITY_CONTROLS[name]()
    controls = validate_controls(ORDERS[order](written), kind, name)
    # ``build_case`` adds the visual with no controls; record them as the
    # viewer's own ``add_*`` does.
    case.viewer._store_controls([case.visual.id], controls)

    def titles_of(items):
        specs = appearance_specs(case.visual, items, case.store, palette=case.palette)
        return [spec.title for spec in specs.specs]

    expected = titles_of(controls)
    if order != "as_written":
        # The rearranged list really is a different dock.
        assert expected != titles_of(written)
        assert sorted(expected) == sorted(titles_of(written))

    titles, stacking, closeables = _built(case, controls, toolkit, presentation)
    try:
        assert titles == expected
        assert stacking == list(range(len(expected)))
    finally:
        for closeable in closeables:
            closeable.close()


# -- Lists with no field control (design F3) ----------------------------------


def _points_viewer(gui, controls):
    import numpy as np

    from cellier.data.points._points_memory_store import PointsMemoryStore

    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui=gui
    )
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    visual = viewer.add_points(
        PointsMemoryStore(positions=positions), controls=controls
    )
    viewer.add_canvas()
    return viewer, visual


@pytest.mark.parametrize("toolkit", list(TOOLKITS))
@pytest.mark.parametrize(
    ("control", "title"),
    [
        (DatasetInfoControl(), "Dataset info"),
        (ClippingPlanesControl(), "Clipping planes"),
    ],
)
def test_a_viewer_builds_one_control_from_a_list_of_one(qtbot, toolkit, control, title):
    """Each used to build nothing unless a field was also asked for."""
    viewer, _visual = _points_viewer(
        "qt" if toolkit == "qt" else "anywidget", [control]
    )
    closeables: list = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            render_dock(AppearanceControls(), viewer, TOOLKITS[toolkit](), closeables)
        (dock,) = closeables
        (widget,) = dock.widgets
        if toolkit == "qt":
            assert qt_control_name(getattr(widget, "widget", widget)) == title
        else:
            assert control_labels_anywidget([widget]) == [title]
    finally:
        for closeable in closeables:
            closeable.close()
        viewer.controller.close()


def test_an_ortho_clipping_only_list_has_no_appearance_entry(qtbot):
    """Design 1.3: the dataset is in the clipping dock and nowhere else."""
    import numpy as np

    from cellier.data.points._points_memory_store import PointsMemoryStore

    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        for scene in ortho.scenes.values():
            ortho.controller.add_canvas(scene.id)
        positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        ortho.add_points(
            PointsMemoryStore(positions=positions),
            name="spots",
            controls=[ClippingPlanesControl()],
        )

        assert appearance_targets(ortho) == []

        closeables: list = []
        render_dock(OrthoClippingControls(), ortho, JupyterHost(), closeables)
        try:
            (widget,) = closeables
            assert widget.datasets == ["spots"]
        finally:
            for closeable in closeables:
                closeable.close()
    finally:
        ortho.controller.close()
