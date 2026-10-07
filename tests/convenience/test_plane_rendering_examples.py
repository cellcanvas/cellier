"""The plane rendering examples build, draw a frame and exit.

``examples/plane_rendering/`` (plane rendering design 10.3, 14.12).  Each Qt
example builds its viewer and its layout at import and calls ``run`` only
under ``__main__``, so a test imports it, renders the layout into a window,
loads the data, runs the viewer's ready callbacks and captures the window.
"""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cellier.convenience import screenshot_window
from cellier.convenience.layout._qt_renderer import render_qt
from tests.render.conftest import drain_loading

EXAMPLES = Path(__file__).parents[2] / "examples" / "plane_rendering"

QT_EXAMPLES = (
    "plane_gizmo",
    "plane_interaction_script",
    "plane_labels_composite",
    "ortho_slice_planes",
    "plane_slicing_validation",
)


@pytest.fixture
def load_example(monkeypatch, tmp_path, qtbot):
    """Import an example by file name, its temporary stores under tmp_path.

    The modules are kept until teardown, and each keeps its window (see
    ``_show``): a window dropped without being closed takes its controls'
    widgets with it and leaves them subscribed to the controller.
    """
    counter = itertools.count()

    def _mkdtemp(*args, **kwargs) -> str:
        path = tmp_path / f"example_{next(counter)}"
        path.mkdir()
        return str(path)

    monkeypatch.setattr(tempfile, "mkdtemp", _mkdtemp)
    loaded = []

    def _load(name: str):
        path = EXAMPLES / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
        return module

    yield _load
    for module in loaded:
        module.viewer.controller.close()


@pytest.fixture
def run_marimo_example(monkeypatch, load_example):
    """Run every cell of a marimo twin; returns what its cells defined.

    ``app.run()`` runs the cells as a script, where marimo does not report a
    running notebook, so the host is named for ``display``.
    """
    pytest.importorskip("marimo")
    viewers = []

    def _run(name: str) -> dict:
        monkeypatch.setattr("cellier.convenience._hosts._marimo_running", lambda: True)
        module = load_example(f"{name}_marimo")
        outputs, defs = module.app.run()
        # What load_example closes.
        module.viewer = defs["viewer"]
        viewers.append(defs["viewer"])
        # The last thing shown before any button is the composed layout.
        assert any(type(output).__name__.endswith("Html") for output in outputs)
        return defs

    return _run


def _scenes(viewer) -> list:
    scenes = getattr(viewer, "scenes", None)
    return list(scenes.values()) if scenes is not None else [viewer.scene]


async def _show(module, qtbot):
    """Do what ``run`` does up to the first loaded frame; return the window."""
    viewer = module.viewer
    controller = viewer.controller
    window = render_qt(module.layout, viewer)
    module.window = window
    qtbot.addWidget(window)
    window.resize(1000, 700)
    window.show()
    for scene in _scenes(viewer):
        controller.fit_camera(scene.id)
        controller.reslice_scene(scene.id)
    await drain_loading(controller)
    # fit="ready" fits again once the data is loaded.
    for scene in _scenes(viewer):
        controller.fit_camera(scene.id)
    await drain_loading(controller)
    for callback in viewer._ready_callbacks:
        callback()
    return window


def _drawn_colours(window, controller) -> int:
    """How many distinct colours the captured window holds."""
    picture = screenshot_window(window, controller)
    return len(np.unique(picture.reshape(-1, picture.shape[-1]), axis=0))


def test_every_example_is_listed():
    scripts = {
        path.stem for path in EXAMPLES.glob("*.py") if not path.stem.endswith("_marimo")
    }
    assert scripts == set(QT_EXAMPLES)
    twins = {path.stem for path in EXAMPLES.glob("*_marimo.py")}
    assert twins == {f"{name}_marimo" for name in QT_EXAMPLES}


@pytest.mark.parametrize("name", QT_EXAMPLES)
async def test_the_example_builds_and_draws(name, load_example, qtbot, offscreen_gpu):
    module = load_example(name)
    window = await _show(module, qtbot)
    # A blank canvas beside the dock's widgets is a few dozen colours; the
    # data drawn through a colour map is hundreds.
    assert _drawn_colours(window, module.viewer.controller) > 200


# -- What each example is there to show ----------------------------------------


def _canvas_id(viewer):
    (canvas_id,) = viewer.canvases
    return canvas_id


async def test_plane_gizmo_puts_a_gizmo_on_its_plane(load_example, qtbot):
    module = load_example("plane_gizmo")
    await _show(module, qtbot)
    session = module.controller.get_plane_gizmo(_canvas_id(module.viewer))
    assert session is not None
    assert session.kind == "render"
    assert session.plane_id == module.plane.id
    assert module.visual.plane_mode()
    # Both extents are bounded, so both scale handles are there.
    assert all(None not in extent for extent in _extents(module.plane))


def _extents(plane):
    return (plane.extent_0, plane.extent_1)


async def test_the_scripted_motion_is_one_interaction_and_one_plan(
    load_example, qtbot, capsys
):
    module = load_example("plane_interaction_script")
    controller, visual = module.controller, module.visual
    log: list[str] = []
    owner_id = controller._id
    controller.on_plane_interaction(
        visual.id, lambda event: log.append(event.phase), owner_id=owner_id
    )
    await _show(module, qtbot)
    controller.on_reslice_started(
        module.viewer.scene.id,
        lambda event: visual.id in event.visual_ids and log.append("plan"),
        owner_id=owner_id,
    )
    (motion,) = module.motions
    await motion
    await drain_loading(controller)

    assert log == ["start", "end", "plan"]
    # Both planes are where the motion ends.
    assert visual.render_planes[0].origin[0] == module.Z_END
    assert visual.clipping_planes[0].plane.offset == module.X_END
    printed = capsys.readouterr().out
    assert "plane interaction: start" in printed
    assert "plane interaction: end (release)" in printed


async def test_the_scripted_motion_without_a_scope_plans_every_step(
    load_example, qtbot, monkeypatch
):
    module = load_example("plane_interaction_script")
    monkeypatch.setattr(module, "USE_SCOPE", False)
    monkeypatch.setattr(module, "STEPS", 3)
    controller, visual = module.controller, module.visual
    await _show(module, qtbot)
    log: list[str] = []
    owner_id = controller._id
    controller.on_plane_interaction(
        visual.id, lambda event: log.append(event.phase), owner_id=owner_id
    )
    controller.on_reslice_started(
        module.viewer.scene.id,
        lambda event: visual.id in event.visual_ids and log.append("plan"),
        owner_id=owner_id,
    )
    (motion,) = module.motions
    await motion
    await drain_loading(controller)
    # Six jumps.  The first step of each motion is where its plane already
    # is, which changes nothing.
    assert log == ["plan"] * 4


async def test_the_composite_and_the_labels_are_on_their_own_planes(
    load_example, qtbot
):
    module = load_example("plane_labels_composite")
    await _show(module, qtbot)
    image, labels = module.image_visual, module.labels_visual
    assert image.composite
    assert image.plane_mode()
    assert labels.plane_mode()
    assert {channel.render_mode for channel in image.channels.values()} == {"plane"}
    assert len(image.render_planes) == len(labels.render_planes) == 2
    # The labels' planes are where the labels are, in world.
    assert labels.render_planes[0].origin[2] == module.LABELS_X + module.SIDE / 2
    extent = module.viewer.controller.visual_world_extent(labels.id)
    low, high = extent[-1]
    assert low < labels.render_planes[0].origin[2] < high


async def test_the_ortho_example_ties_three_planes_to_the_sliders(load_example, qtbot):
    module = load_example("ortho_slice_planes")
    await _show(module, qtbot)
    viewer, vol = module.viewer, module.vol_visual
    assert viewer.plane_controller.mode == "slices"
    assert vol.plane_mode()
    half = module.SIDE / 2
    assert [plane.origin[k] for k, plane in enumerate(vol.render_planes)] == [half] * 3
    # Only the 3D panel's visual carries planes.
    for name in ("xy", "xz", "yz"):
        assert module.visuals[name].render_planes == ()
    viewer.set_slice_positions({0: 20.0})
    assert vol.render_planes[0].origin[0] == 20.0
    assert [plane.origin[k] for k, plane in enumerate(vol.render_planes)][1:] == [
        half,
        half,
    ]


# -- The validation scene: points, labels and image agree at every time --------

_SHOT = 480


def _look_down_z(module) -> None:
    """Look at the volume along -z.

    The plane faces z, so everything on it is at one depth and a disc on it
    is a circle on screen, centred where its centre projects.
    """
    viewer = module.viewer
    # Rendered space is (x, y, z).
    viewer.controller.look_at_visual(
        module.image_visual.id,
        _canvas_id(viewer),
        view_direction=(0, 0, -1),
        up=(0, 1, 0),
    )


async def _only(module, visual) -> np.ndarray:
    """A picture of the scene with one of the three visuals drawn."""
    viewer = module.viewer
    for other in (module.image_visual, module.labels_visual, module.points_visual):
        other.appearance.visible = other is visual
    viewer.controller.reslice_scene(viewer.scene.id)
    await drain_loading(viewer.controller)
    return viewer.screenshot(size=(_SHOT, _SHOT))[..., :3].astype(int)


def _blob(picture: np.ndarray, color) -> tuple[np.ndarray, int]:
    """The centroid ``(row, col)`` and the size of the pixels of one colour."""
    wanted = np.round(np.asarray(color, dtype=float) * 255).astype(int)
    rows, cols = np.nonzero((np.abs(picture - wanted) <= 3).all(axis=-1))
    if rows.size == 0:
        return np.array([np.nan, np.nan]), 0
    return np.array([rows.mean(), cols.mean()]), int(rows.size)


async def test_the_validation_scene_lines_up_at_every_time_point(
    load_example, qtbot, offscreen_gpu
):
    module = load_example("plane_slicing_validation")
    await _show(module, qtbot)
    viewer, controller = module.viewer, module.viewer.controller
    image, labels, points = (
        module.image_visual,
        module.labels_visual,
        module.points_visual,
    )
    # The frame and the gizmo are not part of what is compared.
    viewer.remove_plane_gizmo()
    image.aabb.enabled = False
    controller.update_background_field(viewer.scene.id, "visible", False)
    _look_down_z(module)

    upper, lower = range(4, 8), range(4)
    for t in range(module.N_TIMES):
        viewer.set_slice_positions({0: float(t)})
        # The plane through the four points of the upper half of z.  It is
        # given to the labels; the example carries it over to the image.
        z = float(module.POSITIONS[t, 4, 0])
        plane = labels.render_planes[0].model_copy(
            update={"origin": (z, module.SIDE / 2, module.SIDE / 2)}
        )
        controller.set_render_plane(labels.id, plane.id, plane)
        assert image.render_planes == labels.render_planes
        await drain_loading(controller)

        of_points = await _only(module, points)
        of_labels = await _only(module, labels)
        of_image = await _only(module, image)

        for octant in upper:
            color = module.COLORS[t, octant]
            marker, marker_size = _blob(of_points, color)
            label, label_size = _blob(of_labels, color)
            sphere, sphere_size = _blob(of_image, color * module.IMAGE_INTENSITY)
            assert marker_size > 0, (t, octant)
            # The label and the image are centred on the marker.
            assert np.abs(label - marker).max() < 1.5, (t, octant, label, marker)
            assert np.abs(sphere - marker).max() < 1.5, (t, octant, sphere, marker)
            # And are discs of the two radii: a sphere cut through its centre.
            ratio = (module.IMAGE_RADIUS / module.LABEL_RADIUS) ** 2
            assert sphere_size / label_size == pytest.approx(ratio, rel=0.15)
            # A circle on screen, though a voxel is twice as deep as wide.
            rows, cols = np.nonzero(
                (np.abs(of_image - np.round(color * 127.5)) <= 3).all(axis=-1)
            )
            assert np.ptp(rows) == pytest.approx(np.ptp(cols), abs=3)

        # The plane does not reach the lower half: its markers are drawn
        # (a marker is not on the plane) and nothing else of theirs is.
        for octant in lower:
            color = module.COLORS[t, octant]
            assert _blob(of_points, color)[1] > 0
            assert _blob(of_labels, color)[1] == 0
            assert _blob(of_image, color * module.IMAGE_INTENSITY)[1] == 0

        # Nothing of another time point is drawn by any of the three.
        for other in range(module.N_TIMES):
            if other == t:
                continue
            for octant in range(8):
                color = module.COLORS[other, octant]
                assert _blob(of_points, color)[1] == 0, (t, other, octant)
                assert _blob(of_labels, color)[1] == 0, (t, other, octant)
                dim = color * module.IMAGE_INTENSITY
                assert _blob(of_image, dim)[1] == 0, (t, other, octant)


def test_the_validation_scene_moves_its_points_and_changes_its_colours(
    load_example, qtbot
):
    module = load_example("plane_slicing_validation")
    positions, colors = module.POSITIONS, module.COLORS
    for t in range(module.N_TIMES):
        for octant in range(8):
            # In its octant, with room for the image sphere.
            centre = np.array(
                [0.25 + 0.5 * ((octant >> (2 - axis)) & 1) for axis in range(3)]
            )
            offset = np.abs(positions[t, octant] - centre * module.SIDE)
            assert (offset + module.IMAGE_RADIUS < module.SIDE / 4).all()
    for a in range(module.N_TIMES):
        for b in range(a + 1, module.N_TIMES):
            assert (np.abs(positions[a] - positions[b]).max(axis=-1) > 0).all()
    # 24 colours, all different, and none the dimmed form of another.
    flat = colors.reshape(-1, 3)
    both = np.vstack([flat, flat * module.IMAGE_INTENSITY])
    distance = np.abs(both[:, None] - both[None]).max(axis=-1)
    assert distance[~np.eye(len(both), dtype=bool)].min() > 8 / 255


# -- The marimo twins ----------------------------------------------------------


async def _load(viewer) -> None:
    """Load the viewer's data, as its first frame in a browser would."""
    controller = viewer.controller
    for scene in _scenes(viewer):
        controller.fit_camera(scene.id)
        controller.reslice_scene(scene.id)
    await drain_loading(controller)


def _colours(picture: np.ndarray) -> int:
    return len(np.unique(picture.reshape(-1, picture.shape[-1]), axis=0))


@pytest.mark.parametrize(
    "name", [name for name in QT_EXAMPLES if name != "ortho_slice_planes"]
)
async def test_the_marimo_twin_runs_and_its_viewer_draws(
    name, run_marimo_example, offscreen_gpu
):
    defs = run_marimo_example(name)
    viewer = defs["viewer"]
    assert viewer.gui == "anywidget"
    await _load(viewer)
    # The data through a colour map, not a blank frame.
    assert _colours(viewer.screenshot(size=(300, 300))) > 50


async def test_the_ortho_marimo_twin_runs_and_its_3d_panel_draws(
    run_marimo_example, offscreen_gpu
):
    defs = run_marimo_example("ortho_slice_planes")
    viewer = defs["viewer"]
    assert viewer.plane_controller.mode == "slices"
    assert len(defs["visuals"]["vol"].render_planes) == 3
    await _load(viewer)
    assert _colours(viewer.screenshot(panel="vol", size=(300, 300))) > 50


async def test_the_marimo_motion_is_one_interaction_and_one_plan(run_marimo_example):
    defs = run_marimo_example("plane_interaction_script")
    viewer, log = defs["viewer"], defs["log"]
    await _load(viewer)
    log.clear()
    await defs["move_planes"]()
    await drain_loading(viewer.controller)
    assert log == [
        "plane interaction: start",
        "plane interaction: end (release)",
        "plan",
    ]
